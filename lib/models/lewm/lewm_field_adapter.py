"""Frozen LeWM-family world model as a *field* `F`, plus the ONE progress-covector routine
that both training and closed-loop inference call."""

import os
from typing import Optional

import torch
import torch.nn as nn

from lib.utils.config import get_config
from lib.utils.logger import logger

LEWM_FAMILY = ("LeWM", "LeWMAWM")


def grip_layout(action_dim: int):
    assert action_dim % 7 == 0 and action_dim > 0, (
        f"ACTION_DIM={action_dim} is not 7 x (number of arms): the field reads 6 pose channels and "
        f"one gripper channel per arm")
    n = action_dim // 7
    return 6 * n, n


def find_world_model_state_file(ckpt_dir: str) -> str:
    """The single `<ModelName>.pth.tar` in a `checkpoints/<name>/` folder."""
    cands = [f for f in os.listdir(ckpt_dir)
             if f.endswith(".pth.tar") and f != "train_param.pth.tar"]
    assert len(cands) == 1, f"expected exactly one model-state *.pth.tar in {ckpt_dir}, found {cands}"
    return os.path.join(ckpt_dir, cands[0])


class LeWMFieldAdapter(nn.Module):
    """A co-trained LeWM-family world model, built from its OWN config file and checkpoint."""

    def __init__(self, cfg_path: str, ckpt_dir: str, strict: bool = True):
        super().__init__()
        self.ctx_action_base = "history"
        self.query_population = False
        self._query_stats = None
        from lib.utils.builder import build_model

        wm_cfg = get_config(config_file=cfg_path, merge=False, freeze=False)
        assert wm_cfg.MODEL.TYPE in LEWM_FAMILY, (
            f"WORLD_MODEL.CONFIG {cfg_path} has MODEL.TYPE={wm_cfg.MODEL.TYPE!r}, which is not a "
            f"LeWM-family world model ({LEWM_FAMILY}). This adapter reaches into `.jepa` for "
            f"encode/action_encoder/predict; any other architecture needs its own adapter.")

        wm_cfg.MODEL.INIT_CKPT_DIR = None

        model = build_model(wm_cfg.MODEL, data_preset=wm_cfg.DATA_PRESET, train=wm_cfg.TRAIN)
        model.setup(summary_writer=None)
        assert ckpt_dir, f"LeWMFieldAdapter({cfg_path}) needs the pretrained world model (LEWM_CKPT)"
        sd_path = find_world_model_state_file(ckpt_dir)
        sd = torch.load(sd_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=strict)

        self.model = model
        self.wm_cfg = wm_cfg
        self.cfg_path = cfg_path
        self.ckpt_path = sd_path

        self.history_size = int(wm_cfg.MODEL.HISTORY_SIZE)
        self.num_preds = int(wm_cfg.MODEL.NUM_PREDS)
        self.frameskip = int(wm_cfg.MODEL.FRAMESKIP)
        self.action_dim = int(wm_cfg.MODEL.ACTION_DIM)
        self.embed_dim = int(wm_cfg.MODEL.EMBED_DIM)
        self.num_cameras = int(wm_cfg.MODEL.get("NUM_CAMERAS", 1))
        self.image_size = int(wm_cfg.DATA_PRESET.IMAGE_SIZE)
        wm_train_ds = wm_cfg.DATASET.TRAIN
        self.cameras = list(wm_train_ds.get("CAMERAS", [wm_train_ds.get("PIXEL_KEY", "pixels")]))
        self.action_source = str(wm_train_ds.get("ACTION_SOURCE", "osc"))
        self.normalize_actions = bool(wm_train_ds.get("NORMALIZE_ACTIONS", True))

        assert self.num_preds == 1, (
            f"LeWMFieldAdapter assumes NUM_PREDS=1 (one predicted step per window); got "
            f"{self.num_preds}. `predict_next` reads only the last predictor position.")
        grip_layout(self.action_dim)

        n = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LeWMFieldAdapter: trainable {wm_cfg.MODEL.TYPE} ({n/1e6:.1f}M params) from "
                    f"{sd_path} | cfg={cfg_path} | D={self.embed_dim} H={self.history_size} "
                    f"frameskip={self.frameskip} cameras={self.cameras} "
                    f"action_source={self.action_source} normalize_actions={self.normalize_actions}")

    @property
    def pose_dim(self) -> int:
        return grip_layout(self.action_dim)[0]

    @property
    def n_grip(self) -> int:
        return grip_layout(self.action_dim)[1]


    def query_bn(self) -> nn.modules.batchnorm._BatchNorm:
        """`pred_proj`'s BatchNorm -- the only layer on the base-point query's path that couples
        samples (the action encoder, the predictor and the potential head are per sample)."""
        net = self.model.jepa.pred_proj.net
        assert (len(net) == 4 and isinstance(net[0], nn.Linear)
                and isinstance(net[1], nn.modules.batchnorm._BatchNorm) and isinstance(net[3], nn.Linear)), (
            "QUERY_MODE: population expects pred_proj = Linear -> BatchNorm -> activation -> Linear "
            f"(lib/models/lewm/module.py::MLP), got {net}")
        return net[1]

    def enable_query_population(self) -> None:
        """`QUERY_MODE: population`: register the query population's running statistics, started
        from the world model's current ones (an EMA, so they converge within ~50 training steps;
        the checkpoint stores them)."""
        bn = self.query_bn()
        self.query_population = True
        self.register_buffer("query_bn_mean", bn.running_mean.detach().clone())
        self.register_buffer("query_bn_var", bn.running_var.detach().clone())

    def assert_data_contract(self, dataset_cfg) -> None:
        """Fail loudly when the policy's dataset and the world model disagree about the data."""
        ds_cameras = list(dataset_cfg.get("CAMERAS", [dataset_cfg.get("PIXEL_KEY", "pixels")]))
        ds_frameskip = int(dataset_cfg.FRAMESKIP)
        ds_action_source = str(dataset_cfg.get("ACTION_SOURCE", "osc"))
        ds_normalize = bool(dataset_cfg.get("NORMALIZE_ACTIONS", True))
        assert ds_frameskip == self.frameskip, (
            f"DATASET FRAMESKIP {ds_frameskip} != world model FRAMESKIP {self.frameskip} "
            f"({self.cfg_path})")
        assert ds_cameras == self.cameras, (
            f"DATASET CAMERAS {ds_cameras} != world model CAMERAS {self.cameras} "
            f"({self.cfg_path})")
        assert ds_action_source == self.action_source and ds_normalize == self.normalize_actions, (
            f"DATASET action convention ({ds_action_source}, normalize={ds_normalize}) != world "
            f"model's ({self.action_source}, normalize={self.normalize_actions}) -- `g` would be "
            f"expressed in different units than the expert action it is matched against")


    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """`pixels` (B,T,C,H,W) or (B,T,V,C,H,W) in [0,1] -> `emb` (B,T,D)."""
        return self.model.jepa.encode({"pixels": pixels})["emb"]

    def window_preds(self, emb: torch.Tensor, action: torch.Tensor,
                     pixels: torch.Tensor = None) -> dict:
        """A world-model `preds` dict built from an ALREADY-COMPUTED window embedding."""
        H, P = self.history_size, self.num_preds
        assert emb.dim() == 3, (
            f"window_preds expects a POOLED window embedding (B,T,D); got shape {tuple(emb.shape)}. "
            f"A token-grid latent (B,T,N,D) is out of scope for this adapter -- `progress_covector` "
            f"hands `head` one pooled vector per frame, and `F` would have to predict in token "
            f"space for `g = J_F^T grad_z p` to be defined there.")
        assert emb.shape[1] == H + P, (
            f"window_preds expects a {H + P}-frame window (HISTORY_SIZE {H} + NUM_PREDS {P}), "
            f"got {emb.shape[1]}")
        assert action.shape[1] == H, (
            f"window_preds expects {H} action chunks (one per context frame), got {action.shape[1]}")
        act_emb = self.model.jepa.action_encoder(torch.nan_to_num(action, 0.0))
        preds = {
            "emb": emb,
            "act_emb": act_emb,
            "tgt_emb": emb[:, P:],
            "pred_emb": self.model.jepa.predict(emb[:, :H], act_emb),
        }
        if pixels is not None:
            preds["pixels"] = pixels
        return preds

    def predict_next(self, ctx_emb: torch.Tensor, ctx_act: torch.Tensor,
                     update_bn_stats: bool = True,
                     detach_dynamics: Optional[bool] = None) -> torch.Tensor:
        """`ctx_emb` (B,H,D), `ctx_act` (B,H,frameskip*7) -> predicted next embedding (B,D).
        `detach_dynamics`: the world model's weights enter as constants (gradient only into the
        inputs and the field)."""
        detach = bool(detach_dynamics) and torch.is_grad_enabled()
        if self.query_population:
            return self._predict_next_population(ctx_emb, ctx_act, update_bn_stats, detach)
        assert not detach and (update_bn_stats or not self.model.training), (
            "detach_dynamics / update_bn_stats=False need QUERY_MODE: population")
        act_emb = self.model.jepa.action_encoder(ctx_act)
        return self.model.jepa.predict(ctx_emb, act_emb)[:, -1]

    def _predict_next_population(self, ctx_emb: torch.Tensor, ctx_act: torch.Tensor,
                                 pose_query: bool, detach: bool) -> torch.Tensor:
        """`predict_next` under `QUERY_MODE: population`: `pred_proj`'s BatchNorm normalizes with
        the base-point QUERY population's statistics."""
        step = _PredictNextStep(self.model.jepa)
        subs = {n: p.detach() for n, p in step.named_parameters()} if detach else {}

        def call(*args):
            return torch.func.functional_call(step, subs, args) if subs else step(*args)

        if not self.model.training:
            mean, var = self.query_bn_mean, self.query_bn_var
        elif pose_query:
            x = call(ctx_emb, ctx_act.detach(), None, None, True)                    # (B*H, C)
            mean, var = x.mean(0), x.var(0, unbiased=False)
            bn = self.query_bn()
            with torch.no_grad():
                m = 0.1 if bn.momentum is None else float(bn.momentum)
                n = x.shape[0]
                self.query_bn_mean.mul_(1 - m).add_(mean.detach(), alpha=m)
                self.query_bn_var.mul_(1 - m).add_(var.detach() * (n / max(n - 1, 1)), alpha=m)
            self._query_stats = (mean, var)
        else:
            stats = self._query_stats or (self.query_bn_mean, self.query_bn_var)
            mean, var = (s.detach() for s in stats) if detach else stats
        return call(ctx_emb, ctx_act, mean, var, False)

class _PredictNextStep(nn.Module):
    """`predict_next`'s computation as a module whose weights `functional_call` can substitute."""

    def __init__(self, jepa):
        super().__init__()
        self.jepa = jepa

    def forward(self, ctx_emb, ctx_act, mean=None, var=None, prebn=False):
        jepa = self.jepa
        if mean is None and not prebn:
            return jepa.predict(ctx_emb, jepa.action_encoder(ctx_act))[:, -1]
        h = jepa.predictor(ctx_emb, jepa.action_encoder(ctx_act))                   # (B,H,d)
        lin, bn, act, out = jepa.pred_proj.net
        x = lin(h.reshape(-1, h.shape[-1]))                     # rows in `(b t) d` order, as predict
        if prebn:
            return x
        y = (x - mean) * torch.rsqrt(var + bn.eps) * bn.weight + bn.bias
        return out(act(y)).reshape(h.shape[0], h.shape[1], -1)[:, -1]


def pose_dim(adapter) -> int:
    return int(getattr(adapter, "pose_dim", 6))


def context_history_slots(adapter: LeWMFieldAdapter, ctx_act: torch.Tensor) -> torch.Tensor:
    """`ctx_act` (B,H,k*7) -> slots `0 .. H-2` under `adapter.ctx_action_base`, (B,H-1,k*7)."""
    B = ctx_act.shape[0]
    hist = ctx_act[:, :adapter.history_size - 1]                               # (B,H-1,k*7)
    if getattr(adapter, "ctx_action_base", "history") == "median":
        k, hist = adapter.frameskip, hist.clone()
        h = hist.view(B, adapter.history_size - 1, k, adapter.action_dim)
        h[..., :pose_dim(adapter)] = 0.0
        hist = h.view(B, adapter.history_size - 1, k * adapter.action_dim)
    return hist


def tile_command(a7: torch.Tensor, frameskip: int) -> torch.Tensor:
    """(B,7) single held command -> (B, frameskip*7) chunk, in the dataset's own layout."""
    return a7.repeat(1, frameskip)


def base_point_context(adapter: LeWMFieldAdapter, ctx_emb: torch.Tensor, ctx_act: torch.Tensor,
                       grip_hold: torch.Tensor, u6: torch.Tensor):
    """`(field_ctx (B,H,D), act (B,H,k*7))` -- what `F` is evaluated at, in ONE place."""
    B = ctx_emb.shape[0]
    a7 = torch.cat([u6, grip_hold.to(u6.device, u6.dtype).reshape(B, -1)], dim=-1)
    chunk = tile_command(a7, adapter.frameskip).unsqueeze(1)                       # (B,1,k*7)
    hist = context_history_slots(adapter, ctx_act)                                 # (B,H-1,k*7)
    return ctx_emb, torch.cat([hist, chunk], dim=1)


def predict_at_base(adapter: LeWMFieldAdapter, ctx_emb: torch.Tensor, ctx_act: torch.Tensor,
                    grip_hold: torch.Tensor, u6_base: torch.Tensor = None) -> torch.Tensor:
    """`z_next` at the covector's OWN linearization point, with no derivative taken -- (B,D)."""
    B, dev, dtype = ctx_emb.shape[0], ctx_emb.device, ctx_emb.dtype
    u6 = (torch.zeros(B, pose_dim(adapter), device=dev, dtype=dtype) if u6_base is None
          else u6_base.to(dev, dtype))
    field_ctx, act = base_point_context(adapter, ctx_emb, ctx_act, grip_hold, u6)
    return adapter.predict_next(field_ctx, act)


def progress_covector(adapter: LeWMFieldAdapter, head, ctx_emb: torch.Tensor,
                      ctx_act: torch.Tensor, lang: torch.Tensor, grip_hold: torch.Tensor,
                      u6_base: torch.Tensor = None, create_graph: bool = False):
    """`g = J_F^T grad_z p_theta(z,l)` in the normalized-command frame -- ONE implementation,
    used by both `LeWMGGFieldModel` (training) and `LeWMGGFieldPlanner` (closed loop)."""
    B = ctx_emb.shape[0]
    dev, dtype = ctx_emb.device, ctx_emb.dtype
    with torch.enable_grad():
        u6 = (torch.zeros(B, pose_dim(adapter), device=dev, dtype=dtype) if u6_base is None
              else u6_base.to(dev, dtype))
        u6 = u6.detach().requires_grad_(True)

        field_ctx, act = base_point_context(adapter, ctx_emb, ctx_act, grip_hold, u6)
        z_next = adapter.predict_next(field_ctx, act)                              # (B,D)
        q = head(z_next, lang).squeeze(-1)                                         # (B,)
        (g,) = torch.autograd.grad(q.sum(), u6, create_graph=create_graph)         # (B,6)
    return g, z_next, q
