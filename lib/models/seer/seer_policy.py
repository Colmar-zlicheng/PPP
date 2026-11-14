import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.metrics.basic_metric import AverageMeter, LossMetric
from lib.models.model_abstraction import ModuleAbstract
from lib.models.potential.field_gripper import FieldGripper
from lib.models.potential.film_mlp import FiLMMLPHead
from lib.models.potential.generalized_gradient import GeneralizedGradientGeometry
from lib.models.potential.potential_scaled_common import _spearman, compute_scaled_potential_loss
from lib.utils.builder import MODEL
from lib.utils.config import CN
from lib.utils.logger import logger

from .seer_net import SeerNet, patchify_normalized
from .state_wm import StateWorldModel

_VIS_PREFIX = "net.vision_encoder."


class SeerFieldGripper(FieldGripper):
    def __init__(self, gcfg, control_steps: int):
        super().__init__(gcfg, ctx_action_base="median", frameskip=control_steps)

    def progress_step(self, batch: dict, dev) -> torch.Tensor:
        return self.frameskip / (batch["ep_len"].to(dev).float() - 1.0).clamp_min(1.0)

    def set_scores(self, batch: dict, d_list, closed: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        """d_list: [(B,) q_close - q_open] per context (real, then dwell). -> the command logit."""
        dev = closed.device
        step = self.progress_step(batch, dev)
        tau = self.tau_steps * step
        flip = torch.where(closed, -1.0, 1.0)
        self._stash = {"adv": [flip * d for d in d_list], "tau": tau, "closed": closed,
                       "dz": dz, "step": step}
        return d_list[0] / tau


@MODEL.register_module()
class SeerPolicyModel(nn.Module, ModuleAbstract):
    """Seer (`HEAD: bc`, its own action head) and Seer-PPP (`HEAD: ppp`): the PPP field read through a
    small world model F on the trunk's action-query states, `a = m dp(F(s, u), l)/du` at u = 0."""

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.head_type = str(cfg.get("HEAD", "bc")).lower()
        assert self.head_type in ("bc", "ppp"), f"MODEL.HEAD must be bc or ppp, got {self.head_type}"
        assert not bool(cfg.get("ACTION_CONDITIONED", False)), "MODEL.ACTION_CONDITIONED is not supported"
        sc = cfg.SEER
        self.is_ppp = self.head_type == "ppp"
        assert not self.is_ppp or str(sc.get("PPP_STATE", "trunk")).lower() == "trunk", \
            "SEER.PPP_STATE must be trunk"
        self.policy_mode = "seer_ppp" if self.is_ppp else "seer"
        self.seq_len = int(sc.get("SEQ_LEN", 7))
        self.future = int(sc.get("FUTURE_STEPS", 3))
        self.chunk = int(sc.get("ACTION_PRED_STEPS", 3))
        assert not self.is_ppp or self.chunk == self.future, (
            "the world model F predicts the frame the input chunk ends at: "
            "ACTION_PRED_STEPS must equal FUTURE_STEPS")
        assert not bool(sc.get("SHIFT_PER_WINDOW", False)), "SEER.SHIFT_PER_WINDOW is not supported"
        self.net = SeerNet(
            mae_ckpt=sc.get("MAE_CKPT", None), seq_len=self.seq_len,
            num_resampler_query=int(sc.get("NUM_RESAMPLER_QUERY", 6)),
            num_obs_token_per_image=int(sc.get("NUM_OBS_TOKEN_PER_IMAGE", 9)),
            action_pred_steps=self.chunk, transformer_layers=int(sc.get("TRANSFORMER_LAYERS", 24)),
            hidden_dim=int(sc.get("HIDDEN_DIM", 384)), transformer_heads=int(sc.get("TRANSFORMER_HEADS", 12)),
            rgb_pad=int(sc.get("RGB_PAD", 10)), gripper_pad=int(sc.get("GRIPPER_PAD", 4)),
            bc_head=not self.is_ppp, gpt_dropout=float(sc.get("GPT_DROPOUT", 0.1)),
            query_tokens=self.is_ppp)
        self.wm_head = None
        if self.is_ppp:
            wc = cfg.get("STATE_WM", CN())
            self.wm_head = StateWorldModel(
                hidden=self.net.hidden_dim, n_state=self.chunk, action_steps=self.chunk,
                layers=int(wc.get("LAYERS", 4)), heads=int(wc.get("HEADS", 12)),
                mlp_ratio=int(wc.get("MLP_RATIO", 4)), out_scale=float(wc.get("OUT_SCALE", 0.1)))

        txt = np.load(str(sc.CLIP_TEXT))
        self.register_buffer("text_table", torch.from_numpy(txt["feat"]).float())
        import h5py
        with h5py.File(str(cfg.ACTION_STATS_H5), "r") as f:
            self.register_buffer("action_loc", torch.from_numpy(f["action_loc"][:]).float())
            self.register_buffer("action_scale", torch.from_numpy(f["action_scale"][:]).float())
        self.raw_actions = str(sc.get("ACTION_UNITS", "normalized")).lower() == "raw"
        self.corpus_scale2 = float(self.action_scale[:6].pow(2).mean())
        if self.raw_actions:
            self.action_loc.zero_()
            self.action_scale.fill_(1.0)

        lc = cfg.LOSS
        self.w_arm = float(lc.get("ARM", 1.0))
        self.w_grip_bc = float(lc.get("GRIPPER", 0.01))
        self.w_image = float(lc.get("IMAGE", 0.1))
        self.w_latent = float(lc.get("LATENT", 1.0)) if self.is_ppp else 0.0

        self.head = self.geom = self.field_gripper = None
        if self.is_ppp:
            assert str(lc.get("ACT_TARGET", "per_tick")).lower() == "per_tick", "LOSS.ACT_TARGET must be per_tick"
            assert str(lc.get("ACT_SPACE", "raw")).lower() == "raw", "LOSS.ACT_SPACE must be raw"
            assert str(lc.get("ACT_LOSS", "vec_dir")).lower() == "vec_dir", "LOSS.ACT_LOSS must be vec_dir"
            pc = cfg.get("PROGRESS", CN())
            L = self.chunk * self.net.hidden_dim
            self.state_dim = L
            self.head = FiLMMLPHead(L, int(self.text_table.shape[1]),
                                    int(pc.get("HIDDEN_DIM", 1024)), int(pc.get("NUM_LAYERS", 4)),
                                    int(pc.get("FILM_HIDDEN", 256)), float(pc.get("DROPOUT", 0.0)),
                                    out_dim=1)
            gc = cfg.get("GEOM", CN())
            self.geom = GeneralizedGradientGeometry(
                in_dim=L, action_dim=6, hidden_dim=int(gc.get("HIDDEN_DIM", 256)),
                num_layers=int(gc.get("NUM_LAYERS", 2)), eps=float(gc.get("EPS", 1e-3)))
            self.field_gripper = SeerFieldGripper(cfg.GRIPPER_FIELD, control_steps=self.future)
            self.p_loss_kw = dict(
                lambda_mc=float(lc.get("LAMBDA_MC", 1.0)), lambda_rank=float(lc.get("LAMBDA_RANK", 0.5)),
                rank_tau=float(lc.get("RANK_TAU", 0.05)),
                rank_min_delta=float(lc.get("RANK_MIN_DELTA", 0.0)),
                lambda_boundary=float(lc.get("LAMBDA_BOUNDARY", 1.0)),
                terminal_thresh=float(lc.get("TERMINAL_THRESH", 0.95)),
                nonterminal_eps=float(lc.get("NONTERMINAL_EPS", 0.05)),
                lambda_lang=float(lc.get("LAMBDA_LANG", 0.5)), lang_margin=float(lc.get("LANG_MARGIN", 0.2)),
                lambda_td=float(lc.get("LAMBDA_TD", 1.0)))
            self.lambda_act = float(lc.get("LAMBDA_ACT", 1.0))
            self.lambda_vec = float(lc.get("LAMBDA_VEC", 1.0))
            self.lambda_dir = float(lc.get("LAMBDA_DIR", 1.0))
            self.lambda_grip = float(lc.get("LAMBDA_GRIP", 10.0))
            self.stage2_start = int(cfg.get("STAGES", CN()).get("STAGE2_START_EPOCH", 10))
            self.act_forecast_grad = float(cfg.get("ACT_FORECAST_GRAD", 1.0))
            assert 0.0 <= self.act_forecast_grad <= 1.0
            self.act_trunk_grad = float(cfg.get("ACT_TRUNK_GRAD", 1.0))
            assert 0.0 <= self.act_trunk_grad <= 1.0
            assert float(cfg.get("LP_FORECAST_GRAD", 0.0)) == 0.0, "LP_FORECAST_GRAD must be 0 (L_p reads stop-gradient trunk states)"
            assert cfg.get("READOUT_DROPOUT", True) is True, "READOUT_DROPOUT is not supported"
            self._w_act = 0.0
            self._set_stage(0)
        assert not bool(cfg.get("FREEZE_NET", False)) and int(cfg.get("NET_TRAIN_START_EPOCH", 0)) == 0 \
            and not bool(cfg.get("STAGES", CN()).get("STAGE1_FREEZE_NET", False)), \
            "FREEZE_NET / NET_TRAIN_START_EPOCH / STAGES.STAGE1_FREEZE_NET are not supported"

        self._epoch = 0
        self.loss_metric = LossMetric(cfg)
        self.meters = {k: AverageMeter(k) for k in (
            "act_cos", "act_mse", "act_cos_all", "grip_acc", "image_loss", "latent_loss",
            "latent_copy", "rank_acc", "lang_acc", "m_mean", "g_norm")}
        self._val_prog = ([], [], [])
        self._val_q = ([], [])
        self.summary = None
        self.train_log_interval = int(cfg.TRAIN.get("LOG_INTERVAL", 20))
        self._last_primary_score = float("-inf")
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"{self.name}: {self.policy_mode} | seq {self.seq_len} future {self.future} chunk "
                    f"{self.chunk} | {n_train / 1e6:.1f}M trainable (+{sum(p.numel() for p in self.net.vision_encoder.parameters()) / 1e6:.1f}M frozen MAE)"
                    f" | loss arm {self.w_arm} grip {self.w_grip_bc} image {self.w_image} latent {self.w_latent}"
                    + (f" | PPP state trunk ({self.state_dim}-d): act {self.lambda_act} "
                       f"(vec {self.lambda_vec} dir {self.lambda_dir}) grip {self.lambda_grip} "
                       f"stage2@{self.stage2_start} | world model F "
                       f"({sum(p.numel() for p in self.wm_head.parameters()) / 1e6:.1f}M) on the trunk's queries, "
                       f"forecast-grad {self.act_forecast_grad}, trunk-grad {self.act_trunk_grad} | "
                       + self.field_gripper.describe() if self.is_ppp else "")
                    + (" | raw Seer action units" if self.raw_actions else ""))

    def setup(self, summary_writer=None, **kwargs):
        self.summary = summary_writer

    def state_dict(self, *args, **kwargs):
        """The frozen MAE is rebuilt from SEER.MAE_CKPT and not stored in checkpoints."""
        sd = nn.Module.state_dict(self, *args, **kwargs)
        for k in [k for k in sd if k.startswith(_VIS_PREFIX)]:
            del sd[k]
        return sd

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        if not any(k.startswith(_VIS_PREFIX) for k in state_dict):
            state_dict = dict(state_dict)
            for k, v in self.net.vision_encoder.state_dict().items():
                state_dict[_VIS_PREFIX + k] = v
        return nn.Module.load_state_dict(self, state_dict, strict=strict, **kwargs)

    def param_groups(self, train_cfg):
        lr = float(train_cfg.LR)
        head_lr = float(train_cfg.get("HEAD_LR", lr))
        net, head = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                (head if not n.startswith("net.") else net).append(p)
        groups = [{"params": net, "lr": lr, "name": "SEER"}] if net else []
        if head:
            groups.append({"params": head, "lr": head_lr, "name": "PPP_HEAD"})
        logger.info(f"{self.name}: param groups SEER {len(net)} tensors lr={lr:g}"
                    + (f", PPP_HEAD {len(head)} tensors lr={head_lr:g}" if head else ""))
        return groups

    def _set_stage(self, epoch_idx: int) -> None:
        w = 0.0 if epoch_idx < self.stage2_start else 1.0
        if w != self._w_act:
            logger.warning(f"{self.name}: epoch {epoch_idx}: L_act/L_grip weight {w} "
                           f"(STAGE2_START_EPOCH={self.stage2_start})")
        self._w_act = w

    def on_train_epoch_start(self, epoch_idx: int) -> None:
        self._epoch = epoch_idx
        if self.is_ppp:
            self._set_stage(epoch_idx)

    def frozen_parameters(self) -> list:
        """Parameters whose gradients `on_before_optimizer_step` drops at the current epoch (the trainer
        leaves them out of a global clip norm): the readout gain before L_act is switched on."""
        if self.is_ppp and self._w_act == 0.0:
            return list(self.geom.parameters())
        return []

    def readout_parameters(self) -> list:
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("net.")]

    def on_before_optimizer_step(self) -> None:
        for p in self.frozen_parameters():
            p.grad = None

    def _device(self):
        return self.text_table.device

    def normalize(self, a_raw: torch.Tensor) -> torch.Tensor:
        return (a_raw - self.action_loc[: a_raw.shape[-1]]) / self.action_scale[: a_raw.shape[-1]]

    def model_actions(self, batch: dict) -> torch.Tensor:
        """The expert chunk(s) in the model's action units: the dataset's `actions_norm`, or with
        SEER.ACTION_UNITS raw its `actions_raw` (Seer's own action space). (..., K, 7)."""
        key = "actions_raw" if self.raw_actions else "actions_norm"
        return batch[key].to(self._device()).float()

    def act_loss_terms(self, a_hat: torch.Tensor, expert6: torch.Tensor):
        """vec and dir of L_act in raw command units: a_raw - e_raw = scale * (a - e), vec divided by
        the corpus mean(scale^2) (also under ACTION_UNITS raw, where the model's own scale is 1)."""
        w = self.action_scale[:6]          # model units -> raw command units (1 under ACTION_UNITS raw)
        s2 = self.corpus_scale2 if self.raw_actions else w.pow(2).mean()
        l_vec = ((a_hat - expert6) * w).pow(2).mean() / s2
        l_dir = (1.0 - F.cosine_similarity(a_hat * w, expert6 * w, dim=-1, eps=1e-8)).mean()
        return l_vec, l_dir

    def group_states(self, base: torch.Tensor, groups: list) -> list:
        """One trunk pass, then the world model F(s, u) per group at the group's steps -> the state p
        reads for each group, (B,T,D). The single entry point for p's state in training and deployment
        (lib/planning/seer_planner.py)."""
        s = self.trunk_states(base)[1]
        S = s.shape[1]
        out = []
        for g in groups:
            T = 1 if g.get("last", False) else S
            out.append(self.wm_head(s[:, S - T:], g["u6"], g["grip"]).flatten(-2))
        return out

    def trunk_states(self, base: torch.Tensor):
        """Seer's own pass -- one action-free foresight group read by the action queries -> (the
        foresight outputs (B,S,18,H), read by Seer's pixel loss; the query outputs s (B,S,K,H), the state)."""
        out = self.net.run(base, [{"u6": None, "queries": True}])
        return out["obs"][0], out["queries"][0]

    @contextlib.contextmanager
    def _target_mode(self):
        """The transformer without dropout (eval mode) for the passes that only make TARGETS (F's next
        state, L_p's real-frame states); every module's own train flag is restored after."""
        tb = self.net.transformer_backbone
        flags = [(m, m.training) for m in tb.modules()]
        tb.eval()
        try:
            yield
        finally:
            for m, t in flags:
                m.training = t

    def _trunk_window_state(self, frames: torch.Tensor, states: torch.Tensor, text: torch.Tensor, feat=None):
        with torch.no_grad(), self._target_mode():
            if feat is None:
                feat = self.net.encode_images(self.net.preprocess(frames, augment=self.training))
            img_tok, cls_tok = self.net.frame_tokens(feat)
            base = self.net.base_tokens(text, states, img_tok, cls_tok)
            return self.trunk_states(base)[1]

    def _later_state(self, batch: dict, text: torch.Tensor) -> torch.Tensor:
        dev = self._device()
        assert "later_frames" in batch, "Seer-PPP needs DATASET LATER_WINDOW: true"
        assert "later_window_end" in batch and int(batch["later_window_end"].max()) == 0, \
            "Seer-PPP needs DATASET LATER_WINDOW_END: frame"
        return self._trunk_window_state(batch["later_frames"].to(dev, non_blocking=True),
                                        batch["later_states"].to(dev).float(), text)[:, -1].flatten(-2)

    def _encode(self, batch: dict, n_frames: int):
        """-> pixels (B,n,2,3,224,224), MAE features (B,n,2,197,768), A tokens, text feature."""
        dev = self._device()
        frames = batch["frames"][:, :n_frames].to(dev, non_blocking=True)
        x = self.net.preprocess(frames, augment=self.training)
        n_mae = n_frames if self.is_ppp else self.seq_len    # Seer-PPP: the shifted window too
        feat = self.net.encode_images(x[:, :n_mae])
        img_tok, cls_tok = self.net.frame_tokens(feat[:, :self.seq_len])
        text = self.text_table[batch["task_idx"].to(dev)]
        base = self.net.base_tokens(text, batch["states"].to(dev).float(), img_tok, cls_tok)
        return x, feat, img_tok, cls_tok, text, base

    def _wm_losses(self, x, obs_e) -> dict:
        """Seer's pixel loss on the foresight outputs: predict the frames FUTURE_STEPS ahead."""
        S, Fs = self.seq_len, self.future
        B = x.shape[0]
        tgt = x[:, Fs:Fs + S]                                               # (B,S,2,3,224,224)
        pred = self.net.image_decode(obs_e)                                 # (B*S,2,P,768)
        t0 = patchify_normalized(tgt[:, :, 0].reshape(B * S, *tgt.shape[-3:]), self.net.patch_size)
        t1 = patchify_normalized(tgt[:, :, 1].reshape(B * S, *tgt.shape[-3:]), self.net.patch_size)
        return {"image": 0.5 * (F.mse_loss(pred[:, 0], t0) + F.mse_loss(pred[:, 1], t1))}

    def _groups(self, batch: dict, u0: torch.Tensor = None):
        """[E: the expert chunk, 0: the base point (u0, one command per tick, gripper in force)]."""
        dev = self._device()
        an = self.model_actions(batch)                                      # (B,S,K,7)
        ar = batch["actions_raw"].to(dev).float()
        hold = batch["grip_hold"].to(dev).float()                           # (B,S)
        B, S, K = an.shape[:3]
        E = {"u6": an[..., :6], "grip": torch.where(ar[..., 6] > 0, 1.0, -1.0)}
        if u0 is None:
            u0 = torch.zeros(B, S, 6, device=dev)
        u6 = u0 if u0.dim() == 4 else u0.unsqueeze(2).expand(B, S, K, 6)
        Z = {"u6": u6, "grip": hold.unsqueeze(-1).expand(B, S, K)}
        return E, Z

    def _forward_bc(self, batch: dict) -> dict:
        S = self.seq_len
        n_frames = S + self.future
        x, feat, _, _, _, base = self._encode(batch, n_frames)
        out = self.net.run(base, [{"u6": None}], queries=True, read_group=0)
        arm, glogit = self.net.bc_decode(out["query"])                      # (B,S,K,6), (B,S,K)
        return {"arm": arm, "grip_logit": glogit, "wm": self._wm_losses(x, out["obs"][0])}

    def _bc_loss(self, preds: dict, batch: dict) -> dict:
        ar = batch["actions_raw"].to(self._device()).float()
        l_arm = F.smooth_l1_loss(preds["arm"], ar[..., :6])
        l_grip = F.binary_cross_entropy_with_logits(preds["grip_logit"], (ar[..., 6] > 0).float())
        loss = self.w_arm * l_arm + self.w_grip_bc * l_grip + self.w_image * preds["wm"]["image"]
        ld = {"l_arm": l_arm.detach(), "l_grip": l_grip.detach(), "l_image": preds["wm"]["image"].detach()}
        with torch.no_grad():
            a_hat = self.normalize(preds["arm"].mean(dim=2))                # (B,S,6) normalized
            a_e = self.model_actions(batch)[..., :6].mean(dim=2)
            ld["_stat_act_cos"] = F.cosine_similarity(a_hat[:, -1], a_e[:, -1], dim=-1).mean()
            ld["_stat_act_cos_all"] = F.cosine_similarity(a_hat, a_e, dim=-1).mean()
            ld["_stat_act_mse"] = (a_hat[:, -1] - a_e[:, -1]).pow(2).sum(-1).mean()
            ld["_stat_grip_acc"] = ((preds["grip_logit"][:, -1, 0] > 0) == (ar[:, -1, 0, 6] > 0)).float().mean()
        ld["loss"] = loss
        return ld

    def _wm_call(self, s, u6, grip):
        """The world model F as the readout sees it: F's weights as sg(w) + beta (w - sg w) (ACT_FORECAST_GRAD), so
        L_act's (and L_grip's) gradient into F is scaled by beta and its values are unchanged."""
        b = self.act_forecast_grad
        if b == 1.0:
            return self.wm_head(s, u6, grip)
        params = {n: (p.detach() if b == 0.0 else p.detach() + b * (p - p.detach()))
                  for n, p in self.wm_head.named_parameters()}
        return torch.func.functional_call(self.wm_head, params, (s, u6, grip))

    def _trunk_in(self, s):
        """The trunk's state as the readout sees it: sg(s) + kappa (s - sg s) (ACT_TRUNK_GRAD)."""
        k = self.act_trunk_grad
        d = s.detach()
        return d if k == 0.0 else (s if k == 1.0 else d + k * (s - d))

    def _covector_trunk(self, s, batch, text, create_graph: bool):
        """g = dp(F(s_t, u), l)/du_k at u = 0 for every step t and tick k, exact (F mixes no steps). F is small and
        twice differentiable as written, so no attention context is needed. -> z0 (B,S,K*H), q (B,S), g (B,S,K,6)."""
        B, S = s.shape[:2]
        with torch.enable_grad():
            u0 = torch.zeros(B, S, self.chunk, 6, device=s.device, requires_grad=True)
            _, Z = self._groups(batch, u0)
            z0 = self._wm_call(self._trunk_in(s), Z["u6"], Z["grip"]).flatten(-2)
            q = self.head(z0.reshape(B * S, -1), text.repeat_interleave(S, 0)).view(B, S)
            (g,) = torch.autograd.grad(q.sum(), u0, create_graph=create_graph)
        return z0, q, g

    def _trunk_dwell(self, dwell):
        """The trunk's state at the last step of the dwell context (B,K,H), its gradient into the trunk's weights
        scaled by kappa (functional_call on sg(w) + kappa (w - sg w))."""
        k = self.act_trunk_grad
        if not torch.is_grad_enabled() or k == 1.0:
            return self.trunk_states(dwell)[1][:, -1]
        params = {n: (p.detach() if k == 0.0 else p.detach() + k * (p - p.detach()))
                  for n, p in self.net.named_parameters() if p.requires_grad}
        out = torch.func.functional_call(self.net, params, (dwell, [{"u6": None}]))
        return out[:, -1].view(dwell.shape[0], self.chunk, -1)

    def _grip_scores_trunk(self, s, img_tok, cls_tok, text, batch):
        dev = s.device
        B, S = s.shape[:2]
        K = self.chunk
        with torch.no_grad():
            st = batch["states"].to(dev).float()
            dwell = self.net.base_tokens(text, st[:, -1:].expand(B, S, st.shape[-1]),
                                         img_tok[:, -1:].expand(-1, S, -1, -1),
                                         cls_tok[:, -1:].expand(-1, S, -1, -1))
        iso = self.field_gripper.isolate or (self.act_forecast_grad == 0.0 and self.act_trunk_grad == 0.0)
        with (torch.no_grad() if iso else contextlib.nullcontext()):
            ctx = torch.cat([self._trunk_in(s[:, -1]), self._trunk_dwell(dwell)])          # (2B,K,H)
            z = torch.zeros(2 * B, K, 6, device=dev)
            one = torch.ones(2 * B, K, device=dev)
            z_open = self._wm_call(ctx, z, -one).flatten(-2)
            z_close = self._wm_call(ctx, z, one).flatten(-2)
        lang = text.repeat(2, 1)
        q = self.head(torch.cat([z_open, z_close]), torch.cat([lang, lang])).squeeze(-1)
        return q[2 * B:] - q[:2 * B], (z_close - z_open).detach().norm(dim=-1).mean()

    def _forward_ppp(self, batch: dict, create_graph: bool) -> dict:
        S, Fs, K = self.seq_len, self.future, self.chunk
        dev = self._device()
        x, feat, img_tok, cls_tok, text, base = self._encode(batch, S + Fs)
        obs, s = self.trunk_states(base)                                     # (B,S,18,H), (B,S,K,H)
        wm = self._wm_losses(x, obs)                                         # Seer's pixel loss, its own foresight
        assert "future_states" in batch, "Seer-PPP needs DATASET FUTURE_STATES: true"
        st_next = torch.cat([batch["states"][:, Fs:], batch["future_states"]], dim=1).to(dev).float()
        s_next = self._trunk_window_state(None, st_next, text, feat=feat[:, Fs:Fs + S])
        E, _ = self._groups(batch)
        wm["latent"] = F.mse_loss(self.wm_head(s.detach(), E["u6"], E["grip"]), s_next)
        wm["latent_copy"] = F.mse_loss(s.detach(), s_next).detach()
        z0, q, g = self._covector_trunk(s, batch, text, create_graph)
        B = base.shape[0]
        zr = z0.detach().reshape(B * S, 1, -1).expand(B * S, K, z0.shape[-1]).reshape(B * S * K, -1)
        a_hat, M, _ = self.geom(zr, g.reshape(B * S * K, 6))        # one scalar mobility m(sg z0) on the chunk
        a_hat, m = a_hat.view(B, S, K, 6), M[:, 0, 0].view(B, S, K)[..., 0]
        s_t = s_next[:, -1].flatten(-2)
        s_l = self._later_state(batch, text)
        wrong = self.text_table[batch["wrong_task_idx"].to(dev)]
        p = self.head(torch.cat([s_t, s_l, s_l]), torch.cat([text, text, wrong])).squeeze(-1)
        d, dz = self._grip_scores_trunk(s, img_tok, cls_tok, text, batch)
        return {"a_hat": a_hat, "g": g, "q": q, "m": m,
                "p_t": p[:B], "p_tp1": p[B:2 * B], "p_tp1_wrong": p[2 * B:],
                "grip_d": d, "grip_dz": dz, "wm": wm}

    def _grip_batch(self, batch: dict, dev) -> dict:
        """The field gripper's labels at the window's last step."""
        return {"grip_event_offset": batch["grip_event_offset"][:, -1].to(dev),
                "expert_grip_sign": batch["expert_grip_sign"][:, -1].to(dev),
                "expert_grip_switch": batch["expert_grip_switch"][:, -1].to(dev),
                "ep_len": batch["ep_len"].to(dev)}

    def _ppp_loss(self, preds: dict, batch: dict) -> dict:
        dev = preds["a_hat"].device
        B = preds["a_hat"].shape[0]
        expert6 = self.model_actions(batch)[..., :6]               # (B,S,K,6), one command per tick
        a_hat = preds["a_hat"]
        l_vec, l_dir = self.act_loss_terms(a_hat, expert6)
        l_act = self.lambda_vec * l_vec + self.lambda_dir * l_dir
        a_hat, expert6 = a_hat[:, :, 0], expert6[:, :, 0]          # the metrics below read the chunk's first command

        gb = self._grip_batch(batch, dev)
        closed = batch["grip_hold"][:, -1].to(dev) > 0
        d = preds["grip_d"]
        logit = self.field_gripper.set_scores(gb, [d[:B], d[B:]], closed, preds["grip_dz"])
        l_grip, gstats = self.field_gripper.loss(gb, torch.ones(B, device=dev), logit)

        p_out = compute_scaled_potential_loss(
            preds["p_t"], preds["p_tp1"], preds["p_tp1_wrong"],
            batch["prog_t"].to(dev), batch["prog_tp1"].to(dev), **self.p_loss_kw)
        wm = preds["wm"]
        loss = (p_out["loss"] + self.w_image * wm["image"] + self.w_latent * wm["latent"]
                + self._w_act * (self.lambda_act * l_act + self.lambda_grip * l_grip))
        ld = {"loss": loss, "l_p": p_out["loss"].detach(), "l_act": l_act.detach(),
              "l_vec": l_vec.detach(), "l_dir": l_dir.detach(), "l_grip": l_grip.detach(),
              "l_image": wm["image"].detach(), "l_latent": wm["latent"].detach()}
        for k in ("l_mc", "l_rank", "l_boundary", "l_lang", "l_td"):
            if k in p_out:
                ld[k] = p_out[k]
        with torch.no_grad():
            ld["_stat_act_cos"] = F.cosine_similarity(a_hat[:, -1], expert6[:, -1], dim=-1).mean()
            ld["_stat_act_cos_all"] = F.cosine_similarity(a_hat, expert6, dim=-1).mean()
            ld["_stat_act_mse"] = (a_hat[:, -1] - expert6[:, -1]).pow(2).sum(-1).mean()
            ld["_stat_m_mean"] = preds["m"].mean()
            ld["_stat_g_norm"] = preds["g"].flatten(2).norm(dim=-1).mean()
            ld["_stat_rank_acc"] = (preds["p_tp1"] > preds["p_t"]).float().mean()
            ld["_stat_lang_acc"] = (preds["p_tp1"] > preds["p_tp1_wrong"]).float().mean()
            ld["_stat_latent_copy"] = wm["latent_copy"]
            for k, v in gstats.items():
                ld[f"_stat_{k}"] = v
        return ld

    def compute_loss(self, preds: dict, batch: dict) -> dict:
        return self._ppp_loss(preds, batch) if self.is_ppp else self._bc_loss(preds, batch)

    def _forward(self, batch: dict, train: bool) -> dict:
        return self._forward_ppp(batch, create_graph=train) if self.is_ppp else self._forward_bc(batch)

    def training_step(self, batch: dict, step_idx: int):
        preds = self._forward(batch, train=True)
        ld = self.compute_loss(preds, batch)
        metric = {k: v for k, v in ld.items() if not k.startswith("_stat_")}
        self.loss_metric.feed(metric, batch["frames"].shape[0])
        if self.summary is not None and step_idx % self.train_log_interval == 0:
            for k, v in ld.items():
                self.summary.add_scalar(f"train/{k.replace('_stat_', '')}", float(v), step_idx)
        return preds, ld

    def on_train_finished(self, recorder, epoch_idx: int) -> None:
        recorder.record_loss(self.loss_metric, epoch_idx, comment=f"{self.name}-train")
        self.loss_metric.reset()

    def validation_step(self, batch: dict, step_idx: int):
        n = int(batch["frames"].shape[0])
        if self.is_ppp:
            preds = self._forward_ppp(batch, create_graph=False)
            ld = self._ppp_loss(preds, batch)
            self.field_gripper.update_val(self._grip_batch(batch, preds["a_hat"].device))
            self._val_prog[0].append(batch["task_idx"].numpy())
            self._val_prog[1].append(batch["prog_tp1"].numpy())
            self._val_prog[2].append(preds["p_tp1"].detach().cpu().numpy())
            self._val_q[0].append(batch["prog_t"].numpy())
            self._val_q[1].append(preds["q"][:, -1].detach().cpu().numpy())
        else:
            with torch.no_grad():
                preds = self._forward(batch, train=False)
                ld = self.compute_loss(preds, batch)
        for k in self.meters:
            key = f"_stat_{k}"
            if key in ld:
                self.meters[k].update_by_mean(float(ld[key]), n)
        if "l_image" in ld:
            self.meters["image_loss"].update_by_mean(float(ld["l_image"]), n)
        if "l_latent" in ld:
            self.meters["latent_loss"].update_by_mean(float(ld["l_latent"]), n)
        return preds

    def on_val_finished(self, recorder, epoch_idx: int) -> None:
        meters = [m for m in self.meters.values() if m.count > 0]
        recorder.record_metric(meters, epoch_idx, comment=f"{self.name}-val")
        msg = " ".join(f"{m.name}={m.avg:.4f}" for m in meters)
        if self.meters["latent_copy"].count and self.meters["latent_copy"].avg > 0:
            pvc = self.meters["latent_loss"].avg / self.meters["latent_copy"].avg
            msg += f" pred_vs_copy={pvc:.4f}"
            if self.summary is not None:
                self.summary.add_scalar("val/epoch_pred_vs_copy", pvc, epoch_idx)
        if self.is_ppp and self._val_prog[0]:
            t, pr, pp = (np.concatenate(v) for v in self._val_prog)
            per = [s for k in np.unique(t) if (t == k).sum() >= 2
                   for s in [_spearman(pp[t == k], pr[t == k])] if not np.isnan(s)]
            ms = float(np.mean(per)) if per else float("nan")
            msg += f" macro_spearman={ms:.4f}"
            qp, qq = (np.concatenate(v) for v in self._val_q)
            perq = [s for k in np.unique(t) if (t == k).sum() >= 2
                    for s in [_spearman(qq[t == k], qp[t == k])] if not np.isnan(s)]
            qs = float(np.mean(perq)) if perq else float("nan")
            msg += f" q_spearman={qs:.4f} q_std={float(qq.std()):.4f}"
            if self.summary is not None and perq:
                self.summary.add_scalar("val/epoch_q_spearman", qs, epoch_idx)
            self._val_q = ([], [])
            if self.summary is not None and per:
                self.summary.add_scalar("val/epoch_macro_spearman", ms, epoch_idx)
            self._val_prog = ([], [], [])
            self.field_gripper.log_val(self.summary, logger, self.name, epoch_idx)
        logger.info(f"{self.name} val epoch {epoch_idx}: {msg}")
        if self.summary is not None:
            for m in meters:
                self.summary.add_scalar(f"val/epoch_{m.name}", m.avg, epoch_idx)
        self._last_primary_score = self.meters["act_cos"].avg if self.meters["act_cos"].count else float("-inf")
        for m in self.meters.values():
            m.reset()

    def primary_val_score(self) -> float:
        return self._last_primary_score

    def format_metric(self, mode: str) -> str:
        if mode != "train":
            return ""
        try:
            keys = (["loss", "l_p", "l_act", "l_grip", "l_latent", "l_image"] if self.is_ppp
                    else ["loss", "l_arm", "l_grip", "l_image"])
            return " | ".join(f"{k}={self.loss_metric.get_loss(k):.4f}" for k in keys)
        except (KeyError, ZeroDivisionError):
            return ""

    def forward(self, inputs, step_idx, mode: str = "train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx)
        if mode == "val":
            return self.validation_step(inputs, step_idx)
        raise ValueError(f"Unknown mode: {mode}")
