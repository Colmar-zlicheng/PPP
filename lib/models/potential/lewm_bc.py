import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.metrics.basic_metric import AverageMeter, LossMetric
from lib.metrics.collapse import CollapseMetric
from lib.models.lewm.lewm_field_adapter import LeWMFieldAdapter, predict_at_base
from lib.models.model_abstraction import ModuleAbstract
from lib.utils.builder import MODEL
from lib.utils.config import CN
from lib.utils.logger import logger
from lib.utils.misc import param_size

from .film_mlp import FiLMMLPHead
from .lewm_gg_field import GripperSwitchHead, _WM_PREFIX


@MODEL.register_module()
class LeWMBCModel(nn.Module, ModuleAbstract):
    """Behavior cloning on the same co-trained world model as PPP: the pose command is regressed
    from the world model's reference prediction, a = pi_phi(F(z, [u0, gripper]), l), and the
    gripper comes from a switching head."""

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_cfg = cfg.TRAIN
        self.policy_mode = "bc_wm"

        wmt = cfg.get("WORLD_MODEL_TRAIN", CN())
        assert bool(wmt.get("ENABLED", True)), "MODEL.WORLD_MODEL_TRAIN.ENABLED must be true (co-training)"
        assert str(wmt.get("BN_MODE", "train")).lower() == "train", "MODEL.WORLD_MODEL_TRAIN.BN_MODE must be 'train'"
        self.wm_start_epoch = int(wmt.get("START_EPOCH", 0))
        self.wm = LeWMFieldAdapter(str(cfg.LEWM_CONFIG), str(cfg.LEWM_CKPT))
        vis_dim = self.wm.embed_dim

        act_cfg = cfg.get("ACT_SUPERVISION", CN())
        ctx_base = str(act_cfg.get("CTX_ACTION_BASE", "history"))
        assert ctx_base in ("history", "median"), (
            f"ACT_SUPERVISION.CTX_ACTION_BASE must be 'history' or 'median', got {ctx_base!r}")
        self.wm.ctx_action_base = ctx_base
        assert str(act_cfg.get("CTX_IMAGE_BASE", "history")) == "history", (
            "ACT_SUPERVISION.CTX_IMAGE_BASE must be 'history'")

        self.lang_dim = int(cfg.get("LANG_DIM", 384))
        self.policy_head = FiLMMLPHead(latent_dim=vis_dim, lang_dim=self.lang_dim,
                                       hidden_dim=int(cfg.get("HIDDEN_DIM", 1024)),
                                       num_layers=int(cfg.get("NUM_LAYERS", 4)),
                                       film_hidden=int(cfg.get("FILM_HIDDEN", 256)),
                                       dropout=float(cfg.get("DROPOUT", 0.0)),
                                       out_dim=self.wm.pose_dim)

        grip_cfg = cfg.get("GRIPPER_HEAD", CN())
        self.grip_head = GripperSwitchHead(vis_dim, self.lang_dim,
                                           int(grip_cfg.get("HIDDEN_DIM", 256)),
                                           n_out=self.wm.n_grip)

        loss_cfg = cfg.LOSS
        self.lambda_vec = float(loss_cfg.get("LAMBDA_VEC", 1.0))
        self.lambda_dir = float(loss_cfg.get("LAMBDA_DIR", 1.0))
        self.lambda_grip = float(loss_cfg.get("LAMBDA_GRIP", 1.0))
        self.lambda_wm = float(loss_cfg.get("LAMBDA_WM", 1.0))
        assert self.lambda_wm > 0, (
            "MODEL.LOSS.LAMBDA_WM must be > 0: the world model's own objective (SIGReg in "
            "particular) keeps the action loss from collapsing the representation.")
        self._wm_frozen = self.wm_start_epoch > 0
        self._cur_step = 0

        self.loss_metric = LossMetric(cfg)
        self.act_cos_meter = AverageMeter("act_cos")
        self.act_mse_meter = AverageMeter("act_mse")
        self.act_ref_meter = AverageMeter("act_ref")
        self.act_rep_meter = AverageMeter("act_rep")
        self.grip_acc_meter = AverageMeter("grip_acc")
        self.grip_acc_switch_meter = AverageMeter("grip_acc_switch")
        self.wm_mse_meter = AverageMeter("wm_pred_mse")
        self.wm_collapse = CollapseMetric(name="ValWMCollapse")
        self.summary = None
        self.train_log_interval = int(cfg.TRAIN.LOG_INTERVAL)
        self.val_log_interval = int(cfg.TRAIN.get("VAL_LOG_INTERVAL", 100))
        self.collapse_log_interval = int(cfg.TRAIN.get("COLLAPSE_LOG_INTERVAL", 200))
        self._batch_checked = False
        self._last_primary_score = float("-inf")

        n_wm = sum(p.numel() for p in self.wm.parameters() if p.requires_grad)
        n_pol = sum(p.numel() for n, p in self.named_parameters()
                    if p.requires_grad and not n.startswith(_WM_PREFIX))
        logger.info(
            f"{self.name}: BC -- a = pi_phi(z_next, l) on the field's reference prediction "
            f"(D={vis_dim}, FiLM-MLP trunk, out_dim={self.policy_head.out_dim}, "
            f"{param_size(self.policy_head):.1f} MiB) | policy {n_pol/1e6:.1f}M + world model "
            f"{n_wm/1e6:.1f}M trainable (co-trained, start_epoch={self.wm_start_epoch}) | "
            f"gripper_head=on, ctx_action_base={ctx_base!r} | "
            f"vec={self.lambda_vec} dir={self.lambda_dir} grip={self.lambda_grip} wm={self.lambda_wm}")

    def setup(self, summary_writer=None, **kwargs) -> None:
        self.summary = summary_writer

    def param_groups(self, train_cfg):
        default_lr = float(train_cfg.LR)
        wm_lr = float(train_cfg.get("WM_LR", self.wm.wm_cfg.TRAIN.LR))
        wm_params, head_params = [], []
        for name, prm in self.named_parameters():
            if not prm.requires_grad:
                continue
            (wm_params if name.startswith(_WM_PREFIX) else head_params).append(prm)
        groups = [{"params": head_params, "lr": default_lr, "name": "POLICY"}]
        if wm_params:
            groups.append({"params": wm_params, "lr": wm_lr, "name": "WORLD_MODEL"})
        logger.info(f"{self.name}: param groups -- POLICY {len(head_params)} tensors "
                    f"lr={default_lr:g}, WORLD_MODEL {len(wm_params)} tensors lr={wm_lr:g}")
        return groups

    def on_train_epoch_start(self, epoch_idx: int) -> None:
        was_frozen = self._wm_frozen
        self._wm_frozen = epoch_idx < self.wm_start_epoch
        if was_frozen != self._wm_frozen:
            logger.warning(f"{self.name}: world model "
                           f"{'FROZEN' if self._wm_frozen else 'now UPDATING'} at epoch "
                           f"{epoch_idx} (WORLD_MODEL_TRAIN.START_EPOCH={self.wm_start_epoch})")

    def on_before_optimizer_step(self) -> None:
        if self._wm_frozen:
            for p in self.wm.parameters():
                p.grad = None

    def _device(self):
        return next(self.policy_head.parameters()).device

    def _check_batch(self, batch: dict) -> None:
        assert "pixels_next" in batch, (
            "LeWMBCModel needs the world model's own one-step target frame: set "
            "DATASET.*.EMIT_NEXT_FRAME: True.")
        ctx = batch["ctx_pixels"]
        H = self.wm.history_size
        assert ctx.shape[1] == H, (
            f"ctx_pixels has {ctx.shape[1]} context frames, world model HISTORY_SIZE={H} -- set "
            f"DATASET.*.HISTORY_SIZE to {H} in this config")
        n_views = ctx.shape[2] if ctx.dim() == 6 else 1
        assert n_views == self.wm.num_cameras, (
            f"ctx_pixels carries {n_views} camera view(s), world model NUM_CAMERAS="
            f"{self.wm.num_cameras} -- DATASET.*.CAMERAS must match {self.wm.cameras}")
        want = self.wm.frameskip * self.wm.action_dim
        assert batch["ctx_action"].shape[-1] == want, (
            f"ctx_action width {batch['ctx_action'].shape[-1]} != FRAMESKIP*ACTION_DIM={want} -- "
            f"DATASET.*.FRAMESKIP must be {self.wm.frameskip}")
        logger.info(f"{self.name}: batch contract OK (H={H}, views={n_views}, D={self.wm.embed_dim})")
        self._batch_checked = True

    def _forward_policy(self, batch: dict) -> dict:
        """One fused encode of the world model's own `H + P` window, then the shared base point."""
        if not self._batch_checked:
            self._check_batch(batch)
        dev = self._device()
        H, P = self.wm.history_size, self.wm.num_preds
        ctx = batch["ctx_pixels"].to(dev, non_blocking=True)                   # (B,H,[V,]C,h,w)
        nxt = batch["pixels_next"].to(dev, non_blocking=True).unsqueeze(1)     # (B,1,...)
        window_px = torch.cat([ctx, nxt], dim=1)                               # (B,H+P,...)

        emb = self.wm.encode(window_px)                                        # (B,H+P,D)
        assert emb.dim() == 3, (
            f"the world model returned a {emb.dim()}-d latent {tuple(emb.shape)}; BC needs one "
            f"pooled vector per frame")
        ctx_emb = emb[:, :H]
        z_t = ctx_emb[:, -1]
        lang = batch["lang_embed"].to(dev)

        z_next = predict_at_base(self.wm, ctx_emb, batch["ctx_action"].to(dev),
                                 batch["grip_hold"].to(dev))
        a_hat = self.policy_head(z_next, lang)                                 # (B,6)
        grip_logit = self.grip_head(z_t.detach(), lang)
        return {"a_hat": a_hat, "grip_logit": grip_logit, "z_t": z_t, "z_next": z_next,
                "wm_emb": emb[:, :H + P], "wm_pixels": window_px}

    def _wm_preds(self, batch: dict, preds: dict, with_pixels: bool) -> dict:
        return self.wm.window_preds(
            preds["wm_emb"], batch["ctx_action"].to(preds["wm_emb"].device),
            pixels=preds["wm_pixels"] if with_pixels else None)

    def compute_loss(self, preds: dict, gt: dict) -> dict:
        a_hat, grip_logit = preds["a_hat"], preds["grip_logit"]
        expert6 = gt["expert_a_norm7"].to(a_hat.device)[:, :self.wm.pose_dim]
        w = torch.ones(a_hat.shape[0], device=a_hat.device, dtype=a_hat.dtype)
        denom = w.sum().clamp_min(1.0)

        sq = (a_hat - expert6).pow(2).mean(dim=-1)
        cs = 1.0 - F.cosine_similarity(a_hat, expert6, dim=-1, eps=1e-8)
        l_vec = (sq * w).sum() / denom
        l_dir = (cs * w).sum() / denom
        loss = self.lambda_vec * l_vec + self.lambda_dir * l_dir
        loss_dict = {"l_vec": l_vec.detach(), "l_dir": l_dir.detach()}

        tgt = (gt["expert_grip_sign"].to(grip_logit.device) > 0).float()
        w_g = w.to(grip_logit.device, grip_logit.dtype).reshape(-1)
        per = F.binary_cross_entropy_with_logits(grip_logit, tgt, reduction="none")
        l_grip = (w_g * per).sum() / w_g.sum().clamp_min(1e-8)
        loss = loss + self.lambda_grip * l_grip
        loss_dict["l_grip"] = l_grip.detach()

        stats = {}
        with torch.no_grad():
            a_d = a_hat.detach()
            stats["act_cos"] = F.cosine_similarity(a_d, expert6, dim=-1, eps=1e-8).mean()
            sq_err = (a_d - expert6).pow(2).sum(dim=-1).mean()
            sq_ref = expert6.pow(2).sum(dim=-1).mean()
            stats["act_mse"] = sq_err
            stats["act_nmse"] = sq_err / sq_ref.clamp_min(1e-8)
            correct = ((grip_logit > 0).float() == tgt).float()
            stats["grip_acc"] = correct.mean()
            if "expert_grip_switch" in gt:
                m = gt["expert_grip_switch"].to(grip_logit.device) > 0
                if bool(m.any()):
                    stats["grip_acc_switch"] = correct[m].mean()

        wm_loss = self.wm.model.compute_loss(self._wm_preds(gt, preds, with_pixels=True),
                                             gt, self._cur_step)
        loss = loss + self.lambda_wm * wm_loss["loss"]
        loss_dict["l_wm"] = wm_loss["loss"].detach()
        for k, v in wm_loss.items():
            if k != "loss":
                loss_dict[f"wm_{k}"] = v.detach()

        loss_dict["loss"] = loss
        for k, v in stats.items():
            loss_dict[f"_stat_{k}"] = v
        return loss_dict

    def training_step(self, batch: dict, step_idx: int):
        self._cur_step = int(step_idx)
        preds = self._forward_policy(batch)
        loss_dict = self.compute_loss(preds, batch)
        metric_dict = {k: v for k, v in loss_dict.items() if not k.startswith("_stat_")}
        self.loss_metric.feed(metric_dict, batch["expert_a_norm7"].size(0))
        if step_idx % self.train_log_interval == 0 and self.summary is not None:
            for k, v in metric_dict.items():
                self.summary.add_scalar(f"train/{k}", v.item(), step_idx)
            for k in ("act_cos", "act_mse", "act_nmse", "grip_acc", "grip_acc_switch"):
                key = f"_stat_{k}"
                if key in loss_dict:
                    self.summary.add_scalar(f"train/{k}", loss_dict[key].item(), step_idx)
        if (self.summary is not None and step_idx % self.collapse_log_interval == 0
                and hasattr(self.wm.model, "_collapse_stats")):
            with torch.no_grad():
                cstats = self.wm.model._collapse_stats(self._wm_preds(batch, preds, False))
            for k, v in cstats.items():
                self.summary.add_scalar(f"train/wm_collapse/{k}", v, step_idx)
        return preds, loss_dict

    def on_train_finished(self, recorder, epoch_idx: int) -> None:
        recorder.record_loss(self.loss_metric, epoch_idx, comment=f"{self.name}-train")
        self.loss_metric.reset()

    def validation_step(self, batch: dict, step_idx: int):
        preds = self._forward_policy(batch)
        a_hat, grip_logit = preds["a_hat"], preds["grip_logit"]
        expert6 = batch["expert_a_norm7"].to(a_hat.device)[:, :self.wm.pose_dim]
        with torch.no_grad():
            a_d = a_hat.detach()
            cos = F.cosine_similarity(a_d, expert6, dim=-1, eps=1e-8)
            sq_err = (a_d - expert6).pow(2).sum(dim=-1)
            sq_ref = expert6.pow(2).sum(dim=-1)
            k, H = self.wm.frameskip, self.wm.history_size
            sq_rep = None
            if H >= 2:
                ca = batch["ctx_action"].to(expert6.device)
                a_prev = ca.reshape(ca.shape[0], H, k, -1)[:, H - 2, k - 1, :self.wm.pose_dim]
                sq_rep = (a_prev - expert6).pow(2).sum(dim=-1)
        self.act_cos_meter.update_by_mean(cos.mean().item(), cos.numel())
        self.act_mse_meter.update_by_mean(sq_err.mean().item(), sq_err.numel())
        self.act_ref_meter.update_by_mean(sq_ref.mean().item(), sq_ref.numel())
        if sq_rep is not None:
            self.act_rep_meter.update_by_mean(sq_rep.mean().item(), sq_rep.numel())
        tgt = (batch["expert_grip_sign"].to(grip_logit.device) > 0).float()
        correct = ((grip_logit > 0).float() == tgt).float()
        self.grip_acc_meter.update_by_mean(correct.mean().item(), tgt.numel())
        if "expert_grip_switch" in batch:
            m = batch["expert_grip_switch"].to(grip_logit.device) > 0
            if bool(m.any()):
                self.grip_acc_switch_meter.update_by_mean(correct[m].mean().item(), int(m.sum()))
        wm_preds = self._wm_preds(batch, preds, with_pixels=False)
        n = int(a_hat.shape[0])
        self.wm_mse_meter.update_by_mean(
            (wm_preds["pred_emb"] - wm_preds["tgt_emb"]).pow(2).mean().item(), n)
        if hasattr(self.wm.model, "_collapse_stats"):
            self.wm_collapse.feed(self.wm.model._collapse_stats(wm_preds), n)
        if step_idx % self.val_log_interval == 0 and self.summary is not None:
            self.summary.add_scalar("val/act_cos", self.act_cos_meter.avg, step_idx)
        return preds

    def on_val_finished(self, recorder, epoch_idx: int) -> None:
        meters = [self.act_cos_meter, self.act_mse_meter]
        nmse = (self.act_mse_meter.avg / self.act_ref_meter.avg
                if self.act_ref_meter.count > 0 and self.act_ref_meter.avg > 0 else float("nan"))
        rep = (self.act_mse_meter.avg / self.act_rep_meter.avg
               if self.act_rep_meter.count > 0 and self.act_rep_meter.avg > 0 else float("nan"))
        self._last_act_nmse, self._last_act_nmse_rep = nmse, rep
        if self.grip_acc_meter.count > 0:
            meters.append(self.grip_acc_meter)
        if self.wm_mse_meter.count > 0:
            meters.append(self.wm_mse_meter)
        recorder.record_metric(meters, epoch_idx, comment=f"{self.name}-val")
        if self.summary is not None:
            self.summary.add_scalar("val/epoch_act_cos", self.act_cos_meter.avg, epoch_idx)
            self.summary.add_scalar("val/epoch_act_mse", self.act_mse_meter.avg, epoch_idx)
            if nmse == nmse:
                self.summary.add_scalar("val/epoch_act_nmse", nmse, epoch_idx)
            if rep == rep:
                self.summary.add_scalar("val/epoch_act_nmse_rep", rep, epoch_idx)
            if self.grip_acc_meter.count > 0:
                self.summary.add_scalar("val/epoch_grip_acc", self.grip_acc_meter.avg, epoch_idx)
            if self.grip_acc_switch_meter.count > 0:
                self.summary.add_scalar("val/epoch_grip_acc_switch",
                                        self.grip_acc_switch_meter.avg, epoch_idx)
        sw = self.grip_acc_switch_meter
        logger.info(
            f"{self.name} val epoch {epoch_idx}: act_cos={self.act_cos_meter.avg:.4f} "
            f"act_mse={self.act_mse_meter.avg:.4f} act_nmse={nmse:.4f} (1.0 = a_hat=0) "
            f"act_nmse_rep={rep:.4f} (1.0 = repeating the last command)"
            + (f" | grip_acc={self.grip_acc_meter.avg:.4f}" if self.grip_acc_meter.count else "")
            + (f" grip_acc_switch={sw.avg:.4f} (n={sw.count})" if sw.count else ""))
        if self.wm_mse_meter.count > 0:
            collapse = self.wm_collapse.get_measures() if self.wm_collapse.count > 0 else {}
            logger.info(f"{self.name} val epoch {epoch_idx}: "
                        f"wm_pred_mse={self.wm_mse_meter.avg:.5f} | "
                        + " ".join(f"wm_{k}={v:.4f}" for k, v in collapse.items()))
            if self.summary is not None:
                self.summary.add_scalar("val/epoch_wm_pred_mse", self.wm_mse_meter.avg, epoch_idx)
                for k, v in collapse.items():
                    self.summary.add_scalar(f"val_epoch/wm_collapse/{k}", v, epoch_idx)
            if self.wm_collapse.count > 0:
                recorder.record_metric([self.wm_collapse], epoch_idx, comment=f"{self.name}-val")
        self._last_primary_score = (self.act_cos_meter.avg if self.act_cos_meter.count > 0
                                    else float("-inf"))
        logger.info(f"{self.name}: primary_val_score = val act_cos {self._last_primary_score:.4f}")
        for m in (self.act_cos_meter, self.act_mse_meter, self.act_ref_meter, self.act_rep_meter,
                  self.grip_acc_meter, self.grip_acc_switch_meter, self.wm_mse_meter):
            m.reset()
        self.wm_collapse.reset()

    def primary_val_score(self) -> float:
        return getattr(self, "_last_primary_score", float("-inf"))

    def format_metric(self, mode: str) -> str:
        if mode == "train":
            try:
                return (f"loss={self.loss_metric.get_loss('loss'):.4f} | "
                        f"vec={self.loss_metric.get_loss('l_vec'):.4f} | "
                        f"dir={self.loss_metric.get_loss('l_dir'):.4f} | "
                        f"wm={self.loss_metric.get_loss('l_wm'):.4f}")
            except (KeyError, ZeroDivisionError):
                return ""
        if mode == "val" and self.act_cos_meter.count > 0:
            s = f"act_cos={self.act_cos_meter.avg:.4f} | act_mse={self.act_mse_meter.avg:.4f}"
            if self.wm_mse_meter.count > 0:
                s += f" | wm_mse={self.wm_mse_meter.avg:.5f}"
            return s
        return ""

    def forward(self, inputs, step_idx, mode: str = "train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        if mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        raise ValueError(f"Unknown mode: {mode}")
