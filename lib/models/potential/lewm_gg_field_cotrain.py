"""JOINT optimization of the LeWM world model and the generalized-gradient progress head."""


import torch

from lib.metrics.basic_metric import AverageMeter
from lib.metrics.collapse import CollapseMetric
from lib.utils.builder import MODEL
from lib.utils.config import CN
from lib.utils.logger import logger

from .lewm_gg_field import LeWMGGFieldModel, _WM_PREFIX

def build_field_cfg(cfg: CN) -> CN:
    """The co-train `MODEL` node -> the `MODEL` node `LeWMGGFieldModel.__init__` expects: the world
    model's config and checkpoint (`LEWM_CONFIG`, `LEWM_CKPT`) become `WORLD_MODEL`."""
    wmt = cfg.get("WORLD_MODEL_TRAIN", CN())
    assert bool(wmt.get("ENABLED", True)), "MODEL.WORLD_MODEL_TRAIN.ENABLED must be true (co-training)"
    assert str(wmt.get("BN_MODE", "train")).lower() == "train", "MODEL.WORLD_MODEL_TRAIN.BN_MODE must be 'train'"
    merged = cfg.clone()
    merged.defrost()
    merged.set_new_allowed(True)
    for k in ("LEWM_CONFIG", "LEWM_CKPT", "WORLD_MODEL_TRAIN"):
        merged.pop(k, None)
    merged.WORLD_MODEL = CN({
        "CONFIG": str(cfg.LEWM_CONFIG),
        "CKPT_DIR": str(cfg.LEWM_CKPT),
    })
    return merged


@MODEL.register_module()
class LeWMGGFieldCotrainModel(LeWMGGFieldModel):
    def __init__(self, cfg):
        self.cotrain_cfg = cfg
        super().__init__(build_field_cfg(cfg))

        wmt = cfg.get("WORLD_MODEL_TRAIN", CN())
        self.wm_start_epoch = int(wmt.get("START_EPOCH", 0))
        self.lambda_wm = float(self.cfg.LOSS.get("LAMBDA_WM", 1.0))
        assert self.lambda_wm > 0, (
            "MODEL.LOSS.LAMBDA_WM must be > 0: the world model's own objective (SIGReg in "
            "particular) keeps the field losses from collapsing the representation.")
        self._wm_frozen = self.wm_start_epoch > 0
        self._cur_step = 0

        self.wm_mse_meter = AverageMeter("wm_pred_mse")
        self.wm_collapse = CollapseMetric(name="ValWMCollapse")
        self.collapse_log_interval = int(self.cfg.TRAIN.get("COLLAPSE_LOG_INTERVAL", 200))

        n_wm = sum(p.numel() for p in self.wm.parameters() if p.requires_grad)
        n_field = sum(p.numel() for n, p in self.named_parameters()
                      if p.requires_grad and not n.startswith(_WM_PREFIX))
        logger.info(
            f"{self.name}: JOINT -- world model {n_wm/1e6:.1f}M trainable "
            f"(start_epoch={self.wm_start_epoch}) + field {n_field/1e6:.1f}M | "
            f"lambda_wm={self.lambda_wm} lambda_act={self.lambda_act} "
            f"| world model cfg={cfg.LEWM_CONFIG}")

    def on_train_epoch_start(self, epoch_idx: int) -> None:
        super().on_train_epoch_start(epoch_idx)
        was_frozen = self._wm_frozen
        self._wm_frozen = epoch_idx < self.wm_start_epoch
        if was_frozen != self._wm_frozen:
            logger.warning(f"{self.name}: world model "
                           f"{'FROZEN' if self._wm_frozen else 'now UPDATING'} at epoch "
                           f"{epoch_idx} (WORLD_MODEL_TRAIN.START_EPOCH={self.wm_start_epoch})")

    def on_before_optimizer_step(self) -> None:
        """The parent's stage freeze, plus the world model's own warm-up freeze."""
        super().on_before_optimizer_step()
        if self._wm_frozen:
            for p in self.wm.parameters():
                p.grad = None

    def param_groups(self, train_cfg):
        default_lr = float(train_cfg.LR)
        wm_lr = float(train_cfg.get("WM_LR", self.wm.wm_cfg.TRAIN.LR))
        wm_params, field_params = [], []
        for name, prm in self.named_parameters():
            if not prm.requires_grad:
                continue
            (wm_params if name.startswith(_WM_PREFIX) else field_params).append(prm)
        groups = [{"params": field_params, "lr": default_lr, "name": "FIELD"}]
        if wm_params:
            groups.append({"params": wm_params, "lr": wm_lr, "name": "WORLD_MODEL"})
        logger.info(f"{self.name}: param groups -- FIELD {len(field_params)} tensors lr={default_lr:g}, "
                    f"WORLD_MODEL {len(wm_params)} tensors lr={wm_lr:g}")
        return groups

    def _check_batch(self, batch: dict) -> None:
        assert "pixels_next" in batch, (
            "co-training needs the world model's own one-step target: set "
            "DATASET.*.EMIT_NEXT_FRAME: True in this config. `pixels_tp1` sits at a RANDOM gap "
            "from GAP_SET and is NOT that frame.")
        super()._check_batch(batch)

    def _compute_potentials(self, batch: dict) -> dict:
        """One fused encode for every frame this step needs, WITH a graph into the encoder."""
        if not self._batch_checked:
            self._check_batch(batch)
        dev = self._device()
        H, P = self.wm.history_size, self.wm.num_preds
        ctx = batch["ctx_pixels"].to(dev, non_blocking=True)                       # (B,H,[V,]C,h,w)
        nxt = batch["pixels_next"].to(dev, non_blocking=True).unsqueeze(1)         # (B,1,...)
        later = batch["pixels_tp1"].to(dev, non_blocking=True).unsqueeze(1)        # (B,1,...)
        window_px = torch.cat([ctx, nxt], dim=1)                                   # (B,H+P,...)
        all_px = torch.cat([window_px, later], dim=1)                              # (B,H+P+1,...)

        emb = self.wm.encode(all_px)                                               # (B,H+P+1,D)
        assert emb.dim() == 3, (
            f"the world model returned a {emb.dim()}-d latent {tuple(emb.shape)}; the field "
            f"needs one pooled vector per frame")
        ctx_emb = emb[:, :H]
        z_t = ctx_emb[:, -1]
        z_tp1 = emb[:, H + P]
        lang = batch["lang_embed"].to(dev)

        preds = {
            "p_t": self.head(z_t, lang).squeeze(-1),
            "p_tp1": self.head(z_tp1, lang).squeeze(-1),
            "p_tp1_wrong": None,
            "ctx_emb": ctx_emb,
            "z_t": z_t,
            "wm_emb": emb[:, :H + P],
            "wm_pixels": window_px,
        }
        if "lang_embed_wrong" in batch:
            lw = batch["lang_embed_wrong"].to(dev)
            preds["p_tp1_wrong"] = self.head(z_tp1, lw).squeeze(-1)
        return preds

    def _wm_preds(self, batch: dict, preds: dict, with_pixels: bool) -> dict:
        return self.wm.window_preds(
            preds["wm_emb"], batch["ctx_action"].to(preds["wm_emb"].device),
            pixels=preds["wm_pixels"] if with_pixels else None)

    def compute_loss(self, preds: dict, gt: dict) -> dict:
        """The inherited field objective PLUS the world model's own, on the same window."""
        loss_dict = super().compute_loss(preds, gt)
        wm_preds = self._wm_preds(gt, preds, with_pixels=True)
        wm_loss = self.wm.model.compute_loss(wm_preds, gt, self._cur_step)
        loss_dict["loss"] = loss_dict["loss"] + self.lambda_wm * wm_loss["loss"]
        loss_dict["l_wm"] = wm_loss["loss"].detach()
        for k, v in wm_loss.items():
            if k != "loss":
                loss_dict[f"wm_{k}"] = v.detach()
        return loss_dict

    def training_step(self, batch: dict, step_idx: int):
        self._cur_step = int(step_idx)
        preds, loss_dict = super().training_step(batch, step_idx)
        if (self.summary is not None and step_idx % self.collapse_log_interval == 0
                and hasattr(self.wm.model, "_collapse_stats")):
            with torch.no_grad():
                stats = self.wm.model._collapse_stats(self._wm_preds(batch, preds, False))
            for k, v in stats.items():
                self.summary.add_scalar(f"train/wm_collapse/{k}", v, step_idx)
        return preds, loss_dict

    def validation_step(self, batch: dict, step_idx: int):
        preds = super().validation_step(batch, step_idx)
        wm_preds = self._wm_preds(batch, preds, with_pixels=False)
        n = int(preds["p_t"].numel())
        mse = (wm_preds["pred_emb"] - wm_preds["tgt_emb"]).pow(2).mean()
        self.wm_mse_meter.update_by_mean(mse.item(), n)
        if hasattr(self.wm.model, "_collapse_stats"):
            self.wm_collapse.feed(self.wm.model._collapse_stats(wm_preds), n)
        return preds

    def on_val_finished(self, recorder, epoch_idx: int) -> None:
        if self.wm_mse_meter.count > 0:
            if self.summary is not None:
                self.summary.add_scalar("val/epoch_wm_pred_mse", self.wm_mse_meter.avg, epoch_idx)
            collapse = self.wm_collapse.get_measures() if self.wm_collapse.count > 0 else {}
            logger.info(f"{self.name} val epoch {epoch_idx}: wm_pred_mse="
                        f"{self.wm_mse_meter.avg:.5f} | "
                        + " ".join(f"wm_{k}={v:.4f}" for k, v in collapse.items()))
            if self.summary is not None:
                for k, v in collapse.items():
                    self.summary.add_scalar(f"val_epoch/wm_collapse/{k}", v, epoch_idx)
            recorder.record_metric([self.wm_collapse], epoch_idx, comment=f"{self.name}-val")
            self.wm_mse_meter.reset()
            self.wm_collapse.reset()
        super().on_val_finished(recorder, epoch_idx)

    def format_metric(self, mode: str) -> str:
        s = super().format_metric(mode)
        if mode == "train":
            try:
                s += f" | wm={self.loss_metric.get_loss('l_wm'):.4f}"
            except (KeyError, ZeroDivisionError):
                pass
        elif mode == "val" and self.wm_mse_meter.count > 0:
            s += f" | wm_mse={self.wm_mse_meter.avg:.5f}"
        return s
