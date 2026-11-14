import torch
import torch.nn as nn
from torchvision.utils import make_grid

from lib.metrics.basic_metric import LossMetric
from lib.metrics.collapse import CollapseMetric, action_sensitivity, embedding_collapse_stats
from lib.metrics.jacobian import action_jacobian_stats
from lib.metrics.similarity import PSNR
from lib.utils.builder import MODEL
from lib.utils.logger import logger
from lib.utils.recorder import Recorder
from lib.utils.misc import param_size, param_count
from lib.utils.net_utils import init_weights

from .jepa import JEPA
from .module import ARPredictor, Embedder, MLP, SIGReg, temporal_centered_residual
from lib.models.model_abstraction import ModuleAbstract

try:
    from transformers import ViTConfig, ViTModel
except ImportError as exc:
    raise ImportError("HuggingFace `transformers` is required. Install with: pip install transformers") from exc


def _build_vit(hidden_size: int, image_size: int, patch_size: int) -> ViTModel:
    """Build a ViT with DeiT-Tiny-style hyperparameters (trained from scratch)."""
    cfg = ViTConfig(
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=hidden_size * 4,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        add_pooling_layer=False,
    )
    return ViTModel(cfg)


@MODEL.register_module()
class LeWM(nn.Module, ModuleAbstract):
    """The le-wm JEPA world model."""

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_cfg = cfg.TRAIN
        self.data_preset_cfg = cfg.DATA_PRESET

        embed_dim = cfg.EMBED_DIM  # 192
        history_size = cfg.HISTORY_SIZE  # 3
        num_preds = cfg.NUM_PREDS  # 1
        action_dim = cfg.ACTION_DIM  # 2  (raw per-step dim)
        frameskip = cfg.FRAMESKIP  # 5
        enc_hidden = cfg.ENCODER_HIDDEN_SIZE  # 192

        self.history_size = history_size
        self.num_preds = num_preds

        image_size = cfg.DATA_PRESET.IMAGE_SIZE  # 224
        patch_size = cfg.DATA_PRESET.PATCH_SIZE  # 14

        self.num_cameras = cfg.get("NUM_CAMERAS", 1)

        proj_in = enc_hidden * self.num_cameras     # one CLS token per camera view, concatenated

        encoder = _build_vit(enc_hidden, image_size, patch_size)

        predictor = ARPredictor(
            num_frames=history_size,
            input_dim=embed_dim,
            hidden_dim=enc_hidden,
            output_dim=enc_hidden,
            depth=cfg.PREDICTOR_DEPTH,
            heads=cfg.PREDICTOR_HEADS,
            mlp_dim=cfg.PREDICTOR_MLP_DIM,
            dim_head=cfg.PREDICTOR_DIM_HEAD,
            dropout=cfg.PREDICTOR_DROPOUT,
            emb_dropout=0.0,
        )

        effective_act_dim = frameskip * action_dim  # 10 for pusht
        action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

        projector = MLP(
            input_dim=proj_in,   # per-view pooled widths concatenated, or one token's dim
            hidden_dim=2048,
            output_dim=embed_dim,
            norm_fn=nn.BatchNorm1d,
        )
        pred_proj = MLP(
            input_dim=enc_hidden,
            hidden_dim=2048,
            output_dim=embed_dim,
            norm_fn=nn.BatchNorm1d,
        )

        if enc_hidden < embed_dim:
            logger.warning(
                "LeWM: ENCODER_HIDDEN_SIZE (%d) < EMBED_DIM (%d). `pred_proj` is sized from the "
                "ENCODER width, so the predictor bottlenecks the %d-d latent through %d numbers "
                "and the normalized prediction error cannot go below %.2f by rank alone. Set "
                "ENCODER_HIDDEN_SIZE == EMBED_DIM unless you intend that floor.",
                enc_hidden, embed_dim, embed_dim, enc_hidden,
                (embed_dim - enc_hidden) / embed_dim)

        self.jepa = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
        self.sigreg = SIGReg(knots=17, num_proj=1024)

        for key in ("POOLING", "TOKENS", "USE_DECODER", "SIGREG_TARGET", "AUX_DECODE", "USE_TEMPORAL_VAR_REG"):
            assert not cfg.get(key, None), f"MODEL.{key} is not supported"
        self.sigreg_weight = cfg.LOSS.SIGREG_WEIGHT  # 0.09

        self.loss_metric = LossMetric(cfg)
        self.val_psnr = PSNR()

        self.train_collapse = CollapseMetric(name="TrainCollapse")
        self.val_collapse = CollapseMetric(name="ValCollapse")
        self.collapse_log_interval = cfg.TRAIN.get("COLLAPSE_LOG_INTERVAL", cfg.TRAIN.LOG_INTERVAL)
        self.val_action_sensitivity = cfg.get("VAL_ACTION_SENSITIVITY", True)
        self.val_jacobian = cfg.get("VAL_JACOBIAN", True)
        self.jacobian_sub_batch = cfg.get("JACOBIAN_SUB_BATCH", 64)
        self.jacobian_num_probes = cfg.get("JACOBIAN_NUM_PROBES", 8)
        self.jacobian_log_interval = cfg.TRAIN.get("JACOBIAN_LOG_INTERVAL", 50)
        self.val_jacobian_metric = CollapseMetric(name="ValJacobian")
        self._last_val_step = 0

        self.summary = None
        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        self.val_log_interval = cfg.TRAIN.VAL_LOG_INTERVAL

        self.init_weights()
        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(f"{self.name}: {param_count(self):.2f}M params ({param_size(self):.1f} MiB)")

    def init_weights(self):
        init_weights(self, pretrained=self.cfg.PRETRAINED)

    def setup(self, summary_writer=None, **kwargs):
        self.summary = summary_writer

    def _copy_baseline_emb(self, emb):
        """The embedding each target is predicted FROM, i.e. the trivial no-change prediction."""
        return emb[:, :self.history_size]

    def _collapse_stats(self, preds):
        return embedding_collapse_stats(
            preds["emb"], preds["pred_emb"], preds["tgt_emb"], self._copy_baseline_emb(preds["emb"]))

    def _jacobian_stats(self, preds, batch):
        n = min(self.jacobian_sub_batch, preds["emb"].size(0))
        emb = preds["emb"][:n].detach()
        ctx_emb = emb[:, :self.history_size]
        a_ctx = batch["action"][:n, :self.history_size].to(emb.device).detach()

        def fn(a):
            return self.jepa.predict(ctx_emb, self.jepa.action_encoder(a))

        return action_jacobian_stats(
            fn, a_ctx,
            ref_delta=preds["tgt_emb"][:n].detach() - self._copy_baseline_emb(emb),
            num_probes=self.jacobian_num_probes,
        )

    def compute_loss(self, preds, gt, step_idx):
        emb = preds["emb"]  # (B, T, D)
        pred_emb = preds["pred_emb"]
        tgt_emb = preds["tgt_emb"]

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()

        resid = temporal_centered_residual(emb)                     # (B,T,D), for the logged stats
        sreg_loss = self.sigreg(emb.transpose(0, 1), global_step=step_idx)   # (T,B,D)
        total_loss = pred_loss + self.sigreg_weight * sreg_loss

        loss_dict = {
            "loss": total_loss,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sreg_loss.detach(),
        }

        with torch.no_grad():
            per_dim = lambda x: x.var(dim=0, unbiased=False).mean().detach()   # (B,T,D) -> scalar
            loss_dict["tc_var_resid"] = per_dim(resid)
            loss_dict["tc_var_persist"] = per_dim(emb - resid)
            loss_dict["tc_var_total"] = per_dim(emb)

        return loss_dict

    def _forward_impl(self, batch, **kwargs):
        device = next(self.parameters()).device
        batch["pixels"] = batch["pixels"].to(device, non_blocking=True)
        batch["action"] = torch.nan_to_num(batch["action"].to(device, non_blocking=True), 0.0)
        preds = self.jepa.encode(batch)

        emb = preds["emb"]        # (B, T, D)
        act_emb = preds["act_emb"]  # (B, T, A_emb)
        ctx_emb = emb[:, :self.history_size]
        ctx_act = act_emb[:, :self.history_size]
        preds["tgt_emb"] = emb[:, self.num_preds:]
        preds["pred_emb"] = self.jepa.predict(ctx_emb, ctx_act)

        return preds

    def training_step(self, batch, step_idx, **kwargs):
        batch_size = batch["action"].size(0)
        preds = self._forward_impl(batch, **kwargs)
        loss_dict = self.compute_loss(preds, batch, step_idx)

        self.loss_metric.feed(loss_dict, batch_size)

        if step_idx % self.train_log_interval == 0:
            for k, v in loss_dict.items():
                self.summary.add_scalar(f"{k}", v.item(), step_idx)

        if step_idx % self.collapse_log_interval == 0:
            stats = self._collapse_stats(preds)
            self.train_collapse.feed(stats, batch_size)
            if self.summary is not None:
                for k, v in stats.items():
                    self.summary.add_scalar(f"collapse/{k}", v, step_idx)

        return preds, loss_dict

    def on_train_finished(self, recorder: Recorder, epoch_idx: int):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric([self.train_collapse], epoch_idx, comment=comment)
        self.loss_metric.reset()
        self.train_collapse.reset()

    def validation_step(self, batch, step_idx):
        preds = self._forward_impl(batch)

        pred_emb = preds["pred_emb"]
        tgt_emb = preds["tgt_emb"]
        psnr = self.val_psnr.feed(pred_emb, tgt_emb)

        self.summary.add_scalar("val/epoch_psnr", self.val_psnr.get_result(), step_idx)

        self._last_val_step = step_idx
        stats = self._collapse_stats(preds)
        if self.val_action_sensitivity:
            emb = preds["emb"]
            stats["action_sensitivity"] = action_sensitivity(
                self.jepa.predict,
                emb[:, :self.history_size],
                preds["act_emb"][:, :self.history_size],
                tgt_emb,
                self._copy_baseline_emb(emb),
            )
        self.val_collapse.feed(stats, batch["action"].size(0))

        if self.val_jacobian and step_idx % self.jacobian_log_interval == 0:
            jac = self._jacobian_stats(preds, batch)
            self.val_jacobian_metric.feed(jac, batch["action"].size(0))
            if self.summary is not None:
                for k, v in jac.items():
                    self.summary.add_scalar(f"val/jacobian/{k}", v, step_idx)

        if step_idx % self.val_log_interval == 0 and self.summary is not None:
            self.summary.add_scalar("val/psnr", psnr, step_idx)
            for k, v in stats.items():
                self.summary.add_scalar(f"val/collapse/{k}", v, step_idx)

            pixels = preds["pixels"]  # (B, T, C, H, W) or (B, T, V, C, H, W), values in [0, 1]
            n = min(4, pixels.size(0))
            ctx_imgs = pixels[:n, :self.history_size]  # (n, H_s, [V,] C, H, W)
            tgt_imgs = pixels[:n, self.history_size:]  # (n, P,   [V,] C, H, W)
            img_shape = pixels.shape[-3:]
            n_views = pixels.size(2) if pixels.dim() == 6 else 1
            ctx_flat = ctx_imgs.reshape(-1, *img_shape)  # (n*H_s*V, C, H, W)
            tgt_flat = tgt_imgs.reshape(-1, *img_shape)  # (n*P*V,   C, H, W)
            grid_ctx = make_grid(ctx_flat, nrow=self.history_size * n_views, normalize=False)
            grid_tgt = make_grid(tgt_flat, nrow=self.num_preds * n_views, normalize=False)
            self.summary.add_image("val/context_frames", grid_ctx, step_idx)
            self.summary.add_image("val/target_frames", grid_tgt, step_idx)

        return preds

    def on_val_finished(self, recorder: Recorder, epoch_idx: int):
        comment = f"{self.name}-val"
        recorder.record_metric([self.val_psnr, self.val_collapse, self.val_jacobian_metric],
                               epoch_idx, comment=comment)
        if self.summary is not None:
            for k, v in self.val_collapse.get_measures().items():
                self.summary.add_scalar(f"val_epoch/collapse/{k}", v, self._last_val_step)
            for k, v in self.val_jacobian_metric.get_measures().items():
                self.summary.add_scalar(f"val_epoch/jacobian/{k}", v, self._last_val_step)
        logger.info(f"{self.name} epoch {epoch_idx} collapse watch | " + " | ".join(
            f"{k}: {v:.4f}" for k, v in self.val_collapse.get_measures().items()))
        if not self.val_jacobian_metric.is_empty():
            logger.info(f"{self.name} epoch {epoch_idx} jacobian watch | " + " | ".join(
                f"{k}: {v:.4f}" for k, v in self.val_jacobian_metric.get_measures().items()))
        self.val_psnr.reset()
        self.val_collapse.reset()
        self.val_jacobian_metric.reset()

    def format_metric(self, mode: str) -> str:
        if mode == "train":
            s = (f"loss: {self.loss_metric.get_loss('loss'):.4f} | "
                 f"P: {self.loss_metric.get_loss('pred_loss'):.4f} | "
                 f"R: {self.loss_metric.get_loss('sigreg_loss'):.4f}")
            if not self.train_collapse.is_empty():
                s += (f" | TB: {self.train_collapse.get('time_batch_std_ratio'):.3f}"
                      f" | dR2: {self.train_collapse.get('delta_r2'):.3f}")
            s += (f" | V: {self.loss_metric.get_loss('tc_var_persist'):.2f}"
                  f"/{self.loss_metric.get_loss('tc_var_resid'):.2f}")
            return s
        if mode == "val" and self.val_psnr.avg_meter.count > 0:
            return (f"PSNR: {self.val_psnr.get_result():.2f} dB | "
                    f"TB: {self.val_collapse.get('time_batch_std_ratio'):.3f} | "
                    f"P/C: {self.val_collapse.get('pred_vs_copy_ratio'):.3f} | "
                    f"dR2: {self.val_collapse.get('delta_r2'):.3f} | "
                    f"Jgain: {self.val_jacobian_metric.get('jac_gain_rel'):.3f} | "
                    f"Jrank: {self.val_jacobian_metric.get('jac_eff_rank_frac'):.3f}")
        return ""

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")
