"""Generalized-gradient progress field over a LeWM world model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.metrics.basic_metric import AverageMeter
from lib.models.lewm.lewm_field_adapter import LeWMFieldAdapter, progress_covector
from lib.models.model_abstraction import ModuleAbstract
from lib.models.potential.action_supervision import force_double_backward_safe
from lib.models.potential.field_gripper import FieldGripper
from lib.models.potential.generalized_gradient import (GeneralizedGradientGeometry,
                                                        operator_regularizer,
                                                        progress_alignment_diagnostics)
from lib.utils.logger import logger
from lib.utils.misc import param_size

from .film_mlp import FiLMMLPHead
from .potential_scaled_common import ScaledPotentialTrainMixin

_WM_PREFIX = "wm."


class GripperSwitchHead(nn.Module):
    def __init__(self, in_dim: int, lang_dim: int, hidden_dim: int = 256, n_out: int = 1):
        super().__init__()
        self.n_out = int(n_out)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim + lang_dim),
            nn.Linear(in_dim + lang_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, self.n_out),
        )

    def forward(self, z: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """z (B,D), lang (B,L) -> (B,) logit; (B,n_out) with several grippers."""
        out = self.net(torch.cat([z, lang], dim=-1))
        return out.squeeze(-1) if self.n_out == 1 else out


class LeWMGGFieldModel(ScaledPotentialTrainMixin, nn.Module, ModuleAbstract):
    """The progress field p_theta(z, l) and its action readout. Trained through
    `LeWMGGFieldCotrainModel`, which supplies the world-model encoding (`_compute_potentials`)."""

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.policy_mode = "field"
        self.cfg = cfg
        self.train_cfg = cfg.TRAIN

        wm_cfg = cfg.WORLD_MODEL
        self.wm = LeWMFieldAdapter(wm_cfg.CONFIG, wm_cfg.CKPT_DIR)
        vis_dim = self.wm.embed_dim

        lang_dim = int(cfg.get("LANG_DIM", 384))
        self.lang_dim = lang_dim
        self.head = self._build_head(cfg, vis_dim, lang_dim)

        self._init_scaled_common(cfg)
        self._init_act_supervision(cfg)
        self._batch_checked = False

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"{self.name}: {n_train:,} trainable params ({param_size(self.head):.1f} MiB "
                    f"potential head) over a co-trained "
                    f"{self.wm.wm_cfg.MODEL.TYPE} (D={vis_dim} pooled state, FiLM-MLP head)")

    def _build_head(self, cfg, vis_dim: int, lang_dim: int) -> nn.Module:
        return FiLMMLPHead(
            latent_dim=vis_dim, lang_dim=lang_dim,
            hidden_dim=int(cfg.get("HIDDEN_DIM", 1024)),
            num_layers=int(cfg.get("NUM_LAYERS", 4)),
            film_hidden=int(cfg.get("FILM_HIDDEN", 256)),
            dropout=float(cfg.get("DROPOUT", 0.0)))

    def _init_act_supervision(self, cfg):
        act_cfg = cfg.ACT_SUPERVISION
        assert bool(act_cfg.get("ENABLED", False)), "MODEL.ACT_SUPERVISION.ENABLED must be true"
        self.lambda_act = float(cfg.LOSS.get("LAMBDA_ACT", 0.0))
        assert self.lambda_act > 0, "MODEL.ACT_SUPERVISION.ENABLED=true needs MODEL.LOSS.LAMBDA_ACT > 0"
        self._init_stages(cfg)

        ctx_base = str(act_cfg.get("CTX_ACTION_BASE", "history"))
        assert ctx_base in ("history", "median"), (
            f"ACT_SUPERVISION.CTX_ACTION_BASE must be 'history' or 'median', got {ctx_base!r}")
        self.wm.ctx_action_base = ctx_base
        assert str(act_cfg.get("CTX_IMAGE_BASE", "history")) == "history", (
            "ACT_SUPERVISION.CTX_IMAGE_BASE must be 'history'")

        query_mode = str(act_cfg.get("QUERY_MODE", "population")).lower()
        assert query_mode == "population", (
            f"ACT_SUPERVISION.QUERY_MODE must be 'population', got {query_mode!r}")
        self.wm.enable_query_population()

        geom_cfg = act_cfg.get("GEOM", {})
        assert (str(geom_cfg.get("MODE", "scalar")) == "scalar" and not bool(geom_cfg.get("NORMALIZE_G", False))
                and float(geom_cfg.get("DROPOUT", 0.0)) == 0.0), (
            "ACT_SUPERVISION.GEOM supports MODE: scalar with NORMALIZE_G: false and DROPOUT: 0")
        self.geom = GeneralizedGradientGeometry(
            in_dim=self.wm.embed_dim,
            action_dim=self.wm.pose_dim,
            hidden_dim=int(geom_cfg.get("HIDDEN_DIM", 256)),
            num_layers=int(geom_cfg.get("NUM_LAYERS", 2)),
            eps=float(geom_cfg.get("EPS", 1e-3)),
        )

        self.lambda_grip = float(cfg.LOSS.get("LAMBDA_GRIP", 1.0))
        gf_cfg = cfg.GRIPPER_FIELD
        assert bool(gf_cfg.get("ENABLED", False)), "MODEL.GRIPPER_FIELD.ENABLED must be true"
        self.field_gripper = FieldGripper(gf_cfg, ctx_base, self.wm.frameskip)

        self.lambda_vec = float(cfg.LOSS.get("LAMBDA_VEC", 1.0))
        self.lambda_dir = float(cfg.LOSS.get("LAMBDA_DIR", 1.0))
        self.lambda_m = float(cfg.LOSS.get("LAMBDA_M", 0.0))
        self.lambda_w = float(cfg.LOSS.get("LAMBDA_W", 1e-3))

        self.act_cos_meter = AverageMeter("act_cos")
        self.act_g_cos_meter = AverageMeter("g_cos")
        self.act_mse_meter = AverageMeter("act_mse")
        self.act_ref_meter = AverageMeter("act_ref")
        self.act_rep_meter = AverageMeter("act_rep")
        self.grip_acc_meter = AverageMeter("grip_acc")
        self.grip_acc_switch_meter = AverageMeter("grip_acc_switch")
        n_geom = sum(p.numel() for p in self.geom.parameters())
        logger.info(
            f"{self.name}: geometry mode={self.geom.mode!r} ({n_geom:,} params, task-independent), "
            f"{self.field_gripper.describe()}, "
            f"query_mode={query_mode!r}, lambda_act={self.lambda_act}, "
            f"vec={self.lambda_vec}, dir={self.lambda_dir}, M={self.lambda_m}, W={self.lambda_w}, "
            f"grip={self.lambda_grip}, td={self.lambda_td}, ctx_action_base={ctx_base!r} | "
            f"stages: 1 -> [0,{self.stage2_start}), 2 -> [{self.stage2_start},{self.stage3_start}), "
            f"3 -> [{self.stage3_start},end)")

    def _init_stages(self, cfg):
        stages = cfg.get("STAGES", {})
        self.stage2_start = int(stages.get("STAGE2_START_EPOCH", 0))
        self.stage3_start = int(stages.get("STAGE3_START_EPOCH", 0))
        assert self.stage2_start <= self.stage3_start, (
            f"MODEL.STAGES.STAGE2_START_EPOCH ({self.stage2_start}) must be <= "
            f"STAGE3_START_EPOCH ({self.stage3_start})")
        self._stage = None
        self._set_stage(0)

    _STAGE_NAMES = {1: "calibrate the scalar potential",
                    2: "learn the action-space geometry (p_theta frozen)",
                    3: "joint adaptation"}

    def _set_stage(self, epoch_idx: int):
        stage = 1 if epoch_idx < self.stage2_start else (2 if epoch_idx < self.stage3_start else 3)
        self._w_p, self._w_geom = {1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}[stage]
        if stage != self._stage:
            logger.warning(f"{self.name}: entering STAGE {stage} at epoch {epoch_idx} "
                           f"(w_potential={self._w_p}, w_geometry={self._w_geom}) -- "
                           f"{self._STAGE_NAMES[stage]}")
        self._stage = stage

    def on_train_epoch_start(self, epoch_idx: int) -> None:
        """Called by scripts/train_general.py at every epoch boundary (resume-safe)."""
        self._set_stage(epoch_idx)

    def on_before_optimizer_step(self) -> None:
        if self._w_p == 0.0:
            for p in self.head.parameters():
                p.grad = None
        if self._w_geom == 0.0:
            for p in self.geom.parameters():
                p.grad = None

    def backward_context(self):
        return force_double_backward_safe()

    def _check_batch(self, batch: dict) -> None:
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
        logger.info(f"{self.name}: batch contract OK (H={H}, views={n_views}, "
                    f"chunk={want}, D={self.wm.embed_dim})")
        self._batch_checked = True

    def _device(self):
        return next(self.head.parameters()).device

    def expert_pose(self, batch: dict) -> torch.Tensor:
        """The `L_geom` pose target: the expert command's pose channels, (B,6)."""
        return batch["expert_a_norm7"].to(self._device())[:, :self.wm.pose_dim]

    def _geom_forward(self, batch: dict, ctx_emb: torch.Tensor, create_graph: bool):
        """-> (a_hat, g LIVE, expert6, M, W, grip_logit, q (B,))."""
        dev = self._device()
        lang = batch["lang_embed"].to(dev)
        if create_graph:
            with force_double_backward_safe():
                g, z_next, q = progress_covector(
                    self.wm, self.head, ctx_emb,
                    batch["ctx_action"].to(dev), lang, batch["grip_hold"].to(dev),
                    create_graph=True)
        else:
            g, z_next, q = progress_covector(
                self.wm, self.head, ctx_emb,
                batch["ctx_action"].to(dev), lang, batch["grip_hold"].to(dev),
                create_graph=False)
        a_hat, M, W = self.geom(z_next.detach(), g)
        grip_logit = self.field_gripper.forward(self, batch, ctx_emb, lang)
        self.wm._query_stats = None
        return a_hat, g, self.expert_pose(batch), M, W, grip_logit, q

    def _compute_geom_loss(self, batch: dict, preds: dict):
        a_hat, g_live, expert6, M, W, grip_logit, q = self._geom_forward(
            batch, preds["ctx_emb"], create_graph=True)
        g = g_live.detach()
        w_act = torch.ones(a_hat.shape[0], device=a_hat.device, dtype=a_hat.dtype)
        denom_act = w_act.sum().clamp_min(1.0)
        sq = (a_hat - expert6).pow(2).mean(dim=-1)                     # (B,)
        cs = 1.0 - F.cosine_similarity(a_hat, expert6, dim=-1, eps=1e-8)
        l_vec = (sq * w_act).sum() / denom_act
        l_dir = (cs * w_act).sum() / denom_act
        l_reg, l_m, l_w = operator_regularizer(M, W, self.lambda_m, self.lambda_w)
        l_geom = self.lambda_vec * l_vec + self.lambda_dir * l_dir + l_reg

        parts = {"l_vec": l_vec.detach(), "l_dir": l_dir.detach(),
                 "l_reg_m": l_m, "l_reg_w": l_w}
        stats = progress_alignment_diagnostics(g, expert6)
        l_grip, gstats = self.field_gripper.loss(batch, w_act, grip_logit)
        l_geom = l_geom + self.lambda_grip * l_grip
        parts["l_grip"] = l_grip.detach()
        stats.update(gstats)
        with torch.no_grad():
            a_m = a_hat.detach()
            stats["act_cos"] = F.cosine_similarity(a_m, expert6, dim=-1, eps=1e-8).mean()
            _sq_err = (a_m - expert6).pow(2).sum(dim=-1).mean()
            _sq_ref = expert6.pow(2).sum(dim=-1).mean()
            stats["act_mse"] = _sq_err
            stats["act_nmse"] = _sq_err / _sq_ref.clamp_min(1e-8)
            stats["m_mean"] = M.diagonal(dim1=-2, dim2=-1).mean()
            stats["w_fro"] = W.pow(2).sum(dim=(-1, -2)).sqrt().mean()
        return l_geom, parts, stats

    def compute_loss(self, preds: dict, gt: dict) -> dict:
        loss_dict = super().compute_loss(preds, gt)
        l_geom, parts, stats = self._compute_geom_loss(gt, preds)
        loss_dict["loss"] = self._w_p * loss_dict["loss"] + self._w_geom * self.lambda_act * l_geom
        loss_dict["l_act"] = l_geom.detach()
        loss_dict.update(parts)
        for k, v in stats.items():
            loss_dict[f"_stat_{k}"] = v
        loss_dict["_stat_stage"] = torch.as_tensor(float(self._stage))
        return loss_dict

    def training_step(self, batch: dict, step_idx: int):
        preds, loss_dict = super().training_step(batch, step_idx)
        if step_idx % self.train_log_interval == 0 and self.summary is not None:
            for k in ("g_cos", "g_cos_pos", "g_norm", "g_crit", "act_cos", "act_mse",
                      "act_nmse", "m_mean", "w_fro",
                      "grip_acc", "grip_acc_switch", "grip_dz", "grip_d_steps", "grip_region_frac",
                      "grip_recall", "grip_false", "grip_recall_dwell", "grip_false_dwell",
                      "stage"):
                key = f"_stat_{k}"
                if key in loss_dict:
                    self.summary.add_scalar(f"train/{k}", loss_dict[key].item(), step_idx)
        return preds, loss_dict

    def validation_step(self, batch: dict, step_idx: int):
        preds = super().validation_step(batch, step_idx)
        a_hat, g_live, expert6, _, _, grip_logit, q = self._geom_forward(
            batch, preds["ctx_emb"], create_graph=False)
        with torch.no_grad():
            g = g_live.detach()
            a_m = a_hat.detach()
            cos = F.cosine_similarity(a_m, expert6, dim=-1, eps=1e-8)
            g_cos = F.cosine_similarity(g, expert6, dim=-1, eps=1e-8)
            sq_err = (a_m - expert6).pow(2).sum(dim=-1)
            sq_ref = expert6.pow(2).sum(dim=-1)
            k, H = self.wm.frameskip, self.wm.history_size
            sq_rep = None
            if H >= 2:
                ca = batch["ctx_action"].to(expert6.device)
                a_prev = ca.reshape(ca.shape[0], H, k, -1)[:, H - 2, k - 1, :self.wm.pose_dim]
                sq_rep = (a_prev - expert6).pow(2).sum(dim=-1)
        self.act_cos_meter.update_by_mean(cos.mean().item(), cos.numel())
        self.act_g_cos_meter.update_by_mean(g_cos.mean().item(), g_cos.numel())
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
        self.field_gripper.update_val(batch)
        return preds

    def on_val_finished(self, recorder, epoch_idx: int) -> None:
        """Log the action and gripper meters, then set the checkpoint-selection score."""
        act_cos_avg = self.act_cos_meter.avg if self.act_cos_meter.count > 0 else None
        lang_gate_ok = self.lambda_lang <= 0 or self.lang_acc_meter.avg > 0.5

        self.field_gripper.log_val(self.summary, logger, self.name, epoch_idx)
        if self.grip_acc_meter.count > 0:
            if self.summary is not None:
                self.summary.add_scalar("val/epoch_grip_acc", self.grip_acc_meter.avg, epoch_idx)
            sw = self.grip_acc_switch_meter
            sw_txt = f" grip_acc_switch={sw.avg:.4f} (n={sw.count})" if sw.count > 0 else ""
            if self.summary is not None and sw.count > 0:
                self.summary.add_scalar("val/epoch_grip_acc_switch", sw.avg, epoch_idx)
            logger.info(f"{self.name} val epoch {epoch_idx}: "
                        f"grip_acc={self.grip_acc_meter.avg:.4f}{sw_txt}")
            self.grip_acc_meter.reset()
            self.grip_acc_switch_meter.reset()
        if self.act_ref_meter.count > 0 and self.act_ref_meter.avg > 0:
            nmse = self.act_mse_meter.avg / self.act_ref_meter.avg
            self._last_act_nmse = nmse
            rep = (self.act_mse_meter.avg / self.act_rep_meter.avg
                   if self.act_rep_meter.count > 0 and self.act_rep_meter.avg > 0 else float("nan"))
            self._last_act_nmse_rep = rep
            if self.summary is not None:
                self.summary.add_scalar("val/epoch_act_mse", self.act_mse_meter.avg, epoch_idx)
                self.summary.add_scalar("val/epoch_act_nmse", nmse, epoch_idx)
                if rep == rep:
                    self.summary.add_scalar("val/epoch_act_nmse_rep", rep, epoch_idx)
            logger.info(f"{self.name} val epoch {epoch_idx}: "
                        f"act_mse={self.act_mse_meter.avg:.4f} "
                        f"act_nmse={nmse:.4f} (1.0 = a_hat=0) "
                        f"act_nmse_rep={rep:.4f} (1.0 = repeating the last command)")
            self.act_ref_meter.reset()
            self.act_rep_meter.reset()
        super().on_val_finished(recorder, epoch_idx)

        if act_cos_avg is not None:
            self._last_primary_score = act_cos_avg if lang_gate_ok else float("-inf")
            logger.info(f"{self.name}: primary_val_score = val act_cos {act_cos_avg:.4f} "
                        f"(lang gate {'ok' if lang_gate_ok else 'FAILED -> -inf'}); "
                        f"macro_spearman={getattr(self, '_last_macro_spearman', float('nan')):.4f}")

    def format_metric(self, mode: str) -> str:
        s = super().format_metric(mode)
        if mode == "train":
            try:
                s += (f" | stage={self._stage} | geom={self.loss_metric.get_loss('l_act'):.4f}"
                      f" | vec={self.loss_metric.get_loss('l_vec'):.4f}")
            except (KeyError, ZeroDivisionError):
                pass
        elif mode == "val" and self.act_cos_meter.count > 0:
            s += f" | act_cos={self.act_cos_meter.avg:.4f}"
            if self.act_mse_meter.count > 0:
                s += f" | act_mse={self.act_mse_meter.avg:.4f}"
        return s
