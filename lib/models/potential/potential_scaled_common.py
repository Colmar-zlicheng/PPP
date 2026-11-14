
import numpy as np
import torch
import torch.nn.functional as F

from lib.metrics.basic_metric import AverageMeter, LossMetric
from lib.utils.logger import logger
from lib.utils.recorder import Recorder


def compute_scaled_potential_loss(
    p_t: torch.Tensor,
    p_tp1: torch.Tensor,
    p_tp1_wrong: torch.Tensor,
    prog_t: torch.Tensor,
    prog_tp1: torch.Tensor,
    *,
    lambda_mc: float = 1.0,
    lambda_rank: float = 0.5,
    rank_tau: float = 0.05,
    rank_min_delta: float = 0.0,
    lambda_boundary: float = 1.0,
    terminal_thresh: float = 0.95,
    nonterminal_eps: float = 0.05,
    lambda_lang: float = 0.5,
    lang_margin: float = 0.2,
    lambda_td: float = 0.0,
) -> dict:
    """p_t/p_tp1/p_tp1_wrong: (B,) scalar potentials. prog_t/prog_tp1: (B,)
    in [0,1], frame position / (ep_len-1). p_tp1_wrong may be None (then
    L_lang is forced to 0, same as lambda_lang<=0)."""
    dev = p_t.device
    tgt_t, tgt_tp1 = prog_t - 1.0, prog_tp1 - 1.0

    l_mc = 0.5 * (F.mse_loss(p_t, tgt_t) + F.mse_loss(p_tp1, tgt_tp1))

    l_rank = torch.zeros((), device=dev)
    rank_kept_frac = torch.ones((), device=dev)
    if lambda_rank > 0:
        per_pair = F.softplus(-(p_tp1 - p_t) / rank_tau)
        keep = (tgt_tp1 - tgt_t) > rank_min_delta
        n_keep = int(keep.sum())
        rank_kept_frac = keep.float().mean()
        if n_keep == keep.numel():
            l_rank = per_pair.mean()
        elif n_keep > 0:
            l_rank = per_pair[keep].mean()

    l_boundary = torch.zeros((), device=dev)
    if lambda_boundary > 0:
        p_all = torch.cat([p_t, p_tp1])
        is_terminal = torch.cat([prog_t, prog_tp1]) >= terminal_thresh
        l_term = p_all[is_terminal].pow(2).mean() if is_terminal.any() else torch.zeros((), device=dev)
        l_nonterm = (F.relu(p_all[~is_terminal] + nonterminal_eps).pow(2).mean()
                     if (~is_terminal).any() else torch.zeros((), device=dev))
        l_boundary = l_term + l_nonterm

    l_lang = torch.zeros((), device=dev)
    if lambda_lang > 0 and p_tp1_wrong is not None:
        w = prog_tp1
        l_lang = (w * F.relu(lang_margin - (p_tp1 - p_tp1_wrong))).mean()

    l_td = torch.zeros((), device=dev)
    if lambda_td > 0:
        l_td = ((p_tp1 - p_t) - (tgt_tp1 - tgt_t)).pow(2).mean()

    loss = (lambda_mc * l_mc + lambda_rank * l_rank
            + lambda_boundary * l_boundary + lambda_lang * l_lang + lambda_td * l_td)
    out = {
        "loss": loss,
        "l_mc": l_mc.detach(),
        "l_rank": l_rank.detach(),
        "_stat_rank_kept_frac": rank_kept_frac.detach(),
        "l_boundary": l_boundary.detach(),
        "l_lang": l_lang.detach(),
    }
    if lambda_td > 0:
        out["l_td"] = l_td.detach()
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


class ScaledPotentialTrainMixin:
    def _init_scaled_common(self, cfg):
        loss_cfg = cfg.LOSS
        self.lambda_mc = float(loss_cfg.get("LAMBDA_MC", 1.0))
        self.lambda_rank = float(loss_cfg.get("LAMBDA_RANK", 0.5))
        self.rank_tau = float(loss_cfg.get("RANK_TAU", 0.05))
        self.rank_min_delta = float(loss_cfg.get("RANK_MIN_DELTA", 0.0))
        self.lambda_boundary = float(loss_cfg.get("LAMBDA_BOUNDARY", 1.0))
        self.terminal_thresh = float(loss_cfg.get("TERMINAL_THRESH", 0.95))
        self.nonterminal_eps = float(loss_cfg.get("NONTERMINAL_EPS", 0.05))
        self.lambda_lang = float(loss_cfg.get("LAMBDA_LANG", 0.5))
        self.lang_margin = float(loss_cfg.get("LANG_MARGIN", 0.2))
        self.lambda_td = float(loss_cfg.get("LAMBDA_TD", 0.0))
        assert float(loss_cfg.get("TIME_UNIT", 0.0)) == 0.0, "LOSS.TIME_UNIT is not supported"

        self.loss_metric = LossMetric(cfg)
        self.ttg_mae_meter = AverageMeter("ttg_mae")
        self.rank_acc_meter = AverageMeter("rank_acc")
        self.lang_acc_meter = AverageMeter("lang_acc")
        self._val_task_idx: list[np.ndarray] = []
        self._val_prog_tp1: list[np.ndarray] = []
        self._val_p_tp1: list[np.ndarray] = []

        self.summary = None
        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        self.val_log_interval = cfg.TRAIN.get("VAL_LOG_INTERVAL", 100)

    def setup(self, summary_writer=None, **kwargs) -> None:
        self.summary = summary_writer

    def _ttg_mae(self, p_t, p_tp1, gt: dict):
        """Mean absolute error of p against the normalized-phase target (prog - 1)."""
        dev = p_t.device
        prog_t, prog_tp1 = gt["prog_t"].to(dev), gt["prog_tp1"].to(dev)
        return 0.5 * ((p_t - (prog_t - 1.0)).abs().mean()
                      + (p_tp1 - (prog_tp1 - 1.0)).abs().mean())

    def compute_loss(self, preds: dict, gt: dict) -> dict:
        dev = preds["p_t"].device
        loss_dict = compute_scaled_potential_loss(
            preds["p_t"], preds["p_tp1"], preds.get("p_tp1_wrong"),
            gt["prog_t"].to(dev), gt["prog_tp1"].to(dev),
            lambda_mc=self.lambda_mc, lambda_rank=self.lambda_rank, rank_tau=self.rank_tau,
            rank_min_delta=self.rank_min_delta,
            lambda_boundary=self.lambda_boundary, terminal_thresh=self.terminal_thresh,
            nonterminal_eps=self.nonterminal_eps, lambda_lang=self.lambda_lang, lang_margin=self.lang_margin,
            lambda_td=self.lambda_td,
        )
        p_t, p_tp1 = preds["p_t"], preds["p_tp1"]
        prog_t_d, prog_tp1_d = gt["prog_t"].to(dev), gt["prog_tp1"].to(dev)
        rank_keep = (prog_tp1_d - prog_t_d) > self.rank_min_delta
        loss_dict["_stat_rank_acc"] = ((p_tp1 > p_t)[rank_keep].float().mean().detach()
                                       if bool(rank_keep.any())
                                       else torch.zeros((), device=dev))
        loss_dict["_stat_ttg_mae"] = self._ttg_mae(p_t, p_tp1, gt).detach()
        loss_dict["_stat_p_mean"] = p_t.detach().mean()
        loss_dict["_stat_p_std"] = p_t.detach().std()
        if preds.get("p_tp1_wrong") is not None:
            loss_dict["_stat_lang_acc"] = (p_tp1 > preds["p_tp1_wrong"]).float().mean().detach()
        return loss_dict

    def training_step(self, batch: dict, step_idx: int):
        preds = self._compute_potentials(batch)
        loss_dict = self.compute_loss(preds, batch)
        metric_dict = {k: v for k, v in loss_dict.items() if not k.startswith("_stat_")}
        self.loss_metric.feed(metric_dict, batch["prog_t"].size(0))
        if step_idx % self.train_log_interval == 0 and self.summary is not None:
            for k, v in metric_dict.items():
                self.summary.add_scalar(f"train/{k}", v.item(), step_idx)
            for k in ("rank_acc", "rank_kept_frac", "ttg_mae", "p_mean", "p_std"):
                key = f"_stat_{k}"
                if key in loss_dict:
                    self.summary.add_scalar(f"train/{k}", loss_dict[key].item(), step_idx)
        return preds, loss_dict

    def on_train_finished(self, recorder: Recorder, epoch_idx: int) -> None:
        recorder.record_loss(self.loss_metric, epoch_idx, comment=f"{self.name}-train")
        self.loss_metric.reset()

    def validation_step(self, batch: dict, step_idx: int):
        preds = self._compute_potentials(batch)
        p_t, p_tp1 = preds["p_t"], preds["p_tp1"]
        dev = p_t.device
        prog_t = batch["prog_t"].to(dev)
        prog_tp1 = batch["prog_tp1"].to(dev)
        B = prog_t.size(0)

        self.rank_acc_meter.update_by_mean((p_tp1 > p_t).float().mean().item(), B)
        self.ttg_mae_meter.update_by_mean(self._ttg_mae(p_t, p_tp1, batch).item(), B)
        if preds.get("p_tp1_wrong") is not None:
            self.lang_acc_meter.update_by_mean((p_tp1 > preds["p_tp1_wrong"]).float().mean().item(), B)

        if "task_idx" in batch:
            self._val_task_idx.append(batch["task_idx"].cpu().numpy())
            self._val_prog_tp1.append(prog_tp1.detach().cpu().numpy())
            self._val_p_tp1.append(p_tp1.detach().cpu().numpy())

        if step_idx % self.val_log_interval == 0 and self.summary is not None:
            self.summary.add_scalar("val/rank_acc", self.rank_acc_meter.avg, step_idx)
        return preds

    def on_val_finished(self, recorder: Recorder, epoch_idx: int) -> None:
        macro_spearman = float("nan")
        n_tasks_scored = 0
        if self._val_task_idx:
            task_idx = np.concatenate(self._val_task_idx)
            prog_tp1 = np.concatenate(self._val_prog_tp1)
            p_tp1 = np.concatenate(self._val_p_tp1)
            per_task = []
            for t in np.unique(task_idx):
                mask = task_idx == t
                if mask.sum() < 2:
                    continue
                sp = _spearman(p_tp1[mask], prog_tp1[mask])
                if not np.isnan(sp):
                    per_task.append(sp)
            if per_task:
                macro_spearman = float(np.mean(per_task))
                n_tasks_scored = len(per_task)
        self._val_task_idx.clear()
        self._val_prog_tp1.clear()
        self._val_p_tp1.clear()

        meters = [self.rank_acc_meter, self.ttg_mae_meter]
        if self.lambda_lang > 0:
            meters.append(self.lang_acc_meter)
        act_cos_meter = getattr(self, "act_cos_meter", None)
        has_act_cos = act_cos_meter is not None and act_cos_meter.count > 0
        if has_act_cos:
            meters.append(act_cos_meter)
        g_cos_meter = getattr(self, "act_g_cos_meter", None)
        if g_cos_meter is not None and g_cos_meter.count > 0:
            meters.append(g_cos_meter)
        act_mse_meter = getattr(self, "act_mse_meter", None)
        if act_mse_meter is not None and act_mse_meter.count > 0:
            meters.append(act_mse_meter)
        recorder.record_metric(meters, epoch_idx, comment=f"{self.name}-val")
        if self.summary is not None:
            self.summary.add_scalar("val/epoch_rank_acc", self.rank_acc_meter.avg, epoch_idx)
            self.summary.add_scalar("val/epoch_ttg_mae", self.ttg_mae_meter.avg, epoch_idx)
            self.summary.add_scalar("val/epoch_lang_acc", self.lang_acc_meter.avg, epoch_idx)
            if has_act_cos:
                self.summary.add_scalar("val/epoch_act_cos", act_cos_meter.avg, epoch_idx)
            if g_cos_meter is not None and g_cos_meter.count > 0:
                self.summary.add_scalar("val/epoch_g_cos", g_cos_meter.avg, epoch_idx)
            if not np.isnan(macro_spearman):
                self.summary.add_scalar("val/epoch_macro_spearman", macro_spearman, epoch_idx)
        act_cos_str = f" act_cos={act_cos_meter.avg:.4f}" if has_act_cos else ""
        if g_cos_meter is not None and g_cos_meter.count > 0:
            act_cos_str += f" g_cos={g_cos_meter.avg:.4f}"
        logger.info(f"{self.name} val epoch {epoch_idx}: macro_spearman={macro_spearman:.4f} "
                    f"(n_tasks={n_tasks_scored}) | lang_acc={self.lang_acc_meter.avg:.3f} "
                    f"rank_acc={self.rank_acc_meter.avg:.3f} ttg_mae={self.ttg_mae_meter.avg:.4f}"
                    f"{act_cos_str}")
        self._last_macro_spearman = macro_spearman
        lang_gate_ok = self.lambda_lang <= 0 or self.lang_acc_meter.avg > 0.5
        self._last_primary_score = macro_spearman if (not np.isnan(macro_spearman) and lang_gate_ok) \
            else float("-inf")
        for m in meters:
            m.reset()

    def primary_val_score(self) -> float:
        return getattr(self, "_last_primary_score", float("-inf"))

    def format_metric(self, mode: str) -> str:
        if mode == "train":
            try:
                return (f"loss={self.loss_metric.get_loss('loss'):.4f} | "
                        f"mc={self.loss_metric.get_loss('l_mc'):.4f} | "
                        f"rank={self.loss_metric.get_loss('l_rank'):.4f} | "
                        f"bnd={self.loss_metric.get_loss('l_boundary'):.4f} | "
                        f"lang={self.loss_metric.get_loss('l_lang'):.4f}"
                        + (f" | td={self.loss_metric.get_loss('l_td'):.4f}" if self.lambda_td > 0 else ""))
            except (KeyError, ZeroDivisionError):
                return ""
        if mode == "val":
            sp = getattr(self, "_last_macro_spearman", float("nan"))
            return f"macro_spearman={sp:.3f} | lang_acc={self.lang_acc_meter.avg:.3f} | rank={self.rank_acc_meter.avg:.3f}"
        return ""

    def forward(self, inputs, step_idx, mode: str = "train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        if mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        raise ValueError(f"Unknown mode: {mode}")
