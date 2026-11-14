from typing import Dict

import torch

from lib.metrics.basic_metric import AverageMeter, Metric


@torch.no_grad()
def embedding_collapse_stats(emb: torch.Tensor,
                             pred_emb: torch.Tensor,
                             tgt_emb: torch.Tensor,
                             copy_emb: torch.Tensor) -> Dict[str, float]:
    emb = emb.detach().float()
    pred_emb, tgt_emb, copy_emb = pred_emb.detach().float(), tgt_emb.detach().float(), copy_emb.detach().float()
    B, T, D = emb.shape

    stats = {}
    stats["emb_norm"] = emb.norm(dim=-1).mean().item()

    batch_std = emb.std(dim=0, unbiased=False).mean() if B > 1 else emb.new_zeros(())
    time_std = emb.std(dim=1, unbiased=False).mean() if T > 1 else emb.new_zeros(())
    stats["emb_batch_std"] = batch_std.item()
    stats["emb_time_std"] = time_std.item()
    stats["time_batch_std_ratio"] = (time_std / batch_std.clamp_min(1e-12)).item()

    x = emb.reshape(B * T, D)
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(1, x.size(0) - 1)
    trace = torch.diagonal(cov).sum()
    fro_sq = cov.pow(2).sum()
    stats["emb_eff_rank_frac"] = (trace.pow(2) / fro_sq.clamp_min(1e-12) / D).item()

    pred_mse = (pred_emb - tgt_emb).pow(2).mean()
    copy_mse = (copy_emb - tgt_emb).pow(2).mean()
    stats["copy_mse"] = copy_mse.item()
    stats["pred_vs_copy_ratio"] = (pred_mse / copy_mse.clamp_min(1e-12)).item()

    true_delta = tgt_emb - copy_emb
    delta_var = true_delta.var(dim=0, unbiased=False).mean() if B > 1 else true_delta.new_zeros(())
    stats["delta_r2"] = (1.0 - pred_mse / delta_var.clamp_min(1e-12)).item()

    return stats


@torch.no_grad()
def action_sensitivity(predict_fn, ctx_emb: torch.Tensor, ctx_act: torch.Tensor,
                       tgt_emb: torch.Tensor, copy_emb: torch.Tensor,
                       generator: torch.Generator = None) -> float:
    """How much the prediction depends on the action, relative to the real one-step change."""
    B = ctx_emb.size(0)
    if B < 2:
        return float("nan")
    perm = torch.randperm(B, device=ctx_emb.device, generator=generator)
    if bool((perm == torch.arange(B, device=perm.device)).all()):
        perm = torch.roll(perm, 1)

    pred = predict_fn(ctx_emb, ctx_act)
    pred_shuf = predict_fn(ctx_emb, ctx_act[perm])
    moved = (pred_shuf - pred).pow(2).mean()
    real_step = (tgt_emb - copy_emb).pow(2).mean()
    return (moved / real_step.clamp_min(1e-12)).item()


class CollapseMetric(Metric):
    """Batch-size-weighted running average of `embedding_collapse_stats`' output."""

    def __init__(self, cfg=None, name="Collapse") -> None:
        super().__init__()
        self.name = name
        self._meters: Dict[str, AverageMeter] = {}
        self.reset()

    def reset(self):
        self._meters = {}
        self.count = 0

    def feed(self, stats: Dict[str, float], batch_size: int = 1, **kwargs):
        for k, v in stats.items():
            v = float(v)
            if v != v:  # NaN
                continue
            if k not in self._meters:
                self._meters[k] = AverageMeter(name=k)
            self._meters[k].update_by_mean(v, batch_size)
        self.count += batch_size
        return stats

    def get_measures(self, **kwargs) -> Dict[str, float]:
        return {k: m.avg for k, m in self._meters.items()}

    def get(self, key: str, default: float = float("nan")) -> float:
        return self._meters[key].avg if key in self._meters else default

    def __str__(self):
        parts = " ".join(f"{k}: {m.avg:.4f}" for k, m in self._meters.items())
        return f"{self.name}: {parts}"
