"""Quality metrics for a world model's ACTION JACOBIAN, ``J = d f(s, a) / d a``."""

from contextlib import nullcontext
from typing import Callable, Dict, Sequence

import torch

try:  # torch >= 2.1
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover -- older torch has no backend selector
    SDPBackend = sdpa_kernel = None


def _forward_ad_safe():
    """Context forcing attention onto a kernel that implements forward-mode AD."""
    if sdpa_kernel is None:
        return nullcontext()
    return sdpa_kernel(SDPBackend.MATH)


def _flat_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm of everything but dim 0."""
    return x.reshape(x.size(0), -1).norm(dim=-1)


def action_jacobian_stats(fn: Callable[[torch.Tensor], torch.Tensor],
                          a: torch.Tensor,
                          ref_delta: torch.Tensor,
                          num_probes: int = 8,
                          eps: Sequence[float] = (0.25, 1.0),
                          generator: torch.Generator = None) -> Dict[str, float]:
    """Probe the action Jacobian of `fn` at `a` without ever forming it."""
    stats: Dict[str, float] = {}
    B = a.size(0)
    ref = _flat_norm(ref_delta.detach()).mean().clamp_min(1e-12)

    probes, jvps = [], []
    with torch.no_grad():
        base_pred = fn(a)
    for _ in range(num_probes):
        d = torch.randn(a.shape, device=a.device, dtype=a.dtype, generator=generator)
        d = d / _flat_norm(d).clamp_min(1e-12).reshape(-1, *([1] * (a.dim() - 1)))
        with _forward_ad_safe():  # a fresh context each time; sdpa_kernel is single-use
            _, jvp = torch.func.jvp(fn, (a,), (d,))
        probes.append(d)
        jvps.append(jvp.detach())

    J = torch.stack(jvps, dim=1).reshape(B, num_probes, -1)  # (B, K, out)
    gain = J.norm(dim=-1)                                    # (B, K)
    stats["jac_gain"] = gain.mean().item()
    stats["jac_gain_rel"] = (gain.mean() / ref).item()

    G = J @ J.transpose(1, 2)                                # (B, K, K)
    trace = torch.diagonal(G, dim1=1, dim2=2).sum(-1)
    fro_sq = G.pow(2).sum(dim=(1, 2))
    rank = trace.pow(2) / fro_sq.clamp_min(1e-12) / num_probes
    rank = torch.where(fro_sq > 0, rank, torch.full_like(rank, float("nan")))
    stats["jac_eff_rank_frac"] = rank.mean().item()

    with torch.no_grad():
        for e in eps:
            num, den = 0.0, 0.0
            for d, jvp in zip(probes, jvps):
                df = fn(a + e * d) - base_pred
                num += _flat_norm(df - e * jvp).mean().item()
                den += _flat_norm(df).mean().item()
            stats[f"jac_lin_err_eps{e:g}"] = num / den if den > 0 else float("nan")

        if B > 1:
            perm = torch.randperm(B, device=a.device, generator=generator)
            d_swap = a[perm] - a
            with _forward_ad_safe():
                _, jvp_swap = torch.func.jvp(fn, (a,), (d_swap,))
            df_swap = fn(a[perm]) - base_pred
            swap_den = _flat_norm(df_swap).mean()
            stats["jac_lin_err_swap"] = (
                (_flat_norm(df_swap - jvp_swap).mean() / swap_den).item()
                if float(swap_den) > 0 else float("nan"))
            stats["jac_swap_gain_rel"] = (_flat_norm(df_swap).mean() / ref).item()

    return stats
