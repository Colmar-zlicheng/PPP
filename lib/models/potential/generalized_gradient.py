"""Action readout `a = M_phi(z) g`: a learned positive scalar gain on the field's action gradient."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


class GeneralizedGradientGeometry(nn.Module):
    """Task-independent `z -> M_phi(z) = m_phi(z) I` (a positive scalar gain) plus its application
    to the gradient `g`. `W` (the skew part) is identically zero and returned for the regularizer."""

    def __init__(self, in_dim: int, action_dim: int = 6, hidden_dim: int = 256,
                 num_layers: int = 2, eps: float = 1e-3):
        super().__init__()
        self.mode = "scalar"
        self.action_dim = int(action_dim)
        self.eps = float(eps)

        self.register_buffer("_eye", torch.eye(self.action_dim), persistent=False)
        self._m_bias = math.log(math.e - 1.0)

        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        d = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head_m = _zero_init(nn.Linear(hidden_dim, 1))

    def operators(self, feat: torch.Tensor, batch_size: int, dtype, device):
        """-> (M = m I, W = 0), each `(B,da,da)`."""
        da = self.action_dim
        eye = self._eye.to(device=device, dtype=dtype)
        h = self.trunk(feat)
        m = F.softplus(self.head_m(h).view(batch_size) + self._m_bias) + self.eps  # (B,)
        M = m[..., None, None] * eye
        W = torch.zeros(batch_size, da, da, dtype=dtype, device=device)
        return M, W

    def forward(self, feat: torch.Tensor, g: torch.Tensor):
        assert g.dim() == 2, f"expects g of shape (B,da), got {tuple(g.shape)}"
        M, W = self.operators(feat, g.shape[0], g.dtype, g.device)
        a_hat = torch.einsum("...ij,...j->...i", M + W, g)
        return a_hat, M, W


def _zero_init(linear: nn.Linear) -> nn.Linear:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    return linear


def operator_regularizer(M: torch.Tensor, W: torch.Tensor, lambda_m: float, lambda_w: float):
    eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).unsqueeze(0)
    l_m = (M - eye).pow(2).sum(dim=(-1, -2)).mean()
    l_w = W.pow(2).sum(dim=(-1, -2)).mean()
    return lambda_m * l_m + lambda_w * l_w, l_m.detach(), l_w.detach()


@torch.no_grad()
def progress_alignment_diagnostics(g: torch.Tensor, a_expert: torch.Tensor,
                                    g_zero_thresh: float = 1e-4) -> dict:
    g_norm = g.norm(dim=-1)
    a_norm = a_expert.norm(dim=-1)
    c = F.cosine_similarity(g, a_expert, dim=-1, eps=_EPS)
    live = a_norm > _EPS
    crit = (g_norm < g_zero_thresh) & live
    return {
        "g_cos": c.mean(),
        "g_cos_pos": (c > 0).float().mean(),
        "g_norm": g_norm.mean(),
        "g_crit": crit.float().mean(),
    }
