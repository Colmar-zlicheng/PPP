import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    """Pre-LN transformer block, matmul-softmax attention (twice differentiable), GELU MLP, no dropout."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: int):
        super().__init__()
        assert hidden % heads == 0
        self.heads, self.hd = heads, hidden // heads
        self.ln1, self.ln2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.fc1 = nn.Linear(hidden, mlp_ratio * hidden)
        self.fc2 = nn.Linear(mlp_ratio * hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, L, H = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n, L, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd), dim=-1)
        x = x + self.proj((w @ v).transpose(1, 2).reshape(n, L, H))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class StateWorldModel(nn.Module):
    def __init__(self, hidden: int = 384, n_state: int = 3, action_steps: int = 3, layers: int = 4,
                 heads: int = 12, mlp_ratio: int = 4, out_scale: float = 0.1):
        super().__init__()
        H = hidden
        self.hidden, self.n_state, self.action_steps = H, n_state, action_steps
        self.pose_enc = nn.Linear(6, H)
        self.grip_enc = nn.Linear(2, H)
        self.act_proj = nn.Linear(2 * H, H)
        self.tick = nn.Parameter(torch.zeros(1, action_steps, H))
        self.slot = nn.Parameter(torch.zeros(1, n_state, H))
        self.blocks = nn.ModuleList([_Block(H, heads, mlp_ratio) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(H)
        self.out = nn.Linear(H, H)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.tick, std=0.02)
        nn.init.normal_(self.slot, std=0.02)
        with torch.no_grad():
            self.out.weight.mul_(out_scale)

    def forward(self, s: torch.Tensor, u6: torch.Tensor, grip: torch.Tensor) -> torch.Tensor:
        """s (..., n_state, H) trunk state; u6 (..., K, 6) pose commands (model units); grip (..., K) gripper
        command sign (+1 close) -> the predicted state (..., n_state, H) of the frame the chunk ends at."""
        lead = s.shape[:-2]
        H, n = self.hidden, self.n_state
        onehot = torch.stack([(1.0 - grip) * 0.5, (1.0 + grip) * 0.5], dim=-1).to(u6.dtype)
        a = self.act_proj(torch.cat([self.pose_enc(u6), self.grip_enc(onehot)], dim=-1)) + self.tick
        a = a.expand(*lead, *a.shape[-2:])
        x = torch.cat([s + self.slot, a], dim=-2).reshape(-1, n + a.shape[-2], H)
        for blk in self.blocks:
            x = blk(x)
        d = self.out(self.ln_out(x[:, :n]))
        return s + d.view(*lead, n, H)
