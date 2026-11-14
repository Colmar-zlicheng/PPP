# Adapted from Seer (https://github.com/InternRobotics/Seer), Apache License 2.0.
# Modified for PPP. See THIRD_PARTY_NOTICES.md.
"""Seer's perceiver resampler (`models/perceiver_resampler.py` in Seer), ported without the
`einops_exts` dependency. Parameter names are unchanged so Seer state dicts load as-is."""
import torch
from einops import rearrange, repeat
from torch import einsum, nn


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, inner_dim, bias=False), nn.GELU(),
                         nn.Linear(inner_dim, dim, bias=False))


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """x (b, T, n1, D) media, latents (b, T, n2, D)."""
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        h = self.heads
        q = self.to_q(latents)
        k, v = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)
        q, k, v = (rearrange(t, "b t n (h d) -> b h t n d", h=h) for t in (q, k, v))
        q = q * self.scale
        sim = einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        return self.to_out(rearrange(out, "b h t n d -> b t n (h d)", h=h))


class PerceiverResampler(nn.Module):
    def __init__(self, *, dim, depth=6, dim_head=64, heads=8, num_latents=64, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([
            nn.ModuleList([PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                           FeedForward(dim=dim, mult=ff_mult)]) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x (b, T, F, v, D) -> (b, T, num_latents, D). Seer calls it with T = F = 1."""
        b, T = x.shape[:2]
        x = rearrange(x, "b T F v d -> b T (F v) d")
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)
