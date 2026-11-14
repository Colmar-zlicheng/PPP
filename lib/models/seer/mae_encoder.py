# Adapted from Seer (https://github.com/InternRobotics/Seer), Apache License 2.0.
# Modified for PPP. See THIRD_PARTY_NOTICES.md.
# Seer's version follows the encoder of MAE (https://github.com/facebookresearch/mae),
# CC BY-NC 4.0 License.
"""The frozen MAE ViT-B/16 image encoder of Seer (`models/vit_mae.py` in Seer), encoder half only."""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    emb = np.concatenate([get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]),
                          get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


class MAEEncoder(nn.Module):
    """ViT-B/16 MAE encoder: (N,3,224,224) normalized images -> (N, 1+196, 768), CLS first."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim), requires_grad=False)
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                                           norm_layer=norm) for _ in range(depth)])
        self.norm = norm(embed_dim)
        self.pos_embed.data.copy_(torch.from_numpy(
            get_2d_sincos_pos_embed(embed_dim, int(n ** 0.5), cls_token=True)).float().unsqueeze(0))
        self.embed_dim, self.num_patches = embed_dim, n

    def load_mae(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        enc = {k: v for k, v in sd.items()
               if not (k.startswith("decoder") or k == "mask_token")}
        missing, unexpected = self.load_state_dict(enc, strict=False)
        assert not missing and not unexpected, (
            f"MAE checkpoint {path} does not match the ViT-B encoder: missing={missing}, "
            f"unexpected={unexpected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
