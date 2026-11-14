# Adapted from LeWorldModel (https://github.com/lucas-maes/le-wm), MIT License,
# Copyright (c) 2026 Lucas Maes. Modified for PPP. See THIRD_PARTY_NOTICES.md.
"""JEPA Implementation"""

from einops import rearrange
from torch import nn
from .module import ARPredictor, Embedder, MLP
try:
    from transformers import ViTModel
except ImportError as exc:
    raise ImportError("HuggingFace `transformers` is required. Install with: pip install transformers") from exc


class JEPA(nn.Module):
    def __init__(
        self,
        encoder: ViTModel,
        predictor: ARPredictor,
        action_encoder: Embedder,
        projector: MLP = None,
        pred_proj: MLP = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings."""

        pixels = info['pixels'].float()
        b = pixels.size(0)
        n_views = pixels.size(2) if pixels.dim() == 6 else 1
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        if n_views > 1:
            pixels = rearrange(pixels, "bt v c h w -> (bt v) c h w")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        h = output.last_hidden_state                        # (bt*v, 1+P, D), CLS first

        pixels_emb = h[:, 0]                                # cls token
        if n_views > 1:
            pixels_emb = rearrange(pixels_emb, "(bt v) d -> bt (v d)", v=n_views)
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding"""
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds
