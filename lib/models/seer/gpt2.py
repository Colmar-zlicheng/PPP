# Adapted from Seer (https://github.com/InternRobotics/Seer), Apache License 2.0.
# Modified for PPP. See THIRD_PARTY_NOTICES.md.
# Seer's version is in turn adapted from the GPT-2 implementation of Hugging Face Transformers
# (https://github.com/huggingface/transformers), Apache License 2.0.
"""Seer's causal transformer (`models/gpt2.py` in Seer): a GPT-2 stack with no token or position
embedding of its own, driven by `inputs_embeds` and a dense additive (L, L) attention mask."""
import math

from torch import nn
from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D


class GPT2Attention(nn.Module):
    def __init__(self, hidden, heads, attn_pdrop, resid_pdrop):
        super().__init__()
        assert hidden % heads == 0
        self.embed_dim, self.num_heads, self.head_dim = hidden, heads, hidden // heads
        self.c_attn = Conv1D(3 * hidden, hidden)
        self.c_proj = Conv1D(hidden, hidden)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _split(self, t):
        return t.view(t.shape[:-1] + (self.num_heads, self.head_dim)).permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None):
        q, k, v = (self._split(t) for t in self.c_attn(x).split(self.embed_dim, dim=2))
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.shape[:-2] + (self.embed_dim,))
        return self.resid_dropout(self.c_proj(out))


class GPT2MLP(nn.Module):
    def __init__(self, hidden, inner, resid_pdrop):
        super().__init__()
        self.c_fc = Conv1D(inner, hidden)
        self.c_proj = Conv1D(hidden, inner)
        self.act = ACT2FN["gelu_new"]
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPT2Block(nn.Module):
    def __init__(self, hidden, heads, pdrop, eps):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden, eps=eps)
        self.attn = GPT2Attention(hidden, heads, pdrop, pdrop)
        self.ln_2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = GPT2MLP(hidden, 4 * hidden, pdrop)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        return x + self.mlp(self.ln_2(x))


class GPT2Model(nn.Module):
    def __init__(self, hidden=384, n_layer=24, n_head=12, pdrop=0.1, eps=1e-5,
                 initializer_range=0.02):
        super().__init__()
        self.drop = nn.Dropout(pdrop)
        self.h = nn.ModuleList([GPT2Block(hidden, n_head, pdrop, eps) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(hidden, eps=eps)
        for m in self.modules():
            if isinstance(m, Conv1D):
                m.weight.data.normal_(0.0, initializer_range)
                m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                p.data.normal_(0.0, initializer_range / math.sqrt(2 * n_layer))

    def forward(self, inputs_embeds, attention_mask=None):
        x = self.drop(inputs_embeds)
        for block in self.h:
            x = block(x, attention_mask=attention_mask)
        return self.ln_f(x)
