# Adapted from Seer (https://github.com/InternRobotics/Seer), Apache License 2.0.
# Modified for PPP. See THIRD_PARTY_NOTICES.md.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from .gpt2 import GPT2Model
from .mae_encoder import MAEEncoder, get_2d_sincos_pos_embed
from .perceiver import PerceiverResampler

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

KIND_A, KIND_ACT, KIND_OBS, KIND_QUERY = 0, 1, 2, 3


def random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Seer's `RandomShiftsAug.forward_traj`, per image: replicate-pad by `pad`, then crop at a
    random integer offset in [1, 2 pad] per axis (grid_sample). x (N,C,H,W) square."""
    n, c, h, w = x.shape
    assert h == w
    x = F.pad(x, (pad,) * 4, "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(1, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift *= 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base_grid + shift, padding_mode="zeros", align_corners=False)


class SeerNet(nn.Module):
    def __init__(self, mae_ckpt: Optional[str] = None, seq_len: int = 7, num_resampler_query: int = 6,
                 num_obs_token_per_image: int = 9, action_pred_steps: int = 3,
                 transformer_layers: int = 24, hidden_dim: int = 384, transformer_heads: int = 12,
                 image_size: int = 224, patch_size: int = 16, text_dim: int = 512,
                 rgb_pad: int = 10, gripper_pad: int = 4, bc_head: bool = True,
                 gpt_dropout: float = 0.1, query_tokens: bool = False):
        super().__init__()
        H = hidden_dim
        self.seq_len, self.hidden_dim = seq_len, H
        self.action_pred_steps = action_pred_steps
        self.bc_head = bool(bc_head)
        self.has_queries = self.bc_head or bool(query_tokens)
        self.rgb_pad, self.gripper_pad = rgb_pad, gripper_pad
        self.image_size, self.patch_size = image_size, patch_size
        self.n_obs_img = num_obs_token_per_image
        self.n_obs = 2 * num_obs_token_per_image
        self.n_query = num_resampler_query
        self.num_a = 1 + 1 + 2 * num_resampler_query + 2

        self.text_projector = nn.Linear(text_dim, H)
        self.arm_state_encoder = nn.Linear(6, H)
        self.gripper_state_encoder = nn.Linear(2, H)
        self.state_projector = nn.Linear(2 * H, H)

        self.vision_encoder = MAEEncoder(img_size=image_size, patch_size=patch_size)
        D = self.vision_encoder.embed_dim
        self.vis_dim = D
        self.perceiver_resampler = PerceiverResampler(dim=D, num_latents=num_resampler_query, depth=3)
        self.image_primary_projector = nn.Linear(D, H)
        self.cls_token_primary_projector = nn.Linear(D, H)
        self.image_wrist_projector = nn.Linear(D, H)
        self.cls_token_wrist_projector = nn.Linear(D, H)

        if self.has_queries:
            self.action_pred_token = nn.Parameter(torch.zeros(1, 1, action_pred_steps, H))
        self.obs_tokens = nn.Parameter(torch.zeros(1, 1, self.n_obs, H))
        self.embedding_layer_norm = nn.LayerNorm(H)
        self.transformer_backbone_position_embedding = nn.Parameter(torch.zeros(1, seq_len, 1, H))
        self.transformer_backbone = GPT2Model(hidden=H, n_layer=transformer_layers,
                                              n_head=transformer_heads, pdrop=gpt_dropout)

        if self.bc_head:
            mh = H // 2
            self.action_decoder = nn.Sequential(nn.Linear(H, mh), nn.ReLU(), nn.Linear(mh, mh), nn.ReLU())
            self.arm_action_decoder = nn.Sequential(nn.Linear(mh, 6), nn.Tanh())
            self.gripper_action_decoder = nn.Sequential(nn.Linear(mh, 1), nn.Sigmoid())

        n_patch = (image_size // patch_size) ** 2
        self.n_patch = n_patch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, H))
        self.image_decoder_obs_pred_projector = nn.Linear(H, H)
        self.image_decoder_position_embedding = nn.Parameter(
            torch.zeros(1, self.n_obs_img + n_patch, H), requires_grad=False)
        self.image_decoder = nn.Sequential(*[
            Block(H, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)])
        self.image_decoder_norm = nn.LayerNorm(H)
        self.image_decoder_pred = nn.Linear(H, patch_size ** 2 * 3)

        self._initialize_weights()
        if mae_ckpt:
            self.vision_encoder.load_mae(mae_ckpt)
        self.vision_encoder.requires_grad_(False)
        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)
        self._mask_cache = {}

    def _initialize_weights(self):
        n_obs_grid = int(self.n_obs_img ** 0.5)
        assert n_obs_grid ** 2 == self.n_obs_img, "NUM_OBS_TOKEN_PER_IMAGE must be a square"
        pe = torch.cat([
            torch.from_numpy(get_2d_sincos_pos_embed(self.hidden_dim, n_obs_grid)),
            torch.from_numpy(get_2d_sincos_pos_embed(self.hidden_dim, int(self.n_patch ** 0.5)))]).float()
        self.image_decoder_position_embedding.data.copy_(pe.unsqueeze(0))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.transformer_backbone_position_embedding, std=0.02)

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init)


    def preprocess(self, frames: torch.Tensor, augment: bool) -> torch.Tensor:
        """(..., 2, h, w, 3) uint8 [front, wrist] -> (..., 2, 3, 224, 224) CLIP-normalized float."""
        lead = frames.shape[:-4]
        x = frames.reshape(-1, *frames.shape[-4:]).to(self.clip_mean.dtype) / 255.0
        n = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).reshape(n * 2, 3, frames.shape[-3], frames.shape[-2])
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic",
                          align_corners=False).clamp_(0.0, 1.0)
        x = (x - self.clip_mean) / self.clip_std
        x = x.view(n, 2, 3, self.image_size, self.image_size)
        if augment:
            x = torch.stack([random_shift(x[:, 0], self.rgb_pad),
                             random_shift(x[:, 1], self.gripper_pad)], dim=1)
        return x.reshape(*lead, 2, 3, self.image_size, self.image_size)

    @torch.no_grad()
    def encode_images(self, x: torch.Tensor, chunk: int = 512, bf16: bool = True) -> torch.Tensor:
        """(..., 3, 224, 224) -> (..., 197, 768) frozen MAE features, returned fp32. bf16 autocast by
        default (Seer's `--bf16_module vision_encoder`)."""
        lead = x.shape[:-3]
        flat = x.reshape(-1, *x.shape[-3:])
        outs = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=flat.is_cuda and bf16):
            for i in range(0, flat.shape[0], chunk):
                outs.append(self.vision_encoder(flat[i:i + chunk]).float())
        out = torch.cat(outs)
        return out.view(*lead, *out.shape[-2:])

    def frame_tokens(self, feat: torch.Tensor):
        """(B,T,2,197,D) -> image tokens (B,T,2Q,H) and CLS tokens (B,T,2,H), Seer's projections."""
        B, T = feat.shape[:2]
        D, Q = feat.shape[-1], self.n_query
        cls, patches = feat[..., 0, :], feat[..., 1:, :]

        def resample(p):
            r = self.perceiver_resampler(p.reshape(B * T, -1, D).unsqueeze(1).unsqueeze(1))
            return r.flatten(0, 2)                                         # (B*T*Q, D)
        img = torch.cat([
            self.image_primary_projector(resample(patches[:, :, 0])).view(B, T, Q, -1),
            self.image_wrist_projector(resample(patches[:, :, 1])).view(B, T, Q, -1)], dim=2)
        cls_tok = torch.cat([
            self.cls_token_primary_projector(cls[:, :, 0]).view(B, T, 1, -1),
            self.cls_token_wrist_projector(cls[:, :, 1]).view(B, T, 1, -1)], dim=2)
        return img, cls_tok


    def base_tokens(self, text_feat, state, img_tok, cls_tok) -> torch.Tensor:
        """-> (B,S,16,H) A tokens. text_feat (B,512); state (B,S,8) = [ee_pos, ee_axisangle, finger qpos x2]
        (Seer's LIBERO `--gripper_width` state)."""
        B, S = state.shape[:2]
        text = self.text_projector(text_feat).view(B, 1, 1, -1).expand(B, S, 1, -1)
        st = self.state_projector(torch.cat([self.arm_state_encoder(state[..., :6]),
                                             self.gripper_state_encoder(state[..., 6:8])], dim=-1))
        return torch.cat([text, st.unsqueeze(2), img_tok, cls_tok], dim=2)

    def _mask(self, kinds, steps, groups, device):
        key = (tuple(kinds), tuple(steps), tuple(groups), str(device))
        m = self._mask_cache.get(key)
        if m is None:
            k = torch.tensor(kinds, device=device)
            s = torch.tensor(steps, device=device)
            g = torch.tensor(groups, device=device)
            kr, kc = k[:, None], k[None, :]
            sr, sc = s[:, None], s[None, :]
            gr, gc = g[:, None], g[None, :]
            allowed = (kc == KIND_A) & (sc <= sr)
            allowed |= ((kc == KIND_ACT) & ((kr == KIND_ACT) | (kr == KIND_OBS))
                        & (gc == gr) & (sc == sr))
            allowed |= (kc == KIND_OBS) & (kr == KIND_QUERY) & (gc == gr) & (sc == sr)   # a query set reads its group
            m = torch.zeros(allowed.shape, device=device)
            m.masked_fill_(~allowed, float("-inf"))
            self._mask_cache[key] = m
        return m

    def run(self, base: torch.Tensor, groups: List[dict], queries: bool = False,
            read_group: int = 0) -> dict:
        """One transformer pass."""
        B, S, nA, H = base.shape
        pos = self.transformer_backbone_position_embedding[:, :S]          # (1,S,1,H)
        parts = [(base + pos).reshape(B, S * nA, H)]
        kinds = [KIND_A] * (S * nA)
        steps = [i for i in range(S) for _ in range(nA)]
        gids = [-1] * (S * nA)
        seg = []
        for gi, grp in enumerate(groups):
            T = 1 if grp.get("last", False) else S
            st = list(range(S - T, S))
            gpos = pos[:, S - T:]
            n_act = 0
            assert grp.get("u6") is None, "the Seer network takes no action inputs"
            obs = (self.obs_tokens + gpos).expand(B, T, self.n_obs, H)
            parts.append(obs.reshape(B, T * self.n_obs, H))
            kinds += [KIND_OBS] * (T * self.n_obs)
            steps += [i for i in st for _ in range(self.n_obs)]
            gids += [gi] * (T * self.n_obs)
            seg.append((T, n_act))
        reads = [(read_group, S)] if queries else []             # query sets: (group read, its steps)
        reads += [(gi, T) for gi, (grp, (T, _)) in enumerate(zip(groups, seg)) if grp.get("queries", False)]
        assert len({gi for gi, _ in reads}) == len(reads), "a group is read by one query set"
        n_q = self.action_pred_steps
        for gi, T in reads:
            assert self.has_queries, "this SeerNet has no action queries"
            q = (self.action_pred_token + pos[:, S - T:]).expand(B, T, n_q, H)
            parts.append(q.reshape(B, T * n_q, H))
            kinds += [KIND_QUERY] * (T * n_q)
            steps += [i for i in range(S - T, S) for _ in range(n_q)]
            gids += [gi] * (T * n_q)

        x = self.embedding_layer_norm(torch.cat(parts, dim=1))
        mask = self._mask(kinds, steps, gids, x.device).to(x.dtype)
        y = self.transformer_backbone(x, attention_mask=mask)

        off = S * nA
        obs_out = []
        for T, n_act in seg:
            off += T * n_act
            obs_out.append(y[:, off:off + T * self.n_obs].reshape(B, T, self.n_obs, H))
            off += T * self.n_obs
        q_sets = [None] * len(groups)
        for gi, T in reads:
            q_sets[gi] = y[:, off:off + T * n_q].reshape(B, T, n_q, H)
            off += T * n_q
        return {"obs": obs_out, "query": q_sets[read_group] if queries else None, "queries": q_sets}


    def image_decode(self, obs: torch.Tensor) -> torch.Tensor:
        """(B,T,18,H) foresight outputs -> (B*T, 2, n_patch, p*p*3) predicted future patches (Seer)."""
        B, T = obs.shape[:2]
        e = self.image_decoder_obs_pred_projector(obs.reshape(-1, self.hidden_dim))
        e = e.view(B * T * 2, self.n_obs_img, self.hidden_dim)
        x = torch.cat([e, self.mask_token.expand(B * T * 2, self.n_patch, -1)], dim=1)
        x = self.image_decoder(x + self.image_decoder_position_embedding)
        x = self.image_decoder_norm(x[:, -self.n_patch:].reshape(-1, self.hidden_dim))
        return self.image_decoder_pred(x).view(B * T, 2, self.n_patch, -1)

    def forward(self, base: torch.Tensor, groups: List[dict]):
        """One pass whose first group is read by the action queries -> (B,T,K*H)."""
        return self.run(base, [dict(groups[0], queries=True)] + list(groups[1:]))["queries"][0].flatten(-2)

    def bc_decode(self, q: torch.Tensor):
        h = self.action_decoder(q)
        return self.arm_action_decoder(h), self.gripper_action_decoder[0](h).squeeze(-1)


def patchify_normalized(imgs: torch.Tensor, p: int = 16) -> torch.Tensor:
    """(N,3,H,W) -> (N, L, p*p*3), each patch standardized (Seer's `normalize_patchfied_image`)."""
    N, _, H, W = imgs.shape
    h, w = H // p, W // p
    x = imgs.reshape(N, 3, h, p, w, p)
    x = torch.einsum("nchpwq->nhwpqc", x).reshape(N, h * w, p * p * 3)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return (x - mean) / (var + 1e-6) ** 0.5
