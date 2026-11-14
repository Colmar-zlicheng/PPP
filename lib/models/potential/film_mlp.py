import torch
import torch.nn as nn


class ResFiLMBlock(nn.Module):
    """h -> LN -> Linear -> FiLM(gamma,beta) -> SiLU -> Linear -> (+h)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        x = self.fc1(x)
        x = x * (1.0 + gamma) + beta
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return h + x


class FiLMGenerator(nn.Module):
    """lang_embed (L,) -> per-block (gamma, beta), each (hidden_dim,)."""

    def __init__(self, lang_dim: int, hidden_dim: int, num_layers: int, film_hidden: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(lang_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, 2 * hidden_dim * num_layers),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lang_embed: torch.Tensor):
        out = self.net(lang_embed)  # (B, 2*hidden_dim*num_layers)
        out = out.view(-1, self.num_layers, 2, self.hidden_dim)
        gammas = out[:, :, 0, :]
        betas = out[:, :, 1, :]
        return gammas, betas  # each (B, num_layers, hidden_dim)


class FiLMMLPHead(nn.Module):
    """p(z, lang_embed) -> `out_dim` values, residual FiLM-conditioned MLP."""

    def __init__(self, latent_dim: int, lang_dim: int, hidden_dim: int, num_layers: int,
                 film_hidden: int = 256, dropout: float = 0.0, out_dim: int = 1):
        super().__init__()
        self.out_dim = int(out_dim)
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.film_gen = FiLMGenerator(lang_dim, hidden_dim, num_layers, film_hidden)
        self.blocks = nn.ModuleList([ResFiLMBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, z: torch.Tensor, lang_embed: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z)
        gammas, betas = self.film_gen(lang_embed)
        for i, block in enumerate(self.blocks):
            h = block(h, gammas[:, i, :], betas[:, i, :])
        h = self.out_norm(h)
        return self.out_head(h)
