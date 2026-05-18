"""Mini AlphaQubit-style recurrent transformer decoder."""

from __future__ import annotations

import torch
from torch import nn


class RecurrentSiteBlock(nn.Module):
    """One recurrent per-site update followed by site self-attention and FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Update site states for one time step."""
        b, s, d = h.shape
        updated = self.gru(x_t.reshape(b * s, d), h.reshape(b * s, d)).reshape(b, s, d)
        attn_out, _ = self.attn(updated, updated, updated, need_weights=False)
        h = self.norm_attn(updated + self.dropout(attn_out))
        h = self.norm_ffn(h + self.dropout(self.ffn(h)))
        return h


class MiniAlphaQubitDecoder(nn.Module):
    """Small recurrent transformer-like decoder for dense events ``[B, T, S, F]``."""

    def __init__(
        self,
        num_sites: int,
        num_features: int,
        num_observables: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_sites = num_sites
        self.num_features = num_features
        self.input_encoder = nn.Linear(num_features, hidden_dim)
        self.site_embedding = nn.Embedding(num_sites, hidden_dim)
        self.blocks = nn.ModuleList(
            [RecurrentSiteBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_observables),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """Return logical flip logits."""
        if events.ndim != 4:
            raise ValueError("MiniAlphaQubitDecoder expects dense events with shape [B, T, S, F]")
        b, t, s, f = events.shape
        if s != self.num_sites or f != self.num_features:
            raise ValueError(
                f"expected S={self.num_sites}, F={self.num_features}; got S={s}, F={f}"
            )
        device = events.device
        site_ids = torch.arange(s, device=device)
        site_emb = self.site_embedding(site_ids).unsqueeze(0).expand(b, s, -1)
        h_layers = [torch.zeros_like(site_emb) for _ in self.blocks]
        for ti in range(t):
            x = self.input_encoder(events[:, ti].float()) + site_emb
            for layer_idx, block in enumerate(self.blocks):
                h_layers[layer_idx] = block(h_layers[layer_idx], x)
                x = h_layers[layer_idx]
        pooled = h_layers[-1].mean(dim=1)
        return self.readout(pooled)
