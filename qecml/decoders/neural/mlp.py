"""Flat MLP neural decoder."""

from __future__ import annotations

import torch
from torch import nn


class FlatMLPDecoder(nn.Module):
    """Simple MLP decoder for flat detector events ``[B, N]``."""

    def __init__(
        self,
        num_detectors: int,
        num_observables: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_detectors, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_observables),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """Return logical flip logits."""
        if events.ndim != 2:
            events = events.flatten(start_dim=1)
        return self.net(events.float())
