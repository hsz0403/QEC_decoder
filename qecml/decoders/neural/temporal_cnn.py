"""Temporal CNN decoder for dense detector events."""

from __future__ import annotations

import torch
from torch import nn


class TemporalCNNDecoder(nn.Module):
    """A small 1D convolutional decoder over syndrome rounds."""

    def __init__(
        self,
        num_sites: int,
        num_features: int,
        num_observables: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_channels = num_sites * num_features
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_observables),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """Return logical flip logits from dense events ``[B, T, S, F]``."""
        if events.ndim != 4:
            raise ValueError("TemporalCNNDecoder expects events with shape [B, T, S, F]")
        b, t, s, f = events.shape
        x = events.float().reshape(b, t, s * f).transpose(1, 2)
        return self.net(x)
