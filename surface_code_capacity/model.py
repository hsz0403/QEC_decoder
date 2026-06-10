"""CNN decoder for code-capacity surface-code syndrome data."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBottleneck(nn.Module):
    """A small convolutional residual bottleneck block."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, bottleneck_dim, kernel_size=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(bottleneck_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SurfaceCodeCNNDecoder(nn.Module):
    """Syndrome -> logical-action CNN with check-grid scattering and logical pooling.

    Architecture:
    binary detection event -> learned H-dimensional embedding -> L convolutional
    residual bottleneck blocks -> check-to-data convolution -> pool over a logical
    support mask -> 2-layer MLP readout with hidden dimension 2H.
    """

    def __init__(
        self,
        *,
        detector_grid: torch.Tensor,
        grid_shape: tuple[int, int],
        logical_mask: torch.Tensor,
        num_observables: int,
        hidden_dim: int = 128,
        num_blocks: int = 8,
        bottleneck_ratio: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if detector_grid.ndim != 2 or detector_grid.shape[1] != 2:
            raise ValueError("detector_grid must have shape [num_detectors, 2]")
        if logical_mask.shape != grid_shape:
            raise ValueError("logical_mask shape must match grid_shape")
        self.num_detectors = int(detector_grid.shape[0])
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("detector_grid", detector_grid.long(), persistent=False)
        self.register_buffer("logical_mask", logical_mask.float(), persistent=False)

        self.event_embedding = nn.Embedding(2, hidden_dim)
        bottleneck_dim = max(8, hidden_dim // bottleneck_ratio)
        self.blocks = nn.Sequential(
            *[
                ResidualBottleneck(hidden_dim, bottleneck_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.check_to_data = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, num_observables),
        )

    def _scatter_checks_to_grid(self, embedded: torch.Tensor) -> torch.Tensor:
        batch_size = embedded.shape[0]
        height, width = self.grid_shape
        grid = embedded.new_zeros((batch_size, self.hidden_dim, height, width))
        y = self.detector_grid[:, 0]
        x = self.detector_grid[:, 1]
        grid[:, :, y, x] = embedded.transpose(1, 2)
        return grid

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        if syndrome.ndim != 2:
            raise ValueError("SurfaceCodeCNNDecoder expects syndrome shape [B, num_detectors]")
        if syndrome.shape[1] != self.num_detectors:
            raise ValueError(f"expected {self.num_detectors} detectors, got {syndrome.shape[1]}")
        embedded = self.event_embedding(syndrome.long().clamp(0, 1))
        x = self._scatter_checks_to_grid(embedded)
        x = self.blocks(x)
        x = self.check_to_data(x)
        mask = self.logical_mask.to(dtype=x.dtype, device=x.device)
        denom = mask.sum().clamp_min(1.0)
        pooled = (x * mask[None, None, :, :]).sum(dim=(2, 3)) / denom
        return self.readout(pooled)
