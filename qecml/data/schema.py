"""Common QEC dataset schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class QECBatch:
    """A batch of QEC detector-event data.

    ``events`` can be flat ``[B, N]`` or dense ``[B, T, S, F]``.
    ``labels`` stores logical observable flips with shape ``[B, K]``.
    """

    events: torch.Tensor
    labels: torch.Tensor
    mask: Optional[torch.Tensor] = None
    metadata: Optional[dict[str, Any]] = None
