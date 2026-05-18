"""Loss helpers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary cross entropy with logits for logical observable flips."""
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
