"""Training and evaluation metrics."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def logical_error_rate_np(pred: np.ndarray, labels: np.ndarray) -> float:
    """Return sample-wise logical error rate for binary arrays."""
    pred_bool = np.asarray(pred).astype(bool)
    labels_bool = np.asarray(labels).astype(bool)
    return float(np.mean(np.any(pred_bool != labels_bool, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute binary expected calibration error over all observables."""
    probs = np.asarray(probabilities).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask) * abs(np.mean(probs[mask]) - np.mean(y[mask])))
    return ece


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Compute common neural decoder metrics from logits."""
    labels_f = labels.float()
    probs = torch.sigmoid(logits)
    pred = probs > 0.5
    labels_b = labels_f > 0.5
    bce = F.binary_cross_entropy_with_logits(logits, labels_f).item()
    pred_np = pred.detach().cpu().numpy()
    labels_np = labels_b.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()

    result: dict[str, Any] = {
        "bce": bce,
        "logical_error_rate": logical_error_rate_np(pred_np, labels_np),
        "accuracy": float(np.mean(pred_np == labels_np)),
        "positive_rate": float(np.mean(labels_np)),
        "prediction_positive_rate": float(np.mean(pred_np)),
        "expected_calibration_error": expected_calibration_error(probs_np, labels_np),
    }
    flat_labels = labels_np.reshape(-1).astype(int)
    if len(np.unique(flat_labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(flat_labels, probs_np.reshape(-1)))
    else:
        result["roc_auc"] = None
    return result


@torch.no_grad()
def inference_time_per_sample(
    model: torch.nn.Module,
    events: torch.Tensor,
    *,
    repeats: int = 5,
) -> float:
    """Measure average inference time per sample in seconds."""
    model.eval()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(events)
    elapsed = time.perf_counter() - start
    return elapsed / (repeats * events.shape[0])
