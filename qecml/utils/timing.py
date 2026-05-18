"""Timing helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Measure elapsed wall-clock seconds in a mutable result dict."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
