#!/usr/bin/env python
"""Evaluate a neural decoder checkpoint."""

from __future__ import annotations

import argparse

from qecml.training.evaluate import evaluate_checkpoint
from qecml.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output")
    args = parser.parse_args()
    metrics = evaluate_checkpoint(args.checkpoint, args.data, batch_size=args.batch_size)
    print(metrics)
    if args.output:
        save_json(args.output, metrics)


if __name__ == "__main__":
    main()
