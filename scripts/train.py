#!/usr/bin/env python
"""Train a neural decoder."""

from __future__ import annotations

import argparse

from qecml.training.trainer import train_from_config
from qecml.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = train_from_config(load_yaml(args.config))
    print(result)


if __name__ == "__main__":
    main()
