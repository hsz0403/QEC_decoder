#!/usr/bin/env python
"""Download Zenodo record 6804040 with zenodo_get if available."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/zenodo_6804040/raw")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["zenodo_get", "-o", str(out), "https://zenodo.org/records/6804040"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        msg = "zenodo_get is not installed. Install with: pip install zenodo_get"
        raise SystemExit(msg) from exc


if __name__ == "__main__":
    main()
