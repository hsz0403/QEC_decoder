#!/usr/bin/env python
"""Inspect the Zenodo 6804040 ZIP archive without assuming its schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from qecml.data.zenodo_6804040 import list_zip_tree, read_readmes_from_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        default="data/zenodo_6804040/raw/google_qec3v5_experiment_data.zip",
    )
    parser.add_argument("--output", default="runs/zenodo_6804040_inspection.md")
    args = parser.parse_args()
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"missing ZIP file: {zip_path}")

    names = list_zip_tree(zip_path)
    readmes = read_readmes_from_zip(zip_path)
    tokens = ["sample", "circuit", "predict", "metadata", "readme", ".h5", ".csv", ".json", ".txt"]
    interesting = [n for n in names if any(token in n.lower() for token in tokens)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("# Zenodo 6804040 inspection\n\n")
        f.write(f"ZIP: `{zip_path}`\n\n")
        f.write("## File tree\n\n")
        for name in names:
            f.write(f"- `{name}`\n")
        f.write("\n## README files\n\n")
        if not readmes:
            f.write("No README-like files found.\n\n")
        for name, text in readmes.items():
            f.write(f"### {name}\n\n```text\n{text}\n```\n\n")
        f.write("## Candidate data/metadata files\n\n")
        for name in interesting:
            f.write(f"- `{name}`\n")
    print(f"wrote {output}")
    print(f"files: {len(names)} readmes: {len(readmes)} candidates: {len(interesting)}")


if __name__ == "__main__":
    main()
