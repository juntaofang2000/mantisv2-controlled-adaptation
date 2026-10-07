#!/usr/bin/env python3
"""Scan a benchmark root for fixed train/test numpy splits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive",
        default="auto",
        help="Fixed archive label, or auto to infer UCR/UEA from loaded shape later",
    )
    args = parser.parse_args()

    rows = []
    required = {"X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"}
    for directory in sorted(path for path in args.data_root.rglob("*") if path.is_dir()):
        if required.issubset({path.name for path in directory.iterdir()}):
            rows.append(
                {
                    "dataset": directory.name,
                    "source": "local",
                    "path": str(directory.resolve()),
                    "archive": args.archive,
                }
            )
    for npz in sorted(args.data_root.rglob("*.npz")):
        rows.append(
            {
                "dataset": npz.stem,
                "source": "local",
                "path": str(npz.resolve()),
                "archive": args.archive,
            }
        )
    if not rows:
        raise SystemExit(f"No supported dataset splits found below {args.data_root}")
    names = [row["dataset"] for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate dataset names; rename or edit manually: {duplicates}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "source", "path", "archive"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} datasets to {args.output}")


if __name__ == "__main__":
    main()

