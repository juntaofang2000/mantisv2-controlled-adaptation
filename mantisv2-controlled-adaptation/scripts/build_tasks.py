#!/usr/bin/env python3
"""Expand a dataset manifest into dataset × mode × seed tasks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["frozen", "full", "scratch"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    unknown = set(args.modes) - {"frozen", "full", "scratch"}
    if unknown:
        raise SystemExit(f"Unknown modes: {sorted(unknown)}")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        datasets = list(reader)
        manifest_fields = list(reader.fieldnames or [])
    if not datasets:
        raise SystemExit("Manifest is empty")
    required = {"dataset", "source", "path", "archive"}
    missing = required - set(datasets[0])
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
    rows = []
    for dataset in datasets:
        for mode in args.modes:
            for seed in args.seeds:
                rows.append({**dataset, "mode": mode, "seed": seed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [field for field in manifest_fields if field not in {"mode", "seed"}]
    fields.extend(["mode", "seed"])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()

