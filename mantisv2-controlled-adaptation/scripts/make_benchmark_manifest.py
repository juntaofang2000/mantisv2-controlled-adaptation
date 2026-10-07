#!/usr/bin/env python3
"""Build one manifest from UCR, UEA and MVCore66 dataset roots."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FILES = {"X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"}


def result_key(value: str) -> str:
    """Match the result-directory normalization used by the experiment runner."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return (cleaned.strip("._") or "dataset").casefold()


def scan_root(root: Path, archive: str) -> list[dict[str, str]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{archive} root is not a directory: {root}")

    rows: list[dict[str, str]] = []
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        if REQUIRED_FILES.issubset({path.name for path in directory.iterdir()}):
            rows.append(
                {
                    "dataset": directory.name,
                    "source": "local",
                    "path": str(directory),
                    "archive": archive,
                }
            )

    for npz in sorted(root.rglob("*.npz")):
        rows.append(
            {
                "dataset": npz.stem,
                "source": "local",
                "path": str(npz.resolve()),
                "archive": archive,
            }
        )

    if not rows:
        raise SystemExit(f"No supported dataset splits found below {archive}: {root}")

    by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_key[result_key(row["dataset"])].append(row)
    duplicates = {key: items for key, items in by_key.items() if len(items) > 1}
    if duplicates:
        details = "; ".join(
            f"{key}: {[item['path'] for item in items]}"
            for key, items in sorted(duplicates.items())
        )
        raise SystemExit(f"Duplicate dataset names within {archive}: {details}")
    return rows


def make_names_unique(rows: list[dict[str, str]], prefix_all: bool) -> list[str]:
    """Prefix archive names globally or only where output paths would collide."""
    counts = Counter(result_key(row["dataset"]) for row in rows)
    renamed: list[str] = []
    for row in rows:
        original = row["dataset"]
        if prefix_all or counts[result_key(original)] > 1:
            row["dataset"] = f"{row['archive']}_{original}"
            renamed.append(f"{original} -> {row['dataset']}")

    final_keys = [result_key(row["dataset"]) for row in rows]
    remaining = sorted(key for key, count in Counter(final_keys).items() if count > 1)
    if remaining:
        raise SystemExit(
            "Dataset names still collide after archive prefixing: " + ", ".join(remaining)
        )
    return renamed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan UCR, UEA and MVCore66 roots into one experiment manifest."
    )
    parser.add_argument("--ucr-root", type=Path, required=True)
    parser.add_argument("--uea-root", type=Path, required=True)
    parser.add_argument("--mvcore66-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/datasets.csv"))
    parser.add_argument(
        "--expected-count",
        type=int,
        default=198,
        help="Fail unless this many datasets are found; use 0 to disable (default: 198)",
    )
    parser.add_argument(
        "--prefix-all",
        action="store_true",
        help="Prefix every dataset name with its benchmark, not only collisions",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = [
        (args.ucr_root, "UCR"),
        (args.uea_root, "UEA"),
        (args.mvcore66_root, "MVCore66"),
    ]

    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for root, archive in roots:
        archive_rows = scan_root(root, archive)
        counts[archive] = len(archive_rows)
        rows.extend(archive_rows)

    if args.expected_count and len(rows) != args.expected_count:
        detail = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise SystemExit(
            f"Expected {args.expected_count} datasets, found {len(rows)} ({detail})"
        )

    renamed = make_names_unique(rows, prefix_all=args.prefix_all)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "source", "path", "archive"]
        )
        writer.writeheader()
        writer.writerows(rows)

    detail = ", ".join(f"{name}={count}" for name, count in counts.items())
    print(f"Wrote {len(rows)} datasets to {args.output} ({detail})")
    if renamed:
        print(f"Renamed {len(renamed)} colliding dataset entries:")
        for item in renamed:
            print(f"  {item}")


if __name__ == "__main__":
    main()
