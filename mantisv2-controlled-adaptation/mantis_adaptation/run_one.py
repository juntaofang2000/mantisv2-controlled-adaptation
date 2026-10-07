"""CLI for one task-matrix row."""

from __future__ import annotations

import argparse
import csv
import platform
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .constants import MANTIS_GIT_COMMIT, MANTIS_VERSION
from .io import atomic_write_json, load_dataset, safe_name
from .protocol import Protocol, resolve_device, train_and_score


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def read_task(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if index < 0 or index >= len(rows):
        raise IndexError(f"task-index {index} outside [0, {len(rows) - 1}]")
    required = {"dataset", "source", "mode", "seed"}
    missing = required - set(rows[index])
    if missing:
        raise KeyError(f"Task file is missing columns: {sorted(missing)}")
    return rows[index]


def output_path(root: Path, row: dict[str, str]) -> Path:
    return (
        root
        / "runs"
        / safe_name(row["dataset"])
        / row["mode"]
        / f"seed_{int(row['seed'])}.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="Physical batch size; defaults to task CSV or effective batch size",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        help="Optimizer batch size after accumulation; defaults to task CSV or 128",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-nondeterministic", action="store_true", help="Do not request deterministic kernels"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    row = read_task(args.task_file, args.task_index)
    destination = output_path(args.output_root, row)
    if destination.exists() and not args.overwrite:
        print(f"SKIP {destination}")
        return

    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    protocol = Protocol()
    task_effective = (row.get("effective_batch_size") or "").strip()
    task_micro = (row.get("micro_batch_size") or "").strip()
    effective_batch_size = (
        args.effective_batch_size
        if args.effective_batch_size is not None
        else int(task_effective) if task_effective else protocol.batch_size
    )
    micro_batch_size = (
        args.micro_batch_size
        if args.micro_batch_size is not None
        else int(task_micro) if task_micro else effective_batch_size
    )
    protocol = replace(
        protocol,
        batch_size=effective_batch_size,
        micro_batch_size=micro_batch_size,
    )
    protocol.validate()
    base = {
        "status": "running",
        "task_index": args.task_index,
        "dataset": row["dataset"],
        "source": row.get("source", "local"),
        "archive": row.get("archive", ""),
        "mode": row["mode"],
        "seed": int(row["seed"]),
        "started_at_utc": timestamp,
        "protocol": protocol.to_dict(),
        "provenance": {
            "mantis_package_expected": MANTIS_VERSION,
            "mantis_package_installed": package_version("mantis-tsfm"),
            "mantis_git_commit": MANTIS_GIT_COMMIT,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    try:
        device = resolve_device(args.device)
        X_train, y_train, X_test, y_test, dataset_meta = load_dataset(
            row, protocol.seq_len, args.cache_dir
        )
        metrics = train_and_score(
            X_train,
            y_train,
            X_test,
            y_test,
            mode=row["mode"],
            seed=int(row["seed"]),
            device=device,
            protocol=protocol,
            deterministic=not args.allow_nondeterministic,
        )
        base.update(
            {
                "status": "complete",
                "device": device,
                "dataset_metadata": dataset_meta,
                "metrics": metrics,
            }
        )
    except Exception as exc:
        base.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        base["elapsed_seconds"] = time.perf_counter() - started
        base["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(destination, base)

    print(f"{base['status'].upper()} {destination}")
    if base["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

