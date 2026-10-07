"""Aggregate completed runs into paired task-level statistics and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import MODES, SEEDS


def read_runs(results_root: Path) -> pd.DataFrame:
    records: list[dict] = []
    for path in sorted((results_root / "runs").glob("*/*/seed_*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        meta = payload.get("dataset_metadata", {})
        metrics = payload.get("metrics", {})
        records.append(
            {
                "file": str(path),
                "status": payload.get("status"),
                "dataset": payload.get("dataset"),
                "archive": payload.get("archive") or meta.get("archive", "unknown"),
                "mode": payload.get("mode"),
                "seed": payload.get("seed"),
                "accuracy": metrics.get("accuracy"),
                "n_train": meta.get("n_train"),
                "n_test": meta.get("n_test"),
                "n_classes": meta.get("n_classes"),
                "n_channels": meta.get("n_channels"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "updated_encoder_parameters": metrics.get(
                    "updated_encoder_parameters"
                ),
                "error_type": payload.get("error_type"),
                "error": payload.get("error"),
            }
        )
    if not records:
        raise FileNotFoundError(f"No run JSON files under {results_root / 'runs'}")
    return pd.DataFrame.from_records(records)


def expected_grid(per_run: pd.DataFrame, task_file: Path | None = None) -> pd.DataFrame:
    if task_file is not None:
        expected = pd.read_csv(task_file)[["dataset", "mode", "seed"]].drop_duplicates()
    else:
        datasets = sorted(per_run["dataset"].dropna().unique())
        expected = pd.MultiIndex.from_product(
            [datasets, MODES, SEEDS], names=["dataset", "mode", "seed"]
        ).to_frame(index=False)
    actual = per_run[["dataset", "mode", "seed", "status", "file", "error"]]
    return expected.merge(actual, how="left", on=["dataset", "mode", "seed"])


def make_per_dataset(per_run: pd.DataFrame) -> pd.DataFrame:
    complete = per_run.loc[per_run["status"] == "complete"].copy()
    counts = complete.groupby(["dataset", "mode"])["seed"].nunique()
    valid = counts[counts == len(SEEDS)].index
    complete = complete.set_index(["dataset", "mode"]).loc[
        complete.set_index(["dataset", "mode"]).index.isin(valid)
    ].reset_index()

    info_cols = ["archive", "n_train", "n_test", "n_classes", "n_channels"]
    info = complete.groupby("dataset", as_index=True)[info_cols].first()
    means = complete.pivot_table(
        index="dataset", columns="mode", values="accuracy", aggfunc="mean"
    )
    seed_sd = complete.pivot_table(
        index="dataset", columns="mode", values="accuracy", aggfunc="std"
    ).add_suffix("_seed_sd")
    table = info.join(means).join(seed_sd)
    for mode in MODES:
        if mode not in table:
            table[mode] = np.nan
    table["delta_full_frozen"] = table["full"] - table["frozen"]
    table["delta_full_scratch"] = table["full"] - table["scratch"]
    table["data_type"] = np.where(table["n_channels"] == 1, "UV", "MV")
    table["train_size_group"] = pd.cut(
        table["n_train"],
        bins=[-np.inf, 100, 500, np.inf],
        labels=["<=100", "101-500", ">500"],
    ).astype(str)
    table["class_group"] = pd.cut(
        table["n_classes"],
        bins=[-np.inf, 2, 10, np.inf],
        labels=["2", "3-10", ">10"],
    ).astype(str)
    return table.reset_index()


def wtl(delta: pd.Series, atol: float = 1e-12) -> tuple[int, int, int]:
    values = delta.dropna().to_numpy()
    wins = int(np.sum(values > atol))
    ties = int(np.sum(np.abs(values) <= atol))
    losses = int(np.sum(values < -atol))
    return wins, ties, losses


def fraction_above(values: pd.Series, threshold: float, atol: float = 1e-12) -> float:
    """Fraction strictly above a threshold, robust to binary float noise."""
    valid = values.dropna()
    return float((valid > threshold + atol).mean())


def fraction_below(values: pd.Series, threshold: float, atol: float = 1e-12) -> float:
    """Fraction strictly below a threshold, robust to binary float noise."""
    valid = values.dropna()
    return float((valid < threshold - atol).mean())


def bootstrap_mean_ci(
    values: Iterable[float], repetitions: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    boot_means = array[indices].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def make_main_table(per_dataset: pd.DataFrame) -> pd.DataFrame:
    available = [mode for mode in MODES if per_dataset[mode].notna().any()]
    ranks = per_dataset[available].rank(axis=1, ascending=False, method="average")
    rows = []
    for mode in available:
        versus_frozen = per_dataset[mode] - per_dataset["frozen"]
        versus_scratch = per_dataset[mode] - per_dataset["scratch"]
        valid_vs_frozen = versus_frozen.dropna()
        wins, ties, losses = wtl(valid_vs_frozen)
        rows.append(
            {
                "mode": mode,
                "n_datasets": int(per_dataset[mode].notna().sum()),
                "mean_accuracy_percent": 100 * per_dataset[mode].mean(),
                "average_rank": ranks[mode].mean(),
                "delta_vs_frozen_pp": 100 * versus_frozen.mean(),
                "delta_vs_scratch_pp": 100 * versus_scratch.mean(),
                "wins_vs_frozen": wins,
                "ties_vs_frozen": ties,
                "losses_vs_frozen": losses,
                "gain_gt_1pp_vs_frozen_percent": 100
                * fraction_above(valid_vs_frozen, 0.01),
                "drop_gt_1pp_vs_frozen_percent": 100
                * fraction_below(valid_vs_frozen, -0.01),
            }
        )
    return pd.DataFrame(rows)


def make_paired_effects(
    per_dataset: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    contrasts = {
        "full-frozen": per_dataset["delta_full_frozen"],
        "full-scratch": per_dataset["delta_full_scratch"],
    }
    rows = []
    for offset, (name, delta) in enumerate(contrasts.items()):
        valid = delta.dropna()
        mean, low, high = bootstrap_mean_ci(valid, repetitions, seed + offset)
        wins, ties, losses = wtl(valid)
        rows.append(
            {
                "contrast": name,
                "n_datasets": len(valid),
                "mean_delta_pp": 100 * mean,
                "ci95_low_pp": 100 * low,
                "ci95_high_pp": 100 * high,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "any_negative_percent": 100 * float((valid < 0).mean()),
                "negative_gt_1pp_percent": 100 * fraction_below(valid, -0.01),
                "positive_gt_1pp_percent": 100 * fraction_above(valid, 0.01),
                "bootstrap_unit": "dataset",
                "bootstrap_repetitions": repetitions,
            }
        )
    return pd.DataFrame(rows)


def make_grouped_effects(per_dataset: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for column in ("archive", "data_type", "train_size_group", "class_group"):
        grouped = per_dataset.groupby(column, observed=True, dropna=False)
        summary = grouped.agg(
            n_datasets=("dataset", "size"),
            mean_full_frozen=("delta_full_frozen", "mean"),
            mean_full_scratch=("delta_full_scratch", "mean"),
        ).reset_index(names="group")
        summary.insert(0, "grouping", column)
        summary["mean_full_frozen_pp"] = 100 * summary.pop("mean_full_frozen")
        summary["mean_full_scratch_pp"] = 100 * summary.pop("mean_full_scratch")
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def plot_ecdf(per_dataset: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    contrasts = [
        ("delta_full_frozen", "Full FT − Frozen"),
        ("delta_full_scratch", "Pretrained FT − Scratch"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis, (column, title) in zip(axes, contrasts):
        values = 100 * per_dataset[column].dropna().sort_values().to_numpy()
        y = np.arange(1, len(values) + 1) / len(values)
        axis.step(values, y, where="post", linewidth=2)
        axis.axvline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Accuracy difference (percentage points)")
        axis.set_ylabel("Fraction of datasets")
        axis.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"delta_ecdf.{suffix}", dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument(
        "--task-file",
        type=Path,
        help="Optional task CSV used to detect datasets with no result file at all",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run = read_runs(args.results_root)
    completeness = expected_grid(per_run, args.task_file)
    per_dataset = make_per_dataset(per_run)
    main_table = make_main_table(per_dataset)
    paired = make_paired_effects(
        per_dataset, args.bootstrap_repetitions, args.bootstrap_seed
    )
    grouped = make_grouped_effects(per_dataset)

    per_run.to_csv(args.output_dir / "per_run.csv", index=False)
    completeness.to_csv(args.output_dir / "completeness.csv", index=False)
    per_dataset.to_csv(args.output_dir / "per_dataset.csv", index=False)
    main_table.to_csv(args.output_dir / "main_table.csv", index=False)
    paired.to_csv(args.output_dir / "paired_effects.csv", index=False)
    grouped.to_csv(args.output_dir / "grouped_effects.csv", index=False)
    plot_ecdf(per_dataset, args.output_dir)
    print(main_table.to_string(index=False))
    print("\nPaired effects")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
