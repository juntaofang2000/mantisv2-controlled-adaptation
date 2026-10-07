"""Dataset and result I/O."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ARRAYS = ("X_train", "y_train", "X_test", "y_test")


def safe_name(value: str) -> str:
    """Return a filesystem-safe, stable dataset name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "dataset"


def _load_local(path: Path) -> tuple[Any, np.ndarray, Any, np.ndarray]:
    if path.is_file() and path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in REQUIRED_ARRAYS if key not in data]
            if missing:
                raise KeyError(f"{path} is missing keys: {missing}")
            return tuple(data[key] for key in REQUIRED_ARRAYS)  # type: ignore[return-value]

    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    values = []
    for key in REQUIRED_ARRAYS:
        file_path = path / f"{key}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required array: {file_path}")
        values.append(np.load(file_path, allow_pickle=True))
    return tuple(values)  # type: ignore[return-value]


def _load_aeon(name: str, cache_dir: Path) -> tuple[Any, np.ndarray, Any, np.ndarray]:
    try:
        from aeon.datasets import load_classification
    except ImportError as exc:
        raise ImportError(
            "source=aeon requires the optional dependency: pip install -e '.[archive]'"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"extract_path": str(cache_dir)}
    X_train, y_train = load_classification(name, split="train", **kwargs)
    X_test, y_test = load_classification(name, split="test", **kwargs)
    return X_train, np.asarray(y_train), X_test, np.asarray(y_test)


def _instance_to_channels_time(instance: Any) -> np.ndarray:
    if hasattr(instance, "to_numpy"):
        arr = instance.to_numpy()
        # aeon/pandas instances conventionally use time rows and channel columns.
        if arr.ndim == 2:
            arr = arr.T
    else:
        arr = np.asarray(instance)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Each instance must be 1D or 2D; got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def to_instance_list(X: Any) -> list[np.ndarray]:
    """Normalize common archive containers to a list of (channels, time) arrays."""
    if isinstance(X, np.ndarray) and X.dtype != object:
        if X.ndim == 2:
            return [np.asarray(row[None, :], dtype=np.float32) for row in X]
        if X.ndim == 3:
            return [np.asarray(row, dtype=np.float32) for row in X]
        raise ValueError(f"Expected a 2D/3D collection, got shape {X.shape}")

    if hasattr(X, "columns") and hasattr(X, "groupby"):
        raise TypeError(
            "Nested dataframe collections are not accepted implicitly. Export the "
            "fixed split as numpy arrays or use a current aeon loader."
        )

    return [_instance_to_channels_time(instance) for instance in list(X)]


def resize_collection(X: Any, seq_len: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply PyTorch linear interpolation exactly once to every instance."""
    import torch
    import torch.nn.functional as F

    instances = to_instance_list(X)
    if not instances:
        raise ValueError("Empty time-series collection")
    channel_counts = {arr.shape[0] for arr in instances}
    if len(channel_counts) != 1:
        raise ValueError(f"Channel count varies across instances: {channel_counts}")
    lengths = [int(arr.shape[1]) for arr in instances]
    for idx, arr in enumerate(instances):
        if not np.isfinite(arr).all():
            raise ValueError(
                f"Instance {idx} contains NaN/Inf. Apply the benchmark's fixed "
                "imputation upstream; this runner will not invent one."
            )

    if len(set(lengths)) == 1:
        stacked = np.stack(instances).astype(np.float32, copy=False)
        if lengths[0] != seq_len:
            tensor = torch.as_tensor(stacked, dtype=torch.float32)
            stacked = F.interpolate(
                tensor, size=seq_len, mode="linear", align_corners=False
            ).cpu().numpy()
    else:
        resized = []
        for arr in instances:
            tensor = torch.as_tensor(arr[None, :, :], dtype=torch.float32)
            resized.append(
                F.interpolate(
                    tensor, size=seq_len, mode="linear", align_corners=False
                )[0].cpu().numpy()
            )
        stacked = np.stack(resized).astype(np.float32, copy=False)

    metadata = {
        "n_instances": len(instances),
        "n_channels": int(stacked.shape[1]),
        "original_length_min": min(lengths),
        "original_length_max": max(lengths),
        "resized_length": seq_len,
    }
    return stacked, metadata


def load_dataset(
    row: dict[str, str], seq_len: int, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source = row.get("source", "local").strip().lower()
    name = row["dataset"].strip()
    if source == "local":
        path = Path(os.path.expandvars(os.path.expanduser(row["path"]))).resolve()
        X_train_raw, y_train, X_test_raw, y_test = _load_local(path)
    elif source == "aeon":
        X_train_raw, y_train, X_test_raw, y_test = _load_aeon(name, cache_dir)
        path = cache_dir
    else:
        raise ValueError(f"Unknown source={source!r}; expected 'local' or 'aeon'")

    X_train, train_meta = resize_collection(X_train_raw, seq_len)
    X_test, test_meta = resize_collection(X_test_raw, seq_len)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Train/test channel mismatch: {X_train.shape[1]} vs {X_test.shape[1]}"
        )
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    if len(y_train) != len(X_train) or len(y_test) != len(X_test):
        raise ValueError("Feature/label length mismatch")

    digest = hashlib.sha256()
    for arr in (X_train, y_train.astype(str), X_test, y_test.astype(str)):
        digest.update(np.ascontiguousarray(arr).tobytes())

    labels = np.unique(y_train)
    metadata = {
        "dataset": name,
        "source": source,
        "archive": row.get("archive", "").strip() or "unknown",
        "resolved_path": str(path),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(labels)),
        "n_channels": int(X_train.shape[1]),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "train_original_length_min": train_meta["original_length_min"],
        "train_original_length_max": train_meta["original_length_max"],
        "test_original_length_min": test_meta["original_length_min"],
        "test_original_length_max": test_meta["original_length_max"],
        "preprocessed_sha256": digest.hexdigest(),
    }
    return X_train, y_train, X_test, y_test, metadata


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp, path)

