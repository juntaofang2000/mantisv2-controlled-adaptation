"""Official-model construction and the matched training protocol."""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .constants import (
    ADAM_BETAS,
    BATCH_SIZE,
    CHECKPOINT,
    CHECKPOINT_REVISION,
    EPOCHS,
    LEARNING_RATE,
    SEQ_LEN,
    WEIGHT_DECAY,
)


@dataclass(frozen=True)
class Protocol:
    seq_len: int = SEQ_LEN
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    micro_batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    beta1: float = ADAM_BETAS[0]
    beta2: float = ADAM_BETAS[1]
    learning_rate_adjusting: bool = True
    checkpoint: str = CHECKPOINT
    checkpoint_revision: str = CHECKPOINT_REVISION
    head: str = "LayerNorm+Linear"
    loss: str = "CrossEntropyLoss"
    return_transf_layer: int = -1
    output_token: str = "cls_token"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["effective_batch_size"] = self.batch_size
        payload["gradient_accumulation_steps"] = self.gradient_accumulation_steps
        return payload

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.batch_size // self.micro_batch_size

    def validate(self) -> None:
        if self.batch_size < 1 or self.micro_batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        if self.micro_batch_size > self.batch_size:
            raise ValueError("micro_batch_size cannot exceed effective batch_size")
        if self.batch_size % self.micro_batch_size:
            raise ValueError(
                "effective batch_size must be divisible by micro_batch_size"
            )


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for a matched run."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested!r}, but CUDA is unavailable")
    return requested


def build_backbone(mode: str, device: str, protocol: Protocol):
    """Build an official pretrained or genuinely random MantisV2 backbone."""
    from mantis.architecture import MantisV2

    network = MantisV2(
        device=device,
        return_transf_layer=protocol.return_transf_layer,
        output_token=protocol.output_token,
    )
    if mode != "scratch":
        network = network.from_pretrained(
            protocol.checkpoint, revision=protocol.checkpoint_revision
        )
    return network


def build_head(network, n_channels: int, n_classes: int, device: str):
    """Build the paper's LayerNorm + Linear head."""
    from torch import nn

    input_dim = int(network.hidden_dim) * int(n_channels)
    return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, n_classes)).to(
        device
    )


def count_parameters(module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def scheduled_learning_rate(
    *,
    epochs: int,
    updates_per_epoch: int,
    update_step: int,
    base_learning_rate: float,
) -> float:
    """Match the official v1.0.0 warm-up/cosine schedule per optimizer update."""
    max_steps = epochs * updates_per_epoch
    warmup_steps = 10 * updates_per_epoch
    if update_step < warmup_steps:
        multiplier = update_step / warmup_steps
    else:
        cosine_step = update_step - warmup_steps
        cosine_steps = max_steps - warmup_steps
        multiplier = 0.5 * (
            1 + math.cos(math.pi * cosine_step / (cosine_steps + 1e-15))
        )
    return multiplier * base_learning_rate


def fit_with_gradient_accumulation(
    trainer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    fine_tuning_type: str,
    head,
    protocol: Protocol,
    init_optimizer,
):
    """Mirror official MantisTrainer.fit while accumulating to an effective batch."""
    import torch
    from mantis.trainer.trainer_utils.architecture import FineTuningNetwork
    from mantis.trainer.trainer_utils.dataset import LabeledDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    protocol.validate()
    trainer.fine_tuning_type = fine_tuning_type

    if fine_tuning_type == "head":
        # Encoder inference uses the memory-safe micro-batch. The compact cached
        # embeddings can still train the head with the requested effective batch.
        embeddings = trainer.transform(
            X_train, batch_size=protocol.micro_batch_size, three_dim=False
        )
        trainer.fine_tuned_model = FineTuningNetwork(None, head, None).to(
            trainer.device
        )
        train_dataset = LabeledDataset(embeddings, y_train)
    else:
        trainer.fine_tuned_model = FineTuningNetwork(
            trainer.network, head, None
        ).to(trainer.device)
        train_dataset = LabeledDataset(X_train, y_train)

    parameters = list(
        trainer._get_fine_tuning_params(fine_tuning_type=fine_tuning_type)
    )
    params_to_optimize = set(parameters)
    for parameter in trainer.fine_tuned_model.parameters():
        parameter.requires_grad = parameter in params_to_optimize

    trainer.fine_tuned_model.eval()
    trainer._set_train(fine_tuning_type=fine_tuning_type)
    optimizer = init_optimizer(parameters)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    data_loader = DataLoader(
        train_dataset, batch_size=protocol.micro_batch_size, shuffle=True
    )

    updates_per_epoch = math.ceil(len(train_dataset) / protocol.batch_size)
    update_step = 1
    progress_bar = tqdm(range(protocol.epochs))
    for epoch in progress_bar:
        epoch_loss_sum = 0.0
        epoch_samples = 0
        accumulated_samples = 0
        optimizer.zero_grad()

        for batch_index, (x_batch, y_batch) in enumerate(data_loader):
            if accumulated_samples == 0 and protocol.learning_rate_adjusting:
                learning_rate = scheduled_learning_rate(
                    epochs=protocol.epochs,
                    updates_per_epoch=updates_per_epoch,
                    update_step=update_step,
                    base_learning_rate=protocol.learning_rate,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate

            x_batch = x_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)
            output = trainer.fine_tuned_model(x_batch)
            loss_sum = criterion(output, y_batch)
            loss_sum.backward()

            batch_samples = int(y_batch.shape[0])
            accumulated_samples += batch_samples
            epoch_samples += batch_samples
            epoch_loss_sum += float(loss_sum.detach().item())
            last_batch = batch_index + 1 == len(data_loader)

            if accumulated_samples >= protocol.batch_size or last_batch:
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)
                optimizer.step()
                optimizer.zero_grad()
                accumulated_samples = 0
                update_step += 1

        progress_bar.set_description(
            "Epoch {:d}: Train Loss {:.4f}".format(
                epoch, epoch_loss_sum / max(epoch_samples, 1)
            ),
            refresh=True,
        )
    return trainer.fine_tuned_model


def train_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    mode: str,
    seed: int,
    device: str,
    protocol: Protocol,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run one matched official-code experiment and return last-epoch metrics."""
    if mode not in {"frozen", "full", "scratch"}:
        raise ValueError(f"Unsupported mode: {mode}")
    protocol.validate()

    import torch
    from mantis.trainer import MantisTrainer
    from sklearn.preprocessing import LabelEncoder

    set_seed(seed, deterministic=deterministic)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train).astype(np.int64)
    try:
        y_test_encoded = label_encoder.transform(y_test).astype(np.int64)
    except ValueError as exc:
        raise ValueError("Test split contains a class absent from the training split") from exc

    network = build_backbone(mode, device, protocol)
    # Reset immediately before head creation so all modes share head initialization.
    set_seed(seed + 100_000, deterministic=deterministic)
    head = build_head(network, X_train.shape[1], len(label_encoder.classes_), device)
    trainer = MantisTrainer(device=device, network=network)

    def init_optimizer(params):
        return torch.optim.AdamW(
            params,
            lr=protocol.learning_rate,
            betas=(protocol.beta1, protocol.beta2),
            weight_decay=protocol.weight_decay,
        )

    fine_tuning_type = "head" if mode == "frozen" else mode
    # Reset before MantisTrainer constructs its shuffled DataLoader.
    set_seed(seed + 200_000, deterministic=deterministic)
    fit_with_gradient_accumulation(
        trainer,
        X_train,
        y_train_encoded,
        fine_tuning_type=fine_tuning_type,
        head=head,
        protocol=protocol,
        init_optimizer=init_optimizer,
    )

    predictions = trainer.predict(
        X_test, batch_size=protocol.micro_batch_size
    ).astype(int)
    accuracy = float(np.mean(predictions == y_test_encoded))
    encoder_total = int(sum(p.numel() for p in trainer.network.parameters()))
    head_total = int(sum(p.numel() for p in head.parameters()))
    # In official head mode the cached encoder is not inserted into
    # fine_tuned_model, so requires_grad remains True even though it is absent
    # from the optimizer. Report actually updated parameters, not that flag.
    encoder_updated = 0 if mode == "frozen" else encoder_total
    return {
        "accuracy": accuracy,
        "n_correct": int(np.sum(predictions == y_test_encoded)),
        "n_test": int(len(y_test_encoded)),
        "fine_tuning_type": fine_tuning_type,
        "total_model_parameters": encoder_total + head_total,
        "updated_model_parameters": encoder_updated + head_total,
        "encoder_parameters": encoder_total,
        "updated_encoder_parameters": encoder_updated,
        "effective_batch_size": protocol.batch_size,
        "micro_batch_size": protocol.micro_batch_size,
        "gradient_accumulation_steps": protocol.gradient_accumulation_steps,
        "classes": [str(value) for value in label_encoder.classes_],
    }
