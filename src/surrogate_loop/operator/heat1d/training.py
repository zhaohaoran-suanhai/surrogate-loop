from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from surrogate_loop.operator.config import OperatorRunSpec
from surrogate_loop.operator.heat1d.dataset import HeatDataset, NormalizationStats
from surrogate_loop.operator.heat1d.deeponet import DeepONet, build_deeponet
from surrogate_loop.operator.runtime import seed_everything


class TrainingSplit(Protocol):
    train: HeatDataset
    validation: HeatDataset


@dataclass(frozen=True)
class TrainingRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    state_dict: dict[str, Tensor]
    history: tuple[TrainingRecord, ...]
    best_epoch: int
    stop_reason: str
    device: str
    elapsed_seconds: float
    peak_cuda_memory_mb: float


def train_deeponet(
    spec: OperatorRunSpec,
    split: TrainingSplit,
    normalization: NormalizationStats,
    device: torch.device,
) -> TrainingResult:
    seed_everything(spec.sampling.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = build_deeponet(spec.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=spec.training.lr_factor,
        patience=max(1, spec.training.patience // 3),
        min_lr=spec.training.min_learning_rate,
    )
    train_parameters = normalization.normalize_parameters(split.train.parameters).astype(
        np.float32
    )
    train_coordinates = normalization.normalize_coordinates(
        _coordinate_grid(split.train)
    ).astype(np.float32)
    train_targets = normalization.normalize_targets(split.train.fields).reshape(
        split.train.fields.shape[0], -1
    ).astype(np.float32)
    validation_parameters = normalization.normalize_parameters(
        split.validation.parameters
    ).astype(np.float32)
    validation_coordinates = normalization.normalize_coordinates(
        _coordinate_grid(split.validation)
    ).astype(np.float32)
    validation_targets = normalization.normalize_targets(split.validation.fields).reshape(
        split.validation.fields.shape[0], -1
    ).astype(np.float32)
    rng = np.random.default_rng(spec.sampling.seed)
    best_validation = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] = {}
    epochs_without_improvement = 0
    history: list[TrainingRecord] = []
    stop_reason = "max_epochs"

    for epoch in range(spec.training.max_epochs):
        model.train()
        permutation = rng.permutation(train_parameters.shape[0])
        batch_losses: list[float] = []
        for start in range(0, permutation.size, spec.training.case_batch_size):
            case_indices = permutation[start : start + spec.training.case_batch_size]
            query_count = min(spec.training.query_batch_size, train_coordinates.shape[0])
            query_indices = rng.choice(
                train_coordinates.shape[0], size=query_count, replace=False
            )
            branch = torch.as_tensor(train_parameters[case_indices], device=device)
            trunk = torch.as_tensor(train_coordinates[query_indices], device=device)
            target = torch.as_tensor(
                train_targets[case_indices][:, query_indices], device=device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(branch, trunk)
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("DeepONet 训练损失出现 NaN 或 Inf")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        validation_prediction = _predict_normalized(
            model,
            validation_parameters,
            validation_coordinates,
            device,
            spec.training.query_batch_size,
        )
        validation_loss = float(np.mean((validation_prediction - validation_targets) ** 2))
        if not np.isfinite(validation_loss):
            raise FloatingPointError("DeepONet 验证损失出现 NaN 或 Inf")
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            TrainingRecord(
                epoch=epoch,
                train_loss=float(np.mean(batch_losses)),
                validation_loss=validation_loss,
                learning_rate=learning_rate,
            )
        )
        if best_validation - validation_loss > spec.training.min_delta:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= spec.training.patience:
            stop_reason = "early_stopping"
            break
        if (time.perf_counter() - started) / 60.0 >= spec.training.max_minutes:
            stop_reason = "time_budget"
            break

    elapsed_seconds = time.perf_counter() - started
    if not best_state:
        raise RuntimeError("训练未产生有效的最佳检查点")
    peak_cuda_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "cuda"
        else 0.0
    )
    return TrainingResult(
        state_dict=best_state,
        history=tuple(history),
        best_epoch=best_epoch,
        stop_reason=stop_reason,
        device=str(device),
        elapsed_seconds=elapsed_seconds,
        peak_cuda_memory_mb=peak_cuda_memory_mb,
    )


def predict_dataset(
    model: DeepONet,
    dataset: HeatDataset,
    normalization: NormalizationStats,
    device: torch.device,
    query_batch_size: int,
) -> NDArray[np.float64]:
    normalized_parameters = normalization.normalize_parameters(dataset.parameters).astype(
        np.float32
    )
    normalized_coordinates = normalization.normalize_coordinates(
        _coordinate_grid(dataset)
    ).astype(np.float32)
    normalized_prediction = _predict_normalized(
        model,
        normalized_parameters,
        normalized_coordinates,
        device,
        query_batch_size,
    )
    physical_prediction = normalization.denormalize_targets(normalized_prediction)
    return physical_prediction.reshape(dataset.fields.shape)


def _predict_normalized(
    model: DeepONet,
    parameters: NDArray[np.float32],
    coordinates: NDArray[np.float32],
    device: torch.device,
    query_batch_size: int,
) -> NDArray[np.float32]:
    if query_batch_size <= 0:
        raise ValueError("query_batch_size 必须为正数")
    model.eval()
    branch = torch.as_tensor(parameters, device=device)
    batches: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, coordinates.shape[0], query_batch_size):
            trunk = torch.as_tensor(coordinates[start : start + query_batch_size], device=device)
            prediction = model(branch, trunk)
            batches.append(prediction.cpu().numpy())
    return np.concatenate(batches, axis=1)


def _coordinate_grid(dataset: HeatDataset) -> NDArray[np.float64]:
    return np.stack(np.meshgrid(dataset.x, dataset.t, indexing="xy"), axis=-1).reshape(
        -1, 2
    )
