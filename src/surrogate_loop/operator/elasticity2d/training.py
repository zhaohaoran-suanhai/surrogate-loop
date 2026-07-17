from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from surrogate_loop.operator.elasticity2d.config import ElasticityRunSpec
from surrogate_loop.operator.elasticity2d.deeponet import (
    apply_elasticity_constraints,
    build_elasticity_deeponet,
)
from surrogate_loop.operator.elasticity2d.problem import elasticity_basis_features
from surrogate_loop.operator.field_data import FieldDataset, FieldNormalization
from surrogate_loop.operator.runtime import seed_everything
from surrogate_loop.operator.vector_deeponet import VectorDeepONet


class TrainingPartitions(Protocol):
    train: FieldDataset
    validation: FieldDataset


@dataclass(frozen=True)
class TrainingRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    seed: int
    state_dict: dict[str, Tensor]
    history: tuple[TrainingRecord, ...]
    best_epoch: int
    validation_loss: float
    stop_reason: str
    device: str
    elapsed_seconds: float
    peak_cuda_memory_mb: float


@dataclass(frozen=True)
class SelectedTraining:
    selected_seed: int
    selected: TrainingResult
    candidates: tuple[TrainingResult, ...]


class TrainingFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        failure_epoch: int,
        state_dict: dict[str, Tensor],
        history: tuple[TrainingRecord, ...],
        seed: int,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.failure_epoch = failure_epoch
        self.state_dict = state_dict
        self.history = history
        self.seed = seed


def train_one_seed(
    spec: ElasticityRunSpec,
    partitions: TrainingPartitions,
    normalization: FieldNormalization,
    device: torch.device,
    seed: int,
) -> TrainingResult:
    if seed not in spec.training.seeds:
        raise ValueError("训练种子不在结构化规格中")
    _validate_training_contract(partitions, normalization)
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = build_elasticity_deeponet(spec.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=spec.training.lr_factor,
        patience=max(1, spec.training.patience // 3),
        min_lr=spec.training.min_learning_rate,
    )

    train = _prepare_dataset(partitions.train, normalization)
    validation = _prepare_dataset(partitions.validation, normalization)
    component_rms = _shape_component_rms(train)
    component_rms_tensor = torch.as_tensor(component_rms, device=device)
    rng = np.random.default_rng(seed)
    best_validation = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] = {}
    epochs_without_improvement = 0
    history: list[TrainingRecord] = []
    stop_reason = "max_epochs"

    for epoch in range(spec.training.max_epochs):
        model.train()
        permutation = rng.permutation(train.features.shape[0])
        batch_losses: list[float] = []
        for start in range(0, permutation.size, spec.training.case_batch_size):
            case_indices = permutation[start : start + spec.training.case_batch_size]
            query_count = min(
                spec.training.query_batch_size, train.coordinates.shape[0]
            )
            query_indices = rng.choice(
                train.coordinates.shape[0], size=query_count, replace=False
            )
            try:
                branch = torch.as_tensor(train.features[case_indices], device=device)
                trunk = torch.as_tensor(train.coordinates[query_indices], device=device)
                physical_branch = torch.as_tensor(
                    train.physical_parameters[case_indices], device=device
                )
                physical_trunk = torch.as_tensor(
                    train.physical_coordinates[query_indices], device=device
                )
                target = torch.as_tensor(
                    train.targets[case_indices][:, query_indices], device=device
                )
                optimizer.zero_grad(set_to_none=True)
                raw = model(branch, trunk)
                prediction = apply_elasticity_constraints(
                    raw, physical_branch, physical_trunk
                )
                loss = _balanced_shape_loss(
                    prediction,
                    target,
                    physical_branch,
                    component_rms_tensor,
                )
                if not torch.isfinite(loss):
                    raise _training_failure(
                        "Vector DeepONet 训练损失出现 NaN 或 Inf",
                        "non_finite_train_loss",
                        epoch,
                        model,
                        history,
                        best_state,
                        seed,
                    )
                loss.backward()
                optimizer.step()
            except TrainingFailure:
                raise
            except torch.cuda.OutOfMemoryError as error:
                raise _training_failure(
                    "Vector DeepONet 训练发生 CUDA OOM",
                    "cuda_oom",
                    epoch,
                    model,
                    history,
                    best_state,
                    seed,
                ) from error
            batch_losses.append(float(loss.detach().cpu()))

        try:
            validation_prediction = _predict_physical(
                model,
                validation,
                device,
                spec.training.query_batch_size,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise _training_failure(
                "Vector DeepONet 验证发生 CUDA OOM",
                "cuda_oom",
                epoch,
                model,
                history,
                best_state,
                seed,
            ) from error
        validation_loss = float(
            _balanced_shape_loss(
                torch.as_tensor(validation_prediction),
                torch.as_tensor(validation.targets),
                torch.as_tensor(validation.physical_parameters),
                torch.as_tensor(component_rms),
            )
        )
        if not np.isfinite(validation_loss):
            raise _training_failure(
                "Vector DeepONet 验证损失出现 NaN 或 Inf",
                "non_finite_validation_loss",
                epoch,
                model,
                history,
                best_state,
                seed,
            )
        scheduler.step(validation_loss)
        history.append(
            TrainingRecord(
                epoch=epoch,
                train_loss=float(np.mean(batch_losses)),
                validation_loss=validation_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        if best_validation - validation_loss > spec.training.min_delta:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= spec.training.patience:
            stop_reason = "early_stopping"
            break
        if (time.perf_counter() - started) / 60.0 >= spec.training.max_minutes:
            stop_reason = "time_budget"
            break

    if not best_state:
        raise RuntimeError("训练未产生有效的最佳检查点")
    elapsed_seconds = time.perf_counter() - started
    peak_cuda_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "cuda"
        else 0.0
    )
    return TrainingResult(
        seed=seed,
        state_dict=best_state,
        history=tuple(history),
        best_epoch=best_epoch,
        validation_loss=best_validation,
        stop_reason=stop_reason,
        device=str(device),
        elapsed_seconds=elapsed_seconds,
        peak_cuda_memory_mb=peak_cuda_memory_mb,
    )


def train_and_select(
    spec: ElasticityRunSpec,
    partitions: TrainingPartitions,
    normalization: FieldNormalization,
    device: torch.device,
) -> SelectedTraining:
    candidates = tuple(
        train_one_seed(spec, partitions, normalization, device, seed)
        for seed in spec.training.seeds
    )
    selected = min(
        candidates,
        key=lambda candidate: (candidate.validation_loss, candidate.seed),
    )
    return SelectedTraining(
        selected_seed=selected.seed,
        selected=selected,
        candidates=candidates,
    )


def predict_dataset(
    model: VectorDeepONet,
    dataset: FieldDataset,
    normalization: FieldNormalization,
    device: torch.device,
    query_batch_size: int,
) -> NDArray[np.float64]:
    prepared = _prepare_dataset(dataset, normalization)
    prediction = _predict_physical(model, prepared, device, query_batch_size)
    return prediction.astype(np.float64)


@dataclass(frozen=True)
class _PreparedDataset:
    features: NDArray[np.float32]
    coordinates: NDArray[np.float32]
    physical_parameters: NDArray[np.float32]
    physical_coordinates: NDArray[np.float32]
    targets: NDArray[np.float32]


def _prepare_dataset(
    dataset: FieldDataset,
    normalization: FieldNormalization,
) -> _PreparedDataset:
    physical_parameters = dataset.parameters.astype(np.float32)
    physical_coordinates = dataset.coordinates.astype(np.float32)
    features = normalization.normalize_features(
        elasticity_basis_features(dataset.parameters)
    ).astype(np.float32)
    coordinates = normalization.normalize_coordinates(dataset.coordinates).astype(
        np.float32
    )
    return _PreparedDataset(
        features=features,
        coordinates=coordinates,
        physical_parameters=physical_parameters,
        physical_coordinates=physical_coordinates,
        targets=dataset.fields.astype(np.float32),
    )


def _predict_physical(
    model: VectorDeepONet,
    dataset: _PreparedDataset,
    device: torch.device,
    query_batch_size: int,
) -> NDArray[np.float32]:
    if query_batch_size <= 0:
        raise ValueError("query_batch_size 必须为正数")
    model.eval()
    branch = torch.as_tensor(dataset.features, device=device)
    physical_branch = torch.as_tensor(dataset.physical_parameters, device=device)
    batches: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, dataset.coordinates.shape[0], query_batch_size):
            stop = start + query_batch_size
            trunk = torch.as_tensor(dataset.coordinates[start:stop], device=device)
            physical_trunk = torch.as_tensor(
                dataset.physical_coordinates[start:stop], device=device
            )
            prediction = apply_elasticity_constraints(
                model(branch, trunk), physical_branch, physical_trunk
            )
            batches.append(prediction.cpu().numpy())
    return np.concatenate(batches, axis=1)


def _validate_training_contract(
    partitions: TrainingPartitions,
    normalization: FieldNormalization,
) -> None:
    for role, dataset in (
        ("train", partitions.train),
        ("validation", partitions.validation),
    ):
        if dataset.parameters.shape[1] != 6:
            raise ValueError(f"{role} 参数维数必须为 6")
        if dataset.coordinates.shape[1] != 2 or dataset.fields.shape[2] != 2:
            raise ValueError(f"{role} 必须是二维坐标和二维向量场")
    if not np.array_equal(partitions.train.coordinates, partitions.validation.coordinates):
        raise ValueError("训练集和验证集必须共享观察坐标")
    if (
        normalization.feature_mean.shape != (3,)
        or normalization.coordinate_mean.shape != (2,)
        or normalization.target_rms.shape != (2,)
    ):
        raise ValueError("二维弹性归一化统计维数无效")


def _balanced_shape_loss(
    prediction: Tensor,
    target: Tensor,
    physical_parameters: Tensor,
    component_rms: Tensor,
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("平衡形状损失要求相同的三维预测和目标")
    if physical_parameters.shape != (prediction.shape[0], 6):
        raise ValueError("平衡形状损失的物理参数形状无效")
    if component_rms.shape != (prediction.shape[2],) or torch.any(component_rms <= 0.0):
        raise ValueError("平衡形状损失的分量尺度无效")
    scales = (
        physical_parameters[:, 2] / physical_parameters[:, 0]
    )[:, None, None]
    scaled_difference = (prediction - target) / scales / component_rms[None, None, :]
    scaled_target = target / scales / component_rms[None, None, :]
    difference_energy = torch.mean(scaled_difference.square(), dim=(1, 2))
    target_energy = torch.mean(scaled_target.square(), dim=(1, 2)).clamp_min(1e-12)
    return torch.mean(difference_energy / target_energy)


def _shape_component_rms(dataset: _PreparedDataset) -> NDArray[np.float32]:
    scales = dataset.physical_parameters[:, 2] / dataset.physical_parameters[:, 0]
    shapes = dataset.targets / scales[:, None, None]
    rms = np.sqrt(np.mean(np.square(shapes), axis=(0, 1)))
    return np.maximum(rms, 1e-12).astype(np.float32)




def _cpu_state_dict(model: VectorDeepONet) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _training_failure(
    message: str,
    reason: str,
    epoch: int,
    model: VectorDeepONet,
    history: list[TrainingRecord],
    best_state: dict[str, Tensor],
    seed: int,
) -> TrainingFailure:
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.empty_cache()
    state_dict = (
        {name: tensor.detach().cpu().clone() for name, tensor in best_state.items()}
        if best_state
        else _cpu_state_dict(model)
    )
    return TrainingFailure(
        message,
        reason=reason,
        failure_epoch=epoch,
        state_dict=state_dict,
        history=tuple(history),
        seed=seed,
    )
