from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

_BASE_KEYS = frozenset({"sample_ids", "parameters", "coordinates", "fields"})
_DIAGNOSTIC_PREFIX = "diag__"
_DIAGNOSTIC_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class FieldDataset:
    sample_ids: NDArray[np.str_]
    parameters: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    fields: NDArray[np.float64]
    diagnostics: Mapping[str, NDArray[np.float64]]

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids, dtype=np.str_)
        parameters = np.asarray(self.parameters, dtype=np.float64)
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        fields = np.asarray(self.fields, dtype=np.float64)
        diagnostics = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in self.diagnostics.items()
        }
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))
        if parameters.ndim != 2 or parameters.shape[0] == 0 or parameters.shape[1] == 0:
            raise ValueError("参数形状必须为非空二维数组")
        sample_count = parameters.shape[0]
        if sample_ids.shape != (sample_count,) or np.any(sample_ids == ""):
            raise ValueError("样本 ID 形状必须为 (n_samples,) 且不能为空")
        if len(set(sample_ids.tolist())) != sample_count:
            raise ValueError("样本 ID 必须互不重复")
        if coordinates.ndim != 2 or coordinates.shape[0] == 0 or coordinates.shape[1] == 0:
            raise ValueError("坐标形状必须为非空二维数组")
        expected_fields = (sample_count, coordinates.shape[0])
        if fields.ndim != 3 or fields.shape[:2] != expected_fields or fields.shape[2] == 0:
            raise ValueError(
                "字段形状必须为 (n_samples, n_points, n_components)"
            )
        arrays = (parameters, coordinates, fields, *diagnostics.values())
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("场数据及诊断数组必须全部有限")
        for name, values in diagnostics.items():
            if _DIAGNOSTIC_NAME.fullmatch(name) is None:
                raise ValueError(f"诊断名称无效：{name}")
            if values.ndim == 0 or values.shape[0] != sample_count:
                raise ValueError(f"诊断数组 {name} 的第一维必须为 n_samples")

    def subset(self, indices: NDArray[np.int64]) -> FieldDataset:
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("样本索引必须是一维数组")
        return FieldDataset(
            sample_ids=self.sample_ids[selected].copy(),
            parameters=self.parameters[selected].copy(),
            coordinates=self.coordinates.copy(),
            fields=self.fields[selected].copy(),
            diagnostics={
                name: values[selected].copy() for name, values in self.diagnostics.items()
            },
        )


@dataclass(frozen=True)
class FieldNormalization:
    feature_mean: NDArray[np.float64]
    feature_std: NDArray[np.float64]
    coordinate_mean: NDArray[np.float64]
    coordinate_std: NDArray[np.float64]
    target_rms: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in (
            "feature_mean",
            "feature_std",
            "coordinate_mean",
            "coordinate_std",
            "target_rms",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            object.__setattr__(self, name, value)
            if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
                raise ValueError(f"归一化统计 {name} 必须是有限非空一维数组")
        if self.feature_mean.shape != self.feature_std.shape:
            raise ValueError("特征均值和标准差形状必须一致")
        if self.coordinate_mean.shape != self.coordinate_std.shape:
            raise ValueError("坐标均值和标准差形状必须一致")
        if np.any(self.feature_std <= 0.0) or np.any(self.coordinate_std <= 0.0):
            raise ValueError("归一化标准差必须为正数")
        if np.any(self.target_rms <= 0.0):
            raise ValueError("目标 RMS 必须为正数")

    @classmethod
    def fit(
        cls,
        features: NDArray[np.float64],
        coordinates: NDArray[np.float64],
        fields: NDArray[np.float64],
    ) -> FieldNormalization:
        features = np.asarray(features, dtype=np.float64)
        coordinates = np.asarray(coordinates, dtype=np.float64)
        fields = np.asarray(fields, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("训练特征必须是非空二维数组")
        if coordinates.ndim != 2 or coordinates.shape[0] == 0:
            raise ValueError("训练坐标必须是非空二维数组")
        if fields.ndim != 3 or fields.shape[:2] != (
            features.shape[0],
            coordinates.shape[0],
        ):
            raise ValueError("训练字段必须匹配特征样本数和坐标点数")
        if not (
            np.isfinite(features).all()
            and np.isfinite(coordinates).all()
            and np.isfinite(fields).all()
        ):
            raise ValueError("训练归一化输入必须全部有限")
        return cls(
            feature_mean=features.mean(axis=0),
            feature_std=_safe_std(features, axis=0),
            coordinate_mean=coordinates.mean(axis=0),
            coordinate_std=_safe_std(coordinates, axis=0),
            target_rms=np.maximum(
                np.sqrt(np.mean(np.square(fields), axis=(0, 1))), 1e-12
            ),
        )

    def normalize_features(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_std

    def normalize_coordinates(
        self, coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        values = np.asarray(coordinates, dtype=np.float64) - self.coordinate_mean
        return values / self.coordinate_std

    def scale_targets(self, fields: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(fields, dtype=np.float64) / self.target_rms

    def unscale_targets(self, fields: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(fields, dtype=np.float64) * self.target_rms


def save_field_dataset(path: Path, dataset: FieldDataset) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    arrays: dict[str, NDArray[np.generic]] = {
        "sample_ids": dataset.sample_ids,
        "parameters": dataset.parameters,
        "coordinates": dataset.coordinates,
        "fields": dataset.fields,
    }
    arrays.update(
        {
            f"{_DIAGNOSTIC_PREFIX}{name}": values
            for name, values in sorted(dataset.diagnostics.items())
        }
    )
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def load_field_dataset(path: Path, expected_sha256: str | None = None) -> FieldDataset:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError("场数据集 SHA-256 校验失败")
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if not _BASE_KEYS <= keys:
                raise RuntimeError("场数据集缺少必需数组")
            extra = keys - _BASE_KEYS
            if any(
                not key.startswith(_DIAGNOSTIC_PREFIX)
                or not key.removeprefix(_DIAGNOSTIC_PREFIX)
                for key in extra
            ):
                raise RuntimeError("场数据集包含未知数组")
            diagnostics = {
                key.removeprefix(_DIAGNOSTIC_PREFIX): np.asarray(archive[key]).copy()
                for key in sorted(extra)
            }
            return FieldDataset(
                sample_ids=np.asarray(archive["sample_ids"]).copy(),
                parameters=np.asarray(archive["parameters"]).copy(),
                coordinates=np.asarray(archive["coordinates"]).copy(),
                fields=np.asarray(archive["fields"]).copy(),
                diagnostics=diagnostics,
            )
    except RuntimeError:
        raise
    except (OSError, ValueError) as error:
        raise RuntimeError("无法读取安全的 NPZ 场数据集") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RuntimeError(f"无法读取场数据集：{path}") from error
    return digest.hexdigest()


def _safe_std(
    array: NDArray[np.float64], axis: int | tuple[int, ...]
) -> NDArray[np.float64]:
    value = np.asarray(array, dtype=np.float64).std(axis=axis)
    return np.where(value < 1e-12, 1.0, value)
