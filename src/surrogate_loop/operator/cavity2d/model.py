from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

KERNELS = ("cubic", "thin_plate_spline", "multiquadric")


@dataclass(frozen=True)
class PodBranch:
    field_mean: np.ndarray
    components: np.ndarray
    training_features: np.ndarray
    training_coefficients: np.ndarray
    kernel: str
    smoothing: float


@dataclass(frozen=True)
class CavityPodRbfModel:
    log_re_mean: float
    log_re_std: float
    velocity: PodBranch
    pressure: PodBranch
    n_cells: int
    energy_threshold: float

    def predict(self, reynolds: np.ndarray) -> np.ndarray:
        values = np.asarray(reynolds, dtype=np.float64)
        if (
            values.ndim != 1
            or values.size == 0
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
        ):
            raise ValueError("Reynolds inputs must be positive finite 1D values")
        features = (
            (np.log10(values) - self.log_re_mean) / self.log_re_std
        )[:, None]
        velocity = _predict_branch(self.velocity, features).reshape(
            values.size,
            self.n_cells,
            2,
        )
        pressure = _predict_branch(self.pressure, features).reshape(
            values.size,
            self.n_cells,
            1,
        )
        pressure -= pressure.mean(axis=1, keepdims=True)
        return np.concatenate((velocity, pressure), axis=2)


def _fit_branch(
    features: np.ndarray,
    values: np.ndarray,
    *,
    energy_threshold: float,
    kernel: str,
    smoothing: float,
) -> PodBranch:
    field_mean = values.mean(axis=0)
    centered = values - field_mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    maximum = min(64, values.shape[0] - 1, right_vectors.shape[0])
    if maximum < 1:
        raise ValueError("POD-RBF requires at least two training samples")
    energy = np.square(singular_values[:maximum])
    if energy.sum() <= 0.0:
        selected = 1
    else:
        cumulative = np.cumsum(energy) / energy.sum()
        selected = int(np.searchsorted(cumulative, energy_threshold) + 1)
    selected = min(selected, maximum)
    components = right_vectors[:selected].copy()
    coefficients = centered @ components.T
    return PodBranch(
        field_mean=field_mean,
        components=components,
        training_features=features.copy(),
        training_coefficients=coefficients,
        kernel=kernel,
        smoothing=float(smoothing),
    )


def _rbf(branch: PodBranch) -> RBFInterpolator:
    kwargs: dict[str, object] = {
        "kernel": branch.kernel,
        "smoothing": branch.smoothing,
    }
    if branch.kernel == "multiquadric":
        kwargs["epsilon"] = 1.0
    return RBFInterpolator(
        branch.training_features,
        branch.training_coefficients,
        **kwargs,
    )


def _predict_branch(branch: PodBranch, features: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(_rbf(branch)(features), dtype=np.float64)
    return coefficients @ branch.components + branch.field_mean


def fit_candidate(
    reynolds: np.ndarray,
    fields: np.ndarray,
    *,
    energy_threshold: float,
    kernel: str,
    smoothing: float,
) -> CavityPodRbfModel:
    reynolds = np.asarray(reynolds, dtype=np.float64)
    fields = np.asarray(fields, dtype=np.float64)
    if (
        reynolds.ndim != 1
        or reynolds.size < 2
        or fields.ndim != 3
        or fields.shape[0] != reynolds.size
        or fields.shape[2] != 3
    ):
        raise ValueError("training Reynolds and fields have invalid shapes")
    if (
        not np.isfinite(reynolds).all()
        or np.any(reynolds <= 0.0)
        or len(set(reynolds.tolist())) != reynolds.size
        or not np.isfinite(fields).all()
    ):
        raise ValueError("training Reynolds and fields must be finite and unique")
    if energy_threshold not in {0.999, 0.9999}:
        raise ValueError("POD energy threshold is outside the fixed candidate set")
    if kernel not in KERNELS:
        raise ValueError("RBF kernel is outside the fixed candidate set")
    if smoothing not in {0.0, 1e-10, 1e-8}:
        raise ValueError("RBF smoothing is outside the fixed candidate set")

    log_re = np.log10(reynolds)
    log_re_mean = float(log_re.mean())
    log_re_std = float(log_re.std())
    if log_re_std <= 0.0:
        raise ValueError("training Reynolds must have nonzero log-space variance")
    features = ((log_re - log_re_mean) / log_re_std)[:, None]
    velocity_values = fields[:, :, :2].reshape(reynolds.size, -1)
    pressure_values = fields[:, :, 2].reshape(reynolds.size, -1)
    return CavityPodRbfModel(
        log_re_mean=log_re_mean,
        log_re_std=log_re_std,
        velocity=_fit_branch(
            features,
            velocity_values,
            energy_threshold=energy_threshold,
            kernel=kernel,
            smoothing=smoothing,
        ),
        pressure=_fit_branch(
            features,
            pressure_values,
            energy_threshold=energy_threshold,
            kernel=kernel,
            smoothing=smoothing,
        ),
        n_cells=fields.shape[1],
        energy_threshold=energy_threshold,
    )


def save_cavity_model(
    path: Path,
    model: CavityPodRbfModel,
    *,
    problem_id: str,
    mesh_sha256: str,
    coordinates_sha256: str,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if (path / "model.json").exists() or (path / "model_arrays.npz").exists():
        raise FileExistsError("cavity model artifacts already exist")
    metadata = {
        "schema_version": 1,
        "model_type": "cavity2d_pod_rbf",
        "problem_id": problem_id,
        "mesh_sha256": mesh_sha256.lower(),
        "coordinates_sha256": coordinates_sha256.lower(),
        "log_re_mean": model.log_re_mean,
        "log_re_std": model.log_re_std,
        "n_cells": model.n_cells,
        "energy_threshold": model.energy_threshold,
        "velocity": {
            "kernel": model.velocity.kernel,
            "smoothing": model.velocity.smoothing,
        },
        "pressure": {
            "kernel": model.pressure.kernel,
            "smoothing": model.pressure.smoothing,
        },
    }
    (path / "model.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary = path / "model_arrays.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            velocity_field_mean=model.velocity.field_mean,
            velocity_components=model.velocity.components,
            velocity_training_features=model.velocity.training_features,
            velocity_training_coefficients=model.velocity.training_coefficients,
            pressure_field_mean=model.pressure.field_mean,
            pressure_components=model.pressure.components,
            pressure_training_features=model.pressure.training_features,
            pressure_training_coefficients=model.pressure.training_coefficients,
        )
    os.replace(temporary, path / "model_arrays.npz")


def load_cavity_model(
    path: Path,
    *,
    problem_id: str,
    mesh_sha256: str,
    coordinates_sha256: str,
) -> CavityPodRbfModel:
    metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
    metadata_keys = {
        "schema_version",
        "model_type",
        "problem_id",
        "mesh_sha256",
        "coordinates_sha256",
        "log_re_mean",
        "log_re_std",
        "n_cells",
        "energy_threshold",
        "velocity",
        "pressure",
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != metadata_keys
        or metadata.get("schema_version") != 1
        or metadata.get("model_type") != "cavity2d_pod_rbf"
    ):
        raise RuntimeError("cavity model metadata is invalid")
    if (
        metadata.get("problem_id") != problem_id
        or str(metadata.get("mesh_sha256", "")).lower() != mesh_sha256.lower()
        or str(metadata.get("coordinates_sha256", "")).lower()
        != coordinates_sha256.lower()
    ):
        raise RuntimeError("cavity model identity mismatch")
    try:
        log_re_mean = float(metadata["log_re_mean"])
        log_re_std = float(metadata["log_re_std"])
        energy_threshold = float(metadata["energy_threshold"])
        n_cells = int(metadata["n_cells"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cavity model scalar metadata is invalid") from exc
    if (
        not np.isfinite([log_re_mean, log_re_std, energy_threshold]).all()
        or log_re_std <= 0.0
        or energy_threshold not in {0.999, 0.9999}
    ):
        raise RuntimeError("cavity model scalar metadata is invalid")
    required = {
        "velocity_field_mean",
        "velocity_components",
        "velocity_training_features",
        "velocity_training_coefficients",
        "pressure_field_mean",
        "pressure_components",
        "pressure_training_features",
        "pressure_training_coefficients",
    }
    try:
        with np.load(path / "model_arrays.npz", allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise RuntimeError("cavity model array schema mismatch")
            arrays = {
                name: np.asarray(archive[name], dtype=np.float64).copy()
                for name in required
            }
    except RuntimeError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot load safe cavity model arrays") from exc
    if (
        n_cells <= 0
        or not all(np.isfinite(array).all() for array in arrays.values())
        or arrays["velocity_field_mean"].shape != (2 * n_cells,)
        or arrays["pressure_field_mean"].shape != (n_cells,)
        or arrays["velocity_components"].ndim != 2
        or arrays["velocity_components"].shape[1] != 2 * n_cells
        or arrays["pressure_components"].ndim != 2
        or arrays["pressure_components"].shape[1] != n_cells
    ):
        raise RuntimeError("cavity model array shapes are invalid")

    def branch(name: str) -> PodBranch:
        branch_metadata = metadata[name]
        if (
            not isinstance(branch_metadata, dict)
            or set(branch_metadata) != {"kernel", "smoothing"}
            or branch_metadata.get("kernel") not in KERNELS
            or branch_metadata.get("smoothing") not in {0.0, 1e-10, 1e-8}
        ):
            raise RuntimeError("cavity model branch metadata is invalid")
        features = arrays[f"{name}_training_features"]
        coefficients = arrays[f"{name}_training_coefficients"]
        components = arrays[f"{name}_components"]
        if (
            features.ndim != 2
            or features.shape[1] != 1
            or coefficients.shape != (features.shape[0], components.shape[0])
            or features.shape[0] < 2
        ):
            raise RuntimeError("cavity model branch arrays are invalid")
        return PodBranch(
            field_mean=arrays[f"{name}_field_mean"],
            components=components,
            training_features=features,
            training_coefficients=coefficients,
            kernel=str(branch_metadata["kernel"]),
            smoothing=float(branch_metadata["smoothing"]),
        )

    return CavityPodRbfModel(
        log_re_mean=log_re_mean,
        log_re_std=log_re_std,
        velocity=branch("velocity"),
        pressure=branch("pressure"),
        n_cells=n_cells,
        energy_threshold=energy_threshold,
    )


__all__ = [
    "CavityPodRbfModel",
    "PodBranch",
    "fit_candidate",
    "load_cavity_model",
    "save_cavity_model",
]
