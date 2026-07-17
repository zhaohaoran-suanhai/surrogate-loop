from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from surrogate_loop.operator.field_data import FieldDataset, sha256_file


def write_smoke_diagnostics(
    run_dir: Path,
    dataset: FieldDataset,
    prediction: np.ndarray,
    solver_manifest_path: Path,
) -> dict[str, str]:
    predicted = np.asarray(prediction, dtype=np.float64)
    if predicted.shape != dataset.fields.shape or not np.isfinite(predicted).all():
        raise ValueError("Smoke 诊断预测场形状或数值无效")
    errors = _relative_errors(dataset.fields, predicted)
    representative = int(np.argmax(errors))
    sample_id = str(dataset.sample_ids[representative])
    stress = _stress_summary_for_sample(solver_manifest_path, sample_id)
    diagnostics = run_dir.resolve() / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    displacement_path = diagnostics / "displacement_comparison.png"
    stress_path = diagnostics / "fenicsx_stress_summary.png"
    _plot_displacement(
        displacement_path,
        dataset.coordinates,
        dataset.fields[representative],
        predicted[representative],
        sample_id,
        float(errors[representative]),
    )
    _plot_stress_summary(stress_path, stress, sample_id)
    return {
        str(displacement_path.relative_to(run_dir.resolve())).replace("\\", "/"):
        sha256_file(displacement_path),
        str(stress_path.relative_to(run_dir.resolve())).replace("\\", "/"):
        sha256_file(stress_path),
    }


def _relative_errors(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    differences = np.linalg.norm((prediction - reference).reshape(reference.shape[0], -1), axis=1)
    scales = np.linalg.norm(reference.reshape(reference.shape[0], -1), axis=1)
    return differences / np.maximum(scales, 1e-30)


def _stress_summary_for_sample(path: Path, sample_id: str) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["samples"]
        record = next(item for item in records if item.get("sample_id") == sample_id)
        raw = record["stress_summary"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, StopIteration) as error:
        raise RuntimeError("无法读取代表性样本的 FEniCSx 应力诊断") from error
    required = {
        f"{name}_{statistic}"
        for name in ("stress_xx", "stress_yy", "stress_xy", "von_mises")
        for statistic in ("min", "max", "p95")
    }
    if not isinstance(raw, dict) or not required <= set(raw):
        raise RuntimeError("代表性样本的 FEniCSx 应力诊断字段不完整")
    result = {name: float(raw[name]) for name in required}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("代表性样本的 FEniCSx 应力诊断包含非有限值")
    return result


def _plot_displacement(
    path: Path,
    coordinates: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    sample_id: str,
    relative_error: float,
) -> None:
    figure = Figure(figsize=(12.0, 6.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    errors = np.abs(prediction - reference)
    columns: tuple[tuple[str, np.ndarray], ...] = (
        ("FEniCSx reference", reference),
        ("DeepONet prediction", prediction),
        ("absolute error", errors),
    )
    for component, label in enumerate(("u_x", "u_y")):
        shared = max(
            float(np.max(np.abs(reference[:, component]))),
            float(np.max(np.abs(prediction[:, component]))),
            1e-30,
        )
        for column, (title, values) in enumerate(columns):
            axis = figure.add_subplot(2, 3, component * 3 + column + 1)
            if column < 2:
                colors = values[:, component]
                color_limit = shared
                color_map = "coolwarm"
                minimum, maximum = -color_limit, color_limit
            else:
                colors = values[:, component]
                color_map = "magma"
                minimum, maximum = 0.0, max(float(np.max(colors)), 1e-30)
            artist = axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                c=colors,
                cmap=color_map,
                vmin=minimum,
                vmax=maximum,
                marker="s",
                s=38.0,
                linewidths=0.0,
            )
            axis.set_title(f"{label}: {title}")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_aspect("equal")
            figure.colorbar(artist, ax=axis, shrink=0.8)
    figure.suptitle(f"sample={sample_id} | relative L2={relative_error:.4%}")
    _save_png_atomic(figure, path)


def _plot_stress_summary(path: Path, stress: dict[str, float], sample_id: str) -> None:
    labels = ("sigma_xx", "sigma_yy", "tau_xy", "von Mises")
    keys = ("stress_xx", "stress_yy", "stress_xy", "von_mises")
    minima = np.array([stress[f"{key}_min"] for key in keys])
    maxima = np.array([stress[f"{key}_max"] for key in keys])
    p95 = np.array([stress[f"{key}_p95"] for key in keys])
    centers = (minima + maxima) / 2.0
    figure = Figure(figsize=(8.0, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    positions = np.arange(len(labels))
    axis.errorbar(
        positions,
        centers,
        yerr=np.vstack((centers - minima, maxima - centers)),
        fmt="o",
        capsize=5,
        label="FEniCSx min/max",
    )
    axis.scatter(positions, p95, marker="D", label="FEniCSx p95")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("stress diagnostic value")
    axis.set_title(f"FEniCSx stress summary (not neural output)\nsample={sample_id}")
    axis.legend()
    _save_png_atomic(figure, path)


def _save_png_atomic(figure: Figure, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        figure.savefig(temporary, format="png", dpi=150)
        os.replace(temporary, path)
    finally:
        figure.clear()
        if temporary.exists():
            temporary.unlink()
