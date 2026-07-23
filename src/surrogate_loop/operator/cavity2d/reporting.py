from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from surrogate_loop.operator.cavity2d.evaluation import (
    _centerline_fields,
    _physics_diagnostic_values,
    _regular_fields,
    _relative_l2,
    _vortex_centers,
)


def write_cavity_model_card(
    path: Path,
    *,
    status: str,
    mode: str,
    selected: dict[str, object],
) -> None:
    path.write_text(
        "\n".join(
            (
                "# 二维方腔 POD-RBF 模型卡",
                "",
                f"- 状态：`{status}`",
                f"- 运行模式：`{mode}`",
                "- 问题：固定单位方腔、恒定顶盖速度、稳态不可压缩层流",
                "- 适用 Reynolds 数：`[10,400]`",
                "- 输出：`u、v、p'`，压力按 cell 平均值归零",
                f"- POD 能量阈值：`{selected['energy_threshold']}`",
                f"- RBF kernel：`{selected['kernel']}`",
                f"- RBF smoothing：`{selected['smoothing']}`",
                "",
                "Smoke 最高只表示开发闭环完成；只有 accepted Full 才允许普通推理。",
                "",
            )
        ),
        encoding="utf-8",
    )


def write_cavity_evaluation_report(
    report_dir: Path,
    *,
    sample_ids: np.ndarray,
    reynolds: np.ndarray,
    coordinates: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    fluent_seconds_per_sample: float,
    surrogate_seconds_per_sample: float,
) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=False)
    velocity_errors = _relative_l2(reference[:, :, :2], prediction[:, :, :2])
    reference_pressure = reference[:, :, 2] - reference[:, :, 2].mean(
        axis=1,
        keepdims=True,
    )
    prediction_pressure = prediction[:, :, 2] - prediction[:, :, 2].mean(
        axis=1,
        keepdims=True,
    )
    pressure_errors = _relative_l2(
        reference_pressure[:, :, None],
        prediction_pressure[:, :, None],
    )
    reference_centers = _vortex_centers(coordinates, reference[:, :, :2])
    prediction_centers = _vortex_centers(coordinates, prediction[:, :, :2])
    vortex_errors = np.linalg.norm(prediction_centers - reference_centers, axis=1)
    reference_horizontal, reference_vertical = _centerline_fields(
        coordinates,
        reference,
    )
    prediction_horizontal, prediction_vertical = _centerline_fields(
        coordinates,
        prediction,
    )
    horizontal_errors = _relative_l2(
        reference_horizontal,
        prediction_horizontal,
    )
    vertical_errors = _relative_l2(
        reference_vertical,
        prediction_vertical,
    )
    divergence, momentum = _physics_diagnostic_values(
        coordinates,
        prediction,
        reynolds,
    )
    if momentum is None:
        momentum = np.full(reynolds.shape, np.nan)
    worst_velocity = int(np.argmax(velocity_errors))
    worst_pressure = int(np.argmax(pressure_errors))
    worst_divergence = int(np.nanargmax(divergence))
    worst_momentum = int(np.nanargmax(momentum))
    rows = [
        {
            "sample_id": str(sample_ids[index]),
            "reynolds": float(reynolds[index]),
            "velocity_relative_l2": float(velocity_errors[index]),
            "pressure_relative_l2": float(pressure_errors[index]),
            "horizontal_centerline_velocity_relative_l2": float(
                horizontal_errors[index]
            ),
            "vertical_centerline_velocity_relative_l2": float(
                vertical_errors[index]
            ),
            "reference_vortex_center": reference_centers[index].tolist(),
            "predicted_vortex_center": prediction_centers[index].tolist(),
            "vortex_center_error": float(vortex_errors[index]),
            "divergence_rms": float(divergence[index]),
            "momentum_rms": float(momentum[index]),
        }
        for index in range(reynolds.size)
    ]
    details = {
        "sample_count": int(reynolds.size),
        "fluent_seconds_per_sample": fluent_seconds_per_sample,
        "surrogate_seconds_per_sample": surrogate_seconds_per_sample,
        "cpu_speedup": (
            fluent_seconds_per_sample / surrogate_seconds_per_sample
            if fluent_seconds_per_sample > 0.0
            and surrogate_seconds_per_sample > 0.0
            else 0.0
        ),
        "worst_cases": {
            "velocity": str(sample_ids[worst_velocity]),
            "pressure": str(sample_ids[worst_pressure]),
            "divergence": str(sample_ids[worst_divergence]),
            "momentum": str(sample_ids[worst_momentum]),
        },
        "samples": rows,
        "diagnostic_note": (
            "散度和动量残差是在统一插值观测网格上计算的代理场诊断，"
            "不是 Fluent 原生有限体积离散残差。"
        ),
    }
    details_path = report_dir / "evaluation_details.json"
    details_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    x_values, y_values, regular_reference = _regular_fields(
        coordinates,
        reference[[worst_velocity]],
    )
    _, _, regular_prediction = _regular_fields(
        coordinates,
        prediction[[worst_velocity]],
    )
    arrays_path = report_dir / "evaluation_arrays.npz"
    np.savez_compressed(
        arrays_path,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        reynolds=np.asarray(reynolds, dtype=np.float64),
        horizontal_axis=x_values,
        vertical_axis=y_values,
        reference_horizontal=reference_horizontal,
        prediction_horizontal=prediction_horizontal,
        reference_vertical=reference_vertical,
        prediction_vertical=prediction_vertical,
        reference_vortex_centers=reference_centers,
        prediction_vortex_centers=prediction_centers,
        velocity_relative_l2=velocity_errors,
        pressure_relative_l2=pressure_errors,
        divergence_rms=divergence,
        momentum_rms=momentum,
        worst_sample_id=np.asarray(str(sample_ids[worst_velocity])),
        worst_reference_regular=regular_reference[0],
        worst_prediction_regular=regular_prediction[0],
    )

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    reference_speed = np.linalg.norm(regular_reference[0, :, :, :2], axis=2)
    prediction_speed = np.linalg.norm(regular_prediction[0, :, :, :2], axis=2)
    panels = (
        (reference_speed, "Reference speed"),
        (prediction_speed, "Surrogate speed"),
        (np.abs(prediction_speed - reference_speed), "Absolute speed error"),
        (regular_reference[0, :, :, 2], "Reference mean-free pressure"),
        (regular_prediction[0, :, :, 2], "Surrogate mean-free pressure"),
        (
            np.abs(
                regular_prediction[0, :, :, 2]
                - regular_reference[0, :, :, 2]
            ),
            "Absolute pressure error",
        ),
    )
    for axis, (values, title) in zip(axes.ravel(), panels, strict=True):
        image = axis.pcolormesh(x_values, y_values, values, shading="auto")
        axis.set_title(title)
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis)
    field_figure = report_dir / "field_comparison.png"
    figure.savefig(field_figure, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, values, center, title in (
        (
            axes[0],
            regular_reference[0],
            reference_centers[worst_velocity],
            "Reference streamlines",
        ),
        (
            axes[1],
            regular_prediction[0],
            prediction_centers[worst_velocity],
            "Surrogate streamlines",
        ),
    ):
        axis.streamplot(
            x_values,
            y_values,
            values[:, :, 0],
            values[:, :, 1],
            density=1.1,
        )
        axis.scatter(center[0], center[1], marker="x", color="red")
        axis.set_title(title)
        axis.set_aspect("equal")
    streamline_figure = report_dir / "streamlines.png"
    figure.savefig(streamline_figure, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sample_index = worst_velocity
    axes[0].plot(
        x_values,
        reference_horizontal[sample_index, :, 0],
        label="Reference u",
    )
    axes[0].plot(
        x_values,
        prediction_horizontal[sample_index, :, 0],
        "--",
        label="Surrogate u",
    )
    axes[0].set_title("Horizontal centerline")
    axes[1].plot(
        y_values,
        reference_vertical[sample_index, :, 1],
        label="Reference v",
    )
    axes[1].plot(
        y_values,
        prediction_vertical[sample_index, :, 1],
        "--",
        label="Surrogate v",
    )
    axes[1].set_title("Vertical centerline")
    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.3)
    centerline_figure = report_dir / "centerlines.png"
    figure.savefig(centerline_figure, dpi=160)
    plt.close(figure)

    report_path = report_dir / "README.md"
    report_path.write_text(
        "\n".join(
            (
                "# 二维方腔代理模型评价报告",
                "",
                f"- 样本数：{reynolds.size}",
                f"- 最差速度工况：`{sample_ids[worst_velocity]}`",
                f"- 最差压力工况：`{sample_ids[worst_pressure]}`",
                f"- 单样本 Fluent 平均耗时：{fluent_seconds_per_sample:.6g} s",
                f"- 单样本代理平均耗时：{surrogate_seconds_per_sample:.6g} s",
                "",
                "详细逐样本指标见 `evaluation_details.json`，中心线、主涡和"
                "物理诊断数组见 `evaluation_arrays.npz`。",
                "",
                "散度和动量残差是在统一插值观测网格上计算的代理场诊断，"
                "不是 Fluent 原生有限体积离散残差。",
                "",
            )
        ),
        encoding="utf-8",
    )
    return [
        details_path,
        arrays_path,
        field_figure,
        streamline_figure,
        centerline_figure,
        report_path,
    ]


__all__ = ["write_cavity_evaluation_report", "write_cavity_model_card"]
