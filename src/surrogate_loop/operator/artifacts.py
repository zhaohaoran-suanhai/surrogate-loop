from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch

from surrogate_loop.operator.config import OperatorRunSpec
from surrogate_loop.operator.heat1d.dataset import (
    HeatDataset,
    HeatDatasetSplit,
    NormalizationStats,
)
from surrogate_loop.operator.heat1d.evaluation import FieldMetrics
from surrogate_loop.operator.heat1d.pod_gpr import PodGprBaseline
from surrogate_loop.operator.heat1d.training import TrainingResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def create_operator_run_directory(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    name = f"heat-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = base / name
    run_dir.mkdir(exist_ok=False)
    return run_dir


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_operator_run(
    *,
    run_dir: Path,
    spec: OperatorRunSpec,
    request_text: str,
    dataset: HeatDataset,
    split: HeatDatasetSplit,
    normalization: NormalizationStats,
    baseline: PodGprBaseline,
    pod_metrics: FieldMetrics,
    training: TrainingResult,
    test_metrics: FieldMetrics,
    test_prediction: np.ndarray,
    status: str,
    runtime: Mapping[str, object],
) -> None:
    if status not in {"accepted", "rejected"}:
        raise ValueError("算子运行状态必须为 accepted 或 rejected")
    write_json_atomic(run_dir / "request.json", {"source": "codex", "text": request_text})
    write_json_atomic(run_dir / "spec.json", spec.model_dump(mode="json"))
    _save_dataset(run_dir / "dataset.npz", dataset)
    write_json_atomic(
        run_dir / "split.json",
        {
            "train_parameters": split.train.parameters.tolist(),
            "validation_parameters": split.validation.parameters.tolist(),
            "test_parameters": split.test.parameters.tolist(),
        },
    )
    write_json_atomic(run_dir / "normalization.json", _normalization_payload(normalization))
    write_json_atomic(
        run_dir / "solver_metrics.json",
        {
            "median_relative_l2": float(np.median(dataset.solver_relative_l2)),
            "p95_relative_l2": float(np.quantile(dataset.solver_relative_l2, 0.95)),
            "worst_relative_l2": float(np.max(dataset.solver_relative_l2)),
            "boundary_max_absolute_error": float(
                np.max(np.abs(dataset.fields[:, :, [0, -1]]))
            ),
        },
    )
    _joblib_dump_atomic(run_dir / "pod_gpr.joblib", baseline)
    write_json_atomic(run_dir / "pod_metrics.json", pod_metrics.to_dict())
    _torch_save_atomic(run_dir / "deeponet_state.pt", training.state_dict)
    write_json_atomic(
        run_dir / "network.json",
        {
            "branch_input_dim": 3,
            "trunk_input_dim": 2,
            "hidden_width": spec.model.hidden_width,
            "hidden_layers": spec.model.hidden_layers,
            "latent_dim": spec.model.latent_dim,
        },
    )
    write_json_atomic(
        run_dir / "training_history.json",
        {
            "records": [asdict(record) for record in training.history],
            "best_epoch": training.best_epoch,
            "stop_reason": training.stop_reason,
            "device": training.device,
            "elapsed_seconds": training.elapsed_seconds,
            "peak_cuda_memory_mb": training.peak_cuda_memory_mb,
        },
    )
    write_json_atomic(run_dir / "test_metrics.json", test_metrics.to_dict())
    _save_field_comparison(
        run_dir / "field_comparison.png",
        split.test.x,
        split.test.t,
        split.test.fields[0],
        np.asarray(test_prediction, dtype=np.float64)[0],
    )
    (run_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# 一维热传导 DeepONet 模型卡",
                "",
                f"- 状态：`{status}`",
                f"- 最佳 epoch：`{training.best_epoch}`",
                f"- 停止原因：`{training.stop_reason}`",
                f"- 测试中位相对 L2：`{test_metrics.median_relative_l2:.8g}`",
                f"- 测试 p95 相对 L2：`{test_metrics.p95_relative_l2:.8g}`",
                "- 参数域：`alpha ∈ [0.05,0.2]，A ∈ [0.8,1.2]，B ∈ [-0.3,0.3]`",
            ]
        ),
        encoding="utf-8",
    )
    hashed_files = (
        "spec.json",
        "normalization.json",
        "network.json",
        "deeponet_state.pt",
        "pod_gpr.joblib",
        "test_metrics.json",
    )
    write_json_atomic(
        run_dir / "manifest.json",
        {
            "problem": "heat_1d_operator_v1",
            "status": status,
            "runtime": dict(runtime),
            "sha256": {name: sha256_file(run_dir / name) for name in hashed_files},
        },
    )


def write_failed_run(run_dir: Path, spec_path: Path, error: Exception) -> None:
    write_json_atomic(run_dir / "status.json", {"status": "failed"})
    write_json_atomic(
        run_dir / "error.json",
        {
            "type": type(error).__name__,
            "message": str(error),
            "spec_path": str(spec_path),
        },
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalization_payload(stats: NormalizationStats) -> dict[str, object]:
    return {
        "parameter_mean": stats.parameter_mean.tolist(),
        "parameter_std": stats.parameter_std.tolist(),
        "coordinate_mean": stats.coordinate_mean.tolist(),
        "coordinate_std": stats.coordinate_std.tolist(),
        "target_mean": stats.target_mean,
        "target_std": stats.target_std,
    }


def _save_dataset(path: Path, dataset: HeatDataset) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            parameters=dataset.parameters,
            x=dataset.x,
            t=dataset.t,
            fields=dataset.fields,
            solver_relative_l2=dataset.solver_relative_l2,
        )
    os.replace(temporary, path)


def _joblib_dump_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def _torch_save_atomic(path: Path, state_dict: Mapping[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state_dict), temporary)
    os.replace(temporary, path)


def _save_field_comparison(
    path: Path,
    x: np.ndarray,
    t: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
) -> None:
    difference = prediction - reference
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    extent = (float(x[0]), float(x[-1]), float(t[0]), float(t[-1]))
    for axis, field, title in zip(
        axes,
        (reference, prediction, difference),
        ("Numerical reference", "DeepONet prediction", "Prediction error"),
        strict=True,
    ):
        image = axis.imshow(field, origin="lower", aspect="auto", extent=extent)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=150)
    plt.close(figure)
