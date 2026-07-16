from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np

from surrogate_loop.config import RunSpec
from surrogate_loop.data import CaseDataset
from surrogate_loop.evaluation import SelectionResult
from surrogate_loop.split import DatasetSplit

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def create_run_directory(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = base / name
    run_dir.mkdir(exist_ok=False)
    return run_dir


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_successful_run(
    run_dir: Path,
    spec: RunSpec,
    request_text: str,
    dataset: CaseDataset,
    split: DatasetSplit,
    selection: SelectionResult,
) -> None:
    write_json_atomic(run_dir / "request.json", {"source": "codex", "text": request_text})
    write_json_atomic(run_dir / "spec.json", spec.model_dump(mode="json"))
    np.savez_compressed(run_dir / "dataset.npz", gamma=dataset.gamma, target=dataset.target)
    write_json_atomic(
        run_dir / "split.json",
        {
            "train_gamma": split.train_x.ravel().tolist(),
            "validation_gamma": split.validation_x.ravel().tolist(),
            "test_gamma": split.test_x.ravel().tolist(),
        },
    )
    write_json_atomic(
        run_dir / "validation_metrics.json",
        {name: metrics.to_dict() for name, metrics in selection.validation_metrics.items()},
    )
    write_json_atomic(run_dir / "test_metrics.json", selection.test_metrics.to_dict())
    joblib.dump(selection.selected_model, run_dir / "model.joblib")

    prediction = selection.selected_model.predict(split.test_x)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(split.test_y, prediction)
    low = min(float(np.min(split.test_y)), float(np.min(prediction)))
    high = max(float(np.max(split.test_y)), float(np.max(prediction)))
    axis.plot([low, high], [low, high], "k--")
    axis.set_xlabel("Numerical reference u(1)")
    axis.set_ylabel("Surrogate prediction u(1)")
    axis.set_title("Held-out test predictions")
    figure.tight_layout()
    figure.savefig(run_dir / "prediction.png", dpi=150)
    plt.close(figure)

    status = "accepted" if selection.accepted else "rejected"
    (run_dir / "model_card.md").write_text(
        "\n".join(
            [
                "# 模型卡",
                "",
                f"- 状态：`{status}`",
                f"- 最佳模型：`{selection.selected_name}`",
                f"- 测试 NRMSE：`{selection.test_metrics.nrmse:.8g}`",
                f"- 最大绝对误差：`{selection.test_metrics.max_absolute_error:.8g}`",
                "- 有效参数域：`gamma ∈ [-1, 1]`",
            ]
        ),
        encoding="utf-8",
    )

    hashed_files = ("spec.json", "model.joblib", "test_metrics.json")
    manifest: dict[str, Any] = {
        "status": status,
        "selected_model": selection.selected_name,
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "scikit-learn": version("scikit-learn"),
        },
        "sha256": {name: _sha256(run_dir / name) for name in hashed_files},
    }
    write_json_atomic(run_dir / "manifest.json", manifest)


def write_failed_run(run_dir: Path, spec_path: Path, error: Exception) -> None:
    write_json_atomic(run_dir / "status.json", {"status": "failed"})
    write_json_atomic(
        run_dir / "error.json",
        {"type": type(error).__name__, "message": str(error), "spec_path": str(spec_path)},
    )


def load_verified_model(run_dir: Path) -> tuple[RunSpec, Any, dict[str, object]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if _sha256(run_dir / name) != expected:
            raise RuntimeError(f"运行产物哈希校验失败：{name}")
    spec = RunSpec.model_validate_json((run_dir / "spec.json").read_text(encoding="utf-8"))
    model = joblib.load(run_dir / "model.joblib")
    return spec, model, manifest
