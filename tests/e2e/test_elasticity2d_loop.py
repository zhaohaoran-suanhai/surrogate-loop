from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import surrogate_loop.operator.elasticity2d.artifacts as artifacts_module
import surrogate_loop.operator.elasticity2d.pipeline as pipeline_module
from surrogate_loop.cli import main
from surrogate_loop.operator.elasticity2d.dataset import DatasetFiles
from surrogate_loop.operator.elasticity2d.deeponet import build_elasticity_deeponet
from surrogate_loop.operator.elasticity2d.training import (
    SelectedTraining,
    TrainingRecord,
    TrainingResult,
)
from surrogate_loop.operator.field_data import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tests/fixtures/elasticity_operator_tiny.json"


def test_fake_solver_exercises_validate_run_report_predict_without_claiming_fenicsx(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "generate_or_reuse_dataset",
        lambda spec, plan, run_dir, repo_root: _fake_solver_dataset(run_dir, plan),
    )
    monkeypatch.setattr(pipeline_module, "train_and_select", _fake_training)
    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr(artifacts_module, "_benchmark_fenicsx", lambda *args: 0.5)
    monkeypatch.setattr(artifacts_module, "elasticity_is_acceptable", lambda *args: True)
    runs_dir = tmp_path / "runs"

    assert main(["elasticity2d", "validate", "--config", str(CONFIG)]) == 0
    assert (
        main(
            [
                "elasticity2d",
                "run",
                "--config",
                str(CONFIG),
                "--runs-dir",
                str(runs_dir),
                "--request",
                "测试内固定解析场，仅验证编排",
            ]
        )
        == 0
    )
    run_dir = next(runs_dir.iterdir())
    assert main(["elasticity2d", "report", "--run-dir", str(run_dir)]) == 0
    point_arguments = [
        "elasticity2d",
        "predict",
        "--run-dir",
        str(run_dir),
        "--e",
        "2",
        "--nu",
        ".3",
        "--p",
        ".004",
        "--theta",
        "0",
        "--y0",
        ".5",
        "--w",
        ".1",
        "--x",
        "2",
        "--y",
        ".5",
    ]
    assert main(point_arguments) == 0
    assert '"status": "accepted"' in capsys.readouterr().out

    invalid_parameters = point_arguments.copy()
    invalid_parameters[5] = "6"
    assert main(invalid_parameters) == 2
    protected = run_dir / "spec.json"
    before = sha256_file(protected)
    field_arguments = point_arguments[:-4] + [
        "--nx",
        "3",
        "--ny",
        "3",
        "--output",
        str(protected),
    ]
    assert main(field_arguments) == 2
    assert sha256_file(protected) == before

    assert (
        main(
            [
                "validate",
                "--config",
                str(ROOT / "examples/forced_reaction_scalar/smoke.json"),
            ]
        )
        == 0
    )


def _fake_training(spec, partitions, normalization, device) -> SelectedTraining:
    model = build_elasticity_deeponet(spec.model)
    state_dict = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    record = TrainingRecord(0, 0.2, 0.3, 1e-3)
    candidates = tuple(
        TrainingResult(
            seed=seed,
            state_dict=state_dict,
            history=(record,),
            best_epoch=0,
            validation_loss=0.3 + index * 0.01,
            stop_reason="max_epochs",
            device="cpu",
            elapsed_seconds=0.1,
            peak_cuda_memory_mb=0.0,
        )
        for index, seed in enumerate(spec.training.seeds)
    )
    return SelectedTraining(candidates[0].seed, candidates[0], candidates)


def _fake_solver_dataset(run_dir: Path, plan) -> DatasetFiles:
    x, y = np.meshgrid(np.linspace(0.0, 4.0, 3), np.linspace(0.0, 1.0, 3))
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    parameters = plan.parameters
    clamp = coordinates[:, 0] / 4.0
    shape = np.stack(
        (
            0.5 + parameters[:, 1, None] + 0.1 * coordinates[:, 1],
            np.sin(parameters[:, 3, None]) + 0.2 * coordinates[:, 1],
        ),
        axis=-1,
    )
    fields = shape * (
        (parameters[:, 2] / parameters[:, 0])[:, None, None]
        * clamp[None, :, None]
    )
    output = run_dir / "solver_output/datasets"
    output.mkdir(parents=True, exist_ok=True)
    development_path = output / "development.npz"
    test_path = output / "sealed_test.npz"
    development = np.flatnonzero(np.isin(plan.roles, ["train", "validation"]))
    test = np.flatnonzero(np.isin(plan.roles, ["development_test", "sealed_test"]))
    for path, indices in ((development_path, development), (test_path, test)):
        np.savez_compressed(
            path,
            sample_ids=plan.sample_ids[indices],
            roles=plan.roles[indices],
            parameters=parameters[indices],
            coordinates=coordinates,
            fields=fields[indices],
        )
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(
        '{"evidence":"fake_solver_orchestration_only"}\n', encoding="utf-8"
    )
    return DatasetFiles(
        development_path,
        test_path,
        manifest_path,
        sha256_file(development_path),
        sha256_file(test_path),
    )
