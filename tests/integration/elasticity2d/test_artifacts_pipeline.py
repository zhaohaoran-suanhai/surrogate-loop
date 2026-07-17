from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import surrogate_loop.operator.elasticity2d.artifacts as artifacts_module
import surrogate_loop.operator.elasticity2d.pipeline as pipeline_module
from surrogate_loop.operator.elasticity2d.artifacts import (
    ElasticityRunState,
    evaluate_sealed_once,
    freeze_run,
    read_run_state,
    transition_run,
    verify_freeze_manifest,
)
from surrogate_loop.operator.elasticity2d.config import load_elasticity_spec
from surrogate_loop.operator.elasticity2d.dataset import DatasetFiles
from surrogate_loop.operator.elasticity2d.deeponet import build_elasticity_deeponet
from surrogate_loop.operator.elasticity2d.inference import read_elasticity_report
from surrogate_loop.operator.elasticity2d.pod_rbf import PodRbfBaseline
from surrogate_loop.operator.elasticity2d.problem import elasticity_basis_features
from surrogate_loop.operator.elasticity2d.sampling import build_sample_plan
from surrogate_loop.operator.elasticity2d.training import (
    SelectedTraining,
    TrainingRecord,
    TrainingResult,
)
from surrogate_loop.operator.field_data import FieldDataset, FieldNormalization, sha256_file

ROOT = Path(__file__).resolve().parents[3]


def test_neural_speed_benchmark_measures_one_sample(monkeypatch) -> None:
    calls: list[int] = []
    dataset = FieldDataset(
        sample_ids=np.array([f"sample-{index}" for index in range(5)]),
        parameters=np.tile(np.array([[2.0, 0.3, 0.004, 0.0, 0.5, 0.1]]), (5, 1)),
        coordinates=np.array([[0.0, 0.0], [4.0, 1.0]]),
        fields=np.zeros((5, 2, 2)),
        diagnostics={},
    )
    normalization = FieldNormalization(
        feature_mean=np.zeros(3),
        feature_std=np.ones(3),
        coordinate_mean=np.zeros(2),
        coordinate_std=np.ones(2),
        target_rms=np.ones(2),
    )

    def fake_predict(model, selected, normalization, device, query_batch_size):
        calls.append(selected.parameters.shape[0])
        return selected.fields

    monkeypatch.setattr(artifacts_module, "predict_dataset", fake_predict)

    elapsed = artifacts_module._benchmark_neural(
        torch.nn.Identity(), dataset, normalization, 16
    )

    assert elapsed > 0.0
    assert len(calls) == 110
    assert set(calls) == {1}


def test_run_state_machine_allows_only_declared_atomic_transitions(tmp_path) -> None:
    run_dir = tmp_path / "run"

    transition_run(run_dir, None, ElasticityRunState.CREATED)
    transition_run(
        run_dir,
        ElasticityRunState.CREATED,
        ElasticityRunState.SOLVER_ACCEPTED,
    )
    transition_run(
        run_dir,
        ElasticityRunState.SOLVER_ACCEPTED,
        ElasticityRunState.TRAINED,
    )
    transition_run(
        run_dir,
        ElasticityRunState.TRAINED,
        ElasticityRunState.FROZEN,
    )

    assert read_run_state(run_dir) is ElasticityRunState.FROZEN
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8")) == {
        "state": "frozen"
    }
    assert not list(run_dir.glob("*.tmp"))
    assert not (run_dir / ".state.lock").exists()


def test_run_state_machine_rejects_skips_and_stale_expected_state(tmp_path) -> None:
    run_dir = tmp_path / "run"
    transition_run(run_dir, None, ElasticityRunState.CREATED)

    with pytest.raises(RuntimeError, match="不允许"):
        transition_run(
            run_dir,
            ElasticityRunState.CREATED,
            ElasticityRunState.FROZEN,
        )
    with pytest.raises(RuntimeError, match="当前状态"):
        transition_run(
            run_dir,
            ElasticityRunState.SOLVER_ACCEPTED,
            ElasticityRunState.TRAINED,
        )

    assert read_run_state(run_dir) is ElasticityRunState.CREATED


@pytest.mark.parametrize(
    "source",
    [
        ElasticityRunState.CREATED,
        ElasticityRunState.SOLVER_ACCEPTED,
        ElasticityRunState.TRAINED,
        ElasticityRunState.FROZEN,
    ],
)
def test_active_run_can_be_failed_once(tmp_path, source: ElasticityRunState) -> None:
    run_dir = tmp_path / source.value
    transition_run(run_dir, None, ElasticityRunState.CREATED)
    if source in {
        ElasticityRunState.SOLVER_ACCEPTED,
        ElasticityRunState.TRAINED,
        ElasticityRunState.FROZEN,
    }:
        transition_run(
            run_dir,
            ElasticityRunState.CREATED,
            ElasticityRunState.SOLVER_ACCEPTED,
        )
    if source in {ElasticityRunState.TRAINED, ElasticityRunState.FROZEN}:
        transition_run(
            run_dir,
            ElasticityRunState.SOLVER_ACCEPTED,
            ElasticityRunState.TRAINED,
        )
    if source is ElasticityRunState.FROZEN:
        transition_run(
            run_dir,
            ElasticityRunState.TRAINED,
            ElasticityRunState.FROZEN,
        )

    transition_run(run_dir, source, ElasticityRunState.FAILED)

    assert read_run_state(run_dir) is ElasticityRunState.FAILED
    with pytest.raises(RuntimeError, match="不允许"):
        transition_run(
            run_dir,
            ElasticityRunState.FAILED,
            ElasticityRunState.CREATED,
        )


def test_freeze_run_hashes_scientific_identity_and_selected_checkpoint(tmp_path) -> None:
    inputs = _freeze_inputs(tmp_path)

    manifest = freeze_run(**inputs)

    run_dir = inputs["run_dir"]
    assert read_run_state(run_dir) is ElasticityRunState.FROZEN
    assert manifest.version == 1
    assert manifest.selected_seed == inputs["selected_training"].selected_seed
    assert set(manifest.files) == {
        "spec.json",
        "sample_plan.json",
        "dataset_identity.json",
        "normalization.json",
        "pod_rbf.joblib",
        "network.json",
        "training_candidates.json",
        "deeponet_state.pt",
    }
    verified = verify_freeze_manifest(run_dir)
    assert verified == manifest
    assert all(sha256_file(run_dir / name) == digest for name, digest in manifest.files.items())
    assert json.loads((run_dir / "network.json").read_text(encoding="utf-8")) == {
        "architecture": "directional_linear_v2",
        "branch_input_dim": 3,
        "trunk_input_dim": 2,
        "output_dim": 4,
        "hidden_width": inputs["spec"].model.hidden_width,
        "hidden_layers": inputs["spec"].model.hidden_layers,
        "latent_dim": inputs["spec"].model.latent_dim,
    }


def test_freeze_manifest_rejects_tampered_hashed_artifact(tmp_path) -> None:
    inputs = _freeze_inputs(tmp_path)
    freeze_run(**inputs)
    network = inputs["run_dir"] / "network.json"
    network.write_text(network.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_freeze_manifest(inputs["run_dir"])


def test_sealed_test_cannot_be_read_before_freeze(tmp_path) -> None:
    inputs = _freeze_inputs(tmp_path)

    with pytest.raises(RuntimeError, match="frozen"):
        evaluate_sealed_once(inputs["run_dir"], inputs["dataset_files"])


def test_sealed_test_cannot_be_evaluated_twice(tmp_path, monkeypatch) -> None:
    inputs = _freeze_inputs(tmp_path)
    freeze_run(**inputs)
    monkeypatch.setattr(artifacts_module, "_benchmark_fenicsx", lambda *args: 0.5)

    first = evaluate_sealed_once(inputs["run_dir"], inputs["dataset_files"])

    assert first.status in {"accepted", "rejected"}
    assert first.neural_median_seconds > 0.0
    assert first.fenicsx_median_seconds == 0.5
    assert first.speedup == pytest.approx(0.5 / first.neural_median_seconds)
    assert (inputs["run_dir"] / "acceptance.json").is_file()
    assert (inputs["run_dir"] / "acceptance_stage.json").is_file()
    stage = json.loads(
        (inputs["run_dir"] / "acceptance_stage.json").read_text(encoding="utf-8")
    )
    assert "fenicsx_benchmark_manifest_sha256" in stage
    with pytest.raises(RuntimeError, match="已经消费"):
        evaluate_sealed_once(inputs["run_dir"], inputs["dataset_files"])


def test_sealed_tampering_fails_run_without_second_chance(tmp_path, monkeypatch) -> None:
    inputs = _freeze_inputs(tmp_path)
    freeze_run(**inputs)
    sealed_path = inputs["dataset_files"].sealed_test_path
    with sealed_path.open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setattr(artifacts_module, "_benchmark_fenicsx", lambda *args: 0.5)

    with pytest.raises(RuntimeError, match="SHA-256"):
        evaluate_sealed_once(inputs["run_dir"], inputs["dataset_files"])

    assert read_run_state(inputs["run_dir"]) is ElasticityRunState.FAILED
    with pytest.raises(RuntimeError, match="frozen"):
        evaluate_sealed_once(inputs["run_dir"], inputs["dataset_files"])


def test_smoke_pipeline_uses_development_evidence_and_resumes(tmp_path, monkeypatch) -> None:
    generate_calls = 0

    def fake_generate(spec, sample_plan, run_dir, repo_root, *, reuse_data_from=None):
        nonlocal generate_calls
        generate_calls += 1
        assert reuse_data_from is None
        return _write_protocol_dataset(run_dir, sample_plan)

    def fake_train(spec, partitions, normalization, device):
        state_dict = {
            name: value.detach().cpu().clone()
            for name, value in build_elasticity_deeponet(spec.model).state_dict().items()
        }
        record = TrainingRecord(0, 0.2, 0.3, 1e-3)
        result = TrainingResult(
            seed=spec.training.seeds[0],
            state_dict=state_dict,
            history=(record,),
            best_epoch=0,
            validation_loss=0.3,
            stop_reason="max_epochs",
            device="cpu",
            elapsed_seconds=0.1,
            peak_cuda_memory_mb=0.0,
        )
        return SelectedTraining(result.seed, result, (result,))

    monkeypatch.setattr(pipeline_module, "generate_or_reuse_dataset", fake_generate)
    monkeypatch.setattr(pipeline_module, "train_and_select", fake_train)
    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: torch.device("cpu"))
    config = ROOT / "examples/elasticity_2d_cantilever/smoke.json"

    first = pipeline_module.run_elasticity_pipeline(
        config, tmp_path / "runs", "二维悬臂梁 Smoke"
    )
    second = pipeline_module.run_elasticity_pipeline(
        config, tmp_path / "runs", "二维悬臂梁 Smoke"
    )

    assert first.status == "development_complete"
    assert second == first
    assert generate_calls == 1
    assert read_run_state(first.run_dir) is ElasticityRunState.TRAINED
    assert (first.run_dir / "development_evaluation.json").is_file()
    assert (first.run_dir / "diagnostics/displacement_comparison.png").is_file()
    assert (first.run_dir / "diagnostics/fenicsx_stress_summary.png").is_file()
    assert not (first.run_dir / "freeze_manifest.json").exists()
    state, report = read_elasticity_report(first.run_dir)
    assert state is ElasticityRunState.TRAINED
    assert report["status"] == "development_complete"
    assert report["schema_version"] == 6
    assert report["model_architecture"] == "directional_linear_v2"
    assert set(report["directional_metrics"]) == {
        "near_horizontal",
        "oblique",
        "near_vertical",
    }
    assert report["data_provenance"] == {"mode": "generated"}
    assert report["training"]["selected_seed"] == 20260716
    assert report["timing"]["speedup"] > 0.0

    with (first.run_dir / "development_evaluation.json").open("ab") as stream:
        stream.write(b"tampered")
    recovered = pipeline_module.run_elasticity_pipeline(
        config, tmp_path / "runs", "二维悬臂梁 Smoke"
    )
    assert recovered == first
    assert generate_calls == 2

    (first.run_dir / "dataset_reuse.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="完整性|来源"):
        read_elasticity_report(first.run_dir)


def test_legacy_schema_5_smoke_report_remains_readable(tmp_path) -> None:
    run_dir = tmp_path / "legacy"
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"state":"trained"}', encoding="utf-8")
    displacement = diagnostics / "displacement_comparison.png"
    stress = diagnostics / "fenicsx_stress_summary.png"
    displacement.write_bytes(b"displacement")
    stress.write_bytes(b"stress")
    report = {
        "schema_version": 5,
        "status": "development_complete",
        "deeponet_metrics": {"median_relative_l2": 0.01},
        "pod_rbf_metrics": {"median_relative_l2": 0.02},
        "training": {},
        "timing": {},
    }
    report_path = run_dir / "development_evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    stage = {
        "schema_version": 5,
        "status": "complete",
        "result_sha256": sha256_file(report_path),
        "diagnostic_sha256": {
            "diagnostics/displacement_comparison.png": sha256_file(displacement),
            "diagnostics/fenicsx_stress_summary.png": sha256_file(stress),
        },
    }
    (run_dir / "development_stage.json").write_text(
        json.dumps(stage), encoding="utf-8"
    )

    state, loaded = read_elasticity_report(run_dir)

    assert state is ElasticityRunState.TRAINED
    assert loaded == report


def _freeze_inputs(tmp_path: Path) -> dict[str, object]:
    spec = load_elasticity_spec(ROOT / "examples/elasticity_2d_cantilever/full.json")
    sample_plan = build_sample_plan(spec)
    x, y = np.meshgrid(np.linspace(0.0, 4.0, 3), np.linspace(0.0, 1.0, 3))
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    parameters = sample_plan.parameters
    clamp = coordinates[:, 0] / 4.0
    shape_x = 0.5 + parameters[:, 1, None] + 0.1 * coordinates[:, 1]
    shape_y = np.sin(parameters[:, 3, None]) + 0.2 * coordinates[:, 1]
    fields = np.stack((shape_x, shape_y), axis=-1)
    fields *= (
        (parameters[:, 2] / parameters[:, 0])[:, None, None]
        * clamp[None, :, None]
    )
    train = np.flatnonzero(sample_plan.roles == "train")
    development = np.flatnonzero(
        np.isin(sample_plan.roles, ["train", "validation"])
    )
    sealed = np.flatnonzero(sample_plan.roles == "sealed_test")
    solver_output = tmp_path / "solver_output" / "datasets"
    solver_output.mkdir(parents=True)
    development_path = solver_output / "development.npz"
    sealed_path = solver_output / "sealed_test.npz"
    for path, indices in ((development_path, development), (sealed_path, sealed)):
        np.savez_compressed(
            path,
            sample_ids=sample_plan.sample_ids[indices],
            roles=sample_plan.roles[indices],
            parameters=parameters[indices],
            coordinates=coordinates,
            fields=fields[indices],
        )
    solver_manifest = tmp_path / "solver_output" / "dataset_manifest.json"
    solver_manifest.write_text("{}\n", encoding="utf-8")
    dataset_files = DatasetFiles(
        development_path=development_path,
        sealed_test_path=sealed_path,
        manifest_path=solver_manifest,
        development_sha256=sha256_file(development_path),
        sealed_test_sha256=sha256_file(sealed_path),
    )
    normalization = FieldNormalization.fit(
        elasticity_basis_features(parameters[train]), coordinates, fields[train]
    )
    baseline = PodRbfBaseline(
        energy_threshold=spec.pod.energy_threshold,
        max_components=spec.pod.max_components,
    ).fit(parameters[train], fields[train])
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in build_elasticity_deeponet(spec.model).state_dict().items()
    }
    record = TrainingRecord(epoch=0, train_loss=0.2, validation_loss=0.3, learning_rate=1e-3)
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
    selected_training = SelectedTraining(
        selected_seed=candidates[0].seed,
        selected=candidates[0],
        candidates=candidates,
    )
    run_dir = tmp_path / "run"
    transition_run(run_dir, None, ElasticityRunState.CREATED)
    transition_run(run_dir, ElasticityRunState.CREATED, ElasticityRunState.SOLVER_ACCEPTED)
    transition_run(run_dir, ElasticityRunState.SOLVER_ACCEPTED, ElasticityRunState.TRAINED)
    return {
        "run_dir": run_dir,
        "spec": spec,
        "sample_plan": sample_plan,
        "dataset_files": dataset_files,
        "normalization": normalization,
        "baseline": baseline,
        "selected_training": selected_training,
    }


def _write_protocol_dataset(run_dir: Path, sample_plan) -> DatasetFiles:
    x, y = np.meshgrid(np.linspace(0.0, 4.0, 3), np.linspace(0.0, 1.0, 3))
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    parameters = sample_plan.parameters
    clamp = coordinates[:, 0] / 4.0
    fields = np.stack(
        (
            0.5 + parameters[:, 1, None] + 0.1 * coordinates[:, 1],
            np.sin(parameters[:, 3, None]) + 0.2 * coordinates[:, 1],
        ),
        axis=-1,
    )
    fields *= (
        (parameters[:, 2] / parameters[:, 0])[:, None, None]
        * clamp[None, :, None]
    )
    output = run_dir / "solver_output" / "datasets"
    output.mkdir(parents=True, exist_ok=True)
    development_path = output / "development.npz"
    test_path = output / "sealed_test.npz"
    development = np.flatnonzero(np.isin(sample_plan.roles, ["train", "validation"]))
    test = np.flatnonzero(
        np.isin(sample_plan.roles, ["development_test", "sealed_test"])
    )
    for path, indices in ((development_path, development), (test_path, test)):
        np.savez_compressed(
            path,
            sample_ids=sample_plan.sample_ids[indices],
            roles=sample_plan.roles[indices],
            parameters=parameters[indices],
            coordinates=coordinates,
            fields=fields[indices],
        )
    manifest_path = output / "dataset_manifest.json"
    records = [
        {
            "sample_id": str(sample_id),
            "diagnostics": {"solve_seconds": 0.05},
            "stress_summary": {
                "stress_xx_min": -2.0,
                "stress_xx_max": 3.0,
                "stress_xx_p95": 2.5,
                "stress_yy_min": -1.0,
                "stress_yy_max": 1.5,
                "stress_yy_p95": 1.2,
                "stress_xy_min": -0.5,
                "stress_xy_max": 0.75,
                "stress_xy_p95": 0.6,
                "von_mises_min": 0.0,
                "von_mises_max": 4.0,
                "von_mises_p95": 3.2,
            },
        }
        for sample_id in sample_plan.sample_ids
    ]
    manifest_path.write_text(json.dumps({"samples": records}), encoding="utf-8")
    return DatasetFiles(
        development_path=development_path,
        sealed_test_path=test_path,
        manifest_path=manifest_path,
        development_sha256=sha256_file(development_path),
        sealed_test_sha256=sha256_file(test_path),
    )
