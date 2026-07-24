from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from surrogate_loop.operator.cavity2d.artifacts import (
    CavityRunState,
    consume_sealed_test_once,
    freeze_cavity_run,
    write_artifact_manifest,
    write_cavity_state,
)
from surrogate_loop.operator.cavity2d.config import load_cavity_spec
from surrogate_loop.operator.cavity2d.evaluation import (
    cavity_is_acceptable,
    compute_cavity_metrics,
)
from surrogate_loop.operator.cavity2d.model import (
    CavityPodRbfModel,
    fit_candidate,
    save_cavity_model,
)
from surrogate_loop.operator.cavity2d.protocol import (
    import_verified_cavity_dataset,
)
from surrogate_loop.operator.cavity2d.reporting import (
    write_cavity_evaluation_report,
    write_cavity_model_card,
)
from surrogate_loop.operator.cavity2d.sampling import build_cavity_sample_plan
from surrogate_loop.operator.field_data import FieldDataset, load_field_dataset

ENERGY_THRESHOLDS = (0.999, 0.9999)
KERNELS = ("cubic", "thin_plate_spline", "multiquadric")
SMOOTHING = (0.0, 1e-10, 1e-8)


@dataclass(frozen=True)
class CavityPipelineResult:
    run_dir: Path
    status: str
    selected_model: dict[str, object]
    validation_metrics: dict[str, object]
    test_metrics: dict[str, object] | None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _new_run_dir(runs_dir: Path, mode: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir.resolve() / f"cavity2d-{mode}-{stamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _subset(dataset: FieldDataset, ids: set[str]) -> FieldDataset:
    indices = np.asarray(
        [index for index, sample_id in enumerate(dataset.sample_ids) if sample_id in ids],
        dtype=np.int64,
    )
    if indices.size == 0:
        raise RuntimeError("requested cavity partition is empty")
    return dataset.subset(indices)


def _candidate_record(
    model: CavityPodRbfModel,
    validation: FieldDataset,
) -> tuple[dict[str, object], tuple[object, ...]]:
    prediction = model.predict(validation.parameters[:, 0])
    metrics = compute_cavity_metrics(
        validation.coordinates,
        validation.fields,
        prediction,
        validation.parameters[:, 0],
    )
    record = {
        "energy_threshold": model.energy_threshold,
        "kernel": model.velocity.kernel,
        "smoothing": model.velocity.smoothing,
        "velocity_components": int(model.velocity.components.shape[0]),
        "pressure_components": int(model.pressure.components.shape[0]),
        "metrics": asdict(metrics),
    }
    rank = (
        metrics.velocity_median_relative_l2,
        metrics.pressure_median_relative_l2,
        model.velocity.components.shape[0] + model.pressure.components.shape[0],
        KERNELS.index(model.velocity.kernel),
        model.velocity.smoothing,
        ENERGY_THRESHOLDS.index(model.energy_threshold),
    )
    return record, rank


def _fluent_seconds_per_sample(
    pipeline_complete: Path,
    *,
    sample_ids: set[str],
) -> float:
    payload = json.loads(pipeline_complete.read_text(encoding="utf-8"))
    fluent_sample_seconds: list[float] = []
    for batch in payload.get("batches", []):
        for sample in batch.get("samples", []):
            value = sample.get("wall_time_seconds")
            if (
                sample.get("sample_id") in sample_ids
                and isinstance(value, (int, float))
                and value > 0
            ):
                fluent_sample_seconds.append(float(value))
    return (
        float(np.mean(fluent_sample_seconds))
        if fluent_sample_seconds
        else 0.0
    )


def _cpu_speedup(
    pipeline_complete: Path,
    *,
    inference_seconds: float,
    inference_sample_count: int,
    sample_ids: set[str],
) -> float:
    fluent_seconds_per_sample = _fluent_seconds_per_sample(
        pipeline_complete,
        sample_ids=sample_ids,
    )
    if (
        fluent_seconds_per_sample <= 0.0
        or inference_seconds <= 0.0
        or inference_sample_count <= 0
    ):
        return 0.0
    inference_seconds_per_sample = inference_seconds / inference_sample_count
    return fluent_seconds_per_sample / inference_seconds_per_sample


def run_cavity_pipeline(
    config_path: Path,
    fluent_pipeline: Path,
    runs_dir: Path,
    request_text: str,
) -> CavityPipelineResult:
    spec = load_cavity_spec(config_path)
    if spec.mode not in {"smoke", "full"}:
        raise ValueError("training accepts only smoke or full cavity configurations")
    run_dir = _new_run_dir(runs_dir, spec.mode)
    write_cavity_state(run_dir, CavityRunState.PLANNED)
    try:
        _write_json(run_dir / "request.json", {"text": request_text})
        _write_json(run_dir / "spec.json", spec.model_dump(mode="json"))
        plan = build_cavity_sample_plan(spec)
        _write_json(
            run_dir / "sample_plan.json",
            {
                "sample_ids": list(plan.sample_ids),
                "reynolds": plan.reynolds.tolist(),
                "split": plan.split.tolist(),
            },
        )
        files = import_verified_cavity_dataset(
            fluent_pipeline,
            run_dir / "data",
            expected_sample_ids=plan.sample_ids,
            expected_reynolds=plan.reynolds,
            expected_splits=plan.split,
            expected_mesh_sha256=spec.mesh_sha256,
        )
        write_cavity_state(run_dir, CavityRunState.DATA_VERIFIED)
        development = load_field_dataset(files.development)
        provenance = json.loads(files.manifest.read_text(encoding="utf-8"))
        split_by_id = provenance["split_by_sample_id"]
        train_ids = {
            sample_id for sample_id, split in split_by_id.items() if split == "train"
        }
        validation_ids = {
            sample_id
            for sample_id, split in split_by_id.items()
            if split == "validation"
        }
        train = _subset(development, train_ids)
        validation = _subset(development, validation_ids)

        candidates: list[dict[str, object]] = []
        selected_model: CavityPodRbfModel | None = None
        selected_record: dict[str, object] | None = None
        selected_rank: tuple[object, ...] | None = None
        for energy_threshold, kernel, smoothing in itertools.product(
            ENERGY_THRESHOLDS,
            KERNELS,
            SMOOTHING,
        ):
            model = fit_candidate(
                train.parameters[:, 0],
                train.fields,
                energy_threshold=energy_threshold,
                kernel=kernel,
                smoothing=smoothing,
            )
            record, rank = _candidate_record(model, validation)
            candidates.append(record)
            if selected_rank is None or rank < selected_rank:
                selected_model = model
                selected_record = record
                selected_rank = rank
        assert selected_model is not None and selected_record is not None
        _write_json(
            run_dir / "model_selection.json",
            {"candidates": candidates, "selected": selected_record},
        )
        save_cavity_model(
            run_dir / "model",
            selected_model,
            problem_id=spec.problem.problem_id,
            mesh_sha256=str(provenance["mesh_sha256"]),
            coordinates_sha256=files.coordinates_sha256,
        )
        validation_metrics = dict(selected_record["metrics"])
        _write_json(run_dir / "validation_metrics.json", validation_metrics)
        write_cavity_state(run_dir, CavityRunState.MODEL_SELECTED)
        test_metrics: dict[str, object] | None = None
        report_files: list[Path] = []
        if spec.mode == "smoke":
            development_test_ids = {
                sample_id
                for sample_id, split in split_by_id.items()
                if split == "development_test"
            }
            development_test = _subset(development, development_test_ids)
            development_started = perf_counter()
            development_prediction = selected_model.predict(
                development_test.parameters[:, 0]
            )
            development_seconds = max(
                perf_counter() - development_started,
                1e-12,
            )
            development_metrics = compute_cavity_metrics(
                development_test.coordinates,
                development_test.fields,
                development_prediction,
                development_test.parameters[:, 0],
            )
            test_metrics = dict(asdict(development_metrics))
            _write_json(
                run_dir / "development_test_metrics.json",
                test_metrics,
            )
            development_ids = set(development_test.sample_ids.tolist())
            report_files = write_cavity_evaluation_report(
                run_dir / "report",
                sample_ids=development_test.sample_ids,
                reynolds=development_test.parameters[:, 0],
                coordinates=development_test.coordinates,
                reference=development_test.fields,
                prediction=development_prediction,
                fluent_seconds_per_sample=_fluent_seconds_per_sample(
                    fluent_pipeline,
                    sample_ids=development_ids,
                ),
                surrogate_seconds_per_sample=(
                    development_seconds / development_test.parameters.shape[0]
                ),
            )
        frozen_files = [
            "request.json",
            "spec.json",
            "sample_plan.json",
            "data/solver_provenance.json",
            "data/development.npz",
            "model/model.json",
            "model/model_arrays.npz",
            "model_selection.json",
            "validation_metrics.json",
        ]
        if spec.mode == "smoke":
            frozen_files.append("development_test_metrics.json")
            frozen_files.extend(
                str(path.relative_to(run_dir)) for path in report_files
            )
        freeze_cavity_run(run_dir, frozen_files, mode=spec.mode)
        if spec.mode == "full":
            if files.sealed_test is None:
                raise RuntimeError("Full cavity run is missing physically separated sealed data")
            consume_sealed_test_once(run_dir)
            sealed = load_field_dataset(files.sealed_test)
            started = perf_counter()
            prediction = np.concatenate(
                [
                    selected_model.predict(np.asarray([reynolds]))
                    for reynolds in sealed.parameters[:, 0]
                ],
                axis=0,
            )
            inference_seconds = max(perf_counter() - started, 1e-12)
            metrics = compute_cavity_metrics(
                sealed.coordinates,
                sealed.fields,
                prediction,
                sealed.parameters[:, 0],
            )
            speedup = _cpu_speedup(
                fluent_pipeline,
                inference_seconds=inference_seconds,
                inference_sample_count=sealed.parameters.shape[0],
                sample_ids=set(sealed.sample_ids.tolist()),
            )
            accepted = cavity_is_acceptable(metrics, speedup)
            test_metrics = {**asdict(metrics), "cpu_speedup": speedup}
            _write_json(run_dir / "test_metrics.json", test_metrics)
            _write_json(
                run_dir / "acceptance.json",
                {"accepted": accepted, "thresholds": "cavity2d_v1"},
            )
            sealed_ids = set(sealed.sample_ids.tolist())
            report_files = write_cavity_evaluation_report(
                run_dir / "report",
                sample_ids=sealed.sample_ids,
                reynolds=sealed.parameters[:, 0],
                coordinates=sealed.coordinates,
                reference=sealed.fields,
                prediction=prediction,
                fluent_seconds_per_sample=_fluent_seconds_per_sample(
                    fluent_pipeline,
                    sample_ids=sealed_ids,
                ),
                surrogate_seconds_per_sample=(
                    inference_seconds / sealed.parameters.shape[0]
                ),
            )
            write_cavity_state(
                run_dir,
                CavityRunState.ACCEPTED if accepted else CavityRunState.REJECTED,
            )
        status = (
            CavityRunState.DEVELOPMENT_COMPLETE.value
            if spec.mode == "smoke"
            else (
                CavityRunState.ACCEPTED.value
                if json.loads(
                    (run_dir / "acceptance.json").read_text(encoding="utf-8")
                )["accepted"]
                else CavityRunState.REJECTED.value
            )
        )
        selected_summary = {
            key: selected_record[key]
            for key in ("energy_threshold", "kernel", "smoothing")
        }
        write_cavity_model_card(
            run_dir / "model_card.md",
            status=status,
            mode=spec.mode,
            selected=selected_summary,
        )
        final_files = [
            *frozen_files,
            "freeze_manifest.json",
            "status.json",
            "model_card.md",
        ]
        if spec.mode == "full":
            final_files.extend(
                [
                    "data/sealed_test.npz",
                    "sealed-test-consumed.json",
                    "test_metrics.json",
                    "acceptance.json",
                ]
            )
            final_files.extend(
                str(path.relative_to(run_dir)) for path in report_files
            )
        write_artifact_manifest(run_dir, final_files)
        return CavityPipelineResult(
            run_dir=run_dir,
            status=status,
            selected_model=selected_summary,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )
    except Exception:
        write_cavity_state(run_dir, CavityRunState.FAILED)
        raise


__all__ = ["CavityPipelineResult", "run_cavity_pipeline"]
