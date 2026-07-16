from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from surrogate_loop.operator.artifacts import (
    create_operator_run_directory,
    save_operator_run,
    write_failed_run,
)
from surrogate_loop.operator.config import load_operator_spec
from surrogate_loop.operator.heat1d.dataset import (
    NormalizationStats,
    generate_dataset,
    split_dataset,
)
from surrogate_loop.operator.heat1d.deeponet import build_deeponet
from surrogate_loop.operator.heat1d.evaluation import (
    compute_field_metrics,
    deeponet_is_acceptable,
    solver_is_acceptable,
)
from surrogate_loop.operator.heat1d.pod_gpr import PodGprBaseline
from surrogate_loop.operator.heat1d.training import predict_dataset, train_deeponet
from surrogate_loop.operator.runtime import resolve_device, runtime_summary, seed_everything


@dataclass(frozen=True)
class OperatorRunResult:
    run_dir: Path
    status: str
    deeponet_metrics: dict[str, float]
    pod_metrics: dict[str, float]


def run_operator_pipeline(
    spec_path: Path,
    runs_dir: Path,
    request_text: str,
) -> OperatorRunResult:
    run_dir = create_operator_run_directory(runs_dir)
    try:
        spec = load_operator_spec(spec_path)
        device = resolve_device(spec.runtime.device)
        seed_everything(spec.sampling.seed)
        dataset = generate_dataset(spec)
        if not solver_is_acceptable(dataset, spec.solver_acceptance):
            raise RuntimeError("数值求解器未通过解析解与边界验收")
        split = split_dataset(dataset, spec.sampling)
        normalization = NormalizationStats.fit(split.train)
        baseline = PodGprBaseline(
            energy_threshold=spec.pod.energy_threshold,
            max_components=spec.pod.max_components,
            seed=spec.sampling.seed,
        ).fit(split.train.parameters, split.train.fields)
        training = train_deeponet(spec, split, normalization, device)
        model = build_deeponet(spec.model).to(device)
        model.load_state_dict(training.state_dict)
        test_prediction = predict_dataset(
            model,
            split.test,
            normalization,
            device,
            spec.training.query_batch_size,
        )
        deeponet_metrics = compute_field_metrics(
            split.test.fields,
            test_prediction,
            normalization.target_std,
        )
        pod_prediction = baseline.predict(split.test.parameters)
        pod_metrics = compute_field_metrics(
            split.test.fields,
            pod_prediction,
            normalization.target_std,
        )
        status = (
            "accepted"
            if deeponet_is_acceptable(deeponet_metrics, spec.acceptance)
            else "rejected"
        )
        save_operator_run(
            run_dir=run_dir,
            spec=spec,
            request_text=request_text,
            dataset=dataset,
            split=split,
            normalization=normalization,
            baseline=baseline,
            pod_metrics=pod_metrics,
            training=training,
            test_metrics=deeponet_metrics,
            test_prediction=test_prediction,
            status=status,
            runtime=runtime_summary(device),
        )
    except Exception as error:
        write_failed_run(run_dir, spec_path, error)
        raise
    return OperatorRunResult(
        run_dir=run_dir,
        status=status,
        deeponet_metrics=deeponet_metrics.to_dict(),
        pod_metrics=pod_metrics.to_dict(),
    )
