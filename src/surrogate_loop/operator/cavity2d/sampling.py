from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from surrogate_loop.operator.cavity2d.config import CavityRunSpec


@dataclass(frozen=True)
class CavitySamplePlan:
    sample_ids: tuple[str, ...]
    reynolds: np.ndarray
    split: np.ndarray


def build_cavity_sample_plan(spec: CavityRunSpec) -> CavitySamplePlan:
    if spec.mode in {"vertical", "calibration"}:
        split_name = "protocol" if spec.mode == "vertical" else "calibration"
        reynolds = np.asarray(spec.sampling.explicit_reynolds, dtype=np.float64)
        return CavitySamplePlan(
            sample_ids=tuple(
                f"{spec.mode}-{index:03d}" for index in range(reynolds.size)
            ),
            reynolds=reynolds,
            split=np.asarray([split_name] * reynolds.size),
        )

    count = (
        spec.sampling.train_cases
        + spec.sampling.validation_cases
        + spec.sampling.test_cases
    )
    assert spec.sampling.seed is not None
    sampler = qmc.LatinHypercube(d=1, seed=spec.sampling.seed)
    log_bounds = np.log10([10.0, 400.0])
    reynolds = np.power(
        10.0,
        qmc.scale(
            sampler.random(count),
            [log_bounds[0]],
            [log_bounds[1]],
        )[:, 0],
    )
    split = np.asarray(
        ["train"] * spec.sampling.train_cases
        + ["validation"] * spec.sampling.validation_cases
        + [
            "development_test" if spec.mode == "smoke" else "sealed_test"
        ]
        * spec.sampling.test_cases
    )
    return CavitySamplePlan(
        sample_ids=tuple(f"{spec.mode}-{index:03d}" for index in range(count)),
        reynolds=reynolds,
        split=split,
    )


def write_solver_request(
    output_dir: Path,
    spec: CavityRunSpec,
    plan: CavitySamplePlan,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "solver-request.json"
    samples = [
        {
            "sample_id": sample_id,
            "reynolds": float(reynolds),
            "split": str(split),
        }
        for sample_id, reynolds, split in zip(
            plan.sample_ids,
            plan.reynolds,
            plan.split,
            strict=True,
        )
    ]
    payload = {
        "schema_version": 1,
        "problem_id": spec.problem.problem_id,
        "request_id": f"cavity2d-{spec.mode}",
        "mesh_sha256": spec.mesh_sha256,
        "samples": samples,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "CavitySamplePlan",
    "build_cavity_sample_plan",
    "write_solver_request",
]
