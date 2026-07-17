from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from surrogate_loop.operator.elasticity2d.config import ElasticityRunSpec


@dataclass(frozen=True)
class SamplePlan:
    sample_ids: NDArray[np.str_]
    parameters: NDArray[np.float64]
    roles: NDArray[np.str_]

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids, dtype=np.str_)
        parameters = np.asarray(self.parameters, dtype=np.float64)
        roles = np.asarray(self.roles, dtype=np.str_)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "roles", roles)
        if parameters.ndim != 2 or parameters.shape[1] != 6:
            raise ValueError("二维弹性参数形状必须为 (n_samples, 6)")
        if sample_ids.shape != (parameters.shape[0],):
            raise ValueError("样本 ID 形状必须为 (n_samples,)")
        if roles.shape != (parameters.shape[0],):
            raise ValueError("样本角色形状必须为 (n_samples,)")
        if len(set(sample_ids.tolist())) != sample_ids.size:
            raise ValueError("样本 ID 必须互不重复")
        if np.any(sample_ids == "") or np.any(roles == ""):
            raise ValueError("样本 ID 和角色不能为空")
        if not np.isfinite(parameters).all():
            raise ValueError("二维弹性参数必须全部有限")


def build_sample_plan(spec: ElasticityRunSpec) -> SamplePlan:
    counts = (
        spec.sampling.train_cases,
        spec.sampling.validation_cases,
        spec.sampling.test_cases,
    )
    total = sum(counts)
    unit = qmc.LatinHypercube(d=6, seed=spec.sampling.seed).random(total)
    bounds = (
        spec.problem.young_modulus,
        spec.problem.poisson_ratio,
        spec.problem.load_magnitude,
        spec.problem.load_angle,
        spec.problem.load_center,
        spec.problem.load_width,
    )
    lower = np.array([value.low for value in bounds], dtype=np.float64)
    upper = np.array([value.high for value in bounds], dtype=np.float64)
    parameters = np.asarray(qmc.scale(unit, lower, upper), dtype=np.float64)
    if spec.mode == "calibration":
        roles = np.full(total, "calibration", dtype="<U11")
    else:
        test_role = "sealed_test" if spec.mode == "full" else "development_test"
        roles = np.array(
            ["train"] * counts[0]
            + ["validation"] * counts[1]
            + [test_role] * counts[2],
            dtype=np.str_,
        )
    sample_ids = np.array(
        [
            f"{role}-{index:05d}-{hashlib.sha256(row.tobytes()).hexdigest()[:12]}"
            for index, (role, row) in enumerate(zip(roles, parameters, strict=True))
        ],
        dtype=np.str_,
    )
    return SamplePlan(sample_ids=sample_ids, parameters=parameters, roles=roles)
