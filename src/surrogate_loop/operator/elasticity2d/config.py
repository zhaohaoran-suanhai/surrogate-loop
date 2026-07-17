from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParameterRange(StrictModel):
    low: float
    high: float

    @model_validator(mode="after")
    def validate_bounds(self) -> ParameterRange:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("参数范围必须有限")
        if self.low >= self.high:
            raise ValueError("参数下界必须小于上界")
        return self


class ElasticityProblemSpec(StrictModel):
    template: Literal["elasticity_2d_cantilever_v1"]
    formulation: Literal["plane_stress"]
    length: Literal[4.0]
    height: Literal[1.0]
    young_modulus: ParameterRange
    poisson_ratio: ParameterRange
    load_magnitude: ParameterRange
    load_angle: ParameterRange
    load_center: ParameterRange
    load_width: ParameterRange


class MeshSpec(StrictModel):
    nx: int = Field(gt=0)
    ny: int = Field(gt=0)
    degree: int = Field(gt=0)
    refinement_factor: int = Field(gt=1)


class ObservationSpec(StrictModel):
    nx: int = Field(ge=2)
    ny: int = Field(ge=2)


class ElasticitySamplingSpec(StrictModel):
    seed: Literal[20260716]
    train_cases: int = Field(ge=0)
    validation_cases: int = Field(ge=0)
    test_cases: int = Field(ge=0)


class ElasticitySolverSpec(StrictModel):
    backend: Literal["pyamg", "scipy"]
    max_relative_residual: float = Field(gt=0)
    max_force_balance_error: float = Field(gt=0)
    max_load_linearity_error: float = Field(gt=0)
    max_mesh_convergence_error: float = Field(gt=0)


class PodRbfSpec(StrictModel):
    energy_threshold: float = Field(gt=0, le=1)
    max_components: int = Field(gt=0)


class VectorDeepONetSpec(StrictModel):
    architecture: Literal["directional_linear_v2"]
    hidden_width: int = Field(gt=0)
    hidden_layers: int = Field(gt=0)
    latent_dim: int = Field(gt=0)


class ElasticityTrainingSpec(StrictModel):
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    case_batch_size: int = Field(gt=0)
    query_batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    lr_factor: float = Field(gt=0, lt=1)
    min_learning_rate: float = Field(gt=0)
    min_delta: float = Field(ge=0)
    max_minutes: float = Field(gt=0)
    seeds: tuple[int, ...]

    @model_validator(mode="after")
    def validate_relationships(self) -> ElasticityTrainingSpec:
        if self.patience > self.max_epochs:
            raise ValueError("patience 不能超过 max_epochs")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate 不能超过 learning_rate")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("训练随机种子必须非空且互不重复")
        return self


class ElasticityAcceptanceSpec(StrictModel):
    max_median_relative_l2: float = Field(gt=0)
    max_p95_relative_l2: float = Field(gt=0)
    max_worst_relative_l2: float = Field(gt=0)
    max_p95_tip_error: float = Field(gt=0)
    max_p95_compliance_error: float = Field(gt=0)
    max_clamp_absolute_error: float = Field(gt=0)
    min_cpu_speedup: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_relative_error_order(self) -> ElasticityAcceptanceSpec:
        if not (
            self.max_median_relative_l2
            <= self.max_p95_relative_l2
            <= self.max_worst_relative_l2
        ):
            raise ValueError("median、p95、worst 位移门槛必须依次非递减")
        return self


class RuntimeSpec(StrictModel):
    device: Literal["auto", "cpu", "cuda"]
    dtype: Literal["float32"]


class ElasticityRunSpec(StrictModel):
    mode: Literal["calibration", "smoke", "full"]
    problem: ElasticityProblemSpec
    mesh: MeshSpec
    observation: ObservationSpec
    sampling: ElasticitySamplingSpec
    solver: ElasticitySolverSpec
    pod: PodRbfSpec
    model: VectorDeepONetSpec
    training: ElasticityTrainingSpec
    acceptance: ElasticityAcceptanceSpec
    runtime: RuntimeSpec

    @model_validator(mode="after")
    def validate_scientific_contract(self) -> ElasticityRunSpec:
        expected_problem = (
            (1.0, 5.0),
            (0.2, 0.45),
            (0.002, 0.01),
            (-math.pi, math.pi),
            (0.2, 0.8),
            (0.08, 0.2),
        )
        actual_problem = tuple(
            (bounds.low, bounds.high)
            for bounds in (
                self.problem.young_modulus,
                self.problem.poisson_ratio,
                self.problem.load_magnitude,
                self.problem.load_angle,
                self.problem.load_center,
                self.problem.load_width,
            )
        )
        if actual_problem != expected_problem:
            raise ValueError(f"{self._mode_name()} 问题参数合同必须为 {expected_problem}")
        solver_contract = (
            self.solver.backend,
            self.solver.max_relative_residual,
            self.solver.max_force_balance_error,
            self.solver.max_load_linearity_error,
            self.solver.max_mesh_convergence_error,
        )
        if solver_contract != ("pyamg", 1e-8, 1e-5, 1e-6, 0.01):
            raise ValueError(f"{self._mode_name()} 求解器合同无效")
        if (self.pod.energy_threshold, self.pod.max_components) != (0.999, 64):
            raise ValueError(f"{self._mode_name()} POD-RBF 合同无效")
        acceptance_contract = (
            self.acceptance.max_median_relative_l2,
            self.acceptance.max_p95_relative_l2,
            self.acceptance.max_worst_relative_l2,
            self.acceptance.max_p95_tip_error,
            self.acceptance.max_p95_compliance_error,
            self.acceptance.max_clamp_absolute_error,
            self.acceptance.min_cpu_speedup,
        )
        if acceptance_contract != (0.03, 0.08, 0.15, 0.08, 0.1, 1e-7, 100.0):
            raise ValueError(f"{self._mode_name()} 代理模型验收合同无效")
        expected_by_mode = {
            "calibration": (
                (256, 64, 2, 2),
                (129, 33),
                (16, 0, 0),
                ("directional_linear_v2", 16, 2, 8),
                (1, 1, 1, 1, 1e-3, 0.5, 1e-6, 0.0, 1.0, (20260716,)),
            ),
            "smoke": (
                (128, 32, 2, 2),
                (65, 17),
                (96, 24, 24),
                ("directional_linear_v2", 128, 3, 128),
                (1000, 80, 16, 256, 1e-3, 0.5, 1e-6, 1e-6, 30.0, (20260716,)),
            ),
            "full": (
                (256, 64, 2, 2),
                (129, 33),
                (512, 96, 128),
                ("directional_linear_v2", 128, 3, 128),
                (
                    600,
                    60,
                    32,
                    512,
                    1e-3,
                    0.5,
                    1e-6,
                    1e-6,
                    60.0,
                    (20260716, 20260717, 20260718),
                ),
            ),
        }
        actual = (
            (self.mesh.nx, self.mesh.ny, self.mesh.degree, self.mesh.refinement_factor),
            (self.observation.nx, self.observation.ny),
            (
                self.sampling.train_cases,
                self.sampling.validation_cases,
                self.sampling.test_cases,
            ),
            (
                self.model.architecture,
                self.model.hidden_width,
                self.model.hidden_layers,
                self.model.latent_dim,
            ),
            (
                self.training.max_epochs,
                self.training.patience,
                self.training.case_batch_size,
                self.training.query_batch_size,
                self.training.learning_rate,
                self.training.lr_factor,
                self.training.min_learning_rate,
                self.training.min_delta,
                self.training.max_minutes,
                self.training.seeds,
            ),
        )
        if actual != expected_by_mode[self.mode]:
            raise ValueError(f"{self._mode_name()} 规模与训练合同无效")
        return self

    def _mode_name(self) -> str:
        return {"calibration": "Calibration", "smoke": "Smoke", "full": "Full"}[
            self.mode
        ]


def load_elasticity_spec(path: Path) -> ElasticityRunSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ElasticityRunSpec.model_validate(payload)
