from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ModelName = Literal["prs_1", "prs_2", "prs_3", "gpr", "mlp"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParameterRange(StrictModel):
    low: float
    high: float

    @model_validator(mode="after")
    def validate_bounds(self) -> ParameterRange:
        if self.low < -1.0 or self.high > 1.0:
            raise ValueError("gamma 范围必须位于 [-1, 1]")
        if self.low >= self.high:
            raise ValueError("gamma 下界必须小于上界")
        return self


class ProblemSpec(StrictModel):
    template: Literal["forced_reaction_scalar_endpoint_v1"]
    gamma: ParameterRange
    time_start: Literal[0.0]
    time_end: Literal[1.0]
    initial_value: Literal[0.0]
    forcing_coefficient: Literal[0.5]


class SamplingSpec(StrictModel):
    seed: int
    train_cases: int = Field(gt=0)
    validation_cases: int = Field(gt=0)
    test_cases: int = Field(gt=0)


class ModelSpec(StrictModel):
    candidates: tuple[ModelName, ...] = Field(min_length=1)

    @field_validator("candidates")
    @classmethod
    def candidates_are_unique(cls, value: tuple[ModelName, ...]) -> tuple[ModelName, ...]:
        if len(set(value)) != len(value):
            raise ValueError("候选模型名称不能重复")
        return value


class AcceptanceSpec(StrictModel):
    max_nrmse: float = Field(gt=0)
    max_absolute_error: float = Field(gt=0)


class RunSpec(StrictModel):
    mode: Literal["full", "smoke"]
    problem: ProblemSpec
    sampling: SamplingSpec
    models: ModelSpec
    acceptance: AcceptanceSpec

    @model_validator(mode="after")
    def validate_canonical_scale(self) -> RunSpec:
        actual = (
            self.sampling.train_cases,
            self.sampling.validation_cases,
            self.sampling.test_cases,
        )
        expected = (24, 8, 8) if self.mode == "smoke" else (120, 40, 40)
        if actual != expected:
            raise ValueError(f"{self.mode} 配置的工况数必须为 {expected}")
        return self


def load_spec(path: Path) -> RunSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunSpec.model_validate(payload)
