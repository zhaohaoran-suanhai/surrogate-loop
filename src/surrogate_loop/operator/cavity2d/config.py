from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MESH_SHA256 = "9B09F1287DB71978E10C67A528616C1C95118CFCF4F763ABAD047DF565E6A6DD"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CavityProblemSpec(StrictModel):
    problem_id: Literal["fluent_lid_driven_cavity_steady_v1"]
    reynolds_low: Literal[10.0]
    reynolds_high: Literal[400.0]
    density: Literal[1.0]
    lid_speed: Literal[1.0]
    outputs: tuple[Literal["u"], Literal["v"], Literal["p_prime"]]


class CavitySamplingSpec(StrictModel):
    seed: int | None
    train_cases: int = Field(ge=0)
    validation_cases: int = Field(ge=0)
    test_cases: int = Field(ge=0)
    explicit_reynolds: tuple[float, ...]


class CavityRunSpec(StrictModel):
    mode: Literal["vertical", "calibration", "smoke", "full"]
    problem: CavityProblemSpec
    sampling: CavitySamplingSpec
    mesh_sha256: Literal[MESH_SHA256]

    @model_validator(mode="after")
    def validate_mode_contract(self) -> CavityRunSpec:
        actual = (
            self.sampling.seed,
            self.sampling.train_cases,
            self.sampling.validation_cases,
            self.sampling.test_cases,
            self.sampling.explicit_reynolds,
        )
        expected = {
            "vertical": (None, 0, 0, 0, (100.0,)),
            "calibration": (None, 0, 0, 0, (10.0, 100.0, 400.0)),
            "smoke": (2026072301, 16, 4, 4, ()),
            "full": (2026072302, 80, 20, 20, ()),
        }[self.mode]
        if actual != expected:
            raise ValueError(f"{self.mode} sampling contract must be {expected}")
        return self


def load_cavity_spec(path: Path) -> CavityRunSpec:
    return CavityRunSpec.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = ["CavityRunSpec", "load_cavity_spec"]
