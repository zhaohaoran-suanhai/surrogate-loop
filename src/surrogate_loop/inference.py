from __future__ import annotations

import math
from pathlib import Path

from surrogate_loop.artifacts import load_verified_model


def predict_endpoint(run_dir: Path, gamma: float) -> float:
    if not math.isfinite(gamma):
        raise ValueError("gamma 必须是有限数值")
    spec, model, _ = load_verified_model(run_dir)
    if gamma < spec.problem.gamma.low or gamma > spec.problem.gamma.high:
        raise ValueError(
            f"gamma={gamma} 超出训练参数域 "
            f"[{spec.problem.gamma.low}, {spec.problem.gamma.high}]"
        )
    return float(model.predict([[gamma]])[0])
