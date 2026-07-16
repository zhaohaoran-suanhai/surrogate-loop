from __future__ import annotations

from typing import Any

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_candidates(seed: int, names: tuple[str, ...]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for name in names:
        if name.startswith("prs_"):
            degree = int(name.removeprefix("prs_"))
            candidates[name] = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
                    ("regressor", Ridge(alpha=1e-8)),
                ]
            )
        elif name == "gpr":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
            candidates[name] = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "regressor",
                        GaussianProcessRegressor(
                            kernel=kernel,
                            alpha=1e-10,
                            normalize_y=True,
                            n_restarts_optimizer=2,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        elif name == "mlp":
            candidates[name] = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "regressor",
                        MLPRegressor(
                            hidden_layer_sizes=(32, 32),
                            activation="tanh",
                            alpha=1e-4,
                            max_iter=2000,
                            early_stopping=True,
                            validation_fraction=0.2,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(f"不支持的候选模型：{name}")
    return candidates
