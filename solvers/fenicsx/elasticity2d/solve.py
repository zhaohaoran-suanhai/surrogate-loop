from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
import pyamg
import scipy.sparse.linalg
import ufl
from dolfinx import fem, la

from solvers.fenicsx.elasticity2d.problem import build_elasticity_problem


@dataclass(frozen=True)
class SolvedCase:
    solution: Any
    relative_residual: float
    reaction: np.ndarray
    applied_force: np.ndarray
    solve_seconds: float
    iterations: int
    observed_peak_rss_mb: float
    stress_summary: dict[str, float]


def solve_case(
    parameters: np.ndarray,
    mesh_shape: tuple[int, int],
    backend: str,
    tolerance: float,
) -> SolvedCase:
    if backend not in {"pyamg", "scipy"}:
        raise ValueError("线性求解后端只能是 pyamg 或 scipy")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("线性求解容差必须是有限正数")

    problem = build_elasticity_problem(parameters, mesh_shape)
    memory = _MemoryTracker()

    unconstrained_matrix = fem.assemble_matrix(problem.bilinear_form).to_scipy()
    unconstrained_vector = fem.assemble_vector(problem.linear_form)
    unconstrained_vector.scatter_reverse(la.InsertMode.add)
    raw_rhs = unconstrained_vector.array.copy()

    matrix = fem.assemble_matrix(
        problem.bilinear_form, bcs=[problem.clamp_bc]
    ).to_scipy()
    right_hand_side = fem.assemble_vector(problem.linear_form)
    fem.apply_lifting(
        right_hand_side.array,
        [problem.bilinear_form],
        bcs=[[problem.clamp_bc]],
    )
    right_hand_side.scatter_reverse(la.InsertMode.add)
    problem.clamp_bc.set(right_hand_side.array)
    memory.sample()

    started = time.perf_counter()
    if backend == "pyamg":
        residual_history: list[float] = []
        hierarchy = pyamg.smoothed_aggregation_solver(matrix)
        coefficients = hierarchy.solve(
            right_hand_side.array,
            tol=tolerance,
            residuals=residual_history,
            accel="cg",
        )
        iterations = max(len(residual_history) - 1, 1)
    else:
        coefficients = scipy.sparse.linalg.spsolve(
            matrix.tocsr(), right_hand_side.array
        )
        iterations = 1
    solve_seconds = time.perf_counter() - started
    memory.sample()

    solution = fem.Function(problem.function_space, dtype=np.float64)
    solution.x.array[:] = coefficients
    solution.x.scatter_forward()

    algebraic_residual = matrix @ coefficients - right_hand_side.array
    relative_residual = float(
        np.linalg.norm(algebraic_residual)
        / max(np.linalg.norm(right_hand_side.array), 1e-30)
    )
    physical_residual = unconstrained_matrix @ coefficients - raw_rhs
    reaction = physical_residual.reshape(-1, 2)[problem.clamp_dofs].sum(axis=0)
    applied_force = raw_rhs.reshape(-1, 2).sum(axis=0)

    stress_summary = _stress_summary(problem, solution)
    memory.sample()
    return SolvedCase(
        solution=solution,
        relative_residual=relative_residual,
        reaction=np.asarray(reaction, dtype=np.float64),
        applied_force=np.asarray(applied_force, dtype=np.float64),
        solve_seconds=float(solve_seconds),
        iterations=iterations,
        observed_peak_rss_mb=memory.peak_rss_mb,
        stress_summary=stress_summary,
    )


class _MemoryTracker:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self.peak_rss_mb = 0.0
        self.sample()

    def sample(self) -> None:
        rss_mb = self._process.memory_info().rss / (1024.0 * 1024.0)
        self.peak_rss_mb = max(self.peak_rss_mb, float(rss_mb))


def _stress_summary(problem: Any, solution: Any) -> dict[str, float]:
    strain = problem.strain(solution)
    stress = problem.stress(solution)
    von_mises = ufl.sqrt(
        stress[0, 0] ** 2
        - stress[0, 0] * stress[1, 1]
        + stress[1, 1] ** 2
        + 3.0 * stress[0, 1] ** 2
    )
    diagnostics = ufl.as_vector(
        (
            strain[0, 0],
            strain[1, 1],
            strain[0, 1],
            stress[0, 0],
            stress[1, 1],
            stress[0, 1],
            von_mises,
        )
    )
    space = fem.functionspace(problem.domain, ("DG", 0, (7,)))
    field = fem.Function(space, dtype=np.float64)
    field.interpolate(fem.Expression(diagnostics, space.element.interpolation_points))
    values = field.x.array.reshape(-1, 7)
    labels = (
        "strain_xx",
        "strain_yy",
        "strain_xy",
        "stress_xx",
        "stress_yy",
        "stress_xy",
        "von_mises",
    )
    summary: dict[str, float] = {}
    for index, label in enumerate(labels):
        summary[f"{label}_min"] = float(np.min(values[:, index]))
        summary[f"{label}_max"] = float(np.max(values[:, index]))
        summary[f"{label}_p95"] = float(np.percentile(values[:, index], 95.0))
    if not all(math.isfinite(value) for value in summary.values()):
        raise RuntimeError("应变与应力诊断包含非有限值")
    return summary
