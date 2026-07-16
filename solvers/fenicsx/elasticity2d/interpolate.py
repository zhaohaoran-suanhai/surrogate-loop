from __future__ import annotations

from typing import Any

import numpy as np
from dolfinx import geometry

from solvers.fenicsx.elasticity2d.problem import HEIGHT, LENGTH


def observation_coordinates(nx: int, ny: int) -> np.ndarray:
    if (
        isinstance(nx, bool)
        or isinstance(ny, bool)
        or not isinstance(nx, int)
        or not isinstance(ny, int)
        or nx < 2
        or ny < 2
    ):
        raise ValueError("观测网格的两个方向至少需要两个点")
    x, y = np.meshgrid(
        np.linspace(0.0, LENGTH, nx),
        np.linspace(0.0, HEIGHT, ny),
        indexing="xy",
    )
    return np.column_stack((x.ravel(), y.ravel()))


def interpolate_displacement(solution: Any, coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("插值坐标形状必须为 (n_points, 2)")
    if not np.isfinite(points).all():
        raise ValueError("插值坐标必须全部有限")
    tolerance = 1e-12
    outside = (
        (points[:, 0] < -tolerance)
        | (points[:, 0] > LENGTH + tolerance)
        | (points[:, 1] < -tolerance)
        | (points[:, 1] > HEIGHT + tolerance)
    )
    if np.any(outside):
        raise ValueError("插值坐标包含计算域外点")

    points3 = np.column_stack((points, np.zeros(points.shape[0], dtype=np.float64)))
    domain = solution.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim, padding=1e-12)
    candidates = geometry.compute_collisions_points(tree, points3)
    colliding = geometry.compute_colliding_cells(domain, candidates, points3)
    cells = np.full(points.shape[0], -1, dtype=np.int32)
    for index in range(points.shape[0]):
        links = colliding.links(index)
        if links.size:
            cells[index] = links[0]
    if np.any(cells < 0):
        raise ValueError("至少一个域内插值点未匹配到有限元单元")
    values = np.asarray(solution.eval(points3, cells), dtype=np.float64)
    if values.shape != (points.shape[0], 2) or not np.isfinite(values).all():
        raise RuntimeError("有限元位移插值结果无效")
    return values
