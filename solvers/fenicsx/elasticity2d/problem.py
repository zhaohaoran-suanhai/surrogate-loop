from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

LENGTH = 4.0
HEIGHT = 1.0
LEFT_TAG = 1
RIGHT_TAG = 2


@dataclass(frozen=True)
class ElasticityProblem:
    domain: Any
    function_space: Any
    bilinear_form: Any
    linear_form: Any
    clamp_bc: Any
    clamp_dofs: np.ndarray
    strain: Any
    stress: Any


def build_elasticity_problem(
    parameters: np.ndarray,
    mesh_shape: tuple[int, int],
) -> ElasticityProblem:
    young_modulus, poisson_ratio, load, angle, center, width = _validate_inputs(
        parameters, mesh_shape
    )
    nx, ny = mesh_shape
    domain = mesh.create_rectangle(
        MPI.COMM_SELF,
        [np.array([0.0, 0.0]), np.array([LENGTH, HEIGHT])],
        [nx, ny],
        cell_type=mesh.CellType.triangle,
    )
    function_space = fem.functionspace(domain, ("Lagrange", 1, (2,)))

    facet_dim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(
        domain, facet_dim, lambda x: np.isclose(x[0], 0.0)
    )
    right_facets = mesh.locate_entities_boundary(
        domain, facet_dim, lambda x: np.isclose(x[0], LENGTH)
    )
    facet_indices = np.concatenate((left_facets, right_facets)).astype(np.int32)
    facet_values = np.concatenate(
        (
            np.full(left_facets.size, LEFT_TAG, dtype=np.int32),
            np.full(right_facets.size, RIGHT_TAG, dtype=np.int32),
        )
    )
    order = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(
        domain, facet_dim, facet_indices[order], facet_values[order]
    )

    clamp_dofs = fem.locate_dofs_topological(
        function_space, facet_dim, left_facets
    )
    clamp_bc = fem.dirichletbc(
        np.zeros(2, dtype=np.float64), clamp_dofs, V=function_space
    )

    trial = ufl.TrialFunction(function_space)
    test = ufl.TestFunction(function_space)

    def strain(displacement: Any) -> Any:
        return ufl.sym(ufl.grad(displacement))

    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    plane_stress_lambda = young_modulus * poisson_ratio / (1.0 - poisson_ratio**2)

    def stress(displacement: Any) -> Any:
        epsilon = strain(displacement)
        return (
            2.0 * shear_modulus * epsilon
            + plane_stress_lambda * ufl.tr(epsilon) * ufl.Identity(2)
        )

    coordinate = ufl.SpatialCoordinate(domain)
    normalizer = width * math.sqrt(math.pi / 2.0) * (
        math.erf((HEIGHT - center) / (math.sqrt(2.0) * width))
        - math.erf(-center / (math.sqrt(2.0) * width))
    )
    density = ufl.exp(-0.5 * ((coordinate[1] - center) / width) ** 2) / normalizer
    traction = load * density * ufl.as_vector((math.cos(angle), math.sin(angle)))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    bilinear_form = fem.form(
        ufl.inner(stress(trial), strain(test)) * ufl.dx,
        dtype=np.float64,
    )
    linear_form = fem.form(
        ufl.dot(traction, test) * ds(RIGHT_TAG),
        dtype=np.float64,
    )
    return ElasticityProblem(
        domain=domain,
        function_space=function_space,
        bilinear_form=bilinear_form,
        linear_form=linear_form,
        clamp_bc=clamp_bc,
        clamp_dofs=clamp_dofs,
        strain=strain,
        stress=stress,
    )


def _validate_inputs(
    parameters: np.ndarray,
    mesh_shape: tuple[int, int],
) -> tuple[float, float, float, float, float, float]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("二维线弹性参数必须是含六个有限值的一维数组")
    if (
        len(mesh_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in mesh_shape)
        or any(value <= 0 for value in mesh_shape)
    ):
        raise ValueError("有限元网格尺寸必须是两个正整数")
    young_modulus, poisson_ratio, load, angle, center, width = map(float, values)
    if young_modulus <= 0.0 or not 0.0 < poisson_ratio < 0.5:
        raise ValueError("弹性模量必须为正，泊松比必须位于 (0, 0.5)")
    if load <= 0.0 or load / young_modulus > 1e-2 + 1e-15:
        raise ValueError("载荷必须为正且满足小变形合同 P/E <= 1e-2")
    if not 0.0 <= center <= HEIGHT or width <= 0.0:
        raise ValueError("载荷中心或宽度无效")
    return young_modulus, poisson_ratio, load, angle, center, width
