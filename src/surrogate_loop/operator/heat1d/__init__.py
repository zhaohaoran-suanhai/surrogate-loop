"""一维瞬态热传导科学计算与代理模型。"""

from surrogate_loop.operator.heat1d.analytical import analytical_solution
from surrogate_loop.operator.heat1d.problem import HeatParameters, make_grid
from surrogate_loop.operator.heat1d.solver import solve_case

__all__ = ["HeatParameters", "analytical_solution", "make_grid", "solve_case"]
