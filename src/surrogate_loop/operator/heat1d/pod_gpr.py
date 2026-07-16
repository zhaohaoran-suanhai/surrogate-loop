from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler


class PodGprBaseline:
    def __init__(self, energy_threshold: float, max_components: int, seed: int) -> None:
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("POD 累计能量阈值必须位于 (0, 1]")
        if max_components <= 0:
            raise ValueError("POD 最大模态数必须为正数")
        self.energy_threshold = energy_threshold
        self.max_components = max_components
        self.seed = seed
        self.parameter_scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.regressors: list[GaussianProcessRegressor] = []
        self.field_shape: tuple[int, int] | None = None

    def fit(
        self,
        parameters: NDArray[np.float64],
        fields: NDArray[np.float64],
    ) -> PodGprBaseline:
        parameters = _validate_parameters(parameters)
        fields = np.asarray(fields, dtype=np.float64)
        if fields.ndim != 3 or fields.shape[0] != parameters.shape[0]:
            raise ValueError("字段形状必须为 (n_cases, nt, nx) 并与参数数量一致")
        if not np.isfinite(fields).all():
            raise ValueError("字段必须全部有限")
        flat_fields = fields.reshape(fields.shape[0], -1)
        maximum = min(self.max_components, flat_fields.shape[0], flat_fields.shape[1])
        probe = PCA(n_components=maximum, svd_solver="full").fit(flat_fields)
        cumulative_energy = np.cumsum(probe.explained_variance_ratio_)
        selected = min(
            int(np.searchsorted(cumulative_energy, self.energy_threshold) + 1),
            maximum,
        )
        self.pca = PCA(n_components=selected, svd_solver="full").fit(flat_fields)
        coefficients = self.pca.transform(flat_fields)
        self.parameter_scaler = StandardScaler().fit(parameters)
        scaled_parameters = self.parameter_scaler.transform(parameters)
        self.regressors = []
        for component in range(selected):
            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(3))
            regressor = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-10,
                optimizer=None,
                n_restarts_optimizer=0,
                normalize_y=True,
                random_state=self.seed,
            )
            regressor.fit(scaled_parameters, coefficients[:, component])
            self.regressors.append(regressor)
        self.field_shape = (fields.shape[1], fields.shape[2])
        return self

    def predict(self, parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.pca is None or self.parameter_scaler is None or self.field_shape is None:
            raise RuntimeError("POD/GPR 尚未拟合")
        parameters = _validate_parameters(parameters)
        scaled_parameters = self.parameter_scaler.transform(parameters)
        coefficients = np.column_stack(
            [regressor.predict(scaled_parameters) for regressor in self.regressors]
        )
        flat_fields = self.pca.inverse_transform(coefficients)
        return flat_fields.reshape(parameters.shape[0], *self.field_shape)

    def summary(self) -> dict[str, object]:
        if self.pca is None:
            raise RuntimeError("POD/GPR 尚未拟合")
        return {
            "components": int(self.pca.n_components_),
            "explained_energy": float(np.sum(self.pca.explained_variance_ratio_)),
            "energy_threshold": self.energy_threshold,
            "max_components": self.max_components,
        }


def _validate_parameters(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(parameters, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("参数形状必须为 (n_cases, 3)")
    if not np.isfinite(array).all():
        raise ValueError("参数必须全部有限")
    return array
