from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from surrogate_loop.operator.elasticity2d.problem import (
    elasticity_features,
    validate_parameter_array,
)


class PodRbfBaseline:
    """利用线弹性严格尺度关系构造的 POD-RBF 诊断基线。"""

    def __init__(self, energy_threshold: float, max_components: int) -> None:
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("POD 累计能量阈值必须位于 (0, 1]")
        if max_components <= 0:
            raise ValueError("POD 最大模态数必须为正数")
        self.energy_threshold = energy_threshold
        self.max_components = max_components
        self._pca: PCA | None = None
        self._feature_scaler: StandardScaler | None = None
        self._rbf: RBFInterpolator | None = None
        self._field_shape: tuple[int, int] | None = None
        self._training_samples: int | None = None

    def fit(
        self,
        parameters: NDArray[np.float64],
        fields: NDArray[np.float64],
    ) -> PodRbfBaseline:
        physical_parameters = validate_parameter_array(parameters)
        physical_fields = np.asarray(fields, dtype=np.float64)
        if (
            physical_fields.ndim != 3
            or physical_fields.shape[0] != physical_parameters.shape[0]
            or physical_fields.shape[2] != 2
        ):
            raise ValueError("字段形状必须为 (n_samples, n_points, 2) 并与参数数量一致")
        if not np.isfinite(physical_fields).all():
            raise ValueError("字段必须全部有限")

        features = elasticity_features(physical_parameters)
        if np.unique(features, axis=0).shape[0] != features.shape[0]:
            raise ValueError("POD-RBF 训练参数特征必须互异")

        modulus = physical_parameters[:, 0]
        load = physical_parameters[:, 2]
        shape_fields = physical_fields * (modulus / load)[:, None, None]
        flat_shape_fields = shape_fields.reshape(shape_fields.shape[0], -1)
        maximum = min(
            self.max_components,
            flat_shape_fields.shape[0],
            flat_shape_fields.shape[1],
        )
        probe = PCA(n_components=maximum, svd_solver="full").fit(flat_shape_fields)
        numerical_rank = int(
            np.linalg.matrix_rank(flat_shape_fields - flat_shape_fields.mean(axis=0))
        )
        available = max(1, min(maximum, numerical_rank))
        if self.energy_threshold == 1.0:
            selected = available
        else:
            cumulative_energy = np.cumsum(probe.explained_variance_ratio_[:available])
            selected = min(
                int(np.searchsorted(cumulative_energy, self.energy_threshold) + 1),
                available,
            )

        pca = PCA(n_components=selected, svd_solver="full").fit(flat_shape_fields)
        coefficients = pca.transform(flat_shape_fields)
        feature_scaler = StandardScaler().fit(features)
        scaled_features = feature_scaler.transform(features)
        try:
            rbf = RBFInterpolator(
                scaled_features,
                coefficients,
                kernel="multiquadric",
                epsilon=1.0,
                degree=0,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError("POD-RBF 特征不足以构造稳定插值器") from exc

        self._pca = pca
        self._feature_scaler = feature_scaler
        self._rbf = rbf
        self._field_shape = (physical_fields.shape[1], physical_fields.shape[2])
        self._training_samples = physical_fields.shape[0]
        return self

    def predict(self, parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        if (
            self._pca is None
            or self._feature_scaler is None
            or self._rbf is None
            or self._field_shape is None
        ):
            raise RuntimeError("POD-RBF 尚未拟合")
        physical_parameters = validate_parameter_array(parameters)
        features = elasticity_features(physical_parameters)
        coefficients = self._rbf(self._feature_scaler.transform(features))
        flat_shape_fields = self._pca.inverse_transform(coefficients)
        shape_fields = flat_shape_fields.reshape(
            physical_parameters.shape[0], *self._field_shape
        )
        physical_scale = (
            physical_parameters[:, 2] / physical_parameters[:, 0]
        )[:, None, None]
        return shape_fields * physical_scale

    def summary(self) -> dict[str, object]:
        if self._pca is None or self._field_shape is None or self._training_samples is None:
            raise RuntimeError("POD-RBF 尚未拟合")
        return {
            "components": int(self._pca.n_components_),
            "explained_energy": float(np.sum(self._pca.explained_variance_ratio_)),
            "energy_threshold": self.energy_threshold,
            "max_components": self.max_components,
            "training_samples": self._training_samples,
            "field_shape": list(self._field_shape),
        }
