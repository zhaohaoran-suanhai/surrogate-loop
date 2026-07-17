from __future__ import annotations

import json

import numpy as np

from surrogate_loop.operator.elasticity2d.reporting import write_smoke_diagnostics
from surrogate_loop.operator.field_data import FieldDataset, sha256_file


def test_smoke_report_writes_displacement_and_fenicsx_stress_images(tmp_path) -> None:
    x, y = np.meshgrid(np.linspace(0.0, 4.0, 5), np.linspace(0.0, 1.0, 3))
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    fields = np.stack((0.01 * coordinates[:, 0], -0.02 * coordinates[:, 1]), axis=-1)
    dataset = FieldDataset(
        sample_ids=np.array(["development_test-00000-000000000000"]),
        parameters=np.array([[2.0, 0.3, 0.004, 0.0, 0.5, 0.1]]),
        coordinates=coordinates,
        fields=fields[None, ...],
        diagnostics={},
    )
    prediction = dataset.fields * 0.9
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": str(dataset.sample_ids[0]),
                        "stress_summary": {
                            "stress_xx_min": -2.0,
                            "stress_xx_max": 3.0,
                            "stress_xx_p95": 2.5,
                            "stress_yy_min": -1.0,
                            "stress_yy_max": 1.5,
                            "stress_yy_p95": 1.2,
                            "stress_xy_min": -0.5,
                            "stress_xy_max": 0.75,
                            "stress_xy_p95": 0.6,
                            "von_mises_min": 0.0,
                            "von_mises_max": 4.0,
                            "von_mises_p95": 3.2,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    hashes = write_smoke_diagnostics(tmp_path, dataset, prediction, manifest)

    assert set(hashes) == {
        "diagnostics/displacement_comparison.png",
        "diagnostics/fenicsx_stress_summary.png",
    }
    for relative, digest in hashes.items():
        path = tmp_path / relative
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert digest == sha256_file(path)
