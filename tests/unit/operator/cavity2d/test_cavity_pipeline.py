from __future__ import annotations

import json
from pathlib import Path

from surrogate_loop.operator.cavity2d.pipeline import _cpu_speedup


def test_cpu_speedup_compares_per_sample_costs(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline-complete.json"
    pipeline.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "samples": [
                            {"sample_id": "train-0", "wall_time_seconds": 10.0},
                            {"sample_id": "test-0", "wall_time_seconds": 20.0},
                            {"sample_id": "train-1", "wall_time_seconds": 30.0},
                            {"sample_id": "test-1", "wall_time_seconds": 40.0},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _cpu_speedup(
        pipeline,
        inference_seconds=2.0,
        inference_sample_count=2,
        sample_ids={"test-0", "test-1"},
    ) == 30.0
