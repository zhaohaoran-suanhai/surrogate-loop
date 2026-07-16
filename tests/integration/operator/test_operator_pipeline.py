import json
from pathlib import Path

from surrogate_loop.operator.pipeline import run_operator_pipeline

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/heat_operator_tiny.json"


def test_operator_pipeline_writes_auditable_artifacts(tmp_path) -> None:
    result = run_operator_pipeline(CONFIG, tmp_path, "训练一维热传导神经算子")

    assert result.status == "accepted"
    required = {
        "request.json",
        "spec.json",
        "dataset.npz",
        "split.json",
        "normalization.json",
        "solver_metrics.json",
        "pod_gpr.joblib",
        "pod_metrics.json",
        "deeponet_state.pt",
        "network.json",
        "training_history.json",
        "test_metrics.json",
        "field_comparison.png",
        "manifest.json",
        "model_card.md",
    }
    assert required <= {path.name for path in result.run_dir.iterdir()}
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "accepted"
    assert manifest["problem"] == "heat_1d_operator_v1"
    assert result.deeponet_metrics["median_relative_l2"] >= 0.0
    assert result.pod_metrics["median_relative_l2"] >= 0.0
