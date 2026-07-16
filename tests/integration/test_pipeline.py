import json

from surrogate_loop.pipeline import run_pipeline


def test_smoke_pipeline_writes_accepted_artifacts(tmp_path, smoke_spec_path) -> None:
    result = run_pipeline(smoke_spec_path, tmp_path, "训练论文风格的最小代理模型")

    assert result.status == "accepted"
    required = {
        "request.json",
        "spec.json",
        "dataset.npz",
        "split.json",
        "validation_metrics.json",
        "test_metrics.json",
        "model.joblib",
        "manifest.json",
        "model_card.md",
        "prediction.png",
    }
    assert required <= {path.name for path in result.run_dir.iterdir()}
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "accepted"
