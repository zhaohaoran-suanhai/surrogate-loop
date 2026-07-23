from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_cavity_example_documents_scope_evidence_and_command_chain() -> None:
    content = _read("examples/cavity_2d_fluent/README.md")
    for required in (
        "Re ∈ [10, 400]",
        "POD-RBF",
        "Fluent",
        "vertical",
        "calibration",
        "Smoke",
        "Full",
        "development_complete",
        "accepted",
        "cavity2d plan",
        "lid_driven_cavity.pipeline",
        "--mesh",
        "--solver-request",
        "--pipeline-root",
        "--results-root",
        "--jobs-root",
        "run-ansys-job.ps1",
        "再次运行控制器",
        "cavity2d verify-solver",
        "cavity2d run",
        "cavity2d report",
        "cavity2d predict",
    ):
        assert required in content
    assert "合成" in content
    for evidence in (
        "cavity2d-vertical-re100-20260723-r6",
        "protocol_verified",
        "1337",
        "3550",
    ):
        assert evidence in content
    assert "尚未启动真实 Fluent" not in content


def test_root_and_document_map_link_cavity_example() -> None:
    root = _read("README.md")
    document_map = _read("docs/README.md")
    status = _read("docs/当前能力与状态.md")

    assert "examples/cavity_2d_fluent/README.md" in root
    assert "../examples/cavity_2d_fluent/README.md" in document_map
    for required in ("二维方腔驱动流", "真实 Fluent", "POD-RBF"):
        assert required in status
    assert "cavity2d-vertical-re100-20260723-r6" in status
    assert "protocol_verified" in status
    assert "尚未启动真实 Fluent" not in status
