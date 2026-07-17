from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_agent_documentation_entrypoints_exist() -> None:
    expected = (
        "docs/README.md",
        "docs/当前能力与状态.md",
    )
    assert [path for path in expected if not (ROOT / path).is_file()] == []


def test_document_map_routes_four_agent_tasks() -> None:
    content = _read("docs/README.md")
    for required in ("演示", "运行已有闭环", "诊断", "开发新 PDE"):
        assert required in content
    assert "不要默认全文读取所有实施计划" in content


def test_capability_status_distinguishes_evidence_levels() -> None:
    content = _read("docs/当前能力与状态.md")
    for required in (
        "标量 ODE",
        "一维热传导",
        "二维线弹性",
        "development_complete",
        "accepted",
        "Smoke",
        "Full",
    ):
        assert required in content
    assert "二维线弹性当前未完成 Full" in content


def test_agent_rules_define_minimum_reading_order_and_authority() -> None:
    content = _read("AGENTS.md")
    for required in (
        "docs/README.md",
        "docs/当前能力与状态.md",
        "演示",
        "运行已有闭环",
        "诊断",
        "开发新 PDE",
        "Full",
    ):
        assert required in content


def test_agent_collaboration_guide_has_four_routes_and_progress_contract() -> None:
    content = _read("docs/guides/Agent协作指南.md")
    for required in (
        "只读接管",
        "演示",
        "运行已有闭环",
        "诊断",
        "开发新 PDE",
        "为什么执行",
        "预计成本",
        "当前证据",
        "下一步",
    ):
        assert required in content


def test_root_readme_links_agent_entries() -> None:
    content = _read("README.md")
    assert "docs/README.md" in content
    assert "docs/guides/Agent协作指南.md" in content
