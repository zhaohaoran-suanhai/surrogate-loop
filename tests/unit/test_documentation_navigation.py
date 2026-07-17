import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_agent_documentation_entrypoints_exist() -> None:
    expected = (
        "docs/README.md",
        "docs/当前能力与状态.md",
        "docs/guides/Agent协作指南.md",
        "docs/demos/README.md",
        "docs/demos/二维线弹性演示手册.md",
        "docs/demos/演示Skill内容草案.md",
        "docs/demos/Agent接管验收清单.md",
        "docs/周报/README.md",
        "docs/周报/2026-07-17-第01期-代理模型训练闭环周报.md",
    )
    assert [path for path in expected if not (ROOT / path).is_file()] == []


def test_document_map_routes_four_agent_tasks() -> None:
    content = _read("docs/README.md")
    for required in ("演示", "运行已有闭环", "诊断", "开发新 PDE"):
        assert required in content
    assert "周报" in content
    assert "不要默认全文读取所有实施计划" in content


def test_first_weekly_report_covers_management_story_and_evidence_chain() -> None:
    index = _read("docs/周报/README.md")
    report = _read("docs/周报/2026-07-17-第01期-代理模型训练闭环周报.md")
    assert "第 01 期" in index
    for required in (
        "Executive Summary",
        "基本架构",
        "三个代理模型闭环",
        "核心算法",
        "二维线弹性",
        "Full",
        "能力边界",
        "20 分钟演示",
        "证据链",
        "acceptance_stage.json",
        "acceptance.json",
        "elasticity-full-ba8ff8e584d9",
    ):
        assert required in report
    assert "0.2519%" in report
    assert "931.18" in report


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


def test_elasticity_demo_has_two_modes_formulas_and_six_stages() -> None:
    content = _read("docs/demos/二维线弹性演示手册.md")
    for required in (
        "快速展示模式",
        "从头运行模式",
        "-\\nabla\\cdot\\boldsymbol{\\sigma}",
        "\\mathcal G",
        "\\frac{P}{E}\\frac{x}{L}",
        "进度：3/6",
        "Smoke",
        "development_complete",
        "加速比",
    ):
        assert required in content
    assert "二维线弹性当前未完成 Full" in content


def test_demo_commands_match_current_elasticity_cli() -> None:
    content = _read("docs/demos/二维线弹性演示手册.md")
    commands = (
        "uv run surrogate-loop elasticity2d doctor",
        "uv run surrogate-loop elasticity2d validate --config "
        "examples/elasticity_2d_cantilever/smoke.json",
        "uv run surrogate-loop elasticity2d run --config "
        "examples/elasticity_2d_cantilever/smoke.json --runs-dir runs",
        "uv run surrogate-loop elasticity2d report --run-dir",
    )
    assert all(command in content for command in commands)


def test_demo_index_selects_elasticity_as_main_story() -> None:
    content = _read("docs/demos/README.md")
    assert "二维线弹性" in content
    assert "主线" in content
    assert "一维热传导" in content
    assert "标量 ODE" in content


def test_root_readme_links_elasticity_demo() -> None:
    assert "docs/demos/二维线弹性演示手册.md" in _read("README.md")


def test_future_skill_draft_is_content_only_and_reads_dynamic_facts() -> None:
    content = _read("docs/demos/演示Skill内容草案.md")
    for required in (
        "触发场景",
        "最小读取顺序",
        "任务路由",
        "六阶段",
        "进度播报",
        "Smoke/Full",
        "完成报告",
        "docs/当前能力与状态.md",
    ):
        assert required in content
    assert "本文件不是可安装 Skill" in content
    assert "固定实测数值" in content


def test_agent_rehearsal_covers_demo_run_and_new_pde() -> None:
    content = _read("docs/demos/Agent接管验收清单.md")
    for prompt in (
        "介绍这个仓库并演示二维线弹性的价值",
        "从头运行二维线弹性闭环，并持续告诉我进度",
        "我要扩展一个新的 PDE 代理模型",
    ):
        assert prompt in content
    for required in ("必读文档", "预期路由", "合格判据", "失败表现"):
        assert required in content


def test_local_markdown_links_resolve() -> None:
    broken: list[str] = []
    documents = [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    ]
    for document in documents:
        in_fence = False
        prose: list[str] = []
        for line in document.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if not in_fence:
                prose.append(line)
        content = "\n".join(prose)
        for raw_target in MARKDOWN_LINK.findall(content):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith("#") or "://" in target:
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            candidate = (document.parent / relative).resolve()
            if not candidate.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert broken == []
