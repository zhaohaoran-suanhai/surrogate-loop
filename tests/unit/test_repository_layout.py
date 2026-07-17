from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_required_repository_paths_exist() -> None:
    required_paths = (
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "docs/2026-07-16-标量代理模型闭环设计.md",
        "docs/2026-07-16-二维线弹性神经算子闭环设计.md",
        "docs/2026-07-16-二维线弹性神经算子实施计划.md",
        "docs/2026-07-16-仓库骨架与基础环境实施计划.md",
        "docs/README.md",
        "docs/当前能力与状态.md",
        "docs/architecture/README.md",
        "docs/demos/README.md",
        "docs/demos/二维线弹性演示手册.md",
        "docs/guides/Agent协作指南.md",
        "docs/guides/环境与验证.md",
        "src/surrogate_loop/__init__.py",
        "src/surrogate_loop/__main__.py",
        "src/surrogate_loop/cli.py",
        "examples/forced_reaction_scalar/README.md",
        "environments/fenicsx-0.11.yml",
        "solvers/fenicsx/elasticity2d",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "runs/.gitkeep",
    )

    missing = [path for path in required_paths if not (REPO_ROOT / path).exists()]

    assert missing == []


def test_readme_explains_repository_boundaries() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    for required_text in (
        "项目文档",
        "核心代码",
        "可复现算例",
        "自动化测试",
        "运行产物",
        "已实现接口",
    ):
        assert required_text in readme

    assert "目标接口，尚未实现" not in readme
