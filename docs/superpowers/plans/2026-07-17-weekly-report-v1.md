# First Surrogate-Loop Weekly Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 `docs/周报/`，交付一份可支撑 20 分钟管理层汇报且技术证据可核查的第一期周报，并把稳定文档同步到三个闭环的当前真实状态。

**Architecture:** 采用“单一 Markdown 管理层叙事 + 同文件证据附录”。正文以二维线弹性为约 70% 的主线，通过 Mermaid、能力矩阵和验收表解释架构、核心算法、成果与边界；附录把结论映射到本地 accepted 运行的请求、数据、训练、冻结、封存和验收产物。`runs/` 路径使用代码格式而非 Markdown 链接，确保全新克隆的链接测试不会依赖被 Git 忽略的本地数据。

**Tech Stack:** Markdown、Mermaid、Python 3.11、pytest、Ruff、现有 `surrogate-loop` CLI

## Global Constraints

- 只创建或修改 Markdown 和文档导航测试，不修改模型、训练、求解器、验收或推理代码。
- 不新增 PDE，不重新运行 Full，不重新消费 sealed-test。
- 不创建 PowerPoint、HTML、PDF、Word 或新图片资产。
- 第一周报日期固定为 `2026-07-17`，文件名固定为 `2026-07-17-第01期-代理模型训练闭环周报.md`。
- 汇报面向项目/管理层，时长约 20 分钟，二维线弹性约占正文 70%。
- 关键数字以本地严格可读的 accepted 运行产物为最高事实源。
- `runs/` 不提交 Git；本地证据路径在周报中使用反引号，不创建会在全新克隆中断裂的 Markdown 链接。
- accepted 只表示冻结合同下的域内确认性验收，不表述为生产认证、实验验证或域外泛化保证。
- 历史设计和实施计划不改写；只同步稳定状态文档、演示文档和当前算例说明。
- 所有用户可读内容使用中文；代码标识符、JSON 字段和命令保持英文。

---

## File Map

### Create

- `docs/周报/README.md`：周报定位、命名、索引和证据引用规则。
- `docs/周报/2026-07-17-第01期-代理模型训练闭环周报.md`：第一期 20 分钟管理层汇报及证据附录。

### Modify

- `tests/unit/test_documentation_navigation.py`：保护周报入口、必需章节、Full accepted 状态和稳定文档一致性。
- `AGENTS.md`：把二维弹性证据状态更新为 Full accepted，同时保留新 Full 需明确授权的规则。
- `README.md`：更新第三闭环状态并链接周报。
- `docs/README.md`：增加周报入口。
- `docs/当前能力与状态.md`：更新能力矩阵、Full 指标、证据等级和最近验证。
- `docs/demos/README.md`：把推荐主线切换为 accepted Full 快速展示并链接第一期周报。
- `docs/demos/二维线弹性演示手册.md`：修正“未完成 Full”事实和快速展示路径。
- `docs/guides/二维线弹性闭环操作指南.md`：追加 Full 实测检查点，不覆盖 Smoke 历史。
- `examples/elasticity_2d_cantilever/README.md`：记录 Full accepted，同时保留 Smoke 架构改进对照。

### Evidence Only, Do Not Modify

- `runs/20260716T093821Z-372c012b/`
- `runs/heat-20260716T125250Z-7f1290c7/`
- `runs/elasticity-smoke-1d3a38f34fab/`
- `runs/elasticity-full-a12ace8d157e/`
- `runs/elasticity-full-ba8ff8e584d9/`
- `examples/forced_reaction_scalar/full.json`
- `examples/heat_1d_operator/full.json`
- `examples/elasticity_2d_cantilever/full.json`

---

### Task 1: 建立周报契约、目录和第一期正文

**Files:**
- Create: `docs/周报/README.md`
- Create: `docs/周报/2026-07-17-第01期-代理模型训练闭环周报.md`
- Modify: `docs/README.md`
- Modify: `tests/unit/test_documentation_navigation.py`

**Interfaces:**
- Consumes: 已批准设计 `docs/superpowers/specs/2026-07-17-weekly-report-v1-design.md`；三个 accepted 运行的保存指标。
- Produces: 稳定周报入口和单文件汇报正文，供后续状态文档链接。

- [ ] **Step 1: 先增加周报导航和内容契约测试**

在 `test_agent_documentation_entrypoints_exist` 的 `expected` 中加入：

```python
"docs/周报/README.md",
"docs/周报/2026-07-17-第01期-代理模型训练闭环周报.md",
```

新增：

```python
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
```

在 `test_document_map_routes_four_agent_tasks` 中补充：

```python
assert "周报" in content
```

- [ ] **Step 2: 运行测试并确认先失败**

Run:

```powershell
uv run pytest tests/unit/test_documentation_navigation.py -q
```

Expected: FAIL，原因包括 `docs/周报/README.md` 和第一期周报不存在，或 `docs/README.md` 尚无“周报”。

- [ ] **Step 3: 创建周报索引**

`docs/周报/README.md` 使用以下实际结构：

```markdown
# 项目周报

本目录保存代理模型训练闭环的周期性工作汇报。周报面向项目/管理层，正文可以独立阅读，技术结论同时提供本地运行产物和稳定文档证据。

## 命名规则

`YYYY-MM-DD-第NN期-主题.md`

## 证据引用规则

- Git 中存在的文档、配置和代码使用相对 Markdown 链接。
- 被 Git 忽略的 `runs/` 产物使用反引号路径；关键数字必须同时写入正文。
- accepted 表示当前闭环定义下的域内验收，不代表生产认证或域外保证。

## 周报索引

- [第 01 期：代理模型训练闭环](2026-07-17-第01期-代理模型训练闭环周报.md)
```

- [ ] **Step 4: 创建第一期周报，写入完整管理层叙事**

报告必须按以下顺序包含真实内容：

```markdown
# 代理模型训练闭环项目周报（第 01 期）

## Executive Summary
## 本周工作与项目目标
## 基本架构：从受控需求到可信推理
## 三个代理模型闭环形成能力递进
## 核心算法：把物理结构写进数据和模型
## 二维线弹性成为首个外部有限元 Full 闭环
## Full 封存验收全部通过
## 当前仓库的能力边界
## 20 分钟演示安排
## 下一阶段建议与待确认问题
## 证据链附录
## Caveats and Assumptions
```

正文必须包含以下 Mermaid：

```mermaid
flowchart LR
    A["自然语言需求与人工确认"] --> B["白名单配置"]
    B --> C["数值求解与数据门禁"]
    C --> D["代理模型训练与验证选模"]
    D --> E["模型冻结与分级验收"]
    E --> F["摘要保护的域内推理"]
```

三个闭环表必须使用以下事实：

| 闭环 | 模型 | 最高证据 | 代表结果 |
|---|---|---|---|
| 强迫反应标量 ODE | PRS/GPR/MLP 选模，最终 GPR | accepted | 测试 NRMSE `1.08e-6` |
| 一维热传导 | DeepONet，POD/GPR 基线 | Full accepted | 中位/P95/最差 `0.731%/1.462%/3.013%` |
| 二维线弹性 | `directional_linear_v2` Vector DeepONet | Full accepted | 中位/P95/最差 `0.2519%/1.5152%/4.4492%` |

二维线弹性 Full 表必须逐项列出：

| 指标 | 实测 | 门槛 |
|---|---:|---:|
| 全场相对 L2 中位 | 0.2519% | ≤ 3% |
| 全场相对 L2 P95 | 1.5152% | ≤ 8% |
| 全场最差误差 | 4.4492% | ≤ 15% |
| 自由端误差 P95 | 0.7550% | ≤ 8% |
| 柔度误差 P95 | 3.2833% | ≤ 10% |
| 固支边界误差 | 0 | ≤ `1e-7` |
| CPU 加速 | 931.18× | ≥ 100× |

核心算法段必须解释：

- 标量验证集选模；
- Crank–Nicolson、解析解、POD/GPR 和 DeepONet；
- FEniCSx、PyAMG 和物理门禁；
- `directional_linear_v2` 的 `(nu,y0,w)` Branch、`(x,y)` Trunk、四通道方向响应、`cos(theta)`/`sin(theta)` 叠加、`P/E` 缩放和固支乘子；
- 训练集归一化、验证选模、冻结后一次性 sealed-test。

能力边界必须区分“已经具备”和“尚不具备”，至少覆盖任意 PDE、任意几何、三维/非线性、域外保证、实验验证、生产部署、带外签名和网络应力精度。

证据附录必须包含以下固定路径和作用，路径使用反引号：

```text
runs/elasticity-full-ba8ff8e584d9/request.json
runs/elasticity-full-ba8ff8e584d9/solver_job.json
runs/elasticity-full-ba8ff8e584d9/solver_output/datasets/dataset_manifest.json
runs/elasticity-full-ba8ff8e584d9/solver_output/diagnostics/solver_quality.json
runs/elasticity-full-ba8ff8e584d9/training_candidates.json
runs/elasticity-full-ba8ff8e584d9/freeze_manifest.json
runs/elasticity-full-ba8ff8e584d9/sealed_test_summary.json
runs/elasticity-full-ba8ff8e584d9/acceptance_stage.json
runs/elasticity-full-ba8ff8e584d9/acceptance.json
```

证据摘要表至少记录：

- 请求身份：`ba8ff8e584d965255f010b1b15ecd054feb9fc895cd2b420e66265e71bd53371`；
- development SHA-256：`53a07ea6ab98221520ab097a2946e997b38fc438f917c526b3bc47f886fd8c85`；
- sealed-test SHA-256：`c55cacdd43fcb9049d54e170422698dfe8590619cc69c54e89f020dbda85acd2`；
- freeze manifest SHA-256：`ff3aa6cbcf218fb84b4605505c6bc022138641a86548ce7b77db96eefe053ae9`；
- sealed summary SHA-256：`3c679ce2f995c4b6a7d4104bc795ae1286a0c2e75f6fe1722015397825cc3fe1`；
- acceptance SHA-256：`d1394a37508925ed5647d551ece2043327366d4844021a711903f7cd85efb63d`。

演示脚本按 `3/3/6/5/3` 分钟组织，并提供三个 accepted 点预测命令；强调现场不重新训练 Full。

- [ ] **Step 5: 在文档地图加入周报入口**

在 `docs/README.md` 的稳定文档区增加：

```markdown
- [项目周报](周报/README.md)
- [第 01 期代理模型训练闭环周报](周报/2026-07-17-第01期-代理模型训练闭环周报.md)
```

在“演示”路由中说明管理层工作汇报优先读取第一期周报。

- [ ] **Step 6: 运行周报契约与链接测试**

Run:

```powershell
uv run pytest tests/unit/test_documentation_navigation.py -q
```

Expected: 全部文档导航测试 PASS；Task 2 尚未修改旧状态断言，因此现有旧状态合同仍暂时成立。

- [ ] **Step 7: 提交周报主体**

```powershell
git add docs/周报/README.md docs/周报/2026-07-17-第01期-代理模型训练闭环周报.md docs/README.md tests/unit/test_documentation_navigation.py
git commit -m "docs: add first surrogate-loop weekly report"
```

---

### Task 2: 同步仓库级能力状态

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/当前能力与状态.md`
- Modify: `tests/unit/test_documentation_navigation.py`

**Interfaces:**
- Consumes: Task 1 的周报路径；二维 Full accepted 运行 `elasticity-full-ba8ff8e584d9`。
- Produces: 新 Agent、仓库首页和状态页使用同一最高证据结论。

- [ ] **Step 1: 先把状态测试改成 Full accepted 合同**

将 `test_capability_status_distinguishes_evidence_levels` 的末尾断言替换为：

```python
assert "elasticity-full-ba8ff8e584d9" in content
assert "二维线弹性 Full" in content
assert "二维线弹性当前未完成 Full" not in content
```

在 `test_root_readme_links_agent_entries` 中增加：

```python
assert "docs/周报/2026-07-17-第01期-代理模型训练闭环周报.md" in content
assert "当前未完成二维线弹性 Full" not in content
```

新增：

```python
def test_agent_rules_record_current_elasticity_full_evidence() -> None:
    content = _read("AGENTS.md")
    assert "二维线弹性 Full" in content
    assert "accepted" in content
    assert "未完成 Full" not in content
```

- [ ] **Step 2: 运行状态测试并确认先失败**

Run:

```powershell
uv run pytest tests/unit/test_documentation_navigation.py::test_capability_status_distinguishes_evidence_levels tests/unit/test_documentation_navigation.py::test_root_readme_links_agent_entries tests/unit/test_documentation_navigation.py::test_agent_rules_record_current_elasticity_full_evidence -q
```

Expected: FAIL，因为三个稳定文件仍保留二维 Full 未完成的旧事实。

- [ ] **Step 3: 更新 AGENTS 当前证据和授权规则**

把证据规则更新为：

```markdown
- 二维线弹性已完成 calibration、Smoke 和一次 Full 确认性验收；可信运行 `runs/elasticity-full-ba8ff8e584d9/` 状态为 `accepted`。
- 已有 accepted Full 可以只读报告和域内推理；重新训练、新建 Full 身份或再次消费新的 sealed-test 仍须用户明确确认。
- 不放宽门槛、不绕过摘要、不把失败样本静默删除。
```

- [ ] **Step 4: 更新根 README**

把二维线弹性段落更新为 Full accepted，写入：

- 736 个新 FEniCSx 样本；
- `directional_linear_v2`；
- 运行 ID `elasticity-full-ba8ff8e584d9`；
- 中位/P95/最差 `0.2519%/1.5152%/4.4492%`；
- 约 `931×` CPU 加速；
- 正常可信推理入口已经验证。

在 Agent 接管与演示区链接：

```markdown
[第 01 期代理模型训练闭环周报](docs/周报/2026-07-17-第01期-代理模型训练闭环周报.md)
```

- [ ] **Step 5: 重写当前能力与状态的过期段落**

保持原证据等级表，但将能力矩阵中的二维行改为：

| 闭环 | 当前最高证据 | 可可信推理 |
|---|---|---|
| 二维线弹性 | `accepted` Full | 是，限 accepted Full、训练参数域和完整摘要 |

在二维主线中保留 calibration、原始 Smoke 和同数据架构改进历史，并追加 Full：

- 512/96/128 数据划分；
- 选中 seed `20260718`；
- 0.2519%/1.5152%/4.4492%；
- 自由端 0.7550%、柔度 3.2833%、固支 0；
- 931.18×；
- 总墙钟约 1 小时 16 分 38 秒；
- 状态 `accepted`。

明确首次 `a12ace8d157e` 是缓存权限失败审计，不包含正式样本或 sealed-test 消费。

- [ ] **Step 6: 运行状态测试**

Run:

```powershell
uv run pytest tests/unit/test_documentation_navigation.py::test_capability_status_distinguishes_evidence_levels tests/unit/test_documentation_navigation.py::test_root_readme_links_agent_entries tests/unit/test_documentation_navigation.py::test_agent_rules_record_current_elasticity_full_evidence -q
```

Expected: `3 passed`。

- [ ] **Step 7: 提交仓库级状态同步**

```powershell
git add AGENTS.md README.md docs/当前能力与状态.md tests/unit/test_documentation_navigation.py
git commit -m "docs: record elasticity full acceptance"
```

---

### Task 3: 对齐演示、操作指南和算例说明

**Files:**
- Modify: `docs/demos/README.md`
- Modify: `docs/demos/二维线弹性演示手册.md`
- Modify: `docs/guides/二维线弹性闭环操作指南.md`
- Modify: `examples/elasticity_2d_cantilever/README.md`
- Modify: `tests/unit/test_documentation_navigation.py`

**Interfaces:**
- Consumes: Task 1 周报和 Task 2 当前状态页。
- Produces: 快速演示、详细操作和算例入口对 Full accepted 使用统一措辞。

- [ ] **Step 1: 先更新演示测试合同**

在 `test_elasticity_demo_has_two_modes_formulas_and_six_stages` 中把旧断言替换为：

```python
assert "elasticity-full-ba8ff8e584d9" in content
assert "accepted" in content
assert "二维线弹性当前未完成 Full" not in content
```

在 `test_demo_index_selects_elasticity_as_main_story` 中增加：

```python
assert "../周报/2026-07-17-第01期-代理模型训练闭环周报.md" in content
assert "accepted Full" in content
```

新增：

```python
def test_elasticity_guide_and_example_record_full_acceptance() -> None:
    guide = _read("docs/guides/二维线弹性闭环操作指南.md")
    example = _read("examples/elasticity_2d_cantilever/README.md")
    for content in (guide, example):
        assert "elasticity-full-ba8ff8e584d9" in content
        assert "accepted" in content
    assert "当前未完成二维线弹性 Full" not in guide
```

- [ ] **Step 2: 运行演示测试并确认先失败**

Run:

```powershell
uv run pytest tests/unit/test_documentation_navigation.py::test_elasticity_demo_has_two_modes_formulas_and_six_stages tests/unit/test_documentation_navigation.py::test_demo_index_selects_elasticity_as_main_story tests/unit/test_documentation_navigation.py::test_elasticity_guide_and_example_record_full_acceptance -q
```

Expected: FAIL，因为演示和算例文档仍声明最高证据是 Smoke。

- [ ] **Step 3: 更新演示索引**

`docs/demos/README.md` 的推荐主线改为：

- 默认快速展示 accepted Full；
- 第一阅读入口是第一期周报；
- 若本地运行存在，再展示 `acceptance.json` 和执行 accepted 推理；
- 从头运行只默认建议 Smoke；重新执行 Full 仍需新授权。

加入链接：

```markdown
[第一期代理模型训练闭环周报](../周报/2026-07-17-第01期-代理模型训练闭环周报.md)
```

- [ ] **Step 4: 更新二维线弹性演示手册**

保留快速展示/从头运行双模式和六阶段解释，但：

- 快速展示使用 `runs/elasticity-full-ba8ff8e584d9/`；
- 报告命令指向该 accepted Full；
- 增加 accepted 点预测命令；
- Smoke 仍用于解释开发与架构改进；
- Full 一次性封存过程作为已完成证据讲解，不在现场重跑；
- 删除“当前未完成 Full”的旧结论。

- [ ] **Step 5: 在操作指南追加 Full 实测检查点**

在现有 Smoke 历史之后新增“Full 一次性封存验收”小节，记录：

- 成功运行 `elasticity-full-ba8ff8e584d9`；
- 首次权限失败 `elasticity-full-a12ace8d157e` 的边界；
- 736 个新样本和 512/96/128 划分；
- 三种子耗时与选中 seed；
- 七项指标和门槛；
- accepted 后可信推理命令；
- 重新执行新 Full 仍需明确确认。

- [ ] **Step 6: 更新算例 README**

保留同数据 Smoke 架构改进数字，并紧接着说明该架构已在全新 Full 封存集通过确认性验收。不要把 Smoke 改写成 Full，也不要删除开发过程。

- [ ] **Step 7: 运行演示与链接测试**

Run:

```powershell
uv run pytest tests/unit/test_documentation_navigation.py -q
```

Expected: 全部文档导航测试 PASS。

- [ ] **Step 8: 提交演示资料同步**

```powershell
git add docs/demos/README.md docs/demos/二维线弹性演示手册.md docs/guides/二维线弹性闭环操作指南.md examples/elasticity_2d_cantilever/README.md tests/unit/test_documentation_navigation.py
git commit -m "docs: align elasticity demo with full evidence"
```

---

### Task 4: 完整证据、推理和回归验证

**Files:**
- Verify only; modify a file only if a verification command exposes a task-scoped defect.

**Interfaces:**
- Consumes: Tasks 1–3 的全部 Markdown 和测试变更。
- Produces: 可交付的周报分支和新鲜验证证据。

- [ ] **Step 1: 扫描稳定文档中的过期当前状态**

Run:

```powershell
rg -n "当前未完成二维线弹性 Full|二维线弹性当前未完成 Full|尚无 accepted Full|二维线弹性当前只到 Smoke" AGENTS.md README.md docs/README.md docs/当前能力与状态.md docs/demos docs/guides/二维线弹性闭环操作指南.md examples/elasticity_2d_cantilever/README.md
```

Expected: 无匹配。历史叙述中的“当时没有执行 Full”可以保留，但不能使用现在时声称尚未完成。

- [ ] **Step 2: 严格读取二维 Full 报告**

Run:

```powershell
uv run surrogate-loop elasticity2d report --run-dir runs/elasticity-full-ba8ff8e584d9
```

Expected: exit 0，`state/status` 为 `accepted`，指标与周报一致。

- [ ] **Step 3: 验证三个 accepted 推理入口**

Run:

```powershell
uv run surrogate-loop predict --run-dir runs/20260716T093821Z-372c012b --gamma 0.35
uv run surrogate-loop operator predict --run-dir runs/heat-20260716T125250Z-7f1290c7 --alpha 0.1 --a 1.0 --b 0.1 --x 0.5 --t 0.25
uv run surrogate-loop elasticity2d predict --run-dir runs/elasticity-full-ba8ff8e584d9 --e 3 --nu 0.3 --p 0.006 --theta -1.5707963268 --y0 0.5 --w 0.12 --x 4 --y 0.5
```

Expected outputs include：

```text
"u_at_1": 0.281908366714885
"u": 0.7783732961136168
"u": [-9.54706483753398e-05, -0.53431636095047]
```

- [ ] **Step 4: 运行文档导航测试**

Run:

```powershell
uv run pytest tests/unit/test_documentation_navigation.py -q
```

Expected: 全部 PASS，`test_local_markdown_links_resolve` 无断链。

- [ ] **Step 5: 运行 Ruff 和完整普通测试**

Run:

```powershell
uv run ruff check .
uv run pytest -q
```

Expected: Ruff exit 0；pytest 无失败。允许仓库既定的 FEniCSx 环境条件跳过和显式真实 E2E 跳过，但在交付中报告准确数量。

- [ ] **Step 6: 检查工作区和补丁质量**

Run:

```powershell
git diff --check
git status -sb
git log --oneline --decorate -5
```

Expected: `git diff --check` exit 0；工作区干净；分支为 `docs/weekly-report-v1`；提交只包含设计、计划、周报、稳定文档和文档测试。

- [ ] **Step 7: 仅在验证修复产生新变更时提交**

若 Step 1–6 暴露并修复了任务范围内问题：

```powershell
git add AGENTS.md README.md docs/README.md docs/当前能力与状态.md docs/demos/README.md docs/demos/二维线弹性演示手册.md docs/guides/二维线弹性闭环操作指南.md docs/周报/README.md docs/周报/2026-07-17-第01期-代理模型训练闭环周报.md examples/elasticity_2d_cantilever/README.md tests/unit/test_documentation_navigation.py
git commit -m "docs: finalize first weekly report"
```

若没有新变更，不创建空提交。
