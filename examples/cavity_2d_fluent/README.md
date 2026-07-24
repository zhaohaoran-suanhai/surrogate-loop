# 二维顶盖驱动方腔 Fluent + POD-RBF 闭环

本算例的首版目标是学习固定单位方腔在多种雷诺数下的稳态流场：

```text
输入：Re ∈ [10, 400]
输出：u(x,y)、v(x,y)、p'(x,y)
```

其中上壁速度固定为 `(u,v)=(1,0)`，其余三面为无滑移壁面，流体不可压、层流、稳态；`p'` 是逐样本去均值后的压力。几何、网格、边界条件和输出字段均已冻结，首版不支持改变方腔尺寸、顶盖速度、流体模型、瞬态条件或任意 CFD 问题。

高保真标签由本机 Fluent 生成，训练仓库只接收通过摘要、字段结构和重载审计的结果。代理模型对速度分支和压力分支分别执行 POD，再用 RBF 插值模态系数，简称 POD-RBF。

## 两个仓库的职责

- `surrogate-loop`：确定性采样、生成 Fluent 请求、验证求解器产物、POD-RBF 选模、评价、状态机和受保护推理。
- `fluent-automation`：创建 Fluent 作业、通过可信 Runner 启动求解、导出字段、重载 case/data 并形成追加式证据。

真实 Fluent 只能通过 `fluent-automation/scripts/windows/run-ansys-job.ps1 -Job ...` 启动。训练仓库不直接导入 PyFluent，也不会绕过 Runner 启动求解器。

## 四级执行模式

| 模式 | 样本 | 用途 | 证据含义 |
|---|---:|---|---|
| `vertical` | Re=100 | 检查单个真实 Fluent 求解、字段导出和重载协议 | 只证明协议切片 |
| `calibration` | 少量固定 Re | 测量成本并检查数值稳定性 | 不训练正式模型 |
| `Smoke` | 16/4/4 | 贯通数据、选模、评价和报告 | 状态固定为 `development_complete` |
| `Full` | 80/20/20 | 冻结合同下的一次性确认性验收 | 只有状态为 `accepted` 才能正常推理 |

Smoke 的测试样本是开发证据，不能替代 Full 的封存测试。Full 使用独立随机种子，封存测试只允许消费一次；失败后不能通过修改门槛把同一次运行改写为通过。

## 标准命令链

以下示例假设两个仓库分别位于同一个 GitHub 目录。每个真实求解阶段都应先说明预计会话数、许可证占用和输出目录，并获得明确授权。

### 1. 在训练仓库生成确定性请求

```powershell
uv run surrogate-loop cavity2d validate --config examples/cavity_2d_fluent/vertical.json
uv run surrogate-loop cavity2d plan --config examples/cavity_2d_fluent/vertical.json --output-dir runs/cavity2d-vertical-plan
```

### 2. 在 Fluent 自动化仓库逐步推进追加式流水线

```powershell
.\.venv\Scripts\python.exe -m fluent_automation.cases.lid_driven_cavity.pipeline `
  --mesh .\examples\cases\lid_driven_cavity.msh `
  --solver-request ..\surrogate-loop\runs\cavity2d-vertical-plan\solver-request.json `
  --pipeline-root .\runs\cavity2d-vertical `
  --results-root .\examples\results `
  --jobs-root .\examples\results\jobs
```

控制器每次只创建一个待执行 job，并把 job 路径打印到终端；它本身不会启动 Fluent。复制该路径后，实际启动统一经由：

```powershell
.\scripts\windows\run-ansys-job.ps1 -Job <控制器打印的-job.json>
```

首个 job 成功退出后，用完全相同的五个参数再次运行控制器。控制器会先验证批次证据，再依次为每个批次创建一个代表样本的独立重载 job；继续用 `run-ansys-job.ps1` 执行每个 job。所有批次的代表样本均重载成功后再次运行控制器，才会写出 v2 `pipeline-complete.json`。如果任何证据不一致，流水线停在 `failed`，不会跳过失败阶段；修复根因后可显式传 `--retry-failed-stage` 创建新的追加式尝试，旧失败证据不被覆盖。

每批最多 8 个样本。最终成功必须同时存在 `pipeline-complete.json`、字段 NPZ、求解验收摘要和新会话重载审计。

### 3. 回到训练仓库验证协议

```powershell
uv run surrogate-loop cavity2d verify-solver `
  --config examples/cavity_2d_fluent/vertical.json `
  --fluent-pipeline ..\fluent-automation\runs\cavity2d-vertical\pipeline-complete.json `
  --output-dir runs/cavity2d-vertical-verified
```

验证会把配置中的 sample ID、Re、split 和网格 SHA-256 与 Fluent 请求逐项绑定，并复算字段、case/data、设置回读、transcript、样本 acceptance 和逐批重载摘要；同时检查 Runner 退出、零残留进程、固定网格顺序、字段形状、有限值和压力零均值。新流水线固定使用 `reload_audits` 且数量必须与批次数一致；历史单数 `reload_audit` 只兼容已经封存的单批 v1 证据。

### 4. Smoke 或 Full 训练

先分别用 `cavity2d plan` 和 Fluent 流水线生成对应数据，再执行：

```powershell
uv run surrogate-loop cavity2d run `
  --config examples/cavity_2d_fluent/smoke.json `
  --fluent-pipeline ..\fluent-automation\runs\cavity2d-smoke\pipeline-complete.json `
  --runs-dir runs `
  --request "训练多 Re 二维顶盖驱动方腔 POD-RBF"

uv run surrogate-loop cavity2d report --run-dir runs/<运行标识>
```

### 5. accepted Full 推理

```powershell
uv run surrogate-loop cavity2d predict `
  --run-dir runs/<accepted-full-运行标识> `
  --re 100 `
  --output predictions/cavity-re100.npz
```

推理只接受 `Re ∈ [10, 400]`，只加载清单完整且状态为 `accepted` 的 Full 运行，输出路径也不能位于受保护运行目录内。

## 当前证据

当前代码和真实 Fluent 证据已经贯通跨进程 CLI、严格协议导入、POD-RBF 训练、产物状态、图形报告、Full 封存验收和 accepted-only 推理保护。合成 Fluent 产物仍用于快速工程回归，但以下结论均来自 Ansys Fluent 2024 R1 的实际求解。

### vertical 与 calibration

2026-07-23 完成真实运行 `cavity2d-vertical-re100-20260723-r6`：Re=100 在 1337 次迭代后满足 continuity、x-velocity、y-velocity 的 `1e-6` 残差门限，导出固定网格上 3550 个单元的有限 `u、v、p'`；case/data 在全新 Fluent 会话中独立重载通过。代理仓库复算全部摘要并核对请求、网格、样本身份、坐标、字段和 Runner 证据后状态为 `protocol_verified`。

随后 calibration 的 Re=10/100/400 全部通过，证明训练域两端和代表工况可以在冻结设置下稳定求解。vertical 与 calibration 只验证协议和数值基线，不用于正式模型验收。

### 24 工况 Smoke

Smoke 使用 16 个训练、4 个 validation 和 4 个 development-test 工况。24/24 个 Fluent 样本和三个批次的代表样本独立重载全部通过。运行 `runs/cavity2d-smoke-20260723T171122Z-161b3c09/` 状态为 `development_complete`，选择 POD energy `0.9999`、`multiquadric`、smoothing `1e-8`，速度 4 模态、压力 3 模态。

development-test 的速度相对 L2 中位/P95/最差为 `0.134% / 0.735% / 0.836%`，压力为 `0.967% / 3.121% / 3.406%`，主涡 P95 位置误差为 `0.00838`。Smoke 证明开发链完整，但不替代 Full 的一次性确认性语义。

### 120 工况 Full

Full 使用与 Smoke 无精确 Reynolds 数重叠的 120 个新工况，按 80/20/20 划分 train/validation/sealed-test。15 个求解批次的 120/120 个样本全部通过，15/15 个代表样本在全新 Fluent 会话中独立重载成功；模型、归一化和门槛冻结后才一次性消费 sealed-test。

运行 `runs/cavity2d-full-20260723T185548Z-8cb01c1e/` 最终状态为 `accepted`。选中的模型使用 POD energy `0.9999`、`multiquadric`、smoothing `1e-10`，速度 4 模态、压力 2 模态。20 个 sealed-test 工况的指标为：

| 指标 | 中位 | P95 | 最差 |
|---|---:|---:|---:|
| 速度场相对 L2 | 0.111% | 0.230% | 0.420% |
| 去均值压力场相对 L2 | 1.702% | 7.903% | 8.977% |

主涡中心 P95 位置误差为 `0`，预测字段全部有限。最差速度和压力样本均为 `full-116`、Re=390.59384112。Full 报告中的 Fluent 和代理单样本平均墙钟分别为 `32.1958 s` 与 `0.000243425 s`；历史字段名 `cpu_speedup` 实际是墙钟比，不应解释为累计 CPU time。

### Re=310.5 域内未见工况

Re=310.5 没有出现在 Full 的训练、验证或 sealed-test 样本中。独立 Fluent 求解在 1291 次迭代后收敛并通过全新会话重载；与 accepted 模型比较得到速度场误差 `0.159169%`、压力场误差 `4.045223%`、合并中心线误差 `0.329163%`，两者主涡中心均为 `(0.5690140845, 0.6190426118)`。

本机 Fluent 求解阶段为 `39.0594 s`。代理冷启动端到端 CLI 的 5 次中位数为 `1.8052 s`，约 `21.64×`；模型预加载后的 1000 次纯推理平均为 `0.1819 ms`，墙钟比约 `214688×`。这两种计时范围不同，均只代表当前机器和本次演示。

完整管理摘要和三张可分发对比图见[第 02 期二维方腔代理模型闭环周报](../../docs/周报/2026-07-24-第02期-二维方腔代理模型闭环周报.md)。本地执行细节见 `runs/cavity2d-smoke-full-20260724-execution.md`；`runs/` 不提交 Git。

Smoke/Full 的 `report/` 保留逐样本速度、压力、主涡、水平/竖直中心线、插值观测网格散度与动量诊断、最差工况和单样本耗时；同时生成安全 NPZ、速度/压力/误差对比图、流线/主涡图和中心线图。散度与动量项是统一插值观测网格上的代理场诊断，不是 Fluent 原生有限体积离散残差。
