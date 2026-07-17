# surrogate-loop

一个自然语言驱动、结构化约束、确定性验证的代理模型训练闭环。

## 当前闭环

### 标量代理模型

第一版只支持固定 ODE：

```text
du/dt = gamma*u + 0.5*t
gamma ∈ [-1, 1]
t ∈ [0, 1]
u(0) = 0
```

代理模型的首个预测合同是：

```text
gamma -> u(1)
```

标量闭环已经实现结构化配置校验、ODE 数据生成、候选模型训练、验证集选模、测试集验收、产物保存、重载推理和域外拒绝。

### 一维热传导神经算子

第二个闭环学习完整时空场：

```text
u_t = alpha * u_xx
u(0,t) = u(1,t) = 0
u(x,0) = A*sin(pi*x) + B*sin(2*pi*x)
(alpha, A, B) -> u(x,t)
```

该闭环使用 Crank–Nicolson 生成数值标签、解析解验证求解器、POD/GPR 建立诊断基线，并用 PyTorch DeepONet 学习参数到完整温度场的算子映射。DeepONet 必须独立通过测试集验收，基线不能替代它。

锁定的 Smoke 与 Full 配置均已完成端到端运行。Smoke 只作开发证据；Full 的确认性留出集未参与此前调参，最终状态为 `accepted`，可以通过正常入口执行域内可信推理。当前实测指标和复现说明见状态页与操作指南。

### 二维线弹性神经算子

第三个闭环使用独立 FEniCSx 0.11 环境生成二维平面应力悬臂梁数据，再由 uv/PyTorch 环境训练 Vector DeepONet：

```text
(E, nu, P, theta, y0, w) -> (u_x(x, y), u_y(x, y))
```

已实现严格配置、确定性采样、FEniCSx/PyAMG 求解和物理门禁、版本化 JSON/NPZ 协议、POD-RBF 基线、Vector DeepONet、开发评价、Full 封存状态机及可信推理保护。真实微型跨环境测试、calibration 和 Smoke 均已走通，Smoke 状态为 `development_complete`；它证明流程可运行，但不构成 Full 确认性精度验收。当前未完成二维线弹性 Full。

## 环境要求

- Windows
- Python 3.11
- uv
- 标量闭环仅需要 CPU
- 神经算子闭环使用 PyTorch 2.9.0，CUDA 12.6 优先、CPU 回退
- 二维弹性数据生成另需 Miniforge、FEniCSx 0.11 和 Visual Studio 2022 C++ Build Tools

## 快速开始

```powershell
uv sync --all-groups
uv run surrogate-loop --help
uv run ruff check .
uv run pytest
```

安装神经算子可选依赖：

```powershell
uv sync --extra operator --all-groups
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 已实现接口

以下命令构成第一个可运行闭环：

```powershell
surrogate-loop validate --config examples/forced_reaction_scalar/full.json
surrogate-loop run --config examples/forced_reaction_scalar/smoke.json --smoke
surrogate-loop run --config examples/forced_reaction_scalar/full.json
surrogate-loop report --run-dir runs/示例运行标识
surrogate-loop predict --run-dir runs/示例运行标识 --gamma 0.35
```

神经算子接口：

```powershell
surrogate-loop operator validate --config examples/heat_1d_operator/smoke.json
surrogate-loop operator run --config examples/heat_1d_operator/smoke.json --runs-dir runs --request "训练一维热传导 DeepONet"
surrogate-loop operator report --run-dir runs/示例运行标识
surrogate-loop operator predict --run-dir runs/示例运行标识 --alpha 0.1 --a 1.0 --b 0.1 --x 0.5 --t 0.25
surrogate-loop operator predict --run-dir runs/示例运行标识 --alpha 0.1 --a 1.0 --b 0.1 --nx 129 --nt 101 --output predicted_field.npz
```

二维线弹性接口：

```powershell
uv run surrogate-loop elasticity2d doctor
uv run surrogate-loop elasticity2d validate --config examples/elasticity_2d_cantilever/calibration.json
uv run surrogate-loop elasticity2d calibrate --config examples/elasticity_2d_cantilever/calibration.json --output-dir runs/elasticity-calibration
uv run surrogate-loop elasticity2d run --config examples/elasticity_2d_cantilever/smoke.json --runs-dir runs --request "训练二维悬臂梁位移场代理模型"
uv run surrogate-loop elasticity2d report --run-dir runs/示例运行标识
```

例如，用户可以先用自然语言告诉 Codex：

> 使用 gamma 在 -1 到 1 之间的强迫反应 ODE，运行冒烟训练，比较全部候选模型并预测 gamma=0.35 时的 u(1)。

Codex 将该意图映射到白名单配置 `examples/forced_reaction_scalar/smoke.json`，再调用固定 CLI；Python 程序负责全部科学计算和验收，不执行自然语言生成的代码。

同样地，二维弹性第一版由 Codex 把用户确认后的需求写成受审查的白名单 JSON。运行时 CLI 不调用 LLM、不解析任意方程，也不执行自然语言生成的代码。

## 仓库结构

- 项目文档：`docs/`
- 核心代码：`src/surrogate_loop/`
- 可复现算例：`examples/`
- 自动化测试：`tests/`
- 运行产物：`runs/`

## Agent 接管与演示

- [项目文档地图](docs/README.md)：新对话按演示、运行、诊断或开发新 PDE 选择最短阅读路径。
- [当前能力与状态](docs/当前能力与状态.md)：查看三个闭环的最高证据、当前指标与功能边界。
- [Agent 协作指南](docs/guides/Agent协作指南.md)：了解操作前说明、进度播报、完成报告和授权边界。

二维线弹性是当前推荐演示主线。快速展示和从头运行的演示入口将在 `docs/demos/` 中维护；在该入口完成前，可先阅读状态页和[二维线弹性闭环操作指南](docs/guides/二维线弹性闭环操作指南.md)。

## 文档

- [标量代理模型闭环设计](docs/2026-07-16-标量代理模型闭环设计.md)
- [一维热传导神经算子闭环设计](docs/2026-07-16-一维热传导神经算子闭环设计.md)
- [一维热传导神经算子实施计划](docs/2026-07-16-一维热传导神经算子实施计划.md)
- [二维线弹性神经算子闭环设计](docs/2026-07-16-二维线弹性神经算子闭环设计.md)
- [二维线弹性神经算子实施计划](docs/2026-07-16-二维线弹性神经算子实施计划.md)
- [仓库骨架与基础环境实施计划](docs/2026-07-16-仓库骨架与基础环境实施计划.md)
- [环境与验证指南](docs/guides/环境与验证.md)
- [标量闭环操作指南](docs/guides/标量闭环操作指南.md)
- [一维热传导闭环操作指南](docs/guides/一维热传导闭环操作指南.md)
- [二维线弹性闭环操作指南](docs/guides/二维线弹性闭环操作指南.md)
- [第一个标量 ODE 算例](examples/forced_reaction_scalar/README.md)
- [一维热传导神经算子算例](examples/heat_1d_operator/README.md)
- [二维悬臂梁线弹性神经算子算例](examples/elasticity_2d_cantilever/README.md)

## 明确不支持

当前版本不支持任意 PDE、任意二维/三维几何、非线性或三维弹性、真实 CFD 求解器、FNO、PINN/PINO、多 GPU、Web UI、运行时外部 LLM API、自动外推或生产部署。二维 PDE 目前仅支持已注册的平面应力悬臂梁模板。
