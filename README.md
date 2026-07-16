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

锁定的 Smoke 与 Full 配置均已在 RTX 4060 Laptop GPU 上完成端到端运行。Smoke 留出集在开发校准中用于诊断，因此只作探索性结果；此前未用于调参的 Full 确认性留出集使用 128 个算例，中位相对 L2 误差为 0.731%，状态为 `accepted`。完整指标和复现说明见操作指南。

## 环境要求

- Windows
- Python 3.11
- uv
- 标量闭环仅需要 CPU
- 神经算子闭环使用 PyTorch 2.9.0，CUDA 12.6 优先、CPU 回退

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

例如，用户可以先用自然语言告诉 Codex：

> 使用 gamma 在 -1 到 1 之间的强迫反应 ODE，运行冒烟训练，比较全部候选模型并预测 gamma=0.35 时的 u(1)。

Codex 将该意图映射到白名单配置 `examples/forced_reaction_scalar/smoke.json`，再调用固定 CLI；Python 程序负责全部科学计算和验收，不执行自然语言生成的代码。

## 仓库结构

- 项目文档：`docs/`
- 核心代码：`src/surrogate_loop/`
- 可复现算例：`examples/`
- 自动化测试：`tests/`
- 运行产物：`runs/`

## 文档

- [标量代理模型闭环设计](docs/2026-07-16-标量代理模型闭环设计.md)
- [一维热传导神经算子闭环设计](docs/2026-07-16-一维热传导神经算子闭环设计.md)
- [一维热传导神经算子实施计划](docs/2026-07-16-一维热传导神经算子实施计划.md)
- [仓库骨架与基础环境实施计划](docs/2026-07-16-仓库骨架与基础环境实施计划.md)
- [环境与验证指南](docs/guides/环境与验证.md)
- [标量闭环操作指南](docs/guides/标量闭环操作指南.md)
- [一维热传导闭环操作指南](docs/guides/一维热传导闭环操作指南.md)
- [第一个标量 ODE 算例](examples/forced_reaction_scalar/README.md)
- [一维热传导神经算子算例](examples/heat_1d_operator/README.md)

## 明确不支持

当前版本不支持任意 PDE、二维或三维 PDE、复杂几何、真实 CFD 求解器、FNO、PINN/PINO、多 GPU、Web UI、外部 LLM API、自动外推或生产部署。
