# surrogate-loop

一个自然语言驱动、结构化约束、确定性验证的代理模型训练最小闭环。

## 当前目标

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

当前版本已经实现结构化配置校验、ODE 数据生成、候选模型训练、验证集选模、测试集验收、产物保存、重载推理和域外拒绝。

## 环境要求

- Windows
- Python 3.11
- uv
- CPU-first；第一版不使用 PyTorch、CUDA 或 GPU

## 快速开始

```powershell
uv sync --all-groups
uv run surrogate-loop --help
uv run ruff check .
uv run pytest
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
- [仓库骨架与基础环境实施计划](docs/2026-07-16-仓库骨架与基础环境实施计划.md)
- [环境与验证指南](docs/guides/环境与验证.md)
- [标量闭环操作指南](docs/guides/标量闭环操作指南.md)
- [第一个标量 ODE 算例](examples/forced_reaction_scalar/README.md)

## 明确不支持

第一版不支持任意方程、FEniCSx、DeepONet、FNO、PINN、Web UI、外部 LLM API、运行时多 Agent、自动外推或生产部署。
