# 强迫反应标量 ODE 算例

本目录承载第一个可复现算例：

\[
\frac{du}{dt}=\gamma u+0.5t,
\quad \gamma\in[-1,1],
\quad t\in[0,1],
\quad u(0)=0.
\]

第一阶段代理模型只预测 `gamma -> u(1)`。

本目录提供两份已通过 schema 校验的配置：

- `full.json`：120/40/40 个训练、验证、测试工况；
- `smoke.json`：24/8/8 个训练、验证、测试工况。

运行冒烟闭环：

```powershell
uv run surrogate-loop run --config examples/forced_reaction_scalar/smoke.json --smoke
```

算例目录只保存输入配置和说明，不保存核心实现、虚拟环境或 `runs/` 产物。
