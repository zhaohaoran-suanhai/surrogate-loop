# 强迫反应标量 ODE 算例

本目录承载第一个可复现算例：

\[
\frac{du}{dt}=\gamma u+0.5t,
\quad \gamma\in[-1,1],
\quad t\in[0,1],
\quad u(0)=0.
\]

第一阶段代理模型只预测 `gamma -> u(1)`。

后续任务将在本目录增加：

- `full.json`：正式规模配置；
- `smoke.json`：冒烟配置。

算例目录只保存输入配置和说明，不保存核心实现、虚拟环境或 `runs/` 产物。
