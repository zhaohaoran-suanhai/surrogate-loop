# 一维热传导神经算子算例

## 问题

本算例求解：

\[
u_t=\alpha u_{xx},\qquad u(0,t)=u(1,t)=0,
\]

\[
u(x,0)=A\sin(\pi x)+B\sin(2\pi x).
\]

参数域为 `alpha ∈ [0.05,0.2]`、`A ∈ [0.8,1.2]`、`B ∈ [-0.3,0.3]`。DeepONet 的 Branch 输入是 `(alpha,A,B)`，Trunk 输入是 `(x,t)`，输出是 `u(x,t)`。

## 配置

- `smoke.json`：65×51 网格，64/16/16 个训练、验证、测试算例，最多 1500 epoch；RTX 4060 Laptop GPU 实测训练约 94 秒。
- `full.json`：129×101 网格，512/96/128 个算例，最多 600 epoch。

测试专用小配置位于 `tests/fixtures/heat_operator_tiny.json`，它只验证工程链路，不能作为科学验收结果。

## 运行

```powershell
uv sync --extra operator --all-groups
uv run surrogate-loop operator validate --config examples/heat_1d_operator/smoke.json
uv run surrogate-loop operator run --config examples/heat_1d_operator/smoke.json --runs-dir runs --request "训练并验收一维热传导 DeepONet Smoke 闭环"
```

运行目录会保存数值数据、解析校验、POD/GPR、DeepONet `state_dict`、训练历史、测试指标、误差场、模型卡和 SHA-256 清单。

只有状态为 `accepted` 的运行能够通过正常推理接口加载；`rejected` 运行仍保留完整诊断产物。
