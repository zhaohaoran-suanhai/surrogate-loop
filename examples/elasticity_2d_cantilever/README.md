# 二维悬臂梁线弹性神经算子算例

该算例是仓库的第三个闭环。FEniCSx 0.11 在独立 Conda 环境中求解二维平面应力悬臂梁，PyTorch 在 uv 环境中训练 Vector DeepONet：

```text
(E, nu, P, theta, y0, w) -> (u_x(x, y), u_y(x, y))
```

梁区域固定为 `[0, 4] × [0, 1]`，左端固支，右端承受截断高斯分布载荷。网络只预测位移场；应变、应力和 von Mises 是 FEniCSx 侧的求解器诊断，不是当前神经网络的正式输出。

## 三档配置

- `calibration.json`：16 个校准样本，用于制造解、网格收敛、物理门禁和本机成本测量，不训练模型。
- `smoke.json`：96/24/24 个训练、验证、开发测试样本，用于打通和调试闭环，结果属于开发证据。
- `full.json`：512/96/128 个训练、验证、封存测试样本，用于一次性确认验收；未经再次明确确认不得启动。

三档配置都采用完整、严格、不可动态扩展的 JSON 合同。不要通过直接放宽 Full 的 `3%/8%/15%` 位移误差门槛来处理失败。

## 推荐顺序

```powershell
uv run surrogate-loop elasticity2d doctor
uv run surrogate-loop elasticity2d validate --config examples/elasticity_2d_cantilever/calibration.json
uv run surrogate-loop elasticity2d calibrate --config examples/elasticity_2d_cantilever/calibration.json --output-dir runs/elasticity-calibration
uv run surrogate-loop elasticity2d run --config examples/elasticity_2d_cantilever/smoke.json --runs-dir runs --request "训练二维悬臂梁位移场代理模型"
```

Smoke 成功后可读取报告目录。正常可信推理入口只加载 `accepted` 的 Full 运行；Smoke 是模型开发证据，不能伪装成通过封存验收的生产模型。

详细环境、产物、推理命令和排错说明见 [二维线弹性闭环操作指南](../../docs/guides/二维线弹性闭环操作指南.md)。
