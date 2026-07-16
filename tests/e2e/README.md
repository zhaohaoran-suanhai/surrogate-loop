# 端到端测试

本目录用于子进程级闭环测试。后续测试必须从 CLI 启动，检查运行产物，并在新 Python 进程中重载模型完成预测。

单模块测试放在 `tests/unit/`，模块组合测试放在 `tests/integration/`。
