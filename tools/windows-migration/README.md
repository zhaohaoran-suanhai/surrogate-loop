# Windows 跨机迁移工具

本目录提供一套安全引导型 PowerShell 工具，用于把 `surrogate-loop` 迁移到另一台 Windows 11 x64 + NVIDIA GPU 电脑，并验证 uv/PyTorch 与 Conda/FEniCSx 双环境。完整解释、故障处理和证据边界见[Windows 跨机迁移指南](../../docs/guides/Windows跨机迁移指南.md)。

## 支持范围

- Windows 11 x64，Windows PowerShell 5.1 或 PowerShell 7。
- uv 训练环境：Python `>=3.11,<3.12`、PyTorch 2.9.0、CUDA 12.6 wheel。
- Conda 求解环境：`surrogate-loop-fenicsx-0.11`、Python 3.12、DOLFINx 0.11、PyAMG。
- accepted 运行类型仅允许 `scalar`、`heat1d`、`elasticity2d`。
- `FullChain` 验证环境、科学测试、真实微型 FEniCSx E2E 和已有 accepted 推理，不创建新的 Full 训练身份。

## 工具不会执行的操作

工具不会自动安装 Git、uv、Miniforge、Visual Studio Build Tools、Windows SDK 或 NVIDIA 驱动；不会调用 `winget`、自动提权、修改 Execution Policy、删除或重建 Conda 环境。所有外部命令均为固定可执行文件和参数数组。

迁移包的 SHA-256 只用于发现传输或内容变化，不表示数字签名、来源认证或可信时间戳。工具拒绝覆盖已有 ZIP、sidecar、验证报告和目标运行目录。

## 源电脑：导出 accepted 运行

先创建一个仓库外部输出目录，例如 `D:\surrogate-loop-transfer`，再运行：

```powershell
& .\tools\windows-migration\Export-AcceptedRun.ps1 `
    -RunDir .\runs\elasticity-full-ba8ff8e584d9 `
    -ModelKind elasticity2d `
    -OutputDirectory D:\surrogate-loop-transfer
```

脚本先执行固定 report 和代表性 predict，确认状态为 accepted 后才生成：

- `<run-id>.surrogate-run.zip`
- `<run-id>.surrogate-run.sha256.json`

`D:\surrogate-loop-transfer` 是需替换的示例外部目录；工具不会自动创建或覆盖它。

## 目标电脑：检查与初始化

```powershell
& .\tools\windows-migration\Test-Prerequisites.ps1
& .\tools\windows-migration\Initialize-Environments.ps1 -WhatIf
& .\tools\windows-migration\Initialize-Environments.ps1
```

`-WhatIf` 只返回固定计划，不执行 uv 同步或 Conda 创建/更新。实际初始化在固定环境已存在时使用 `conda env update`，不使用 `--prune`，也不删除环境。

## 目标电脑：导入 accepted 运行

```powershell
& .\tools\windows-migration\Import-AcceptedRun.ps1 `
    -ArchivePath D:\surrogate-loop-transfer\elasticity-full-ba8ff8e584d9.surrogate-run.zip `
    -ChecksumPath D:\surrogate-loop-transfer\elasticity-full-ba8ff8e584d9.surrogate-run.sha256.json
```

导入依次验证 sidecar、ZIP 总哈希、安全路径、bundle schema、逐文件清单和 accepted 推理；全部通过后才发布到 `runs/<run-id>`。失败 staging 只会在通过所有权检查后清理。

## 分级验证

```powershell
& .\tools\windows-migration\Test-Installation.ps1 -Level Prerequisites
& .\tools\windows-migration\Test-Installation.ps1 -Level Python
& .\tools\windows-migration\Test-Installation.ps1 -Level Fenicsx
& .\tools\windows-migration\Test-Installation.ps1 `
    -Level FullChain `
    -AcceptedRunDir .\runs\elasticity-full-ba8ff8e584d9 `
    -ModelKind elasticity2d
```

`FullChain` 不会启动 calibration、Smoke、正式 Full 或 sealed-test 消费。若要把结果保存为回执，可增加 `-ReportPath .\migration-verification-full-chain.json`；已有文件会被拒绝覆盖。

## 参数与退出码速查

所有入口支持 `-Json`，JSON 模式 stdout 只输出一个结果对象。

| 脚本 | 主要参数 |
| --- | --- |
| `Test-Prerequisites.ps1` | `-Json` |
| `Initialize-Environments.ps1` | `-WhatIf`、`-Json` |
| `Test-Installation.ps1` | `-Level`、`-AcceptedRunDir`、`-ModelKind`、`-ReportPath`、`-Json` |
| `Export-AcceptedRun.ps1` | `-RunDir`、`-ModelKind`、`-OutputDirectory`、`-Json` |
| `Import-AcceptedRun.ps1` | `-ArchivePath`、`-ChecksumPath`、`-Json` |

| 退出码 | 含义 |
| ---: | --- |
| 0 | 通过、完成或已生成 `WhatIf` 计划 |
| 2 | 参数、前置条件或 accepted 状态不满足 |
| 3 | 外部环境、命令或 accepted 推理验证失败 |
| 4 | sidecar、ZIP、schema 或逐文件完整性失败 |
| 5 | 输出、报告或目标运行已存在，拒绝覆盖 |

## 完整手册

跨电脑操作顺序、人工安装清单、三档目标、故障处理和证据等级见[Windows 跨机迁移指南](../../docs/guides/Windows跨机迁移指南.md)。
