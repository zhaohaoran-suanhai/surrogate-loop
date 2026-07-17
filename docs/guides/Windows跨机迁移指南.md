# Windows 跨机迁移指南

本指南面向“另一台 Windows 11 x64 + NVIDIA GPU 电脑”，目标是在不自动改动系统配置的前提下，建立 `surrogate-loop` 的 uv/PyTorch 与 Conda/FEniCSx 双环境、迁移 accepted 运行，并分级核查完整链路。命令速查见[Windows 跨机迁移工具](../../tools/windows-migration/README.md)。

## 1. 先选择迁移目标

| 目标 | 需要的环境 | 可以证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| accepted 只读推理 | Git、uv、Python 3.11、PyTorch；对应 accepted bundle | 已封存运行在固定域内可重新加载、报告和推理 | 不能生成新 FEniCSx 样本，也不是重新训练验收 |
| 开发与普通测试 | 上述环境及全部 uv 开发依赖 | CLI、单元测试、普通 E2E 和代码质量门可运行 | 日常 pytest 会跳过显式真实 FEniCSx E2E |
| 完整 FEniCSx 训练链 | 再加 Miniforge、DOLFINx 0.11、PyAMG、VS Build Tools、Windows SDK | 求解环境、跨环境协议、真实微型 E2E 和 accepted 推理可验证 | `FullChain` 不等于新一次正式 Full 训练或封存验收 |

本套件覆盖第三档的环境建立与链路验证，但不会启动 calibration、Smoke 或 Full，也不会消费 sealed-test。若未来确需重新训练或创建新 Full 身份，仍须单独获得用户明确授权。

## 2. 支持基线与人工前置条件

正式支持范围固定为：

- Windows 11 x64；Windows PowerShell 5.1 或 PowerShell 7。
- NVIDIA GPU 和兼容驱动。
- uv 训练环境：Python `>=3.11,<3.12`、PyTorch `2.9.0`、官方 CUDA 12.6 wheel。
- Conda 求解环境：`surrogate-loop-fenicsx-0.11`、Python 3.12、DOLFINx 0.11、PyAMG。
- accepted 模型 allowlist：`scalar`、`heat1d`、`elasticity2d`。

请先从各项目官方安装入口人工安装并确认：

1. Git for Windows。
2. uv。
3. Miniforge。
4. Visual Studio 2022 Build Tools，勾选“使用 C++ 的桌面开发”、MSVC 和 Windows SDK。
5. NVIDIA 显卡驱动。

工具不会自动安装上述系统依赖，不调用 `winget`，不自动提权，不修改 Execution Policy，也不删除或重建已有 Conda 环境。若组织策略阻止脚本运行，可只读查看：

```powershell
Get-ExecutionPolicy -List
```

请把结果交给组织管理员处理；本指南不要求、更不会自动修改执行策略。

## 3. 源电脑：确认并导出 accepted 运行

### 3.1 选择可信运行

当前可作为迁移源的代表性 accepted 运行包括：

- 标量：`runs/20260716T093821Z-372c012b`
- 一维热传导：`runs/heat-20260716T125250Z-7f1290c7`
- 二维线弹性 Full：`runs/elasticity-full-ba8ff8e584d9`

导出入口会自动执行固定 report 和代表性 predict。若希望人工先查看二维线弹性证据，可运行：

```powershell
uv run surrogate-loop elasticity2d report --run-dir .\runs\elasticity-full-ba8ff8e584d9
uv run surrogate-loop elasticity2d predict `
    --run-dir .\runs\elasticity-full-ba8ff8e584d9 `
    --e 3 --nu 0.3 --p 0.006 --theta -1.5707963268 `
    --y0 0.5 --w 0.12 --x 4 --y 0.5
```

### 3.2 创建仓库外输出目录并导出

先人工创建外部目录；以下 `D:\surrogate-loop-transfer` 只是示例，不由工具自动创建或覆盖：

```powershell
& .\tools\windows-migration\Export-AcceptedRun.ps1 `
    -RunDir .\runs\elasticity-full-ba8ff8e584d9 `
    -ModelKind elasticity2d `
    -OutputDirectory D:\surrogate-loop-transfer
```

成功后得到 ZIP 和 checksum sidecar。bundle 内含 schema、模型类型、run id、导出提交、仓库 dirty 状态、UTC 时间和逐文件 SHA-256 清单。ZIP 与 sidecar 任一已存在时，导出都会停止而不是覆盖。

### 3.3 传输与独立核对

将 ZIP 和 sidecar 一起传输。建议通过独立渠道传递或人工核对 sidecar 中的 `archive_sha256`，但必须理解：**SHA-256 不等于数字签名**。它只能证明所校验内容没有发生未被发现的变化，不能证明发布者身份、来源真实性或可信时间戳。

## 4. 目标电脑：克隆、检查和初始化

### 4.1 使用完整 Git 克隆

全新克隆不含被忽略的 `runs/`。不要只复制一个权重文件：可信推理还需要配置、数据摘要、归一化器、模型元数据、验收报告和完整清单；单个权重文件不能通过本工具的 accepted 验证。

在仓库根目录运行只读前置检查：

```powershell
& .\tools\windows-migration\Test-Prerequisites.ps1
& .\tools\windows-migration\Test-Prerequisites.ps1 -Json
```

检查项包括 Windows 版本与架构、PowerShell、Git、uv、Conda、MSVC、Windows SDK、`nvidia-smi`、仓库必要文件和可用磁盘空间。缺项只返回 guidance，不会自动安装。

### 4.2 先查看环境计划

```powershell
& .\tools\windows-migration\Initialize-Environments.ps1 -WhatIf
& .\tools\windows-migration\Initialize-Environments.ps1 -WhatIf -Json
```

`-WhatIf` 返回固定的 uv、Conda、导入和 doctor 计划，但不执行计划命令。确认无误后执行：

```powershell
& .\tools\windows-migration\Initialize-Environments.ps1
```

工具依次固定执行 Python 3.11 pin、`uv sync --extra operator --all-groups`、Conda create 或 update、PyTorch 导入检查和 elasticity doctor。已有 Conda 环境只更新，不使用 `--prune`，不删除或重建。

## 5. 目标电脑：导入 accepted 运行

```powershell
& .\tools\windows-migration\Import-AcceptedRun.ps1 `
    -ArchivePath D:\surrogate-loop-transfer\elasticity-full-ba8ff8e584d9.surrogate-run.zip `
    -ChecksumPath D:\surrogate-loop-transfer\elasticity-full-ba8ff8e584d9.surrogate-run.sha256.json
```

导入顺序不可跳过：

1. sidecar 字段、ZIP 名称、字节数和总 SHA-256；
2. ZIP traversal、绝对路径、ADS、大小写别名和目标越界检查；
3. bundle schema、固定 `ModelKind`、run id 和逐文件清单；
4. 在 `runs/.migration-staging-*` 中逐项安全解包；
5. 逐文件大小与 SHA-256 复核；
6. 固定 accepted report 和代表性 predict；
7. 最后原子发布到 `runs/<run-id>`。

目标目录已经存在时 exit 5，工具不会合并或覆盖。导出提交与目标提交不同只产生 warning，不会跳过 accepted 验证。提交相同也不意味着逐位复现：驱动、硬件、底层数值库和缓存仍可能影响执行；本工具的目标是兼容复现和证据完整性，不承诺跨机器位级一致。

## 6. 分级验证与证据边界

`Test-Installation.ps1` 要求显式选择等级，避免无意运行较高成本验证：

| Level | 新增验证 | 证据解释 |
| --- | --- | --- |
| `Prerequisites` | 系统和仓库只读检查 | 只证明前置依赖可发现 |
| `Python` | CLI、版本、CUDA 前后向、Ruff、普通 pytest | 证明训练环境与普通自动化门可运行 |
| `Fenicsx` | 再加 doctor 和 `tests/solver/elasticity2d` | 证明固定求解环境与科学测试可运行 |
| `FullChain` | 再加显式真实微型 FEniCSx E2E、已有 accepted report/predict | 证明跨环境链和已有 accepted 推理完整，不是新的 Full 验收 |

示例：

```powershell
& .\tools\windows-migration\Test-Installation.ps1 -Level Python `
    -ReportPath .\migration-verification-python.json
& .\tools\windows-migration\Test-Installation.ps1 -Level Fenicsx `
    -ReportPath .\migration-verification-fenicsx.json
& .\tools\windows-migration\Test-Installation.ps1 `
    -Level FullChain `
    -AcceptedRunDir .\runs\elasticity-full-ba8ff8e584d9 `
    -ModelKind elasticity2d `
    -ReportPath .\migration-verification-full-chain.json
```

| 术语 | 证据等级 | 是否由迁移工具自动启动 |
| --- | --- | --- |
| 普通 pytest | 编排、保护机制和非显式求解器测试 | `Python` 及以上会运行 |
| 真实微型 FEniCSx E2E | uv 与 Conda 跨环境协议 | 仅 `FullChain` 显式运行 |
| calibration | 数值与物理门禁 | 不运行 |
| Smoke | 开发证据 | 不运行 |
| 正式 Full / sealed-test | 一次性确认性验收 | 不运行，仍需单独授权 |

## 7. 常见故障与安全处理

### `nvidia-smi` 成功但 CUDA 失败

`nvidia-smi` 只证明驱动层可以看到 GPU，不等于 PyTorch CUDA 前后向可用。以 `Test-Installation.ps1 -Level Python` 中的真实张量前后向为准，并检查 PyTorch 版本、`torch.version.cuda` 和 `torch.cuda.is_available()`。

### FFCx 缓存权限失败

常见症状是 JIT/FFCx 缓存目录拒绝访问、缓存文件由另一身份创建或临时目录权限异常。这类错误不等于求解器数值失败，也不等于代理模型失败。先确认当前用户对缓存和临时目录有权限，清理由当前用户拥有且明确定位的故障缓存，修复后以新进程、新身份重新运行 doctor、solver tests 和显式 E2E；不要通过删除失败样本绕过门禁。

### Conda 环境已存在

初始化脚本使用 `conda env update -n surrogate-loop-fenicsx-0.11`，不会运行环境删除，也不会加 `--prune`。若环境内容异常，先保存诊断并人工决定；工具不会替你破坏性重建。

### 退出码

- `0`：通过、完成或成功生成 `WhatIf` 计划。
- `2`：参数、前置条件或 accepted 状态不满足。
- `3`：外部命令、环境或 accepted 推理验证失败。
- `4`：sidecar、ZIP、schema 或逐文件完整性失败。
- `5`：目标或报告已存在，拒绝覆盖。

所有脚本支持 `-Json`；JSON 模式 stdout 只包含一个结果对象，便于留存证据。迁移 ZIP、sidecar、staging、验证回执和 `runs/` 均被 Git 忽略，不应提交到仓库。
