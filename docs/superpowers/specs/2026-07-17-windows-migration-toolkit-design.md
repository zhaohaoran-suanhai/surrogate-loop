# Windows 11 跨机迁移工具套件设计

## 1. 背景与结论

`surrogate-loop` 的源码和环境定义不是当前电脑专属：Python 训练环境由 `.python-version`、`pyproject.toml` 和 `uv.lock` 约束，二维线弹性求解环境由 `environments/fenicsx-0.11.yml` 约束，现有 CLI 还提供 FEniCSx `doctor` 和 accepted 产物完整性校验。

但仓库目前只能称为“可人工迁移”，不能称为“迁移流程已产品化”：系统依赖需要人工拼接，缺少从新克隆开始的顺序化检查和分级验证；`runs/` 被 Git 忽略，已有 accepted 模型及证据不会随源码迁移；也没有安全的打包、导入和逐文件校验工具。

本设计新增一套位于 `tools/` 的 Windows 安全引导型迁移工具和配套中文手册，使另一台兼容的 Windows 11 + NVIDIA GPU 电脑可以：

1. 检查但不自动安装系统级依赖；
2. 创建或更新 uv/PyTorch 与 Conda/FEniCSx 双环境；
3. 分级证明普通 Python、CUDA、FEniCSx 科学测试和真实微型跨环境链路可用；
4. 安全迁移三个现有闭环的 accepted 运行目录，并在目标机重新执行报告、摘要校验和代表性域内推理；
5. 在用户另行确认成本和 sealed-test 授权后，具备执行 calibration、Smoke 和新 Full 的环境基础。

## 2. 目标与非目标

### 2.1 目标

- 正式支持 Windows 11 x64、NVIDIA GPU 和 Windows PowerShell 5.1 / PowerShell 7。
- 提供 PowerShell 脚本与 Markdown 手册，不把迁移逻辑写入模型核心包。
- 系统依赖采用安全引导：检查、解释和链接人工步骤，不提权、不静默安装。
- 环境依赖就绪后，自动执行固定、可审查的 uv 和 Conda 命令。
- 把环境验证拆成明确等级，默认验证行为不启动正式 calibration、Smoke 或 Full。
- accepted 迁移覆盖 `scalar`、`heat1d` 和 `elasticity2d`，二维线弹性作为主验收路径。
- 所有导出、导入、外部命令和文件写入都拒绝静默覆盖。
- 迁移工具支持人工可读输出和机器可读 JSON 摘要。

### 2.2 非目标

- 不在首版支持 Linux、macOS、ARM64、AMD GPU 或容器化部署。
- 不自动安装 Git、uv、Miniforge、Visual Studio Build Tools、Windows SDK 或 NVIDIA 驱动。
- 不修改系统或用户 Execution Policy，不调用 `winget`，不自动提权。
- 不删除或重建已有 Conda 环境，不自动更换驱动和 CUDA 工具链。
- 不把 `runs/`、迁移 ZIP、缓存或本地环境提交 Git。
- 不把环境验证描述为新 Full 验收，不自动创建 Full 身份或消费 sealed-test。
- 不承诺不同 GPU、驱动和求解器构建得到逐位一致的训练结果或相同性能。
- SHA-256 只证明内容一致性，不冒充带外数字签名、可信时间戳或来源认证。

## 3. 正式支持矩阵

| 项目 | 首版合同 |
|---|---|
| 操作系统 | Windows 11 x64 |
| PowerShell | Windows PowerShell 5.1 或 PowerShell 7 |
| 训练环境 | Python 3.11、uv、仓库 `uv.lock` |
| 神经算子 | PyTorch 2.9.0、官方 CUDA 12.6 wheel、NVIDIA GPU |
| 求解环境 | Miniforge、独立 Python 3.12、DOLFINx 0.11、PyAMG |
| 编译工具 | Visual Studio 2022 Build Tools、MSVC x64、Windows SDK |
| accepted 迁移 | 标量 ODE、一维热传导、二维线弹性 |
| 训练与验收 | 环境具备后可按现有指南显式运行；迁移脚本本身不触发 |

当前本机 `.venv` 文件体积约 `4.18 GiB`，二维线弹性 Full accepted 目录约 `49.88 MiB`，PyTorch CUDA wheel 首次下载约 `2.4 GiB`。这些数字只作为当前检查点，不作为目标机固定预算；手册应要求为 uv/Conda 缓存、两个环境、新样本和运行产物预留额外空间。

## 4. 目录与职责

```text
tools/
└── windows-migration/
    ├── README.md
    ├── Test-Prerequisites.ps1
    ├── Initialize-Environments.ps1
    ├── Test-Installation.ps1
    ├── Export-AcceptedRun.ps1
    ├── Import-AcceptedRun.ps1
    └── SurrogateLoopMigration.psm1

docs/
└── guides/
    └── Windows跨机迁移指南.md
```

职责边界：

- `tools/windows-migration/` 保存仓库维护者和使用者可直接运行的 Windows 迁移工具。
- `SurrogateLoopMigration.psm1` 只封装仓库根目录解析、固定外部进程调用、统一结果、ZIP 安全检查、文件清单和 SHA-256；不包含模型训练或求解逻辑。
- 五个 `.ps1` 是稳定用户入口，负责参数校验、流程编排和退出码。
- `tools/windows-migration/README.md` 提供参数速查和最短命令链。
- `docs/guides/Windows跨机迁移指南.md` 提供完整源电脑/目标电脑流程、支持矩阵、授权边界和故障处理。
- `src/surrogate_loop/`、`solvers/` 和现有科学配置不因迁移工具而改变行为。

所有脚本从 `$PSScriptRoot` 向上解析仓库根目录，不依赖调用者当前工作目录，也不写入 `tools/`。迁移包输出目录由用户显式提供；导入目标固定为仓库 `runs/`。

## 5. 公共 PowerShell 模块合同

`SurrogateLoopMigration.psm1` 提供边界清晰的内部函数：

- 解析和验证仓库根目录，至少要求 `pyproject.toml`、`uv.lock` 和 `environments/fenicsx-0.11.yml` 存在。
- 以可执行文件和参数数组调用外部命令，捕获退出码、stdout、stderr 和耗时；禁止 `Invoke-Expression` 和用户文本命令拼接。
- 创建统一结果对象，字段至少包含 `status`、`stage`、`message`、`evidence`、`exit_code` 和 `elapsed_seconds`。
- 计算文件 SHA-256，生成按相对路径排序的文件清单，并验证大小与哈希。
- 检查 ZIP 条目：拒绝绝对路径、盘符、UNC、`..`、NTFS ADS、目标根目录逃逸和大小写不敏感的重复路径。
- 把 JSON 以 UTF-8 写入用户指定位置；不依赖系统中文代码页。

用户入口默认输出阶段化中文信息。传入 `-Json` 时只在 stdout 输出单个 JSON 对象，诊断信息进入 stderr，便于自动化消费。

退出码固定为：

| 退出码 | 含义 |
|---:|---|
| 0 | 成功 |
| 2 | 前置条件或参数不满足 |
| 3 | 固定外部命令失败 |
| 4 | 迁移包路径或完整性验证失败 |
| 5 | 目标冲突或拒绝覆盖 |

## 6. 前置条件检查

`Test-Prerequisites.ps1` 是只读工具，检查：

- Windows 11、x64、PowerShell 版本；
- Git 和 uv 是否可执行；
- Conda 是否在 PATH 或现有代码支持的 Miniforge 默认路径；
- Visual Studio Installer 的 `vswhere.exe`、MSVC x64 工具与 Windows SDK 组件；
- `nvidia-smi`、可见 NVIDIA GPU、驱动版本和可用显存；
- 仓库关键文件是否存在；
- 目标磁盘剩余空间，只报告证据和建议，不以当前机器体积作为永久硬门槛。

缺失项必须给出：检测方式、检测结果、人工安装或修复建议、需要重新打开终端的提示。脚本不得下载、安装、提权或修改系统配置。

## 7. 双环境初始化

`Initialize-Environments.ps1` 先调用与前置检查相同的固定检查；关键系统依赖缺失时，在任何环境写入前停止。它实现 `SupportsShouldProcess`，支持 `-WhatIf`，实际动作固定为：

1. `uv python pin 3.11`；
2. `uv sync --extra operator --all-groups`；
3. 查询 `surrogate-loop-fenicsx-0.11` 是否存在；
4. 不存在时执行 `conda env create -f environments/fenicsx-0.11.yml`；
5. 存在时执行不带 `--prune` 的 `conda env update -n surrogate-loop-fenicsx-0.11 -f environments/fenicsx-0.11.yml`；
6. 运行最小版本导入和 `elasticity2d doctor`。

脚本不删除已有环境。若现有环境无法收敛到合同版本，应失败并在指南中给出人工备份、删除和重建步骤；不得在默认流程中自动执行这些破坏性动作。

## 8. 分级安装验证

`Test-Installation.ps1` 接受白名单参数 `-Level Prerequisites|Python|Fenicsx|FullChain`。高等级包含低等级检查：

### 8.1 `Prerequisites`

运行第 6 节全部只读检查。

### 8.2 `Python`

在前置检查之后运行：

- `uv run surrogate-loop --help` 与版本；
- PyTorch、CUDA 和 GPU 信息；
- 一个小型 CUDA 前向/反向与有限值检查；
- `uv run ruff check .`；
- `uv run pytest -q`。

### 8.3 `Fenicsx`

在 `Python` 之后运行：

- `uv run surrogate-loop elasticity2d doctor`；
- `conda run -n surrogate-loop-fenicsx-0.11 python -m pytest tests/solver/elasticity2d -v`。

### 8.4 `FullChain`

在 `Fenicsx` 之后：

- 临时设置 `SURROGATE_LOOP_RUN_FENICSX_E2E=1`，运行真实微型跨环境 E2E，并在 `finally` 中恢复调用前环境变量状态；
- 要求 `-AcceptedRunDir` 和相应 `-ModelKind`，执行固定 accepted 报告和代表性域内推理。

`FullChain` 只表示“安装和真实微型链路完整”，不能被描述为正式 Full。它不运行 calibration、Smoke、正式数据生成、训练或 sealed-test。

脚本可选 `-ReportPath` 保存环境与验证回执；未指定时不额外写文件。回执记录操作系统、PowerShell、Python、uv、PyTorch、CUDA、GPU、Conda、DOLFINx、PyAMG、SciPy、Git 提交、各阶段命令与结果，但不记录访问令牌、用户名目录之外的敏感环境变量或完整系统环境。

## 9. accepted 运行导出

`Export-AcceptedRun.ps1` 必须显式接收：

- `-RunDir`：待导出的完整运行目录；
- `-ModelKind scalar|heat1d|elasticity2d`；
- `-OutputDirectory`：迁移包输出目录。

流程：

1. 解析真实路径并确认目录存在；
2. 按 `ModelKind` 调用固定报告入口，解析 JSON 并确认状态为 `accepted`；
3. 枚举普通文件，拒绝重解析点和离开运行目录的路径；
4. 生成 bundle schema 1 清单，至少包含 `schema_version`、`model_kind`、`run_id`、`export_repo_commit`、`export_repo_dirty`、`created_at_utc`、文件相对路径、大小和 SHA-256；这里的提交只记录导出工具所在仓库状态，不冒充运行最初生成时的代码提交；
5. 创建结构固定的 ZIP：根级 `bundle.json` 与 `run/<run-id>/...`；
6. 计算 ZIP SHA-256，生成外部 `<run-id>.surrogate-run.sha256.json`；
7. 同名 ZIP 或 sidecar 已存在时返回退出码 5，不覆盖。

输出文件名固定为 `<run-id>.surrogate-run.zip` 和 `<run-id>.surrogate-run.sha256.json`。sidecar 与 ZIP 应通过独立传输渠道核对时才能增强来源信任；两者一同拷贝只提供传输错误检测。

## 10. accepted 运行导入

`Import-AcceptedRun.ps1` 接收 `-ArchivePath` 和 `-ChecksumPath`，流程为：

1. 校验 sidecar schema、文件名和 ZIP 总 SHA-256；
2. 读取 ZIP 中央目录并执行安全路径检查，确认只含一个 `bundle.json` 和一个 `run/<run-id>/` 树；
3. 解析 bundle schema，确认 `model_kind` 白名单、`run_id` 与目录一致；
4. 若最终目标 `runs/<run-id>` 已存在，立即返回退出码 5；
5. 在仓库 `runs/` 下创建工具自有的 `.migration-staging-<guid>`，安全解压并逐文件校验大小和 SHA-256；
6. 在 staging 目录对运行执行固定报告和代表性域内推理；
7. 全部通过后，把 staging 中的运行目录移动为最终 `runs/<run-id>`；
8. 成功或失败均只清理本次工具创建且路径已验证的 staging 目录，不删除任何既有运行目录或迁移包。

若 bundle 的 `export_repo_commit` 与目标仓库提交不同，导入工具给出兼容性警告但不单凭提交差异拒绝：文档变更和向后兼容代码可能不影响模型。最终准入仍以目标代码实际完成内部摘要校验、accepted 报告和固定推理为准；需要严格历史复现时，指南要求检出经过记录和评审的对应版本。

三类固定验证入口：

| `ModelKind` | 报告 | 代表性推理 |
|---|---|---|
| `scalar` | `surrogate-loop report` | `gamma=0.35` |
| `heat1d` | `surrogate-loop operator report` | `alpha=0.1,A=1,B=0.1,x=0.5,t=0.25` |
| `elasticity2d` | `surrogate-loop elasticity2d report` | `E=3,nu=0.3,P=0.006,theta=-π/2,y0=0.5,w=0.12,x=4,y=0.5` |

用户不能通过参数替换命令名、Python 模块或任意推理表达式。

## 11. 用户文档

### 11.1 工具速查

`tools/windows-migration/README.md` 包含：

- 五个脚本的用途、参数和退出码；
- 源电脑最短导出命令；
- 目标电脑最短检查、初始化、导入和 `FullChain` 验证命令；
- `-WhatIf`、`-Json` 和 `-ReportPath` 示例；
- 明确说明工具不会做的系统变更。

### 11.2 完整迁移指南

`docs/guides/Windows跨机迁移指南.md` 包含：

1. 支持矩阵和三种目标：只读 accepted 推理、开发/测试、完整 FEniCSx 训练链；
2. 源电脑检查、accepted 导出、外部介质复制和哈希独立核对；
3. 目标电脑人工安装 Git、uv、Miniforge、VS Build Tools/SDK 和 NVIDIA 驱动的官方入口与检查方法；
4. 克隆、双环境初始化、分级验证、accepted 导入和推理；
5. calibration、Smoke 和 Full 的授权与成本边界；
6. Conda 未发现、MSVC/SDK 缺失、FFCx 缓存权限、CUDA 不可用、磁盘不足、下载失败、哈希不一致和目标冲突的处理；
7. “兼容复现不等于逐位复现”、SHA-256 不等于签名、`runs/` 不随 Git 分发等 caveats。

同时更新根 `README.md`、`docs/README.md` 和 `docs/guides/环境与验证.md`，只增加清晰入口和当前支持结论，不复制完整手册。

## 12. 安全与失败语义

- 所有外部进程使用可执行文件和参数数组，禁止 shell 拼接、`Invoke-Expression` 和动态模块名。
- 路径必须解析为绝对路径后再做边界检查；导入 ZIP 在解压前验证条目。
- 不读取或打包运行目录之外的文件，不跟随重解析点。
- 不覆盖环境、ZIP、sidecar、目标运行目录或验证回执。
- 外部命令失败时保存退出码和受限长度的 stderr 摘要，停止后续步骤。
- 初始化中途失败时保留现有环境供诊断，不自动回滚或删除。
- 导入只有在传输哈希、逐文件哈希、accepted 报告和推理全部通过后才发布到最终目录。
- 任何安装验证失败都不能通过放宽科学门槛、跳过摘要或删除失败样本来转为成功。

## 13. 测试与验收

### 13.1 自动化测试

使用现有 pytest 驱动 Windows PowerShell，不引入 Pester。新增测试应覆盖：

- 所有脚本在 Windows PowerShell 5.1 下可解析，帮助和参数白名单可读；
- 从 `$PSScriptRoot` 解析仓库根目录，不依赖当前目录；
- SHA-256 文件清单生成、排序和校验；
- ZIP 绝对路径、盘符、UNC、`..`、ADS、根目录逃逸和大小写重复条目拒绝；
- 目标目录、迁移包和回执拒绝覆盖；
- `-Json` 成功与失败输出字段稳定；
- 临时目录成功和失败路径都只清理工具自建 staging；
- `ModelKind` 只映射到三组固定报告和推理参数；
- 静态禁止 `Invoke-Expression`、`winget`、自动提权和修改 Execution Policy；
- 文档地图、根 README 和环境指南链接可解析，并包含 Full 授权边界。

纯打包、路径和哈希测试使用 pytest 临时目录和合成文件，不依赖被 Git 忽略的真实 `runs/`。非 Windows 环境只允许明确跳过 PowerShell 5.1 专属测试，不能把跳过描述为 Windows 迁移能力通过。

### 13.2 本机真实验收

在当前 Windows 机器上执行：

1. `Test-Prerequisites.ps1` 人工输出与 `-Json`；
2. `Initialize-Environments.ps1 -WhatIf`，确认无系统安装和破坏性动作；
3. `Test-Installation.ps1 -Level Python`；
4. `Test-Installation.ps1 -Level Fenicsx`；
5. 使用 `runs/elasticity-full-ba8ff8e584d9/` 导出到临时外部目录；
6. 在隔离临时仓库或临时目标路径完成导入前校验、accepted 报告和代表性推理；
7. `Test-Installation.ps1 -Level FullChain`，其中真实微型 E2E 通过且二维 accepted 推理值与现有证据一致；
8. `uv run ruff check .` 和 `uv run pytest -q` 全部通过；
9. 扫描工作区，确认 ZIP、sidecar、staging、环境回执和 `runs/` 没有被加入 Git。

本次真实验收不重新运行 calibration、Smoke 或 Full，不创建新的 sealed-test 身份。

## 14. 完成标准

满足以下条件才可声明“Windows 跨机迁移套件完成”：

- 五个脚本和公共模块职责独立、参数白名单明确、安全约束有自动化测试；
- 新用户可以仅按迁移指南从全新克隆到双环境可用，不需要从历史计划拼接命令；
- 分级验证能准确区分环境、科学测试、真实微型链路和正式 Full 证据；
- 三类 accepted 运行可以完整导出，导入前后逐文件一致，目标机报告和代表性推理通过；
- 不自动安装系统软件、不自动提权、不覆盖或删除既有环境与运行目录；
- 根 README、文档地图、环境指南和工具 README 形成一致入口；
- 完整 Ruff、pytest 和本机真实迁移验收提供新鲜证据。
