# Windows 11 Migration Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `tools/windows-migration/` 交付一套安全引导型 PowerShell 工具和中文手册，使另一台 Windows 11 x64 + NVIDIA GPU 电脑能够建立 uv/PyTorch 与 Conda/FEniCSx 双环境、分级验证完整链路，并安全迁移三个闭环的 accepted 运行。

**Architecture:** 公共 PowerShell 模块提供仓库定位、固定命令调用、统一结果、路径安全、ZIP 与 SHA-256 原语；五个薄入口脚本分别负责编排前置检查、环境初始化、分级验证、accepted 导出和 accepted 导入。pytest 通过 Windows PowerShell 5.1 驱动模块级和脚本级测试，真实 FEniCSx 与 accepted 迁移只在最终本机验收阶段运行，不在自动化单元测试中依赖 Git 忽略的 `runs/`。

**Tech Stack:** Windows 11 x64、Windows PowerShell 5.1 / PowerShell 7、Python 3.11、uv、pytest、.NET `System.IO.Compression`、Miniforge/Conda、Python 3.12、DOLFINx 0.11、PyAMG、PyTorch 2.9.0 CUDA 12.6

## Global Constraints

- 正式支持目标固定为 Windows 11 x64、NVIDIA GPU、Windows PowerShell 5.1 或 PowerShell 7。
- Python 训练环境固定为 `>=3.11,<3.12`，通过 `.python-version`、`pyproject.toml` 和 `uv.lock` 创建。
- 神经算子依赖固定使用 `torch==2.9.0` 与官方 CUDA 12.6 wheel。
- FEniCSx 求解环境固定名为 `surrogate-loop-fenicsx-0.11`，由 `environments/fenicsx-0.11.yml` 创建，使用独立 Python 3.12、DOLFINx 0.11 和 PyAMG。
- 工具不自动安装 Git、uv、Miniforge、Visual Studio Build Tools、Windows SDK 或 NVIDIA 驱动。
- 工具不调用 `winget`、不自动提权、不修改 Execution Policy、不删除或重建已有 Conda 环境。
- 所有外部命令使用固定可执行文件和参数数组；禁止 `Invoke-Expression`、动态 Python 模块名和用户文本命令拼接。
- 所有写入拒绝静默覆盖；导入只在 ZIP 总哈希、逐文件哈希、accepted 报告和固定域内推理全部通过后发布。
- accepted 迁移只允许 `scalar`、`heat1d`、`elasticity2d` 三个 `ModelKind`。
- `FullChain` 只表示环境、科学测试、真实微型 E2E 与 imported accepted 推理完整；不得运行或暗示正式 calibration、Smoke、Full 或 sealed-test 消费。
- SHA-256 只表示传输与内容一致性，不表示带外签名、来源认证或可信时间戳。
- `runs/`、迁移 ZIP、checksum sidecar、staging 和本地验证回执不得提交 Git。
- 用户可读内容使用中文；脚本名、参数、JSON 字段、PowerShell 函数和第三方接口保持英文。

---

## File Map

### Create

- `tools/windows-migration/SurrogateLoopMigration.psm1`：公共安全原语、环境计划、验证计划、bundle 打包/解包函数。
- `tools/windows-migration/Test-Prerequisites.ps1`：只读系统依赖检查。
- `tools/windows-migration/Initialize-Environments.ps1`：支持 `-WhatIf` 的双环境创建/更新入口。
- `tools/windows-migration/Test-Installation.ps1`：`Prerequisites|Python|Fenicsx|FullChain` 分级验证入口。
- `tools/windows-migration/Export-AcceptedRun.ps1`：accepted 运行验证、打包与 sidecar 生成入口。
- `tools/windows-migration/Import-AcceptedRun.ps1`：安全校验、staging、accepted 验证与发布入口。
- `tools/windows-migration/README.md`：参数和最短命令链速查。
- `docs/guides/Windows跨机迁移指南.md`：完整源电脑/目标电脑迁移手册。
- `tests/__init__.py`：让 PowerShell 测试驱动可被稳定导入。
- `tests/tools/__init__.py`：Windows 工具测试包标志。
- `tests/tools/powershell.py`：共享 Windows PowerShell 5.1 子进程驱动。
- `tests/tools/test_windows_migration_module.py`：公共模块、SHA-256 与 ZIP 路径安全测试。
- `tests/tools/test_windows_environment_scripts.py`：前置检查、环境计划和分级验证脚本测试。
- `tests/tools/test_windows_run_transfer.py`：bundle 导出、导入、冲突和 staging 测试。

### Modify

- `.gitignore`：拒绝意外跟踪迁移 ZIP、checksum sidecar 和验证回执。
- `AGENTS.md`：增加 `tools/` 目录边界与 FullChain 授权语义。
- `README.md`：增加 Windows 跨机迁移入口与当前支持结论。
- `docs/README.md`：把迁移手册加入稳定文档与任务路由。
- `docs/guides/环境与验证.md`：从单机环境说明路由到迁移套件。
- `tests/unit/test_documentation_navigation.py`：保护迁移文档、工具 README 和本地链接。

### Evidence Only, Do Not Modify

- `runs/20260716T093821Z-372c012b/`
- `runs/heat-20260716T125250Z-7f1290c7/`
- `runs/elasticity-full-ba8ff8e584d9/`
- `environments/fenicsx-0.11.yml`
- `uv.lock`

---

### Task 1: 建立 PowerShell 测试驱动和公共安全模块

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/powershell.py`
- Create: `tests/tools/test_windows_migration_module.py`
- Create: `tools/windows-migration/SurrogateLoopMigration.psm1`

**Interfaces:**
- Consumes: 仓库根目录标志文件 `pyproject.toml`、`uv.lock`、`environments/fenicsx-0.11.yml`。
- Produces: `Get-SurrogateRepositoryRoot`, `New-MigrationResult`, `ConvertTo-MigrationJson`, `Invoke-FixedCommand`, `Get-FileManifest`, `Test-FileManifest`, `Test-SafeZipEntries`, `Write-MigrationOutput`。
- Result object fields: `status: string`, `stage: string`, `message: string`, `evidence: object`, `exit_code: int`, `elapsed_seconds: double`。

- [ ] **Step 1: 先创建 Windows PowerShell pytest 驱动和失败测试**

创建空的 `tests/__init__.py` 和 `tests/tools/__init__.py`。创建 `tests/tools/powershell.py`：

```python
from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell.exe")

def ps_quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run_powershell(code: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    prelude = (
        "$ErrorActionPreference='Stop';"
        "[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);"
    )
    encoded = base64.b64encode((prelude + code).encode("utf-16le")).decode("ascii")
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
```

创建 `tests/tools/test_windows_migration_module.py`：

```python
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from tests.tools.powershell import POWERSHELL, ps_quote, run_powershell

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/windows-migration/SurrogateLoopMigration.psm1"

pytestmark = pytest.mark.skipif(
    POWERSHELL is None,
    reason="Windows PowerShell 5.1 is required for Windows migration tool tests",
)


def test_module_resolves_repository_root_independent_of_current_directory(tmp_path: Path) -> None:
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "(Get-SurrogateRepositoryRoot).Path",
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).resolve() == ROOT


def test_migration_result_has_stable_json_schema() -> None:
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "$r=New-MigrationResult -Status 'ok' -Stage 'unit' -Message 'done' "
        "-Evidence @{value=3} -ExitCode 0 -ElapsedSeconds 1.25;"
        "$r | ConvertTo-MigrationJson"
    )
    payload = json.loads(completed.stdout)
    assert payload == {
        "elapsed_seconds": 1.25,
        "evidence": {"value": 3},
        "exit_code": 0,
        "message": "done",
        "stage": "unit",
        "status": "ok",
    }


def test_fixed_command_preserves_argument_boundaries() -> None:
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        "$r=Invoke-FixedCommand "
        f"-FilePath {ps_quote(sys.executable)} "
        "-Arguments @('-c','import json,sys; print(json.dumps(sys.argv[1:]))',"
        "'space value','semi;colon','quote\"inside') "
        f"-WorkingDirectory {ps_quote(ROOT)};"
        "$r | ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    process = json.loads(completed.stdout)
    assert process["exit_code"] == 0
    assert json.loads(process["stdout"]) == [
        "space value",
        "semi;colon",
        'quote"inside',
    ]


def test_file_manifest_is_sorted_and_detects_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "b.txt").write_text("b", encoding="utf-8")
    (run_dir / "a.txt").write_text("a", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"$m=Get-FileManifest -Root {ps_quote(run_dir)};"
        f"[IO.File]::WriteAllText({ps_quote(manifest_path)},"
        "($m | ConvertTo-Json -Depth 10),"
        "(New-Object Text.UTF8Encoding($false)));"
        f"Test-FileManifest -Root {ps_quote(run_dir)} -Files $m"
    )
    assert completed.returncode == 0, completed.stderr
    files = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["path"] for item in files] == ["a.txt", "b.txt"]

    (run_dir / "a.txt").write_text("changed", encoding="utf-8")
    failed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"$m=Get-Content {ps_quote(manifest_path)} -Raw | ConvertFrom-Json;"
        f"Test-FileManifest -Root {ps_quote(run_dir)} -Files $m"
    )
    assert failed.returncode != 0
    assert "SHA-256" in failed.stderr


@pytest.mark.parametrize(
    "entry",
    ("../escape.txt", "/absolute.txt", "C:/drive.txt", "run/item.txt:ads", "run/../escape"),
)
def test_zip_entry_validation_rejects_unsafe_paths(tmp_path: Path, entry: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(entry, "bad")
    completed = run_powershell(
        f"Import-Module {ps_quote(MODULE)} -Force;"
        f"Test-SafeZipEntries -ArchivePath {ps_quote(archive)} "
        f"-DestinationRoot {ps_quote(tmp_path / 'destination')}"
    )
    assert completed.returncode != 0
    assert "ZIP" in completed.stderr
```

- [ ] **Step 2: 运行公共模块测试并确认先失败**

Run:

```powershell
uv run pytest tests/tools/test_windows_migration_module.py -q
```

Expected: FAIL，因为 `SurrogateLoopMigration.psm1` 尚不存在。

- [ ] **Step 3: 实现公共模块最小安全原语**

创建 `tools/windows-migration/SurrogateLoopMigration.psm1`。文件顶部和仓库解析使用：

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-SurrogateRepositoryRoot {
    [CmdletBinding()]
    param()
    $root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    foreach ($required in @(
        'pyproject.toml',
        'uv.lock',
        'environments\fenicsx-0.11.yml'
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $required) -PathType Leaf)) {
            throw "仓库根目录缺少必要文件：$required"
        }
    }
    Get-Item -LiteralPath $root
}

function New-MigrationResult {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Status,
        [Parameter(Mandatory)][string]$Stage,
        [Parameter(Mandatory)][string]$Message,
        [Parameter(Mandatory)]$Evidence,
        [Parameter(Mandatory)][int]$ExitCode,
        [Parameter(Mandatory)][double]$ElapsedSeconds
    )
    [pscustomobject][ordered]@{
        status = $Status
        stage = $Stage
        message = $Message
        evidence = $Evidence
        exit_code = $ExitCode
        elapsed_seconds = $ElapsedSeconds
    }
}

function ConvertTo-MigrationJson {
    [CmdletBinding()]
    param([Parameter(Mandatory, ValueFromPipeline)]$InputObject)
    process { $InputObject | ConvertTo-Json -Depth 20 -Compress }
}
```

实现 `ConvertTo-WindowsCommandLineArgument` 和 `Invoke-FixedCommand`。前者按 Windows `CommandLineToArgvW` 规则处理空参数、空白、双引号和双引号前/结尾反斜杠：无空白/引号的非空参数原样返回；其余用双引号包围，普通反斜杠原样，双引号前连续反斜杠加倍后再加一个转义反斜杠，结尾反斜杠加倍。后者使用 `System.Diagnostics.ProcessStartInfo`，固定 `UseShellExecute = $false`、重定向 stdout/stderr、设置 UTF-8 输出与 `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`，把每个参数单独调用上述函数后以空格连接，不接受一整段命令字符串。返回对象字段固定为 `file_path`、`arguments`、`exit_code`、`stdout`、`stderr`、`elapsed_seconds`；非零退出码由调用方决定映射，函数本身不静默忽略。

文件清单函数必须先枚举全部子项并拒绝任何文件或目录 `ReparsePoint`，再只处理普通文件，把相对路径统一为 `/`，并按 `Ordinal` 排序：

```powershell
function Get-FileManifest {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Root)
    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    $prefix = $resolvedRoot.TrimEnd('\') + '\'
    $items = foreach ($file in Get-ChildItem -LiteralPath $resolvedRoot -File -Recurse) {
        if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "运行目录不能包含重解析点：$($file.FullName)"
        }
        if (-not $file.FullName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "文件离开运行目录：$($file.FullName)"
        }
        [pscustomobject][ordered]@{
            path = $file.FullName.Substring($prefix.Length).Replace('\', '/')
            bytes = [int64]$file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    @($items | Sort-Object -Property path)
}
```

`Test-FileManifest` 必须对每项重新解析到根目录内，检查文件存在、大小和 SHA-256，拒绝清单重复路径，并检查实际普通文件集合与清单完全一致。

`Test-SafeZipEntries` 使用 `System.IO.Compression.ZipFile.OpenRead` 在解压前检查：空名称、绝对路径、`^[A-Za-z]:`、`^[/\\]`、任意 `:`、路径分段 `..`、目标根逃逸，以及 `OrdinalIgnoreCase` 重复条目。安全目标路径用 `GetFullPath(Join-Path ...)` 计算并要求以目标根加目录分隔符开头。

`Write-MigrationOutput` 接受 result、`-Json` 和输出流选择；`-Json` 时 stdout 只写一个压缩 JSON，对人模式写简短中文状态。最后显式导出本任务接口：

```powershell
Export-ModuleMember -Function @(
    'Get-SurrogateRepositoryRoot',
    'New-MigrationResult',
    'ConvertTo-MigrationJson',
    'Invoke-FixedCommand',
    'Get-FileManifest',
    'Test-FileManifest',
    'Test-SafeZipEntries',
    'Write-MigrationOutput'
)
```

- [ ] **Step 4: 运行公共模块测试并修正 PowerShell 5.1 兼容性**

Run:

```powershell
uv run pytest tests/tools/test_windows_migration_module.py -q
```

Expected: 全部 PASS；不得出现未批准 verb 警告、编码异常或 PowerShell 7 专属语法。

- [ ] **Step 5: 提交公共模块检查点**

```powershell
git add tests/__init__.py tests/tools/__init__.py tests/tools/powershell.py tests/tools/test_windows_migration_module.py tools/windows-migration/SurrogateLoopMigration.psm1
git commit -m "feat: add Windows migration core module"
```

---

### Task 2: 实现系统前置检查和双环境初始化

**Files:**
- Modify: `tools/windows-migration/SurrogateLoopMigration.psm1`
- Create: `tools/windows-migration/Test-Prerequisites.ps1`
- Create: `tools/windows-migration/Initialize-Environments.ps1`
- Create: `tests/tools/test_windows_environment_scripts.py`

**Interfaces:**
- Consumes: Task 1 的 repository root、fixed command、result 和 JSON 输出函数。
- Produces: `Find-CondaExecutable`, `Get-PrerequisiteReport`, `Get-EnvironmentPlan`。
- `Get-PrerequisiteReport() -> PSCustomObject`：`status`, `checks`, `summary`。
- `Get-EnvironmentPlan([bool]$CondaEnvironmentExists, [string]$UvPath='uv', [string]$CondaPath='conda', [string]$RepositoryRoot=(Get-SurrogateRepositoryRoot).Path) -> object[]`：每项含 `name`, `file_path`, `arguments`, `working_directory`；纯计划生成不探测可执行文件，入口脚本在 prerequisite 通过后传入真实路径。

- [ ] **Step 1: 先写环境计划和安全边界失败测试**

创建 `tests/tools/test_windows_environment_scripts.py`，文件顶部使用共享驱动：

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.tools.powershell import (
    POWERSHELL,
    ps_quote as _ps_quote,
    run_powershell as _run_powershell,
)

pytestmark = pytest.mark.skipif(
    POWERSHELL is None,
    reason="Windows PowerShell 5.1 is required for Windows migration tool tests",
)
```

随后加入：

```python
ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/windows-migration/SurrogateLoopMigration.psm1"
TOOLS = ROOT / "tools/windows-migration"


def _environment_plan(exists: bool) -> list[dict[str, object]]:
    literal = "$true" if exists else "$false"
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Get-EnvironmentPlan -CondaEnvironmentExists {literal} | "
        "ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    return payload if isinstance(payload, list) else [payload]


def test_environment_plan_creates_or_updates_without_prune() -> None:
    create = _environment_plan(False)
    update = _environment_plan(True)
    assert [item["name"] for item in create[:2]] == ["uv-python", "uv-sync"]
    assert create[2]["arguments"][:2] == ["env", "create"]
    assert update[2]["arguments"][:4] == [
        "env", "update", "-n", "surrogate-loop-fenicsx-0.11"
    ]
    assert "--prune" not in update[2]["arguments"]


def test_prerequisite_json_always_has_actionable_schema() -> None:
    script = TOOLS / "Test-Prerequisites.ps1"
    completed = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-File", script, "-Json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode in {0, 2}
    payload = json.loads(completed.stdout)
    assert payload["stage"] == "prerequisites"
    assert payload["exit_code"] == completed.returncode
    assert isinstance(payload["evidence"]["checks"], list)
    for check in payload["evidence"]["checks"]:
        assert {"name", "status", "evidence", "guidance"} <= set(check)


def test_environment_scripts_do_not_install_or_escalate() -> None:
    forbidden = (
        "Invoke-Expression",
        "winget ",
        "-Verb RunAs",
        "Set-ExecutionPolicy",
        "conda env remove",
        "Remove-Item",
    )
    content = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            TOOLS / "Test-Prerequisites.ps1",
            TOOLS / "Initialize-Environments.ps1",
        )
    )
    assert all(token not in content for token in forbidden)
    assert "SupportsShouldProcess" in content
```

- [ ] **Step 2: 运行环境脚本测试并确认先失败**

```powershell
uv run pytest tests/tools/test_windows_environment_scripts.py -q
```

Expected: FAIL，因为两个脚本和环境函数尚不存在。

- [ ] **Step 3: 在公共模块实现前置探测和环境命令计划**

新增 `Find-CondaExecutable`：先 `Get-Command conda.exe, conda`，再检查 `$HOME\miniforge3\Scripts\conda.exe`、`$HOME\Miniforge3\Scripts\conda.exe`、`$HOME\AppData\Local\miniforge3\Scripts\conda.exe`，找不到返回 `$null`。

新增 `Get-PrerequisiteReport`，返回以下固定检查：

```powershell
$checks = @(
    New-PrerequisiteCheck 'windows11' ($IsWindows11) $WindowsEvidence '需要 Windows 11 x64。'
    New-PrerequisiteCheck 'powershell' ($PSVersionTable.PSVersion.Major -ge 5) $PSVersionTable.PSVersion.ToString() '需要 Windows PowerShell 5.1 或 PowerShell 7。'
    New-PrerequisiteCheck 'git' ($null -ne $git) $git.Source '安装 Git for Windows 后重新打开终端。'
    New-PrerequisiteCheck 'uv' ($null -ne $uv) $uv.Source '按 uv 官方说明安装后重新打开终端。'
    New-PrerequisiteCheck 'conda' ($null -ne $conda) $conda '安装 Miniforge 后重新打开终端。'
    New-PrerequisiteCheck 'msvc' $MsvcFound $VsEvidence '安装 Visual Studio 2022 Build Tools 的“使用 C++ 的桌面开发”。'
    New-PrerequisiteCheck 'windows_sdk' $SdkFound $SdkEvidence '在 Build Tools 中安装 Windows SDK。'
    New-PrerequisiteCheck 'nvidia_gpu' $GpuFound $GpuEvidence '安装兼容 NVIDIA 驱动并确认 nvidia-smi 可用。'
    New-PrerequisiteCheck 'repository' $RepoFilesFound $RepoEvidence '从完整 Git 克隆运行工具。'
    New-PrerequisiteCheck 'disk' $DiskProbeSucceeded $DiskEvidence '确认磁盘可容纳 uv/Conda 缓存、两个环境和新样本。'
)
```

Windows 11 使用 Windows build `>= 22000` 和 `[Environment]::Is64BitOperatingSystem` 判断。Visual Studio 依次检查 `${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe`，并固定调用 `-latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`；Windows SDK 检查 `${env:ProgramFiles(x86)}\Windows Kits\10\Include` 下是否有至少一个版本目录。GPU 使用固定 `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits`。命令失败只产生 failed check，不安装任何内容。

新增 `Get-EnvironmentPlan`，使用可选的 `UvPath`、`CondaPath` 和 `RepositoryRoot` 参数精确返回，保证单元测试不依赖当前机器已安装 Conda：

```powershell
@(
    New-CommandSpec 'uv-python' $uv @('python', 'pin', '3.11') $root
    New-CommandSpec 'uv-sync' $uv @('sync', '--extra', 'operator', '--all-groups') $root
    if ($CondaEnvironmentExists) {
        New-CommandSpec 'conda-environment' $conda @(
            'env', 'update', '-n', 'surrogate-loop-fenicsx-0.11',
            '-f', (Join-Path $root 'environments\fenicsx-0.11.yml')
        ) $root
    } else {
        New-CommandSpec 'conda-environment' $conda @(
            'env', 'create', '-f', (Join-Path $root 'environments\fenicsx-0.11.yml')
        ) $root
    }
    New-CommandSpec 'python-imports' $uv @(
        'run', 'python', '-c',
        'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
    ) $root
    New-CommandSpec 'fenicsx-doctor' $uv @(
        'run', 'surrogate-loop', 'elasticity2d', 'doctor'
    ) $root
)
```

把三个新函数加入 `Export-ModuleMember`。

- [ ] **Step 4: 实现只读前置脚本**

`Test-Prerequisites.ps1` 使用：

```powershell
[CmdletBinding()]
param([switch]$Json)

Import-Module (Join-Path $PSScriptRoot 'SurrogateLoopMigration.psm1') -Force
$watch = [Diagnostics.Stopwatch]::StartNew()
try {
    $report = Get-PrerequisiteReport
    $exitCode = if ($report.status -eq 'pass') { 0 } else { 2 }
    $result = New-MigrationResult -Status $report.status -Stage 'prerequisites' `
        -Message $report.summary -Evidence $report -ExitCode $exitCode `
        -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit $exitCode
} catch {
    $result = New-MigrationResult -Status 'error' -Stage 'prerequisites' `
        -Message $_.Exception.Message -Evidence @{} -ExitCode 2 `
        -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit 2
}
```

- [ ] **Step 5: 实现支持 `-WhatIf` 的环境初始化脚本**

`Initialize-Environments.ps1` 使用 `[CmdletBinding(SupportsShouldProcess=$true, ConfirmImpact='Medium')]` 和 `[switch]$Json`。先读取 prerequisite report；失败时输出 exit 2 且不调用任何计划命令。通过时用 `conda env list --json` 判断固定环境是否存在，再获得 `Get-EnvironmentPlan`。

循环计划时必须：

```powershell
foreach ($command in $plan) {
    if ($PSCmdlet.ShouldProcess($command.name, "$($command.file_path) $($command.arguments -join ' ')")) {
        $completed = Invoke-FixedCommand -FilePath $command.file_path `
            -Arguments $command.arguments -WorkingDirectory $command.working_directory
        if ($completed.exit_code -ne 0) {
            throw "环境阶段 $($command.name) 失败：$($completed.stderr)"
        }
    }
}
```

`-WhatIf` 时仍返回 `status=planned`、完整固定 plan 和 exit 0，但不得执行 `uv` 或 `conda`。实际成功返回 `status=pass`；外部命令失败映射 exit 3。

- [ ] **Step 6: 运行环境测试**

```powershell
uv run pytest tests/tools/test_windows_environment_scripts.py -q
```

Expected: 全部 PASS；当前机器缺某个系统依赖时，JSON schema 测试仍允许脚本以 exit 2 返回可操作检查结果。

- [ ] **Step 7: 提交环境工具检查点**

```powershell
git add tools/windows-migration/SurrogateLoopMigration.psm1 tools/windows-migration/Test-Prerequisites.ps1 tools/windows-migration/Initialize-Environments.ps1 tests/tools/test_windows_environment_scripts.py
git commit -m "feat: add Windows environment bootstrap tools"
```

---

### Task 3: 实现分级安装验证和固定模型验证计划

**Files:**
- Modify: `tools/windows-migration/SurrogateLoopMigration.psm1`
- Create: `tools/windows-migration/Test-Installation.ps1`
- Modify: `tests/tools/test_windows_environment_scripts.py`

**Interfaces:**
- Consumes: Task 2 的 prerequisites 和 command spec。
- Produces: `Get-ModelVerificationPlan`, `Invoke-ModelVerification`, `Get-InstallationPlan`；计划函数提供默认字符串 `UvPath='uv'` / `CondaPath='conda'`，脚本在 prerequisite 通过后传入真实解析路径；入口参数 `-Level Prerequisites|Python|Fenicsx|FullChain`, `-AcceptedRunDir`, `-ModelKind`, `-ReportPath`, `-Json`。
- `Invoke-ModelVerification(...) -> PSCustomObject`：`status=accepted`, `report`, `prediction`, `model_kind`, `run_dir`。

- [ ] **Step 1: 先增加验证等级和固定命令失败测试**

在 `tests/tools/test_windows_environment_scripts.py` 增加：

```python
@pytest.mark.parametrize(
    ("kind", "report_tokens", "predict_tokens"),
    (
        ("scalar", ["report"], ["predict", "--gamma", "0.35"]),
        ("heat1d", ["operator", "report"], ["operator", "predict", "--alpha", "0.1"]),
        ("elasticity2d", ["elasticity2d", "report"], ["elasticity2d", "predict", "--e", "3"]),
    ),
)
def test_model_verification_plan_is_allowlisted(
    tmp_path: Path,
    kind: str,
    report_tokens: list[str],
    predict_tokens: list[str],
) -> None:
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Get-ModelVerificationPlan -ModelKind '{kind}' -RunDir {_ps_quote(tmp_path)} | "
        "ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert all(token in payload[0]["arguments"] for token in report_tokens)
    assert all(token in payload[1]["arguments"] for token in predict_tokens)


def test_unknown_model_kind_and_incomplete_full_chain_are_rejected(tmp_path: Path) -> None:
    unknown = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Get-ModelVerificationPlan -ModelKind 'shell' -RunDir {_ps_quote(tmp_path)}"
    )
    assert unknown.returncode != 0

    incomplete = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "Get-InstallationPlan -Level 'FullChain'"
    )
    assert incomplete.returncode != 0


def test_installation_plans_never_run_formal_training(tmp_path: Path) -> None:
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Get-InstallationPlan -Level 'FullChain' -ModelKind 'elasticity2d' "
        f"-AcceptedRunDir {_ps_quote(tmp_path)} | ConvertTo-Json -Depth 10 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    text = completed.stdout
    assert "test_elasticity2d_fenicsx_loop.py" in text
    for forbidden in ("calibrate", "full.json", "elasticity2d run", "sealed-test"):
        assert forbidden not in text


def test_installation_script_restores_e2e_environment_and_refuses_report_overwrite() -> None:
    content = (TOOLS / "Test-Installation.ps1").read_text(encoding="utf-8")
    assert "try" in content and "finally" in content
    assert "SURROGATE_LOOP_RUN_FENICSX_E2E" in content
    assert "Test-Path -LiteralPath $ReportPath" in content
    assert "exit 5" in content
```

- [ ] **Step 2: 运行新增测试并确认先失败**

```powershell
uv run pytest tests/tools/test_windows_environment_scripts.py -q
```

Expected: FAIL，因为验证计划函数与 `Test-Installation.ps1` 不存在。

- [ ] **Step 3: 实现三类固定 accepted 报告和推理计划**

`Get-ModelVerificationPlan` 的 `ModelKind` 使用 `[ValidateSet('scalar','heat1d','elasticity2d')]`，精确映射：

```powershell
switch ($ModelKind) {
    'scalar' {
        $report = @('run','surrogate-loop','report','--run-dir',$resolvedRun)
        $predict = @('run','surrogate-loop','predict','--run-dir',$resolvedRun,'--gamma','0.35')
    }
    'heat1d' {
        $report = @('run','surrogate-loop','operator','report','--run-dir',$resolvedRun)
        $predict = @(
            'run','surrogate-loop','operator','predict','--run-dir',$resolvedRun,
            '--alpha','0.1','--a','1.0','--b','0.1','--x','0.5','--t','0.25'
        )
    }
    'elasticity2d' {
        $report = @('run','surrogate-loop','elasticity2d','report','--run-dir',$resolvedRun)
        $predict = @(
            'run','surrogate-loop','elasticity2d','predict','--run-dir',$resolvedRun,
            '--e','3','--nu','0.3','--p','0.006','--theta','-1.5707963268',
            '--y0','0.5','--w','0.12','--x','4','--y','0.5'
        )
    }
}
```

`Invoke-ModelVerification` 执行两个 command spec，报告 JSON 必须满足 `state == accepted` 或 `status == accepted`；predict 必须是合法 JSON。任何失败映射为异常，由入口脚本转 exit 3。

- [ ] **Step 4: 实现分级命令计划**

`Get-InstallationPlan` 按等级累加：

```powershell
$python = @(
    New-CommandSpec 'cli-help' $uv @('run','surrogate-loop','--help') $root
    New-CommandSpec 'cli-version' $uv @('run','python','-m','surrogate_loop','--version') $root
    New-CommandSpec 'cuda-backward' $uv @(
        'run','python','-c',
        "import torch; x=torch.randn(128,128,device='cuda',requires_grad=True); y=x.square().mean(); y.backward(); print(torch.cuda.get_device_name(0), torch.isfinite(x.grad).all().item())"
    ) $root
    New-CommandSpec 'ruff' $uv @('run','ruff','check','.') $root
    New-CommandSpec 'pytest' $uv @('run','pytest','-q') $root
)
$fenicsx = @(
    New-CommandSpec 'fenicsx-doctor' $uv @('run','surrogate-loop','elasticity2d','doctor') $root
    New-CommandSpec 'solver-tests' $conda @(
        'run','-n','surrogate-loop-fenicsx-0.11','python','-m','pytest',
        'tests/solver/elasticity2d','-v'
    ) $root
)
$fullChain = @(
    New-CommandSpec 'real-fenicsx-e2e' $uv @(
        'run','pytest','tests/e2e/test_elasticity2d_fenicsx_loop.py','-v'
    ) $root
)
```

`FullChain` 没有 `AcceptedRunDir` 或 `ModelKind` 时抛出明确错误；model report/predict 不混入 command plan，而由脚本在 E2E 后调用 `Invoke-ModelVerification`。

把 `Get-ModelVerificationPlan`、`Invoke-ModelVerification` 和 `Get-InstallationPlan` 加入模块 `Export-ModuleMember`。

- [ ] **Step 5: 实现 `Test-Installation.ps1`**

入口把 `Level` 声明为 `[Parameter(Mandatory)]` 并使用 `[ValidateSet('Prerequisites','Python','Fenicsx','FullChain')]`，避免隐式选择验证成本。所有等级先执行 prerequisite report；`Prerequisites` 直接返回，其余执行计划并把每项结果加入 evidence。

FullChain 临时环境变量必须按以下模式恢复：

```powershell
$hadE2E = Test-Path Env:SURROGATE_LOOP_RUN_FENICSX_E2E
$previousE2E = if ($hadE2E) { $env:SURROGATE_LOOP_RUN_FENICSX_E2E } else { $null }
try {
    $env:SURROGATE_LOOP_RUN_FENICSX_E2E = '1'
    # 只在这里运行 real-fenicsx-e2e，然后调用 Invoke-ModelVerification。
} finally {
    if ($hadE2E) {
        $env:SURROGATE_LOOP_RUN_FENICSX_E2E = $previousE2E
    } else {
        Remove-Item Env:SURROGATE_LOOP_RUN_FENICSX_E2E -ErrorAction SilentlyContinue
    }
}
```

`-ReportPath` 必须在执行前解析；已存在立即输出 exit 5。成功后以 UTF-8 无 BOM 写完整 result。`-Json` stdout 只输出一个 result；命令 stdout/stderr 收集到 evidence，不直接污染 JSON。

- [ ] **Step 6: 运行分级验证测试**

```powershell
uv run pytest tests/tools/test_windows_environment_scripts.py -q
```

Expected: 全部 PASS；测试只检查计划和 `Prerequisites` JSON，不运行 CUDA、FEniCSx 或真实 E2E。

- [ ] **Step 7: 提交分级验证检查点**

```powershell
git add tools/windows-migration/SurrogateLoopMigration.psm1 tools/windows-migration/Test-Installation.ps1 tests/tools/test_windows_environment_scripts.py
git commit -m "feat: add tiered Windows installation verification"
```

---

### Task 4: 实现 accepted 运行 bundle 导出

**Files:**
- Modify: `tools/windows-migration/SurrogateLoopMigration.psm1`
- Create: `tools/windows-migration/Export-AcceptedRun.ps1`
- Create: `tests/tools/test_windows_run_transfer.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Task 3 `Invoke-ModelVerification` 的 accepted payload；Task 1 文件 manifest。
- Produces: `New-RunBundleArchive`。
- Archive: `$runId.surrogate-run.zip`；sidecar: `$runId.surrogate-run.sha256.json`。
- Bundle schema 1: `schema_version`, `model_kind`, `run_id`, `export_repo_commit`, `export_repo_dirty`, `created_at_utc`, `files`。

- [ ] **Step 1: 先写合成 accepted bundle 导出失败测试**

创建 `tests/tools/test_windows_run_transfer.py`，顶部精确导入：

```python
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tests.tools.powershell import (
    POWERSHELL,
    ps_quote as _ps_quote,
    run_powershell as _run_powershell,
)

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/windows-migration/SurrogateLoopMigration.psm1"
TOOLS = ROOT / "tools/windows-migration"

pytestmark = pytest.mark.skipif(
    POWERSHELL is None,
    reason="Windows PowerShell 5.1 is required for Windows migration tool tests",
)
```

随后加入：

```python
def test_bundle_archive_contains_manifest_run_tree_and_checksum(tmp_path: Path) -> None:
    run_dir = tmp_path / "accepted-run"
    output = tmp_path / "output"
    run_dir.mkdir()
    output.mkdir()
    (run_dir / "status.json").write_text('{"status":"accepted"}', encoding="utf-8")
    (run_dir / "weights.bin").write_bytes(b"weights")
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "$verification=[pscustomobject]@{status='accepted'};"
        f"New-RunBundleArchive -RunDir {_ps_quote(run_dir)} -ModelKind 'elasticity2d' "
        f"-OutputDirectory {_ps_quote(output)} -RepositoryRoot {_ps_quote(ROOT)} "
        "-Verification $verification | ConvertTo-Json -Depth 20 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    archive = Path(result["archive_path"])
    checksum = Path(result["checksum_path"])
    assert archive.name == "accepted-run.surrogate-run.zip"
    assert checksum.name == "accepted-run.surrogate-run.sha256.json"
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        manifest = json.loads(bundle.read("bundle.json"))
    assert "run/accepted-run/status.json" in names
    assert "run/accepted-run/weights.bin" in names
    assert manifest["schema_version"] == 1
    assert manifest["model_kind"] == "elasticity2d"
    assert manifest["run_id"] == "accepted-run"
    assert manifest["export_repo_commit"]
    assert isinstance(manifest["export_repo_dirty"], bool)
    sidecar = json.loads(checksum.read_text(encoding="utf-8"))
    assert sidecar["archive_name"] == archive.name
    assert len(sidecar["archive_sha256"]) == 64


def test_bundle_export_rejects_nonaccepted_and_existing_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output = tmp_path / "output"
    run_dir.mkdir()
    output.mkdir()
    rejected = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "$verification=[pscustomobject]@{status='rejected'};"
        f"New-RunBundleArchive -RunDir {_ps_quote(run_dir)} -ModelKind 'scalar' "
        f"-OutputDirectory {_ps_quote(output)} -RepositoryRoot {_ps_quote(ROOT)} "
        "-Verification $verification"
    )
    assert rejected.returncode != 0
    assert "accepted" in rejected.stderr

    (output / "run.surrogate-run.zip").write_bytes(b"existing")
    conflict = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "$verification=[pscustomobject]@{status='accepted'};"
        f"New-RunBundleArchive -RunDir {_ps_quote(run_dir)} -ModelKind 'scalar' "
        f"-OutputDirectory {_ps_quote(output)} -RepositoryRoot {_ps_quote(ROOT)} "
        "-Verification $verification"
    )
    assert conflict.returncode != 0
    assert "已存在" in conflict.stderr


def test_export_script_validates_before_packaging_and_has_no_arbitrary_command() -> None:
    content = (TOOLS / "Export-AcceptedRun.ps1").read_text(encoding="utf-8")
    assert content.index("Invoke-ModelVerification") < content.index("New-RunBundleArchive")
    assert "ValidateSet('scalar', 'heat1d', 'elasticity2d')" in content
    assert "Invoke-Expression" not in content
```

- [ ] **Step 2: 运行导出测试并确认先失败**

```powershell
uv run pytest tests/tools/test_windows_run_transfer.py -q
```

Expected: FAIL，因为 bundle 函数和导出脚本不存在。

- [ ] **Step 3: 实现 `New-RunBundleArchive`**

函数参数固定为 `RunDir`, `ModelKind`, `OutputDirectory`, `RepositoryRoot`, `Verification`。实现顺序：

1. `Verification.status` 必须是 `accepted`；
2. `run_id` 必须等于目录 basename 且只允许 `[A-Za-z0-9._-]+`；
3. 输出目录必须存在；archive/sidecar 任一存在即抛出“已存在”；
4. `Get-FileManifest` 生成 run 文件清单；
5. `git -C $RepositoryRoot rev-parse HEAD` 和 `git -C $RepositoryRoot status --porcelain` 生成 `export_repo_commit` / `export_repo_dirty`；
6. 在系统 temp 下创建工具自有 staging，写 UTF-8 无 BOM `bundle.json`，复制 run 普通文件到 `run/$runId`；
7. 使用 `[IO.Compression.ZipFile]::CreateFromDirectory` 创建 ZIP；
8. 计算 ZIP 大小和 SHA-256，写 schema 1 sidecar；
9. finally 只删除本函数创建且位于系统 temp 下、basename 以 `surrogate-loop-export-` 开头的 staging。

bundle object 精确为：

```powershell
[pscustomobject][ordered]@{
    schema_version = 1
    model_kind = $ModelKind
    run_id = $runId
    export_repo_commit = $commit
    export_repo_dirty = [bool]$dirty
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    files = $files
}
```

sidecar 精确为：

```powershell
[pscustomobject][ordered]@{
    schema_version = 1
    archive_name = $archiveName
    archive_bytes = [int64](Get-Item -LiteralPath $archivePath).Length
    archive_sha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
}
```

把 `New-RunBundleArchive` 加入模块 `Export-ModuleMember`。

- [ ] **Step 4: 实现导出入口和忽略规则**

`Export-AcceptedRun.ps1` 参数：

```powershell
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RunDir,
    [Parameter(Mandatory)]
    [ValidateSet('scalar', 'heat1d', 'elasticity2d')]
    [string]$ModelKind,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [switch]$Json
)
```

先 `Invoke-ModelVerification`，再 `New-RunBundleArchive`。参数/accepted 错误 exit 2，外部命令错误 exit 3，完整性错误 exit 4，输出冲突 exit 5。不要通过异常文本猜测所有分类；为模块自定义异常设置稳定 `Data['MigrationExitCode']`，入口优先读取该值。

在 `.gitignore` 增加：

```gitignore
*.surrogate-run.zip
*.surrogate-run.sha256.json
migration-verification*.json
```

- [ ] **Step 5: 运行导出和公共模块测试**

```powershell
uv run pytest tests/tools/test_windows_run_transfer.py tests/tools/test_windows_migration_module.py -q
```

Expected: 全部 PASS；测试产物只存在 pytest 临时目录。

- [ ] **Step 6: 提交导出检查点**

```powershell
git add .gitignore tools/windows-migration/SurrogateLoopMigration.psm1 tools/windows-migration/Export-AcceptedRun.ps1 tests/tools/test_windows_run_transfer.py
git commit -m "feat: export accepted run bundles"
```

---

### Task 5: 实现安全 bundle 导入与发布

**Files:**
- Modify: `tools/windows-migration/SurrogateLoopMigration.psm1`
- Create: `tools/windows-migration/Import-AcceptedRun.ps1`
- Modify: `tests/tools/test_windows_run_transfer.py`

**Interfaces:**
- Consumes: Task 4 bundle schema 1、sidecar schema 1；Task 3 accepted verification。
- Produces: `Expand-VerifiedRunBundle`, `Publish-ImportedRun`, `Remove-OwnedStagingDirectory`。
- `Expand-VerifiedRunBundle(...) -> PSCustomObject`：`bundle`, `staging_root`, `run_dir`, `target_run_dir`, `commit_warning`。

- [ ] **Step 1: 先增加安全导入、篡改、冲突与清理失败测试**

在 `tests/tools/test_windows_run_transfer.py` 增加完整 helper：

```python
def _make_bundle(base: Path) -> tuple[Path, Path]:
    run_dir = base / "accepted-run"
    output = base / "output"
    run_dir.mkdir(parents=True)
    output.mkdir()
    (run_dir / "status.json").write_text('{"status":"accepted"}', encoding="utf-8")
    (run_dir / "weights.bin").write_bytes(b"weights")
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        "$verification=[pscustomobject]@{status='accepted'};"
        f"New-RunBundleArchive -RunDir {_ps_quote(run_dir)} -ModelKind 'elasticity2d' "
        f"-OutputDirectory {_ps_quote(output)} -RepositoryRoot {_ps_quote(ROOT)} "
        "-Verification $verification | ConvertTo-Json -Depth 20 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    return Path(payload["archive_path"]), Path(payload["checksum_path"])
```

随后增加：

```python
def test_verified_bundle_expands_only_to_owned_staging(tmp_path: Path) -> None:
    archive, checksum = _make_bundle(tmp_path)
    runs = tmp_path / "target-runs"
    runs.mkdir()
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Expand-VerifiedRunBundle -ArchivePath {_ps_quote(archive)} "
        f"-ChecksumPath {_ps_quote(checksum)} -RunsDirectory {_ps_quote(runs)} "
        f"-TargetRepositoryRoot {_ps_quote(ROOT)} | ConvertTo-Json -Depth 20 -Compress"
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    staging = Path(payload["staging_root"])
    assert staging.parent == runs
    assert staging.name.startswith(".migration-staging-")
    assert (Path(payload["run_dir"]) / "weights.bin").read_bytes() == b"weights"


def test_bundle_import_rejects_tampered_archive_and_target_conflict(tmp_path: Path) -> None:
    archive, checksum = _make_bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tamper")
    runs = tmp_path / "runs"
    runs.mkdir()
    tampered = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Expand-VerifiedRunBundle -ArchivePath {_ps_quote(archive)} "
        f"-ChecksumPath {_ps_quote(checksum)} -RunsDirectory {_ps_quote(runs)} "
        f"-TargetRepositoryRoot {_ps_quote(ROOT)}"
    )
    assert tampered.returncode != 0
    assert "SHA-256" in tampered.stderr

    archive, checksum = _make_bundle(tmp_path / "fresh")
    (runs / "accepted-run").mkdir()
    conflict = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Expand-VerifiedRunBundle -ArchivePath {_ps_quote(archive)} "
        f"-ChecksumPath {_ps_quote(checksum)} -RunsDirectory {_ps_quote(runs)} "
        f"-TargetRepositoryRoot {_ps_quote(ROOT)}"
    )
    assert conflict.returncode != 0
    assert "已存在" in conflict.stderr


def test_owned_staging_cleanup_refuses_unowned_paths(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Remove-OwnedStagingDirectory -Path {_ps_quote(ordinary)} "
        f"-RunsDirectory {_ps_quote(tmp_path)}"
    )
    assert completed.returncode != 0
    assert ordinary.exists()


def test_import_script_verifies_before_publish_and_cleans_in_finally() -> None:
    content = (TOOLS / "Import-AcceptedRun.ps1").read_text(encoding="utf-8")
    assert content.index("Expand-VerifiedRunBundle") < content.index("Invoke-ModelVerification")
    assert content.index("Invoke-ModelVerification") < content.index("Publish-ImportedRun")
    assert "finally" in content
    assert "Remove-OwnedStagingDirectory" in content
    assert "Invoke-Expression" not in content
```

为恶意 ZIP 增加完整测试；sidecar 重新计算为匹配恶意 ZIP，证明不是总哈希失败掩盖路径检查：

```python
@pytest.mark.parametrize(
    "entries",
    (
        {"bundle.json": "{}", "../escape.txt": "bad"},
        {
            "bundle.json": "{}",
            "run/id/A.txt": "one",
            "run/id/a.txt": "two",
        },
    ),
)
def test_bundle_import_rejects_unsafe_or_case_duplicate_entries(
    tmp_path: Path,
    entries: dict[str, str],
) -> None:
    import hashlib

    archive = tmp_path / "malicious.surrogate-run.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, value in entries.items():
            bundle.writestr(name, value)
    checksum = tmp_path / "malicious.surrogate-run.sha256.json"
    checksum.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archive_name": archive.name,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    completed = _run_powershell(
        f"Import-Module {_ps_quote(MODULE)} -Force;"
        f"Expand-VerifiedRunBundle -ArchivePath {_ps_quote(archive)} "
        f"-ChecksumPath {_ps_quote(checksum)} -RunsDirectory {_ps_quote(runs)} "
        f"-TargetRepositoryRoot {_ps_quote(ROOT)}"
    )
    assert completed.returncode != 0
    assert "ZIP" in completed.stderr
```

- [ ] **Step 2: 运行导入测试并确认先失败**

```powershell
uv run pytest tests/tools/test_windows_run_transfer.py -q
```

Expected: FAIL，因为三个导入函数和入口脚本不存在。

- [ ] **Step 3: 实现 sidecar、ZIP 和逐文件校验解包**

`Expand-VerifiedRunBundle` 按以下不可换序的步骤实现：

1. archive、checksum 必须是现有普通文件；
2. sidecar 字段集合和 schema 1 精确校验；
3. sidecar `archive_name`、`archive_bytes`、`archive_sha256` 与 ZIP 一致；
4. `Test-SafeZipEntries`，要求 ZIP 恰有 `bundle.json` 和单一 `run/$runId/` 树；
5. bundle schema、`ModelKind`、run_id 正则和 files 清单精确校验；
6. `runs/$runId` 已存在时抛出 exit 5；
7. 在 runs 下创建 `.migration-staging-$guid`；
8. 对每个 ZIP entry 用经过边界校验的绝对路径逐项创建目录/文件，不调用不受控的整包 `ExtractToDirectory`；
9. `Test-FileManifest` 校验 extracted `run/$runId`；
10. 比较 `export_repo_commit` 与目标 `git rev-parse HEAD`，只生成 warning，不跳过 accepted 验证。

异常对象的 `Data['MigrationExitCode']`：sidecar/ZIP/manifest 为 4，目标冲突为 5。

- [ ] **Step 4: 实现安全发布和 staging 清理**

`Publish-ImportedRun` 再次确认 final 不存在、staging run 是 staging root 子目录、目标是 runs 直接子目录，然后用 `Move-Item -LiteralPath` 发布。

`Remove-OwnedStagingDirectory` 必须同时满足：

- path 与 runs 均能解析为绝对路径；
- path 的 parent 等于 runs；
- basename 匹配 `^.migration-staging-[0-9a-f-]+$`；
- path 不是 reparse point。

不满足则抛错且不删除；满足才 `Remove-Item -LiteralPath $resolved -Recurse -Force`。该函数只用于清理工具本次创建的 staging。

把 `Expand-VerifiedRunBundle`、`Publish-ImportedRun` 和 `Remove-OwnedStagingDirectory` 加入模块 `Export-ModuleMember`。

- [ ] **Step 5: 实现导入入口**

`Import-AcceptedRun.ps1` 参数只有 `ArchivePath`, `ChecksumPath`, `Json`。入口从 bundle 读取 `ModelKind`，用户不能覆盖。流程：

```powershell
$expanded = $null
try {
    $expanded = Expand-VerifiedRunBundle -ArchivePath $ArchivePath `
        -ChecksumPath $ChecksumPath -RunsDirectory (Join-Path $root 'runs') `
        -TargetRepositoryRoot $root
    $verification = Invoke-ModelVerification -ModelKind $expanded.bundle.model_kind `
        -RunDir $expanded.run_dir -RepositoryRoot $root
    $published = Publish-ImportedRun -ExpandedBundle $expanded
    # 输出 imported result，evidence 包含 warning、verification 和最终路径。
} catch {
    # 使用稳定 MigrationExitCode，默认完整性失败 4，外部 accepted 验证失败 3。
} finally {
    if ($null -ne $expanded -and (Test-Path -LiteralPath $expanded.staging_root)) {
        Remove-OwnedStagingDirectory -Path $expanded.staging_root `
            -RunsDirectory (Join-Path $root 'runs')
    }
}
```

若 publish 成功，staging root 只剩空目录并由 finally 删除；最终 run 不在 staging 下，不会被清理。

- [ ] **Step 6: 运行所有迁移工具单元测试**

```powershell
uv run pytest tests/tools -q
```

Expected: 全部 PASS；不访问真实 `runs/`，不执行真实 uv/Conda/FEniCSx。

- [ ] **Step 7: 提交导入检查点**

```powershell
git add tools/windows-migration/SurrogateLoopMigration.psm1 tools/windows-migration/Import-AcceptedRun.ps1 tests/tools/test_windows_run_transfer.py
git commit -m "feat: import verified accepted run bundles"
```

---

### Task 6: 编写工具速查、完整迁移手册和稳定导航

**Files:**
- Create: `tools/windows-migration/README.md`
- Create: `docs/guides/Windows跨机迁移指南.md`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/guides/环境与验证.md`
- Modify: `tests/unit/test_documentation_navigation.py`

**Interfaces:**
- Consumes: Tasks 2–5 的最终脚本名、参数、等级、退出码和安全语义。
- Produces: 从根 README 到完整迁移指南和工具 README 的稳定入口。

- [ ] **Step 1: 先增加文档入口和关键安全合同测试**

在 `test_agent_documentation_entrypoints_exist` 增加：

```python
"docs/guides/Windows跨机迁移指南.md",
"tools/windows-migration/README.md",
```

新增：

```python
def test_windows_migration_docs_define_safe_full_chain_contract() -> None:
    guide = _read("docs/guides/Windows跨机迁移指南.md")
    tools = _read("tools/windows-migration/README.md")
    for required in (
        "Windows 11 x64",
        "NVIDIA GPU",
        "Test-Prerequisites.ps1",
        "Initialize-Environments.ps1",
        "Test-Installation.ps1",
        "Export-AcceptedRun.ps1",
        "Import-AcceptedRun.ps1",
        "FullChain",
        "accepted",
        "SHA-256",
    ):
        assert required in guide
        assert required in tools
    assert "不会自动安装" in guide
    assert "不会启动 calibration、Smoke 或 Full" in guide
    assert "SHA-256 不等于数字签名" in guide


def test_root_and_document_map_link_windows_migration_guide() -> None:
    root = _read("README.md")
    document_map = _read("docs/README.md")
    environment = _read("docs/guides/环境与验证.md")
    assert "docs/guides/Windows跨机迁移指南.md" in root
    assert "Windows跨机迁移指南.md" in document_map
    assert "Windows跨机迁移指南.md" in environment
```

把 `test_local_markdown_links_resolve` 的 documents 改为：

```python
documents = [
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    *sorted((ROOT / "docs").rglob("*.md")),
    *sorted((ROOT / "tools").rglob("*.md")),
]
```

- [ ] **Step 2: 运行文档测试并确认先失败**

```powershell
uv run pytest tests/unit/test_documentation_navigation.py -q
```

Expected: FAIL，因为两个新 README/指南和现有入口尚未增加。

- [ ] **Step 3: 编写工具 README**

`tools/windows-migration/README.md` 按以下顺序写完整内容：

```markdown
# Windows 跨机迁移工具

## 支持范围
## 工具不会执行的操作
## 源电脑：导出 accepted 运行
## 目标电脑：检查与初始化
## 目标电脑：导入 accepted 运行
## 分级验证
## 参数与退出码速查
## 完整手册
```

最短命令链必须使用真实脚本参数：

```powershell
& .\tools\windows-migration\Test-Prerequisites.ps1
& .\tools\windows-migration\Initialize-Environments.ps1 -WhatIf
& .\tools\windows-migration\Initialize-Environments.ps1
& .\tools\windows-migration\Export-AcceptedRun.ps1 -RunDir .\runs\elasticity-full-ba8ff8e584d9 -ModelKind elasticity2d -OutputDirectory D:\surrogate-loop-transfer
& .\tools\windows-migration\Import-AcceptedRun.ps1 -ArchivePath D:\surrogate-loop-transfer\elasticity-full-ba8ff8e584d9.surrogate-run.zip -ChecksumPath D:\surrogate-loop-transfer\elasticity-full-ba8ff8e584d9.surrogate-run.sha256.json
& .\tools\windows-migration\Test-Installation.ps1 -Level FullChain -AcceptedRunDir .\runs\elasticity-full-ba8ff8e584d9 -ModelKind elasticity2d
```

说明这些命令不运行正式 Full；`D:\surrogate-loop-transfer` 是用户需替换的示例外部目录，不由工具自动创建或覆盖。

- [ ] **Step 4: 编写完整 Windows 迁移指南**

`docs/guides/Windows跨机迁移指南.md` 使用规格第 11 节的七部分，并加入：

- 三档目标表：accepted 只读推理、开发/测试、完整 FEniCSx 训练链；
- 源电脑先运行 report/predict，再导出；ZIP 与 sidecar 建议通过独立渠道核对哈希；
- 目标电脑系统依赖只引用官方安装页面名称，不把不稳定下载 URL 写死为脚本逻辑；
- PowerShell 受组织策略阻止时查看 `Get-ExecutionPolicy -List` 并联系管理员，文档不要求修改策略；
- `nvidia-smi` 成功不等于 PyTorch CUDA 成功，后者由 `Level Python` 前后向验证；
- FFCx 缓存权限失败的症状、该失败不等于求解器/模型失败、修复后用新身份重试；
- `FullChain`、calibration、Smoke、Full 的证据等级表；
- accepted bundle 的完整性、提交差异 warning、兼容复现和逐位复现边界；
- 全新克隆不含 `runs/`，只复制单个权重文件不能通过可信推理。

- [ ] **Step 5: 更新稳定入口和 Agent 规则**

`AGENTS.md` 的目录边界增加：

```markdown
- `tools/`：仓库维护和跨机迁移工具；不得在其中复制模型训练、求解器或推理核心实现。
```

证据与授权增加：

```markdown
- Windows 迁移工具的 `FullChain` 只验证环境、真实微型 E2E 和已有 accepted 推理；它不是新的 Full 验收，不授权创建 Full 身份或消费 sealed-test。
```

根 README 增加“Windows 跨机迁移”小节，链接 `[Windows 跨机迁移指南](docs/guides/Windows跨机迁移指南.md)` 与 `[工具速查](tools/windows-migration/README.md)`。说明当前支持 Windows 11 x64 + NVIDIA GPU，系统依赖人工安装，工具负责检查、环境创建、分级验证和 accepted 传输。

`docs/README.md` 在“运行已有闭环”和“稳定文档”加入迁移指南；`docs/guides/环境与验证.md` 在开头说明本页适合当前机器，另一台 Windows 电脑应从迁移指南开始，并删除过期的“73 项测试”固定计数或改为“以当前 pytest 输出为准”。

- [ ] **Step 6: 运行文档和工具测试**

```powershell
uv run pytest tests/unit/test_documentation_navigation.py tests/tools -q
```

Expected: 全部 PASS，工具 README 中的相对链接无断链。

- [ ] **Step 7: 提交文档检查点**

```powershell
git add AGENTS.md README.md docs/README.md docs/guides/环境与验证.md docs/guides/Windows跨机迁移指南.md tools/windows-migration/README.md tests/unit/test_documentation_navigation.py
git commit -m "docs: add Windows migration guide"
```

---

### Task 7: 真实 Windows 环境、accepted 导出导入和完整回归验收

**Files:**
- Verify all implementation files.
- Modify only task-scoped files if a verification command exposes a defect.

**Interfaces:**
- Consumes: Tasks 1–6 全部工具和文档；本地 accepted 运行 `runs/elasticity-full-ba8ff8e584d9/`。
- Produces: 新鲜的 Windows 5.1、CUDA、FEniCSx、真实微型 E2E、accepted 迁移和全量测试证据。

- [ ] **Step 1: 扫描危险命令和意外产物**

```powershell
$unsafeMatches = rg -n "Invoke-Expression|winget\s|Set-ExecutionPolicy|-Verb\s+RunAs|conda\s+env\s+remove" tools/windows-migration
if ($LASTEXITCODE -eq 0) { $unsafeMatches; throw '发现禁止的危险命令' }
if ($LASTEXITCODE -gt 1) { throw '危险命令扫描执行失败' }
git status --short --ignored
```

Expected: 扫描无匹配；迁移 ZIP、sidecar、staging、`.venv`、缓存和 `runs/` 显示为 ignored，不能显示为 tracked/untracked 待提交。

- [ ] **Step 2: 运行前置检查的人工与 JSON 模式**

```powershell
& .\tools\windows-migration\Test-Prerequisites.ps1
& .\tools\windows-migration\Test-Prerequisites.ps1 -Json
```

Expected: 当前 Windows 11 x64、PowerShell 5.1、Git、uv、Miniforge、MSVC/SDK 和 NVIDIA GPU 均为 pass；JSON 是单一对象且 exit 0。若真实依赖缺失，停止并报告，不伪造通过。

- [ ] **Step 3: 验证环境初始化 `-WhatIf` 不写入**

记录运行前 `.python-version`、`.venv` 和 Conda env 列表，再执行并逐项比较：

```powershell
$pythonPinBefore = (Get-FileHash -LiteralPath .\.python-version -Algorithm SHA256).Hash
$venvPythonBefore = (Get-Item -LiteralPath .\.venv\Scripts\python.exe).LastWriteTimeUtc
$condaBefore = (& conda env list --json | ConvertFrom-Json).envs
& .\tools\windows-migration\Initialize-Environments.ps1 -WhatIf
& .\tools\windows-migration\Initialize-Environments.ps1 -WhatIf -Json
$pythonPinAfter = (Get-FileHash -LiteralPath .\.python-version -Algorithm SHA256).Hash
$venvPythonAfter = (Get-Item -LiteralPath .\.venv\Scripts\python.exe).LastWriteTimeUtc
$condaAfter = (& conda env list --json | ConvertFrom-Json).envs
if ($pythonPinBefore -ne $pythonPinAfter) { throw '.python-version 被 WhatIf 修改' }
if ($venvPythonBefore -ne $venvPythonAfter) { throw '.venv 被 WhatIf 修改' }
if (Compare-Object $condaBefore $condaAfter) { throw 'Conda 环境列表被 WhatIf 修改' }
```

Expected: 输出固定 uv/Conda/import/doctor 计划，状态为 `planned`；Git diff、`.python-version` 内容、`.venv` 关键文件时间和 Conda env 列表没有被脚本改变。

- [ ] **Step 4: 运行 Python 与 FEniCSx 分级验证**

```powershell
& .\tools\windows-migration\Test-Installation.ps1 -Level Python -ReportPath .\migration-verification-python.json
& .\tools\windows-migration\Test-Installation.ps1 -Level Fenicsx -ReportPath .\migration-verification-fenicsx.json
```

Expected: 两级 exit 0；Python 级 CUDA 前后向、Ruff 和普通 pytest 通过；Fenicsx 级 doctor 与 `tests/solver/elasticity2d` 通过。两个 report 被 `.gitignore` 忽略。

- [ ] **Step 5: 导出真实二维 accepted bundle**

使用明确的系统临时目录，不写仓库：

```powershell
$migrationVerifyRoot = Join-Path ([IO.Path]::GetTempPath()) ('surrogate-loop-migration-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $migrationVerifyRoot | Out-Null
& .\tools\windows-migration\Export-AcceptedRun.ps1 `
    -RunDir .\runs\elasticity-full-ba8ff8e584d9 `
    -ModelKind elasticity2d `
    -OutputDirectory $migrationVerifyRoot
```

Expected: exit 0，生成约 50 MiB 量级 ZIP 和 sidecar；原运行目录无修改；报告状态为 accepted。

- [ ] **Step 6: 在隔离 detached worktree 导入真实 bundle**

使用已被忽略的 `.worktrees/migration-import-verification`。创建前确认目标不存在且 `.worktrees` 被 Git 忽略：

```powershell
$verificationWorktree = (Join-Path (git rev-parse --show-toplevel) '.worktrees\migration-import-verification')
git check-ignore .worktrees
if (Test-Path -LiteralPath $verificationWorktree) { throw "验证 worktree 已存在：$verificationWorktree" }
git worktree add --detach $verificationWorktree HEAD
```

让 detached worktree 复用当前已验证 `.venv`，只在当前 PowerShell 进程临时设置并在 finally 恢复：

```powershell
$previousUvEnvironment = $env:UV_PROJECT_ENVIRONMENT
$locationPushed = $false
try {
    $env:UV_PROJECT_ENVIRONMENT = (Join-Path (git rev-parse --show-toplevel) '.venv')
    Push-Location $verificationWorktree
    $locationPushed = $true
    & .\tools\windows-migration\Import-AcceptedRun.ps1 `
        -ArchivePath (Join-Path $migrationVerifyRoot 'elasticity-full-ba8ff8e584d9.surrogate-run.zip') `
        -ChecksumPath (Join-Path $migrationVerifyRoot 'elasticity-full-ba8ff8e584d9.surrogate-run.sha256.json')
} finally {
    if ($locationPushed) { Pop-Location }
    if ($null -eq $previousUvEnvironment) { Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue }
    else { $env:UV_PROJECT_ENVIRONMENT = $previousUvEnvironment }
}
```

Expected: exit 0；worktree 中出现完整 `runs/elasticity-full-ba8ff8e584d9/`，没有遗留 `.migration-staging-*`；导入时 accepted 报告和二维代表性预测通过。

- [ ] **Step 7: 在隔离 worktree 运行 FullChain**

```powershell
$previousUvEnvironment = $env:UV_PROJECT_ENVIRONMENT
$locationPushed = $false
try {
    $env:UV_PROJECT_ENVIRONMENT = (Join-Path (git rev-parse --show-toplevel) '.venv')
    Push-Location $verificationWorktree
    $locationPushed = $true
    & .\tools\windows-migration\Test-Installation.ps1 `
        -Level FullChain `
        -AcceptedRunDir .\runs\elasticity-full-ba8ff8e584d9 `
        -ModelKind elasticity2d `
        -ReportPath .\migration-verification-full-chain.json
} finally {
    if ($locationPushed) { Pop-Location }
    if ($null -eq $previousUvEnvironment) { Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue }
    else { $env:UV_PROJECT_ENVIRONMENT = $previousUvEnvironment }
}
```

Expected: exit 0；solver 科学测试、显式真实微型 FEniCSx E2E、accepted report 和代表性预测全部通过；没有 calibration、Smoke、Full 或 sealed-test 新运行。

- [ ] **Step 8: 安全清理仅由本任务创建的验证目录**

先验证 worktree 精确位于仓库 `.worktrees` 且 temp 目录位于系统 temp：

```powershell
$repoRoot = [IO.Path]::GetFullPath((git rev-parse --show-toplevel))
$expectedWorktreeRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot '.worktrees'))
$resolvedWorktree = [IO.Path]::GetFullPath($verificationWorktree)
$resolvedTemp = [IO.Path]::GetFullPath($migrationVerifyRoot)
$systemTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
if (-not $resolvedWorktree.StartsWith($expectedWorktreeRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw '拒绝清理非预期 worktree' }
if (-not $resolvedTemp.StartsWith($systemTemp, [StringComparison]::OrdinalIgnoreCase)) { throw '拒绝清理非系统临时目录' }
git worktree remove --force $resolvedWorktree
git worktree prune
Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
```

Expected: 只移除本任务创建的 detached worktree 和临时迁移包；当前分支、原 accepted run 和现有环境不受影响。

- [ ] **Step 9: 运行完整自动化质量门**

```powershell
uv run ruff check .
uv run pytest -q
```

Expected: Ruff exit 0；pytest 无失败。准确记录 Windows PowerShell 工具测试、普通条件跳过和显式真实 E2E 的数量，不能把 skip 表述为通过。

- [ ] **Step 10: 检查补丁、提交和工作区**

```powershell
git diff --check
git status -sb
git log --oneline --decorate -10
git ls-files "*.surrogate-run.zip" "*.surrogate-run.sha256.json" "migration-verification*.json" "runs/*"
```

Expected: `git diff --check` exit 0；分支为 `docs/weekly-report-v1`；工作区干净；最后一条不列出迁移包、回执或 generated runs（允许既有 `runs/.gitkeep`）。

- [ ] **Step 11: 仅在真实验证修复产生新变更时提交**

若 Step 1–10 暴露并修复了任务范围内缺陷，先运行对应 RED/GREEN 回归，再提交：

```powershell
git add .gitignore AGENTS.md README.md docs/README.md docs/guides/环境与验证.md docs/guides/Windows跨机迁移指南.md tools/windows-migration tests/tools tests/unit/test_documentation_navigation.py
git commit -m "test: verify Windows migration toolkit"
```

若没有新变更，不创建空提交。
