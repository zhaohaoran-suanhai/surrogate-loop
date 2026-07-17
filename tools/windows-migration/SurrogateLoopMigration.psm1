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
    $item = Get-Item -LiteralPath $root
    Add-Member -InputObject $item -NotePropertyName Path -NotePropertyValue $root -PassThru
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

function New-MigrationException {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Message,
        [Parameter(Mandatory)][int]$ExitCode
    )

    $exception = New-Object InvalidOperationException($Message)
    $exception.Data['MigrationExitCode'] = $ExitCode
    $exception
}

function ConvertTo-MigrationJson {
    [CmdletBinding()]
    param([Parameter(Mandatory, ValueFromPipeline)]$InputObject)

    process {
        $InputObject | ConvertTo-Json -Depth 20 -Compress
    }
}

function ConvertTo-WindowsCommandLineArgument {
    [CmdletBinding()]
    param([AllowEmptyString()][string]$Argument)

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashCount = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashCount++
            continue
        }

        if ($character -eq '"') {
            for ($index = 0; $index -lt (($backslashCount * 2) + 1); $index++) {
                [void]$builder.Append('\')
            }
            [void]$builder.Append('"')
            $backslashCount = 0
            continue
        }

        for ($index = 0; $index -lt $backslashCount; $index++) {
            [void]$builder.Append('\')
        }
        [void]$builder.Append($character)
        $backslashCount = 0
    }

    for ($index = 0; $index -lt ($backslashCount * 2); $index++) {
        [void]$builder.Append('\')
    }
    [void]$builder.Append('"')
    $builder.ToString()
}

function Invoke-FixedCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][AllowEmptyString()][AllowEmptyCollection()][string[]]$Arguments,
        [Parameter(Mandatory)][string]$WorkingDirectory
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = (($Arguments | ForEach-Object {
        ConvertTo-WindowsCommandLineArgument -Argument $_
    }) -join ' ')
    $startInfo.WorkingDirectory = [IO.Path]::GetFullPath($WorkingDirectory)
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    $startInfo.StandardOutputEncoding = $utf8
    $startInfo.StandardErrorEncoding = $utf8

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    $hadPythonUtf8 = Test-Path -LiteralPath 'Env:PYTHONUTF8'
    $previousPythonUtf8 = $env:PYTHONUTF8
    $hadPythonIoEncoding = Test-Path -LiteralPath 'Env:PYTHONIOENCODING'
    $previousPythonIoEncoding = $env:PYTHONIOENCODING
    try {
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        $started = $process.Start()
    }
    finally {
        if ($hadPythonUtf8) {
            $env:PYTHONUTF8 = $previousPythonUtf8
        }
        else {
            Remove-Item -LiteralPath 'Env:PYTHONUTF8' -WhatIf:$false -Confirm:$false
        }
        if ($hadPythonIoEncoding) {
            $env:PYTHONIOENCODING = $previousPythonIoEncoding
        }
        else {
            Remove-Item -LiteralPath 'Env:PYTHONIOENCODING' -WhatIf:$false -Confirm:$false
        }
    }
    if (-not $started) {
        throw "无法启动固定命令：$FilePath"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $stopwatch.Stop()

    [pscustomobject][ordered]@{
        file_path = $FilePath
        arguments = @($Arguments)
        exit_code = $process.ExitCode
        stdout = $stdout
        stderr = $stderr
        elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
    }
}

function Get-FileManifest {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Root)

    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    if (((Get-Item -LiteralPath $resolvedRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "运行目录不能是重解析点：$resolvedRoot"
    }
    $prefix = $resolvedRoot.TrimEnd('\') + '\'
    $children = @(Get-ChildItem -LiteralPath $resolvedRoot -Force -Recurse)
    foreach ($item in $children) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "运行目录不能包含重解析点：$($item.FullName)"
        }
    }

    $items = New-Object 'System.Collections.Generic.List[object]'
    foreach ($file in $children | Where-Object { -not $_.PSIsContainer }) {
        if (-not $file.FullName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "文件离开运行目录：$($file.FullName)"
        }
        [void]$items.Add([pscustomobject][ordered]@{
            path = $file.FullName.Substring($prefix.Length).Replace('\', '/')
            bytes = [int64]$file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        })
    }
    $items.Sort([Comparison[object]]{
        param($left, $right)
        [StringComparer]::Ordinal.Compare([string]$left.path, [string]$right.path)
    })
    foreach ($item in $items) {
        Write-Output $item
    }
}

function Test-FileManifest {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)]$Files
    )

    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    $prefix = $resolvedRoot.TrimEnd('\') + '\'
    $paths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $expectedPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in @($Files)) {
        $relativePath = [string]$entry.path
        if ([string]::IsNullOrWhiteSpace($relativePath) -or [IO.Path]::IsPathRooted($relativePath)) {
            throw "清单包含无效路径：$relativePath"
        }
        $segments = $relativePath -split '[\\/]'
        if ($segments -contains '..') {
            throw "清单路径不能离开运行目录：$relativePath"
        }
        $candidate = [IO.Path]::GetFullPath((Join-Path $resolvedRoot $relativePath.Replace('/', '\')))
        if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "清单路径离开运行目录：$relativePath"
        }
        if (-not $paths.Add($candidate)) {
            throw "清单包含重复路径：$relativePath"
        }
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "清单文件不存在：$relativePath"
        }
        $file = Get-Item -LiteralPath $candidate
        if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "清单文件不能是重解析点：$relativePath"
        }
        $hash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne [string]$entry.sha256) {
            throw "文件 SHA-256 与清单不一致：$relativePath"
        }
        if ([int64]$file.Length -ne [int64]$entry.bytes) {
            throw "文件大小与清单不一致：$relativePath"
        }
        [void]$expectedPaths.Add($relativePath.Replace('\', '/'))
    }

    $actualFiles = @(Get-FileManifest -Root $resolvedRoot)
    if ($actualFiles.Count -ne $expectedPaths.Count) {
        throw '运行目录中的普通文件集合与清单不一致'
    }
    foreach ($actual in $actualFiles) {
        if (-not $expectedPaths.Contains([string]$actual.path)) {
            throw "运行目录包含清单外文件：$($actual.path)"
        }
    }
}

function Test-SafeZipEntries {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$ArchivePath,
        [Parameter(Mandatory)][string]$DestinationRoot
    )

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $destination = [IO.Path]::GetFullPath($DestinationRoot)
    if (Test-Path -LiteralPath $destination) {
        $destinationItem = Get-Item -LiteralPath $destination
        if (($destinationItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "ZIP destination root cannot be a reparse point: $destination"
        }
    }
    $trimCharacters = [char[]]@([char]92, [char]47)
    $destinationPrefix = $destination.TrimEnd($trimCharacters) + [IO.Path]::DirectorySeparatorChar
    $destinationPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        foreach ($entry in $archive.Entries) {
            $name = [string]$entry.FullName
            if ([string]::IsNullOrEmpty($name)) {
                throw 'ZIP contains an empty entry name'
            }
            if ([IO.Path]::IsPathRooted($name) -or $name -match '^[A-Za-z]:' -or $name -match '^[\\/]') {
                throw "ZIP contains an absolute entry path: $name"
            }
            if ($name.Contains(':')) {
                throw "ZIP entry contains a colon: $name"
            }
            if (($name -split '[\\/]') -contains '..') {
                throw "ZIP entry leaves the destination root: $name"
            }
            $candidate = [IO.Path]::GetFullPath((Join-Path $destination $name.Replace('/', '\')))
            if (-not $candidate.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "ZIP entry leaves the destination root: $name"
            }
            if (-not $destinationPaths.Add($candidate)) {
                throw "ZIP contains a duplicate destination path: $name"
            }
        }
    }
    finally {
        $archive.Dispose()
    }
}

function Write-MigrationOutput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]$Result,
        [switch]$Json,
        [Alias('OutputStream')]
        [ValidateSet('Output', 'Information', 'Error')]
        [string]$Stream = 'Information'
    )

    if ($Json) {
        [Console]::Out.WriteLine(($Result | ConvertTo-MigrationJson))
        return
    }

    $message = "状态：$($Result.status)；阶段：$($Result.stage)；$($Result.message)"
    switch ($Stream) {
        'Output' { Write-Output $message }
        'Information' { Write-Information $message -InformationAction Continue }
        'Error' { Write-Error $message }
    }
}

function Find-CondaExecutable {
    [CmdletBinding()]
    param()

    foreach ($commandName in @('conda.exe', 'conda')) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $command) {
            foreach ($candidate in @($command.Path, $command.Source, $command.Definition)) {
                if (-not [string]::IsNullOrWhiteSpace([string]$candidate) -and
                    (Test-Path -LiteralPath $candidate -PathType Leaf)) {
                    return [IO.Path]::GetFullPath([string]$candidate)
                }
            }
        }
    }

    $userProfile = [Environment]::GetFolderPath('UserProfile')
    foreach ($relativePath in @(
        'miniforge3\Scripts\conda.exe',
        'Miniforge3\Scripts\conda.exe',
        'AppData\Local\miniforge3\Scripts\conda.exe'
    )) {
        $candidate = Join-Path $userProfile $relativePath
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return [IO.Path]::GetFullPath($candidate)
        }
    }
    return $null
}

function New-PrerequisiteCheck {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][bool]$Passed,
        [AllowNull()]$Evidence,
        [Parameter(Mandatory)][string]$Guidance
    )

    [pscustomobject][ordered]@{
        name = $Name
        status = if ($Passed) { 'pass' } else { 'fail' }
        evidence = $Evidence
        guidance = $Guidance
    }
}

function Get-ApplicationPath {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Names)

    foreach ($name in $Names) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $command) {
            continue
        }
        foreach ($candidate in @($command.Path, $command.Source, $command.Definition)) {
            if (-not [string]::IsNullOrWhiteSpace([string]$candidate) -and
                (Test-Path -LiteralPath $candidate -PathType Leaf)) {
                return [IO.Path]::GetFullPath([string]$candidate)
            }
        }
    }
    return $null
}

function Get-PrerequisiteReport {
    [CmdletBinding()]
    param()

    $repositoryRoot = $null
    $repositoryEvidence = $null
    $repositoryFound = $false
    try {
        $repositoryRoot = (Get-SurrogateRepositoryRoot).Path
        $repositoryEvidence = $repositoryRoot
        $repositoryFound = $true
    }
    catch {
        $repositoryEvidence = $_.Exception.Message
    }

    $windowsVersion = [Environment]::OSVersion.Version
    $windows11 = [Environment]::Is64BitOperatingSystem -and $windowsVersion.Build -ge 22000
    $windowsEvidence = [pscustomobject][ordered]@{
        version = $windowsVersion.ToString()
        build = $windowsVersion.Build
        architecture = if ([Environment]::Is64BitOperatingSystem) { 'x64' } else { 'non-x64' }
    }

    $git = Get-ApplicationPath -Names @('git.exe', 'git')
    $uv = Get-ApplicationPath -Names @('uv.exe', 'uv')
    $conda = Find-CondaExecutable

    $programFilesX86 = [Environment]::GetFolderPath('ProgramFilesX86')
    $vswhere = Join-Path $programFilesX86 'Microsoft Visual Studio\Installer\vswhere.exe'
    $msvcFound = $false
    $vsEvidence = if (Test-Path -LiteralPath $vswhere -PathType Leaf) { $vswhere } else { 'vswhere.exe 未找到' }
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        try {
            $vsResult = Invoke-FixedCommand -FilePath $vswhere -Arguments @(
                '-latest', '-products', '*', '-requires',
                'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'
            ) -WorkingDirectory $programFilesX86
            $vsEvidence = $vsResult.stdout.Trim()
            $msvcFound = $vsResult.exit_code -eq 0 -and -not [string]::IsNullOrWhiteSpace($vsEvidence)
            if (-not $msvcFound -and -not [string]::IsNullOrWhiteSpace($vsResult.stderr)) {
                $vsEvidence = $vsResult.stderr.Trim()
            }
        }
        catch {
            $vsEvidence = $_.Exception.Message
        }
    }

    $sdkInclude = Join-Path $programFilesX86 'Windows Kits\10\Include'
    $sdkVersions = @()
    if (Test-Path -LiteralPath $sdkInclude -PathType Container) {
        $sdkVersions = @(Get-ChildItem -LiteralPath $sdkInclude -Directory -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Name)
    }
    $sdkFound = $sdkVersions.Count -gt 0
    $sdkEvidence = if ($sdkFound) { $sdkVersions -join ', ' } else { $sdkInclude }

    $nvidiaSmi = Get-ApplicationPath -Names @('nvidia-smi.exe', 'nvidia-smi')
    $gpuFound = $false
    $gpuEvidence = if ($null -eq $nvidiaSmi) { 'nvidia-smi 未找到' } else { $nvidiaSmi }
    if ($null -ne $nvidiaSmi) {
        try {
            $gpuResult = Invoke-FixedCommand -FilePath $nvidiaSmi -Arguments @(
                '--query-gpu=name,driver_version,memory.total',
                '--format=csv,noheader,nounits'
            ) -WorkingDirectory $(if ($repositoryFound) { $repositoryRoot } else { $PWD.Path })
            $gpuEvidence = $gpuResult.stdout.Trim()
            $gpuFound = $gpuResult.exit_code -eq 0 -and -not [string]::IsNullOrWhiteSpace($gpuEvidence)
            if (-not $gpuFound -and -not [string]::IsNullOrWhiteSpace($gpuResult.stderr)) {
                $gpuEvidence = $gpuResult.stderr.Trim()
            }
        }
        catch {
            $gpuEvidence = $_.Exception.Message
        }
    }

    $diskProbeSucceeded = $false
    $diskEvidence = '仓库根目录不可用，未检查磁盘空间'
    if ($repositoryFound) {
        try {
            $rootPath = [IO.Path]::GetPathRoot($repositoryRoot)
            $drive = New-Object IO.DriveInfo($rootPath)
            $freeGiB = [Math]::Round($drive.AvailableFreeSpace / 1GB, 2)
            $diskProbeSucceeded = $freeGiB -ge 20
            $diskEvidence = [pscustomobject][ordered]@{
                drive = $rootPath
                free_gib = $freeGiB
                required_gib = 20
            }
        }
        catch {
            $diskEvidence = $_.Exception.Message
        }
    }

    $checks = @(
        New-PrerequisiteCheck 'windows11' $windows11 $windowsEvidence '需要 Windows 11 x64。'
        New-PrerequisiteCheck 'powershell' ($PSVersionTable.PSVersion.Major -ge 5) $PSVersionTable.PSVersion.ToString() '需要 Windows PowerShell 5.1 或 PowerShell 7。'
        New-PrerequisiteCheck 'git' ($null -ne $git) $git '安装 Git for Windows 后重新打开终端。'
        New-PrerequisiteCheck 'uv' ($null -ne $uv) $uv '按 uv 官方说明安装后重新打开终端。'
        New-PrerequisiteCheck 'conda' ($null -ne $conda) $conda '安装 Miniforge 后重新打开终端。'
        New-PrerequisiteCheck 'msvc' $msvcFound $vsEvidence '安装 Visual Studio 2022 Build Tools 的“使用 C++ 的桌面开发”。'
        New-PrerequisiteCheck 'windows_sdk' $sdkFound $sdkEvidence '在 Build Tools 中安装 Windows SDK。'
        New-PrerequisiteCheck 'nvidia_gpu' $gpuFound $gpuEvidence '安装兼容 NVIDIA 驱动并确认 nvidia-smi 可用。'
        New-PrerequisiteCheck 'repository' $repositoryFound $repositoryEvidence '从完整 Git 克隆运行工具。'
        New-PrerequisiteCheck 'disk' $diskProbeSucceeded $diskEvidence '确认磁盘至少有 20 GiB 可用空间，以容纳 uv/Conda 缓存、两个环境和新样本。'
    )
    $failed = @($checks | Where-Object { $_.status -ne 'pass' })
    [pscustomobject][ordered]@{
        status = if ($failed.Count -eq 0) { 'pass' } else { 'fail' }
        checks = $checks
        summary = if ($failed.Count -eq 0) {
            '系统前置条件检查通过。'
        }
        else {
            "有 $($failed.Count) 项前置条件未通过，请按 guidance 处理。"
        }
    }
}

function New-CommandSpec {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][AllowEmptyCollection()][string[]]$Arguments,
        [Parameter(Mandatory)][string]$WorkingDirectory
    )

    [pscustomobject][ordered]@{
        name = $Name
        file_path = $FilePath
        arguments = @($Arguments)
        working_directory = [IO.Path]::GetFullPath($WorkingDirectory)
    }
}

function Get-EnvironmentPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][bool]$CondaEnvironmentExists,
        [string]$UvPath = 'uv',
        [string]$CondaPath = 'conda',
        [string]$RepositoryRoot = (Get-SurrogateRepositoryRoot).Path
    )

    $root = [IO.Path]::GetFullPath($RepositoryRoot)
    $environmentFile = Join-Path $root 'environments\fenicsx-0.11.yml'
    @(
        New-CommandSpec 'uv-python' $UvPath @('python', 'pin', '3.11') $root
        New-CommandSpec 'uv-sync' $UvPath @('sync', '--extra', 'operator', '--all-groups') $root
        if ($CondaEnvironmentExists) {
            New-CommandSpec 'conda-environment' $CondaPath @(
                'env', 'update', '-n', 'surrogate-loop-fenicsx-0.11', '-f', $environmentFile
            ) $root
        }
        else {
            New-CommandSpec 'conda-environment' $CondaPath @('env', 'create', '-f', $environmentFile) $root
        }
        New-CommandSpec 'python-imports' $UvPath @(
            'run', 'python', '-c',
            'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
        ) $root
        New-CommandSpec 'fenicsx-doctor' $UvPath @(
            'run', 'surrogate-loop', 'elasticity2d', 'doctor'
        ) $root
    )
}

function Get-ModelVerificationPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('scalar', 'heat1d', 'elasticity2d')]
        [string]$ModelKind,
        [Parameter(Mandatory)][string]$RunDir,
        [string]$UvPath = 'uv',
        [string]$RepositoryRoot = (Get-SurrogateRepositoryRoot).Path
    )

    $resolvedRun = (Resolve-Path -LiteralPath $RunDir).Path
    if (-not (Test-Path -LiteralPath $resolvedRun -PathType Container)) {
        throw "accepted 运行目录不存在：$RunDir"
    }
    if (((Get-Item -LiteralPath $resolvedRun).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "accepted 运行目录不能是重解析点：$resolvedRun"
    }
    $root = [IO.Path]::GetFullPath($RepositoryRoot)

    switch ($ModelKind) {
        'scalar' {
            $report = @('run', 'surrogate-loop', 'report', '--run-dir', $resolvedRun)
            $predict = @(
                'run', 'surrogate-loop', 'predict', '--run-dir', $resolvedRun,
                '--gamma', '0.35'
            )
        }
        'heat1d' {
            $report = @('run', 'surrogate-loop', 'operator', 'report', '--run-dir', $resolvedRun)
            $predict = @(
                'run', 'surrogate-loop', 'operator', 'predict', '--run-dir', $resolvedRun,
                '--alpha', '0.1', '--a', '1.0', '--b', '0.1', '--x', '0.5', '--t', '0.25'
            )
        }
        'elasticity2d' {
            $report = @('run', 'surrogate-loop', 'elasticity2d', 'report', '--run-dir', $resolvedRun)
            $predict = @(
                'run', 'surrogate-loop', 'elasticity2d', 'predict', '--run-dir', $resolvedRun,
                '--e', '3', '--nu', '0.3', '--p', '0.006', '--theta', '-1.5707963268',
                '--y0', '0.5', '--w', '0.12', '--x', '4', '--y', '0.5'
            )
        }
    }

    @(
        New-CommandSpec 'model-report' $UvPath $report $root
        New-CommandSpec 'model-predict' $UvPath $predict $root
    )
}

function Invoke-ModelVerification {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('scalar', 'heat1d', 'elasticity2d')]
        [string]$ModelKind,
        [Parameter(Mandatory)][string]$RunDir,
        [string]$UvPath = 'uv',
        [string]$RepositoryRoot = (Get-SurrogateRepositoryRoot).Path
    )

    $plan = @(Get-ModelVerificationPlan -ModelKind $ModelKind -RunDir $RunDir `
        -UvPath $UvPath -RepositoryRoot $RepositoryRoot)
    $outputs = New-Object 'System.Collections.Generic.List[object]'
    foreach ($command in $plan) {
        $completed = Invoke-FixedCommand -FilePath $command.file_path `
            -Arguments $command.arguments -WorkingDirectory $command.working_directory
        if ($completed.exit_code -ne 0) {
            throw "accepted 验证阶段 $($command.name) 失败：$($completed.stderr)"
        }
        try {
            $payload = $completed.stdout | ConvertFrom-Json
        }
        catch {
            throw "accepted 验证阶段 $($command.name) 未返回合法 JSON：$($_.Exception.Message)"
        }
        [void]$outputs.Add($payload)
    }

    $reportPayload = $outputs[0]
    $stateProperty = $reportPayload.PSObject.Properties['state']
    $statusProperty = $reportPayload.PSObject.Properties['status']
    $acceptedState = if ($null -ne $stateProperty) {
        [string]$stateProperty.Value
    }
    elseif ($null -ne $statusProperty) {
        [string]$statusProperty.Value
    }
    else {
        ''
    }
    if ($acceptedState -ne 'accepted') {
        throw "运行报告状态不是 accepted：$acceptedState"
    }

    [pscustomobject][ordered]@{
        status = 'accepted'
        report = $reportPayload
        prediction = $outputs[1]
        model_kind = $ModelKind
        run_dir = (Resolve-Path -LiteralPath $RunDir).Path
    }
}

function Get-InstallationPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('Prerequisites', 'Python', 'Fenicsx', 'FullChain')]
        [string]$Level,
        [string]$AcceptedRunDir,
        [ValidateSet('scalar', 'heat1d', 'elasticity2d')]
        [string]$ModelKind,
        [string]$UvPath = 'uv',
        [string]$CondaPath = 'conda',
        [string]$RepositoryRoot = (Get-SurrogateRepositoryRoot).Path
    )

    $root = [IO.Path]::GetFullPath($RepositoryRoot)
    if ($Level -eq 'FullChain') {
        if ([string]::IsNullOrWhiteSpace($AcceptedRunDir) -or
            [string]::IsNullOrWhiteSpace($ModelKind)) {
            throw 'FullChain 必须同时提供 AcceptedRunDir 和 ModelKind。'
        }
        $null = Resolve-Path -LiteralPath $AcceptedRunDir
    }

    $python = @(
        New-CommandSpec 'cli-help' $UvPath @('run', 'surrogate-loop', '--help') $root
        New-CommandSpec 'cli-version' $UvPath @(
            'run', 'python', '-m', 'surrogate_loop', '--version'
        ) $root
        New-CommandSpec 'cuda-backward' $UvPath @(
            'run', 'python', '-c',
            "import torch; x=torch.randn(128,128,device='cuda',requires_grad=True); y=x.square().mean(); y.backward(); print(torch.cuda.get_device_name(0), torch.isfinite(x.grad).all().item())"
        ) $root
        New-CommandSpec 'ruff' $UvPath @('run', 'ruff', 'check', '.') $root
        New-CommandSpec 'pytest' $UvPath @('run', 'pytest', '-q') $root
    )
    $fenicsx = @(
        New-CommandSpec 'fenicsx-doctor' $UvPath @(
            'run', 'surrogate-loop', 'elasticity2d', 'doctor'
        ) $root
        New-CommandSpec 'solver-tests' $CondaPath @(
            'run', '-n', 'surrogate-loop-fenicsx-0.11', 'python', '-m', 'pytest',
            'tests/solver/elasticity2d', '-v'
        ) $root
    )
    $fullChain = @(
        New-CommandSpec 'real-fenicsx-e2e' $UvPath @(
            'run', 'pytest', 'tests/e2e/test_elasticity2d_fenicsx_loop.py', '-v'
        ) $root
    )

    switch ($Level) {
        'Prerequisites' { return @() }
        'Python' { return @($python) }
        'Fenicsx' { return @($python + $fenicsx) }
        'FullChain' { return @($python + $fenicsx + $fullChain) }
    }
}

function New-RunBundleArchive {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$RunDir,
        [Parameter(Mandatory)]
        [ValidateSet('scalar', 'heat1d', 'elasticity2d')]
        [string]$ModelKind,
        [Parameter(Mandatory)][string]$OutputDirectory,
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)]$Verification
    )

    $statusProperty = $Verification.PSObject.Properties['status']
    if ($null -eq $statusProperty -or [string]$statusProperty.Value -ne 'accepted') {
        throw (New-MigrationException -Message '只有 accepted 验证结果可以导出。' -ExitCode 2)
    }

    $resolvedRun = (Resolve-Path -LiteralPath $RunDir).Path
    if (-not (Test-Path -LiteralPath $resolvedRun -PathType Container)) {
        throw (New-MigrationException -Message "运行目录不存在：$RunDir" -ExitCode 2)
    }
    $runId = Split-Path -Leaf $resolvedRun
    if ($runId -notmatch '^[A-Za-z0-9._-]+$') {
        throw (New-MigrationException -Message "run_id 包含不允许的字符：$runId" -ExitCode 2)
    }

    $resolvedOutput = (Resolve-Path -LiteralPath $OutputDirectory).Path
    if (-not (Test-Path -LiteralPath $resolvedOutput -PathType Container)) {
        throw (New-MigrationException -Message "输出目录不存在：$OutputDirectory" -ExitCode 2)
    }
    if (((Get-Item -LiteralPath $resolvedOutput).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw (New-MigrationException -Message "输出目录不能是重解析点：$resolvedOutput" -ExitCode 2)
    }

    $archiveName = "$runId.surrogate-run.zip"
    $checksumName = "$runId.surrogate-run.sha256.json"
    $archivePath = Join-Path $resolvedOutput $archiveName
    $checksumPath = Join-Path $resolvedOutput $checksumName
    if ((Test-Path -LiteralPath $archivePath) -or (Test-Path -LiteralPath $checksumPath)) {
        throw (New-MigrationException -Message '导出 archive 或 checksum 已存在，拒绝覆盖。' -ExitCode 5)
    }

    $files = @(Get-FileManifest -Root $resolvedRun)
    $resolvedRepository = (Resolve-Path -LiteralPath $RepositoryRoot).Path
    $gitPath = Get-ApplicationPath -Names @('git.exe', 'git')
    if ($null -eq $gitPath) {
        throw (New-MigrationException -Message '未找到 Git，无法记录导出提交。' -ExitCode 3)
    }
    $commitResult = Invoke-FixedCommand -FilePath $gitPath `
        -Arguments @('-C', $resolvedRepository, 'rev-parse', 'HEAD') `
        -WorkingDirectory $resolvedRepository
    if ($commitResult.exit_code -ne 0) {
        throw (New-MigrationException -Message "读取 Git 提交失败：$($commitResult.stderr)" -ExitCode 3)
    }
    $statusResult = Invoke-FixedCommand -FilePath $gitPath `
        -Arguments @('-C', $resolvedRepository, 'status', '--porcelain') `
        -WorkingDirectory $resolvedRepository
    if ($statusResult.exit_code -ne 0) {
        throw (New-MigrationException -Message "读取 Git 状态失败：$($statusResult.stderr)" -ExitCode 3)
    }
    $commit = $commitResult.stdout.Trim()
    $dirty = -not [string]::IsNullOrWhiteSpace($statusResult.stdout)
    $bundle = [pscustomobject][ordered]@{
        schema_version = 1
        model_kind = $ModelKind
        run_id = $runId
        export_repo_commit = $commit
        export_repo_dirty = [bool]$dirty
        created_at_utc = [DateTime]::UtcNow.ToString('o')
        files = $files
    }

    $stagingRoot = Join-Path ([IO.Path]::GetTempPath()) `
        ('surrogate-loop-export-' + [guid]::NewGuid().ToString('N'))
    $contentRoot = Join-Path $stagingRoot 'content'
    $stagingRun = Join-Path $contentRoot (Join-Path 'run' $runId)
    $stagingArchive = Join-Path $stagingRoot $archiveName
    $stagingChecksum = Join-Path $stagingRoot $checksumName
    $archivePublished = $false
    try {
        $null = New-Item -ItemType Directory -Path $stagingRun
        $bundlePath = Join-Path $contentRoot 'bundle.json'
        [IO.File]::WriteAllText(
            $bundlePath,
            ($bundle | ConvertTo-Json -Depth 20 -Compress),
            (New-Object Text.UTF8Encoding($false))
        )
        foreach ($file in $files) {
            $relative = ([string]$file.path).Replace('/', '\')
            $source = Join-Path $resolvedRun $relative
            $destination = Join-Path $stagingRun $relative
            $destinationParent = Split-Path -Parent $destination
            if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) {
                $null = New-Item -ItemType Directory -Path $destinationParent
            }
            [IO.File]::Copy($source, $destination, $false)
        }

        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::CreateFromDirectory(
            $contentRoot,
            $stagingArchive,
            [IO.Compression.CompressionLevel]::Optimal,
            $false
        )
        $sidecar = [pscustomobject][ordered]@{
            schema_version = 1
            archive_name = $archiveName
            archive_bytes = [int64](Get-Item -LiteralPath $stagingArchive).Length
            archive_sha256 = (Get-FileHash -LiteralPath $stagingArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        [IO.File]::WriteAllText(
            $stagingChecksum,
            ($sidecar | ConvertTo-Json -Depth 10 -Compress),
            (New-Object Text.UTF8Encoding($false))
        )

        [IO.File]::Move($stagingArchive, $archivePath)
        $archivePublished = $true
        [IO.File]::Move($stagingChecksum, $checksumPath)

        [pscustomobject][ordered]@{
            archive_path = $archivePath
            checksum_path = $checksumPath
            bundle = $bundle
            verification = $Verification
        }
    }
    catch {
        if ($archivePublished -and (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
            Remove-Item -LiteralPath $archivePath -Force -Confirm:$false
        }
        if ($null -eq $_.Exception.Data['MigrationExitCode']) {
            $_.Exception.Data['MigrationExitCode'] = 4
        }
        throw
    }
    finally {
        $resolvedStaging = [IO.Path]::GetFullPath($stagingRoot)
        $tempPrefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
        $stagingName = Split-Path -Leaf $resolvedStaging
        if ((Test-Path -LiteralPath $resolvedStaging) -and
            $resolvedStaging.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -and
            $stagingName -match '^surrogate-loop-export-[0-9a-f]{32}$' -and
            ((Get-Item -LiteralPath $resolvedStaging).Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
            Remove-Item -LiteralPath $resolvedStaging -Recurse -Force -Confirm:$false
        }
    }
}

Export-ModuleMember -Function @(
    'Get-SurrogateRepositoryRoot',
    'New-MigrationResult',
    'ConvertTo-MigrationJson',
    'Invoke-FixedCommand',
    'Get-FileManifest',
    'Test-FileManifest',
    'Test-SafeZipEntries',
    'Write-MigrationOutput',
    'Find-CondaExecutable',
    'Get-PrerequisiteReport',
    'Get-EnvironmentPlan',
    'Get-ModelVerificationPlan',
    'Invoke-ModelVerification',
    'Get-InstallationPlan',
    'New-RunBundleArchive'
)
