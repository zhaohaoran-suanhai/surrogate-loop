[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('Prerequisites', 'Python', 'Fenicsx', 'FullChain')]
    [string]$Level,
    [string]$AcceptedRunDir,
    [ValidateSet('scalar', 'heat1d', 'elasticity2d')]
    [string]$ModelKind,
    [string]$ReportPath,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'SurrogateLoopMigration.psm1') -Force

function Write-NewUtf8File {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Content
    )

    $stream = [IO.File]::Open(
        $Path,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None
    )
    try {
        $writer = New-Object IO.StreamWriter(
            $stream,
            (New-Object Text.UTF8Encoding($false))
        )
        try {
            $writer.Write($Content)
            $writer.Flush()
        }
        finally {
            $writer.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

$watch = [Diagnostics.Stopwatch]::StartNew()
$stageExitCode = 2
$resolvedReportPath = $null

if (-not [string]::IsNullOrWhiteSpace($ReportPath)) {
    if (Test-Path -LiteralPath $ReportPath) {
        $result = New-MigrationResult -Status 'error' -Stage 'installation' `
            -Message "拒绝覆盖已有验证报告：$ReportPath" -Evidence @{} -ExitCode 5 `
            -ElapsedSeconds $watch.Elapsed.TotalSeconds
        Write-MigrationOutput -Result $result -Json:$Json
        exit 5
    }
    $resolvedReportPath = if ([IO.Path]::IsPathRooted($ReportPath)) {
        [IO.Path]::GetFullPath($ReportPath)
    }
    else {
        [IO.Path]::GetFullPath((Join-Path $PWD.Path $ReportPath))
    }
}

try {
    $prerequisites = Get-PrerequisiteReport
    $prerequisiteExitCode = if ($prerequisites.status -eq 'pass') { 0 } else { 2 }
    if ($Level -eq 'Prerequisites') {
        $result = New-MigrationResult -Status $prerequisites.status `
            -Stage 'installation-prerequisites' -Message $prerequisites.summary `
            -Evidence $prerequisites -ExitCode $prerequisiteExitCode `
            -ElapsedSeconds $watch.Elapsed.TotalSeconds
        if ($prerequisiteExitCode -eq 0 -and $null -ne $resolvedReportPath) {
            $stageExitCode = 5
            Write-NewUtf8File -Path $resolvedReportPath `
                -Content ($result | ConvertTo-MigrationJson)
        }
        Write-MigrationOutput -Result $result -Json:$Json
        exit $prerequisiteExitCode
    }
    if ($prerequisites.status -ne 'pass') {
        $result = New-MigrationResult -Status 'fail' -Stage 'installation' `
            -Message '前置条件未通过，未执行安装验证命令。' `
            -Evidence $prerequisites -ExitCode 2 -ElapsedSeconds $watch.Elapsed.TotalSeconds
        Write-MigrationOutput -Result $result -Json:$Json
        exit 2
    }

    $uvPath = (Get-Command uv.exe, uv -ErrorAction Stop | Select-Object -First 1).Source
    $condaPath = Find-CondaExecutable
    if ([string]::IsNullOrWhiteSpace($condaPath)) {
        throw '未找到可执行的 Conda。'
    }
    $repositoryRoot = (Get-SurrogateRepositoryRoot).Path
    $planParameters = @{
        Level = $Level
        UvPath = $uvPath
        CondaPath = $condaPath
        RepositoryRoot = $repositoryRoot
    }
    if ($Level -eq 'FullChain') {
        $planParameters['AcceptedRunDir'] = $AcceptedRunDir
        $planParameters['ModelKind'] = $ModelKind
    }
    $plan = @(Get-InstallationPlan @planParameters)
    $stageExitCode = 3
    $commandResults = New-Object 'System.Collections.Generic.List[object]'
    $verification = $null

    foreach ($command in $plan | Where-Object { $_.name -ne 'real-fenicsx-e2e' }) {
        $completed = Invoke-FixedCommand -FilePath $command.file_path `
            -Arguments $command.arguments -WorkingDirectory $command.working_directory
        [void]$commandResults.Add($completed)
        if ($completed.exit_code -ne 0) {
            throw "安装验证阶段 $($command.name) 失败：$($completed.stderr)"
        }
    }

    if ($Level -eq 'FullChain') {
        $e2eCommand = $plan | Where-Object { $_.name -eq 'real-fenicsx-e2e' } |
            Select-Object -First 1
        $hadE2E = Test-Path Env:SURROGATE_LOOP_RUN_FENICSX_E2E
        $previousE2E = if ($hadE2E) { $env:SURROGATE_LOOP_RUN_FENICSX_E2E } else { $null }
        try {
            $env:SURROGATE_LOOP_RUN_FENICSX_E2E = '1'
            $completed = Invoke-FixedCommand -FilePath $e2eCommand.file_path `
                -Arguments $e2eCommand.arguments -WorkingDirectory $e2eCommand.working_directory
            [void]$commandResults.Add($completed)
            if ($completed.exit_code -ne 0) {
                throw "安装验证阶段 $($e2eCommand.name) 失败：$($completed.stderr)"
            }
            $verification = Invoke-ModelVerification -ModelKind $ModelKind `
                -RunDir $AcceptedRunDir -UvPath $uvPath -RepositoryRoot $repositoryRoot
        }
        finally {
            if ($hadE2E) {
                $env:SURROGATE_LOOP_RUN_FENICSX_E2E = $previousE2E
            }
            else {
                Remove-Item Env:SURROGATE_LOOP_RUN_FENICSX_E2E `
                    -ErrorAction SilentlyContinue -Confirm:$false
            }
        }
    }

    $result = New-MigrationResult -Status 'pass' -Stage "installation-$($Level.ToLowerInvariant())" `
        -Message "安装验证等级 $Level 通过。" `
        -Evidence ([pscustomobject][ordered]@{
            level = $Level
            prerequisites = $prerequisites
            commands = $commandResults.ToArray()
            verification = $verification
        }) -ExitCode 0 -ElapsedSeconds $watch.Elapsed.TotalSeconds
    if ($null -ne $resolvedReportPath) {
        $stageExitCode = 5
        Write-NewUtf8File -Path $resolvedReportPath `
            -Content ($result | ConvertTo-MigrationJson)
    }
    Write-MigrationOutput -Result $result -Json:$Json
    exit 0
}
catch {
    $result = New-MigrationResult -Status 'error' -Stage 'installation' `
        -Message $_.Exception.Message -Evidence @{} -ExitCode $stageExitCode `
        -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit $stageExitCode
}
