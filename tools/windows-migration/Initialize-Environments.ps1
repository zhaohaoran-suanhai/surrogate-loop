[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param([switch]$Json)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'SurrogateLoopMigration.psm1') -Force
$watch = [Diagnostics.Stopwatch]::StartNew()
$stageExitCode = 2
$requestedWhatIf = [bool]$WhatIfPreference
try {
    $prerequisites = Get-PrerequisiteReport
    if ($prerequisites.status -ne 'pass') {
        $result = New-MigrationResult -Status 'fail' -Stage 'initialize-environments' `
            -Message '前置条件未通过，未执行任何环境命令。' `
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
    $condaList = Invoke-FixedCommand -FilePath $condaPath `
        -Arguments @('env', 'list', '--json') -WorkingDirectory $repositoryRoot
    if ($condaList.exit_code -ne 0) {
        $stageExitCode = 3
        throw "读取 Conda 环境列表失败：$($condaList.stderr)"
    }
    $condaPayload = $condaList.stdout | ConvertFrom-Json
    $environmentExists = @($condaPayload.envs | ForEach-Object {
        [IO.Path]::GetFileName(([string]$_).TrimEnd('\', '/'))
    }) -contains 'surrogate-loop-fenicsx-0.11'
    $plan = @(Get-EnvironmentPlan -CondaEnvironmentExists $environmentExists `
        -UvPath $uvPath -CondaPath $condaPath -RepositoryRoot $repositoryRoot)

    $executed = New-Object 'System.Collections.Generic.List[object]'
    foreach ($command in $plan) {
        if ($requestedWhatIf) {
            continue
        }
        if ($PSCmdlet.ShouldProcess(
            $command.name,
            "$($command.file_path) $($command.arguments -join ' ')"
        )) {
            $stageExitCode = 3
            $completed = Invoke-FixedCommand -FilePath $command.file_path `
                -Arguments $command.arguments -WorkingDirectory $command.working_directory
            [void]$executed.Add($completed)
            if ($completed.exit_code -ne 0) {
                throw "环境阶段 $($command.name) 失败：$($completed.stderr)"
            }
        }
    }

    $planned = $requestedWhatIf
    $result = New-MigrationResult -Status $(if ($planned) { 'planned' } else { 'pass' }) `
        -Stage 'initialize-environments' `
        -Message $(if ($planned) { '已生成环境计划，WhatIf 未执行命令。' } else { '双环境初始化和导入检查完成。' }) `
        -Evidence ([pscustomobject][ordered]@{
            prerequisites = $prerequisites
            plan = $plan
            executed = $executed.ToArray()
        }) -ExitCode 0 -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit 0
}
catch {
    $result = New-MigrationResult -Status 'error' -Stage 'initialize-environments' `
        -Message $_.Exception.Message -Evidence @{} -ExitCode $stageExitCode `
        -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit $stageExitCode
}
