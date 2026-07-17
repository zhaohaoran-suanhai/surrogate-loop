[CmdletBinding()]
param([switch]$Json)

$ErrorActionPreference = 'Stop'
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
}
catch {
    $result = New-MigrationResult -Status 'error' -Stage 'prerequisites' `
        -Message $_.Exception.Message -Evidence @{} -ExitCode 2 `
        -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit 2
}
