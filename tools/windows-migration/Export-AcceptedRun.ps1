[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RunDir,
    [Parameter(Mandatory)]
    [ValidateSet('scalar', 'heat1d', 'elasticity2d')]
    [string]$ModelKind,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'SurrogateLoopMigration.psm1') -Force
$watch = [Diagnostics.Stopwatch]::StartNew()
$stageExitCode = 2
try {
    $repositoryRoot = (Get-SurrogateRepositoryRoot).Path
    $uvPath = (Get-Command uv.exe, uv -ErrorAction Stop | Select-Object -First 1).Source
    $stageExitCode = 3
    $verification = Invoke-ModelVerification -ModelKind $ModelKind -RunDir $RunDir `
        -UvPath $uvPath -RepositoryRoot $repositoryRoot
    $bundle = New-RunBundleArchive -RunDir $RunDir -ModelKind $ModelKind `
        -OutputDirectory $OutputDirectory -RepositoryRoot $repositoryRoot `
        -Verification $verification
    $result = New-MigrationResult -Status 'accepted' -Stage 'export-accepted-run' `
        -Message "accepted 运行已导出：$($bundle.archive_path)" `
        -Evidence $bundle -ExitCode 0 -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit 0
}
catch {
    $exitCode = $stageExitCode
    $migrationCode = $_.Exception.Data['MigrationExitCode']
    if ($null -ne $migrationCode) {
        $exitCode = [int]$migrationCode
    }
    $result = New-MigrationResult -Status 'error' -Stage 'export-accepted-run' `
        -Message $_.Exception.Message -Evidence @{} -ExitCode $exitCode `
        -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit $exitCode
}
