[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ArchivePath,
    [Parameter(Mandatory)][string]$ChecksumPath,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'SurrogateLoopMigration.psm1') -Force
$watch = [Diagnostics.Stopwatch]::StartNew()
$expanded = $null
$runsDirectory = $null
$stageExitCode = 4
try {
    $root = (Get-SurrogateRepositoryRoot).Path
    $runsDirectory = Join-Path $root 'runs'
    $expanded = Expand-VerifiedRunBundle -ArchivePath $ArchivePath `
        -ChecksumPath $ChecksumPath -RunsDirectory $runsDirectory `
        -TargetRepositoryRoot $root
    $uvPath = (Get-Command uv.exe, uv -ErrorAction Stop | Select-Object -First 1).Source
    $stageExitCode = 3
    $verification = Invoke-ModelVerification -ModelKind $expanded.bundle.model_kind `
        -RunDir $expanded.run_dir -UvPath $uvPath -RepositoryRoot $root
    $published = Publish-ImportedRun -ExpandedBundle $expanded
    $result = New-MigrationResult -Status 'imported' -Stage 'import-accepted-run' `
        -Message "accepted 运行已导入：$($published.run_dir)" `
        -Evidence ([pscustomobject][ordered]@{
            commit_warning = $expanded.commit_warning
            verification = $verification
            run_id = $published.run_id
            run_dir = $published.run_dir
        }) -ExitCode 0 -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit 0
}
catch {
    $exitCode = $stageExitCode
    $migrationCode = $_.Exception.Data['MigrationExitCode']
    if ($null -ne $migrationCode) {
        $exitCode = [int]$migrationCode
    }
    if ($stageExitCode -eq 3 -and $exitCode -ne 5) {
        $exitCode = 3
    }
    $result = New-MigrationResult -Status 'error' -Stage 'import-accepted-run' `
        -Message $_.Exception.Message -Evidence @{} -ExitCode $exitCode `
        -ElapsedSeconds $watch.Elapsed.TotalSeconds
    Write-MigrationOutput -Result $result -Json:$Json
    exit $exitCode
}
finally {
    if ($null -ne $expanded -and $null -ne $runsDirectory -and
        (Test-Path -LiteralPath $expanded.staging_root)) {
        Remove-OwnedStagingDirectory -Path $expanded.staging_root `
            -RunsDirectory $runsDirectory
    }
}
