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
        [Parameter(Mandatory)][string[]]$Arguments,
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
    $startInfo.EnvironmentVariables['PYTHONUTF8'] = '1'
    $startInfo.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    if (-not $process.Start()) {
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

    $items = foreach ($file in $children | Where-Object { -not $_.PSIsContainer }) {
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
    $trimCharacters = [char[]]@([char]92, [char]47)
    $destinationPrefix = $destination.TrimEnd($trimCharacters) + [IO.Path]::DirectorySeparatorChar
    $entryNames = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
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
            if (-not $entryNames.Add($name)) {
                throw "ZIP contains a duplicate entry: $name"
            }
            $candidate = [IO.Path]::GetFullPath((Join-Path $destination $name.Replace('/', '\')))
            if (-not $candidate.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "ZIP entry leaves the destination root: $name"
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
