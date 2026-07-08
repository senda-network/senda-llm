param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$OutputDir = "dist",
    [string]$Flavor = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$buildBinDir = if ($env:SENDA_LLAMA_BUILD_BIN_DIR) { $env:SENDA_LLAMA_BUILD_BIN_DIR } else { Join-Path $repoRoot ".deps\llama.cpp\build\bin" }
$releaseBinDir = Join-Path $repoRoot "target\release"

Add-Type -AssemblyName System.IO.Compression.FileSystem

function Normalize-RecipeArgument {
    param(
        [AllowEmptyString()]
        [string]$Value,
        [string[]]$KnownNames = @()
    )

    if ($null -eq $Value) {
        return $Value
    }

    $normalized = $Value.Trim()
    if (-not $normalized) {
        return ""
    }

    if ($normalized -match '^(?<name>[A-Za-z_][A-Za-z0-9_-]*)=(?<value>.*)$') {
        $matchedName = $Matches.name
        $isKnownName = $KnownNames.Count -eq 0
        foreach ($knownName in $KnownNames) {
            if ($matchedName.Equals($knownName, [System.StringComparison]::OrdinalIgnoreCase)) {
                $isKnownName = $true
                break
            }
        }

        if ($isKnownName) {
            $normalized = $Matches.value
        }
    }

    if ($normalized.Length -ge 2) {
        $first = $normalized[0]
        $last = $normalized[$normalized.Length - 1]
        if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
            $normalized = $normalized.Substring(1, $normalized.Length - 2)
        }
    }

    return $normalized.Trim()
}

function Get-BinaryFlavor {
    param([string]$RequestedFlavor)

    if ($RequestedFlavor) {
        switch ($RequestedFlavor.ToLowerInvariant()) {
            "hip" { return "rocm" }
            default { return $RequestedFlavor.ToLowerInvariant() }
        }
    }

    return "cpu"
}

function Get-FlavorSuffix {
    param([string]$BinaryFlavor)

    if (-not $BinaryFlavor -or $BinaryFlavor -in @("cpu", "metal")) {
        return ""
    }

    return "-$BinaryFlavor"
}

function New-ReleaseAssetName {
    param(
        [string]$Prefix,
        [string]$TargetTriple,
        [string]$ArchiveExt,
        [string]$BinaryFlavor
    )

    return "$Prefix-$TargetTriple$(Get-FlavorSuffix $BinaryFlavor).$ArchiveExt"
}

function Get-BundleBinaryName {
    param(
        [string]$BaseName,
        [string]$BinaryFlavor
    )

    if ($BaseName -eq "senda" -or $BaseName -eq "senda") {
        return "senda.exe"
    }

    if ($BinaryFlavor) {
        return "$BaseName-$BinaryFlavor.exe"
    }

    return "$BaseName.exe"
}

function Copy-RuntimeLibs {
    param([string]$BundleDir)

    Get-ChildItem -Path $buildBinDir -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination (Join-Path $BundleDir $_.Name) -Force
    }
}

function Copy-BenchmarkBinaries {
    param([string]$BundleDir)

    Get-ChildItem -Path $releaseBinDir -Filter "membench-fingerprint*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination (Join-Path $BundleDir $_.Name) -Force
    }
}

function New-ZipArchive {
    param(
        [string]$SourceDir,
        [string]$ArchivePath
    )

    if (Test-Path $ArchivePath) {
        Remove-Item $ArchivePath -Force
    }

    $parent = Split-Path -Parent $ArchivePath
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $SourceDir,
        $ArchivePath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $true
    )
}

function Require-File {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Required file not found: $Path"
    }
}

$Version = Normalize-RecipeArgument $Version @("version")
$OutputDir = Normalize-RecipeArgument $OutputDir @("output", "output_dir", "outputdir")
$Flavor = Normalize-RecipeArgument $Flavor @("flavor", "backend")

$binaryFlavor = Get-BinaryFlavor $Flavor
$targetTriple = "x86_64-pc-windows-msvc"
$archiveExt = "zip"
$stableAsset = "senda-windows-x86_64$(Get-FlavorSuffix $binaryFlavor).$archiveExt"
$versionedAsset = "senda-$Version-windows-x86_64$(Get-FlavorSuffix $binaryFlavor).$archiveExt"

$meshBinary = Join-Path $releaseBinDir "senda.exe"
$rpcBinary = Join-Path $buildBinDir "rpc-server.exe"
$llamaBinary = Join-Path $buildBinDir "llama-server.exe"
$moeAnalyzeBinary = Join-Path $buildBinDir "llama-moe-analyze.exe"
$moeSplitBinary = Join-Path $buildBinDir "llama-moe-split.exe"

Require-File $meshBinary
Require-File $rpcBinary
Require-File $llamaBinary
Require-File $moeAnalyzeBinary
Require-File $moeSplitBinary

$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    [System.IO.Path]::GetFullPath($OutputDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}
New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null

$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("senda-release-" + [System.Guid]::NewGuid().ToString("N"))
$bundleDir = Join-Path $stagingRoot "mesh-bundle"
New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

try {
    Copy-Item $meshBinary -Destination (Join-Path $bundleDir (Get-BundleBinaryName "senda" $binaryFlavor)) -Force
    Copy-Item $rpcBinary -Destination (Join-Path $bundleDir (Get-BundleBinaryName "rpc-server" $binaryFlavor)) -Force
    Copy-Item $llamaBinary -Destination (Join-Path $bundleDir (Get-BundleBinaryName "llama-server" $binaryFlavor)) -Force
    # MoE helpers are flavor-agnostic on POSIX (one binary, no -flavor suffix);
    # mirror that on Windows so senda.exe finds them by their plain name
    # — `desktop/src/mesh.rs::MOVE_PREFIXES` and the runtime resolver both look
    # for `llama-moe-analyze.exe` / `llama-moe-split.exe`.
    Copy-Item $moeAnalyzeBinary -Destination (Join-Path $bundleDir "llama-moe-analyze.exe") -Force
    Copy-Item $moeSplitBinary -Destination (Join-Path $bundleDir "llama-moe-split.exe") -Force
    Copy-RuntimeLibs $bundleDir
    Copy-BenchmarkBinaries $bundleDir

    $versionedPath = Join-Path $resolvedOutputDir $versionedAsset
    $stablePath = Join-Path $resolvedOutputDir $stableAsset

    New-ZipArchive -SourceDir $bundleDir -ArchivePath $versionedPath
    New-ZipArchive -SourceDir $bundleDir -ArchivePath $stablePath

    Write-Host "Created release archives:"
    Get-ChildItem -Path $resolvedOutputDir -File | Sort-Object Name | ForEach-Object {
        Write-Host $_.FullName
    }
} finally {
    if (Test-Path $stagingRoot) {
        Remove-Item $stagingRoot -Recurse -Force
    }
}
