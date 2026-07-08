# release-senda.ps1 — package the senda binary for the senda.network installer (Windows).
#
# Produces dist-release\senda-windows-x86_64-<flavor>.zip, where <flavor>
# is 'cuda' (vulkan/cpu wired up later). Mirrors scripts/release-senda.sh
# for the macOS/Linux side.
#
# Bundle is SELF-CONTAINED and ships:
#   - senda.exe                          (target/release/, built by cargo)
#   - rpc-server.exe / llama-server.exe       (.deps/llama.cpp/build/bin/, Senda-patched)
#   - llama-moe-analyze.exe / llama-moe-split.exe (same, MoE tools added by patch 0003)
#   - all ggml-* / llama* / mtmd DLLs         (.deps/llama.cpp/build/bin/, when BUILD_SHARED_LIBS=ON)
#   - cudart / cublas / cublasLt / nvrtc DLLs (CUDA toolkit redist, CUDA flavor only)
#   - libomp140.x86_64.dll                    (MSVC OpenMP runtime, when present in build/bin)
#
# Pre-0.66.10 the script downloaded ggml-org/llama.cpp's official Windows
# release for rpc-server / llama-server / DLLs. That bundle has none of our
# patches (--gguf, --mesh-port, the moe tools), so compound-RAM MoE serving
# on Windows was structurally broken — every Mixtral-style host attempt
# died at "error: invalid argument: --mesh-port" or "moe-split not found".
# We now build everything from third_party/llama.cpp/patches via
# scripts/build-windows.ps1 -Backend cuda, and stage the locally built
# binaries instead.
#
# Usage:
#   powershell -NoProfile -File scripts/release-senda.ps1 -Flavor cuda [-OutputDir dist-release]

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('cuda')]
    [string]$Flavor,

    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Resolve-Path (Join-Path $scriptDir '..')
$defaultDist = Join-Path $repoRoot 'dist-release'
if (-not $OutputDir) { $OutputDir = $defaultDist }

# Windows installer only ships x86_64 today; aarch64 (Snapdragon X) is future work.
$platformSuffix = "windows-x86_64-$Flavor"
$asset = "senda-$platformSuffix.zip"
$zipPath = Join-Path $OutputDir $asset
$shaPath = "$zipPath.sha256"

$mesh        = Join-Path $repoRoot 'target\release\senda.exe'
$llamaBinDir = Join-Path $repoRoot '.deps\llama.cpp\build\bin'

if (-not (Test-Path -PathType Leaf $mesh)) {
    Write-Error "release-senda: built senda.exe not found at $mesh. Run scripts/build-windows.ps1 -Backend $Flavor first."
    exit 1
}
if (-not (Test-Path -PathType Container $llamaBinDir)) {
    Write-Error "release-senda: patched llama.cpp build dir not found at $llamaBinDir. Run scripts/build-windows.ps1 -Backend $Flavor first."
    exit 1
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

function Require-Built {
    param([string]$RelativeName)
    $p = Join-Path $llamaBinDir $RelativeName
    if (-not (Test-Path -PathType Leaf $p)) {
        throw "Expected build output not found: $p (build-windows.ps1 -Backend $Flavor must produce it)."
    }
    return $p
}

function Copy-IfExists {
    param([string]$Source, [string]$Destination)
    if (Test-Path -PathType Leaf $Source) {
        Copy-Item -Force $Source $Destination
        return $true
    }
    return $false
}

# Resolve CUDA runtime DLLs from the toolkit install. Required at runtime
# for any binary that links libcudart / libcublas (rpc-server, llama-server,
# and ggml-cuda.dll all do). The Jimver/cuda-toolkit action exports
# CUDA_PATH; locally / in interactive use we look in the standard install
# location too.
function Resolve-CudaBinDir {
    $candidates = @()
    if ($env:CUDA_PATH) { $candidates += (Join-Path $env:CUDA_PATH 'bin') }
    if ($env:CUDA_HOME) { $candidates += (Join-Path $env:CUDA_HOME 'bin') }
    if ($env:ProgramFiles) {
        $toolkitRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
        if (Test-Path $toolkitRoot) {
            Get-ChildItem -Path $toolkitRoot -Directory | Sort-Object Name -Descending | ForEach-Object {
                $candidates += (Join-Path $_.FullName 'bin')
            }
        }
    }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    throw "CUDA toolkit not found. Set CUDA_PATH or install the CUDA toolkit before running this script."
}

$stage = New-Item -ItemType Directory -Path (Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString()))
try {
    Copy-Item $mesh (Join-Path $stage 'senda.exe')

    $licensePath = Join-Path $repoRoot 'LICENSE'
    if (Test-Path $licensePath) {
        Copy-Item $licensePath (Join-Path $stage 'LICENSE')
    }

    $taskRef = Join-Path $repoRoot 'dist\senda-task.xml'
    if (Test-Path $taskRef) {
        Copy-Item $taskRef (Join-Path $stage 'senda-task.xml')
    }

    foreach ($exe in @('rpc-server.exe', 'llama-server.exe', 'llama-moe-analyze.exe', 'llama-moe-split.exe')) {
        $src = Require-Built $exe
        Copy-Item -Force $src (Join-Path $stage $exe)
    }

    # Bundle every DLL produced by the llama.cpp build (ggml-base, ggml-cpu-*,
    # ggml-cuda, ggml-rpc, llama, llama-common, mtmd, libomp140 if shipped, …).
    # When BUILD_SHARED_LIBS=ON these all live alongside the executables.
    Get-ChildItem -Path $llamaBinDir -Filter '*.dll' -File | ForEach-Object {
        Copy-Item -Force $_.FullName (Join-Path $stage $_.Name)
    }

    if ($Flavor -eq 'cuda') {
        $cudaBin = Resolve-CudaBinDir
        # Match the historical bundle: cudart, cublas, cublasLt, nvrtc.
        # Glob each so the CUDA major/minor suffix in the filename
        # (cudart64_12.dll vs cudart64_13.dll) tracks the installed toolkit
        # without needing this script edited every time CUDA bumps.
        foreach ($pattern in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll', 'nvrtc64_*.dll', 'nvrtc-builtins64_*.dll')) {
            $matches = Get-ChildItem -Path $cudaBin -Filter $pattern -File -ErrorAction SilentlyContinue
            foreach ($m in $matches) {
                Copy-Item -Force $m.FullName (Join-Path $stage $m.Name)
            }
        }
    }

    if (Test-Path $zipPath) { Remove-Item $zipPath }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zipPath

    $hash = (Get-FileHash -Algorithm SHA256 -Path $zipPath).Hash.ToLower()
    Set-Content -Path $shaPath -Value $hash -NoNewline -Encoding ASCII
} finally {
    Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "  Archive: $zipPath"
Write-Host "  SHA256:  $hash"
Write-Host ""
