#!/usr/bin/env bash
# release-senda.sh — package the senda binary for the senda.network installer.
#
# Produces dist-release/senda-<platform-suffix>.tar.gz, where <platform-suffix>
# matches what senda/public/install.sh asks for. Runs alongside the existing
# scripts/package-release.sh (which produces the upstream senda bundle).
#
# What goes in the tarball depends on the flavor:
#
#   - metal (macOS arm64) / cpu (Linux): the full runtime — senda,
#     rpc-server-<flavor>, llama-server-<flavor>, llama-moe-{analyze,split},
#     and any runtime shared libraries. These flavors are small enough (~40MB
#     on macOS, ~45MB on Linux CPU) to ship as a single-tarball installer
#     payload that works end-to-end after `curl … | sh`.
#
#   - cuda / rocm / vulkan: only the main senda binary. These flavors'
#     llama.cpp runtime bundles balloon from tens of MB to hundreds of MB
#     (ROCm) or low gigabytes (CUDA) because they drag in cuBLAS / HIP
#     shared libraries. We keep the installer tarball slim; install.sh
#     does a second download of the matching `senda-v<ver>-<target>.tar.gz`
#     (produced by package-release.sh) for the llama.cpp runtime.
#
# Every flavor always includes:
#   <archive>/senda
#   <archive>/network.senda.runtime.plist  (macOS only, reference)
#   <archive>/senda.service               (Linux only, systemd reference)
#   <archive>/LICENSE
#
# Usage:
#   scripts/release-senda.sh [--flavor cpu|cuda|rocm|vulkan|metal] [output_dir]
#
# Backend flavor defaults to:
#   - macOS arm64: metal
#   - Linux x86_64 / aarch64: cpu
#   - Override via --flavor or MESH_RELEASE_FLAVOR (matches package-release.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_BIN_DIR="$REPO_ROOT/target/release"
BUILD_BIN_DIR="${SENDA_LLAMA_BUILD_BIN_DIR:-$REPO_ROOT/.deps/llama.cpp/build/bin}"
DIST_DIR_DEFAULT="$REPO_ROOT/dist-release"

FLAVOR="${MESH_RELEASE_FLAVOR:-}"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --flavor)
            FLAVOR="${2:-}"
            shift 2
            ;;
        --flavor=*)
            FLAVOR="${1#--flavor=}"
            shift
            ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            if [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
                shift
            else
                echo "release-senda: unexpected argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

OUTPUT_DIR="${OUTPUT_DIR:-$DIST_DIR_DEFAULT}"

os="$(uname -s)"
arch="$(uname -m)"

case "$arch" in
    x86_64|amd64) arch_canon="x86_64" ;;
    arm64|aarch64) arch_canon="aarch64" ;;
    *) arch_canon="$arch" ;;
esac

# Default flavor by platform.
if [[ -z "$FLAVOR" ]]; then
    case "$os/$arch_canon" in
        Darwin/aarch64) FLAVOR="metal" ;;
        Linux/x86_64|Linux/aarch64) FLAVOR="cpu" ;;
        *)
            echo "release-senda: unsupported platform $os/$arch_canon" >&2
            exit 1
            ;;
    esac
fi

# Compose platform-suffix used in the archive name. This must stay in lockstep
# with the detect_target() function in senda/public/install.sh.
case "$os/$arch_canon/$FLAVOR" in
    Darwin/aarch64/metal) platform_suffix="darwin-aarch64" ;;
    Linux/x86_64/cpu)     platform_suffix="linux-x86_64-cpu" ;;
    Linux/x86_64/cuda)    platform_suffix="linux-x86_64-cuda" ;;
    Linux/x86_64/rocm)    platform_suffix="linux-x86_64-rocm" ;;
    Linux/x86_64/vulkan)  platform_suffix="linux-x86_64-vulkan" ;;
    Linux/aarch64/cpu)    platform_suffix="linux-aarch64-cpu" ;;
    Linux/aarch64/vulkan) platform_suffix="linux-aarch64-vulkan" ;;
    *)
        echo "release-senda: unsupported os/arch/flavor combo: $os/$arch_canon/$FLAVOR" >&2
        exit 1
        ;;
esac

asset="senda-${platform_suffix}.tar.gz"

bin="$RELEASE_BIN_DIR/senda"
if [[ ! -x "$bin" ]]; then
    echo "release-senda: built binary not found at $bin" >&2
    echo "                    run 'just release-build' (or the platform-specific recipe) first." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
# Resolve to absolute path so the tarball path stays valid after cd-into-staging.
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
tarball="$OUTPUT_DIR/$asset"
sha_file="$tarball.sha256"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

cp "$bin" "$stage/senda"
chmod +x "$stage/senda"

# Reference service definitions: ship per-platform so the tarball is
# self-documenting, but install.sh / install.ps1 generate the live unit
# at install time with the correct paths.
case "$os" in
    Darwin)
        if [[ -f "$REPO_ROOT/dist/network.senda.runtime.plist" ]]; then
            cp "$REPO_ROOT/dist/network.senda.runtime.plist" "$stage/network.senda.runtime.plist"
        fi
        ;;
    Linux)
        if [[ -f "$REPO_ROOT/dist/senda.service" ]]; then
            cp "$REPO_ROOT/dist/senda.service" "$stage/senda.service"
        fi
        ;;
esac

[[ -f "$REPO_ROOT/LICENSE" ]] && cp "$REPO_ROOT/LICENSE" "$stage/LICENSE"

# For metal/cpu flavors the llama.cpp runtime is small enough (tens of MB) to
# ship alongside the main binary, so the installer tarball works end-to-end
# after a single download. GPU flavors stay slim here — install.sh pulls the
# full bundle separately to avoid a 1.7GB curl | sh on CUDA.
case "$FLAVOR" in
    metal|cpu)
        include_runtime=1
        ;;
    *)
        include_runtime=0
        ;;
esac

if (( include_runtime )); then
    missing_bins=()
    for bin in "rpc-server" "llama-server" "llama-moe-analyze" "llama-moe-split"; do
        src="$BUILD_BIN_DIR/$bin"
        [[ ! -x "$src" ]] && missing_bins+=("$bin") && continue
        # Flavor-suffix the two that senda's launch.rs expects flavored.
        case "$bin" in
            rpc-server|llama-server) dest="$stage/${bin}-${FLAVOR}" ;;
            *)                       dest="$stage/${bin}" ;;
        esac
        cp "$src" "$dest"
        chmod +x "$dest"
    done

    if (( ${#missing_bins[@]} > 0 )); then
        echo "release-senda: missing llama.cpp binaries for flavor '$FLAVOR':" >&2
        printf '  - %s\n' "${missing_bins[@]}" >&2
        echo "                    expected under $BUILD_BIN_DIR" >&2
        echo "                    run 'just release-build' first." >&2
        exit 1
    fi

    # Pull in any runtime shared libs llama-server links against (dylibs on
    # Darwin, .so on Linux). For a static Metal build this is usually empty,
    # which is fine — nothing to copy.
    shopt -s nullglob
    runtime_libs=()
    case "$os" in
        Darwin) runtime_libs=("$BUILD_BIN_DIR"/*.dylib) ;;
        Linux)  runtime_libs=("$BUILD_BIN_DIR"/*.so "$BUILD_BIN_DIR"/*.so.*) ;;
    esac
    shopt -u nullglob
    if (( ${#runtime_libs[@]} > 0 )); then
        for lib in "${runtime_libs[@]}"; do
            cp "$lib" "$stage/"
        done
    fi

    # macOS needs the binaries to look next to themselves for their dylibs;
    # mirror what package-release.sh does for the full bundle.
    if [[ "$os" == "Darwin" ]]; then
        for bin in \
            "$stage/senda" \
            "$stage/rpc-server-${FLAVOR}" \
            "$stage/llama-server-${FLAVOR}" \
            "$stage/llama-moe-analyze" \
            "$stage/llama-moe-split"; do
            [[ -f "$bin" ]] || continue
            install_name_tool -add_rpath @executable_path/ "$bin" 2>/dev/null || true
        done
    fi
fi

(cd "$stage" && tar -czf "$tarball" ./*)

if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$tarball" | awk '{print $1}' > "$sha_file"
elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$tarball" | awk '{print $1}' > "$sha_file"
fi

echo
echo "  Tarball: $tarball"
[[ -f "$sha_file" ]] && echo "  SHA256:  $(cat "$sha_file")"
echo
