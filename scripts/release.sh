#!/usr/bin/env bash
# scripts/release.sh — build a Senda macOS arm64 release tarball.
#
# Output: dist-release/senda-darwin-aarch64.tar.gz + .sha256
#
#   ./scripts/release.sh
#
# Then upload the tarball as an asset on a GitHub Release on senda-network/senda-llm.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARCH="darwin-aarch64"
ASSET_NAME="senda-${ARCH}.tar.gz"
DIST_DIR="$REPO_ROOT/dist-release"
TARBALL="$DIST_DIR/$ASSET_NAME"
SHA="$DIST_DIR/$ASSET_NAME.sha256"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "release.sh currently builds for darwin-aarch64 only (you are on $(uname -s) $(uname -m))." >&2
    exit 1
fi

echo "==> cargo build --release"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}" \
    cargo build --release -p senda

BIN="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/senda"
if [[ ! -x "$BIN" ]]; then
    echo "release: built binary not found at $BIN" >&2
    exit 1
fi

echo "==> packaging $ASSET_NAME"
mkdir -p "$DIST_DIR"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$BIN" "$STAGE/senda"
chmod +x "$STAGE/senda"
cp "$REPO_ROOT/dist/network.senda.runtime.plist" "$STAGE/network.senda.runtime.plist"
[[ -f "$REPO_ROOT/LICENSE" ]] && cp "$REPO_ROOT/LICENSE" "$STAGE/LICENSE"

tar -C "$STAGE" -czf "$TARBALL" senda network.senda.runtime.plist $([[ -f "$STAGE/LICENSE" ]] && echo LICENSE)

shasum -a 256 "$TARBALL" | awk '{print $1}' > "$SHA"

echo
echo "  Tarball: $TARBALL"
echo "  SHA256:  $(cat "$SHA")"
echo
echo "Upload this tarball to a GitHub Release on senda-network/senda-llm:"
echo "  gh release create v\$VERSION '$TARBALL' --repo senda-network/senda-llm \\"
echo "    --title 'Senda v\$VERSION' --notes 'Senda release.'"
