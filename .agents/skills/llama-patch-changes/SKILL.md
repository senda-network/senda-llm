---
name: llama-patch-changes
description: Use when changing senda's llama.cpp patch queue, upstream pin, prepare/build scripts, or carried RPC, MoE, and mesh-hook llama.cpp patches.
---

# llama-patch-changes

Use this skill when editing the llama.cpp patch queue, refreshing patches from a
llama.cpp checkout, updating the pinned upstream SHA, or changing build scripts
that prepare or consume patched llama.cpp.

## Boundaries

- Keep durable llama-side changes in `third_party/llama.cpp/patches/*.patch`.
- Keep the upstream pin in `third_party/llama.cpp/upstream.txt`.
- Keep `LLAMA_CPP_SHA` as a compatibility mirror of `upstream.txt` while this
  repository still has legacy readers.
- Do not add a submodule, vendor a llama checkout, or depend on the old
  Mesh-LLM llama.cpp fork.
- Do not treat edits in `.deps/llama.cpp` as durable until the patch queue has
  been regenerated and committed.
- Do not add llama-stage ABI/static in-process patches unless the task
  explicitly asks for that integration pass.
- Prefer small, reviewable llama commits with one capability per patch.

## Local Flow

Prepare the pinned upstream checkout and current patch queue:

```bash
scripts/prepare-llama.sh pinned
```

For actual llama-side editing, prefer a normal llama.cpp checkout or branch
where commits can be named and inspected. Base the branch on upstream
`ggml-org/llama.cpp` `master`, then carry the Mesh-LLM patch commits on top.

After editing and committing in that llama checkout, regenerate the patch queue
from its upstream merge base:

```bash
rm -rf /path/to/senda/third_party/llama.cpp/patches
mkdir -p /path/to/senda/third_party/llama.cpp/patches
git format-patch \
  --output-directory /path/to/senda/third_party/llama.cpp/patches \
  "$(git merge-base HEAD upstream/master)..HEAD"
```

If the llama checkout uses `origin` for upstream instead of `upstream`, replace
`upstream/master` with `origin/master`.

## Validation

Validate that patches apply in a clean checkout:

```bash
tmp_llama="$(mktemp -d /tmp/senda-llama.XXXXXX)"
rm -rf "$tmp_llama"
LLAMA_WORKDIR="$tmp_llama" scripts/prepare-llama.sh pinned
```

For normal senda validation, use the repository build workflow:

```bash
just build
```

For Rust-only fallout from build-system or runtime call-site changes:

```bash
cargo fmt --all -- --check
cargo check -p senda
```

Run Cargo commands serially. This repo frequently hits Cargo lock conflicts
when multiple Cargo commands run at once.

## Updating The Upstream Pin

Test the queue against current upstream without moving the pin:

```bash
scripts/prepare-llama.sh latest
just build
cargo test -p senda --lib
```

If the queue applies and validation passes, update both pin files:

```bash
cp third_party/llama.cpp/upstream.txt /tmp/old-llama-upstream.txt
git -C .deps/llama.cpp rev-parse "$(cat .deps/llama.cpp/.git/senda-upstream-sha)" > third_party/llama.cpp/upstream.txt
cp third_party/llama.cpp/upstream.txt LLAMA_CPP_SHA
```

Commit the pin update with any patch refreshes.

## Gotchas

- `scripts/prepare-llama.sh` configures local git identity for `git am`; keep
  that responsibility there for fresh CI checkouts.
- Patch files are mail-format artifacts and may intentionally contain
  whitespace that `git diff --check` reports. Do not hand-normalize patches in
  a way that changes or breaks `git am`.
- Build outputs live under `.deps/llama.cpp/build`; the root `llama.cpp`
  symlink is compatibility-only.
- Important backend flags include `GGML_RPC=ON`, `BUILD_SHARED_LIBS=OFF`, and
  `LLAMA_OPENSSL=OFF`; preserve CPU, Metal, CUDA, Vulkan, and ROCm behavior
  when touching build scripts.
- See `senda/docs/LLAMA_CPP_FORK.md` for the full patch-queue maintenance
  notes and `senda/docs/LLAMA_STAGE_INTEGRATION_PLAN.md` for deferred
  llama-stage integration.
