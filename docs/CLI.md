# CLI User Guide

This is a practical user guide to the `senda` CLI.
It explains what to run for common tasks, then documents each command and switch.

Catalog id definition: a catalog id is the model id shown in `senda models recommended` (for example `Qwen3-0.6B-Q4_K_M`).

## Get help

```bash
senda --help
senda <command> --help
senda models --help
senda models <subcommand> --help
```

## Start here (common tasks)

If you want to:

1. Start serving right away:

```bash
senda serve --auto
```

2. Find a model you can run:

```bash
senda models search gemma --gguf
senda models search smoll --mlx
```

3. Inspect a model before downloading:

```bash
senda models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

4. Download a model:

```bash
senda models download unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

5. Check what is already installed:

```bash
senda models installed
```

## Runtime entrypoints (`serve` / `client`)

If you want to start serving, join a mesh, or run as an API-only client, start here.

Examples:

```bash
senda serve
senda serve --model Qwen3-0.6B-Q4_K_M
senda client --auto
```

Runtime switches:

- `--join <TOKEN>`: join a specific mesh using an invite token (repeatable).
- `--discover [QUERY]`: discover a mesh via Nostr and join.
- `--auto`: auto-join the best discovered mesh.
- `--model <MODEL>`: model to serve (catalog id from `models recommended`, HF ref/URL, or path).
- `--gguf <GGUF>`: serve a specific local GGUF file directly (repeatable).
- `--port <PORT>`: API port (default `9337`).
- `--client`: API-only mode (no GPU/model serving).
- `--console <CONSOLE>`: console/API management port (default `3131`).
- `--headless`: disable the embedded web UI; keep the management API on the `--console` port.
- `--publish`: publish your mesh for discovery.
- `--mesh-name <MESH_NAME>`: friendly mesh name in discovery.
- `--region <REGION>`: region hint for discovery.
- `--blackboard`: enable blackboard on public meshes.
- `--name <NAME>`: your blackboard display name.
- `--max-vram <MAX_VRAM>`: cap VRAM used for planning and fit decisions.
- `--llama-flavor <LLAMA_FLAVOR>`: force backend binary flavor (`cpu|cuda|rocm|vulkan|metal`).
- `--config <CONFIG>`: explicit config file path.
- `--owner-key <OWNER_KEY>`: keystore used to attest this runtime node.
- `--owner-required`: fail startup if owner attestation cannot be loaded.
- `--node-label <NODE_LABEL>`: attach a human label to this runtime node certificate.
- `--trust-policy <TRUST_POLICY>`: override peer ownership trust policy.
- `--trust-owner <TRUST_OWNER>`: add trusted owner IDs on top of the local trust store.

## Commands

### `models`

Start with `models` when you’re working with models: finding them, checking details, downloading them, or checking update state.

Subcommands:

- `recommended`
- `installed`
- `search`
- `show`
- `download`
- `updates`

### `models recommended`

Run this when you want the official built-in model IDs (catalog IDs) and sizes.

Switches:

- `--json`: machine-readable output.

### `models installed`

Run this when you want to see what’s already on your machine.

Switches:

- `--json`: machine-readable output.

### `models search`

Use this to find something you can actually download and run (GGUF or MLX).

Usage:

```bash
senda models search gemma --gguf
senda models search smoll --mlx --limit 5
senda models search qwen --catalog
```

Switches:

- `--gguf`: GGUF-only search (default).
- `--mlx`: MLX-only search.
- `--catalog`: search only built-in catalog.
- `--limit <LIMIT>`: max results (default `20`).
- `--json`: machine-readable output.

### `models show`

Use this when you want to sanity-check one exact model ref before you download or serve it.

Usage:

```bash
senda models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
senda models show mlx-community/SmolLM-135M-8bit
```

Switches:

- `--json`: machine-readable output.

### `models download`

Use this when you’re ready to download one specific resolved model.

Usage:

```bash
senda models download unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
senda models download mlx-community/SmolLM-135M-8bit
```

Switches:

- `--draft`: also download the recommended draft model (if available).
- `--json`: machine-readable output.

### `models updates`

Use this when you want to check for new upstream revisions or refresh cached repo metadata.

Usage:

```bash
senda models updates --check
senda models updates --all
senda models updates unsloth/gemma-4-31B-it-GGUF
```

Switches:

- `--all`: operate on all cached HF repos.
- `--check`: check only; do not refresh cache.
- `--json`: machine-readable output.

### `download`

Use this to quickly download by built-in catalog ID or shorthand.

Usage:

```bash
senda download
senda download 32b
senda download Qwen3-0.6B-Q4_K_M --draft
```

Switches:

- `--draft`: download recommended draft model too.

### `update`

Use this to update senda and exit.

Switches:
- `--auto-update`: available on most commands; when set, senda checks for a newer bundled release before proceeding.


### `gpus`

Use this to inspect local GPU identity and capacity, including per-device VRAM, unified-memory state, and cached benchmark-derived bandwidth when present.


### `load`

Use this to load a model into an already-running local senda runtime.

Usage:

```bash
senda load Qwen3-0.6B-Q4_K_M
```

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `unload`

Use this to unload a model from a running local runtime.

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `status`

Use this to inspect model status from a running local runtime.

Switches:

- `--port <PORT>`: target management/API port (default `3131`).

### `discover`

Use this to discover meshes via Nostr and optionally select one automatically.

Switches:

- `--model <MODEL>`: filter discovered meshes by model name substring.
- `--min-vram <MIN_VRAM>`: filter by minimum VRAM (GB).
- `--region <REGION>`: filter by region.
- `--auto`: print best invite token (useful for piping).
- `--relay <RELAY>`: custom relay URL(s).

### `goose`

Use this to launch Goose already wired to senda’s OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--port <PORT>`: senda API port (default `9337`).

### `claude`

Use this to launch Claude Code already wired to senda’s OpenAI-compatible endpoint.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--port <PORT>`: senda API port (default `9337`).

### `opencode`

Use this to launch OpenCode already wired to senda’s OpenAI-compatible endpoint.

It injects a temporary OpenCode config through `OPENCODE_CONFIG_CONTENT` at launch time, so it does not edit persistent OpenCode config files unless you explicitly pass `--write`.

Switches:

- `--model <MODEL>`: model id from `/v1/models`.
- `--host <HOST|HOST:PORT|URL>`: OpenCode target host or URL (default `127.0.0.1:9337`). Bare host forms assume `http`, default inference port `9337`, and default management port `3131`.
- `--write`: write a merged `~/.config/opencode/opencode.json` that preserves unrelated root keys and sibling providers. If only `opencode.jsonc` exists, senda errors and tells you to rename or migrate it to `opencode.json` first.

### `stop`

Use this to stop local `senda`, `llama-server`, and `rpc-server` processes.


### `blackboard`

Use this to post/search/read shared mesh notes, or to run blackboard as MCP over stdio.

Usage:

```bash
senda blackboard
senda blackboard "STATUS: testing gguf resolution"
senda blackboard --search "gemma"
senda blackboard --mcp
```

Switches:

- `--search <SEARCH>`: search blackboard entries.
- `--from <FROM>`: filter by author.
- `--since <SINCE>`: last N hours.
- `--limit <LIMIT>`: max rows (default `20`).
- `--port <PORT>`: target management/API port (default `3131`).
- `--mcp`: run as MCP server over stdio.

### `plugin`

Use this to inspect plugin status or run plugin compatibility shims.

Subcommands:

- `plugin list`: list auto-registered/configured plugins.
- `plugin install <NAME>`: old install workflow shim.


### `auth`

Use this to manage owner identity and keystore files.

Subcommands:

- `auth init`: generate/save owner keypair.
- `auth status`: show identity/keystore status.

`auth init` switches:

- `--owner-key <OWNER_KEY>`: keystore path.
- `--force`: overwrite existing keystore.
- `--no-passphrase`: leave keys unencrypted.
- `--keychain`: store random unlock passphrase in OS keychain.

`auth status` switches:

- `--owner-key <OWNER_KEY>`: keystore path.

`auth sign-node` / `auth renew-node` / `auth rotate-node` switches:

- `--owner-key <OWNER_KEY>`: keystore path.
- `--node-label <NODE_LABEL>`: attach a human label to the signed node certificate.

`auth rotate-owner` switches:

- `--owner-key <OWNER_KEY>`: keystore path.

### `moe`

Use this to inspect MoE rankings, generate new rankings locally, plan expert placement for a target memory budget, or submit local rankings to the canonical dataset.

Subcommands:

- `moe plan <MODEL>`: resolve a ranking and compute a placement recommendation.
- `moe analyze full <MODEL>`: run a full local MoE analysis and cache the result.
- `moe analyze micro <MODEL>`: run the canonical micro analysis and cache the result.
- `moe share <MODEL>`: validate a local ranking artifact and open a contribution PR to the canonical dataset.

### `moe plan`

Use this when you want to know whether a MoE model is likely to fit on your hardware, how many nodes it needs, and which ranking artifact was used to make that decision.

Usage:

```bash
senda moe plan unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_S
senda moe plan unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_S --max-vram 16
senda moe plan unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_S --max-vram 16 --json
```

Behavior:

- Checks local senda ranking cache first.
- Checks `meshllm/moe-rankings` on Hugging Face and uses it when it provides a stronger ranking than local cache. If local and published rankings have the same analyzer strength, the local cache wins.
- Keeps Hugging Face dataset files in the normal Hugging Face cache; it does not copy dataset artifacts into `~/.cache/senda`.
- Prefers `full-*` rankings over `micro-*` for the same model/distribution.
- Prints ranking provenance clearly so you can see whether the planner used local cache, Hugging Face, or an explicit override.

Switches:

- `--max-vram <MAX_VRAM>`: target per-node VRAM budget in GB.
- `--nodes <NODES>`: optional explicit node count to evaluate.
- `--ranking-file <RANKING_FILE>`: bypass normal ranking resolution and use one specific ranking CSV.
- `--json`: machine-readable output.

### `moe analyze full`

Use this when you want to produce a full local MoE ranking for a model and cache it for planning or later submission.

Usage:

```bash
senda moe analyze full unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_S
```

Behavior:

- Runs `llama-moe-analyze` locally using the full analyzer contract.
- Shows progress for long-running work.
- Writes durable logs so failures can be inspected after the command exits.
- Caches the generated ranking locally for later `moe plan`, `serve`, or `moe share`.

### `moe analyze micro`

Use this when you want a faster local fallback ranking using the canonical micro prompt set.

Usage:

```bash
senda moe analyze micro unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ2_M
```

Behavior:

- Runs the canonical `micro-v1` analysis locally.
- Uses the fixed micro prompt set so results are comparable across machines.
- Caches the generated ranking locally for later planning or submission.
- Writes a durable log path on success or failure.

### `moe share`

Use this to open a contribution PR for a local ranking artifact on the canonical dataset repo on Hugging Face.

Usage:

```bash
senda moe share unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_S
senda moe share unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_S --ranking-file ~/.cache/senda/moe-rankings/local-gemma-4-26b-a4b-it-ud-q4_k_s.micro-p8-t128-all.csv
```

Behavior:

- Validates the local artifact and computes its canonical dataset path.
- Checks `meshllm/moe-rankings` first and exits cleanly when the artifact is already published.
- Opens a dataset PR instead of writing directly to `main`.
- Includes `ranking.csv`, `metadata.json`, and `run.log` when available.
- Uses the same Hugging Face commit API pattern as the Python publisher, with PR creation enabled.

Requirements:

- Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` with write access to the destination dataset repo.

Switches:

- `--ranking-file <RANKING_FILE>`: share one specific local ranking file instead of resolving the default cached artifact.
- `--dataset-repo <DATASET_REPO>`: override the target dataset repo (default `meshllm/moe-rankings`).

## Model reference formats

Supported for `models show`, `models download`, and `serve --model`:

1. Catalog id (an id from `senda models recommended`):

```bash
senda models show Qwen3-0.6B-Q4_K_M
```

2. HF repo or GGUF selector:

```bash
senda models show unsloth/gemma-4-31B-it-GGUF
senda models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL
```

3. HF URL:

```bash
senda models show https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
```

4. Revision pin:

```bash
senda models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL@main
senda models show unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL@<commit-sha>
senda models show mlx-community/SmolLM-135M-8bit@<commit-sha>
senda models show https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/main
```

For MLX, use repo shorthand (not `/model`):

```bash
senda models show mlx-community/SmolLM-135M-8bit
senda models download mlx-community/SmolLM-135M-8bit
```

## Model resolution behavior

Resolution order:

1. exact catalog id
2. exact HF ref
3. HF URL
4. bare-name discovery

GGUF behavior:

1. GGUF search uses Hub `gguf` pre-filter.
2. Excludes sidecars like `mmproj*.gguf`.
3. Split GGUF uses first shard (`-00001-of-...`) for selection/display.
4. `repo` with no selector uses fit-aware ranking against local VRAM.
5. `repo:SELECTOR` resolves exact quant/variant.

MLX behavior:

1. MLX search uses Hub `mlx` pre-filter.
2. Model must include weight files (`model.safetensors` or split first shard).
3. `model.safetensors.index.json` by itself is not treated as a model artifact.
4. Display reference stays repo shorthand.

## Machine-readable output (`--json`)

All `models` subcommands support `--json`. `senda moe plan` also supports `--json`.

Examples:

```bash
senda models search smoll --mlx --limit 1 --json | jq .
senda models show mlx-community/SmolLM-135M-8bit --json | jq .
senda models download Qwen3-0.6B-Q4_K_M --json | jq .
senda models installed --json | jq .
senda models recommended --json | jq .
senda models updates --check --json | jq .
senda moe plan unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_S --max-vram 16 --json | jq .
```

Shape summary:

- `search --json`: `{ filter, query, machine, results[] }`
- `show --json`: resolved model + `variants[]`
- `download --json`: requested/resolved refs + local `path`
- `installed --json`: `{ cache_dir, results[] }`
- `recommended --json`: `{ source, results[] }`
- `updates --json`: check/update results
- `moe plan --json`: plan result + ranking provenance + sizing summary

Automation tips:

1. Prefer explicit refs in scripts.
2. Pin `@<commit-sha>` when reproducibility matters.
3. Parse stable keys such as `type`, `ref`, `fit`, `path`, and `results`.
