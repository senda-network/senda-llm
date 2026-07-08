<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/senda-logo-dark.svg">
    <img src="docs/senda-logo-light.svg" alt="Senda" width="380">
  </picture>
</p>

<h1 align="center">Senda LLM</h1>

<p align="center">
  <strong>The peer-to-peer inference runtime that powers Senda.</strong><br/>
  Run real open-weight models end-to-end on the hardware you already own — Apple Silicon Macs, NVIDIA / AMD / Intel boxes — and expose the result as one OpenAI-compatible API at <code>localhost:9337/v1</code>.
</p>

<p align="center">
  <a href="https://senda.network">senda.network</a> · 
  <a href="docs/USAGE.md">Usage</a> · 
  <a href="CONTRIBUTING.md">Contributing</a> · 
  <a href="ROADMAP.md">Roadmap</a>
</p>

---

Senda LLM is the open-source inference engine behind [Senda](https://senda.network) — a private LLM product for teams and individuals that don't want to send their conversations to a third-party LLM API. It's built for the work open-weight models already do well — summarizing documents and codebases, classifying and labeling data at scale, long-running background agents, synthetic-data generation — where keeping data in-house and keeping per-token costs flat matter more than shaving a second off every reply. The chat product runs in the browser. The runtime in this repo runs on the machines you already own and presents them as one OpenAI-compatible API.

The design principle: **the unit of work is a session, not a token.** On residential and laptop-class networks, per-token cross-machine traffic is fatal — RTTs are 20–200 ms, bandwidth is variable, and any architecture that puts the network on the per-token critical path collapses to <1 t/s. So we don't:

- **Replication is the default.** If a model fits on one machine, it runs there end-to-end at full quality. The mesh's job is to find the best peer for each session, not to stitch weights across slow links.
- **Speculative decoding is a single-node opt-in, not a multi-peer route.** A small draft model proposes 4–8 tokens that the verifier accepts in one batched forward pass — a real win when both models run on the same box. The cross-peer variant was benchmarked and shelved (2026-06): even the single-GPU best case ceilinged at ~1.3–1.4×, and a WAN hop per draft→verify cycle erases that.
- **Models can collaborate during inference.** A text-only model consults a vision peer for image captions; an uncertain model gets a second opinion from a different architecture; small models nudge a stuck verifier out of repetition loops. The caller sees one seamless response.
- **Every node exposes the same local API** at `http://localhost:9337/v1`.

For the rare model that doesn't fit on any single peer (Llama 3.1 405B, DeepSeek-V3, very large MoEs), the runtime can fall back to pipeline parallelism or MoE expert sharding across multiple peers. Those modes are documented in the [Power-user / no-single-peer-fits fallbacks](#power-user--no-single-peer-fits-fallbacks) section below — they exist for completeness, not as the headline experience. Real-world decode rates on residential links cap out below 1 t/s for those modes; we don't recommend running them as a daily driver.

### Apple Silicon is the hero hardware

M-series unified memory turns a $2.5–4.5k laptop into a 30B–70B-capable inference box at speeds Windows GPU setups at the same price can't match. An M3 Pro 18 GB serves Qwen 3 8B comfortably; an M4 Pro / Max 36–48 GB serves Qwen 3 14B and 30B-A3B MoE; an M4 Max / Ultra 64–128 GB serves Llama 3.3 70B and 100B+ MoEs end-to-end. CUDA / ROCm / Vulkan boxes happily join too, and excel in different niches (an RTX 4090 24 GB is the best 32–70B GPU-only host, gaming PCs run smaller models faster, etc.).

## How this repo fits into the Senda product

| Piece | Repo | What it is |
|---|---|---|
| Chat product surface | [`senda-network/senda`](https://github.com/senda-network/senda) | The Next.js chat UI hosted at `senda.network` and the desktop app / local controller. |
| Inference runtime | **this repo** (`senda-network/senda-llm`) | The `senda` binary. OpenAI-compatible API, peer-to-peer mesh, pipeline + MoE parallelism, capability-aware routing. |

The two are versioned and released independently. Most teams only ever install the runtime — the chat product talks to it for them.

> Senda LLM is a fork of [Mesh-LLM/senda](https://github.com/senda-network/senda-llm). The Rust crate names are kept upstream-compatible (`senda`, `mesh-client`, etc.) so we can rebase cleanly; the **shipped binary**, **service label**, and **data directory** are all `senda`-branded.

## Quick start

Install the runtime on macOS, Linux, or WSL2:

```bash
curl -fsSL https://senda.network/install | sh
```

```powershell
# Windows (PowerShell, no admin needed):
iwr -useb https://senda.network/install.ps1 | iex
```

Auto-start at login (writes a launchd plist on macOS, a systemd `--user` unit on Linux, or a Scheduled Task on Windows):

```bash
curl -fsSL https://senda.network/install | sh -s -- --service
```

Once installed:

```bash
senda serve --auto                       # join the best public mesh, start serving
senda service start                      # or run as a background service
senda service status                     # status, pid, last logs
```

You now have:

- An OpenAI-compatible API at `http://localhost:9337/v1`.
- A live admin/management surface at `http://localhost:3131`.
- A local data dir at `~/.senda/`.

Send a request:

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

## Common workflows

### Private mesh (single-tenant / company fleet)

```bash
senda serve --private-only --model Qwen2.5-32B
```

`--private-only` is a hard lock for deployments that must never accidentally publish to or auto-join a public mesh — internal company fleets, on-prem clusters, regulated environments. It conflicts at parse time with `--auto`, `--publish`, `--discover`, `--mesh-name`, and `--region`, so a misconfigured invocation fails fast instead of silently joining the wrong mesh. Joining via an explicit invite token (`--join <token>`) still works; only Nostr-based discovery and publishing are disabled.

### Add a second machine

The first time `senda serve` runs, it prints an invite token. Anyone with reasonable hardware (a decent GPU, an Apple Silicon Mac, or even a beefy CPU-only Linux box) can join. The router learns each node's capability and only dispatches work that node can actually serve.

```bash
senda serve --join <invite-token>
```

A 70B-class request only routes to nodes that advertise enough VRAM; an 8B model happily hops between an M-series Mac, an RTX 4090 box, and a Vulkan laptop in the same conversation.

### Multi-model

```bash
senda serve --model Qwen2.5-32B --model GLM-4.7-Flash
```

Different nodes serve different models simultaneously. The API proxy peeks at the `model` field in each request and routes to the right node:

```bash
curl localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

`/v1/models` lists everything available across the mesh.

### Inspect local hardware

```bash
senda gpus                                # local GPUs, stable IDs, VRAM
senda gpus --json                         # machine-readable inventory
senda gpu benchmark --json                # refresh the local fingerprint
```

Use only pinnable `Stable ID` / `stable_id` values from `senda gpus` for pinned startup config. Stable-ID fallback values such as `index:*` or backend-device names like `CUDA0` / `HIP0` / `MTL0` can still be printed for inventory purposes, but they are not valid pin targets.

## How it works

Senda LLM keeps the user-facing surface simple: talk to `localhost:9337`, pick a model, and let the mesh decide how to serve it. The decision tree, in order:

1. **One peer fits the model?** Run it solo on that peer, end-to-end, full speed, zero per-token network overhead. This is the default and the case the runtime is optimised for.
2. **Speculative decoding on that peer?** If the serving peer also holds a compatible small draft model, it can propose 4–8 tokens that the verifier accepts in one batched forward pass — an opt-in, same-box throughput win. (Cross-peer draft/verify pairing was benchmarked and shelved in 2026-06: best-case speedup ~1.3–1.4× even with zero network hop, below the bar at which the added WAN latency pays for itself.)
3. **Multi-model collaboration?** A text-only model silently consults a vision peer for image captions; an uncertain model gets a second opinion from a different architecture; a stuck verifier gets nudged out of a repetition loop. The caller sees one seamless response — they don't know multiple models collaborated. See [senda/docs/VIRTUAL_LLM.md](senda/docs/VIRTUAL_LLM.md).
4. **Multi-model serving** — different peers serve different models simultaneously. The API proxy peeks at the `model` field in each request and routes to the right peer via QUIC tunnel. `/v1/models` lists everything advertised by the mesh.
5. **Demand-aware rebalancing** — a unified demand map tracks which models the mesh wants (from `--model` flags, API requests, and gossip). Demand signals propagate across peers and decay via TTL. Standby peers auto-promote to serve unserved models with active demand, or rebalance when one model is significantly hotter than others.
6. **Capability matching** — requests only go to peers whose backend (Metal / CUDA / ROCm / Vulkan / CPU) and VRAM can actually serve the model. Offline peers are routed around automatically.

### Latency design

HTTP streaming is latency-tolerant; per-token RPC is latency-multiplied. So `llama-server` always runs on the same box as the GPU, and the mesh tunnels the HTTP request to that box — cross-network latency only affects time-to-first-token, not per-token throughput. **Per-token traffic stays inside one machine** for the default replication path, which is what makes residential and laptop-class hardware viable as inference peers in the first place. Speculative decoding is the deliberate exception: it sends 4–8 candidate tokens per network hop, so the per-hop overhead amortises over the batch.

### Throughput optimisations

- **Zero-transfer GGUF loading** — `SET_TENSOR_GGUF` tells rpc-server to read weights from local disk. Dropped model load from 111 s → 5 s.
- **Speculative decoding (same box)** — local draft model proposes tokens, the verifier accepts them in one batched forward pass. +38 % throughput on code (75 % acceptance) when draft and verifier share a machine. The cross-peer variant is shelved: measured best case ~1.3–1.4× with no network hop, which a residential RTT per draft→verify cycle erases.
- **RPC round-trip reduction** — cached `get_alloc_size`, skip GGUF lookups for intermediates. Per-token round-trips: 558 → 8 (used by the fallback splits below).
- **Direct server-to-server transfers** — intermediate tensors pushed directly between rpc-servers via TCP, not relayed through the client (also used by the fallback splits).

### Power-user / no-single-peer-fits fallbacks

For models that genuinely don't fit on any single peer (Llama 3.1 405B, DeepSeek-V3, very large dense models), the runtime can still split the model across multiple peers. **These modes are not recommended as a daily driver on residential WAN** — RTTs in the 20–200 ms range push per-token decode rates well below 1 t/s — but they exist for completeness, controlled LAN setups, and "I want to see my 405B run somehow" moments.

- **Pipeline parallelism** (dense models). Layers are distributed across peers proportional to per-peer VRAM. `llama-server` runs on the highest-VRAM peer and coordinates via RPC; each `rpc-server` loads only its assigned layers from local disk. Latency-aware peer selection: lowest-RTT peers first, 80 ms hard cap — higher-latency peers stay in the mesh as API clients but don't participate in splits.
- **MoE expert sharding** (Mixture-of-Experts models — Qwen3-MoE, GLM, OLMoE, Mixtral, DeepSeek). Auto-detected from the GGUF header. The mesh reads expert routing statistics to identify which experts matter most, then assigns each peer an overlapping shard: a shared core of critical experts replicated everywhere, plus unique experts distributed across peers. Each peer gets a standalone GGUF with the full trunk + its expert subset and runs its own independent `llama-server` — zero cross-peer traffic during inference. Sessions are hash-routed to peers for KV cache locality. With `overlap=1` (the default for two peers), each peer holds a complete-enough expert set to serve sessions end-to-end, so this mode is in practice a special case of replication and remains viable.

**Demos and benchmarks.** By default the runtime prefers solo replication whenever a peer can host a model end-to-end. To force the legacy behavior for a specific model, set `force_split = true` in `~/.senda/config.toml` (the desktop "Run on the mesh" toggle writes this). Entry-node routing uses `SENDA_FORCE_SPLIT_ROUTING=1` to disable solo-first target ordering when you need A/B catalog comparisons between solo and pooled-split rows. `SENDA_FORCE_DUPLICATE_HOSTS=1` still forces duplicate solo hosts for routing benchmarks.

## Service mode

Senda runs as a per-user background service on all three platforms:

```bash
senda service start            # start (auto-installs the unit on first run)
senda service stop             # stop
senda service status           # state, pid, recent logs
senda service logs             # tail logs
```

Under the hood:

- macOS installs a launchd agent at `~/Library/LaunchAgents/network.senda.runtime.plist`.
- Linux installs a systemd `--user` unit at `~/.config/systemd/user/senda.service`.
- Windows registers a Scheduled Task at user login.
- Shared environment config lives in `~/.config/senda/service.env`.
- Startup models live in `~/.senda/config.toml`.

On Linux this is a user service, so if you want it to keep running after reboot before login, enable lingering once:

```bash
sudo loginctl enable-linger "$USER"
```

## Output modes

`senda` has two terminal output modes:

- `--log-format pretty` — human-readable. In `serve` on an interactive TTY, this becomes the full dashboard; otherwise it falls back to line-oriented pretty output.
- `--log-format json` — newline-delimited JSON records to `stdout`. Safe for `jq`, log shippers, and shell pipelines.

```json
{"timestamp":"...","level":"info","event":"llama_ready","model":"Qwen3-32B","port":8001,"ctx_size":8192,"message":"Qwen3-32B ready on internal port 8001"}
{"timestamp":"...","level":"info","event":"ready","api_url":"http://localhost:9337","console_url":"http://localhost:3131","api_port":9337,"console_port":3131,"models_count":2,"message":"senda runtime ready"}
```

For the full event taxonomy and field reference, see [senda/src/cli/output/EVENTS.md](senda/src/cli/output/EVENTS.md).

## Verify which backend loaded

```bash
curl -s http://localhost:3131/api/status | jq '.capability'
# {
#   "backend": "metal",            # metal | cuda | rocm | vulkan | cpu
#   "vendor": "apple",
#   "vram_total_mb": 16384,
#   "compute_class": "mid",
#   ...
# }
```

`capability.backend` is the source of truth — it's what the rest of the mesh sees when matching your node to inference requests, and what the Senda chat product surfaces in its status pill and `/control` Nodes tab.

## Multimodal support

Senda LLM supports multimodal requests on:

- `POST /v1/chat/completions`
- `POST /v1/responses`

| Family / model type | Vision | Audio |
|---|---|---|
| `Qwen3-VL`, `Qwen3VL` | yes | no |
| `Qwen2-VL`, `Qwen2.5-VL` | yes | no |
| `LLaVA`, `mllama`, `PaliGemma`, `Idefics`, `Molmo`, `InternVL`, `GLM-4V`, `Ovis`, `Florence` | yes | no |
| `Qwen2-Audio`, `SeaLLM-Audio`, `Ultravox`, `Whisper` | no | yes |
| `Omni` | metadata-dependent | yes |
| Any GGUF with `mmproj` sidecar | yes | depends |
| Any model with `vision_config` / vision token IDs | yes | depends |
| Any model with `audio_config` / audio token IDs | depends | yes |

For the full capability and transport details, see [senda/docs/MULTI_MODAL.md](senda/docs/MULTI_MODAL.md).

## Using with agents

Senda LLM exposes an OpenAI-compatible API on `localhost:9337`. Any tool that supports custom OpenAI endpoints works. `/v1/models` lists available models; the `model` field in requests routes to the right node.

Built-in launcher integrations:

```bash
senda goose                     # launch goose against the local mesh
senda opencode                  # launch opencode against the local mesh
```

- Goose and Claude reuse a local mesh on `--port` and auto-start a local client if needed.
- OpenCode targets `--host` (default `127.0.0.1:9337`) and only auto-starts a local client for loopback/localhost targets.
- If `--model` is omitted, the launcher picks the strongest tool-capable model available on the mesh.

### External OpenAI-compatible backends (vLLM, TGI, Ollama, Lemonade, etc.)

The `openai-endpoint` plugin routes inference to any server that speaks the OpenAI `/v1/chat/completions` API. The server does all the inference work — Senda just discovers its models and routes requests to it.

Enable the plugin in `~/.senda/config.toml`:

```toml
# vLLM
[[plugin]]
name = "openai-endpoint"
url = "http://gpu-box:8000/v1"

# Ollama
[[plugin]]
name = "openai-endpoint"
url = "http://localhost:11434/v1"
```

The plugin health-checks the backend by probing `GET /v1/models` — models appear and disappear automatically as the backend starts and stops.

## Web console

Each running node exposes a live admin/management UI:

```text
http://localhost:3131
```

The console shows live topology with `Client`, `Standby`, `Loading`, and `Serving` badges, plus VRAM usage, loaded models, and built-in chat. It's backed by `/api/status` and `/api/events` (SSE).

To run without the embedded UI (for example, in a headless server environment), pass `--headless`. The management API (`/api/*`) stays fully available on the `--console` port.

## Hardware support

The installer detects OS, CPU architecture and GPU vendor, then pulls the matching runtime build:

| OS | Hardware | Backend |
|---|---|---|
| macOS | Apple Silicon | Metal |
| Linux | x86_64 · NVIDIA | CUDA |
| Linux | x86_64 · AMD (`rocminfo`) | ROCm |
| Linux | x86_64 · Intel / Vulkan-only | Vulkan |
| Linux | x86_64 · CPU-only | CPU |
| Linux | aarch64 | Vulkan / CPU |
| WSL2 | x86_64 · NVIDIA passthrough | CUDA |
| Windows 10/11 | x86_64 · NVIDIA | CUDA |
| Windows 10/11 | x86_64 · AMD / Intel / other | Vulkan |

Override auto-detection with `SENDA_BACKEND=cuda|rocm|vulkan|cpu` when running the installer (handy for dual-GPU boxes or unusual setups).

## Build from source

Building from source only makes sense if you're hacking on the runtime itself or running on a platform we don't ship binaries for yet.

```bash
git clone https://github.com/senda-network/senda-llm
cd senda-llm
just build           # patched llama.cpp + senda binary + UI (~30 min first time)
./target/release/senda models download Qwen3-0.6B-Q4_K_M  # fast, 397MB
./target/release/senda models download Qwen3-8B-Q4_K_M    # demo model, 5GB
```

Requirements: `just`, `cmake`, Rust toolchain, Node.js 24 + npm. NVIDIA builds need `nvcc` (CUDA toolkit). AMD builds need ROCm/HIP. Vulkan builds need the Vulkan development files plus `glslc`. CPU-only and Jetson/Tegra also work.

On Linux, `just build` auto-detects CUDA vs ROCm vs Vulkan, or you can force `backend=rocm` / `backend=vulkan`. Windows source builds are supported for `cuda`, `rocm`/`hip`, `vulkan`, and `cpu` via `just build`.

For full build-from-source instructions and UI development, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Repo layout

```
senda/                Rust crate — runtime, mesh, CLI (binary name: senda)
mesh-client/             OpenAI-compatible client crate
mesh-host-core/          Host-side primitives shared between binaries
mesh-api/, mesh-api-ffi/ Management API surface and FFI shim
sdk/                     Language SDKs
tools/                   Internal tooling
evals/                   Benchmarking and evaluation scripts
docs/                    Design, usage, CLI reference
dist/                    OS service unit templates (launchd, systemd, scheduled task)
scripts/                 Release, install, smoke-test scripts
```

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — service installs, model commands, storage, runtime control
- [docs/CLI.md](docs/CLI.md) — full CLI reference
- [docs/AGENTS.md](docs/AGENTS.md) — Goose, Claude Code, pi, OpenCode, curl, blackboard
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — benchmark numbers and context
- [docs/VERIFICATION.md](docs/VERIFICATION.md) — model-identity peer verification (fingerprints, the audit loop, observe vs enforce, privacy boundary)
- [CONTRIBUTING.md](CONTRIBUTING.md) — local development and build workflows
- [PLUGINS.md](PLUGINS.md) — plugin system and blackboard internals
- [senda/docs/VIRTUAL_LLM.md](senda/docs/VIRTUAL_LLM.md) — inter-model collaboration design
- [senda/docs/LLAMA_CPP_FORK.md](senda/docs/LLAMA_CPP_FORK.md) — llama.cpp patch queue maintenance
- [ROADMAP.md](ROADMAP.md) — future directions

## License

Apache-2.0 / MIT, dual-licensed. See [LICENSE](LICENSE).

Senda LLM is a fork of [Mesh-LLM/senda](https://github.com/senda-network/senda-llm) under the same dual-license terms.
