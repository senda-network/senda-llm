<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/closedmesh-logo-dark.svg">
    <img src="docs/closedmesh-logo-light.svg" alt="ClosedMesh" width="380">
  </picture>
</p>

<h1 align="center">ClosedMesh LLM</h1>

<p align="center">
  <strong>The peer-to-peer inference runtime that powers ClosedMesh.</strong><br/>
  Pool spare GPU capacity across machines you own. Expose the result as one OpenAI-compatible API at <code>localhost:9337/v1</code>.
</p>

<p align="center">
  <a href="https://closedmesh.com">closedmesh.com</a> · 
  <a href="docs/USAGE.md">Usage</a> · 
  <a href="CONTRIBUTING.md">Contributing</a> · 
  <a href="ROADMAP.md">Roadmap</a>
</p>

---

ClosedMesh LLM is the open-source inference engine behind [ClosedMesh](https://closedmesh.com) — a private LLM product for teams that don't want to send their conversations to a third-party LLM API. The chat product runs in the browser. The runtime in this repo runs on the machines you already own and stitches them together into a single addressable inference pool.

If a model fits on one machine, it runs there. If it doesn't, ClosedMesh LLM automatically spreads the work:

- Dense models use **pipeline parallelism** — layers split across nodes proportional to VRAM.
- MoE models use **expert sharding** — experts split across nodes with zero cross-node inference traffic.
- Models can **collaborate during inference** — a text-only model consults a vision peer, an uncertain model gets a second opinion from a different architecture.
- Every node exposes the same local API at `http://localhost:9337/v1`.

## How this repo fits into the ClosedMesh product

| Piece | Repo | What it is |
|---|---|---|
| Chat product surface | private | The Next.js chat UI hosted at `closedmesh.com` and the local controller installed on each teammate's machine. |
| Inference runtime | **this repo** (`closedmesh/closedmesh-llm`) | The `closedmesh` binary. OpenAI-compatible API, peer-to-peer mesh, pipeline + MoE parallelism, capability-aware routing. |

The two are versioned and released independently. Most teams only ever install the runtime — the chat product talks to it for them.

> ClosedMesh LLM is a fork of [Mesh-LLM/closedmesh](https://github.com/closedmesh/closedmesh-llm). The Rust crate names are kept upstream-compatible (`closedmesh`, `mesh-client`, etc.) so we can rebase cleanly; the **shipped binary**, **service label**, and **data directory** are all `closedmesh`-branded.

## Quick start

Install the runtime on macOS, Linux, or WSL2:

```bash
curl -fsSL https://closedmesh.com/install | sh
```

```powershell
# Windows (PowerShell, no admin needed):
iwr -useb https://closedmesh.com/install.ps1 | iex
```

Auto-start at login (writes a launchd plist on macOS, a systemd `--user` unit on Linux, or a Scheduled Task on Windows):

```bash
curl -fsSL https://closedmesh.com/install | sh -s -- --service
```

Once installed:

```bash
closedmesh serve --auto                       # join the best public mesh, start serving
closedmesh service start                      # or run as a background service
closedmesh service status                     # status, pid, last logs
```

You now have:

- An OpenAI-compatible API at `http://localhost:9337/v1`.
- A live admin/management surface at `http://localhost:3131`.
- A local data dir at `~/.closedmesh/`.

Send a request:

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-8B-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

## Common workflows

### Private mesh (single-tenant / company fleet)

```bash
closedmesh serve --private-only --model Qwen2.5-32B
```

`--private-only` is a hard lock for deployments that must never accidentally publish to or auto-join a public mesh — internal company fleets, on-prem clusters, regulated environments. It conflicts at parse time with `--auto`, `--publish`, `--discover`, `--mesh-name`, and `--region`, so a misconfigured invocation fails fast instead of silently joining the wrong mesh. Joining via an explicit invite token (`--join <token>`) still works; only Nostr-based discovery and publishing are disabled.

### Add a second machine

The first time `closedmesh serve` runs, it prints an invite token. Anyone with reasonable hardware (a decent GPU, an Apple Silicon Mac, or even a beefy CPU-only Linux box) can join. The router learns each node's capability and only dispatches work that node can actually serve.

```bash
closedmesh serve --join <invite-token>
```

A 70B-class request only routes to nodes that advertise enough VRAM; an 8B model happily hops between an M-series Mac, an RTX 4090 box, and a Vulkan laptop in the same conversation.

### Multi-model

```bash
closedmesh serve --model Qwen2.5-32B --model GLM-4.7-Flash
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
closedmesh gpus                                # local GPUs, stable IDs, VRAM
closedmesh gpus --json                         # machine-readable inventory
closedmesh gpu benchmark --json                # refresh the local fingerprint
```

Use only pinnable `Stable ID` / `stable_id` values from `closedmesh gpus` for pinned startup config. Stable-ID fallback values such as `index:*` or backend-device names like `CUDA0` / `HIP0` / `MTL0` can still be printed for inventory purposes, but they are not valid pin targets.

## How it works

ClosedMesh LLM keeps the user-facing surface simple: talk to `localhost:9337`, pick a model, and let the mesh decide how to serve it.

- **Model fits on one machine?** It runs solo, full speed, no network overhead.
- **Dense model too big?** Pipeline parallelism — layers split across nodes.
- **MoE model too big?** Expert parallelism — experts split across nodes, zero cross-node traffic during inference.
- **Mixed hardware?** Capability matching ensures requests only go to nodes that can actually serve them.

If a node has enough VRAM, it always runs the full model. Splitting only happens when it has to.

**Pipeline parallelism** — for dense models that don't fit on one machine, layers are distributed across nodes proportional to VRAM. llama-server runs on the highest-VRAM node and coordinates via RPC. Each rpc-server loads only its assigned layers from local disk. Latency-aware: peers are selected by lowest RTT first, with an 80ms hard cap — high-latency nodes stay in the mesh as API clients but don't participate in splits.

**MoE expert parallelism** — Mixture-of-Experts models (Qwen3-MoE, GLM, OLMoE, Mixtral, DeepSeek — increasingly the best-performing architectures) are auto-detected from the GGUF header. The mesh reads expert routing statistics to identify which experts matter most, then assigns each node an overlapping shard: a shared core of critical experts replicated everywhere, plus unique experts distributed across nodes. Each node gets a standalone GGUF with the full trunk + its expert subset and runs its own independent llama-server — zero cross-node traffic during inference. Sessions are hash-routed to nodes for KV cache locality.

**Multi-model** — different nodes serve different models simultaneously. The API proxy peeks at the `model` field in each request and routes to the right node via QUIC tunnel. `/v1/models` lists everything available.

**Demand-aware rebalancing** — a unified demand map tracks which models the mesh wants (from `--model` flags, API requests, and gossip). Demand signals propagate infectiously across all nodes and decay naturally via TTL. Standby nodes auto-promote to serve unserved models with active demand, or rebalance when one model is significantly hotter than others.

**Inter-model collaboration** — models on the mesh help each other during inference. When a text-only model receives an image, it silently consults a vision model for a caption and generates from that. When a small model is uncertain, it races two peers for a second opinion and injects the winner's answer as context. The caller sees one seamless response — they don't know multiple models collaborated. See [closedmesh/docs/VIRTUAL_LLM.md](closedmesh/docs/VIRTUAL_LLM.md).

**Latency design** — HTTP streaming is latency-tolerant; RPC is latency-multiplied. llama-server always runs on the same box as the GPU. The mesh tunnels HTTP, so cross-network latency only affects time-to-first-token, not per-token throughput. RPC only crosses the network for pipeline splits where the model physically doesn't fit on one machine.

### Network optimizations

- **Zero-transfer GGUF loading** — `SET_TENSOR_GGUF` tells rpc-server to read weights from local disk. Dropped model load from 111s → 5s.
- **RPC round-trip reduction** — cached `get_alloc_size`, skip GGUF lookups for intermediates. Per-token round-trips: 558 → 8.
- **Direct server-to-server transfers** — intermediate tensors pushed directly between rpc-servers via TCP, not relayed through the client.
- **Speculative decoding** — draft model runs locally on the host, proposes tokens verified in one batched forward pass. +38% throughput on code (75% acceptance).

## Service mode

ClosedMesh runs as a per-user background service on all three platforms:

```bash
closedmesh service start            # start (auto-installs the unit on first run)
closedmesh service stop             # stop
closedmesh service status           # state, pid, recent logs
closedmesh service logs             # tail logs
```

Under the hood:

- macOS installs a launchd agent at `~/Library/LaunchAgents/dev.closedmesh.closedmesh.plist`.
- Linux installs a systemd `--user` unit at `~/.config/systemd/user/closedmesh.service`.
- Windows registers a Scheduled Task at user login.
- Shared environment config lives in `~/.config/closedmesh/service.env`.
- Startup models live in `~/.closedmesh/config.toml`.

On Linux this is a user service, so if you want it to keep running after reboot before login, enable lingering once:

```bash
sudo loginctl enable-linger "$USER"
```

## Output modes

`closedmesh` has two terminal output modes:

- `--log-format pretty` — human-readable. In `serve` on an interactive TTY, this becomes the full dashboard; otherwise it falls back to line-oriented pretty output.
- `--log-format json` — newline-delimited JSON records to `stdout`. Safe for `jq`, log shippers, and shell pipelines.

```json
{"timestamp":"...","level":"info","event":"llama_ready","model":"Qwen3-32B","port":8001,"ctx_size":8192,"message":"Qwen3-32B ready on internal port 8001"}
{"timestamp":"...","level":"info","event":"ready","api_url":"http://localhost:9337","console_url":"http://localhost:3131","api_port":9337,"console_port":3131,"models_count":2,"message":"closedmesh runtime ready"}
```

For the full event taxonomy and field reference, see [closedmesh/src/cli/output/EVENTS.md](closedmesh/src/cli/output/EVENTS.md).

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

`capability.backend` is the source of truth — it's what the rest of the mesh sees when matching your node to inference requests, and what the ClosedMesh chat product surfaces in its status pill and `/control` Nodes tab.

## Multimodal support

ClosedMesh LLM supports multimodal requests on:

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

For the full capability and transport details, see [closedmesh/docs/MULTI_MODAL.md](closedmesh/docs/MULTI_MODAL.md).

## Using with agents

ClosedMesh LLM exposes an OpenAI-compatible API on `localhost:9337`. Any tool that supports custom OpenAI endpoints works. `/v1/models` lists available models; the `model` field in requests routes to the right node.

Built-in launcher integrations:

```bash
closedmesh goose                     # launch goose against the local mesh
closedmesh opencode                  # launch opencode against the local mesh
```

- Goose and Claude reuse a local mesh on `--port` and auto-start a local client if needed.
- OpenCode targets `--host` (default `127.0.0.1:9337`) and only auto-starts a local client for loopback/localhost targets.
- If `--model` is omitted, the launcher picks the strongest tool-capable model available on the mesh.

### External OpenAI-compatible backends (vLLM, TGI, Ollama, Lemonade, etc.)

The `openai-endpoint` plugin routes inference to any server that speaks the OpenAI `/v1/chat/completions` API. The server does all the inference work — ClosedMesh just discovers its models and routes requests to it.

Enable the plugin in `~/.closedmesh/config.toml`:

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

Override auto-detection with `CLOSEDMESH_BACKEND=cuda|rocm|vulkan|cpu` when running the installer (handy for dual-GPU boxes or unusual setups).

## Build from source

Building from source only makes sense if you're hacking on the runtime itself or running on a platform we don't ship binaries for yet.

```bash
git clone https://github.com/closedmesh/closedmesh-llm
cd closedmesh-llm
just build           # patched llama.cpp + closedmesh binary + UI (~30 min first time)
./target/release/closedmesh models download Qwen3-0.6B-Q4_K_M  # fast, 397MB
./target/release/closedmesh models download Qwen3-8B-Q4_K_M    # demo model, 5GB
```

Requirements: `just`, `cmake`, Rust toolchain, Node.js 24 + npm. NVIDIA builds need `nvcc` (CUDA toolkit). AMD builds need ROCm/HIP. Vulkan builds need the Vulkan development files plus `glslc`. CPU-only and Jetson/Tegra also work.

On Linux, `just build` auto-detects CUDA vs ROCm vs Vulkan, or you can force `backend=rocm` / `backend=vulkan`. Windows source builds are supported for `cuda`, `rocm`/`hip`, `vulkan`, and `cpu` via `just build`.

For full build-from-source instructions and UI development, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Repo layout

```
closedmesh/                Rust crate — runtime, mesh, CLI (binary name: closedmesh)
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
- [CONTRIBUTING.md](CONTRIBUTING.md) — local development and build workflows
- [PLUGINS.md](PLUGINS.md) — plugin system and blackboard internals
- [closedmesh/docs/VIRTUAL_LLM.md](closedmesh/docs/VIRTUAL_LLM.md) — inter-model collaboration design
- [closedmesh/docs/LLAMA_CPP_FORK.md](closedmesh/docs/LLAMA_CPP_FORK.md) — llama.cpp patch queue maintenance
- [ROADMAP.md](ROADMAP.md) — future directions

## License

Apache-2.0 / MIT, dual-licensed. See [LICENSE](LICENSE).

ClosedMesh LLM is a fork of [Mesh-LLM/closedmesh](https://github.com/closedmesh/closedmesh-llm) under the same dual-license terms.
