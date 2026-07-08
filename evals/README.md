# senda Router Evals

A/B comparison of pi agent performance through senda's multi-model router vs a frontier cloud model.

## Setup

### Mesh (local multi-model)
```bash
# 3 models on M4 Max 52GB (~27GB total, room for KV cache)
SENDA_EPHEMERAL_KEY=1 senda \
  --model Qwen2.5-32B-Instruct-Q4_K_M \
  --model Qwen2.5-Coder-7B-Instruct-Q4_K_M \
  --model Hermes-2-Pro-Mistral-7B-Q4_K_M
```

Router auto-classifies each request and picks the best model:
- **Qwen2.5-32B** (tier 3) — reasoning, chat, complex code, tool use
- **Qwen2.5-Coder-7B** (tier 2) — code generation/review, fast (85 tok/s)
- **Hermes-7B** (tier 2) — fast chat, simple Q&A (87 tok/s, no tool use)

`SENDA_EPHEMERAL_KEY=1` uses a fresh identity so no external peers connect.

### Cloud baseline
Sonnet via `pi --provider anthropic --model claude-sonnet-4-20250514`.

## Scenarios

Multi-turn conversations that start with chat and progress to tool use:

| Scenario | Turns | What it tests |
|---|---|---|
| **chat-to-code** | 4 | Chat→write code→write tests→review (router must switch models) |
| **debug-session** | 4 | Read files→run code→find/fix bugs→verify (tool-heavy) |
| **edit-file** | 3 | Analyze→multi-step edits→verify (structured editing) |
| **html-app** | 3 | Generate code→validate→iterate (code generation) |
| **explore-repo** | 4 | Bash tools→read files→summarize (repo navigation) |
| **refactor** | 3 | Code review→refactor→verify (code quality) |

## Running

### Multi-turn (recommended — realistic)
```bash
# Single scenario
./evals/run-multi.sh mesh chat-to-code
./evals/run-multi.sh opus chat-to-code

# Compare results
./evals/compare.sh chat-to-code
```

### One-shot (quick, less realistic)
```bash
./evals/run.sh mesh edit-file
./evals/run.sh opus edit-file
```

## Results

Results go to `evals/results/<provider>/<scenario>/`:
- Working files (copied from scenario, edited by agent)
- `_output.txt` — full session capture
- `_screen_turnN.txt` — screen state after each turn
- `_time.txt` — wall clock seconds
- `_turns.txt` — number of turns completed

## What to look for

1. **Correctness** — Did it complete all turns? Are edits right?
2. **Tool use** — Did it use read/edit/bash appropriately?
3. **Routing** — Check `/tmp/senda-local.log` for which model handled each turn
4. **Speed** — Wall clock per scenario
5. **Model switching** — Does quality degrade when router changes models mid-conversation?
6. **Chat quality** — Are quick chat responses from Hermes comparable to 32B?

## Model capabilities (from testing)

| Model | Tool use | Code gen | Chat | Speed |
|---|---|---|---|---|
| Qwen2.5-32B | ✅ works | ✅ good | ✅ good | ~18 tok/s |
| Qwen2.5-Coder-7B | ✅ works | ✅ great | ⚠️ ok | ~85 tok/s |
| Hermes-7B | ❌ broken | ⚠️ basic | ✅ fast | ~87 tok/s |
| Qwen3-30B-A3B | ❌ thinking format | ✅ good | ❌ empty content | ~22 tok/s |
