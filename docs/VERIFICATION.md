# Peer verification

Senda is an open network: anyone can run the runtime and advertise that
they serve a model. Verification answers one question — **is a peer that claims
to serve model X actually running model X, honestly?** — without trusting the
peer's self-report and without ever touching real user traffic.

It shipped in `v0.66.57` and runs in **observe-mode by default**: the verifier
logs verdicts and does not act on them unless enforcement is explicitly enabled.

## Why it exists

The native-baseline measurement ([benchmark honesty](#see-also)) already keeps
peers honest about *speed*. It does not catch a peer that serves at the claimed
speed while running a smaller/cheaper model than it advertises, or returning
canned text. Model-identity verification closes that gap, which is the
prerequisite for any reward/staking layer where misrepresenting a model is
profitable.

## The model-identity fingerprint

When a peer's `llama-server` becomes Ready for a model, the runtime issues a
single deterministic probe (`temperature=0`, fixed seed) directly to its own
`llama-server` and records a compact fingerprint:

- `output_sha256` — SHA-256 of the full greedy-decoded output text.
- `token_count` — number of decoded tokens.
- `prefix_tokens` — the first N decoded token strings (`FINGERPRINT_PREFIX_LEN`).
- `top_k_tokens` — the per-position top-k candidate token sets
  (`top_logprobs`), aligned 1:1 with `prefix_tokens`.

A different or smaller model, or canned text, produces a different greedy decode
for the same fixed prompt and diverges within the first few tokens. The
fingerprint is cached on disk alongside the timing baseline and gossiped to the
mesh.

> **Top-k candidates, not logprob values.** `top_logprobs` is requested at
> `temperature=0` purely to capture the per-position candidate *token sets*; the
> logprob magnitudes are not stored or compared. Token identity within the
> top-k is what lets the oracle tell an honest cross-backend near-tie flip from a
> wrong model (see below). Older fingerprints that predate this field still
> deserialize (the field defaults empty) and fall back to exact prefix matching.

## The comparison oracle

A verifier compares a *reference* fingerprint against a *candidate* fingerprint
produced by the suspect peer. Naively comparing token prefixes is unsound across
a heterogeneous mesh: even greedy `temperature=0` decoding diverges across
Metal / CUDA / Vulkan when an early token sits on a near-tie and two backends'
floating-point logits break it the other way — and because greedy decoding then
conditions on a different prefix, that single flip **cascades** the rest of the
sequence apart. A live observe-mode review found this false-flagged a genuinely
honest peer roughly half the time.

The oracle therefore **classifies the first divergence distributionally** rather
than counting prefix agreement:

1. Walk the prefix to the first token where the two decodes disagree. Up to
   there both sides decoded from a byte-identical prefix, so that position's
   candidate distributions are directly comparable; past it nothing is.
2. An honest **near-tie flip** keeps each side's chosen token inside the *other*
   side's top-k at that position → `Match`. A wrong/smaller model's token is
   **absent** from the real model's top-k → `Mismatch`.
3. When one side's fingerprint predates top-k capture, the oracle falls back to
   the original bounded-prefix agreement gate.

The verdict is one of `Match`, `Mismatch`, or `Inconclusive`.

## Two probe modes

The verifier loop runs on entry nodes, samples `(peer, model)` pairs, re-probes,
and logs the verdict.

- **Self-oracle (preferred).** When the verifying node also serves the model,
  each audit generates a fresh **nonce-randomized** probe, runs it on its own
  `llama-server` to get ground truth, and sends the identical probe to the
  suspect. Because the probe is unpredictable, a peer cannot recognise "the
  probe" and serve the real model only for it — this closes the known-prompt
  spoof.
- **Fixed reference (fallback).** When the verifier does not serve the model,
  the suspect is compared against a precomputed reference for a fixed probe.
  Spoofable by a peer that recognises the known prompt, but still catches the
  common cases — wrong/smaller model, canned replies, misconfiguration.

## Privacy boundary

**Verification only ever re-executes synthetic probes the verifier generates. It
never samples, replays, or duplicates real user traffic.**

Replaying a real user request against a second node would be more robust against
a peer that fingerprints synthetic probes — but it would fan a user's private
prompt out to a node that played no part in serving that request, expanding
plaintext exposure beyond the minimal serving path (the entry plus the one
host). That conflicts with Senda's privacy promise, so it is deliberately
not done. This boundary is intentional; do not "improve" verification by
sampling organic traffic.

## Observe vs enforce

Demotion is the one consequential lever — a false positive punishes an honest
contributor — so it is gated three ways:

1. **Off unless `SENDA_VERIFY_ENFORCE` is set** to a truthy value
   (`1`/`true`/`yes`/`on`). Default is observe-only: verdicts are logged, nothing
   is demoted.
2. **Requires several *consecutive* `Mismatch` verdicts** for the same
   `(peer, model)` before acting — never a single flaky probe. `Inconclusive`
   never counts toward conviction.
3. **The action is reversible and time-boxed.** A convicted peer is removed from
   the routable set for that model only, stays in the mesh, keeps being
   re-probed, and is reinstated on the next `Match` or when the cooldown lapses.
   This is route demotion, not slashing.

| `SENDA_VERIFY_ENFORCE` | Behaviour |
|---|---|
| unset / falsey (default) | Observe-only. Verdicts logged; routing unaffected. |
| `1` / `true` / `yes` / `on` | A peer with sustained mismatch is demoted from the routable set for that model, reversibly. |

## Establishing reference fingerprints

An auditor can capture a ground-truth reference for a `(model, quant)` from a
known-good local server:

```bash
senda benchmark capture-reference --model <model-id>
```

The embedded defaults live in `senda/src/inference/reference_fingerprints.json`
and now carry `top_k_tokens` so the fixed-reference path (used by CPU-only entry
nodes) gets the distributional classifier rather than the legacy prefix gate.
Recapture them against the current production server config when the bundled
decode drifts from what honest peers produce.

## Limits and deferred work

- **Coverage is limited to models a verifier serves locally** for the strong
  (self-oracle) path. Multi-peer consensus (proof-of-sampling) for models no
  verifier serves is deferred.
- A determined adversary who can statistically distinguish synthetic probes from
  organic traffic is out of scope here — that is left to a future
  staking/attestation layer, not to prompt snooping.
- Verdicts are not yet surfaced on the public status catalog.

## See also

- `senda/src/inference/verify.rs` — the oracle, the verifier loop, and the
  authoritative module docstring (including the privacy boundary).
- `senda/src/inference/native_baseline.rs` — fingerprint capture and the
  native timing baseline it rides alongside.
- [senda/docs/DESIGN.md](../senda/docs/DESIGN.md) — architecture and module map.
