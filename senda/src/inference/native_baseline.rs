//! v0.66.49 Phase 3.0 — benchmark honesty.
//!
//! When a peer's local llama-server becomes Ready for a model, the runtime
//! issues a single synthetic chat completion directly to
//! `127.0.0.1:<llama_port>/v1/chat/completions`, times decode + TTFT, and
//! records the result as that model's *native baseline* — what the
//! hardware can do without the entry tunnel, the auth gateway, or the
//! routing layer in the way. This number is gossiped alongside the
//! through-mesh `model_timings` collected from real chat traffic; the
//! catalog renders the ratio per `(peer, model)` pair so it is visible
//! whether the slow numbers we publish today are hardware-bound or
//! mesh-overhead-bound.
//!
//! Refresh policy:
//! - One baseline per `(model, model-file-mtime)` is cached on disk at
//!   `~/.senda/native-baselines.json`. The mtime is part of the cache
//!   key so swapping the model file (re-quantization, redownload) forces
//!   a fresh measurement.
//! - The cache TTL is 24h. Past TTL the existing baseline is still
//!   gossiped (so the catalog never goes blank) but a fresh measurement
//!   runs in the background and replaces it on completion.
//! - Failures (timeout, llama-server 5xx, network blip) are silent; we
//!   keep the previous baseline if any, retry on the next refresh tick,
//!   and emit a tracing warning rather than a user-visible error.
//!
//! Methodology caveats (deliberately simple):
//! - `samples = 3` per refresh, median of 3 published. v0.66.49 → v0.66.51
//!   ran a single sample per refresh, which a config sweep on the same
//!   M3 Pro showed had ~30 % single-shot variance — large enough to
//!   make the published number meaningfully different from the actual
//!   capability. v0.66.52 raised the count to 3 (sorted, middle value
//!   wins). The `samples` field on the wire was always there for this;
//!   the gossip schema didn't need to change.
//! - Each sample asks for ~80–128 output tokens with `temperature=0`. The
//!   first sample sends the fixed deterministic probe (which also produces
//!   the gossiped model-identity fingerprint); the remaining samples send
//!   *unique* cache-busting probes ([`probe_messages_for`] with a fresh
//!   nonce). This is deliberate: v0.66.49 → v0.66.56 sent the *same* probe
//!   for all 3 samples, so llama-server's prompt-KV cache replayed it and
//!   the published TTFT was ~20x optimistic (a cache hit, not a real
//!   prefill). Unique probes force a real prefill per sample, so the
//!   baseline reflects what a first-time user prompt actually pays. This is
//!   still not apples-to-apples with through-mesh traffic (variable prompt +
//!   output shapes); it is a *baseline* the through-mesh number references.
//! - Inter-sample delay is 1 s to let the GPU command queue drain between
//!   runs, small enough that the whole 3-sample sweep finishes in ~30 s on
//!   Apple Silicon. Failed samples in a sweep are dropped silently; we still
//!   publish a median across the surviving subset (down to 1) rather than
//!   blank the cache.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::mesh::NativeBaselineEntry;

const CACHE_FILENAME: &str = "native-baselines.json";
const CACHE_TTL_SECS: u64 = 24 * 60 * 60;
/// Match the metric path's `DECODE_DURATION_FLOOR` from `network::metrics`.
/// Non-streaming responses can collapse decode duration to ~0ms; falling
/// back to wall-clock is the same fix Phase 1 made to stop publishing
/// 953,000 t/s readings.
const DECODE_DURATION_FLOOR: Duration = Duration::from_millis(100);
/// Hard ceiling on a single synthetic baseline run. Real Apple-Silicon
/// llama-server cold-starts have been observed at ~20 s TTFT with a
/// further ~4 s decode for 128 tokens; budget is generous so the
/// collector doesn't kill a slow-but-honest peer's first measurement.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);
/// Output tokens we ask the model to produce. ~80–128 is enough that
/// decode rate dominates over TTFT noise without waiting forever on
/// slow hardware. `pub(crate)` so the remote verifier sends the identical
/// budget — fingerprints are only comparable across a byte-identical probe.
pub(crate) const MAX_TOKENS: u32 = 128;
/// Samples per refresh. 3 gives a median that tolerates one outlier
/// run without inflating measurement time past ~30 s. See module-level
/// methodology note for why this isn't 1 anymore.
const SAMPLES_PER_REFRESH: u32 = 3;
/// Pause between consecutive samples in a single refresh sweep. Gives the
/// GPU command queue a moment to drain between runs; short enough that 3
/// samples still finish in ~30 s on Apple Silicon. Note: this delay does
/// *not* defeat prompt-cache reuse (llama-server's cache TTL is far longer) —
/// that is why timing samples now use unique probes, not the same prompt.
const INTER_SAMPLE_DELAY: Duration = Duration::from_secs(1);
/// How long to wait before retrying a refresh that was deferred because the
/// node was busy serving real traffic (idle-gating). Short relative to the
/// TTL/2 steady-state cadence so a popular host still publishes a clean
/// baseline from its next idle gap instead of going stale for hours.
const BUSY_RETRY_DELAY: Duration = Duration::from_secs(120);
/// How many decoded tokens the fingerprint covers. The greedy decode of a
/// fixed prompt is a strong model-identity signal; a bounded prefix lets a
/// verifier compare with tolerance instead of demanding an exact
/// cross-backend match.
const FINGERPRINT_PREFIX_LEN: usize = 32;
/// Per-position candidate set size captured from `top_logprobs`. The oracle
/// uses it to tell an honest cross-backend near-tie flip (the flipped token is
/// in both sides' top-k) from a genuinely wrong model (its token isn't). `5`
/// is what `logprobs` clients conventionally request and is plenty to contain
/// a near-tie pair while staying tight enough that a wrong model falls out.
pub(crate) const TOP_K_PER_POSITION: usize = 5;

/// The exact prompt + sampling params the deterministic probe sends.
/// Shared by the local baseline collector and the remote verifier so
/// their fingerprints are comparable — the fingerprint is only meaningful
/// if both sides issue a byte-identical request.
pub(crate) const PROBE_SYSTEM_PROMPT: &str = "You are a benchmark probe. Respond concisely.";
pub(crate) const PROBE_USER_PROMPT: &str = "Write a short paragraph (about 100 words) explaining \
     why direct measurement beats marketing claims when \
     comparing distributed systems performance.";
/// Sampling seed shared by the baseline + verifier probes.
pub(crate) const PROBE_SEED: u64 = 42;

/// The probe's `messages` array. Identical on both the local and remote
/// sides so the greedy decode (and thus the fingerprint) lines up.
pub(crate) fn probe_messages() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({"role": "system", "content": PROBE_SYSTEM_PROMPT}),
        serde_json::json!({"role": "user", "content": PROBE_USER_PROMPT}),
    ]
}

/// Topic pool for the *randomized* verifier probe. Combined with a per-audit
/// nonce these make each probe unique, so a peer can't precompute answers or
/// recognize "the probe" and serve the real model only for it. The topics are
/// substantive enough that a wrong/smaller model diverges on the greedy decode.
pub(crate) const PROBE_TOPICS: &[&str] = &[
    "latency percentiles",
    "cache invalidation",
    "consensus under network partition",
    "backpressure",
    "idempotent retries",
    "tail latency amplification",
    "clock skew",
    "load shedding",
    "connection pooling",
    "write amplification",
    "quorum reads",
    "head-of-line blocking",
    "exactly-once delivery",
    "circuit breakers",
    "bloom filters",
    "vector clocks",
];

/// A fresh, unpredictable probe keyed by `nonce`. Used by the verifier's
/// self-oracle: it generates a nonce per audit, runs this probe against its
/// own llama-server for ground truth, and sends the *identical* probe to the
/// suspect. Deterministic for a given nonce (temp=0 greedy), so the two sides
/// are comparable, but unpredictable across audits.
pub(crate) fn probe_messages_for(nonce: u64) -> Vec<serde_json::Value> {
    let topic = PROBE_TOPICS[(nonce as usize) % PROBE_TOPICS.len()];
    let user = format!(
        "Probe {nonce}. In about 80 words, explain why {topic} matters when \
         operating a distributed inference system. Be concrete and concise."
    );
    vec![
        serde_json::json!({"role": "system", "content": PROBE_SYSTEM_PROMPT}),
        serde_json::json!({"role": "user", "content": user}),
    ]
}

// ── Layer 1 — first-token top-k battery (gross-fraud verifier) ───────────────
//
// The deterministic *output-prefix* fingerprint above can only distinguish a
// GROSS substitution on this stack: the bundled llama.cpp (b9109) attaches
// `top_logprobs` to only the first generated token in chat completions, and
// greedy decode is reproducible only on an idle server, so a multi-token
// prefix compare reads cross-backend drift + load jitter as divergence (see
// internal/RESILIENCE.md, 2026-06-04). The robust, calibrated signal is the
// *first token's top-k candidate set*, captured over a battery of short prompts
// via the native `/completion` endpoint (`n_predict:1`, which always carries
// first-token top-k, even on b9109). A genuine peer is a mutual near-tie with
// the reference on ~100 % of the battery — idle, under load, and across
// Metal/CUDA backends — while canned output or a wrong/much-smaller model falls
// far below. Same-family downgrades (e.g. 4B-as-8B) are deliberately NOT in
// scope here; they are left to the performance-profile layer (L4).

/// A battery of short, discriminative completion prompts. Each one has a
/// confident, specific first token for the genuine model, so a wrong/much-
/// smaller model or canned text diverges out of the top-k. Sent verbatim as a
/// native `/completion` prompt (no chat template) so the first token is a raw
/// continuation, matching how the reference battery is captured. Changing this
/// list requires recapturing the embedded reference battery (see
/// `verify::ReferenceBatteryStore`); entries are paired by index.
pub(crate) const VERIFY_BATTERY: &[&str] = &[
    "Q: 2+2? A:",
    "The chemical symbol for gold is",
    "The opposite of 'hot' is",
    "import numpy as",
    "def add(a, b):\n    return",
    "The capital of Japan is",
    "Roses are red, violets are",
    "The first three prime numbers are 2, 3, and",
    "In Python, to print to the console you use the function named",
    "The largest planet in our solar system is",
    "To create a new git branch you run: git",
    "The square root of 144 is",
    "HTML stands for HyperText Markup",
    "The author of 'Romeo and Juliet' is William",
    "1, 1, 2, 3, 5, 8, 13,",
    "Water is made of hydrogen and",
];

/// A single battery probe's result: the genuine first token and its top-k
/// candidate set (`top_logprobs[].token`, includes the chosen token). The
/// verifier compares these by *mutual top-k membership* (a near-tie), which is
/// robust to the cross-backend argmax flips that broke exact matching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstTokenProbe {
    pub token: String,
    pub top_k: Vec<String>,
}

/// The native `/completion` request body for one battery probe. Shared by the
/// local reference capture and the remote verifier so both sides issue a
/// byte-identical request — `cache_prompt:false` is load-bearing: it forces a
/// real prefill so the first-token logits are deterministic instead of replayed
/// from a KV-cache hit (see RESILIENCE.md).
pub(crate) fn first_token_request_body(prompt: &str) -> serde_json::Value {
    serde_json::json!({
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0,
        "seed": PROBE_SEED,
        "n_probs": TOP_K_PER_POSITION,
        "cache_prompt": false,
        "stream": false,
    })
}

/// Parse a native `/completion` response into the first token + its top-k. The
/// native endpoint reports per-token candidates under
/// `completion_probabilities[].top_logprobs[].token`; we only need position 0.
pub(crate) fn first_token_from_completion_json(v: &serde_json::Value) -> Option<FirstTokenProbe> {
    let first = v.get("completion_probabilities")?.as_array()?.first()?;
    let token = first.get("token")?.as_str()?.to_string();
    let top_k = first
        .get("top_logprobs")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| c.get("token").and_then(|t| t.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Some(FirstTokenProbe { token, top_k })
}

/// Capture one battery probe against a local llama-server's native
/// `/completion` endpoint. Used for the self-oracle reference and for the
/// `benchmark capture-reference-battery` subcommand.
pub(crate) async fn local_first_token(
    http_port: u16,
    prompt: &str,
) -> anyhow::Result<FirstTokenProbe> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| anyhow::anyhow!("build reqwest client: {e}"))?;
    let url = format!("http://127.0.0.1:{http_port}/completion");
    let resp = client
        .post(&url)
        .json(&first_token_request_body(prompt))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("local first-token probe failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("local first-token probe returned {status}");
    }
    let parsed: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("parse local first-token probe: {e}"))?;
    first_token_from_completion_json(&parsed)
        .ok_or_else(|| anyhow::anyhow!("local first-token probe had no usable output"))
}

/// Capture the full reference battery from a local llama-server. Returns the
/// per-prompt first-token probes in `VERIFY_BATTERY` order (index-aligned).
pub(crate) async fn capture_reference_battery(
    http_port: u16,
) -> anyhow::Result<Vec<FirstTokenProbe>> {
    let mut out = Vec::with_capacity(VERIFY_BATTERY.len());
    for prompt in VERIFY_BATTERY {
        out.push(local_first_token(http_port, prompt).await?);
    }
    Ok(out)
}

/// On-disk cache shape. Keyed by model name; the entry stores
/// `model_file_mtime_secs` so reinstalling/requantizing the model
/// invalidates the cache without us needing to hash GBs of weights.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NativeBaselineCache {
    #[serde(default = "current_schema_version")]
    pub version: u32,
    #[serde(default)]
    pub entries: HashMap<String, CachedNativeBaseline>,
}

fn current_schema_version() -> u32 {
    1
}

/// Runtime crate version. Stamped on every cache entry so a runtime upgrade
/// busts the baseline cache even when the model file is byte-identical: a new
/// release can change launch flags (e.g. v0.66.59 enabled no-think serving),
/// which changes both the timing baseline and the model-identity fingerprint,
/// and the mtime/TTL freshness keys can't see a flag change.
const RUNTIME_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedNativeBaseline {
    pub model: String,
    pub native_tps_p50: f64,
    pub native_ttft_ms_p50: u64,
    pub measured_at_unix_secs: u64,
    pub samples: u32,
    pub backend: String,
    /// `mtime` of the model file at measurement time. Cache miss when
    /// this differs from the live file's mtime — covers re-quantization
    /// and redownload without us having to hash the weights.
    pub model_file_mtime_secs: Option<u64>,
    /// Deterministic model-identity fingerprint from the same probe.
    /// `None` for caches written before this field existed (serde default)
    /// or when the probe produced no output. Captured, cached, and gossiped;
    /// consumed by the verifier loop in `inference::verify`. Enforcement
    /// (reversible peer demotion) is gated behind `SENDA_VERIFY_ENFORCE`
    /// and off by default — see that module for the observe-vs-enforce policy.
    #[serde(default)]
    pub logit_fingerprint: Option<LogitFingerprint>,
    /// Runtime crate version that wrote this entry (see [`RUNTIME_VERSION`]).
    /// `None` for caches written before this field existed → treated as stale
    /// so they get re-measured under the current runtime. A version mismatch
    /// busts the cache because a release can change launch flags that alter
    /// the baseline/fingerprint without touching the model file.
    #[serde(default)]
    pub runtime_version: Option<String>,
}

/// Deterministic model-identity fingerprint captured from the same
/// temp=0 / seed=42 probe that produces the timing baseline.
///
/// Why it exists: a peer that claims to serve model X but actually runs a
/// smaller/different model — or returns canned text — produces a different
/// greedy decode for this fixed prompt. A verifier (the entry, or a sampled
/// second peer) re-runs the identical probe and compares. We store the
/// full-output hash plus a bounded prefix of decoded token strings so the
/// comparison can be tolerance-based (prefix agreement) rather than an exact
/// byte match — exact equality is impossible across Metal / CUDA / Vulkan
/// because of floating-point divergence in the argmax tail.
///
/// Per-token logprobs were intentionally dropped: at `temperature=0` the
/// chosen token's logprob is definitionally 0 (prob 1.0) and llama.cpp
/// returns no alternatives, so they carried no signal. The token sequence
/// and output hash are the discriminators.
///
/// This struct is captured, cached, and gossiped, and is consumed by the
/// verifier loop in [`crate::inference::verify`]. Enforcement (reversible,
/// time-boxed peer demotion on sustained mismatch) is gated behind
/// `SENDA_VERIFY_ENFORCE` and off by default, so capturing/gossiping a
/// fingerprint has the same risk profile as the timing baseline — observe-only
/// until enforcement is explicitly enabled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitFingerprint {
    /// Number of decoded tokens observed for the probe (`0` when the backend
    /// returned no per-token data).
    pub token_count: u32,
    /// Lowercase-hex SHA-256 of the full greedy-decoded output text.
    pub output_sha256: String,
    /// First `FINGERPRINT_PREFIX_LEN` decoded token strings. Empty when the
    /// backend returned no per-token data — the output hash still
    /// fingerprints the model in that case.
    pub prefix_tokens: Vec<String>,
    /// Per-position top-k candidate token strings (`top_logprobs[].token`),
    /// aligned 1:1 with `prefix_tokens`, each inner vec ≤ `TOP_K_PER_POSITION`.
    /// Lets the oracle distinguish an honest cross-backend near-tie flip (the
    /// other side's token sits in this position's top-k) from a wrong model.
    /// Empty when the backend returned no `top_logprobs`; `#[serde(default)]`
    /// so older gossiped/bundled fingerprints (which lack the field) still
    /// deserialize and the oracle falls back to exact prefix matching.
    #[serde(default)]
    pub top_k_tokens: Vec<Vec<String>>,
}

/// Build a [`LogitFingerprint`] from the decoded token stream and the
/// concatenated output text. Pure + deterministic so it is unit-testable
/// without an llama-server.
pub(crate) fn build_fingerprint(
    tokens: &[String],
    top_k_tokens: &[Vec<String>],
    full_text: &str,
) -> LogitFingerprint {
    let mut hasher = Sha256::new();
    hasher.update(full_text.as_bytes());
    let output_sha256 = hex::encode(hasher.finalize());
    let prefix_tokens = tokens
        .iter()
        .take(FINGERPRINT_PREFIX_LEN)
        .cloned()
        .collect();
    let top_k_tokens = top_k_tokens
        .iter()
        .take(FINGERPRINT_PREFIX_LEN)
        .cloned()
        .collect();
    LogitFingerprint {
        token_count: tokens.len() as u32,
        output_sha256,
        prefix_tokens,
        top_k_tokens,
    }
}

/// Pull the chosen-token stream and the per-position top-k candidate sets out
/// of an OpenAI `logprobs.content` array (`[{token, top_logprobs:[{token}]}]`).
/// A backend that omits `top_logprobs` yields empty inner vecs, which makes the
/// oracle fall back to exact prefix matching for that fingerprint.
fn extract_token_stream(content: &[serde_json::Value]) -> (Vec<String>, Vec<Vec<String>>) {
    let mut tokens = Vec::new();
    let mut top_k = Vec::new();
    for entry in content {
        let Some(tok) = entry.get("token").and_then(|t| t.as_str()) else {
            continue;
        };
        tokens.push(tok.to_string());
        let mut candidates = Vec::new();
        if let Some(arr) = entry.get("top_logprobs").and_then(|a| a.as_array()) {
            for c in arr {
                if let Some(t) = c.get("token").and_then(|t| t.as_str()) {
                    candidates.push(t.to_string());
                }
            }
        }
        top_k.push(candidates);
    }
    (tokens, top_k)
}

/// Build a fingerprint from a *non-streaming* chat-completion JSON value —
/// the shape the remote verifier reads back over the mesh tunnel. Mirrors
/// the SSE path: text from `choices[0].message.content`, the per-token
/// stream from `choices[0].logprobs.content[].token`. `None` when the
/// response carried no output text.
pub(crate) fn fingerprint_from_completion_json(v: &serde_json::Value) -> Option<LogitFingerprint> {
    let choice = v.get("choices")?.get(0)?;
    let text = choice
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    if text.is_empty() {
        return None;
    }
    let (tokens, top_k) = choice
        .get("logprobs")
        .and_then(|l| l.get("content"))
        .and_then(|c| c.as_array())
        .map(|content| extract_token_stream(content))
        .unwrap_or_default();
    Some(build_fingerprint(&tokens, &top_k, &text))
}

/// Issue a probe to a *local* llama-server (non-streaming) and build the
/// fingerprint via the exact same code path the remote verifier uses
/// ([`fingerprint_from_completion_json`]) — so a local oracle's fingerprint
/// and a remote candidate are guaranteed to be constructed identically.
/// `messages`/`seed` are the probe spec; pass [`probe_messages`] + [`PROBE_SEED`]
/// for the fixed reference, or [`probe_messages_for`] + the nonce for the
/// verifier's randomized self-oracle.
pub(crate) async fn local_probe_fingerprint(
    http_port: u16,
    model: &str,
    messages: Vec<serde_json::Value>,
    seed: u64,
) -> anyhow::Result<LogitFingerprint> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| anyhow::anyhow!("build reqwest client: {e}"))?;
    let url = format!("http://127.0.0.1:{http_port}/v1/chat/completions");
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": seed,
        "stream": false,
        "logprobs": true,
        // Per-position candidate sets for the distributional oracle (see
        // verify::compare_fingerprints). Lets it accept an honest cross-backend
        // near-tie flip while still rejecting a wrong model.
        "top_logprobs": TOP_K_PER_POSITION,
        // Same no-think pin as the streaming probe so the reference fingerprint
        // matches what the daily driver gossips (see measure_baseline).
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("local probe request failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("local probe llama-server returned {status}");
    }
    let parsed: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("parse local probe response: {e}"))?;
    fingerprint_from_completion_json(&parsed)
        .ok_or_else(|| anyhow::anyhow!("local probe response had no usable output"))
}

/// Capture an auditor *reference* fingerprint with the fixed deterministic
/// probe. Used by the `benchmark capture-reference` subcommand.
pub(crate) async fn capture_reference_fingerprint(
    http_port: u16,
    model: &str,
) -> anyhow::Result<LogitFingerprint> {
    local_probe_fingerprint(http_port, model, probe_messages(), PROBE_SEED).await
}

/// Parse an SSE chat-completion body into the concatenated output text and
/// the per-token stream. Mirrors the lenient line-walking the usage parser
/// uses; a backend that omits `logprobs` yields an empty token stream (the
/// text is still recovered). Used only to build the deterministic
/// fingerprint.
fn parse_output_and_tokens_from_sse(buf: &[u8]) -> (String, Vec<String>) {
    let mut text = String::new();
    let mut tokens: Vec<String> = Vec::new();
    let Ok(body) = std::str::from_utf8(buf) else {
        return (text, tokens);
    };
    for line in body.lines() {
        let line = line.trim();
        let payload = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        let Some(choice) = v.get("choices").and_then(|c| c.get(0)) else {
            continue;
        };
        if let Some(content) = choice
            .get("delta")
            .and_then(|d| d.get("content"))
            .and_then(|c| c.as_str())
        {
            text.push_str(content);
        }
        if let Some(content) = choice
            .get("logprobs")
            .and_then(|l| l.get("content"))
            .and_then(|c| c.as_array())
        {
            for entry in content {
                if let Some(tok) = entry.get("token").and_then(|t| t.as_str()) {
                    tokens.push(tok.to_string());
                }
            }
        }
    }
    (text, tokens)
}

impl From<&CachedNativeBaseline> for NativeBaselineEntry {
    fn from(c: &CachedNativeBaseline) -> Self {
        NativeBaselineEntry {
            model: c.model.clone(),
            native_tps_p50: c.native_tps_p50,
            native_ttft_ms_p50: c.native_ttft_ms_p50,
            measured_at_unix_secs: c.measured_at_unix_secs,
            samples: c.samples,
            backend: c.backend.clone(),
            logit_fingerprint: c.logit_fingerprint.clone(),
        }
    }
}

/// Returns the path of the native-baseline cache file (`~/.senda/
/// native-baselines.json`). Honors `SENDA_HOME` / `HOME` the same
/// way the rest of the runtime does.
pub fn cache_path() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("SENDA_HOME") {
        return Some(PathBuf::from(custom).join(CACHE_FILENAME));
    }
    let home = dirs::home_dir()?;
    Some(home.join(".senda").join(CACHE_FILENAME))
}

pub fn load_cache(path: &std::path::Path) -> NativeBaselineCache {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
        Err(_) => NativeBaselineCache::default(),
    }
}

/// Persist the cache atomically (write tmp, rename). If the parent
/// directory doesn't exist, create it first.
pub fn save_cache(path: &std::path::Path, cache: &NativeBaselineCache) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(cache).unwrap_or_else(|_| "{}".to_string());
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)
}

/// Inputs for a single baseline measurement. The model and port are
/// known to the caller (the election loop holds them when it spawns the
/// llama-server); the backend label is passed through so we don't have
/// to re-read hardware here.
#[derive(Debug, Clone)]
pub struct BaselineRequest {
    pub model: String,
    pub http_port: u16,
    pub backend: String,
}

/// Outcome of a single synthetic baseline run.
#[derive(Debug, Clone)]
pub struct BaselineMeasurement {
    pub native_tps_p50: f64,
    pub native_ttft_ms_p50: u64,
    pub backend: String,
    pub measured_at_unix_secs: u64,
    /// Deterministic logit fingerprint from this run. `None` when the
    /// probe produced no output bytes.
    pub logit_fingerprint: Option<LogitFingerprint>,
}

/// Issue a synthetic chat completion to the local llama-server and
/// measure TTFT + decode rate. Streaming so TTFT is the byte-level
/// time-to-first-chunk, not the request total.
///
/// `messages` is the probe to send. Pass [`probe_messages`] for the fixed
/// deterministic reference (so the fingerprint stays comparable across
/// peers) or [`probe_messages_for`] with a fresh nonce for a unique,
/// cache-busting timing sample. `capture_fingerprint` must be `true` only
/// for the fixed probe: a unique probe's fingerprint is not comparable to
/// any reference, so timing samples leave it `None` and never feed the
/// gossiped value.
///
/// Errors when the request times out, returns a non-200, or doesn't
/// produce any output tokens. Callers log + retain the previous cached
/// baseline rather than blanking the gossiped value.
pub async fn measure_baseline(
    req: &BaselineRequest,
    messages: Vec<serde_json::Value>,
    capture_fingerprint: bool,
) -> anyhow::Result<BaselineMeasurement> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| anyhow::anyhow!("build reqwest client: {e}"))?;

    let url = format!("http://127.0.0.1:{}/v1/chat/completions", req.http_port);
    let body = serde_json::json!({
        "model": req.model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": PROBE_SEED,
        "stream": true,
        "stream_options": {"include_usage": true},
        // NB: deliberately NO `logprobs` here. Requesting logprobs makes
        // llama.cpp run a per-token full-vocab probability extraction that
        // ~halves decode throughput (measured on v0.66.70: 110→45 tok/s for
        // Qwen3-8B on a 4080 SUPER) — the same full-vocab-op defect class as
        // the v0.66.69 mesh-hook tax. This is the *timing* probe, so it must
        // measure what real chat traffic — which never asks for logprobs —
        // actually pays. The gossiped fingerprint is captured separately by the
        // non-streaming `local_probe_fingerprint` request below (with its own
        // top_logprobs); the SSE fallback just yields an output-hash-only
        // fingerprint (empty token prefix), which still fingerprints the model.
        // Pin thinking off so the timing baseline and the gossiped fingerprint
        // reflect the same no-think path the daily driver actually serves
        // (launch.rs sets this server-wide; we also send it per-request so the
        // fingerprint is identical regardless of a peer's server launch flags,
        // and the timing samples don't pay the empty-`<think>` scaffold tax).
        "chat_template_kwargs": {"enable_thinking": false},
    });

    let request_started = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("native baseline request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("native baseline llama-server returned {status}");
    }

    // Stream the body and time the first byte separately from the rest of
    // the body. The OpenAI streaming `usage` chunk lands in the SSE tail;
    // we extract `completion_tokens` to compute decode rate honestly.
    let mut stream = resp.bytes_stream();
    use tokio_stream::StreamExt;

    let mut first_byte_at: Option<Instant> = None;
    let mut last_chunk_at: Option<Instant> = None;
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| anyhow::anyhow!("streaming read failed: {e}"))?;
        if first_byte_at.is_none() {
            first_byte_at = Some(Instant::now());
        }
        last_chunk_at = Some(Instant::now());
        buf.extend_from_slice(&chunk);
    }

    let first_byte_at =
        first_byte_at.ok_or_else(|| anyhow::anyhow!("native baseline produced no bytes"))?;
    let last_chunk_at = last_chunk_at.unwrap_or(first_byte_at);
    let ttft = first_byte_at.saturating_duration_since(request_started);
    let decode_duration = last_chunk_at.saturating_duration_since(first_byte_at);

    let completion_tokens = parse_completion_tokens_from_sse_tail(&buf).unwrap_or(0);
    if completion_tokens == 0 {
        anyhow::bail!("native baseline streaming response had no usage chunk");
    }

    let effective_decode = if decode_duration < DECODE_DURATION_FLOOR {
        ttft.saturating_add(decode_duration)
    } else {
        decode_duration
    };
    let secs = effective_decode.as_secs_f64();
    if secs <= 0.0 {
        anyhow::bail!("native baseline decode window collapsed to zero");
    }
    let native_tps_p50 = (completion_tokens as f64) / secs;

    // Build the deterministic fingerprint from the same buffered body — but
    // only for the fixed reference probe. Unique cache-busting timing probes
    // produce per-probe output that isn't comparable to any reference, so we
    // skip the build and leave the fingerprint `None`; this keeps the gossiped
    // fingerprint exactly the fixed-probe one (or `None`), never a stray
    // timing-probe hash. No extra request; the probe already streamed it.
    let logit_fingerprint = if capture_fingerprint {
        // The gossiped fingerprint has to line up with what the verifier
        // compares against, and the verifier always probes non-streaming on
        // both sides (`local_probe_fingerprint` / `remote_probe_fingerprint`).
        // Streaming logprobs are lossy under speculative decoding — several
        // draft-accepted tokens can arrive in one chunk behind a single
        // logprob entry — so parsing the SSE stream yields a token list that
        // wouldn't line up with the reference (observed: gossiped prefix
        // `["Direct", " objective", ...]` vs reference `["Direct",
        // " measurement", " provides", " objective", ...]`). Capture the
        // gossiped fingerprint from the same canonical non-streaming probe
        // (one extra request, fixed sample only, once per refresh). Fall back
        // to the streaming buffer only if that probe fails, so a transient
        // error still yields *some* fingerprint rather than none.
        match local_probe_fingerprint(req.http_port, &req.model, probe_messages(), PROBE_SEED).await
        {
            Ok(fp) => Some(fp),
            Err(_) => {
                // The timing probe no longer requests logprobs (it must not pay
                // the per-token tax — see the request body above), so the SSE
                // buffer carries no per-token segmentation: `tokens` is empty
                // and the oracle falls back to exact prefix matching for this
                // (rare) fingerprint. The output hash still fingerprints the model.
                let (output_text, tokens) = parse_output_and_tokens_from_sse(&buf);
                (!output_text.is_empty()).then(|| build_fingerprint(&tokens, &[], &output_text))
            }
        }
    } else {
        None
    };

    Ok(BaselineMeasurement {
        native_tps_p50,
        native_ttft_ms_p50: ttft.as_millis().min(u64::MAX as u128) as u64,
        backend: req.backend.clone(),
        measured_at_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        logit_fingerprint,
    })
}

/// Extract `usage.completion_tokens` from an SSE response body. Mirrors
/// the same scan logic the through-mesh metric path uses
/// (`network::openai::transport::parse_completion_tokens_from_sse_tail`).
/// Inlined here so the baseline collector doesn't depend on an internal
/// transport module that may move.
fn parse_completion_tokens_from_sse_tail(buf: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(buf).ok()?;
    // Walk SSE lines in reverse looking for the `usage` chunk.
    for line in text.lines().rev() {
        let line = line.trim();
        let payload = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(tokens) = v
            .get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|t| t.as_u64())
        {
            return Some(tokens);
        }
    }
    None
}

/// True when `entry` is fresh enough for the catalog to display without
/// rerunning the measurement. Past TTL the entry is still useful (we
/// keep gossiping it while the new run executes) but a refresh tick
/// kicks off in the background.
pub fn is_fresh(entry: &CachedNativeBaseline, model_file_mtime_secs: Option<u64>) -> bool {
    // A runtime upgrade can change launch flags (e.g. enabling no-think
    // serving) which changes the baseline and the fingerprint even though the
    // model file is byte-identical. The mtime/TTL keys below can't see that, so
    // bust the cache whenever the entry was written by a different runtime
    // version. `None` (pre-stamp caches) is treated as stale for the same
    // reason — re-measure once under the current runtime to stamp it.
    if entry.runtime_version.as_deref() != Some(RUNTIME_VERSION) {
        return false;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if now.saturating_sub(entry.measured_at_unix_secs) > CACHE_TTL_SECS {
        return false;
    }
    match (entry.model_file_mtime_secs, model_file_mtime_secs) {
        // Both known and equal — fresh.
        (Some(a), Some(b)) if a == b => true,
        // Mtime missing on one side — be conservative and consider fresh
        // only when the wall-clock TTL hasn't expired (which we already
        // checked above).
        (None, _) | (_, None) => true,
        // Both known and different — model file changed, force re-run.
        _ => false,
    }
}

/// Convenience: read the live mtime of a model file as Unix seconds.
/// Returns `None` for missing files, permission errors, or platforms
/// that don't expose mtime.
pub fn model_file_mtime_secs(path: &std::path::Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    mtime.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

/// Reduce a vec of single-sample measurements to one median measurement
/// plus the count of surviving samples. Pulled out of
/// [`measure_baseline_median`] so the median selection is testable
/// without spinning up an HTTP server.
///
/// Returns `None` for an empty input. TPS and TTFT are sorted and
/// medianed independently — they're correlated (a slow run is slow on
/// both) but not identical, and per-axis sort matches what a downstream
/// reader would compute from the raw samples themselves.
fn median_measurement(
    samples: Vec<BaselineMeasurement>,
    backend: String,
) -> Option<(BaselineMeasurement, u32)> {
    if samples.is_empty() {
        return None;
    }
    let mut tps: Vec<f64> = samples.iter().map(|m| m.native_tps_p50).collect();
    let mut ttft: Vec<u64> = samples.iter().map(|m| m.native_ttft_ms_p50).collect();
    tps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ttft.sort();
    let mid = samples.len() / 2;
    // Only the fixed-probe sample (index 0) carries a fingerprint; the
    // cache-busting timing samples leave it `None`. `find_map` therefore
    // yields the fixed-probe fingerprint when sample 0 succeeded, else `None`
    // — never a stray timing-probe hash.
    let logit_fingerprint = samples.iter().find_map(|m| m.logit_fingerprint.clone());
    let sample_count = samples.len() as u32;
    Some((
        BaselineMeasurement {
            native_tps_p50: tps[mid],
            native_ttft_ms_p50: ttft[mid],
            backend,
            measured_at_unix_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            logit_fingerprint,
        },
        sample_count,
    ))
}

/// A per-sample nonce for the cache-busting timing probe. Time-derived so it
/// differs across samples (the inter-sample delay guarantees distinct nanos)
/// and across refreshes, so [`probe_messages_for`] yields a prompt
/// llama-server's prompt-KV cache has never seen — forcing a real prefill.
fn baseline_timing_nonce(sample_index: u32) -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        .wrapping_add(sample_index as u64)
}

/// Why a fresh measurement didn't produce a clean published baseline.
enum BaselineError {
    /// The node was serving real traffic for every sample window, so no
    /// uncontended sample was taken. Transient — keep the prior clean value
    /// and retry on a short cadence.
    Busy,
    /// Every sample errored (timeout, non-200, no output). Genuine failure.
    Failed(anyhow::Error),
}

/// Run [`measure_baseline`] up to `n` times with a 1 s gap between
/// samples and return the median TPS / TTFT across the runs that
/// succeeded. `Err` only when *every* sample failed; partial sample
/// sets (1 of 3, 2 of 3) still produce a usable measurement so a
/// flaky network blip doesn't blank a peer's published baseline.
///
/// **Idle-gated.** Each sample is only kept if no real request was in flight
/// on `node` for the whole sample window (checked immediately before and after
/// the call). A sample that overlaps live traffic is contended — the GPU is
/// shared, so decode tok/s reads far below the true rate — and would otherwise
/// be cached as "truth" and gossiped for hours (the 25-vs-121 tok/s
/// discrepancy that made this signal untrustworthy for the perf-profile
/// verifier). If every sample is contended this returns
/// [`BaselineError::Busy`] so the caller keeps the prior clean value and
/// retries soon, rather than publishing a contended number.
async fn measure_baseline_median(
    node: &crate::mesh::Node,
    req: &BaselineRequest,
    n: u32,
) -> Result<(BaselineMeasurement, u32), BaselineError> {
    let mut samples: Vec<BaselineMeasurement> = Vec::with_capacity(n as usize);
    let mut last_err: Option<anyhow::Error> = None;
    let mut skipped_busy = false;
    for i in 0..n {
        if i > 0 {
            tokio::time::sleep(INTER_SAMPLE_DELAY).await;
        }
        // Don't measure while a real request shares the GPU — the sample would
        // read low and poison the gossiped baseline.
        if node.inflight_requests() > 0 {
            skipped_busy = true;
            continue;
        }
        // Sample 0 sends the fixed deterministic probe: it yields the
        // canonical gossiped fingerprint plus one timing sample. Samples 1..
        // send unique, cache-busting probes so TTFT reflects real first-time
        // prefill instead of llama-server's prompt-KV cache replaying an
        // identical prompt (the v0.66.49 bug: 3 back-to-back identical probes
        // made samples 2/3 cache hits, so the median TTFT was ~20x optimistic).
        // With n=3 the lone fixed-probe sample is the extreme on both axes, so
        // the median excludes it and reports realistic timing.
        let (messages, capture_fp) = if i == 0 {
            (probe_messages(), true)
        } else {
            (probe_messages_for(baseline_timing_nonce(i)), false)
        };
        match measure_baseline(req, messages, capture_fp).await {
            Ok(m) => {
                // Re-check: if a request started during the sample window the
                // tail of the decode was contended — discard it.
                if node.inflight_requests() > 0 {
                    skipped_busy = true;
                } else {
                    samples.push(m);
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "senda::native_baseline",
                    model = %req.model,
                    sample_index = i,
                    "native baseline sample failed: {e}"
                );
                last_err = Some(e);
            }
        }
    }
    if let Some(out) = median_measurement(samples, req.backend.clone()) {
        return Ok(out);
    }
    // No clean sample. Distinguish "node too busy" (keep prior, retry soon)
    // from "all samples errored" (genuine failure).
    if skipped_busy {
        Err(BaselineError::Busy)
    } else {
        Err(BaselineError::Failed(last_err.unwrap_or_else(|| {
            anyhow::anyhow!("all native baseline samples failed")
        })))
    }
}

/// Settle delay before issuing the first synthetic measurement. Lets
/// llama-server warm up its KV cache and any backend-side
/// pre-allocation, so the first baseline doesn't measure a cold start.
const SETTLE_DELAY_SECS: u64 = 30;

/// Spawn a background task that owns the native-baseline lifecycle for
/// one `(model, llama_port)` pair: it loads the cache, decides whether
/// to publish-cached vs measure-fresh, persists the result with the
/// model-file mtime as a cache key, calls `Node::record_native_baseline`
/// to trigger gossip, then sleeps until the next refresh tick.
///
/// Errors are logged via `tracing::warn` and never block the runtime —
/// a missing baseline is the right wire shape for "not yet measured",
/// not "broken". The catalog falls back to "—" for that column.
pub fn spawn_collector(
    node: crate::mesh::Node,
    model: String,
    http_port: u16,
    backend: String,
    model_file_path: Option<std::path::PathBuf>,
) {
    tracing::info!(
        target: "senda::native_baseline",
        model = %model,
        port = http_port,
        backend = %backend,
        "native baseline collector spawned"
    );
    tokio::spawn(async move {
        // Initial settle delay — some backends spin up Metal/CUDA buffers
        // lazily on the first request and we don't want our baseline to
        // include that one-time cost.
        tokio::time::sleep(Duration::from_secs(SETTLE_DELAY_SECS)).await;

        loop {
            let cache_file = cache_path();
            let live_mtime = model_file_path.as_deref().and_then(model_file_mtime_secs);

            // Step 1: try the cache. If a fresh entry exists, publish it
            // to the Node (which gossips it) and skip the measurement.
            if let Some(ref cp) = cache_file {
                let cache = load_cache(cp);
                if let Some(entry) = cache.entries.get(&model) {
                    if is_fresh(entry, live_mtime) {
                        node.record_native_baseline(NativeBaselineEntry::from(entry))
                            .await;
                        tokio::time::sleep(Duration::from_secs(CACHE_TTL_SECS / 2)).await;
                        continue;
                    }
                }
            }

            // Step 2: cache miss or stale → run a fresh synthetic
            // measurement against the local llama-server. Take
            // `SAMPLES_PER_REFRESH` shots and publish the median; see
            // the module docstring for why single-shot was insufficient.
            let req = BaselineRequest {
                model: model.clone(),
                http_port,
                backend: backend.clone(),
            };
            match measure_baseline_median(&node, &req, SAMPLES_PER_REFRESH).await {
                Ok((meas, sample_count)) => {
                    let entry = NativeBaselineEntry {
                        model: model.clone(),
                        native_tps_p50: meas.native_tps_p50,
                        native_ttft_ms_p50: meas.native_ttft_ms_p50,
                        measured_at_unix_secs: meas.measured_at_unix_secs,
                        samples: sample_count,
                        backend: meas.backend.clone(),
                        logit_fingerprint: meas.logit_fingerprint.clone(),
                    };
                    if let Some(ref cp) = cache_file {
                        let mut cache = load_cache(cp);
                        cache.entries.insert(
                            model.clone(),
                            CachedNativeBaseline {
                                model: entry.model.clone(),
                                native_tps_p50: entry.native_tps_p50,
                                native_ttft_ms_p50: entry.native_ttft_ms_p50,
                                measured_at_unix_secs: entry.measured_at_unix_secs,
                                samples: entry.samples,
                                backend: entry.backend.clone(),
                                model_file_mtime_secs: live_mtime,
                                logit_fingerprint: meas.logit_fingerprint.clone(),
                                runtime_version: Some(RUNTIME_VERSION.to_string()),
                            },
                        );
                        if let Err(err) = save_cache(cp, &cache) {
                            tracing::warn!(
                                target: "senda::native_baseline",
                                model = %model,
                                "failed to persist native-baselines cache: {err}"
                            );
                        }
                    }
                    tracing::info!(
                        target: "senda::native_baseline",
                        model = %model,
                        native_tps_p50 = meas.native_tps_p50,
                        native_ttft_ms_p50 = meas.native_ttft_ms_p50,
                        samples = sample_count,
                        backend = %backend,
                        "recorded native baseline"
                    );
                    node.record_native_baseline(entry).await;
                }
                Err(BaselineError::Busy) => {
                    // Node was serving real traffic for every sample window.
                    // Keep the prior clean baseline and retry on a short cadence
                    // so we grab the next idle gap rather than waiting TTL/2.
                    tracing::debug!(
                        target: "senda::native_baseline",
                        model = %model,
                        port = http_port,
                        "native baseline deferred — node busy, retrying soon"
                    );
                    tokio::time::sleep(BUSY_RETRY_DELAY).await;
                    continue;
                }
                Err(BaselineError::Failed(err)) => {
                    tracing::warn!(
                        target: "senda::native_baseline",
                        model = %model,
                        port = http_port,
                        "native baseline measurement failed: {err}"
                    );
                    // Don't blank a previously-cached entry on failure —
                    // it's better to keep gossiping the last known
                    // baseline than to flap to empty and back.
                }
            }

            // Re-check every TTL/2 so an mtime change is picked up
            // promptly without us hammering the model on every loop.
            tokio::time::sleep(Duration::from_secs(CACHE_TTL_SECS / 2)).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn measure_baseline_median_defers_when_node_busy() {
        // A real request in flight must prevent a (contended) sample from being
        // taken or published — that contention is exactly what made native_tps
        // read 5x low and untrustworthy for the perf-profile verifier. With the
        // node busy, the port is never even contacted; we must get `Busy` so the
        // collector keeps the prior clean value and retries soon.
        let node = crate::mesh::tests::make_test_node(crate::mesh::NodeRole::Host { http_port: 0 })
            .await
            .unwrap();
        let _guard = node.begin_inflight_request();
        assert!(node.inflight_requests() > 0);

        let req = BaselineRequest {
            model: "test-model".to_string(),
            http_port: 0, // never contacted — the busy check short-circuits first
            backend: "cpu".to_string(),
        };
        let result = measure_baseline_median(&node, &req, 1).await;
        assert!(
            matches!(result, Err(BaselineError::Busy)),
            "busy node must defer, not measure a contended sample"
        );
    }

    #[tokio::test]
    async fn measure_baseline_median_reports_failed_when_idle_but_unreachable() {
        // Idle node but a dead llama-server port: this is a genuine failure, not
        // a busy-defer. The collector should warn and keep the prior value via
        // the TTL/2 path, not the short busy-retry path.
        let node = crate::mesh::tests::make_test_node(crate::mesh::NodeRole::Host { http_port: 0 })
            .await
            .unwrap();
        assert_eq!(node.inflight_requests(), 0);

        let req = BaselineRequest {
            model: "test-model".to_string(),
            http_port: 1, // nothing listening → connection refused
            backend: "cpu".to_string(),
        };
        let result = measure_baseline_median(&node, &req, 1).await;
        assert!(
            matches!(result, Err(BaselineError::Failed(_))),
            "idle + unreachable must surface as a genuine failure, not Busy"
        );
    }

    #[test]
    fn parse_usage_from_sse_tail_extracts_completion_tokens() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                    data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"completion_tokens\":42,\"prompt_tokens\":7}}\n\
                    data: [DONE]\n";
        assert_eq!(
            parse_completion_tokens_from_sse_tail(body.as_bytes()),
            Some(42)
        );
    }

    #[test]
    fn parse_usage_returns_none_when_absent() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                    data: [DONE]\n";
        assert_eq!(parse_completion_tokens_from_sse_tail(body.as_bytes()), None);
    }

    #[test]
    fn cache_load_save_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("native-baselines.json");

        let mut cache = NativeBaselineCache::default();
        cache.entries.insert(
            "Qwen3-8B-Q4_K_M".to_string(),
            CachedNativeBaseline {
                model: "Qwen3-8B-Q4_K_M".to_string(),
                native_tps_p50: 28.4,
                native_ttft_ms_p50: 320,
                measured_at_unix_secs: 1_747_700_000,
                samples: 1,
                backend: "metal".to_string(),
                model_file_mtime_secs: Some(1_747_640_000),
                logit_fingerprint: None,
                runtime_version: Some(RUNTIME_VERSION.to_string()),
            },
        );

        save_cache(&path, &cache).unwrap();
        let loaded = load_cache(&path);
        let entry = loaded.entries.get("Qwen3-8B-Q4_K_M").unwrap();
        assert!((entry.native_tps_p50 - 28.4).abs() < 1e-9);
        assert_eq!(entry.native_ttft_ms_p50, 320);
        assert_eq!(entry.samples, 1);
        assert_eq!(entry.backend, "metal");
        assert_eq!(entry.model_file_mtime_secs, Some(1_747_640_000));
    }

    #[test]
    fn is_fresh_invalidates_on_mtime_change() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let entry = CachedNativeBaseline {
            model: "x".to_string(),
            native_tps_p50: 10.0,
            native_ttft_ms_p50: 100,
            measured_at_unix_secs: now,
            samples: 1,
            backend: "metal".to_string(),
            model_file_mtime_secs: Some(1_000_000),
            logit_fingerprint: None,
            runtime_version: Some(RUNTIME_VERSION.to_string()),
        };
        // Same mtime → fresh.
        assert!(is_fresh(&entry, Some(1_000_000)));
        // Different mtime → not fresh, force re-run.
        assert!(!is_fresh(&entry, Some(2_000_000)));
    }

    #[test]
    fn is_fresh_invalidates_on_runtime_version_change() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let base = CachedNativeBaseline {
            model: "x".to_string(),
            native_tps_p50: 10.0,
            native_ttft_ms_p50: 100,
            measured_at_unix_secs: now,
            samples: 1,
            backend: "metal".to_string(),
            model_file_mtime_secs: Some(1_000_000),
            logit_fingerprint: None,
            runtime_version: Some(RUNTIME_VERSION.to_string()),
        };
        // Current version + same mtime → fresh.
        assert!(is_fresh(&base, Some(1_000_000)));
        // Different runtime version → stale, even with matching mtime/TTL.
        let stale_version = CachedNativeBaseline {
            runtime_version: Some("0.0.0-old".to_string()),
            ..base.clone()
        };
        assert!(!is_fresh(&stale_version, Some(1_000_000)));
        // Pre-stamp cache (None) → stale.
        let unstamped = CachedNativeBaseline {
            runtime_version: None,
            ..base
        };
        assert!(!is_fresh(&unstamped, Some(1_000_000)));
    }

    fn meas(tps: f64, ttft: u64) -> BaselineMeasurement {
        BaselineMeasurement {
            native_tps_p50: tps,
            native_ttft_ms_p50: ttft,
            backend: "metal".to_string(),
            measured_at_unix_secs: 0,
            logit_fingerprint: None,
        }
    }

    #[test]
    fn median_picks_middle_of_three_per_axis() {
        // Real-world example from the v0.66.51 sweep: same hardware,
        // same config, three back-to-back runs returned 12.34, 14.30,
        // 17.58 t/s. Median is the middle value, not the mean.
        let samples = vec![meas(14.30, 1159), meas(12.34, 1220), meas(17.58, 1014)];
        let (m, n) = median_measurement(samples, "metal".to_string()).unwrap();
        assert_eq!(n, 3);
        assert!((m.native_tps_p50 - 14.30).abs() < 1e-9);
        assert_eq!(m.native_ttft_ms_p50, 1159);
    }

    #[test]
    fn median_is_resilient_to_partial_sample_failures() {
        // 1 of 3 succeeded — still publish that one rather than blank
        // the cache. (The collector logs the failures separately.)
        let only_one = vec![meas(14.41, 734)];
        let (m, n) = median_measurement(only_one, "metal".to_string()).unwrap();
        assert_eq!(n, 1);
        assert_eq!(m.native_ttft_ms_p50, 734);

        // Empty input → None; the collector translates that to keeping
        // the previously-cached entry, not gossiping nothing.
        assert!(median_measurement(vec![], "metal".to_string()).is_none());
    }

    #[test]
    fn is_fresh_invalidates_past_ttl() {
        let entry = CachedNativeBaseline {
            model: "x".to_string(),
            native_tps_p50: 10.0,
            native_ttft_ms_p50: 100,
            measured_at_unix_secs: 1, // ancient
            samples: 1,
            backend: "metal".to_string(),
            model_file_mtime_secs: Some(1_000_000),
            logit_fingerprint: None,
            runtime_version: Some(RUNTIME_VERSION.to_string()),
        };
        assert!(!is_fresh(&entry, Some(1_000_000)));
    }

    #[test]
    fn fingerprint_is_stable_and_separates_different_output() {
        let toks_a = vec!["Hello".to_string(), " world".to_string()];
        let a1 = build_fingerprint(&toks_a, &[], "Hello world");
        let a2 = build_fingerprint(&toks_a, &[], "Hello world");
        // Deterministic: same input → identical fingerprint.
        assert_eq!(a1, a2);
        assert_eq!(a1.token_count, 2);
        assert_eq!(a1.prefix_tokens[0], "Hello");
        assert_eq!(a1.prefix_tokens[1], " world");

        // A wrong/smaller model returning different text → different hash.
        let b = build_fingerprint(&toks_a, &[], "Hola mundo");
        assert_ne!(a1.output_sha256, b.output_sha256);
    }

    #[test]
    fn randomized_probe_is_deterministic_per_nonce_and_varies_across_nonces() {
        // Same nonce → byte-identical probe (so the self-oracle and the
        // suspect issue the same request and their fingerprints are
        // comparable).
        assert_eq!(probe_messages_for(7), probe_messages_for(7));
        // Different nonces → different user content (unpredictable across
        // audits; a peer can't precompute or recognize the probe).
        let a = probe_messages_for(7);
        let b = probe_messages_for(8);
        assert_ne!(a, b);
        // The nonce itself appears in the prompt, so even same-topic probes
        // (nonce + TOPICS.len()) differ.
        let same_topic = probe_messages_for(7 + PROBE_TOPICS.len() as u64);
        assert_ne!(a, same_topic);
    }

    /// The v0.66.57 baseline-honesty fix: a refresh sweep must send a
    /// *distinct* prompt for every sample (sample 0 = fixed reference probe,
    /// samples 1.. = unique cache-busting probes), otherwise llama-server's
    /// prompt-KV cache replays an identical prompt and the published TTFT is
    /// a ~20x-optimistic cache hit instead of a real prefill.
    #[test]
    fn baseline_refresh_sends_distinct_probe_per_sample() {
        fn user_text(msgs: &[serde_json::Value]) -> String {
            msgs.iter()
                .find(|m| m["role"] == "user")
                .and_then(|m| m["content"].as_str())
                .unwrap_or_default()
                .to_string()
        }
        let mut texts = std::collections::HashSet::new();
        // Sample 0: fixed deterministic probe.
        assert!(texts.insert(user_text(&probe_messages())));
        // Samples 1..: unique probes keyed by the per-sample timing nonce.
        for i in 1..SAMPLES_PER_REFRESH {
            let probe = probe_messages_for(baseline_timing_nonce(i));
            assert!(
                texts.insert(user_text(&probe)),
                "sample {i} probe collided with an earlier probe in the sweep"
            );
        }
        assert_eq!(texts.len(), SAMPLES_PER_REFRESH as usize);
    }

    #[test]
    fn fingerprint_prefix_is_bounded() {
        let toks: Vec<String> = (0..100).map(|i| format!("t{i}")).collect();
        let top_k: Vec<Vec<String>> = (0..100).map(|i| vec![format!("t{i}")]).collect();
        let fp = build_fingerprint(&toks, &top_k, "irrelevant");
        // Full count recorded, prefix + top-k both capped at FINGERPRINT_PREFIX_LEN.
        assert_eq!(fp.token_count, 100);
        assert_eq!(fp.prefix_tokens.len(), FINGERPRINT_PREFIX_LEN);
        assert_eq!(fp.top_k_tokens.len(), FINGERPRINT_PREFIX_LEN);
    }

    #[test]
    fn parse_sse_recovers_text_and_tokens() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"logprobs\":{\"content\":[{\"token\":\"Hi\",\"logprob\":-0.2}]}}]}\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\" there\"},\"logprobs\":{\"content\":[{\"token\":\" there\",\"logprob\":-1.5}]}}]}\n\
                    data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"completion_tokens\":2}}\n\
                    data: [DONE]\n";
        let (text, toks) = parse_output_and_tokens_from_sse(body.as_bytes());
        assert_eq!(text, "Hi there");
        assert_eq!(toks, vec!["Hi".to_string(), " there".to_string()]);
    }

    #[test]
    fn parse_sse_recovers_text_when_logprobs_absent() {
        // A backend that ignores `logprobs` still yields a usable output
        // hash from the deltas; the prefix is just empty.
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\
                    data: [DONE]\n";
        let (text, toks) = parse_output_and_tokens_from_sse(body.as_bytes());
        assert_eq!(text, "hello");
        assert!(toks.is_empty());
        let fp = build_fingerprint(&toks, &[], &text);
        assert_eq!(fp.token_count, 0);
        assert!(fp.prefix_tokens.is_empty());
        assert!(fp.top_k_tokens.is_empty());
        assert!(!fp.output_sha256.is_empty());
    }

    #[test]
    fn cache_roundtrips_logit_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("native-baselines.json");
        let mut cache = NativeBaselineCache::default();
        cache.entries.insert(
            "m".to_string(),
            CachedNativeBaseline {
                model: "m".to_string(),
                native_tps_p50: 10.0,
                native_ttft_ms_p50: 100,
                measured_at_unix_secs: 1,
                samples: 1,
                backend: "metal".to_string(),
                model_file_mtime_secs: None,
                logit_fingerprint: Some(LogitFingerprint {
                    token_count: 1,
                    output_sha256: "abc".to_string(),
                    prefix_tokens: vec!["hi".to_string()],
                    top_k_tokens: vec![vec!["hi".to_string(), "hey".to_string()]],
                }),
                runtime_version: Some(RUNTIME_VERSION.to_string()),
            },
        );
        save_cache(&path, &cache).unwrap();
        let loaded = load_cache(&path);
        let fp = loaded
            .entries
            .get("m")
            .unwrap()
            .logit_fingerprint
            .clone()
            .unwrap();
        assert_eq!(fp.token_count, 1);
        assert_eq!(fp.output_sha256, "abc");
        assert_eq!(fp.prefix_tokens, vec!["hi".to_string()]);
        assert_eq!(
            fp.top_k_tokens,
            vec![vec!["hi".to_string(), "hey".to_string()]]
        );
    }

    #[test]
    fn fingerprint_from_completion_json_matches_sse_path() {
        // The non-streaming verifier path must produce the same fingerprint
        // as the streaming baseline path for the same tokens + text.
        let json = serde_json::json!({
            "choices": [{
                "message": {"content": "Hi there"},
                "logprobs": {"content": [
                    {"token": "Hi", "logprob": -0.2, "top_logprobs": [
                        {"token": "Hi", "logprob": -0.2},
                        {"token": "Hey", "logprob": -1.1}
                    ]},
                    {"token": " there", "logprob": -1.5, "top_logprobs": [
                        {"token": " there", "logprob": -1.5},
                        {"token": " world", "logprob": -2.0}
                    ]}
                ]}
            }]
        });
        let from_json = fingerprint_from_completion_json(&json).unwrap();
        // Per-position top-k is captured from `top_logprobs[].token`.
        assert_eq!(
            from_json.top_k_tokens,
            vec![
                vec!["Hi".to_string(), "Hey".to_string()],
                vec![" there".to_string(), " world".to_string()],
            ]
        );

        let toks = vec!["Hi".to_string(), " there".to_string()];
        let top_k = vec![
            vec!["Hi".to_string(), "Hey".to_string()],
            vec![" there".to_string(), " world".to_string()],
        ];
        let from_stream = build_fingerprint(&toks, &top_k, "Hi there");
        assert_eq!(from_json, from_stream);
    }

    #[test]
    fn fingerprint_from_completion_json_none_when_empty() {
        let json = serde_json::json!({"choices": [{"message": {"content": ""}}]});
        assert!(fingerprint_from_completion_json(&json).is_none());
    }

    #[test]
    fn legacy_cache_without_fingerprint_deserializes() {
        // Caches written before the field existed must still load.
        let raw = r#"{"version":1,"entries":{"m":{"model":"m","native_tps_p50":10.0,"native_ttft_ms_p50":100,"measured_at_unix_secs":1,"samples":1,"backend":"metal","model_file_mtime_secs":null}}}"#;
        let cache: NativeBaselineCache = serde_json::from_str(raw).unwrap();
        let entry = cache.entries.get("m").unwrap();
        assert!(entry.logit_fingerprint.is_none());
    }
}
