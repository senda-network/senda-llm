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
//!   `~/.closedmesh/native-baselines.json`. The mtime is part of the cache
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
//! - The synthetic prompt asks for ~80–128 output tokens with `temperature=0`
//!   and a fixed seed, so the same model produces the same shape of
//!   output across runs. This is not an apples-to-apples comparison
//!   with through-mesh traffic (which has variable prompt + output
//!   shapes); it is a *baseline* the through-mesh number can be
//!   referenced against.
//! - Inter-sample delay is 1 s so consecutive runs don't race the same
//!   prompt cache state, but small enough that the whole 3-sample
//!   sweep finishes in ~30 s on Apple Silicon. Failed samples in a
//!   sweep are dropped silently; we still publish a median across the
//!   surviving subset (down to 1) rather than blank the cache.

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
/// Pause between consecutive samples in a single refresh sweep. Long
/// enough that prompt-cache state from the previous run isn't reused
/// (llama-server's prompt cache TTL is much higher, but the delay also
/// gives the GPU command queue a moment to drain), short enough that
/// 3 samples still finish in ~30 s on Apple Silicon.
const INTER_SAMPLE_DELAY: Duration = Duration::from_secs(1);
/// How many decoded tokens the fingerprint covers. The greedy decode of a
/// fixed prompt is a strong model-identity signal; a bounded prefix lets a
/// verifier compare with tolerance instead of demanding an exact
/// cross-backend match.
const FINGERPRINT_PREFIX_LEN: usize = 32;

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
    /// Deterministic logit fingerprint from the same probe. `None` for
    /// caches written before this field existed (serde default) or when
    /// the probe produced no output. Captured + cached only today — not
    /// yet gossiped or used for enforcement.
    #[serde(default)]
    pub logit_fingerprint: Option<LogitFingerprint>,
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
/// This struct is captured + cached + gossiped, but drives no enforcement
/// yet — pure instrumentation, same risk profile as the timing baseline.
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
}

/// Build a [`LogitFingerprint`] from the decoded token stream and the
/// concatenated output text. Pure + deterministic so it is unit-testable
/// without an llama-server.
pub(crate) fn build_fingerprint(tokens: &[String], full_text: &str) -> LogitFingerprint {
    let mut hasher = Sha256::new();
    hasher.update(full_text.as_bytes());
    let output_sha256 = hex::encode(hasher.finalize());
    let prefix_tokens = tokens.iter().take(FINGERPRINT_PREFIX_LEN).cloned().collect();
    LogitFingerprint {
        token_count: tokens.len() as u32,
        output_sha256,
        prefix_tokens,
    }
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
    let mut tokens: Vec<String> = Vec::new();
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
    Some(build_fingerprint(&tokens, &text))
}

/// Capture an auditor *reference* fingerprint by issuing the deterministic
/// probe to a local llama-server (non-streaming) and building the fingerprint
/// via the exact same code path the remote verifier uses
/// ([`fingerprint_from_completion_json`]) — so a reference and a candidate are
/// guaranteed to be constructed identically. Used by the
/// `benchmark capture-reference` subcommand.
pub(crate) async fn capture_reference_fingerprint(
    http_port: u16,
    model: &str,
) -> anyhow::Result<LogitFingerprint> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| anyhow::anyhow!("build reqwest client: {e}"))?;
    let url = format!("http://127.0.0.1:{http_port}/v1/chat/completions");
    let body = serde_json::json!({
        "model": model,
        "messages": probe_messages(),
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": PROBE_SEED,
        "stream": false,
        "logprobs": true,
    });
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("reference probe request failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("reference probe llama-server returned {status}");
    }
    let parsed: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("parse reference probe response: {e}"))?;
    fingerprint_from_completion_json(&parsed)
        .ok_or_else(|| anyhow::anyhow!("reference probe response had no usable output"))
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

/// Returns the path of the native-baseline cache file (`~/.closedmesh/
/// native-baselines.json`). Honors `CLOSEDMESH_HOME` / `HOME` the same
/// way the rest of the runtime does.
pub fn cache_path() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("CLOSEDMESH_HOME") {
        return Some(PathBuf::from(custom).join(CACHE_FILENAME));
    }
    let home = dirs::home_dir()?;
    Some(home.join(".closedmesh").join(CACHE_FILENAME))
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
/// Errors when the request times out, returns a non-200, or doesn't
/// produce any output tokens. Callers log + retain the previous cached
/// baseline rather than blanking the gossiped value.
pub async fn measure_baseline(req: &BaselineRequest) -> anyhow::Result<BaselineMeasurement> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| anyhow::anyhow!("build reqwest client: {e}"))?;

    let url = format!("http://127.0.0.1:{}/v1/chat/completions", req.http_port);
    let body = serde_json::json!({
        "model": req.model,
        "messages": probe_messages(),
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": PROBE_SEED,
        "stream": true,
        "stream_options": {"include_usage": true},
        // `logprobs` gives us per-token segmentation for the fingerprint
        // prefix. Backends that don't return it just leave the prefix empty
        // (the output hash still fingerprints the model).
        "logprobs": true,
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

    // Build the deterministic fingerprint from the same buffered body.
    // No extra request; the probe already streamed it.
    let (output_text, tokens) = parse_output_and_tokens_from_sse(&buf);
    let logit_fingerprint = if output_text.is_empty() {
        None
    } else {
        Some(build_fingerprint(&tokens, &output_text))
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
    // The probe is deterministic (temp=0/seed=42), so every surviving
    // sample's fingerprint should be identical; take the first present.
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

/// Run [`measure_baseline`] up to `n` times with a 1 s gap between
/// samples and return the median TPS / TTFT across the runs that
/// succeeded. `Err` only when *every* sample failed; partial sample
/// sets (1 of 3, 2 of 3) still produce a usable measurement so a
/// flaky network blip doesn't blank a peer's published baseline.
async fn measure_baseline_median(
    req: &BaselineRequest,
    n: u32,
) -> anyhow::Result<(BaselineMeasurement, u32)> {
    let mut samples: Vec<BaselineMeasurement> = Vec::with_capacity(n as usize);
    let mut last_err: Option<anyhow::Error> = None;
    for i in 0..n {
        if i > 0 {
            tokio::time::sleep(INTER_SAMPLE_DELAY).await;
        }
        match measure_baseline(req).await {
            Ok(m) => samples.push(m),
            Err(e) => {
                tracing::warn!(
                    target: "closedmesh::native_baseline",
                    model = %req.model,
                    sample_index = i,
                    "native baseline sample failed: {e}"
                );
                last_err = Some(e);
            }
        }
    }
    median_measurement(samples, req.backend.clone()).ok_or_else(|| {
        last_err.unwrap_or_else(|| anyhow::anyhow!("all native baseline samples failed"))
    })
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
        target: "closedmesh::native_baseline",
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
            match measure_baseline_median(&req, SAMPLES_PER_REFRESH).await {
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
                        },
                        );
                        if let Err(err) = save_cache(cp, &cache) {
                            tracing::warn!(
                                target: "closedmesh::native_baseline",
                                model = %model,
                                "failed to persist native-baselines cache: {err}"
                            );
                        }
                    }
                    tracing::info!(
                        target: "closedmesh::native_baseline",
                        model = %model,
                        native_tps_p50 = meas.native_tps_p50,
                        native_ttft_ms_p50 = meas.native_ttft_ms_p50,
                        samples = sample_count,
                        backend = %backend,
                        "recorded native baseline"
                    );
                    node.record_native_baseline(entry).await;
                }
                Err(err) => {
                    tracing::warn!(
                        target: "closedmesh::native_baseline",
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
        };
        // Same mtime → fresh.
        assert!(is_fresh(&entry, Some(1_000_000)));
        // Different mtime → not fresh, force re-run.
        assert!(!is_fresh(&entry, Some(2_000_000)));
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
        };
        assert!(!is_fresh(&entry, Some(1_000_000)));
    }

    #[test]
    fn fingerprint_is_stable_and_separates_different_output() {
        let toks_a = vec!["Hello".to_string(), " world".to_string()];
        let a1 = build_fingerprint(&toks_a, "Hello world");
        let a2 = build_fingerprint(&toks_a, "Hello world");
        // Deterministic: same input → identical fingerprint.
        assert_eq!(a1, a2);
        assert_eq!(a1.token_count, 2);
        assert_eq!(a1.prefix_tokens[0], "Hello");
        assert_eq!(a1.prefix_tokens[1], " world");

        // A wrong/smaller model returning different text → different hash.
        let b = build_fingerprint(&toks_a, "Hola mundo");
        assert_ne!(a1.output_sha256, b.output_sha256);
    }

    #[test]
    fn fingerprint_prefix_is_bounded() {
        let toks: Vec<String> = (0..100).map(|i| format!("t{i}")).collect();
        let fp = build_fingerprint(&toks, "irrelevant");
        // Full count recorded, prefix capped at FINGERPRINT_PREFIX_LEN.
        assert_eq!(fp.token_count, 100);
        assert_eq!(fp.prefix_tokens.len(), FINGERPRINT_PREFIX_LEN);
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
        let fp = build_fingerprint(&toks, &text);
        assert_eq!(fp.token_count, 0);
        assert!(fp.prefix_tokens.is_empty());
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
                }),
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
    }

    #[test]
    fn fingerprint_from_completion_json_matches_sse_path() {
        // The non-streaming verifier path must produce the same fingerprint
        // as the streaming baseline path for the same tokens + text.
        let json = serde_json::json!({
            "choices": [{
                "message": {"content": "Hi there"},
                "logprobs": {"content": [
                    {"token": "Hi", "logprob": -0.2},
                    {"token": " there", "logprob": -1.5}
                ]}
            }]
        });
        let from_json = fingerprint_from_completion_json(&json).unwrap();

        let toks = vec!["Hi".to_string(), " there".to_string()];
        let from_stream = build_fingerprint(&toks, "Hi there");
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
