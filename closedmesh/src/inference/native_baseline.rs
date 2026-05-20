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
/// slow hardware.
const MAX_TOKENS: u32 = 128;
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
        "messages": [
            {"role": "system", "content": "You are a benchmark probe. Respond concisely."},
            {"role": "user",
             "content": "Write a short paragraph (about 100 words) explaining \
                         why direct measurement beats marketing claims when \
                         comparing distributed systems performance."}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 42,
        "stream": true,
        "stream_options": {"include_usage": true},
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

    Ok(BaselineMeasurement {
        native_tps_p50,
        native_ttft_ms_p50: ttft.as_millis().min(u64::MAX as u128) as u64,
        backend: req.backend.clone(),
        measured_at_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
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
    Some((
        BaselineMeasurement {
            native_tps_p50: tps[mid],
            native_ttft_ms_p50: ttft[mid],
            backend,
            measured_at_unix_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        },
        samples.len() as u32,
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
        };
        assert!(!is_fresh(&entry, Some(1_000_000)));
    }
}
