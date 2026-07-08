//! Keep-warm loop: holds the local llama-server's GPU residency hot during and
//! shortly after active use, so a follow-up request doesn't pay the cold-start
//! tax.
//!
//! Background (measured 2026-06-06, M3 Pro): when a model sits idle, the Metal
//! residency set releases its GPU residency after ~180 s (a hardcoded
//! `keep_alive` in the bundled ggml build — no flag/env to extend it). The next
//! request then re-pages the weights, spiking TTFT ~10x (~2,050 ms vs ~196 ms
//! warm). CUDA shows a smaller version of the same (clock/context ramp). Any
//! inference resets the timer, so a periodic 1-token self-ping holds it warm.
//!
//! This is deliberately *windowed*: we only ping while the node has served a
//! real request within `KEEPWARM_WINDOW_SECS`. That kills cold-starts during
//! and around active sessions while letting a truly-idle contributor GPU sleep
//! — important for laptop contributors on battery, who should not have their
//! GPU pinned awake 24/7 to shave 2 s off an occasional first request. The very
//! first request after a long idle still pays one cold-start; that is the
//! intended trade.

use crate::mesh;
use std::time::Duration;

/// Ping cadence. Half the Metal residency `keep_alive` (180 s) for ~2x margin.
const KEEPWARM_INTERVAL_SECS: u64 = 90;

/// Keep warm for this long after the last locally-served request, then stop.
const KEEPWARM_WINDOW_SECS: u64 = 15 * 60;

/// Per-ping timeout. The ping itself can be slow when it lands right as
/// residency was about to lapse; that cost is paid by the ping, not a user.
const PING_TIMEOUT: Duration = Duration::from_secs(20);

/// Spawn the background keep-warm loop. Safe to run on any node: it is a no-op
/// unless this node is locally serving at least one model and has served a real
/// request recently.
pub fn spawn_keepwarm(node: mesh::Node) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(KEEPWARM_INTERVAL_SECS));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;

            // Only within the activity window — otherwise let the GPU sleep.
            match node.seconds_since_last_local_request() {
                Some(secs) if secs < KEEPWARM_WINDOW_SECS => {}
                _ => continue,
            }

            // A real request in flight already holds residency; don't pile on.
            if node.inflight_requests() > 0 {
                continue;
            }

            for (model, port) in node.local_model_ports_snapshot().await {
                if let Err(e) = ping(port).await {
                    tracing::debug!(
                        target: "senda::keepwarm",
                        %model,
                        port,
                        "keep-warm ping failed: {e}"
                    );
                }
            }
        }
    });
}

/// Fire a minimal 1-token completion at the local llama-server. This touches
/// the model's forward pass on the GPU, which is all that's needed to reset the
/// residency timer. It hits the llama-server port directly (not the mesh
/// ingress path), so it does not count as a "real request" and never extends
/// the keep-warm window.
async fn ping(http_port: u16) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().timeout(PING_TIMEOUT).build()?;
    let url = format!("http://127.0.0.1:{http_port}/completion");
    let body = serde_json::json!({
        "prompt": "ok",
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": false,
    });
    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("keep-warm ping returned {}", resp.status());
    }
    Ok(())
}
