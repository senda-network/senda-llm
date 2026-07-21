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
//! Policy (2026-07-21, Qwen TTFT dig on 0xSenda):
//!   - Warm path is chat-viable (~1 s TTFT). Cold / residency drops are 10–17 s.
//!   - While this node is *serving* a local model we keep pinging by default so
//!     the public daily-driver surface stays warm. Battery / thermal-sensitive
//!     contributors can opt out with `SENDA_KEEPWARM_WHILE_SERVING=0`, which
//!     falls back to the activity window only.
//!   - After the last real request we still keep the windowed pings for a while
//!     so a short pause mid-session does not re-cold the GPU.

use crate::mesh;
use std::time::Duration;

/// Ping cadence. Under Metal's ~180 s residency `keep_alive` with margin.
/// Was 90 s; 60 s leaves ~3 pings per residency window.
const KEEPWARM_INTERVAL_SECS: u64 = 60;

/// After the last locally-served *real* request, keep windowed pings for this
/// long even if `SENDA_KEEPWARM_WHILE_SERVING=0`. Was 15 min; 60 min covers a
/// typical chat session without pinning forever.
const KEEPWARM_WINDOW_SECS: u64 = 60 * 60;

/// Per-ping timeout. The ping itself can be slow when it lands right as
/// residency was about to lapse; that cost is paid by the ping, not a user.
const PING_TIMEOUT: Duration = Duration::from_secs(20);

/// Default: keep warm whenever a model is loaded locally (serving).
/// Set `SENDA_KEEPWARM_WHILE_SERVING=0` to only ping inside the activity window
/// (better for battery laptops that rarely take chat traffic).
fn keepwarm_while_serving_enabled() -> bool {
    match std::env::var("SENDA_KEEPWARM_WHILE_SERVING") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        }
        Err(_) => true,
    }
}

/// Spawn the background keep-warm loop. Safe to run on any node: it is a no-op
/// unless this node is locally serving at least one model and (by default)
/// keeps that residency hot, or has served a real request recently.
pub fn spawn_keepwarm(node: mesh::Node) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(KEEPWARM_INTERVAL_SECS));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;

            let ports = node.local_model_ports_snapshot().await;
            if ports.is_empty() {
                continue;
            }

            let in_activity_window = matches!(
                node.seconds_since_last_local_request(),
                Some(secs) if secs < KEEPWARM_WINDOW_SECS
            );
            let while_serving = keepwarm_while_serving_enabled();
            if !in_activity_window && !while_serving {
                continue;
            }

            // A real request in flight already holds residency; don't pile on.
            if node.inflight_requests() > 0 {
                continue;
            }

            for (model, port) in ports {
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

#[cfg(test)]
mod tests {
    #[test]
    fn while_serving_defaults_on() {
        fn parse(v: Option<&str>) -> bool {
            match v {
                Some(v) => {
                    let v = v.trim().to_ascii_lowercase();
                    !(v == "0" || v == "false" || v == "off" || v == "no")
                }
                None => true,
            }
        }
        assert!(parse(None));
        assert!(parse(Some("1")));
        assert!(parse(Some("true")));
        assert!(!parse(Some("0")));
        assert!(!parse(Some("OFF")));
        assert!(!parse(Some("false")));
    }
}
