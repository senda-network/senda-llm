//! Keep-warm loop: holds the local llama-server's GPU residency hot so a
//! follow-up request after a pause doesn't pay the cold-start tax.
//!
//! Background (measured 2026-06-06, M3 Pro): when a model sits idle, the Metal
//! residency set releases its GPU residency after ~180 s (a hardcoded
//! `keep_alive` in the bundled ggml build — no flag/env to extend it). The next
//! request then re-pages the weights, spiking TTFT ~10x (~2,050 ms vs ~196 ms
//! warm). CUDA shows a smaller version of the same (clock/context ramp). Any
//! inference resets the timer, so a periodic 1-token self-ping holds it warm.
//!
//! Policy (2026-07-22, mid-burst TTFT dig on 0xSenda):
//!   - Warm local path is fine (~0.2–1 s content TTFT). Mid-session 5–12 s
//!     spikes correlated with keepwarm `/completion` pings overlapping chat
//!     on `--parallel 4`: Metal serializes and *both* slots stall
//!     (llama-server: ~9 s prompt-eval on a 1–6 token keepwarm *and* on the
//!     concurrent chat prefill).
//!   - Therefore: **never ping while real traffic is recent**. Real requests
//!     already reset residency. Only bridge the idle gap as we approach the
//!     ~180 s Metal cliff (and optionally keep bridging while this node is
//!     serving, for long quiet periods between public chats).
//!   - `SENDA_KEEPWARM_WHILE_SERVING=0` disables the long-quiet bridge; pings
//!     then only happen inside the post-activity window.

use crate::mesh;
use std::time::Duration;

/// How often we *consider* a ping. Actual pings are gated by idle age.
const KEEPWARM_INTERVAL_SECS: u64 = 30;

/// Skip keepwarm entirely if a real local request happened more recently
/// than this. Measured Metal keep_alive is ~180 s; staying well under that
/// means live traffic is already holding residency, and a ping would only
/// contend for the GPU (the mid-burst failure mode).
const KEEPWARM_MIN_IDLE_SECS: u64 = 90;

/// After the last locally-served *real* request, allow idle-gap pings for
/// this long. Covers a typical chat pause without pinning forever when
/// `SENDA_KEEPWARM_WHILE_SERVING=0`.
const KEEPWARM_WINDOW_SECS: u64 = 60 * 60;

/// Per-ping timeout. The ping itself can be slow when it lands right as
/// residency was about to lapse; that cost is paid by the ping, not a user.
const PING_TIMEOUT: Duration = Duration::from_secs(30);

/// Default: after the activity window, keep bridging quiet periods while a
/// model is loaded locally. Set `SENDA_KEEPWARM_WHILE_SERVING=0` to stop
/// pinging once the window expires (better for battery laptops).
fn keepwarm_while_serving_enabled() -> bool {
    match std::env::var("SENDA_KEEPWARM_WHILE_SERVING") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        }
        Err(_) => true,
    }
}

/// Decide whether a keepwarm ping should fire given idle age.
///
/// `idle_secs = None` means no real local request has been observed yet
/// (fresh process) — treat as "long idle" so while-serving can warm up.
fn should_keepwarm_ping(idle_secs: Option<u64>, while_serving: bool) -> bool {
    match idle_secs {
        // Live traffic is holding residency; a ping would only contend.
        Some(secs) if secs < KEEPWARM_MIN_IDLE_SECS => false,
        Some(secs) if secs < KEEPWARM_WINDOW_SECS => true,
        // Past the window: only bridge if we're still serving publicly.
        Some(_) | None => while_serving,
    }
}

/// Spawn the background keep-warm loop. Safe to run on any node: it is a no-op
/// unless this node is locally serving at least one model and the idle-gap
/// policy says a ping is useful.
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

            let idle = node.seconds_since_last_local_request();
            if !should_keepwarm_ping(idle, keepwarm_while_serving_enabled()) {
                continue;
            }

            // A real request in flight already holds residency; don't pile on.
            if node.inflight_requests() > 0 {
                continue;
            }

            for (model, port) in ports {
                // Re-check: a chat may have arrived while we were pinging
                // another model / between loop iterations.
                if node.inflight_requests() > 0 {
                    break;
                }
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
/// the keep-warm window / idle clock.
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
    use super::{should_keepwarm_ping, KEEPWARM_MIN_IDLE_SECS, KEEPWARM_WINDOW_SECS};

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

    #[test]
    fn skips_ping_while_traffic_is_recent() {
        assert!(!should_keepwarm_ping(Some(0), true));
        assert!(!should_keepwarm_ping(Some(2), true));
        assert!(!should_keepwarm_ping(
            Some(KEEPWARM_MIN_IDLE_SECS - 1),
            true
        ));
    }

    #[test]
    fn pings_in_idle_gap_before_metal_cliff() {
        assert!(should_keepwarm_ping(Some(KEEPWARM_MIN_IDLE_SECS), true));
        assert!(should_keepwarm_ping(Some(150), false));
        assert!(should_keepwarm_ping(
            Some(KEEPWARM_WINDOW_SECS - 1),
            false
        ));
    }

    #[test]
    fn long_quiet_only_when_while_serving() {
        assert!(should_keepwarm_ping(Some(KEEPWARM_WINDOW_SECS), true));
        assert!(!should_keepwarm_ping(Some(KEEPWARM_WINDOW_SECS), false));
        assert!(should_keepwarm_ping(None, true));
        assert!(!should_keepwarm_ping(None, false));
    }
}
