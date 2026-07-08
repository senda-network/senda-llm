//! Mesh-visibility audit and auto-heal.
//!
//! ## Why
//!
//! The desktop UI's big green "READY" pill is computed from the *local*
//! runtime's `node_state: "serving"` flag — that flag flips on the second
//! the runtime decides to commit to serving a model, with no regard for
//! whether anyone in the wider mesh can actually reach us. Symptomatically
//! this means the dashboard can confidently report "everything is fine"
//! while a peer is silently invisible to the entry node and therefore
//! invisible to every chat request routed through the public website.
//!
//! That mismatch is the cause of the May 2026 MSI-invisible-but-green
//! incident: MSI's local view showed it peered with LYU and the entry,
//! its `serving_models` was set, the pill was green; meanwhile
//! `https://entry.senda.network/api/status` returned exactly one peer
//! object (LYU) and zero requests ever landed on MSI.
//!
//! ## What this module does
//!
//! - **Audit (Slice 1).** Every 30 s, GET `<entry_url>/api/status` and
//!   check whether our own `node_id` appears in `entry.peers[].id` (or
//!   matches `entry.node_id` itself, for the entry-as-self case).
//!   Classify the outcome as one of [`MeshVisibilityState`] and write
//!   it to a shared [`MeshVisibilitySnapshot`] that `/api/status`
//!   surfaces to clients verbatim.
//!
//! - **Soft heal (Slice 2).** After [`SOFT_RECONNECT_THRESHOLD`]
//!   consecutive non-`Visible` probes the loop refetches a fresh join
//!   token from `<entry_url>/api/status` and calls `node.join(token)`
//!   in-process. Rate-limited by [`SOFT_RECONNECT_COOLDOWN`] so a
//!   genuinely-offline entry doesn't make us thrash.
//!
//! - **Hard reset (Slice 2).** After [`HARD_RESET_THRESHOLD`]
//!   consecutive non-`Visible` probes the loop calls
//!   [`std::process::exit`] with code 75 (`EX_TEMPFAIL`). The desktop
//!   app's launchd / scheduled-task / systemd supervisor restarts us
//!   immediately, which re-runs the whole join from scratch (fresh
//!   token from `--join-url`, fresh iroh dial). This is heavier than a
//!   soft reconnect — it tears down `llama-server` children and forces
//!   a model re-mmap — but if we've been invisible for ~4 minutes the
//!   system is already broken and a clean restart is the right move.
//!
//! ## Why not run this for every node
//!
//! The audit only makes sense when this runtime is a *member* of a mesh
//! anchored at a specific entry URL. The entry node itself has no parent
//! entry to probe, and a fully-local mesh has no upstream to verify
//! against. [`spawn_mesh_visibility_monitor`] is only invoked when
//! `--join-url` is set; everywhere else the snapshot stays at
//! [`MeshVisibilityState::Unknown`] and the dashboard renders accordingly.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::RwLock;

/// How often the audit loop polls the entry node. Chosen to be small
/// enough that an invisible peer is detected within a couple of status
/// page refreshes (the website polls every 8 s) but large enough that
/// we're not hammering the entry — the entry can already handle
/// hundreds of these per minute easily, but we want to keep room for
/// other status consumers.
const AUDIT_INTERVAL: Duration = Duration::from_secs(30);

/// Initial settle delay before the first probe. Gives our own iroh
/// `join()` and the entry's gossip propagation a chance to land — if we
/// probe at T+0 we sometimes catch the entry mid-accept and falsely
/// classify ourselves as Invisible.
const INITIAL_DELAY: Duration = Duration::from_secs(15);

/// Hard cap on a single probe. 8 s is generous compared to the typical
/// 100–400 ms entry round-trip yet short enough that a flaky connection
/// doesn't stall the audit cadence.
const PROBE_TIMEOUT: Duration = Duration::from_secs(8);

/// Number of consecutive non-`Visible` probes before the loop attempts
/// a soft reconnect. With [`AUDIT_INTERVAL`] = 30 s this is ~90 s of
/// confirmed invisibility — enough to ride out a one-tick blip on a
/// healthy mesh without thrashing.
const SOFT_RECONNECT_THRESHOLD: u32 = 3;

/// Number of consecutive non-`Visible` probes before the loop escalates
/// to a hard reset (process exit). ~4 minutes of invisibility.
const HARD_RESET_THRESHOLD: u32 = 8;

/// Minimum time between two soft reconnect attempts. Without a cap the
/// loop would call `node.join()` every 30 s for a genuinely-offline
/// entry, which is wasted work and pollutes logs.
const SOFT_RECONNECT_COOLDOWN: Duration = Duration::from_secs(60);

/// Minimum time between two hard reset attempts. Mostly a safety belt:
/// once we've `exit(75)`'d, the supervisor needs ~1–5 s to respawn us,
/// then we start fresh and the counter is back at 0, so the cooldown
/// only matters if `exit()` somehow returns without terminating.
const HARD_RESET_COOLDOWN: Duration = Duration::from_secs(5 * 60);

/// Outcome of one mesh-visibility probe against the configured entry
/// node.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshVisibilityState {
    /// No probe has run yet. Distinct from [`Self::EntryUnreachable`]
    /// so the dashboard can render "checking…" during the first 30 s
    /// after launch instead of a scary red banner.
    Unknown,
    /// Most recent probe found our `node_id` in the entry's peers list
    /// (or matched the entry's own `node_id`, for the entry-as-self
    /// case).
    Visible,
    /// Most recent probe reached the entry but our `node_id` was not
    /// in its peers list. This is the silent-broken-peer failure mode
    /// that motivated the whole module — iroh thinks it's connected,
    /// we're advertising models locally, but the entry has evicted us
    /// or never registered us.
    Invisible,
    /// Most recent probe could not reach the entry at all (network
    /// error, timeout, non-2xx response). Operationally distinct from
    /// [`Self::Invisible`] because a transient WAN blip needs a
    /// reconnect to *us*, not a reset of *our* mesh state.
    EntryUnreachable,
}

impl MeshVisibilityState {
    /// True for the two states that count as "we should escalate". Kept
    /// as a method (rather than inlined) so the test below can exercise
    /// the classification without duplicating the match arms.
    #[cfg(test)]
    fn counts_as_invisible(self) -> bool {
        matches!(self, Self::Invisible | Self::EntryUnreachable)
    }
}

/// Public read-side projection of the audit loop's state. Embedded
/// verbatim in `/api/status` so the desktop UI and the public status
/// page can render the truth.
#[derive(Clone, Debug, Serialize)]
pub struct MeshVisibilitySnapshot {
    pub state: MeshVisibilityState,
    /// Unix-seconds timestamp of the most recent probe attempt (any
    /// outcome). `None` until the first probe completes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_check_unix: Option<u64>,
    /// Unix-seconds timestamp of the most recent probe that returned
    /// [`MeshVisibilityState::Visible`]. Survives subsequent failures
    /// so the UI can show "last seen 3 min ago" while we're in a
    /// degraded state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_visible_unix: Option<u64>,
    /// Consecutive non-`Visible` probes since the last `Visible`. Reset
    /// to 0 on `Visible`. Drives the auto-heal escalation thresholds.
    pub consecutive_invisible_count: u32,
    /// Short human-readable reason for the most recent failure. Cleared
    /// on the next `Visible`. Surfaced verbatim in the dashboard so
    /// users can tell "entry returned 404" apart from "we just aren't
    /// in the peer list".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    /// URL of the entry node being probed. Surfaced so operators can
    /// audit which mesh they're being measured against without having
    /// to inspect the launchd plist by hand.
    pub entry_url: String,
    /// True if the auto-heal loop has issued at least one soft
    /// reconnect since the last `Visible`. Cleared on next `Visible`.
    pub soft_reconnect_triggered: bool,
    /// True if the auto-heal loop has decided the runtime needs a
    /// hard reset. Set ~1 s before [`std::process::exit`] so a fast
    /// status consumer can record the trigger before the process dies.
    pub hard_reset_triggered: bool,
}

impl MeshVisibilitySnapshot {
    fn new(entry_url: String) -> Self {
        Self {
            state: MeshVisibilityState::Unknown,
            last_check_unix: None,
            last_visible_unix: None,
            consecutive_invisible_count: 0,
            last_error: None,
            entry_url,
            soft_reconnect_triggered: false,
            hard_reset_triggered: false,
        }
    }
}

/// Shared handle into the visibility monitor's current state. Cheap to
/// clone (one `Arc`), read-locked from `/api/status`, write-locked only
/// from the single audit task that owns the loop.
#[derive(Clone)]
pub struct MeshVisibilityHandle {
    inner: Arc<RwLock<MeshVisibilitySnapshot>>,
}

impl MeshVisibilityHandle {
    /// Take a cheap clone of the current snapshot. Held read-lock is
    /// released before this returns, so concurrent audit-loop writes
    /// are never blocked by status consumers.
    pub async fn snapshot(&self) -> MeshVisibilitySnapshot {
        self.inner.read().await.clone()
    }
}

/// Configuration for the peer-report phone-home path (Slice 4).
///
/// Each peer POSTs its `MeshVisibilitySnapshot` to
/// `<report_url>/api/peer-report` after every audit cycle. The Vercel
/// app retains the report for a few minutes and surfaces
/// claimed-but-invisible peers on the public status page — the only
/// way the operator-side view can know about a peer that has silently
/// dropped its iroh connection (the very state we cannot learn from
/// the entry's gossip view).
///
/// When `None`, the audit loop runs without phoning home — useful for
/// the entry itself (which doesn't need to report about itself), and
/// for headless/dev installations that don't want the network egress.
#[derive(Clone)]
pub struct PeerReportConfig {
    /// Base URL of the website receiving reports (e.g.
    /// `https://senda.network`). `/api/peer-report` is appended.
    pub url: String,
    /// Node hostname surfaced in the report — used by the public UI to
    /// label the claimed-but-invisible row without needing to look up
    /// the iroh `EndpointId`.
    pub hostname: Option<String>,
    /// Runtime version. Same purpose as `hostname`.
    pub version: Option<String>,
}

/// Spawn the visibility audit loop and return a handle into its shared
/// state plus the task `JoinHandle`.
///
/// `entry_url` is the base URL of the entry node (e.g.
/// `https://entry.senda.network`) — typically the same URL the runtime
/// was started with via `--join-url`. The `/api/status` suffix is
/// appended internally.
///
/// `fallback_tokens` is used by the soft-reconnect path when refetching
/// a fresh token from `entry_url` fails (captive portal, DNS hiccup,
/// entry currently rebooting). Pass the startup `--join` tokens that
/// were already resolved by `resolve_join_url`.
///
/// `peer_report` enables Slice 4's phone-home channel. Pass `None`
/// for installs that should audit but not phone home.
pub fn spawn_mesh_visibility_monitor(
    node: crate::mesh::Node,
    entry_url: String,
    fallback_tokens: Vec<String>,
    peer_report: Option<PeerReportConfig>,
) -> (MeshVisibilityHandle, tokio::task::JoinHandle<()>) {
    let snapshot = Arc::new(RwLock::new(MeshVisibilitySnapshot::new(entry_url.clone())));
    let handle = MeshVisibilityHandle {
        inner: snapshot.clone(),
    };

    let task = tokio::spawn(async move {
        run_audit_loop(node, entry_url, fallback_tokens, snapshot, peer_report).await;
    });

    (handle, task)
}

#[derive(Debug)]
struct ProbeOutcome {
    state: MeshVisibilityState,
    error: Option<String>,
    /// If the probe succeeded, the entry's currently-advertised invite
    /// token. The soft-reconnect path prefers this over the startup
    /// fallback tokens because the entry rotates its iroh identity on
    /// every container restart, so the fallback can be stale.
    fresh_token: Option<String>,
}

// `last_hard_reset` is written immediately before `std::process::exit`,
// which is `!`-typed and so the compiler flags the assignment as dead.
// Suppress at the function level rather than the binding because the
// binding-level attribute does not propagate to assignment sites in
// stable Rust. We want the explicit `last_hard_reset = …` so the
// cooldown invariant is visible to a reader and would still apply
// hypothetically if `exit` ever didn't terminate (PID 1 namespace
// edge cases, mock harnesses).
#[allow(unused_assignments)]
async fn run_audit_loop(
    node: crate::mesh::Node,
    entry_url: String,
    fallback_tokens: Vec<String>,
    snapshot: Arc<RwLock<MeshVisibilitySnapshot>>,
    peer_report: Option<PeerReportConfig>,
) {
    tokio::time::sleep(INITIAL_DELAY).await;

    let mut last_soft_reconnect: Option<Instant> = None;
    let mut last_hard_reset: Option<Instant> = None;
    let mut cached_fresh_token: Option<String> = None;

    loop {
        let outcome = probe_once(&node, &entry_url).await;

        if let Some(token) = outcome.fresh_token.clone() {
            cached_fresh_token = Some(token);
        }

        let consecutive_invisible = update_snapshot(&snapshot, &outcome).await;

        if let Some(report_cfg) = peer_report.as_ref() {
            // Best-effort, fire-and-forget. Failures are logged at debug
            // level so we don't spam stderr when the website is briefly
            // unreachable — the audit loop itself is what matters; the
            // phone-home is purely for operator visibility.
            let snap = snapshot.read().await.clone();
            let report_cfg = report_cfg.clone();
            let node_for_report = node.clone();
            tokio::spawn(async move {
                if let Err(e) = send_peer_report(&report_cfg, &node_for_report, &snap).await {
                    tracing::debug!("mesh visibility: peer-report POST failed: {e}");
                }
            });
        }

        if consecutive_invisible >= HARD_RESET_THRESHOLD
            && last_hard_reset.is_none_or(|t| t.elapsed() >= HARD_RESET_COOLDOWN)
        {
            tracing::warn!(
                "mesh visibility: {consecutive_invisible} consecutive non-Visible probes; \
                 triggering hard reset via process exit (supervisor will restart)"
            );
            {
                let mut s = snapshot.write().await;
                s.hard_reset_triggered = true;
            }
            last_hard_reset = Some(Instant::now());
            // Give the SSE push and any in-flight `/api/status` consumers
            // a beat to record the trigger before the process dies.
            tokio::time::sleep(Duration::from_secs(1)).await;
            // 75 = EX_TEMPFAIL — distinguishes "asked to restart cleanly"
            // from a panic/crash. systemd/launchd treat it as a normal
            // exit and restart per Restart=always / KeepAlive.
            std::process::exit(75);
        } else if consecutive_invisible >= SOFT_RECONNECT_THRESHOLD
            && last_soft_reconnect.is_none_or(|t| t.elapsed() >= SOFT_RECONNECT_COOLDOWN)
        {
            tracing::warn!(
                "mesh visibility: {consecutive_invisible} consecutive non-Visible probes; \
                 attempting soft reconnect via {entry_url}"
            );
            {
                let mut s = snapshot.write().await;
                s.soft_reconnect_triggered = true;
            }
            last_soft_reconnect = Some(Instant::now());
            attempt_soft_reconnect(
                &node,
                &entry_url,
                cached_fresh_token.as_deref(),
                &fallback_tokens,
            )
            .await;
        }

        tokio::time::sleep(AUDIT_INTERVAL).await;
    }
}

async fn probe_once(node: &crate::mesh::Node, entry_url: &str) -> ProbeOutcome {
    let my_id = node.id().fmt_short().to_string();
    probe_visibility(&my_id, entry_url).await
}

/// HTTP-only probe extracted from [`probe_once`] so tests can exercise
/// the classification logic against a real `axum::serve`-hosted fake
/// entry without building a full `mesh::Node`. The node-id string is
/// the only piece of node state we actually need.
async fn probe_visibility(my_id: &str, entry_url: &str) -> ProbeOutcome {
    let endpoint = format!("{}/api/status", entry_url.trim_end_matches('/'));

    let client = match reqwest::Client::builder().timeout(PROBE_TIMEOUT).build() {
        Ok(c) => c,
        Err(e) => {
            return ProbeOutcome {
                state: MeshVisibilityState::EntryUnreachable,
                error: Some(format!("Failed to build http client: {e}")),
                fresh_token: None,
            };
        }
    };

    let res = match client.get(&endpoint).send().await {
        Ok(r) => r,
        Err(e) => {
            return ProbeOutcome {
                state: MeshVisibilityState::EntryUnreachable,
                error: Some(format!("Probe to {endpoint} failed: {e}")),
                fresh_token: None,
            };
        }
    };

    if !res.status().is_success() {
        return ProbeOutcome {
            state: MeshVisibilityState::EntryUnreachable,
            error: Some(format!("Probe returned HTTP {}", res.status())),
            fresh_token: None,
        };
    }

    let json: serde_json::Value = match res.json().await {
        Ok(j) => j,
        Err(e) => {
            return ProbeOutcome {
                state: MeshVisibilityState::EntryUnreachable,
                error: Some(format!("Probe response was not JSON: {e}")),
                fresh_token: None,
            };
        }
    };

    classify_entry_response(&json, my_id)
}

/// Pure classification of an entry `/api/status` body against our
/// `my_id`. Split out so unit tests can hit every realistic JSON shape
/// (peer present, peer missing, entry-as-self, malformed peers list)
/// without spinning up a server.
fn classify_entry_response(json: &serde_json::Value, my_id: &str) -> ProbeOutcome {
    let entry_node_id = json
        .get("node_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let fresh_token = json
        .get("token")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);

    if entry_node_id == my_id {
        return ProbeOutcome {
            state: MeshVisibilityState::Visible,
            error: None,
            fresh_token,
        };
    }

    let peers = json
        .get("peers")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let in_peers = peers
        .iter()
        .any(|p| p.get("id").and_then(serde_json::Value::as_str) == Some(my_id));

    if in_peers {
        ProbeOutcome {
            state: MeshVisibilityState::Visible,
            error: None,
            fresh_token,
        }
    } else {
        ProbeOutcome {
            state: MeshVisibilityState::Invisible,
            error: Some(format!(
                "Entry returned {} peer(s) but my node_id {my_id} is not among them",
                peers.len(),
            )),
            fresh_token,
        }
    }
}

/// Update the shared snapshot with the latest probe outcome. Returns
/// the post-update `consecutive_invisible_count` so the caller can
/// decide whether to escalate without taking a second lock.
async fn update_snapshot(
    snapshot: &Arc<RwLock<MeshVisibilitySnapshot>>,
    outcome: &ProbeOutcome,
) -> u32 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut s = snapshot.write().await;
    s.state = outcome.state;
    s.last_check_unix = Some(now);
    match outcome.state {
        MeshVisibilityState::Visible => {
            s.last_visible_unix = Some(now);
            s.consecutive_invisible_count = 0;
            s.last_error = None;
            s.soft_reconnect_triggered = false;
            s.hard_reset_triggered = false;
        }
        MeshVisibilityState::Invisible | MeshVisibilityState::EntryUnreachable => {
            s.consecutive_invisible_count = s.consecutive_invisible_count.saturating_add(1);
            s.last_error = outcome.error.clone();
        }
        MeshVisibilityState::Unknown => {
            // The audit loop never writes Unknown; only the initial
            // constructor does. Treat as a no-op so we don't accidentally
            // reset counters.
        }
    }
    s.consecutive_invisible_count
}

async fn attempt_soft_reconnect(
    node: &crate::mesh::Node,
    entry_url: &str,
    cached_fresh_token: Option<&str>,
    fallback_tokens: &[String],
) {
    // Prefer the most-recently observed token from the probe loop, but also
    // perform a best-effort fresh fetch during reconnect. That covers entry
    // restarts where the cached token is stale or a proxy returned a partial
    // status payload without updating our cache.
    let mut candidates: Vec<String> = Vec::new();
    if let Some(t) = cached_fresh_token {
        candidates.push(t.to_string());
    }
    match fetch_fresh_token(entry_url).await {
        Ok(t) => {
            if !candidates.iter().any(|existing| existing == &t) {
                candidates.push(t);
            }
        }
        Err(e) => tracing::warn!(
            "mesh visibility: soft reconnect token refresh failed ({e}); \
             falling back to cached/startup tokens"
        ),
    }
    candidates.extend(fallback_tokens.iter().cloned());

    for token in &candidates {
        match node.join(token).await {
            Ok(()) => {
                tracing::info!(
                    "mesh visibility: soft reconnect via node.join succeeded (token len {})",
                    token.len()
                );
                return;
            }
            Err(e) => tracing::warn!("mesh visibility: node.join failed: {e}"),
        }
    }
    if candidates.is_empty() {
        tracing::warn!("mesh visibility: soft reconnect skipped — no join tokens available");
    }
}

/// POST this node's most recent audit snapshot to the website (Slice 4).
///
/// Fire-and-forget on the runtime side. The website normalizes /
/// clamps every field, so we can pass values verbatim — but we still
/// keep the body small (drop `last_error` longer than ~1 KB) to avoid
/// trivially DOSing ourselves on a chatty error path.
async fn send_peer_report(
    cfg: &PeerReportConfig,
    node: &crate::mesh::Node,
    snapshot: &MeshVisibilitySnapshot,
) -> anyhow::Result<()> {
    let url = format!("{}/api/peer-report", cfg.url.trim_end_matches('/'));
    let client = reqwest::Client::builder().timeout(PROBE_TIMEOUT).build()?;
    let serving_models = node.serving_models().await;
    let body = build_peer_report_body(
        &node.id().fmt_short().to_string(),
        cfg.hostname.as_deref(),
        cfg.version.as_deref(),
        &serving_models,
        snapshot,
    );

    let res = client.post(&url).json(&body).send().await?;
    if !res.status().is_success() {
        anyhow::bail!("peer-report returned HTTP {}", res.status());
    }
    Ok(())
}

/// Build the on-the-wire peer-report body. Split out so the unit test
/// suite can assert the shape matches the Vercel route's
/// `RuntimeReportWire` type without spinning up a `Node`. Field names
/// here are part of the public contract between the runtime and
/// `app/api/peer-report/route.ts`; do not rename without updating the
/// website ingest in lockstep.
fn build_peer_report_body(
    node_id: &str,
    hostname: Option<&str>,
    version: Option<&str>,
    serving_models: &[String],
    snapshot: &MeshVisibilitySnapshot,
) -> serde_json::Value {
    // Pre-clamp the on-the-wire error string defensively. The receiver
    // does this too, but trimming locally saves bandwidth on bad days.
    let last_error = snapshot.last_error.as_ref().map(|e| {
        if e.len() > 1024 {
            format!("{}…[truncated]", &e[..1024])
        } else {
            e.clone()
        }
    });

    serde_json::json!({
        "node_id": node_id,
        "hostname": hostname,
        "version": version,
        "serving_models": serving_models,
        "mesh_visibility": {
            "state": snapshot.state,
            "last_check_unix": snapshot.last_check_unix,
            "last_visible_unix": snapshot.last_visible_unix,
            "consecutive_invisible_count": snapshot.consecutive_invisible_count,
            "last_error": last_error,
            "entry_url": snapshot.entry_url,
            "soft_reconnect_triggered": snapshot.soft_reconnect_triggered,
            "hard_reset_triggered": snapshot.hard_reset_triggered,
        },
    })
}

async fn fetch_fresh_token(entry_url: &str) -> anyhow::Result<String> {
    let endpoint = format!("{}/api/status", entry_url.trim_end_matches('/'));
    let client = reqwest::Client::builder().timeout(PROBE_TIMEOUT).build()?;
    let res = client.get(&endpoint).send().await?.error_for_status()?;
    let v: serde_json::Value = res.json().await?;
    let token = v
        .get("token")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("entry status response missing token field"))?;
    Ok(token.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn update_snapshot_marks_visible_and_clears_counters() {
        let snap = Arc::new(RwLock::new(MeshVisibilitySnapshot::new(
            "https://example/".into(),
        )));
        // Two invisible outcomes raise the counter.
        let invisible = ProbeOutcome {
            state: MeshVisibilityState::Invisible,
            error: Some("not in peers".into()),
            fresh_token: None,
        };
        let count_after_two = {
            update_snapshot(&snap, &invisible).await;
            update_snapshot(&snap, &invisible).await
        };
        assert_eq!(count_after_two, 2);

        // Visible resets everything.
        let visible = ProbeOutcome {
            state: MeshVisibilityState::Visible,
            error: None,
            fresh_token: Some("tok".into()),
        };
        let count = update_snapshot(&snap, &visible).await;
        assert_eq!(count, 0);
        let s = snap.read().await;
        assert_eq!(s.state, MeshVisibilityState::Visible);
        assert!(s.last_visible_unix.is_some());
        assert!(s.last_error.is_none());
        assert!(!s.soft_reconnect_triggered);
        assert!(!s.hard_reset_triggered);
    }

    #[tokio::test]
    async fn update_snapshot_distinguishes_invisible_from_unreachable() {
        let snap = Arc::new(RwLock::new(MeshVisibilitySnapshot::new(
            "https://example/".into(),
        )));
        let unreachable = ProbeOutcome {
            state: MeshVisibilityState::EntryUnreachable,
            error: Some("timeout".into()),
            fresh_token: None,
        };
        let invisible = ProbeOutcome {
            state: MeshVisibilityState::Invisible,
            error: Some("not in peers".into()),
            fresh_token: None,
        };
        update_snapshot(&snap, &unreachable).await;
        update_snapshot(&snap, &invisible).await;
        let s = snap.read().await;
        // Both count toward the same escalation counter.
        assert_eq!(s.consecutive_invisible_count, 2);
        // Latest error reflects the most recent probe, not a sticky one.
        assert_eq!(s.last_error.as_deref(), Some("not in peers"));
        // State reflects the most recent probe.
        assert_eq!(s.state, MeshVisibilityState::Invisible);
    }

    #[test]
    fn invisible_and_unreachable_both_count_as_invisible() {
        assert!(MeshVisibilityState::Invisible.counts_as_invisible());
        assert!(MeshVisibilityState::EntryUnreachable.counts_as_invisible());
        assert!(!MeshVisibilityState::Visible.counts_as_invisible());
        assert!(!MeshVisibilityState::Unknown.counts_as_invisible());
    }

    // ---------------------------------------------------------------
    // Pure classification — every realistic entry response shape.
    //
    // These tests are the regression net for the May 2026 MSI bug:
    // the dashboard's green pill claimed everything was fine, but the
    // entry's peer list returned exactly one entry (LYU). The bug-fix
    // is meaningless if `classify_entry_response` can't tell the
    // difference between "I'm in the list" and "I'm not in the list".
    // ---------------------------------------------------------------

    fn entry_status_with_peers(peer_ids: &[&str], extra: serde_json::Value) -> serde_json::Value {
        let mut body = serde_json::json!({
            "node_id": "ENTRY_NODE_ID_abc123",
            "peers": peer_ids
                .iter()
                .map(|id| serde_json::json!({ "id": id, "hostname": "fake" }))
                .collect::<Vec<_>>(),
        });
        if let serde_json::Value::Object(map) = &mut body {
            if let serde_json::Value::Object(extra_map) = extra {
                for (k, v) in extra_map {
                    map.insert(k, v);
                }
            }
        }
        body
    }

    #[test]
    fn classify_visible_when_my_id_in_peers() {
        let body = entry_status_with_peers(&["LYU_id", "MSI_id"], serde_json::json!({}));
        let outcome = classify_entry_response(&body, "MSI_id");
        assert_eq!(outcome.state, MeshVisibilityState::Visible);
        assert!(outcome.error.is_none());
    }

    #[test]
    fn classify_invisible_when_my_id_missing_from_peers() {
        // Exactly the May 2026 MSI shape: entry sees one peer (LYU)
        // and not us, even though our local view says we're connected.
        let body = entry_status_with_peers(&["LYU_id"], serde_json::json!({}));
        let outcome = classify_entry_response(&body, "MSI_id");
        assert_eq!(outcome.state, MeshVisibilityState::Invisible);
        assert!(outcome.error.is_some());
        let err = outcome.error.unwrap();
        assert!(err.contains("MSI_id"), "error must name our id: {err}");
        assert!(err.contains("1 peer"), "error must count peers: {err}");
    }

    #[test]
    fn classify_visible_when_entry_is_self() {
        let body = entry_status_with_peers(&[], serde_json::json!({}));
        let outcome = classify_entry_response(&body, "ENTRY_NODE_ID_abc123");
        assert_eq!(outcome.state, MeshVisibilityState::Visible);
    }

    #[test]
    fn classify_captures_fresh_token() {
        let body =
            entry_status_with_peers(&["MSI_id"], serde_json::json!({ "token": "fresh-tok-xyz" }));
        let outcome = classify_entry_response(&body, "MSI_id");
        assert_eq!(outcome.state, MeshVisibilityState::Visible);
        assert_eq!(outcome.fresh_token.as_deref(), Some("fresh-tok-xyz"));
    }

    #[test]
    fn classify_invisible_response_still_captures_token() {
        // Crucial for soft-reconnect: even when we're invisible, the
        // entry's currently-advertised token is what we'll use to
        // re-join — so we must extract it from the same response.
        let body =
            entry_status_with_peers(&["LYU_id"], serde_json::json!({ "token": "fresh-tok-xyz" }));
        let outcome = classify_entry_response(&body, "MSI_id");
        assert_eq!(outcome.state, MeshVisibilityState::Invisible);
        assert_eq!(outcome.fresh_token.as_deref(), Some("fresh-tok-xyz"));
    }

    #[test]
    fn classify_invisible_when_peers_field_missing() {
        // Defensive: an old entry build with no `peers` key must still
        // be classified as Invisible (not crash, not Visible).
        let body = serde_json::json!({ "node_id": "ENTRY" });
        let outcome = classify_entry_response(&body, "MSI_id");
        assert_eq!(outcome.state, MeshVisibilityState::Invisible);
    }

    // ---------------------------------------------------------------
    // Real HTTP — exercises the full probe path against an axum server.
    // ---------------------------------------------------------------

    use axum::response::IntoResponse;
    use axum::{routing::get, Json, Router};
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicU32, Ordering};

    async fn spawn_fake_entry(
        handler_body: serde_json::Value,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let body = Arc::new(handler_body);
        let app = Router::new().route(
            "/api/status",
            get(move || {
                let body = body.clone();
                async move { Json((*body).clone()) }
            }),
        );
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), task)
    }

    async fn spawn_fake_entry_500() -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/api/status",
            get(|| async { (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
        );
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), task)
    }

    async fn spawn_fake_entry_garbage() -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let app = Router::new().route("/api/status", get(|| async { "this is not JSON" }));
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), task)
    }

    #[tokio::test]
    async fn probe_visibility_visible_against_real_server() {
        let body = entry_status_with_peers(
            &["MSI_id", "LYU_id"],
            serde_json::json!({ "token": "tok-real" }),
        );
        let (url, task) = spawn_fake_entry(body).await;
        let outcome = probe_visibility("MSI_id", &url).await;
        task.abort();
        assert_eq!(outcome.state, MeshVisibilityState::Visible);
        assert_eq!(outcome.fresh_token.as_deref(), Some("tok-real"));
    }

    #[tokio::test]
    async fn probe_visibility_invisible_against_real_server() {
        let body = entry_status_with_peers(&["LYU_id"], serde_json::json!({}));
        let (url, task) = spawn_fake_entry(body).await;
        let outcome = probe_visibility("MSI_id", &url).await;
        task.abort();
        assert_eq!(outcome.state, MeshVisibilityState::Invisible);
        assert!(outcome.error.unwrap().contains("MSI_id"));
    }

    #[tokio::test]
    async fn probe_visibility_unreachable_on_http_500() {
        let (url, task) = spawn_fake_entry_500().await;
        let outcome = probe_visibility("MSI_id", &url).await;
        task.abort();
        assert_eq!(outcome.state, MeshVisibilityState::EntryUnreachable);
        let err = outcome.error.unwrap();
        assert!(err.contains("500"), "should report status code, got: {err}");
    }

    #[tokio::test]
    async fn probe_visibility_unreachable_on_non_json_response() {
        let (url, task) = spawn_fake_entry_garbage().await;
        let outcome = probe_visibility("MSI_id", &url).await;
        task.abort();
        assert_eq!(outcome.state, MeshVisibilityState::EntryUnreachable);
    }

    #[tokio::test]
    async fn probe_visibility_unreachable_on_dead_port() {
        // Bind then drop to grab a port no-one is listening on. There
        // is a TOCTOU window where the OS could re-allocate the same
        // port to a different process, but in practice the test
        // window is sub-millisecond.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let url = format!("http://{addr}");
        let outcome = probe_visibility("MSI_id", &url).await;
        assert_eq!(outcome.state, MeshVisibilityState::EntryUnreachable);
    }

    #[tokio::test]
    async fn probe_visibility_trims_trailing_slash_in_entry_url() {
        let body = entry_status_with_peers(&["MSI_id"], serde_json::json!({}));
        let (url, task) = spawn_fake_entry(body).await;
        let url_with_slash = format!("{url}/");
        let outcome = probe_visibility("MSI_id", &url_with_slash).await;
        task.abort();
        assert_eq!(outcome.state, MeshVisibilityState::Visible);
    }

    #[tokio::test]
    async fn probe_visibility_recovers_after_transient_failure() {
        // Real bug-shape from the May 2026 incident: entry returns
        // 503 for ~30 s during a deploy, then comes back. We must
        // not stay stuck in EntryUnreachable once it recovers.
        let counter = Arc::new(AtomicU32::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let c = counter.clone();
        let app = Router::new().route(
            "/api/status",
            get(move || {
                let c = c.clone();
                async move {
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    if n < 2 {
                        // First two calls: HTTP 503.
                        (
                            axum::http::StatusCode::SERVICE_UNAVAILABLE,
                            axum::Json(serde_json::json!({"error": "deploying"})),
                        )
                            .into_response()
                    } else {
                        axum::Json(serde_json::json!({
                            "node_id": "ENTRY",
                            "peers": [{"id": "MSI_id"}],
                            "token": "post-deploy-tok",
                        }))
                        .into_response()
                    }
                }
            }),
        );
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let url = format!("http://{addr}");

        let first = probe_visibility("MSI_id", &url).await;
        assert_eq!(first.state, MeshVisibilityState::EntryUnreachable);
        let second = probe_visibility("MSI_id", &url).await;
        assert_eq!(second.state, MeshVisibilityState::EntryUnreachable);
        let third = probe_visibility("MSI_id", &url).await;
        assert_eq!(third.state, MeshVisibilityState::Visible);
        assert_eq!(third.fresh_token.as_deref(), Some("post-deploy-tok"));
        task.abort();
    }

    // ---------------------------------------------------------------
    // Peer-report wire shape — assert every field the website expects
    // is present and named correctly. The peer-report endpoint on
    // senda.network is the only thing standing between an invisible
    // node and zero operator visibility, so a silent rename here
    // becomes a silent operational blackout.
    // ---------------------------------------------------------------

    #[test]
    fn peer_report_body_matches_website_schema() {
        let snap = MeshVisibilitySnapshot {
            state: MeshVisibilityState::Invisible,
            last_check_unix: Some(1_700_000_010),
            last_visible_unix: Some(1_700_000_000),
            consecutive_invisible_count: 3,
            last_error: Some("not in peers".into()),
            entry_url: "https://entry.senda.network".into(),
            soft_reconnect_triggered: true,
            hard_reset_triggered: false,
        };
        let body = build_peer_report_body(
            "MSI_id",
            Some("MSI"),
            Some("0.66.18"),
            &["DeepSeek-R1-Distill-70B-Q4_K_M".into()],
            &snap,
        );

        // Top-level keys (must match `app/api/peer-report/route.ts`).
        assert_eq!(body["node_id"], "MSI_id");
        assert_eq!(body["hostname"], "MSI");
        assert_eq!(body["version"], "0.66.18");
        assert_eq!(body["serving_models"][0], "DeepSeek-R1-Distill-70B-Q4_K_M");

        // Nested mesh_visibility keys.
        let mv = &body["mesh_visibility"];
        assert_eq!(mv["state"], "invisible");
        assert_eq!(mv["last_check_unix"], 1_700_000_010);
        assert_eq!(mv["last_visible_unix"], 1_700_000_000);
        assert_eq!(mv["consecutive_invisible_count"], 3);
        assert_eq!(mv["last_error"], "not in peers");
        assert_eq!(mv["entry_url"], "https://entry.senda.network");
        assert_eq!(mv["soft_reconnect_triggered"], true);
        assert_eq!(mv["hard_reset_triggered"], false);
    }

    #[test]
    fn peer_report_body_serializes_state_in_snake_case() {
        // The website's `RuntimeMeshVisibility` discriminator union is
        // matched by string literal — `"entry_unreachable"`, not
        // `"EntryUnreachable"`. Any drift here makes the website
        // silently down-classify the peer.
        for (variant, expected) in [
            (MeshVisibilityState::Unknown, "unknown"),
            (MeshVisibilityState::Visible, "visible"),
            (MeshVisibilityState::Invisible, "invisible"),
            (MeshVisibilityState::EntryUnreachable, "entry_unreachable"),
        ] {
            let snap = MeshVisibilitySnapshot {
                state: variant,
                last_check_unix: None,
                last_visible_unix: None,
                consecutive_invisible_count: 0,
                last_error: None,
                entry_url: "x".into(),
                soft_reconnect_triggered: false,
                hard_reset_triggered: false,
            };
            let body = build_peer_report_body("id", None, None, &[], &snap);
            assert_eq!(
                body["mesh_visibility"]["state"], expected,
                "expected snake_case for {variant:?}"
            );
        }
    }

    #[test]
    fn peer_report_body_clamps_oversized_last_error() {
        // Defensive against pathological error strings (e.g. a stack
        // trace from llama-server) blowing past the 8 KB Vercel
        // ingest cap. The website also clamps, but local trimming
        // saves egress on bad days.
        let long_err = "x".repeat(5_000);
        let snap = MeshVisibilitySnapshot {
            state: MeshVisibilityState::Invisible,
            last_check_unix: Some(0),
            last_visible_unix: None,
            consecutive_invisible_count: 1,
            last_error: Some(long_err.clone()),
            entry_url: "x".into(),
            soft_reconnect_triggered: false,
            hard_reset_triggered: false,
        };
        let body = build_peer_report_body("id", None, None, &[], &snap);
        let err = body["mesh_visibility"]["last_error"].as_str().unwrap();
        assert!(
            err.len() <= 1024 + 32,
            "should be trimmed: len = {}",
            err.len()
        );
        assert!(err.contains("truncated"), "should mark as truncated");
    }

    #[test]
    fn peer_report_body_omits_optional_null_fields_as_expected() {
        // Sanity-check on the wire shape when hostname/version are not
        // configured (e.g. dev / headless). `serde_json::json!` keeps
        // `Option::None` as a JSON `null`, which the website tolerates
        // — but we want to make that contract explicit so a future
        // refactor doesn't accidentally start sending the key as
        // missing-vs-null and break the ingest validator.
        let snap = MeshVisibilitySnapshot {
            state: MeshVisibilityState::Unknown,
            last_check_unix: None,
            last_visible_unix: None,
            consecutive_invisible_count: 0,
            last_error: None,
            entry_url: "x".into(),
            soft_reconnect_triggered: false,
            hard_reset_triggered: false,
        };
        let body = build_peer_report_body("id", None, None, &[], &snap);
        assert!(body["hostname"].is_null());
        assert!(body["version"].is_null());
        assert!(body["mesh_visibility"]["last_check_unix"].is_null());
    }
}
