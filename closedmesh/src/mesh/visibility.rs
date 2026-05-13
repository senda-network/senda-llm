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
//! `https://mesh.closedmesh.com/api/status` returned exactly one peer
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
    /// `https://closedmesh.com`). `/api/peer-report` is appended.
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
/// `https://mesh.closedmesh.com`) — typically the same URL the runtime
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
            && last_hard_reset.map_or(true, |t| t.elapsed() >= HARD_RESET_COOLDOWN)
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
            && last_soft_reconnect.map_or(true, |t| t.elapsed() >= SOFT_RECONNECT_COOLDOWN)
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
            attempt_soft_reconnect(&node, &entry_url, cached_fresh_token.as_deref(), &fallback_tokens)
                .await;
        }

        tokio::time::sleep(AUDIT_INTERVAL).await;
    }
}

async fn probe_once(node: &crate::mesh::Node, entry_url: &str) -> ProbeOutcome {
    let my_id = node.id().fmt_short().to_string();
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
        .any(|p| p.get("id").and_then(serde_json::Value::as_str) == Some(my_id.as_str()));

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
                "Entry returned {} peer(s) but my node_id {} is not among them",
                peers.len(),
                my_id
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
    // Prefer the most-recently observed token from the probe loop —
    // that's by definition fresh as of <30 s ago. If the cache is empty
    // (first audit window, or every probe so far has been
    // EntryUnreachable) try a one-shot fetch. Both can fail, so cycle
    // through any startup fallback tokens last.
    let mut candidates: Vec<String> = Vec::new();
    if let Some(t) = cached_fresh_token {
        candidates.push(t.to_string());
    }
    if candidates.is_empty() {
        match fetch_fresh_token(entry_url).await {
            Ok(t) => candidates.push(t),
            Err(e) => tracing::warn!(
                "mesh visibility: soft reconnect token refresh failed ({e}); \
                 falling back to startup tokens"
            ),
        }
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

    // Pre-clamp the on-the-wire error string defensively. The receiver
    // does this too, but trimming locally saves bandwidth on bad days.
    let last_error = snapshot.last_error.as_ref().map(|e| {
        if e.len() > 1024 {
            format!("{}…[truncated]", &e[..1024])
        } else {
            e.clone()
        }
    });

    let serving_models = node.serving_models().await;
    let body = serde_json::json!({
        "node_id": node.id().fmt_short().to_string(),
        "hostname": cfg.hostname,
        "version": cfg.version,
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
    });

    let res = client.post(&url).json(&body).send().await?;
    if !res.status().is_success() {
        anyhow::bail!("peer-report returned HTTP {}", res.status());
    }
    Ok(())
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
}
