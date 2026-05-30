//! Verification v1 — the comparison oracle for deterministic model-identity
//! fingerprints.
//!
//! Given a *reference* fingerprint (auditor-established ground truth for a
//! `(model, quant)` today; majority-consensus later) and a *candidate*
//! fingerprint produced by a peer that claims to serve the same model, decide
//! whether the candidate is plausibly that model run honestly.
//!
//! Why this shape:
//! - **Prefix agreement is the gate.** The greedy decode of a fixed
//!   `temp=0 / seed=42` prompt is a strong model-identity signal: a different
//!   or smaller model, or canned text, diverges within the first few tokens.
//!   We compare only a bounded prefix and allow a small disagreement budget,
//!   because even greedy decoding can diverge in the tail across
//!   Metal / CUDA / Vulkan from floating-point differences in near-tie
//!   argmaxes.
//! - **No logprob signal.** At `temp=0` the chosen token's logprob is
//!   definitionally 0 and llama.cpp returns no alternatives, so per-token
//!   logprobs carried no information; a nonzero temperature would recover
//!   them but make the *token* sequence far less stable across backends,
//!   which is the opposite of what we want. The token sequence and output
//!   hash are the discriminators.
//!
//! Limitation (documented honestly): a fixed probe is spoofable by a
//! sophisticated peer that detects the known prompt and runs the real model
//! only for it. v1 targets the common threat — wrong/smaller model, canned
//! replies, misconfiguration. Unpredictable probes + second-node recompute
//! (proof-of-sampling) is a later increment.

use super::native_baseline::{self, LogitFingerprint};
use crate::mesh;
use iroh::EndpointId;
use std::collections::HashMap;
use std::time::Duration;

/// Tunables for [`compare_fingerprints`]. Defaults are deliberately
/// conservative: they favor *not* convicting an honest peer over catching
/// every cheat, because a false mismatch demotes a real contributor.
#[derive(Debug, Clone)]
pub struct VerifyThresholds {
    /// Minimum fraction of compared prefix tokens that must match for a
    /// `Match`. Below 1.0 to tolerate cross-backend tail divergence.
    pub min_prefix_agreement: f64,
    /// Need at least this many overlapping tokens to render a token verdict;
    /// fewer → `Inconclusive` (not enough signal).
    pub min_compared_tokens: usize,
}

impl Default for VerifyThresholds {
    fn default() -> Self {
        Self {
            min_prefix_agreement: 0.9,
            min_compared_tokens: 8,
        }
    }
}

/// Outcome of comparing a candidate fingerprint against a reference.
#[derive(Debug, Clone, PartialEq)]
pub enum FingerprintVerdict {
    /// Candidate is plausibly the same model, run honestly.
    Match {
        prefix_agreement: f64,
        compared_tokens: usize,
    },
    /// Candidate clearly diverges — wrong/smaller model or canned output.
    Mismatch {
        prefix_agreement: f64,
        compared_tokens: usize,
        reason: &'static str,
    },
    /// Not enough signal to decide. Must never be used to demote a peer.
    Inconclusive { reason: &'static str },
}

/// Compare a `candidate` fingerprint against a `reference` under `thresholds`.
/// Pure + deterministic so it is unit-testable without a network.
pub fn compare_fingerprints(
    reference: &LogitFingerprint,
    candidate: &LogitFingerprint,
    thresholds: &VerifyThresholds,
) -> FingerprintVerdict {
    // No token-level material on either side → fall back to the output hash.
    // An exact hash match is a clean positive; a mismatch with no prefix is
    // ambiguous (expected across backends), so we decline to judge.
    if reference.prefix_tokens.is_empty() || candidate.prefix_tokens.is_empty() {
        if !reference.output_sha256.is_empty() && reference.output_sha256 == candidate.output_sha256
        {
            return FingerprintVerdict::Match {
                prefix_agreement: 1.0,
                compared_tokens: 0,
            };
        }
        return FingerprintVerdict::Inconclusive {
            reason: "no token prefix to compare and output hashes differ \
                     (expected across backends) — cannot judge with tolerance",
        };
    }

    let compared = reference
        .prefix_tokens
        .len()
        .min(candidate.prefix_tokens.len());
    if compared < thresholds.min_compared_tokens {
        return FingerprintVerdict::Inconclusive {
            reason: "too few overlapping tokens to render a verdict",
        };
    }

    let token_matches = (0..compared)
        .filter(|&i| reference.prefix_tokens[i] == candidate.prefix_tokens[i])
        .count();
    let prefix_agreement = token_matches as f64 / compared as f64;

    if prefix_agreement < thresholds.min_prefix_agreement {
        return FingerprintVerdict::Mismatch {
            prefix_agreement,
            compared_tokens: compared,
            reason: "greedy decode diverges from reference — likely a \
                     different/smaller model or canned output",
        };
    }

    FingerprintVerdict::Match {
        prefix_agreement,
        compared_tokens: compared,
    }
}

// ── Verifier loop ────────────────────────────────────────────────────────────
//
// Observe by default, enforce only when explicitly enabled. The loop probes
// peers, runs the oracle, and logs verdicts. Demotion is the one
// irreversible-ish lever (a false positive punishes an honest contributor), so
// it is gated three ways: (1) off unless `CLOSEDMESH_VERIFY_ENFORCE` is set,
// (2) requires several *consecutive* `Mismatch` verdicts for the same
// `(peer, model)` before acting, never a single flaky probe, and (3) the
// action is a reversible, time-boxed route demotion — the peer stays in the
// mesh, keeps being re-probed, and is reinstated on the next `Match` or when
// the cooldown lapses. `Inconclusive` never counts toward conviction.

/// Set to a truthy value (`1`/`true`/`yes`/`on`) to let the verifier demote
/// convicted peers instead of only logging.
const ENFORCE_ENV: &str = "CLOSEDMESH_VERIFY_ENFORCE";

/// Config for the background verifier loop.
#[derive(Debug, Clone)]
pub struct VerifierConfig {
    /// Delay before the first audit — let gossip + native baselines settle.
    pub settle: Duration,
    /// Pause between audits. One `(peer, model)` is checked per tick.
    pub interval: Duration,
    /// Per-probe timeout (re-probing a slow peer over the tunnel).
    pub probe_timeout: Duration,
    pub thresholds: VerifyThresholds,
    /// When `false` (default), the loop only logs verdicts. When `true`, a
    /// convicted `(peer, model)` is demoted from routing for `demotion`.
    pub enforce: bool,
    /// Consecutive `Mismatch` verdicts required before a demotion. `>= 1`.
    pub min_consecutive_mismatches: u32,
    /// How long a demotion lasts before the peer is auto-reinstated.
    pub demotion: Duration,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            settle: Duration::from_secs(60),
            interval: Duration::from_secs(120),
            probe_timeout: Duration::from_secs(60),
            thresholds: VerifyThresholds::default(),
            enforce: false,
            min_consecutive_mismatches: 2,
            demotion: Duration::from_secs(900), // 15 min, reversible
        }
    }
}

fn enforce_from_env() -> bool {
    std::env::var(ENFORCE_ENV)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Auditor-established reference fingerprints, keyed by model name.
///
/// Bootstrap (Option A): references we generate by running canonical models on
/// our own hardware. They ship two ways, layered:
/// 1. **Embedded** in the binary (`reference_fingerprints.json`, `include_str!`)
///    so every node — including GPU-less entry routers — has them with zero
///    config. Regenerated by copying a capture into that file before a release.
/// 2. **On-disk override** at `~/.closedmesh/reference-fingerprints.json`,
///    which extends/overrides the embedded set without a rebuild (local
///    recapture, new models). This is what `benchmark capture-reference` writes.
///
/// When no reference exists for a model at all, the verifier falls back to
/// *this* node's own native baseline fingerprint, but only if this node serves
/// that model — a node that genuinely runs the model is a valid auditor for it.
#[derive(Debug, Clone, Default)]
pub struct ReferenceStore {
    refs: HashMap<String, LogitFingerprint>,
}

/// References baked into the binary at build time (Option A defaults).
const EMBEDDED_REFERENCES: &str = include_str!("reference_fingerprints.json");

impl ReferenceStore {
    /// Load embedded references, then layer the on-disk file on top so a local
    /// capture overrides/extends the shipped set without a rebuild.
    pub fn load() -> Self {
        let mut refs: HashMap<String, LogitFingerprint> =
            serde_json::from_str(EMBEDDED_REFERENCES).unwrap_or_default();
        if let Some(path) = reference_path() {
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Ok(disk) = serde_json::from_str::<HashMap<String, LogitFingerprint>>(&raw) {
                    refs.extend(disk);
                }
            }
        }
        Self { refs }
    }

    pub fn get(&self, model: &str) -> Option<&LogitFingerprint> {
        self.refs.get(model)
    }
}

pub(crate) fn reference_path() -> Option<std::path::PathBuf> {
    if let Ok(custom) = std::env::var("CLOSEDMESH_HOME") {
        return Some(std::path::PathBuf::from(custom).join("reference-fingerprints.json"));
    }
    dirs::home_dir().map(|h| h.join(".closedmesh").join("reference-fingerprints.json"))
}

/// Insert or replace one model's reference fingerprint in the on-disk store
/// and persist it (pretty JSON). Returns the path written. The store owns its
/// own file format here so capture and load stay in sync.
pub(crate) fn upsert_reference(
    model: &str,
    fp: &LogitFingerprint,
) -> anyhow::Result<std::path::PathBuf> {
    let path = reference_path()
        .ok_or_else(|| anyhow::anyhow!("could not resolve reference-fingerprints.json path"))?;
    let mut store = ReferenceStore::load();
    store.refs.insert(model.to_string(), fp.clone());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, serde_json::to_string_pretty(&store.refs)?)?;
    Ok(path)
}

/// Spawn the background verifier. Intended to run on entry / proxy nodes.
/// Observe-only unless `CLOSEDMESH_VERIFY_ENFORCE` is set, in which case
/// repeatedly-convicted peers are demoted from routing for a cooldown.
pub fn spawn_verifier(node: mesh::Node, mut config: VerifierConfig) {
    if enforce_from_env() {
        config.enforce = true;
    }
    config.min_consecutive_mismatches = config.min_consecutive_mismatches.max(1);
    tokio::spawn(async move {
        tokio::time::sleep(config.settle).await;
        let store = ReferenceStore::load();
        tracing::info!(
            target: "closedmesh::verify",
            shipped_refs = store.refs.len(),
            interval_secs = config.interval.as_secs(),
            enforce = config.enforce,
            min_consecutive_mismatches = config.min_consecutive_mismatches,
            demotion_secs = config.demotion.as_secs(),
            "verifier loop started ({})",
            if config.enforce { "enforcing" } else { "observe-only" }
        );
        // Consecutive-mismatch streak per (peer, model). Reset on Match;
        // untouched by Inconclusive. Drives conviction before any demotion.
        let mut streaks: HashMap<(EndpointId, String), u32> = HashMap::new();
        loop {
            tokio::time::sleep(config.interval).await;
            if let Err(e) = run_one_audit(&node, &store, &config, &mut streaks).await {
                tracing::debug!(target: "closedmesh::verify", "audit tick skipped: {e}");
            }
        }
    });
}

/// This node's own native-baseline fingerprints, keyed by model — the
/// self-baseline reference fallback for models this node genuinely serves.
async fn self_baseline_fingerprints(node: &mesh::Node) -> HashMap<String, LogitFingerprint> {
    node.native_baselines_snapshot()
        .await
        .into_iter()
        .filter_map(|b| b.logit_fingerprint.map(|fp| (b.model, fp)))
        .collect()
}

/// Pick one auditable `(peer, model)`, establish ground truth, re-probe the
/// peer with the identical probe, compare, and log + enforce the verdict.
/// `Err` only signals "nothing to do this tick".
///
/// Ground truth is established one of two ways, preferring the stronger:
/// - **Self-oracle (preferred).** When *we* serve the model locally, generate
///   a fresh nonce-randomized probe, run it on our own llama-server, and send
///   the identical probe to the suspect. Because the probe is unpredictable,
///   a peer can't recognize it and serve the real model only for it — this is
///   what closes the known-prompt spoof.
/// - **Fixed reference (fallback).** Otherwise compare the suspect's fixed
///   probe against the precomputed reference (embedded / on-disk / our own
///   baseline). Spoofable by a peer that detects the known prompt, but still
///   catches wrong/smaller models, canned text, and misconfiguration.
async fn run_one_audit(
    node: &mesh::Node,
    store: &ReferenceStore,
    config: &VerifierConfig,
    streaks: &mut HashMap<(EndpointId, String), u32>,
) -> anyhow::Result<()> {
    let local_id = node.id();
    let self_refs = self_baseline_fingerprints(node).await;
    let local_ports = node.local_model_ports_snapshot().await;

    // (peer, model) pairs we can establish ground truth for — either we serve
    // the model (self-oracle) or we hold a fixed reference for it.
    let mut candidates: Vec<(EndpointId, String)> = Vec::new();
    for p in node.peers().await {
        if p.id == local_id {
            continue; // never audit ourselves
        }
        if !matches!(p.role, mesh::NodeRole::Host { .. }) {
            continue; // only HTTP-routable hosts are reachable for a probe
        }
        for model in p.http_routable_models() {
            let auditable = local_ports.contains_key(&model)
                || store.get(&model).is_some()
                || self_refs.contains_key(&model);
            if auditable {
                candidates.push((p.id, model));
            }
        }
    }
    if candidates.is_empty() {
        anyhow::bail!("no auditable (peer, model) pairs");
    }

    let idx = (rand::random::<u64>() as usize) % candidates.len();
    let (peer_id, model) = candidates.swap_remove(idx);

    let (reference, candidate, mode) = if let Some(&port) = local_ports.get(&model) {
        // Self-oracle: fresh unpredictable probe, ground truth from our own model.
        let nonce: u64 = rand::random();
        let messages = native_baseline::probe_messages_for(nonce);
        let reference =
            native_baseline::local_probe_fingerprint(port, &model, messages.clone(), nonce).await?;
        let candidate = tokio::time::timeout(
            config.probe_timeout,
            remote_probe_fingerprint(node, peer_id, &model, messages, nonce),
        )
        .await
        .map_err(|_| anyhow::anyhow!("probe timed out"))??;
        (reference, candidate, "self_oracle_random")
    } else {
        // Fixed reference fallback.
        let reference = store
            .get(&model)
            .cloned()
            .or_else(|| self_refs.get(&model).cloned())
            .ok_or_else(|| anyhow::anyhow!("reference vanished between selection and probe"))?;
        let candidate = tokio::time::timeout(
            config.probe_timeout,
            remote_probe_fingerprint(
                node,
                peer_id,
                &model,
                native_baseline::probe_messages(),
                native_baseline::PROBE_SEED,
            ),
        )
        .await
        .map_err(|_| anyhow::anyhow!("probe timed out"))??;
        (reference, candidate, "fixed_reference")
    };

    let verdict = compare_fingerprints(&reference, &candidate, &config.thresholds);
    log_verdict(peer_id, &model, mode, &verdict);
    apply_enforcement(node, peer_id, &model, &verdict, config, streaks).await;
    Ok(())
}

/// What the verifier should do to the routable set after a verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnforcementAction {
    /// Leave routing untouched.
    None,
    /// Lift any active demotion for this `(peer, model)` (it behaved).
    Reinstate,
    /// Demote this `(peer, model)` for the cooldown.
    Demote,
}

/// Pure conviction state machine: given a verdict, the current consecutive-
/// mismatch streak, and config, return the new streak and the routing action.
/// Kept side-effect-free so the safety rules are unit-testable without a node.
///
/// Rules:
/// - `Inconclusive` → streak unchanged, no action (never convicts).
/// - `Match` → streak reset to 0; `Reinstate` when enforcing (else `None`).
/// - `Mismatch` → streak +1; `Demote` only when enforcing *and* the streak has
///   reached `min_consecutive_mismatches`. A single mismatch never demotes.
fn next_streak_and_action(
    verdict: &FingerprintVerdict,
    current_streak: u32,
    config: &VerifierConfig,
) -> (u32, EnforcementAction) {
    match verdict {
        FingerprintVerdict::Inconclusive { .. } => (current_streak, EnforcementAction::None),
        FingerprintVerdict::Match { .. } => (
            0,
            if config.enforce {
                EnforcementAction::Reinstate
            } else {
                EnforcementAction::None
            },
        ),
        FingerprintVerdict::Mismatch { .. } => {
            let streak = current_streak.saturating_add(1);
            let action = if config.enforce && streak >= config.min_consecutive_mismatches {
                EnforcementAction::Demote
            } else {
                EnforcementAction::None
            };
            (streak, action)
        }
    }
}

/// Apply [`next_streak_and_action`] to the shared streak map and the node's
/// routable set.
async fn apply_enforcement(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    verdict: &FingerprintVerdict,
    config: &VerifierConfig,
    streaks: &mut HashMap<(EndpointId, String), u32>,
) {
    let key = (peer_id, model.to_string());
    let current = streaks.get(&key).copied().unwrap_or(0);
    let (new_streak, action) = next_streak_and_action(verdict, current, config);
    if new_streak == 0 {
        streaks.remove(&key);
    } else {
        streaks.insert(key, new_streak);
    }
    match action {
        EnforcementAction::None => {}
        EnforcementAction::Reinstate => {
            node.clear_peer_model_demotion(peer_id, model).await;
        }
        EnforcementAction::Demote => {
            let until = std::time::Instant::now() + config.demotion;
            node.demote_peer_model(peer_id, model, until).await;
            tracing::warn!(
                target: "closedmesh::verify",
                peer = %peer_id.fmt_short(),
                model,
                consecutive_mismatches = new_streak,
                demotion_secs = config.demotion.as_secs(),
                "verify: DEMOTED (cooldown; reversible, peer stays in mesh)"
            );
        }
    }
}

/// Re-probe a specific peer with a byte-identical probe over the mesh QUIC
/// tunnel and build the candidate fingerprint from the (non-streaming)
/// response. `messages`/`seed` must match whatever the reference side used —
/// the fixed probe, or a randomized self-oracle probe.
async fn remote_probe_fingerprint(
    node: &mesh::Node,
    peer_id: EndpointId,
    model: &str,
    messages: Vec<serde_json::Value>,
    seed: u64,
) -> anyhow::Result<LogitFingerprint> {
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": native_baseline::MAX_TOKENS,
        "temperature": 0,
        "seed": seed,
        "stream": false,
        "logprobs": true,
        // Don't let the peer recurse into its own consultation hooks.
        "mesh_hooks": false,
    });
    let body_bytes = serde_json::to_vec(&body)?;
    let http_request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         \r\n",
        body_bytes.len()
    );
    let mut raw = http_request.into_bytes();
    raw.extend_from_slice(&body_bytes);

    let (mut send, mut recv) = node.open_http_tunnel(peer_id).await?;
    send.write_all(&raw).await?;
    send.finish()?;

    let response_bytes = recv.read_to_end(256 * 1024).await?;
    let response_str = String::from_utf8_lossy(&response_bytes);
    let header_end = response_str
        .find("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP response: no header terminator"))?;
    let status_code: u16 = response_str[..header_end]
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if status_code != 200 {
        anyhow::bail!("peer returned HTTP {status_code}");
    }
    let body = &response_str[header_end + 4..];
    let parsed: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| anyhow::anyhow!("failed to parse peer probe response: {e}"))?;
    native_baseline::fingerprint_from_completion_json(&parsed)
        .ok_or_else(|| anyhow::anyhow!("peer probe response had no usable output"))
}

fn log_verdict(peer_id: EndpointId, model: &str, mode: &str, verdict: &FingerprintVerdict) {
    match verdict {
        FingerprintVerdict::Match {
            prefix_agreement,
            compared_tokens,
        } => {
            tracing::info!(
                target: "closedmesh::verify",
                peer = %peer_id.fmt_short(),
                model,
                mode,
                prefix_agreement,
                compared_tokens,
                "verify: MATCH"
            );
        }
        FingerprintVerdict::Mismatch {
            prefix_agreement,
            compared_tokens,
            reason,
        } => {
            tracing::warn!(
                target: "closedmesh::verify",
                peer = %peer_id.fmt_short(),
                model,
                mode,
                prefix_agreement,
                compared_tokens,
                reason,
                "verify: MISMATCH"
            );
        }
        FingerprintVerdict::Inconclusive { reason } => {
            tracing::debug!(
                target: "closedmesh::verify",
                peer = %peer_id.fmt_short(),
                model,
                mode,
                reason,
                "verify: inconclusive"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(tokens: &[&str], hash: &str) -> LogitFingerprint {
        LogitFingerprint {
            token_count: tokens.len() as u32,
            output_sha256: hash.to_string(),
            prefix_tokens: tokens.iter().map(|t| t.to_string()).collect(),
        }
    }

    fn ten() -> Vec<&'static str> {
        vec![
            "The", " cat", " sat", " on", " the", " mat", " and", " then", " ran", " away",
        ]
    }

    fn is_match(v: &FingerprintVerdict) -> bool {
        matches!(v, FingerprintVerdict::Match { .. })
    }
    fn is_mismatch(v: &FingerprintVerdict) -> bool {
        matches!(v, FingerprintVerdict::Mismatch { .. })
    }

    #[test]
    fn identical_fingerprints_match_perfectly() {
        let r = fp(&ten(), "h1");
        let v = compare_fingerprints(&r, &r, &VerifyThresholds::default());
        assert!(is_match(&v));
        if let FingerprintVerdict::Match {
            prefix_agreement, ..
        } = v
        {
            assert!((prefix_agreement - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn wrong_model_mismatches() {
        let r = fp(&ten(), "h1");
        let wrong = fp(
            &[
                "Hola",
                " mundo",
                " esto",
                " es",
                " otro",
                " modelo",
                " muy",
                " distinto",
                " aqui",
                " vale",
            ],
            "h2",
        );
        let v = compare_fingerprints(&r, &wrong, &VerifyThresholds::default());
        assert!(is_mismatch(&v));
    }

    #[test]
    fn one_token_off_is_within_budget() {
        let r = fp(&ten(), "h1");
        let mut toks = ten();
        toks[5] = " rug"; // 9/10 agree = 0.9 >= threshold
        let c = fp(&toks, "h2");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(is_match(&v), "9/10 agreement should pass the 0.9 gate");
    }

    #[test]
    fn two_tokens_off_breaks_budget() {
        let r = fp(&ten(), "h1");
        let mut toks = ten();
        toks[5] = " rug";
        toks[7] = " later"; // 8/10 = 0.8 < 0.9
        let c = fp(&toks, "h2");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(is_mismatch(&v), "8/10 agreement should fail the 0.9 gate");
    }

    #[test]
    fn empty_prefix_equal_hash_matches() {
        let r = fp(&[], "samehash");
        let c = fp(&[], "samehash");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(is_match(&v));
    }

    #[test]
    fn empty_prefix_diff_hash_is_inconclusive() {
        let r = fp(&[], "h1");
        let c = fp(&[], "h2");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(matches!(v, FingerprintVerdict::Inconclusive { .. }));
    }

    #[test]
    fn too_few_tokens_is_inconclusive() {
        let r = fp(&["a", "b", "c"], "h1");
        let c = fp(&["a", "b", "c"], "h1");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(matches!(v, FingerprintVerdict::Inconclusive { .. }));
    }

    #[test]
    fn embedded_references_parse_and_include_daily_driver() {
        let refs: HashMap<String, LogitFingerprint> = serde_json::from_str(EMBEDDED_REFERENCES)
            .expect("embedded reference_fingerprints.json must be valid JSON");
        let fp = refs
            .get("Qwen3-8B-Q4_K_M")
            .expect("embedded references must include the canonical daily driver");
        assert!(!fp.output_sha256.is_empty());
        assert_eq!(fp.prefix_tokens.len(), 32);
    }

    // ── conviction / enforcement state machine ────────────────────────────

    fn mismatch() -> FingerprintVerdict {
        FingerprintVerdict::Mismatch {
            prefix_agreement: 0.1,
            compared_tokens: 32,
            reason: "test",
        }
    }
    fn matched() -> FingerprintVerdict {
        FingerprintVerdict::Match {
            prefix_agreement: 1.0,
            compared_tokens: 32,
        }
    }
    fn inconclusive() -> FingerprintVerdict {
        FingerprintVerdict::Inconclusive { reason: "test" }
    }

    fn observe() -> VerifierConfig {
        VerifierConfig::default() // enforce = false
    }
    fn enforcing() -> VerifierConfig {
        VerifierConfig {
            enforce: true,
            min_consecutive_mismatches: 2,
            ..VerifierConfig::default()
        }
    }

    #[test]
    fn observe_mode_never_demotes_however_many_mismatches() {
        let cfg = observe();
        let mut streak = 0;
        for _ in 0..10 {
            let (s, action) = next_streak_and_action(&mismatch(), streak, &cfg);
            streak = s;
            assert_eq!(action, EnforcementAction::None);
        }
        assert_eq!(streak, 10); // streak still tracked for observability
    }

    #[test]
    fn single_mismatch_does_not_demote_when_enforcing() {
        let cfg = enforcing(); // needs 2
        let (streak, action) = next_streak_and_action(&mismatch(), 0, &cfg);
        assert_eq!(streak, 1);
        assert_eq!(action, EnforcementAction::None);
    }

    #[test]
    fn consecutive_mismatches_demote_at_threshold() {
        let cfg = enforcing();
        let (streak, action) = next_streak_and_action(&mismatch(), 1, &cfg);
        assert_eq!(streak, 2);
        assert_eq!(action, EnforcementAction::Demote);
    }

    #[test]
    fn inconclusive_never_advances_streak_or_demotes() {
        let cfg = enforcing();
        // One prior mismatch on record, then an inconclusive probe.
        let (streak, action) = next_streak_and_action(&inconclusive(), 1, &cfg);
        assert_eq!(streak, 1, "inconclusive must not advance the streak");
        assert_eq!(action, EnforcementAction::None);
    }

    #[test]
    fn match_resets_streak_and_reinstates_when_enforcing() {
        let cfg = enforcing();
        let (streak, action) = next_streak_and_action(&matched(), 5, &cfg);
        assert_eq!(streak, 0);
        assert_eq!(action, EnforcementAction::Reinstate);
    }

    #[test]
    fn match_in_observe_mode_resets_but_takes_no_routing_action() {
        let cfg = observe();
        let (streak, action) = next_streak_and_action(&matched(), 5, &cfg);
        assert_eq!(streak, 0);
        assert_eq!(action, EnforcementAction::None);
    }
}
