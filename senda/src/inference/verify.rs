//! Verification v1 — the comparison oracle for deterministic model-identity
//! fingerprints.
//!
//! Given a *reference* fingerprint (auditor-established ground truth for a
//! `(model, quant)` today; majority-consensus later) and a *candidate*
//! fingerprint produced by a peer that claims to serve the same model, decide
//! whether the candidate is plausibly that model run honestly.
//!
//! Why this shape:
//! - **Distributional classification of the first divergence.** The greedy
//!   decode of a fixed `temp=0 / seed=42` prompt is a strong model-identity
//!   signal: a different/smaller model, or canned text, diverges within the
//!   first few tokens. But even *honest* greedy decoding can diverge across
//!   Metal / CUDA / Vulkan when an early token sits on a near-tie and the
//!   backends' floating-point logits break it the other way — and because
//!   greedy decoding then conditions on a different prefix, that single flip
//!   *cascades* the rest of the sequence apart. Naive prefix-agreement
//!   therefore false-flagged honest peers ~half the time (see
//!   internal/RESILIENCE.md). Instead we walk to the *first* token divergence
//!   — up to which both sides decoded from an identical prefix, so the
//!   distributions are comparable — and classify it: an honest near-tie flip
//!   keeps each side's chosen token inside the *other* side's top-k candidate
//!   set, while a wrong model's token is absent from it. Past the first
//!   divergence nothing is comparable, so that one position is the verdict.
//! - **Top-k candidates, not logprob values.** `top_logprobs` is requested at
//!   `temp=0` purely to capture the per-position candidate *token sets*; the
//!   logprob magnitudes are not stored or compared (raising temperature to get
//!   richer values would make the token sequence itself less stable across
//!   backends, the opposite of what we want). Token identity within the top-k
//!   is the discriminator; the output hash backs it up when no prefix exists.
//!
//! Two probe modes (see `run_one_audit`):
//! - **Self-oracle (preferred).** When this node serves the model, each audit
//!   generates a fresh nonce-randomized probe, runs it on our own llama-server
//!   for ground truth, and sends the identical probe to the suspect. Because
//!   the probe is unpredictable, a peer can't detect "the probe" and serve the
//!   real model only for it — this closes the known-prompt spoof.
//! - **Fixed reference (fallback).** When we don't serve the model, the suspect
//!   is compared against a precomputed reference for the fixed probe. Spoofable
//!   by a peer that recognizes the known prompt, but still catches the common
//!   threat — wrong/smaller model, canned replies, misconfiguration.
//!
//! PRIVACY BOUNDARY (deliberate, do not "improve" by sampling real traffic):
//! verification only ever re-executes *synthetic* probes we generate. Sampling
//! real user requests and replaying them against a second node would be more
//! robust against a peer that fingerprints synthetic traffic — but it would
//! fan a user's private prompt out to a node that played no part in serving
//! the request, expanding plaintext exposure beyond the minimal serving path
//! (entry + the one host). That conflicts with the privacy promise, so it is
//! intentionally not done. The determined adversary who can statistically
//! distinguish synthetic probes from organic traffic is left to the later
//! staking / attestation layer, not to prompt snooping.
//!
//! Deferred: multi-peer consensus (proof-of-sampling) for models *no* verifier
//! serves locally, so the self-oracle's coverage isn't limited to served models.

use super::native_baseline::{self, FirstTokenProbe, LogitFingerprint};
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
    /// `Match`. **Legacy fallback only** — used when one side's fingerprint
    /// predates top-k capture (old bundled reference or pre-upgrade gossiped
    /// baseline). Fingerprints with top-k use the distributional first-
    /// divergence classifier, which ignores this. Below 1.0 to tolerate
    /// cross-backend tail divergence.
    pub min_prefix_agreement: f64,
    /// Need at least this many overlapping tokens to render a token verdict;
    /// fewer → `Inconclusive` (not enough signal).
    pub min_compared_tokens: usize,
    /// **Layer 1 gross-fraud gate.** Minimum fraction of battery probes whose
    /// first token is a *mutual near-tie* with the reference (each side's token
    /// sits in the other's top-k) for a `Match`. Calibrated honest peers score
    /// ~1.0 across backends and under load; canned output and wrong/much-smaller
    /// models score far below. Deliberately loose — Layer 1 catches only GROSS
    /// substitution; same-family downgrades (e.g. 4B-as-8B) are out of scope and
    /// left to the performance-profile layer (L4). See internal/RESILIENCE.md.
    pub min_battery_agreement: f64,
    /// Minimum battery probes that must return a usable first-token top-k on
    /// *both* sides to render a battery verdict; fewer → `Inconclusive`.
    pub min_battery_probes: usize,
}

impl Default for VerifyThresholds {
    fn default() -> Self {
        Self {
            // Calibrated against measured honest behavior, not a guess. The
            // daily driver serves with speculative decoding (`-md` draft +
            // continuous batching), which is not bit-reproducible run-to-run:
            // the deterministic *opening* of the answer (~the first sentence)
            // is rock-solid, but the model can branch at a later sentence
            // boundary, and a cold first request after boot sometimes takes a
            // different branch than the warm steady state. Measured over 8
            // captures of the same `(model, probe)`: the first 27/32 prefix
            // tokens were identical every time; one cold capture diverged for
            // the final 5, i.e. 0.844 agreement. A genuinely wrong/smaller
            // model or canned text diverges from the first token (agreement
            // ~0), so 0.75 cleanly separates honest (>=0.84) from dishonest
            // (~0) with margin for an even shorter deterministic opening
            // (24/32) on other backends. Raising this back toward 1.0 would
            // false-flag honest, freshly-booted peers.
            min_prefix_agreement: 0.75,
            min_compared_tokens: 8,
            // Honest first-token near-tie is 1.000 (idle, under load, and
            // cross-backend Metal/CUDA — measured, see RESILIENCE.md). A gross
            // substitution (different family / much smaller / canned) collapses
            // well below 0.5. 0.5 leaves a wide safety margin on the honest side
            // while still firing on anything genuinely fraudulent; it
            // intentionally does NOT try to separate a same-family 4B (which
            // also scores ~1.0 here — that is L4's job).
            min_battery_agreement: 0.5,
            // ~2/3 of the 16-probe battery must come back usable. Below that the
            // sample is too thin to convict; the audit is just inconclusive.
            min_battery_probes: 10,
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

    // Find the first position where the greedy decodes diverge. Up to here both
    // sides decoded from a byte-identical prefix, so position `d`'s next-token
    // distributions are directly comparable; *past* `d` the prefixes differ and
    // nothing downstream is comparable (this is the cascade that made naive
    // prefix-agreement false-flag honest peers — see internal/RESILIENCE.md).
    let first_div =
        (0..compared).find(|&i| reference.prefix_tokens[i] != candidate.prefix_tokens[i]);

    let Some(d) = first_div else {
        // Whole compared window agrees token-for-token — unambiguous match.
        return FingerprintVerdict::Match {
            prefix_agreement: 1.0,
            compared_tokens: compared,
        };
    };

    let prefix_agreement = d as f64 / compared as f64;

    // Distributional classification of the single comparable divergence. An
    // honest cross-backend near-tie flip keeps each side's chosen token inside
    // the *other* side's top-k at position `d` (both tokens were near the top
    // of both backends' logits; the FP tie just broke the other way). A
    // genuinely wrong/smaller model — or canned text — picks a token the real
    // model assigns negligible mass, so it's absent from the top-k. Needs top-k
    // on both sides; older fingerprints (no top-k) take the legacy gate below.
    if let (Some(ref_topk), Some(cand_topk)) =
        (reference.top_k_tokens.get(d), candidate.top_k_tokens.get(d))
    {
        if !ref_topk.is_empty() && !cand_topk.is_empty() {
            let ref_tok = &reference.prefix_tokens[d];
            let cand_tok = &candidate.prefix_tokens[d];
            let mutual_near_tie =
                cand_topk.iter().any(|t| t == ref_tok) && ref_topk.iter().any(|t| t == cand_tok);
            if mutual_near_tie {
                return FingerprintVerdict::Match {
                    prefix_agreement,
                    compared_tokens: compared,
                };
            }
            return FingerprintVerdict::Mismatch {
                prefix_agreement,
                compared_tokens: compared,
                reason: "greedy decode diverges and the divergent token is absent \
                         from the model's own top-k — a different/smaller model or \
                         canned output, not a cross-backend near-tie flip",
            };
        }
    }

    // Legacy fallback: at least one side predates top-k capture (old bundled
    // reference or pre-upgrade gossiped baseline). Fall back to the original
    // tolerant prefix-agreement gate over the whole window.
    let token_matches = (0..compared)
        .filter(|&i| reference.prefix_tokens[i] == candidate.prefix_tokens[i])
        .count();
    let legacy_agreement = token_matches as f64 / compared as f64;
    if legacy_agreement < thresholds.min_prefix_agreement {
        return FingerprintVerdict::Mismatch {
            prefix_agreement: legacy_agreement,
            compared_tokens: compared,
            reason: "greedy decode diverges from reference — likely a \
                     different/smaller model or canned output",
        };
    }
    FingerprintVerdict::Match {
        prefix_agreement: legacy_agreement,
        compared_tokens: compared,
    }
}

// ── Layer 1 — first-token top-k battery oracle (gross-fraud) ─────────────────
//
// The preferred verdict path. Instead of comparing a long greedy-decode prefix
// (which on the bundled b9109 stack reads cross-backend drift + load jitter as
// divergence — see `compare_fingerprints` and RESILIENCE.md), it compares the
// *first token's top-k candidate set* over a battery of short prompts, captured
// via native `/completion` where first-token top-k is always present. Honest
// peers are a ~100 % mutual near-tie idle, under load, and across backends;
// gross fraud collapses far below the gate. Scoped to GROSS substitution by
// design — see `VerifyThresholds::min_battery_agreement`.

/// Compare a candidate's per-prompt first-token probes against the reference
/// battery, index-aligned (`reference[i]` and `candidate[i]` are the same
/// `VERIFY_BATTERY` prompt; `None` candidate entries are dropped probes). A
/// position counts toward agreement when it is a *mutual near-tie*: the
/// candidate's first token is in the reference's top-k AND vice versa. Pure +
/// deterministic so it is unit-testable without a network.
pub fn battery_verdict(
    reference: &[FirstTokenProbe],
    candidate: &[Option<FirstTokenProbe>],
    thresholds: &VerifyThresholds,
) -> FingerprintVerdict {
    let mut compared = 0usize;
    let mut near_tie = 0usize;
    for (r, c) in reference.iter().zip(candidate.iter()) {
        let Some(c) = c else { continue };
        if r.top_k.is_empty() || c.top_k.is_empty() {
            continue; // no candidate set on one side — not comparable
        }
        compared += 1;
        let mutual = c.top_k.iter().any(|t| t == &r.token) && r.top_k.iter().any(|t| t == &c.token);
        if mutual {
            near_tie += 1;
        }
    }

    if compared < thresholds.min_battery_probes {
        return FingerprintVerdict::Inconclusive {
            reason: "too few battery probes returned usable first-token top-k to judge",
        };
    }

    let agreement = near_tie as f64 / compared as f64;
    if agreement < thresholds.min_battery_agreement {
        return FingerprintVerdict::Mismatch {
            prefix_agreement: agreement,
            compared_tokens: compared,
            reason: "first-token top-k battery agreement far below the honest floor — \
                     gross substitution (wrong/much-smaller model or canned output)",
        };
    }
    FingerprintVerdict::Match {
        prefix_agreement: agreement,
        compared_tokens: compared,
    }
}

// ── Verifier loop ────────────────────────────────────────────────────────────
//
// Observe by default, enforce only when explicitly enabled. The loop probes
// peers, runs the oracle, and logs verdicts. Demotion is the one
// irreversible-ish lever (a false positive punishes an honest contributor), so
// it is gated three ways: (1) off unless `SENDA_VERIFY_ENFORCE` is set,
// (2) requires several *consecutive* `Mismatch` verdicts for the same
// `(peer, model)` before acting, never a single flaky probe, and (3) the
// action is a reversible, time-boxed route demotion — the peer stays in the
// mesh, keeps being re-probed, and is reinstated on the next `Match` or when
// the cooldown lapses. `Inconclusive` never counts toward conviction.

/// Set to a truthy value (`1`/`true`/`yes`/`on`) to let the verifier demote
/// convicted peers instead of only logging.
const ENFORCE_ENV: &str = "SENDA_VERIFY_ENFORCE";

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
/// 2. **On-disk override** at `~/.senda/reference-fingerprints.json`,
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

/// Layer-1 reference batteries (first-token top-k per `VERIFY_BATTERY` prompt),
/// keyed by model name. Same two-layer shape as [`ReferenceStore`]: embedded in
/// the binary so a GPU-less entry router has them with zero config, plus an
/// on-disk override at `~/.senda/reference-batteries.json` (what
/// `benchmark capture-reference-battery` writes) that extends/replaces the
/// shipped set without a rebuild. Entries are index-aligned to `VERIFY_BATTERY`,
/// so the battery list and the captures must be regenerated together.
#[derive(Debug, Clone, Default)]
pub struct ReferenceBatteryStore {
    refs: HashMap<String, Vec<FirstTokenProbe>>,
}

/// Reference batteries baked into the binary at build time.
const EMBEDDED_BATTERIES: &str = include_str!("reference_batteries.json");

impl ReferenceBatteryStore {
    pub fn load() -> Self {
        let mut refs: HashMap<String, Vec<FirstTokenProbe>> =
            serde_json::from_str(EMBEDDED_BATTERIES).unwrap_or_default();
        if let Some(path) = battery_reference_path() {
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Ok(disk) =
                    serde_json::from_str::<HashMap<String, Vec<FirstTokenProbe>>>(&raw)
                {
                    refs.extend(disk);
                }
            }
        }
        Self { refs }
    }

    pub fn get(&self, model: &str) -> Option<&Vec<FirstTokenProbe>> {
        self.refs.get(model)
    }
}

pub(crate) fn battery_reference_path() -> Option<std::path::PathBuf> {
    if let Ok(custom) = std::env::var("SENDA_HOME") {
        return Some(std::path::PathBuf::from(custom).join("reference-batteries.json"));
    }
    dirs::home_dir().map(|h| h.join(".senda").join("reference-batteries.json"))
}

/// Insert or replace one model's reference battery in the on-disk store and
/// persist it (pretty JSON). Returns the path written. Used by
/// `benchmark capture-reference-battery`.
pub(crate) fn upsert_reference_battery(
    model: &str,
    battery: &[FirstTokenProbe],
) -> anyhow::Result<std::path::PathBuf> {
    let path = battery_reference_path()
        .ok_or_else(|| anyhow::anyhow!("could not resolve reference-batteries.json path"))?;
    let mut store = ReferenceBatteryStore::load();
    store.refs.insert(model.to_string(), battery.to_vec());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, serde_json::to_string_pretty(&store.refs)?)?;
    Ok(path)
}

pub(crate) fn reference_path() -> Option<std::path::PathBuf> {
    if let Ok(custom) = std::env::var("SENDA_HOME") {
        return Some(std::path::PathBuf::from(custom).join("reference-fingerprints.json"));
    }
    dirs::home_dir().map(|h| h.join(".senda").join("reference-fingerprints.json"))
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
/// Observe-only unless `SENDA_VERIFY_ENFORCE` is set, in which case
/// repeatedly-convicted peers are demoted from routing for a cooldown.
pub fn spawn_verifier(node: mesh::Node, mut config: VerifierConfig) {
    if enforce_from_env() {
        config.enforce = true;
    }
    config.min_consecutive_mismatches = config.min_consecutive_mismatches.max(1);
    tokio::spawn(async move {
        tokio::time::sleep(config.settle).await;
        let store = ReferenceStore::load();
        let battery_store = ReferenceBatteryStore::load();
        tracing::info!(
            target: "senda::verify",
            shipped_refs = store.refs.len(),
            shipped_batteries = battery_store.refs.len(),
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
            // L4 (observe-only): cheap, probe-free perf-profile consistency over
            // all peers' gossiped telemetry. Runs every tick regardless of
            // whether L1 has an auditable reference for any peer.
            super::perf_profile::observe_peers(&node).await;
            if let Err(e) =
                run_one_audit(&node, &store, &battery_store, &config, &mut streaks).await
            {
                tracing::debug!(target: "senda::verify", "audit tick skipped: {e}");
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
    battery_store: &ReferenceBatteryStore,
    config: &VerifierConfig,
    streaks: &mut HashMap<(EndpointId, String), u32>,
) -> anyhow::Result<()> {
    let local_id = node.id();
    let self_refs = self_baseline_fingerprints(node).await;
    let local_ports = node.local_model_ports_snapshot().await;

    // (peer, model) pairs we can establish ground truth for — either we serve
    // the model (self-oracle) or we hold a fixed reference for it. A battery
    // reference (Layer 1) or a fingerprint reference (legacy) both qualify.
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
                || battery_store.get(&model).is_some()
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

    // ── Layer 1 (preferred): first-token top-k battery, gross-fraud scoped. ──
    // Ground truth is our own model when we serve it (same-hardware capture),
    // else the embedded/on-disk reference battery. A decisive verdict short-
    // circuits; an `Inconclusive` (peer returned too few probes) falls through
    // to the legacy fingerprint oracle below.
    let ref_battery: Option<(Vec<FirstTokenProbe>, &'static str)> =
        if let Some(&port) = local_ports.get(&model) {
            native_baseline::capture_reference_battery(port)
                .await
                .ok()
                .map(|b| (b, "battery_self"))
        } else {
            battery_store
                .get(&model)
                .cloned()
                .map(|b| (b, "battery_fixed"))
        };
    if let Some((reference, mode)) = ref_battery {
        // Cap per-probe wait so a stalled peer can't drag one tick across the
        // whole battery; a single decoded token is fast on a live server.
        let per_probe = config.probe_timeout.min(Duration::from_secs(20));
        let candidate = remote_first_token_battery(node, peer_id, per_probe).await;
        let verdict = battery_verdict(&reference, &candidate, &config.thresholds);
        if !matches!(verdict, FingerprintVerdict::Inconclusive { .. }) {
            log_verdict(peer_id, &model, mode, &verdict);
            apply_enforcement(node, peer_id, &model, mode, &verdict, config, streaks).await;
            return Ok(());
        }
    }

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
    apply_enforcement(node, peer_id, &model, mode, &verdict, config, streaks).await;
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
    mode: &str,
    verdict: &FingerprintVerdict,
    config: &VerifierConfig,
    streaks: &mut HashMap<(EndpointId, String), u32>,
) {
    // Record the verdict for the status API on every tick, independent of
    // enforcement, so observe-only meshes still surface "independently
    // verified" / "failed verification".
    let (vk, agreement, compared, reason): (&str, f64, usize, Option<&str>) = match verdict {
        FingerprintVerdict::Match {
            prefix_agreement,
            compared_tokens,
        } => ("match", *prefix_agreement, *compared_tokens, None),
        FingerprintVerdict::Mismatch {
            prefix_agreement,
            compared_tokens,
            reason,
        } => (
            "mismatch",
            *prefix_agreement,
            *compared_tokens,
            Some(*reason),
        ),
        FingerprintVerdict::Inconclusive { reason } => ("inconclusive", 0.0, 0, Some(*reason)),
    };
    node.record_verify_verdict(peer_id, model, vk, agreement, compared, mode, reason)
        .await;

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
                target: "senda::verify",
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
        // Per-position candidate sets so the oracle can classify the first
        // divergence as an honest cross-backend near-tie flip vs a wrong model.
        // Driven by our request, so an honest peer on any backend returns them
        // even if its own baseline predates top-k capture.
        "top_logprobs": native_baseline::TOP_K_PER_POSITION,
        // Don't let the peer recurse into its own consultation hooks.
        "mesh_hooks": false,
        // Pin thinking off on both sides of the comparison so the audit is
        // apples-to-apples regardless of the suspect's server launch flags
        // (matches the baseline/reference probes; honest llama.cpp peers honor
        // it, and a peer that ignores it diverging is itself a signal).
        "chat_template_kwargs": {"enable_thinking": false},
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

/// Capture a peer's first-token battery over the mesh QUIC tunnel: one native
/// `/completion` probe per `VERIFY_BATTERY` prompt, index-aligned with the
/// reference (`None` for a dropped/failed probe). The peer's inbound HTTP
/// tunnel forwards to its elected-model backend proxy regardless of path, so
/// `/completion` reaches the same llama-server the chat probe would.
async fn remote_first_token_battery(
    node: &mesh::Node,
    peer_id: EndpointId,
    per_probe_timeout: Duration,
) -> Vec<Option<FirstTokenProbe>> {
    let mut out = Vec::with_capacity(native_baseline::VERIFY_BATTERY.len());
    for prompt in native_baseline::VERIFY_BATTERY {
        let probe =
            tokio::time::timeout(per_probe_timeout, remote_first_token(node, peer_id, prompt))
                .await
                .ok()
                .and_then(|r| r.ok());
        out.push(probe);
    }
    out
}

/// Send one battery prompt to a peer's native `/completion` over the tunnel and
/// parse its first-token top-k. Mirrors [`remote_probe_fingerprint`]'s raw-HTTP
/// framing; the body is the shared [`native_baseline::first_token_request_body`]
/// so both sides issue a byte-identical request.
async fn remote_first_token(
    node: &mesh::Node,
    peer_id: EndpointId,
    prompt: &str,
) -> anyhow::Result<FirstTokenProbe> {
    let body_bytes = serde_json::to_vec(&native_baseline::first_token_request_body(prompt))?;
    let http_request = format!(
        "POST /completion HTTP/1.1\r\n\
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
        .map_err(|e| anyhow::anyhow!("failed to parse peer first-token response: {e}"))?;
    native_baseline::first_token_from_completion_json(&parsed)
        .ok_or_else(|| anyhow::anyhow!("peer first-token response had no usable output"))
}

fn log_verdict(peer_id: EndpointId, model: &str, mode: &str, verdict: &FingerprintVerdict) {
    match verdict {
        FingerprintVerdict::Match {
            prefix_agreement,
            compared_tokens,
        } => {
            tracing::info!(
                target: "senda::verify",
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
                target: "senda::verify",
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
                target: "senda::verify",
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

    /// Legacy fingerprint (no top-k) — exercises the prefix-agreement fallback.
    fn fp(tokens: &[&str], hash: &str) -> LogitFingerprint {
        LogitFingerprint {
            token_count: tokens.len() as u32,
            output_sha256: hash.to_string(),
            prefix_tokens: tokens.iter().map(|t| t.to_string()).collect(),
            top_k_tokens: Vec::new(),
        }
    }

    /// Fingerprint with per-position top-k candidate sets — exercises the
    /// distributional first-divergence classifier.
    fn fp_tk(tokens: &[&str], top_k: &[&[&str]], hash: &str) -> LogitFingerprint {
        LogitFingerprint {
            token_count: tokens.len() as u32,
            output_sha256: hash.to_string(),
            prefix_tokens: tokens.iter().map(|t| t.to_string()).collect(),
            top_k_tokens: top_k
                .iter()
                .map(|row| row.iter().map(|t| t.to_string()).collect())
                .collect(),
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
    fn is_inconclusive(v: &FingerprintVerdict) -> bool {
        matches!(v, FingerprintVerdict::Inconclusive { .. })
    }

    fn probe(token: &str, top_k: &[&str]) -> FirstTokenProbe {
        FirstTokenProbe {
            token: token.to_string(),
            top_k: top_k.iter().map(|t| t.to_string()).collect(),
        }
    }

    /// A 12-prompt honest reference: each first token sits at the top of its
    /// own top-k.
    fn honest_reference() -> Vec<FirstTokenProbe> {
        (0..12)
            .map(|i| {
                let t = format!(" t{i}");
                FirstTokenProbe {
                    token: t.clone(),
                    top_k: vec![t, format!(" alt{i}"), " __".to_string()],
                }
            })
            .collect()
    }

    #[test]
    fn battery_identical_is_a_clean_match() {
        let r = honest_reference();
        let cand: Vec<Option<FirstTokenProbe>> = r.iter().cloned().map(Some).collect();
        let v = battery_verdict(&r, &cand, &VerifyThresholds::default());
        assert!(is_match(&v));
        if let FingerprintVerdict::Match {
            prefix_agreement, ..
        } = v
        {
            assert!((prefix_agreement - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn battery_tolerates_cross_backend_near_tie_flips() {
        // Every position flips to the reference's *second* candidate, but the
        // reference token is still in the candidate's top-k and vice versa — a
        // mutual near-tie, exactly the honest cross-backend case.
        let r = honest_reference();
        let cand: Vec<Option<FirstTokenProbe>> = r
            .iter()
            .map(|p| {
                let flipped = p.top_k[1].clone();
                Some(FirstTokenProbe {
                    token: flipped,
                    top_k: p.top_k.clone(), // shares the reference token, so mutual
                })
            })
            .collect();
        let v = battery_verdict(&r, &cand, &VerifyThresholds::default());
        assert!(
            is_match(&v),
            "honest near-tie flips must not convict: {v:?}"
        );
    }

    #[test]
    fn battery_catches_gross_substitution() {
        // A wrong model: every chosen token is absent from the reference's
        // top-k and its own top-k excludes the reference token.
        let r = honest_reference();
        let cand: Vec<Option<FirstTokenProbe>> = (0..12)
            .map(|i| Some(probe(&format!(" x{i}"), &[&format!(" x{i}"), " y", " z"])))
            .collect();
        let v = battery_verdict(&r, &cand, &VerifyThresholds::default());
        assert!(is_mismatch(&v), "gross substitution must convict: {v:?}");
    }

    #[test]
    fn battery_too_few_probes_is_inconclusive() {
        // Only 5 probes returned (< min_battery_probes default 10) → no verdict,
        // never a demotion.
        let r = honest_reference();
        let mut cand: Vec<Option<FirstTokenProbe>> = r.iter().cloned().map(Some).collect();
        for slot in cand.iter_mut().skip(5) {
            *slot = None;
        }
        let v = battery_verdict(&r, &cand, &VerifyThresholds::default());
        assert!(is_inconclusive(&v));
    }

    #[test]
    fn battery_empty_top_k_positions_are_skipped_not_counted() {
        // A position with no candidate set on the suspect side is not
        // comparable and must not count toward the total — otherwise a backend
        // that drops some top-k would dilute agreement.
        let r = honest_reference();
        let cand: Vec<Option<FirstTokenProbe>> = r
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i % 2 == 0 {
                    Some(p.clone())
                } else {
                    Some(probe(&p.token, &[])) // present token, empty top-k
                }
            })
            .collect();
        // 6 comparable positions, all matching → still inconclusive (< 10), and
        // crucially not a mismatch from diluted agreement.
        let v = battery_verdict(&r, &cand, &VerifyThresholds::default());
        assert!(
            is_inconclusive(&v),
            "empty-top-k positions must be skipped: {v:?}"
        );
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
    fn two_tokens_off_is_within_budget() {
        let r = fp(&ten(), "h1");
        let mut toks = ten();
        toks[5] = " rug";
        toks[7] = " later"; // 8/10 agree = 0.8 >= 0.75 threshold
        let c = fp(&toks, "h2");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(is_match(&v), "8/10 agreement should pass the 0.75 gate");
    }

    #[test]
    fn three_tokens_off_breaks_budget() {
        let r = fp(&ten(), "h1");
        let mut toks = ten();
        toks[5] = " rug";
        toks[7] = " later";
        toks[9] = " nope"; // 7/10 = 0.7 < 0.75
        let c = fp(&toks, "h2");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(is_mismatch(&v), "7/10 agreement should fail the 0.75 gate");
    }

    // ── distributional (top-k) first-divergence classifier ───────────────
    //
    // These exercise the actual cross-hardware fix: a single early divergence
    // that is a near-tie flip (the other side's token is in this position's
    // top-k) is honest; one whose token is absent from the top-k is a wrong
    // model. Regression guard for the ~52% false-positive bug documented in
    // internal/RESILIENCE.md.

    #[test]
    fn early_near_tie_flip_matches_despite_low_prefix_agreement() {
        // Honest cross-backend: diverges at token 1 (so naive prefix agreement
        // is ~0.1 and the old gate would FALSE-FLAG), but the flipped tokens are
        // mutually in each other's top-k → near-tie → Match.
        let ref_toks = ten();
        let mut cand = ten();
        cand[1] = " kitten"; // flip at position 1
                             // Both honest backends rank " cat" and " kitten" near the top at the
                             // tie position; every other position is confident on its single token.
        let mut rows: Vec<Vec<&str>> = ref_toks.iter().map(|t| vec![*t]).collect();
        rows[1] = vec![" cat", " kitten", "_pad"];
        let rrows: Vec<&[&str]> = rows.iter().map(|r| r.as_slice()).collect();
        let r = fp_tk(&ref_toks, &rrows, "h1");
        let c = fp_tk(&cand, &rrows, "h2");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(
            is_match(&v),
            "an early near-tie flip must NOT demote an honest peer: {v:?}"
        );
    }

    #[test]
    fn early_divergence_not_in_topk_mismatches() {
        // Wrong model: diverges at token 1 with a token the real model never
        // ranks (not in top-k) → Mismatch even though only one token differs.
        let ref_toks = ten();
        let mut cand = ten();
        cand[1] = " perro"; // wrong-model token, absent from the real top-k
                            // top-k rows: the real model's candidates never include " perro".
        let rows: Vec<Vec<&str>> = ref_toks
            .iter()
            .map(|t| vec![*t, "_alt1", "_alt2", "_alt3", "_alt4"])
            .collect();
        let rrows: Vec<&[&str]> = rows.iter().map(|r| r.as_slice()).collect();
        let r = fp_tk(&ref_toks, &rrows, "h1");
        let c = fp_tk(&cand, &rrows, "h2");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(
            is_mismatch(&v),
            "a divergent token absent from top-k is a wrong model: {v:?}"
        );
    }

    #[test]
    fn topk_full_agreement_matches() {
        let toks = ten();
        let rows: Vec<Vec<&str>> = toks.iter().map(|t| vec![*t, "_alt1", "_alt2"]).collect();
        let rrows: Vec<&[&str]> = rows.iter().map(|r| r.as_slice()).collect();
        let r = fp_tk(&toks, &rrows, "h1");
        let v = compare_fingerprints(&r, &r, &VerifyThresholds::default());
        assert!(is_match(&v));
    }

    #[test]
    fn one_sided_topk_falls_back_to_prefix_gate() {
        // Candidate has top-k but reference predates it: must not crash or use
        // the near-tie path; falls back to the prefix-agreement gate. 7/10
        // agreement < 0.75 → Mismatch.
        let r = fp(&ten(), "h1"); // legacy, no top-k
        let mut toks = ten();
        toks[5] = " rug";
        toks[7] = " later";
        toks[9] = " nope"; // 7/10 = 0.7
        let rows: Vec<Vec<&str>> = toks.iter().map(|t| vec![*t, "_x"]).collect();
        let rrows: Vec<&[&str]> = rows.iter().map(|r| r.as_slice()).collect();
        let c = fp_tk(&toks, &rrows, "h2");
        let v = compare_fingerprints(&r, &c, &VerifyThresholds::default());
        assert!(is_mismatch(&v), "one-sided top-k must use the legacy gate");
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
        // Embedded reference must carry per-position top-k so the entry-node
        // verifier (fixed_reference mode) uses the distributional classifier
        // rather than falling back to the legacy prefix gate.
        assert_eq!(fp.top_k_tokens.len(), 32);
        assert!(fp.top_k_tokens.iter().all(|row| !row.is_empty()));
    }

    #[test]
    fn embedded_battery_parses_and_aligns_with_the_probe_battery() {
        let refs: HashMap<String, Vec<FirstTokenProbe>> = serde_json::from_str(EMBEDDED_BATTERIES)
            .expect("embedded reference_batteries.json must be valid JSON");
        let battery = refs
            .get("Qwen3-8B-Q4_K_M")
            .expect("embedded batteries must include the canonical daily driver");
        // Index-aligned with VERIFY_BATTERY — a mismatch means the battery was
        // changed without recapturing the reference.
        assert_eq!(battery.len(), native_baseline::VERIFY_BATTERY.len());
        // Every probe must carry a non-empty top-k or the gross-fraud gate has
        // nothing to compare on that position.
        assert!(battery
            .iter()
            .all(|p| !p.token.is_empty() && !p.top_k.is_empty()));
        // The reference battery must clear its own gate against itself with a
        // wide margin (sanity: the embedded data isn't degenerate).
        let cand: Vec<Option<FirstTokenProbe>> = battery.iter().cloned().map(Some).collect();
        let v = battery_verdict(battery, &cand, &VerifyThresholds::default());
        assert!(is_match(&v));
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
