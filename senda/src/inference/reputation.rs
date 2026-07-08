//! Phase 3.2 — entry-side reputation accumulator.
//!
//! The verifier loop in [`crate::inference::verify`] produces a
//! sample-and-verify verdict (`match` / `mismatch` / `inconclusive`) for a
//! `(peer, model)` pair on every audit tick, but until now those verdicts
//! only lived for an hour in `MeshState::verify_verdicts` — a single flaky
//! probe was indistinguishable from a peer that has reproduced the reference
//! fingerprint a hundred times, and every entry restart wiped the slate.
//!
//! This module turns that ephemeral verdict stream into a *persistent*,
//! time-weighted trust **score** per `(peer, model)`. Each conclusive verdict
//! is folded into an exponentially-weighted moving average so a peer accrues a
//! durable "trusted" signal across many independent probes, and a peer that
//! starts diverging is pulled down gradually rather than flipped by one tick.
//! The score survives restarts via `~/.senda/reputation.json`.
//!
//! **Observe-mode (Phase 3.2, not 3.3).** The score is surfaced on
//! `/api/status` for the catalog. It does **not** gate routing — that remains
//! the verifier's fast, flag-gated consecutive-mismatch demotion
//! (`SENDA_VERIFY_ENFORCE`). Reputation is the slow, accumulated companion
//! to that streak signal: the thing a future Phase 5 credit ledger weights
//! payouts by, and a future 3.3 lever can read.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

const STORE_FILENAME: &str = "reputation.json";

/// EWMA weight for the newest sample. 0.25 keeps the score stable across a
/// single flaky probe while still moving meaningfully over a handful of
/// consistent results (~5 samples to cross the trust boundary from a cold
/// start, more to recover from a mismatch).
pub const DEFAULT_ALPHA: f64 = 0.25;

/// Minimum conclusive samples folded before a score is allowed to read as
/// `Trusted` rather than `Unproven` — one or two matches shouldn't mint trust.
pub const MIN_SAMPLES_FOR_TRUST: u64 = 5;

/// Score at/above which a sufficiently-sampled peer reads as `Trusted`.
pub const TRUST_SCORE_THRESHOLD: f64 = 0.80;

/// Entries not updated within this window are pruned on load so the store file
/// stays bounded as peers come and go. 30 days is long enough to remember a
/// peer across a holiday but short enough to forget a one-off contributor.
const MAX_AGE_SECS: u64 = 30 * 24 * 3600;

/// In-memory store key: `(full peer id string, model id)`. The full
/// `EndpointId` string form is used (not `fmt_short`) so the key is stable and
/// collision-free across restarts; the status builder converts `peer.id` the
/// same way, so writes (verifier) and reads (status) always agree.
pub type RepKey = (String, String);

/// Accumulated trust for one `(peer, model)` pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReputationScore {
    /// EWMA of conclusive sample values (match = 1.0, mismatch = 0.0) in
    /// `[0, 1]`. Seeded by the first conclusive sample.
    pub score: f64,
    /// Conclusive samples folded so far (matches + mismatches). Inconclusive
    /// probes are counted separately and never move `score` or `samples`.
    pub samples: u64,
    pub matches: u64,
    pub mismatches: u64,
    pub inconclusive: u64,
    /// The most recent verdict string (`match` / `mismatch` / `inconclusive`).
    pub last_verdict: String,
    pub first_seen_unix_secs: u64,
    pub updated_at_unix_secs: u64,
}

impl ReputationScore {
    fn new(now: u64) -> Self {
        ReputationScore {
            score: 0.0,
            samples: 0,
            matches: 0,
            mismatches: 0,
            inconclusive: 0,
            last_verdict: String::new(),
            first_seen_unix_secs: now,
            updated_at_unix_secs: now,
        }
    }
}

/// Coarse trust grade derived from a score. Surfaced to the catalog so the UI
/// can render a chip without re-implementing the thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReputationGrade {
    /// Enough samples and a high score — independently verified over time.
    Trusted,
    /// Has produced at least one mismatch and hasn't recovered above the trust
    /// threshold. The peer to keep an eye on.
    Watch,
    /// Not enough conclusive samples yet to say either way.
    Unproven,
}

impl ReputationGrade {
    pub fn as_str(self) -> &'static str {
        match self {
            ReputationGrade::Trusted => "trusted",
            ReputationGrade::Watch => "watch",
            ReputationGrade::Unproven => "unproven",
        }
    }
}

/// Fold one verifier verdict into the prior score (or a fresh one). Pure and
/// unit-testable: no clock, no IO — the caller passes `now`.
///
/// - `match` / `mismatch`: seed `score` on the first conclusive sample, else
///   EWMA toward 1.0 / 0.0; increments `samples`.
/// - anything else (`inconclusive`): counted, but leaves `score` and `samples`
///   untouched — a probe that couldn't decide carries no trust signal.
pub fn fold(prev: Option<ReputationScore>, verdict: &str, now: u64, alpha: f64) -> ReputationScore {
    let mut s = prev.unwrap_or_else(|| ReputationScore::new(now));
    s.last_verdict = verdict.to_string();
    s.updated_at_unix_secs = now;
    match verdict {
        "match" => {
            s.score = if s.samples == 0 {
                1.0
            } else {
                alpha + (1.0 - alpha) * s.score
            };
            s.samples += 1;
            s.matches += 1;
        }
        "mismatch" => {
            s.score = if s.samples == 0 {
                0.0
            } else {
                (1.0 - alpha) * s.score
            };
            s.samples += 1;
            s.mismatches += 1;
        }
        _ => {
            s.inconclusive += 1;
        }
    }
    s
}

/// Derive the coarse [`ReputationGrade`] for a score.
pub fn grade(s: &ReputationScore) -> ReputationGrade {
    if s.samples >= MIN_SAMPLES_FOR_TRUST && s.score >= TRUST_SCORE_THRESHOLD {
        ReputationGrade::Trusted
    } else if s.mismatches > 0 {
        ReputationGrade::Watch
    } else {
        ReputationGrade::Unproven
    }
}

// ---- Persistence --------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedEntry {
    peer: String,
    model: String,
    #[serde(flatten)]
    score: ReputationScore,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PersistedStore {
    entries: Vec<PersistedEntry>,
}

/// Path of the reputation store file (`~/.senda/reputation.json`). Honors
/// `SENDA_HOME` / `HOME` the same way the native-baseline cache does.
pub fn store_path() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("SENDA_HOME") {
        return Some(PathBuf::from(custom).join(STORE_FILENAME));
    }
    let home = dirs::home_dir()?;
    Some(home.join(".senda").join(STORE_FILENAME))
}

/// Load the store into the in-memory map, pruning entries older than
/// [`MAX_AGE_SECS`]. Missing/corrupt file → empty map (never fails).
pub fn load_store(path: &std::path::Path, now: u64) -> HashMap<RepKey, ReputationScore> {
    let persisted: PersistedStore = match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
        Err(_) => PersistedStore::default(),
    };
    persisted
        .entries
        .into_iter()
        .filter(|e| now.saturating_sub(e.score.updated_at_unix_secs) <= MAX_AGE_SECS)
        .map(|e| ((e.peer, e.model), e.score))
        .collect()
}

/// Persist the map atomically (write tmp, rename), creating the parent dir if
/// needed. Entries are sorted by key for a stable, diff-friendly file.
pub fn save_store(
    path: &std::path::Path,
    map: &HashMap<RepKey, ReputationScore>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut entries: Vec<PersistedEntry> = map
        .iter()
        .map(|((peer, model), score)| PersistedEntry {
            peer: peer.clone(),
            model: model.clone(),
            score: score.clone(),
        })
        .collect();
    entries.sort_by(|a, b| (&a.peer, &a.model).cmp(&(&b.peer, &b.model)));
    let store = PersistedStore { entries };
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(&store).unwrap_or_else(|_| "{}".to_string());
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_match_seeds_full_trust_but_stays_unproven() {
        let s = fold(None, "match", 100, DEFAULT_ALPHA);
        assert_eq!(s.score, 1.0);
        assert_eq!(s.samples, 1);
        assert_eq!(s.matches, 1);
        // One sample is not enough to mint trust.
        assert_eq!(grade(&s), ReputationGrade::Unproven);
    }

    #[test]
    fn consistent_matches_reach_trusted() {
        let mut s = None;
        for t in 0..MIN_SAMPLES_FOR_TRUST {
            s = Some(fold(s, "match", t, DEFAULT_ALPHA));
        }
        let s = s.unwrap();
        assert_eq!(s.samples, MIN_SAMPLES_FOR_TRUST);
        assert!(s.score >= TRUST_SCORE_THRESHOLD);
        assert_eq!(grade(&s), ReputationGrade::Trusted);
    }

    #[test]
    fn first_mismatch_seeds_zero() {
        let s = fold(None, "mismatch", 1, DEFAULT_ALPHA);
        assert_eq!(s.score, 0.0);
        assert_eq!(s.mismatches, 1);
        assert_eq!(grade(&s), ReputationGrade::Watch);
    }

    #[test]
    fn mismatch_pulls_a_trusted_peer_down_to_watch() {
        // Build up trust, then a mismatch arrives.
        let mut s = None;
        for t in 0..10 {
            s = Some(fold(s, "match", t, DEFAULT_ALPHA));
        }
        let before = s.clone().unwrap();
        assert_eq!(grade(&before), ReputationGrade::Trusted);
        let after = fold(s, "mismatch", 11, DEFAULT_ALPHA);
        assert!(after.score < before.score);
        // A single mismatch off a long match streak shouldn't instantly drop
        // below the trust score, but the grade reflects the blemish.
        assert_eq!(grade(&after), ReputationGrade::Watch);
    }

    #[test]
    fn inconclusive_never_moves_score_or_samples() {
        let s = fold(None, "match", 1, DEFAULT_ALPHA);
        let s2 = fold(Some(s.clone()), "inconclusive", 2, DEFAULT_ALPHA);
        assert_eq!(s2.score, s.score);
        assert_eq!(s2.samples, s.samples);
        assert_eq!(s2.inconclusive, 1);
        assert_eq!(s2.last_verdict, "inconclusive");
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reputation.json");
        let mut map: HashMap<RepKey, ReputationScore> = HashMap::new();
        let s = fold(None, "match", 1000, DEFAULT_ALPHA);
        map.insert(("peerA".to_string(), "qwen3-8b".to_string()), s.clone());
        save_store(&path, &map).unwrap();

        let loaded = load_store(&path, 1001);
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded.get(&("peerA".to_string(), "qwen3-8b".to_string())),
            Some(&s)
        );
    }

    #[test]
    fn load_prunes_stale_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reputation.json");
        let mut map: HashMap<RepKey, ReputationScore> = HashMap::new();
        // updated long ago
        let old = fold(None, "match", 0, DEFAULT_ALPHA);
        map.insert(("old".to_string(), "m".to_string()), old);
        save_store(&path, &map).unwrap();

        // "now" is well past MAX_AGE_SECS → entry is dropped on load.
        let loaded = load_store(&path, MAX_AGE_SECS + 10);
        assert!(loaded.is_empty());
    }
}
