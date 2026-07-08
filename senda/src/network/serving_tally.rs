//! Persisted per-model completion-token tally over a rolling 7-day window.
//!
//! Powers the desktop dashboard's "estimated earnings this week" preview:
//! a contributor's runtime accumulates the completion tokens it actually
//! served (via the `record_completion` serving hook in `mesh::Node`),
//! bucketed by UTC day, and persists them to
//! `~/.senda/serving-tally.json` so the count survives the silent
//! ~6h auto-upgrade restarts that would otherwise reset an in-memory
//! counter on every release.
//!
//! This is an ESTIMATE input, not a ledger: there is no payment, no signed
//! receipt, and no gossip — the number never leaves the local node. It
//! exists so the install-and-share loop can show a contributor "this is
//! roughly what your machine served this week" against an illustrative
//! rate card, well before Phase 5's real credit ledger.
//!
//! Why per-model (not a single total): the eventual peer-payout rate card
//! is keyed by `(model, tier)`, so keeping tokens split by model lets the
//! frontend apply a per-tier rate without a runtime change later.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const TALLY_FILENAME: &str = "serving-tally.json";
/// Rolling window length. The dashboard renders this as "this week".
const WINDOW_DAYS: u64 = 7;
const SECS_PER_DAY: u64 = 24 * 60 * 60;
/// Don't write to disk more than once per this interval from the serving
/// hot path. A crash loses at most this much of the most recent burst —
/// acceptable for an estimate, and it keeps disk I/O off the per-request
/// path the rest of the time.
const FLUSH_INTERVAL_SECS: u64 = 60;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn today_bucket() -> u64 {
    now_secs() / SECS_PER_DAY
}

/// Mirror of `RoutingMetrics`' model-name normalization so the tally
/// ignores the same non-model sentinels the timing window does.
fn normalized_model(model: Option<&str>) -> Option<&str> {
    model.filter(|m| !m.is_empty() && *m != "auto")
}

#[derive(Default, Serialize, Deserialize)]
struct TallyState {
    /// `model -> (unix_day -> completion tokens served that day)`.
    by_model_day: HashMap<String, HashMap<u64, u64>>,
}

impl TallyState {
    /// Drop day buckets older than the rolling window and any model whose
    /// buckets are now all gone.
    fn prune(&mut self, today: u64) {
        let cutoff = today.saturating_sub(WINDOW_DAYS - 1);
        for days in self.by_model_day.values_mut() {
            days.retain(|&day, _| day >= cutoff);
        }
        self.by_model_day.retain(|_, days| !days.is_empty());
    }

    /// Per-model token totals within the rolling window. Models with a
    /// zero in-window total are omitted (missing = "served nothing this
    /// week", same convention as the other per-model status maps).
    fn window_totals(&self, today: u64) -> HashMap<String, u64> {
        let cutoff = today.saturating_sub(WINDOW_DAYS - 1);
        let mut out = HashMap::new();
        for (model, days) in &self.by_model_day {
            let sum: u64 = days
                .iter()
                .filter(|(&day, _)| day >= cutoff)
                .map(|(_, &tokens)| tokens)
                .sum();
            if sum > 0 {
                out.insert(model.clone(), sum);
            }
        }
        out
    }
}

/// Disk-persisted rolling tally of completion tokens served per model.
/// Cheap to clone via `Arc`; all interior state is behind a `Mutex`.
pub struct ServingTally {
    /// `None` disables persistence (used by lightweight/test constructions
    /// so they never touch `~/.senda`).
    path: Option<PathBuf>,
    state: Mutex<TallyState>,
    last_flush_secs: AtomicU64,
}

impl ServingTally {
    pub fn new(path: Option<PathBuf>) -> Self {
        let state = path.as_deref().map(load).unwrap_or_default();
        Self {
            path,
            state: Mutex::new(state),
            last_flush_secs: AtomicU64::new(0),
        }
    }

    /// Add `completion_tokens` served for `model` to today's bucket. No-ops
    /// for empty / `auto` model names and for zero tokens. Persists at most
    /// once per `FLUSH_INTERVAL_SECS` (the disk write happens outside the
    /// state lock).
    pub fn record(&self, model: Option<&str>, completion_tokens: u64) {
        let Some(model) = normalized_model(model) else {
            return;
        };
        if completion_tokens == 0 {
            return;
        }
        let today = today_bucket();
        let pending_json = {
            let mut state = self.state.lock().unwrap();
            *state
                .by_model_day
                .entry(model.to_string())
                .or_default()
                .entry(today)
                .or_insert(0) += completion_tokens;
            state.prune(today);
            let now = now_secs();
            let last = self.last_flush_secs.load(Ordering::Relaxed);
            if self.path.is_some() && now.saturating_sub(last) >= FLUSH_INTERVAL_SECS {
                self.last_flush_secs.store(now, Ordering::Relaxed);
                Some(serde_json::to_string_pretty(&*state).unwrap_or_else(|_| "{}".to_string()))
            } else {
                None
            }
        };
        if let (Some(json), Some(path)) = (pending_json, self.path.as_deref()) {
            let _ = write_atomic(path, &json);
        }
    }

    /// Per-model completion tokens served over the rolling 7-day window.
    /// Empty when nothing has been served. Consumed by `/api/status` for
    /// the desktop earnings-preview card.
    pub fn snapshot(&self) -> HashMap<String, u64> {
        let today = today_bucket();
        let mut state = self.state.lock().unwrap();
        state.prune(today);
        state.window_totals(today)
    }

    /// Best-effort force-persist of the current state. Safe to call from a
    /// shutdown path; a no-op when persistence is disabled.
    pub fn flush(&self) {
        let Some(path) = self.path.as_deref() else {
            return;
        };
        let json = {
            let state = self.state.lock().unwrap();
            serde_json::to_string_pretty(&*state).unwrap_or_else(|_| "{}".to_string())
        };
        let _ = write_atomic(path, &json);
    }
}

/// Path of the tally file (`~/.senda/serving-tally.json`). Honors
/// `SENDA_HOME` / `HOME` exactly like the native-baseline cache.
pub fn tally_path() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("SENDA_HOME") {
        return Some(PathBuf::from(custom).join(TALLY_FILENAME));
    }
    let home = dirs::home_dir()?;
    Some(home.join(".senda").join(TALLY_FILENAME))
}

fn load(path: &Path) -> TallyState {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
        Err(_) => TallyState::default(),
    }
}

fn write_atomic(path: &Path, json: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_accumulates_per_model_in_window() {
        let tally = ServingTally::new(None);
        tally.record(Some("Qwen3-8B-Q4_K_M"), 100);
        tally.record(Some("Qwen3-8B-Q4_K_M"), 50);
        tally.record(Some("Llama-3.1-8B-Instruct-Q4_K_M"), 20);
        let snap = tally.snapshot();
        assert_eq!(snap.get("Qwen3-8B-Q4_K_M"), Some(&150));
        assert_eq!(snap.get("Llama-3.1-8B-Instruct-Q4_K_M"), Some(&20));
    }

    #[test]
    fn record_ignores_auto_empty_and_zero() {
        let tally = ServingTally::new(None);
        tally.record(Some("auto"), 100);
        tally.record(Some(""), 100);
        tally.record(None, 100);
        tally.record(Some("Qwen3-8B-Q4_K_M"), 0);
        assert!(tally.snapshot().is_empty());
    }

    #[test]
    fn prune_drops_buckets_outside_window() {
        let today = today_bucket();
        let mut state = TallyState::default();
        let m = "Qwen3-8B-Q4_K_M".to_string();
        let days = state.by_model_day.entry(m.clone()).or_default();
        days.insert(today, 100); // in window
        days.insert(today - 3, 40); // in window (within 7 days)
        days.insert(today - 7, 999); // outside window (8th day back)
        days.insert(today - 30, 999); // long expired
                                      // window_totals must only sum in-window buckets.
        assert_eq!(state.window_totals(today).get(&m), Some(&140));
        // prune must physically drop the expired buckets.
        state.prune(today);
        let remaining = &state.by_model_day[&m];
        assert!(remaining.contains_key(&today));
        assert!(remaining.contains_key(&(today - 3)));
        assert!(!remaining.contains_key(&(today - 7)));
        assert!(!remaining.contains_key(&(today - 30)));
    }

    #[test]
    fn empty_model_pruned_when_all_buckets_expire() {
        let today = today_bucket();
        let mut state = TallyState::default();
        state
            .by_model_day
            .entry("old-model".to_string())
            .or_default()
            .insert(today - 100, 5);
        state.prune(today);
        assert!(state.by_model_day.is_empty());
    }

    #[test]
    fn persists_and_reloads_across_restart() {
        let dir = std::env::temp_dir().join(format!("ct-tally-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("serving-tally.json");

        let tally = ServingTally::new(Some(path.clone()));
        tally.record(Some("Qwen3-8B-Q4_K_M"), 321);
        tally.flush();

        // Simulate a restart: a fresh instance loads the persisted file.
        let reloaded = ServingTally::new(Some(path.clone()));
        assert_eq!(reloaded.snapshot().get("Qwen3-8B-Q4_K_M"), Some(&321));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
