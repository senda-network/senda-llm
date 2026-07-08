//! Layer 4 — performance-profile consistency (observe-only).
//!
//! Catches the same-family substitution that output-matching (L1) deliberately
//! doesn't: a peer that serves a smaller/cheaper model (e.g. a 4B) while
//! claiming a bigger one (an 8B). Such a peer decodes *faster* than the claimed
//! model physically can.
//!
//! The check is a hard physical bound, not a learned threshold. Autoregressive
//! decode is memory-bandwidth bound: every token streams the whole weight set
//! through memory once, so a dense model of `S` bytes cannot exceed
//! `bandwidth / S` tokens/sec — regardless of how fast the GPU's compute is. We
//! already gossip each peer's *measured* memory bandwidth (the startup
//! benchmark, `gpu_mem_bandwidth_gbps`) and the claimed model's byte size is in
//! the catalog, so we can compute the largest model that *could* sustain the
//! peer's observed `native_tps`:
//!
//!     implied_max_bytes = bandwidth_bytes_per_sec / observed_tps
//!
//! and compare it to the claimed size. Honest peers sit comfortably *above* the
//! claimed size — real decode reaches only ~60–80% of the memcpy-benchmark
//! bandwidth, so `implied_max_bytes` lands ~1.3–1.7x the true size. A peer
//! whose throughput implies a model well *below* what it claims is serving
//! something smaller. Flagging at `implied < FLAG_FRACTION x claimed` cleanly
//! separates an honest model (implied ~1.4x) from a 2x-smaller substitute
//! (implied ~0.7–0.9x).
//!
//! Detection floor (be honest about it): the bound flags a 4B-as-8B only when
//! the fake's *absolute* rate exceeds the claimed model's ceiling, which needs
//! inference to reach ≳0.6 of memcpy bandwidth — true for modern CUDA/Metal
//! llama.cpp (typically 0.6–0.8), but at very low efficiency a slow 4B can hide
//! inside an 8B's plausible range. The asymmetry that matters: an honest peer's
//! implied fraction is always `1/efficiency > 1` (it decodes *below* its own
//! ceiling), so it is **never** false-flagged regardless of efficiency. This
//! layer trades detection sensitivity for zero false positives by design.
//!
//! Scope and honesty:
//! - **Observe-only.** This logs a signal; it never demotes. The bound relies
//!   on a *measured* bandwidth we don't independently verify, so it informs
//!   reputation / cross-checks rather than convicting on its own. It depends on
//!   a trustworthy `native_tps`, which is why the baseline collector idle-gates
//!   its samples (see `native_baseline`); a contended sample reads low and
//!   would only ever *suppress* a flag, never manufacture one.
//! - **Dense models only.** MoE decodes only its *active* experts, not the full
//!   file, so file size badly overstates per-token reads — we skip MoE (the
//!   catalog flags it) to avoid false positives.
//! - **One-sided.** We only flag *too fast* (smaller-than-claimed). Too slow is
//!   ambiguous (load, throttling, a genuinely bigger model) and not a fraud
//!   signal here.

use crate::mesh;
use crate::models::catalog;

/// Flag when the throughput-implied model size falls below this fraction of the
/// claimed catalog size. The gap between honest peers (implied ~1.3x+, since
/// inference reaches only a fraction of memcpy bandwidth) and a 2x-smaller
/// substitute (implied ~0.7–0.9x) is wide; `0.9` sits in that gap with margin
/// for bandwidth-measurement noise.
const FLAG_FRACTION: f64 = 0.9;

/// Need at least this many timing samples behind a gossiped baseline before we
/// read anything into it.
const MIN_SAMPLES: u32 = 2;

/// Ignore baselines older than this — a stale figure may predate a model or
/// hardware change. Matches the collector's clean-sample TTL.
const MAX_BASELINE_AGE_SECS: u64 = 24 * 60 * 60;

/// Outcome of one perf-profile consistency check.
#[derive(Debug, Clone, PartialEq)]
pub enum PerfVerdict {
    /// Throughput is consistent with (or slower than) the claimed model.
    Consistent { implied_fraction: f64 },
    /// Throughput is too high for the claimed model on this measured bandwidth —
    /// the peer is likely serving something smaller than it claims.
    Suspicious {
        implied_fraction: f64,
        observed_tps: f64,
        ceiling_tps: f64,
    },
    /// Not enough trustworthy data to judge.
    Inconclusive { reason: &'static str },
}

/// Classify one (observed throughput, measured bandwidth, claimed size) triple.
/// Pure and side-effect-free so the physical bound is unit-testable without a
/// node. `bandwidth_gbps` is the peer's measured memory bandwidth in GB/s
/// (10^9 bytes/s); `claimed_model_bytes` is the catalog size of the model the
/// peer claims to serve.
pub fn classify(
    observed_tps: f64,
    bandwidth_gbps: f64,
    claimed_model_bytes: u64,
    samples: u32,
) -> PerfVerdict {
    if samples < MIN_SAMPLES {
        return PerfVerdict::Inconclusive {
            reason: "too few baseline samples",
        };
    }
    if !observed_tps.is_finite() || observed_tps <= 0.0 {
        return PerfVerdict::Inconclusive {
            reason: "no observed throughput",
        };
    }
    if !bandwidth_gbps.is_finite() || bandwidth_gbps <= 0.0 {
        return PerfVerdict::Inconclusive {
            reason: "no measured memory bandwidth",
        };
    }
    if claimed_model_bytes == 0 {
        return PerfVerdict::Inconclusive {
            reason: "unknown claimed model size",
        };
    }

    let bandwidth_bytes_per_sec = bandwidth_gbps * 1e9;
    // Hard physical ceiling: a model of `claimed_model_bytes` can't decode
    // faster than this on the measured bandwidth (one full weight read/token).
    let ceiling_tps = bandwidth_bytes_per_sec / claimed_model_bytes as f64;
    // Largest model that could sustain the observed rate at that ceiling.
    let implied_max_bytes = bandwidth_bytes_per_sec / observed_tps;
    let implied_fraction = implied_max_bytes / claimed_model_bytes as f64;

    if implied_fraction < FLAG_FRACTION {
        PerfVerdict::Suspicious {
            implied_fraction,
            observed_tps,
            ceiling_tps,
        }
    } else {
        PerfVerdict::Consistent { implied_fraction }
    }
}

/// Run the L4 perf-profile check over every peer's gossiped baselines and log
/// the verdicts. Observe-only — never touches routing. Cheap (no probing): it
/// reads only already-gossiped telemetry, so it's safe to call every audit tick.
pub async fn observe_peers(node: &mesh::Node) {
    let local_id = node.id();
    let now = now_secs();
    for p in node.peers().await {
        if p.id == local_id {
            continue; // never audit ourselves
        }
        let bandwidth_gbps = match parse_total_bandwidth_gbps(p.gpu_mem_bandwidth_gbps.as_deref()) {
            Some(bw) => bw,
            None => continue, // no measured bandwidth → nothing to bound against
        };
        for b in &p.native_baselines {
            // Dense models only; MoE active-param decode breaks the file-size bound.
            let Some(model) = catalog::find_model(&b.model) else {
                continue;
            };
            if model.moe.is_some() {
                continue;
            }
            if now.saturating_sub(b.measured_at_unix_secs) > MAX_BASELINE_AGE_SECS {
                continue;
            }
            let claimed_bytes = (catalog::parse_size_gb(&model.size) * 1e9) as u64;
            let verdict = classify(b.native_tps_p50, bandwidth_gbps, claimed_bytes, b.samples);
            log_verdict(p.id, &b.model, &verdict);
        }
    }
}

/// Sum the per-GPU measured memory bandwidths from the gossiped CSV string.
/// Summing is the conservative direction here: it can only *raise* the ceiling,
/// making us *less* likely to flag, so a multi-GPU peer is never falsely
/// accused by over-counting bandwidth.
fn parse_total_bandwidth_gbps(csv: Option<&str>) -> Option<f64> {
    let csv = csv?;
    let total: f64 = csv
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .sum();
    (total > 0.0).then_some(total)
}

fn log_verdict(peer: iroh::EndpointId, model: &str, verdict: &PerfVerdict) {
    match verdict {
        PerfVerdict::Suspicious {
            implied_fraction,
            observed_tps,
            ceiling_tps,
        } => {
            tracing::warn!(
                target: "senda::verify::l4",
                peer = %peer.fmt_short(),
                %model,
                observed_tps = format!("{observed_tps:.1}"),
                ceiling_tps = format!("{ceiling_tps:.1}"),
                implied_fraction = format!("{implied_fraction:.2}"),
                "L4 perf-profile SUSPICIOUS (observe-only): {observed_tps:.0} tok/s exceeds the \
                 {ceiling_tps:.0} tok/s ceiling for the claimed model — throughput implies a model \
                 ~{:.0}% of the claimed size; possible smaller-model substitution",
                implied_fraction * 100.0
            );
        }
        PerfVerdict::Consistent { implied_fraction } => {
            tracing::debug!(
                target: "senda::verify::l4",
                peer = %peer.fmt_short(),
                %model,
                implied_fraction = format!("{implied_fraction:.2}"),
                "L4 perf-profile consistent"
            );
        }
        PerfVerdict::Inconclusive { reason } => {
            tracing::trace!(
                target: "senda::verify::l4",
                peer = %peer.fmt_short(),
                %model,
                "L4 perf-profile inconclusive: {reason}"
            );
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Qwen3 Q4_K_M reference sizes (bytes), and a representative consumer-GPU
    // memory bandwidth (RTX 4080-class ≈ 716 GB/s).
    const BYTES_8B: u64 = 4_700_000_000;
    const BYTES_4B: u64 = 2_500_000_000;
    const BW_4080: f64 = 716.0;

    /// Honest 8B: real decode reaches a fraction of memcpy bandwidth, so the
    /// observed rate is well under the ceiling and the implied size is ABOVE
    /// the claimed size. Must be `Consistent`.
    #[test]
    fn honest_8b_is_consistent() {
        // ceiling ≈ 716e9 / 4.7e9 ≈ 152 tok/s; honest llama.cpp ~120.
        let v = classify(120.0, BW_4080, BYTES_8B, 3);
        match v {
            PerfVerdict::Consistent { implied_fraction } => {
                assert!(
                    implied_fraction > 1.0,
                    "honest peer implied {implied_fraction}"
                );
            }
            other => panic!("expected Consistent, got {other:?}"),
        }
    }

    /// A 4B served while claiming the 8B: it decodes ~2x faster, so its
    /// throughput implies a model well below the claimed 8B size. Must flag.
    #[test]
    fn four_b_claiming_eight_b_is_suspicious() {
        // honest 4B at ~0.7 efficiency ≈ 0.7 * 716e9/2.5e9 ≈ 200 tok/s, served
        // under the "8B" label.
        let v = classify(200.0, BW_4080, BYTES_8B, 3);
        match v {
            PerfVerdict::Suspicious {
                implied_fraction, ..
            } => {
                assert!(
                    implied_fraction < FLAG_FRACTION,
                    "fake implied {implied_fraction}"
                );
            }
            other => panic!("expected Suspicious, got {other:?}"),
        }
    }

    /// A throughput physically impossible for the claimed model (faster than the
    /// memcpy ceiling) is the clearest fraud and must flag.
    #[test]
    fn faster_than_physical_ceiling_is_suspicious() {
        let ceiling = BW_4080 * 1e9 / BYTES_8B as f64; // ~152
        let v = classify(ceiling * 1.5, BW_4080, BYTES_8B, 3);
        assert!(matches!(v, PerfVerdict::Suspicious { .. }));
    }

    /// A slow (loaded/throttled) honest peer is never flagged — too-slow is not
    /// the fraud this layer targets.
    #[test]
    fn slow_peer_is_consistent_not_flagged() {
        let v = classify(20.0, BW_4080, BYTES_8B, 3);
        assert!(matches!(v, PerfVerdict::Consistent { .. }));
    }

    #[test]
    fn missing_or_thin_data_is_inconclusive() {
        assert!(matches!(
            classify(120.0, BW_4080, BYTES_8B, 1),
            PerfVerdict::Inconclusive { .. }
        ));
        assert!(matches!(
            classify(0.0, BW_4080, BYTES_8B, 3),
            PerfVerdict::Inconclusive { .. }
        ));
        assert!(matches!(
            classify(120.0, 0.0, BYTES_8B, 3),
            PerfVerdict::Inconclusive { .. }
        ));
        assert!(matches!(
            classify(120.0, BW_4080, 0, 3),
            PerfVerdict::Inconclusive { .. }
        ));
    }

    #[test]
    fn bandwidth_csv_sums_multi_gpu() {
        assert_eq!(parse_total_bandwidth_gbps(Some("716.0")), Some(716.0));
        assert_eq!(
            parse_total_bandwidth_gbps(Some("716.0,716.0")),
            Some(1432.0)
        );
        assert_eq!(parse_total_bandwidth_gbps(Some(" 700 , 0 ")), Some(700.0));
        assert_eq!(parse_total_bandwidth_gbps(Some("")), None);
        assert_eq!(parse_total_bandwidth_gbps(None), None);
    }

    /// Across the realistic inference-efficiency range for the backends we serve
    /// (≳0.6 of memcpy bandwidth on modern CUDA/Metal llama.cpp), the 4B
    /// substitution stays caught and the honest 8B stays clear — the separation
    /// isn't knife-edge. Below ~0.6 the 2x size gap can't clear the noise (a
    /// slow 4B hides inside the 8B's range); that's the documented detection
    /// floor, and crucially the honest peer is still never flagged there.
    #[test]
    fn separation_holds_across_efficiency_range() {
        for eff in [0.6, 0.7, 0.8, 0.85] {
            let honest_8b = eff * BW_4080 * 1e9 / BYTES_8B as f64;
            assert!(
                matches!(
                    classify(honest_8b, BW_4080, BYTES_8B, 3),
                    PerfVerdict::Consistent { .. }
                ),
                "honest 8B flagged at eff={eff}"
            );
            let fake_4b = eff * BW_4080 * 1e9 / BYTES_4B as f64;
            assert!(
                matches!(
                    classify(fake_4b, BW_4080, BYTES_8B, 3),
                    PerfVerdict::Suspicious { .. }
                ),
                "4B-as-8B missed at eff={eff}"
            );
        }
    }

    /// Zero false positives by construction: an honest peer decodes below its
    /// own ceiling, so its implied fraction is `1/efficiency > 1` — never
    /// flagged, even at implausibly low efficiency where fake detection fails.
    #[test]
    fn honest_peer_never_flagged_even_at_low_efficiency() {
        for eff in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99] {
            let honest = eff * BW_4080 * 1e9 / BYTES_8B as f64;
            assert!(
                matches!(
                    classify(honest, BW_4080, BYTES_8B, 3),
                    PerfVerdict::Consistent { .. }
                ),
                "honest 8B false-flagged at eff={eff}"
            );
        }
    }
}
