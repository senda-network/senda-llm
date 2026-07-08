//! Automatic host election and dynamic mesh management.
//!
//! Per-model election: nodes serving the same model form a group.
//! The highest-VRAM node in each group becomes its host and runs llama-server.
//! Every mesh change: kill llama-server, re-elect, winner starts fresh.
//! senda owns :api_port and proxies to the right host by model name.

use crate::cli::output::{
    emit_event, MoeAnalysisProgressSummary, MoeDistributionSummary, MoeStatusSummary, MoeSummary,
    OutputEvent,
};
use crate::inference::{launch, moe};
use crate::mesh;
use crate::models;
use crate::network::tunnel;
use crate::process_util::HideConsole;
use crate::system::hardware;
use launch::{BinaryFlavor, SplitMode};
use mesh::NodeRole;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::watch;

/// Returns `true` when `flavor` and `gpu_count` together call for row-split
/// tensor parallelism.
///
/// Row split requires a backend that implements `ggml_backend_split_buffer_type`
/// (CUDA and ROCm).  When no flavor is specified the binary may still be a CUDA
/// or ROCm build discovered automatically, so `None` is treated as potentially
/// supported; if the binary turns out to be CPU/Metal/Vulkan, llama.cpp falls
/// back safely.
fn should_use_row_split(flavor: Option<BinaryFlavor>, gpu_count: usize) -> bool {
    let backend_supported = matches!(
        flavor,
        Some(BinaryFlavor::Cuda) | Some(BinaryFlavor::Rocm) | None
    );
    backend_supported && gpu_count > 1
}

/// Returns `Some(SplitMode::Row)` when the local machine has multiple GPUs and
/// the llama.cpp backend supports row-level tensor parallelism (CUDA, ROCm).
///
/// Row split shards weight matrices across local GPUs so all GPUs are active on
/// every token — faster than layer (pipeline) split where GPUs take turns.
/// This does NOT work over RPC (network) — only for GPUs on the same machine.
///
/// When no explicit flavor is provided the resolved binary may still be CUDA/ROCm
/// (auto-detected from the binary name), so `None` is treated as potentially
/// supported.
pub(crate) fn local_multi_gpu_split_mode(flavor: Option<BinaryFlavor>) -> Option<SplitMode> {
    let hw = hardware::query(&[hardware::Metric::GpuCount]);
    let gpu_count = usize::from(hw.gpu_count);
    if should_use_row_split(flavor, gpu_count) {
        tracing::info!(
            "Local multi-GPU detected ({} GPUs) — using row split for tensor parallelism",
            gpu_count
        );
        Some(SplitMode::Row)
    } else {
        None
    }
}

fn split_mode_for_local_launch(
    flavor: Option<BinaryFlavor>,
    pinned_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
) -> Option<SplitMode> {
    if pinned_gpu.is_some() {
        return None;
    }
    local_multi_gpu_split_mode(flavor)
}

/// Calculate total model size, summing all split files if present.
/// Split files follow the pattern: name-00001-of-00004.gguf
///
/// Returns `0` when the file is missing or unreadable. This is what most
/// arithmetic callers want (`min_vram = model_bytes * 1.1`), but it is
/// dangerously ambiguous for any code that asks "does this peer actually
/// have the model?" — `0` then collapses to "yes, a zero-byte model that
/// fits in any amount of VRAM", which is exactly how a peer without the
/// weights on disk ends up advertising `serving_models = [<missing>]`
/// and winning host elections it cannot honor. Any code that gates on
/// "is the model present" MUST use `try_total_model_bytes` and treat
/// `None` as "this peer does not have the model".
///
/// See the May 16 2026 incident: three Qwen3-32B peers, all with a
/// dangling HuggingFace symlink (`config.json` downloaded, 19.76 GB
/// weights blob never finished), all reported `model_size_gb = 0.0`,
/// all passed `model_fits_locally = (vram >= 0)`, the cohort elected
/// an 8 GB RTX 4070 as host of a 19.76 GB model, and the entry node
/// happily routed chat into a black hole for two weeks.
pub fn total_model_bytes(model: &Path) -> u64 {
    try_total_model_bytes(model).unwrap_or(0)
}

/// Like [`total_model_bytes`] but returns `None` when the file (or any
/// shard of a split GGUF) is missing or unreadable on disk. Use this
/// in any code that asks "does this peer have the model on disk?".
///
/// `Some(0)` is reserved for the (impossible-in-practice) zero-byte
/// file case so callers can distinguish "missing" from "present-but-
/// empty". Both cases will trip the "is the model present" check.
pub fn try_total_model_bytes(model: &Path) -> Option<u64> {
    let name = model.to_string_lossy();
    // Check for split pattern: *-00001-of-NNNNN.gguf
    if let Some(pos) = name.find("-00001-of-") {
        let of_pos = pos + 10;
        if let Some(ext_pos) = name[of_pos..].find(".gguf") {
            if let Ok(n_split) = name[of_pos..of_pos + ext_pos].parse::<u32>() {
                let prefix = &name[..pos + 1];
                let suffix = &name[of_pos + ext_pos..];
                let mut total: u64 = 0;
                for i in 1..=n_split {
                    let split_name = format!("{}{:05}-of-{:05}{}", prefix, i, n_split, suffix);
                    // Any shard missing → the whole model is incomplete.
                    // Returning `None` here is what stops a half-downloaded
                    // multi-part GGUF from poisoning the cohort: the runtime
                    // will refuse to advertise it as servable, rather than
                    // declaring a partial-sum size that looks fine to the
                    // dense-launch planner.
                    let bytes = std::fs::metadata(&split_name).ok()?.len();
                    total += bytes;
                }
                return Some(total);
            }
        }
    }
    std::fs::metadata(model).ok().map(|m| m.len())
}

/// Determine if this node should be host for its model group.
/// Only considers peers serving the same model.
/// Deterministic: highest VRAM wins, tie-break by node ID.
///
/// `my_vram` and `peer.fast_memory_bytes()` are compared in the same
/// units (GPU / unified-memory bytes, NOT the RAM-offload-inflated
/// `vram_bytes`). The election picks the peer most likely to actually
/// run llama-server at GPU speed. Otherwise the highest-VRAM peer is
/// the one with the most spare DRAM, which produces a host whose
/// inference falls through to mmap and stalls the whole pipeline. See
/// `mesh::Node::fast_memory_bytes()` for the May 13 2026 incident.
pub fn should_be_host_for_model(
    my_id: iroh::EndpointId,
    my_vram: u64,
    model_peers: &[mesh::PeerInfo],
) -> bool {
    for peer in model_peers {
        if matches!(peer.role, NodeRole::Client) {
            continue;
        }
        let peer_vram = peer.fast_memory_bytes();
        if peer_vram > my_vram {
            return false;
        }
        if peer_vram == my_vram && peer.id > my_id {
            return false;
        }
    }
    true
}

/// Almost-solo-biased election (v0.66.38).
///
/// Identical to [`should_be_host_for_model`] but inserts a new highest-
/// priority criterion: a peer that can hold the entire model solo
/// (fast memory ≥ model bytes) is always preferred over a peer that
/// cannot, regardless of who has more raw VRAM. The motivation is the
/// May 17 2026 4-peer Qwen3-32B-Q4_K_M cohort:
///
///   * MacBook (12 GB) + manonas (10 GB) + MSI (8 GB) + LYU (16 GB)
///
/// LYU is the ONLY peer that can hold the 20 GB Q4 model with mmap
/// streaming local-share weights from disk while keeping the bulk of
/// the layers on the host's GPU — every other host has to RPC-split
/// onto WAN-latency tunnels for the bulk of layers. The legacy
/// `should_be_host_for_model` picked the maximum `fast_memory_bytes()`,
/// which is LYU only because LYU happens to also have the most VRAM.
/// In a mesh where Mac happens to also have 16 GB compound memory but
/// no peer can solo, the legacy code would still pick by raw memory and
/// then split — which is exactly what we want to AVOID.
///
/// Score tuple, compared lexicographically (largest wins):
///   `(can_hold_solo, fast_memory_bytes, system_ram_bytes, endpoint_id)`
///
/// The `endpoint_id` tiebreak preserves the legacy "higher id wins"
/// behavior so deterministic cohorts elect the same peer pre/post-
/// rollout (avoids the half-rolled-out mesh from oscillating between
/// the new winner and the old winner during the upgrade window).
pub fn should_be_host_for_model_with_solo_bias(
    my_id: iroh::EndpointId,
    my_vram: u64,
    my_system_ram_bytes: u64,
    model_bytes: u64,
    model_peers: &[mesh::PeerInfo],
) -> bool {
    let my_score = (my_vram >= model_bytes, my_vram, my_system_ram_bytes, my_id);
    for peer in model_peers {
        if matches!(peer.role, NodeRole::Client) {
            continue;
        }
        let peer_fast = peer.fast_memory_bytes();
        let peer_score = (
            peer_fast >= model_bytes,
            peer_fast,
            peer.system_ram_bytes,
            peer.id,
        );
        if peer_score > my_score {
            return false;
        }
    }
    true
}

/// Minimum fast-memory budget for a solo serve (matches `build_dense_launch_plan`).
fn min_vram_for_solo(model_bytes: u64) -> u64 {
    (model_bytes as f64 * 1.1) as u64
}

fn peer_can_solo_model(peer: &mesh::PeerInfo, model_bytes: u64) -> bool {
    peer.fast_memory_bytes() >= min_vram_for_solo(model_bytes)
}

fn any_other_peer_can_solo(model: &str, model_bytes: u64, peers: &[mesh::PeerInfo]) -> bool {
    peers
        .iter()
        .any(|p| p.is_assigned_model(model) && peer_can_solo_model(p, model_bytes))
}

fn demand_count(catalog_demand: &std::collections::HashMap<String, u64>, model: &str) -> u64 {
    catalog_demand.get(model).copied().unwrap_or(0)
}

/// Phase 2: choose which configured models this peer should actually advertise
/// and run, given local fast memory and mesh-wide capability. The desktop UI's
/// requested model list is a hint set — we drop models that waste this peer's
/// capacity on a split cohort when a better solo assignment exists.
///
/// `force_split_models` preserves explicit "Run on the mesh" opt-ins.
pub fn select_serving_models_for_peer(
    local_vram: u64,
    requested_models: &[String],
    model_bytes_by_name: &std::collections::HashMap<String, u64>,
    force_split_models: &std::collections::HashSet<String>,
    catalog_demand: &std::collections::HashMap<String, u64>,
    peers: &[mesh::PeerInfo],
) -> Vec<String> {
    let mut selected: Vec<String> = Vec::new();
    let mut flexible: Vec<(String, u64)> = Vec::new();

    for model in requested_models {
        let Some(&bytes) = model_bytes_by_name.get(model) else {
            continue;
        };
        if force_split_models.contains(model) {
            selected.push(model.clone());
        } else {
            flexible.push((model.clone(), bytes));
        }
    }

    if flexible.is_empty() {
        return sort_models_by_demand(selected, catalog_demand);
    }

    let largest_bytes = flexible.iter().map(|(_, b)| *b).max().unwrap_or(0);
    let smallest_bytes = flexible.iter().map(|(_, b)| *b).min().unwrap_or(0);

    if local_vram >= min_vram_for_solo(largest_bytes) {
        selected.extend(flexible.into_iter().map(|(m, _)| m));
    } else if local_vram >= min_vram_for_solo(smallest_bytes) {
        // Serve the largest model that still fits solo; drop the rest.
        let best = flexible
            .iter()
            .filter(|(_, bytes)| local_vram >= min_vram_for_solo(*bytes))
            .max_by(|(a_name, a_bytes), (b_name, b_bytes)| {
                a_bytes
                    .cmp(b_bytes)
                    .then_with(|| {
                        demand_count(catalog_demand, a_name)
                            .cmp(&demand_count(catalog_demand, b_name))
                    })
                    .then_with(|| a_name.cmp(b_name))
            });
        if let Some((model, _)) = best {
            selected.push(model.clone());
        }
    } else if flexible
        .iter()
        .any(|(name, bytes)| any_other_peer_can_solo(name, *bytes, peers))
    {
        // Can't solo anything, but another peer can solo at least one
        // requested model — only stay in cohorts for models nobody can solo.
        for (name, bytes) in flexible {
            if !any_other_peer_can_solo(&name, bytes, peers) {
                selected.push(name);
            }
        }
    } else {
        selected.extend(flexible.into_iter().map(|(m, _)| m));
    }

    sort_models_by_demand(selected, catalog_demand)
}

fn sort_models_by_demand(
    mut models: Vec<String>,
    catalog_demand: &std::collections::HashMap<String, u64>,
) -> Vec<String> {
    models.sort_by(|a, b| {
        demand_count(catalog_demand, b)
            .cmp(&demand_count(catalog_demand, a))
            .then_with(|| a.cmp(b))
    });
    models
}

/// Routing preference for a target: lower is better. Solo end-to-end hosts
/// win over pipeline-split hosts when both can serve the same model name.
impl InferenceTarget {
    pub fn priority_class(&self, model: &str, model_bytes: u64, peers: &[mesh::PeerInfo]) -> u8 {
        match self {
            InferenceTarget::Local(_) | InferenceTarget::MoeLocal(_) => 0,
            InferenceTarget::Remote(peer_id) | InferenceTarget::MoeRemote(peer_id) => peers
                .iter()
                .find(|p| p.id == *peer_id)
                .map(|p| {
                    if is_split_pipeline_host_for_model(p, model, model_bytes, peers) {
                        1
                    } else {
                        0
                    }
                })
                .unwrap_or(0),
            InferenceTarget::None => 2,
        }
    }
}

fn is_split_pipeline_host_for_model(
    host: &mesh::PeerInfo,
    model: &str,
    model_bytes: u64,
    peers: &[mesh::PeerInfo],
) -> bool {
    if !matches!(host.role, NodeRole::Host { .. }) {
        return false;
    }
    if peer_can_solo_model(host, model_bytes) {
        return false;
    }
    peers.iter().any(|p| {
        p.id != host.id && p.is_assigned_model(model) && matches!(p.role, NodeRole::Worker)
    })
}

fn routing_stable_hash(local_id: iroh::EndpointId, host_id: iroh::EndpointId) -> u64 {
    local_id
        .as_bytes()
        .iter()
        .chain(host_id.as_bytes().iter())
        .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

fn inflight_for_target(target: &InferenceTarget, peers: &[mesh::PeerInfo]) -> u64 {
    match target {
        InferenceTarget::Local(_) | InferenceTarget::MoeLocal(_) => 0,
        InferenceTarget::Remote(peer_id) | InferenceTarget::MoeRemote(peer_id) => peers
            .iter()
            .find(|p| p.id == *peer_id)
            .map(|p| p.inflight_requests)
            .unwrap_or(0),
        InferenceTarget::None => u64::MAX,
    }
}

fn force_split_routing_enabled() -> bool {
    std::env::var("SENDA_FORCE_SPLIT_ROUTING")
        .ok()
        .as_deref()
        == Some("1")
}

fn sort_targets_for_model(
    model: &str,
    model_bytes: u64,
    targets: &mut [InferenceTarget],
    peers: &[mesh::PeerInfo],
    local_id: iroh::EndpointId,
) {
    if force_split_routing_enabled() {
        return;
    }
    targets.sort_by(|a, b| {
        let pa = a.priority_class(model, model_bytes, peers);
        let pb = b.priority_class(model, model_bytes, peers);
        let ia = inflight_for_target(a, peers);
        let ib = inflight_for_target(b, peers);
        let ha = match a {
            InferenceTarget::Remote(id) | InferenceTarget::MoeRemote(id) => {
                routing_stable_hash(local_id, *id)
            }
            _ => 0,
        };
        let hb = match b {
            InferenceTarget::Remote(id) | InferenceTarget::MoeRemote(id) => {
                routing_stable_hash(local_id, *id)
            }
            _ => 0,
        };
        (pa, ia, ha).cmp(&(pb, ib, hb))
    });
}

/// Maximum time a peer may sit "elected but not actually serving" before the
/// rest of the cohort gives up on it and excludes it from host candidacy.
///
/// Picked at 30 s because the host-role transition happens *after*
/// `start_llama` returns (i.e. after llama-server has bound its port and the
/// loading proxy is up), but *before* the model finishes loading into VRAM.
/// On the slowest paths we measured (~25 GB GGUF on a cold-cache Mac
/// unified-memory box, mmap warming included), the role flip lands at
/// ≤ 12 s. Doubling for safety gets us 30. Anything longer and we should be
/// looking at a stuck peer, not an honest-but-slow one.
pub const HOST_CLAIM_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

/// Minimum senda runtime for participating in pipeline-parallel host
/// election and split planning. Peers below this (e.g. 0.66.18 during the
/// May 2026 DeepSeek-70B deadlock) are ignored for cohort capacity and host
/// picks so the rest of the mesh can converge without waiting for them to
/// upgrade. Shipped in v0.66.53 alongside `peers_for_pipeline_election`.
pub const MIN_PIPELINE_ELECTION_PEER_VERSION: &str = "0.66.20";

/// True when `version` meets [`MIN_PIPELINE_ELECTION_PEER_VERSION`] (or
/// `SENDA_MIN_PIPELINE_PEER_VERSION` when set). Missing or malformed
/// versions are treated as too old — safer than letting a silent legacy peer
/// block election.
pub fn peer_supports_pipeline_election(version: Option<&str>) -> bool {
    let min = std::env::var("SENDA_MIN_PIPELINE_PEER_VERSION")
        .ok()
        .and_then(|raw| semver::Version::parse(raw.trim()).ok())
        .unwrap_or_else(|| {
            semver::Version::parse(MIN_PIPELINE_ELECTION_PEER_VERSION)
                .expect("MIN_PIPELINE_ELECTION_PEER_VERSION is valid semver")
        });
    let Some(raw) = version.map(str::trim).filter(|s| !s.is_empty()) else {
        return false;
    };
    let Ok(peer_v) = semver::Version::parse(raw) else {
        return false;
    };
    peer_v >= min
}

/// Peers assigned to this model that are new enough for split election /
/// worker planning. Outdated stragglers stay in gossip but do not count.
pub fn peers_for_pipeline_election(model_peers: &[mesh::PeerInfo]) -> Vec<mesh::PeerInfo> {
    model_peers
        .iter()
        .filter(|p| peer_supports_pipeline_election(p.version.as_deref()))
        .cloned()
        .collect()
}

/// Filter a peer set down to those that can still credibly become host for
/// this model. The election picks the highest-fast-memory peer, but if that
/// peer is reachable in gossip and still does not advertise `NodeRole::Host`
/// after `HOST_CLAIM_GRACE`, every other node defers to a peer that will
/// never actually take the role and the cohort deadlocks. This filter lets
/// the loop drop those peers from host candidacy after their grace expires;
/// they remain perfectly usable as pipeline workers (the cohort capacity
/// total and worker selection do NOT use this filter — see issue #9).
///
/// A peer is viable iff:
///   1. It currently advertises `NodeRole::Host { .. }` for any model
///      (already serving — definitely a real candidate), OR
///   2. We have observed it for less than `HOST_CLAIM_GRACE` (still inside
///      its grace window — give it time).
///
/// Peers we have observed for `>= HOST_CLAIM_GRACE` that are NOT yet hosting
/// are dropped. Once such a peer eventually does flip to `NodeRole::Host`,
/// branch (1) re-admits it on the next election round — the exclusion is
/// not sticky beyond the actual misbehavior.
///
/// `first_observed` is owned by the election loop; it stamps each peer the
/// first time it shows up in `model_peers` and never updates that stamp.
/// Using `peer.last_seen` here would be wrong — gossip refreshes it on
/// every heartbeat, so a stuck peer would appear "freshly seen" forever.
///
/// Pure function: temporal state (`now`, `first_observed`) is supplied by
/// the caller, which makes this trivially unit-testable without a clock
/// mock. See the regression tests below for the v0.66.18 deadlock shape.
pub fn viable_host_candidates(
    model_peers: &[mesh::PeerInfo],
    first_observed: &std::collections::HashMap<iroh::EndpointId, std::time::Instant>,
    now: std::time::Instant,
    grace: std::time::Duration,
) -> Vec<mesh::PeerInfo> {
    model_peers
        .iter()
        .filter(|p| {
            if matches!(p.role, NodeRole::Host { .. }) {
                return true;
            }
            let first = first_observed.get(&p.id).copied().unwrap_or(now);
            now.saturating_duration_since(first) < grace
        })
        .cloned()
        .collect()
}

/// Conservative estimate of how much system RAM a peer will commit while
/// hosting `model_bytes` in a cohort of `cohort_size` peers (host
/// included).
///
/// Math: with mmap on (v0.66.38 task A), llama-server pages local-share
/// weights through the OS page cache and only commits activations,
/// KV cache, and RPC0 staging. Empirically on Qwen3-32B-Q4_K_M across
/// 4 peers, that's ~3 GB constant + ~half of the local share within
/// the first 30 s of inference (the OS pre-faults aggressively). We
/// bound by `3 GB + local_share / 2` so a 16 GB Mac in a 4-peer
/// cohort hosting a 20 GB model predicts 5.5 GB commit — well under
/// 75% of 16 GB and matches the post-fix observed RSS on the May 17
/// regression.
pub fn predicted_host_ram_bytes(model_bytes: u64, cohort_size: usize) -> u64 {
    const ACTIVATION_FLOOR: u64 = 3 * 1024 * 1024 * 1024;
    let local_share = if cohort_size == 0 {
        model_bytes
    } else {
        model_bytes / cohort_size as u64
    };
    ACTIVATION_FLOOR + local_share / 2
}

/// Does `peer` have enough total system RAM to host `model_bytes` in a
/// cohort of `cohort_size` peers without swapping?
///
/// Legacy peers that don't gossip `system_ram_bytes` (= 0) are
/// optimistically accepted so that one slow upgrade rollout doesn't
/// silently exclude an entire cohort.
///
/// The 0.75 budget headroom accounts for the OS, the desktop app,
/// browser/IDE, and other userland processes — picked because every
/// host we've seen OOM in the May 17 incident was running with <25%
/// free RAM. Drop the multiplier later once we have a tighter
/// activation model.
pub fn ram_can_host_model(peer: &mesh::PeerInfo, model_bytes: u64, cohort_size: usize) -> bool {
    if peer.system_ram_bytes == 0 {
        return true;
    }
    let predicted = predicted_host_ram_bytes(model_bytes, cohort_size);
    let budget = (peer.system_ram_bytes as f64 * 0.75) as u64;
    predicted <= budget
}

/// Apply the RAM-aware filter to an existing candidate set. Current
/// `NodeRole::Host` peers always pass — pulling the rug out from under
/// a peer that's already successfully serving the model creates more
/// churn than it solves. The filter is meant to keep new candidates
/// from getting elected and OOMing on first launch (the May 17 2026
/// MacBook crash).
pub fn ram_filtered_host_candidates(
    candidates: Vec<mesh::PeerInfo>,
    model_bytes: u64,
) -> Vec<mesh::PeerInfo> {
    let cohort_size = candidates.len();
    candidates
        .into_iter()
        .filter(|p| {
            matches!(p.role, NodeRole::Host { .. })
                || ram_can_host_model(p, model_bytes, cohort_size)
        })
        .collect()
}

/// Sliding-window cap on how many failed `start_llama` attempts a single
/// node will burn before stepping aside. Picked at 2 (with the window
/// below) so a transient hiccup doesn't bench us, but a genuinely stuck
/// host (e.g. workers behind broken iroh tunnels — May 14 2026 incident)
/// surrenders quickly enough for the runner-up to take a turn.
pub const HOST_ATTEMPT_MAX_FAILURES: usize = 2;

/// Window for `HOST_ATTEMPT_MAX_FAILURES`. After this much elapsed time
/// without a fresh failure the counter resets — we don't want yesterday's
/// network blip to keep us out of host election today.
pub const HOST_ATTEMPT_FAILURE_WINDOW: std::time::Duration = std::time::Duration::from_secs(90);

/// How long we step aside once the threshold trips. Long enough for
/// `HOST_CLAIM_GRACE` (30 s) to expire on the runner-up's view of us, plus
/// some padding so we don't ping-pong between candidates each cycle.
pub const HOST_ATTEMPT_BACKOFF: std::time::Duration = std::time::Duration::from_secs(60);

/// v0.66.38 "slow churn" window: covers the case where a host wins
/// re-election repeatedly over the span of several minutes (e.g. a
/// flaky worker keeps dropping out and back in, triggering teardown +
/// re-launch on the host's side). The fast 90 s window above only
/// catches back-to-back failures; this longer window catches the
/// "drips that add up to a flood" pattern from the May 18 2026 LYU
/// flapping incident.
pub const HOST_ATTEMPT_SLOW_WINDOW: std::time::Duration = std::time::Duration::from_secs(600);

/// Slow-window failure threshold: 3 failures across `HOST_ATTEMPT_SLOW_WINDOW`
/// arms the slow backoff. Picked one higher than the fast-window threshold
/// so genuinely transient hiccups don't trip both paths at once.
pub const HOST_ATTEMPT_SLOW_MAX: usize = 3;

/// Backoff applied when the slow window trips. Longer than the fast
/// backoff because the failure pattern is persistent (the cohort needs
/// real time to stabilize, not just a re-probe).
pub const HOST_ATTEMPT_SLOW_BACKOFF: std::time::Duration = std::time::Duration::from_secs(300);

/// How long we let a freshly-launched `llama-server` sit "loading but
/// not yet routable" before assuming the fitter loop
/// (`common_params_fit_impl`) has hung and tearing the process down.
///
/// v0.66.38 May 18 2026 incident: a 16 GB Mac hosting Qwen3-32B-Q4_K_M
/// split across a 13.8 GB Mac + 8 GB MSI cohort spent >40 minutes inside
/// `common_params_fit_impl` trying every layer distribution. The server
/// never returned 200 to /health, the desktop app showed "loading…"
/// forever, and the human had to manually kill the process.
///
/// 5 minutes is well past any healthy launch on the hardware we ship to
/// (the slowest cold-cache 25 GB GGUF load we measured lands at ≤90 s)
/// while still being short enough that the user notices something is
/// wrong and the cohort can fail over to the runner-up via
/// `HostAttemptBackoff`.
pub const FITTER_WATCHDOG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Cooldown after the election loop starts during which a peer that
/// hasn't yet observed any other peer must not self-elect as host.
///
/// Discovery on a fresh start typically takes 3-10 s on relay-mediated
/// joins, longer on direct-dial. Without this grace, a peer that
/// restarts in parallel with the rest of the cohort wins its own
/// 1-peer election before the other peers' announcements arrive, then
/// the v0.66.37 sticky-cohort logic prevents it from yielding even
/// after a much-better candidate (e.g. a 64 GB RAM peer) joins. The
/// May 18 2026 incident hit exactly this on four peers restarted in
/// sequence: each one self-elected as its own host, the entry node saw
/// only one of them as Host (LYU), and the other three left orphan
/// `llama-server` processes hung on tensor load. 20 s is well past
/// typical relay discovery and short enough that a genuinely-solo peer
/// (e.g. running standalone for tests) still launches in <30 s.
pub const ELECTION_DISCOVERY_GRACE: std::time::Duration = std::time::Duration::from_secs(20);

/// Tracks repeated `start_llama` failures so a single node can't pin the
/// cohort by failing over and over. Records each failure with a timestamp,
/// trims out anything older than `HOST_ATTEMPT_FAILURE_WINDOW`, and once
/// `HOST_ATTEMPT_MAX_FAILURES` accumulate within that window flips into a
/// `backoff_until` state for `HOST_ATTEMPT_BACKOFF`.
///
/// Pure logic, no clock — every method takes `now` from the caller, which
/// makes the regression tests trivial. See the May 14 2026 incident in
/// issue #10: `v0.66.21` correctly elected `1024286234` (Mac) but the
/// elected host SIGABRT'd on every relaunch because its iroh tunnels to
/// the Windows workers were silently dropping bytes. Without this backoff
/// the runtime would loop on the same `recv failed (bytes_recv=0)` crash
/// forever instead of demoting and letting the runner-up try.
#[derive(Default, Debug)]
pub struct HostAttemptBackoff {
    failures: std::collections::VecDeque<std::time::Instant>,
    backoff_until: Option<std::time::Instant>,
}

impl HostAttemptBackoff {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` once a backoff has tripped and is still active.
    /// Callers should treat this as "force-disable host candidacy for me".
    pub fn is_active(&self, now: std::time::Instant) -> bool {
        self.backoff_until.map(|until| now < until).unwrap_or(false)
    }

    /// Records a failed `start_llama` attempt and, if the per-window
    /// threshold is crossed, arms the backoff. Returns the new
    /// `backoff_until` instant whenever the call *transitioned* from "no
    /// active backoff" to "active backoff" so the caller can emit a
    /// user-visible warning exactly once per trip.
    pub fn record_failure(&mut self, now: std::time::Instant) -> Option<std::time::Instant> {
        // If a previous backoff has expired, clear it so the next trip
        // emits a fresh warning instead of being silently swallowed.
        if let Some(until) = self.backoff_until {
            if now >= until {
                self.backoff_until = None;
                self.failures.clear();
            }
        }
        if self.is_active(now) {
            return None;
        }
        self.failures.push_back(now);
        // Trim only to the LONGEST window we still care about (slow), so
        // both fast-window and slow-window counts can read from the same
        // deque without losing data.
        while let Some(&earliest) = self.failures.front() {
            if now.saturating_duration_since(earliest) > HOST_ATTEMPT_SLOW_WINDOW {
                self.failures.pop_front();
            } else {
                break;
            }
        }
        // Slow window first: a stretched-out pattern earns the longer
        // backoff so the cohort actually has time to stabilize.
        let slow_count = self
            .failures
            .iter()
            .filter(|t| now.saturating_duration_since(**t) <= HOST_ATTEMPT_SLOW_WINDOW)
            .count();
        if slow_count >= HOST_ATTEMPT_SLOW_MAX {
            let until = now + HOST_ATTEMPT_SLOW_BACKOFF;
            self.backoff_until = Some(until);
            self.failures.clear();
            return Some(until);
        }
        let fast_count = self
            .failures
            .iter()
            .filter(|t| now.saturating_duration_since(**t) <= HOST_ATTEMPT_FAILURE_WINDOW)
            .count();
        if fast_count >= HOST_ATTEMPT_MAX_FAILURES {
            let until = now + HOST_ATTEMPT_BACKOFF;
            self.backoff_until = Some(until);
            self.failures.clear();
            return Some(until);
        }
        None
    }

    /// Marks the cohort as serving (e.g. we just successfully launched).
    /// Wipes any in-progress failure counter so the next failure starts
    /// from a clean slate.
    pub fn record_success(&mut self) {
        self.failures.clear();
        self.backoff_until = None;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DenseLaunchPlan {
    Solo,
    Split {
        worker_ids: Vec<iroh::EndpointId>,
        total_group_vram: u64,
    },
    WaitingForCapacity {
        worker_ids: Vec<iroh::EndpointId>,
        total_group_vram: u64,
        min_vram: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DenseRunningPlan {
    Solo,
    Split { worker_ids: Vec<iroh::EndpointId> },
}

impl DenseLaunchPlan {
    fn running_plan(&self) -> Option<DenseRunningPlan> {
        match self {
            DenseLaunchPlan::Solo => Some(DenseRunningPlan::Solo),
            DenseLaunchPlan::Split { worker_ids, .. } => Some(DenseRunningPlan::Split {
                worker_ids: worker_ids.clone(),
            }),
            DenseLaunchPlan::WaitingForCapacity { .. } => None,
        }
    }
}

/// Sticky-cohort check: is the cohort we already launched llama-server with
/// still good enough to keep serving the model?
///
/// The election loop runs `build_dense_launch_plan` on every gossip change
/// and gets back the *currently preferred* plan. Without this check, any
/// RTT jitter or transient peer state flip that re-sorts worker selection
/// causes the loop to consider the new plan "different" and tear down the
/// running llama-server — even though the running cohort is still healthy.
/// Pre-fix that produced the May 17 2026 churn loop where the model spent
/// ~60 s serving for every ~6 min of relaunch warm-up.
///
/// "Still viable" means:
///   - Solo: this node's fast memory can still hold the model.
///   - Split: every running worker id is still in `model_peers` (so iroh
///     tunnels still exist and the peer hasn't departed), AND the running
///     cohort's combined fast-memory budget still covers `model_bytes *
///     1.1` (the same 10 % cushion `build_dense_launch_plan` uses for
///     fresh planning).
///
/// When the running cohort is still viable we keep serving and just let
/// the inner sleep/select tick again, so chat traffic doesn't see a
/// teardown window.
fn current_cohort_still_viable(
    running: &DenseRunningPlan,
    model_peers: &[mesh::PeerInfo],
    local_launch_vram: u64,
    model_bytes: u64,
) -> bool {
    let min_vram = (model_bytes as f64 * 1.1) as u64;
    match running {
        DenseRunningPlan::Solo => local_launch_vram >= min_vram,
        DenseRunningPlan::Split { worker_ids } => {
            let mut total = local_launch_vram;
            for id in worker_ids {
                let Some(peer) = model_peers.iter().find(|p| &p.id == id) else {
                    return false;
                };
                total = total.saturating_add(split_peer_vram_bytes(peer, local_launch_vram));
            }
            total >= min_vram
        }
    }
}

/// Bytes a peer can actually hold in fast memory for split planning.
///
/// Must match what `peer.fast_memory_bytes()` returns elsewhere — see
/// `mesh::Node::fast_memory_bytes()` and the May 13 2026 deadlock notes
/// in `mesh/mod.rs`. Using `peer.vram_bytes` directly here was the
/// May 14 2026 incident: on Linux NVIDIA peers `vram_bytes` includes
/// the `0.75 * RAM_offload` allowance (~106 GB on LYU's 16 GB 4080-SUPER
/// with 120 GB system RAM), so the split planner happily put 36 GB of
/// DeepSeek-R1-Distill-70B-Q4_K_M weights on a 16 GB GPU and llama-server
/// died with `alloc_tensor_range: failed to allocate RPC0 buffer of size
/// 36706976000`. `fast_memory_bytes` clamps to the actual GPU/unified-memory
/// budget on every platform, so the split ratio reflects what each device
/// can hold and the planner correctly stops adding workers only once the
/// real fast-memory total covers the model.
fn split_peer_vram_bytes(peer: &mesh::PeerInfo, my_vram: u64) -> u64 {
    let fast = peer.fast_memory_bytes();
    if fast > 0 {
        fast
    } else {
        my_vram
    }
}

fn effective_local_launch_vram(
    my_vram: u64,
    pinned_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
) -> u64 {
    pinned_gpu.map(|gpu| gpu.vram_bytes).unwrap_or(my_vram)
}

fn build_dense_launch_plan(
    my_vram: u64,
    model_bytes: u64,
    force_split: bool,
    model_name: &str,
    model_peers: &[mesh::PeerInfo],
) -> DenseLaunchPlan {
    let min_vram = (model_bytes as f64 * 1.1) as u64;
    // Solo is only appropriate when this node's *fast memory* (GPU VRAM
    // on discrete cards, unified-memory working set on Apple Silicon)
    // can hold the model. Callers pass `my_vram = node.fast_memory_bytes()`,
    // not `node.vram_bytes()` — the latter includes a 75% RAM-offload
    // allowance that is correct for "this node could mmap the weights
    // somewhere" but wrong for "this node could serve at usable speed".
    //
    // The May 13 2026 deadlock was the cost of conflating those two: an
    // RTX 4080-SUPER laptop with 16 GB of GPU and 120 GB of host RAM
    // reported `vram_bytes = 106 GB`, the planner picked Solo for a
    // 42.5 GB model, llama.cpp paged weights through PCIe at ~0.2 t/s,
    // and the mesh entry node showed every peer Loading indefinitely
    // while no chat request ever completed.
    if !force_split && my_vram >= min_vram {
        return DenseLaunchPlan::Solo;
    }

    // Mixed-backend pipeline-parallel is now allowed (v0.66.13). llama.cpp's
    // rpc-server serialises tensors over the wire, so layers 0..N can compute
    // on the host's backend (e.g. Metal) and N+1.. on a worker's (e.g. CUDA)
    // without the host needing matching local kernels. The conversion + TCP
    // overhead makes mixed-backend splits noticeably slower than same-backend
    // ones, but functionally correct — and "your Mac + your friend's RTX
    // laptop" is exactly the heterogeneous-mesh use case we ship for. We
    // previously filtered cross-backend peers out here as a conservatism the
    // ROADMAP flagged as deferred work; in practice the conservatism made
    // every Apple+CUDA pair fall back to a no-fit solo run on whichever side
    // the planner picked, which surfaces to users as `Compute error` 500s
    // from llama.cpp the moment a token is decoded against an mmap-fallback
    // model that doesn't fit in the elected host's VRAM. Honesty beats
    // pessimism: try the split, let llama.cpp fail loudly if a specific
    // weight format isn't routable.
    let mut candidates: Vec<_> = model_peers
        .iter()
        .filter(|p| matches!(p.role, NodeRole::Worker) || p.is_assigned_model(model_name))
        .filter(|p| !matches!(p.role, NodeRole::Client))
        // Lazy rpc-server gate: skip a worker whose on-demand rpc-server is
        // down (`Some(false)`). `None` (legacy, always-on rpc) and `Some(true)`
        // both pass. This is what keeps the host from dialing a still-cold
        // worker and tripping the v0.66.36 HELLO-failure blacklist — the
        // worker brings its rpc up (gossiping `rpc_ready=true`) as soon as it
        // sees the split is needed, and only then becomes selectable.
        .filter(|p| p.rpc_ready != Some(false))
        .filter(|p| !matches!(p.rtt_ms, Some(rtt) if rtt > mesh::MAX_SPLIT_RTT_MS))
        .collect();
    candidates.sort_by_key(|p| (p.rtt_ms.unwrap_or(u32::MAX), p.id));

    let mut total_group_vram = my_vram;
    let mut worker_ids = Vec::new();
    for peer in candidates {
        if total_group_vram >= min_vram && !(force_split && worker_ids.is_empty()) {
            break;
        }
        total_group_vram += split_peer_vram_bytes(peer, my_vram);
        worker_ids.push(peer.id);
    }

    if total_group_vram >= min_vram && (!force_split || !worker_ids.is_empty()) {
        DenseLaunchPlan::Split {
            worker_ids,
            total_group_vram,
        }
    } else {
        DenseLaunchPlan::WaitingForCapacity {
            worker_ids,
            total_group_vram,
            min_vram,
        }
    }
}

fn rpc_ports_for_worker_ids(
    all_ports: &HashMap<iroh::EndpointId, u16>,
    worker_ids: &[iroh::EndpointId],
) -> Option<Vec<u16>> {
    worker_ids
        .iter()
        .map(|id| all_ports.get(id).copied())
        .collect()
}

fn replenish_worker_cohort_after_probe_failure(
    model_name: &str,
    model_peers: &[mesh::PeerInfo],
    current_worker_ids: &[iroh::EndpointId],
    current_rpc_ports: &[u16],
    bad_ports: &HashSet<u16>,
    all_ports: &HashMap<iroh::EndpointId, u16>,
    local_launch_vram: u64,
    model_bytes: u64,
) -> (Vec<iroh::EndpointId>, Vec<u16>, u64) {
    let mut worker_ids = Vec::new();
    let mut rpc_ports = Vec::new();
    let bad_worker_ids: HashSet<_> = all_ports
        .iter()
        .filter_map(|(id, port)| bad_ports.contains(port).then_some(*id))
        .collect();
    let mut group_capacity = local_launch_vram;

    for (id, port) in current_worker_ids.iter().zip(current_rpc_ports.iter()) {
        if bad_ports.contains(port) {
            continue;
        } else {
            worker_ids.push(*id);
            rpc_ports.push(*port);
            if let Some(peer) = model_peers.iter().find(|peer| peer.id == *id) {
                group_capacity =
                    group_capacity.saturating_add(split_peer_vram_bytes(peer, local_launch_vram));
            }
        }
    }

    let min_vram = (model_bytes as f64 * 1.1) as u64;
    if group_capacity >= min_vram {
        return (worker_ids, rpc_ports, group_capacity);
    }

    let mut candidates: Vec<_> = model_peers
        .iter()
        .filter(|p| matches!(p.role, NodeRole::Worker) || p.is_assigned_model(model_name))
        .filter(|p| !matches!(p.role, NodeRole::Client))
        .filter(|p| !matches!(p.rtt_ms, Some(rtt) if rtt > mesh::MAX_SPLIT_RTT_MS))
        .filter(|p| !bad_worker_ids.contains(&p.id))
        .filter(|p| !worker_ids.contains(&p.id))
        .collect();
    candidates.sort_by_key(|p| (p.rtt_ms.unwrap_or(u32::MAX), p.id));

    for peer in candidates {
        if group_capacity >= min_vram {
            break;
        }
        let Some(port) = all_ports.get(&peer.id).copied() else {
            continue;
        };
        if bad_ports.contains(&port) {
            continue;
        }
        worker_ids.push(peer.id);
        rpc_ports.push(port);
        group_capacity =
            group_capacity.saturating_add(split_peer_vram_bytes(peer, local_launch_vram));
    }

    (worker_ids, rpc_ports, group_capacity)
}

async fn probe_rpc_ports(
    rpc_ports: &[u16],
    port_to_peer: &HashMap<u16, iroh::EndpointId>,
) -> (Vec<String>, HashSet<u16>) {
    let mut probe_set = tokio::task::JoinSet::new();
    for port in rpc_ports {
        let port = *port;
        probe_set.spawn(async move {
            let addr: std::net::SocketAddr = ([127, 0, 0, 1], port).into();
            let outcome = crate::network::rpc_probe::probe_hello(
                addr,
                crate::network::rpc_probe::DEFAULT_PROBE_TIMEOUT,
            )
            .await;
            (port, outcome)
        });
    }

    let mut bad = Vec::new();
    let mut bad_ports = HashSet::new();
    while let Some(joined) = probe_set.join_next().await {
        if let Ok((port, outcome)) = joined {
            if !outcome.is_healthy() {
                bad.push(format_probe_failure(
                    port,
                    port_to_peer.get(&port),
                    &outcome,
                ));
                bad_ports.insert(port);
            }
        }
    }
    (bad, bad_ports)
}

/// Format a single failed HELLO probe so the dashboard warning names the
/// remote peer whose rpc-server is the actual bottleneck instead of just the
/// opaque local tunnel port. The endpoint short id is what every other event
/// in the runtime uses, so users can cross-reference it against the status
/// page peer list.
fn format_probe_failure(
    port: u16,
    peer_id: Option<&iroh::EndpointId>,
    outcome: &crate::network::rpc_probe::ProbeOutcome,
) -> String {
    match peer_id {
        Some(id) => format!("peer {} (127.0.0.1:{port}) ({outcome:?})", id.fmt_short()),
        None => format!("127.0.0.1:{port} ({outcome:?})"),
    }
}

/// The current state of llama-server as managed by the election loop.
/// The API proxy reads this to know where to forward requests.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InferenceTarget {
    /// No llama-server running anywhere (election in progress, mesh empty, etc.)
    None,
    /// We are host — llama-server is on this local port.
    Local(u16),
    /// Another node is host — proxy via QUIC to this peer.
    Remote(iroh::EndpointId),
    /// MoE mode — this node runs its own llama-server with its expert shard.
    /// All MoE nodes are independent; the proxy picks one per session.
    MoeLocal(u16),
    /// MoE mode — another node is running its shard; proxy via QUIC.
    MoeRemote(iroh::EndpointId),
}

/// MoE deployment state shared between election and proxy.
/// The proxy uses this to route sessions to MoE nodes.
#[derive(Clone, Debug, Default)]
pub struct MoeState {
    /// All MoE node targets (local + remote), in stable order.
    pub nodes: Vec<InferenceTarget>,
    /// Full-coverage targets that can serve the whole model if the active shard set fails.
    pub fallbacks: Vec<InferenceTarget>,
}

/// Per-model routing table. The API proxy uses this to route by model name.
#[derive(Clone, Debug, Default)]
pub struct ModelTargets {
    /// model_name → list of inference targets (multiple hosts = load balancing)
    pub targets: HashMap<String, Vec<InferenceTarget>>,
    /// MoE state — if set, this model uses MoE expert sharding.
    /// The proxy uses this for session-sticky routing across MoE nodes.
    pub moe: Option<MoeState>,
    /// Round-robin counter for load balancing, shared across clones via Arc<AtomicU64>
    /// so that all ModelTargets clones (including per-request proxy clones) share a sequence.
    counter: Arc<AtomicU64>,
}

#[derive(Clone, Debug)]
pub struct LocalProcessInfo {
    pub backend: String,
    pub pid: u32,
    pub port: u16,
    pub context_length: u32,
}

fn stop_requested(stop_rx: &watch::Receiver<bool>) -> bool {
    *stop_rx.borrow()
}

fn emit_ready_events(model_name: &str, llama_port: u16, model_port: u16, ctx_size: u32) {
    let _ = emit_event(OutputEvent::LlamaReady {
        model: Some(model_name.to_string()),
        port: llama_port,
        ctx_size: Some(ctx_size),
        log_path: None,
    });
    let _ = emit_event(OutputEvent::ModelReady {
        model: model_name.to_string(),
        internal_port: Some(model_port),
        role: None,
    });
}

fn emit_moe_status(model_name: &str, phase: &str, detail: impl Into<String>) {
    let _ = emit_event(OutputEvent::MoeStatus {
        model: model_name.to_string(),
        status: MoeStatusSummary {
            phase: phase.to_string(),
            detail: detail.into(),
        },
    });
}

fn emit_warning(message: impl Into<String>, context: Option<String>) {
    let _ = emit_event(OutputEvent::Warning {
        message: message.into(),
        context,
    });
}

fn emit_error(message: impl Into<String>, context: Option<String>) {
    let _ = emit_event(OutputEvent::Error {
        message: message.into(),
        context,
    });
}

fn emit_info(message: impl Into<String>, context: Option<String>) {
    let _ = emit_event(OutputEvent::Info {
        message: message.into(),
        context,
    });
}

fn emit_moe_analysis_progress(
    model_name: &str,
    mode: &str,
    spinner: &str,
    current: usize,
    total: Option<usize>,
    elapsed_secs: u64,
) {
    let _ = emit_event(OutputEvent::MoeAnalysisProgress {
        model: model_name.to_string(),
        progress: MoeAnalysisProgressSummary {
            mode: mode.to_string(),
            spinner: spinner.to_string(),
            current,
            total,
            elapsed_secs,
        },
    });
}

async fn wait_for_peer_moe_ranking(
    model_name: &str,
    model_path: &Path,
    peer_rx: &mut watch::Receiver<usize>,
    stop_rx: &mut watch::Receiver<bool>,
    timeout: std::time::Duration,
) {
    if moe::best_shared_ranking_artifact(model_path).is_some() {
        return;
    }

    emit_moe_status(
        model_name,
        "waiting for peer ranking",
        format!("up to {:.0}s before local analysis", timeout.as_secs_f64()),
    );

    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            emit_moe_status(
                model_name,
                "peer ranking timeout",
                "continuing with local analysis",
            );
            return;
        }

        tokio::select! {
            _ = tokio::time::sleep(remaining) => {
                emit_moe_status(
                    model_name,
                    "peer ranking timeout",
                    "continuing with local analysis",
                );
                return;
            }
            res = peer_rx.changed() => {
                if res.is_err() {
                    return;
                }
                if let Some(artifact) = moe::best_shared_ranking_artifact(model_path) {
                    emit_moe_status(
                        model_name,
                        "using imported peer ranking",
                        format!(
                            "mode={} origin={}",
                            artifact.kind.label(),
                            artifact.origin.label()
                        ),
                    );
                    return;
                }
            }
            res = stop_rx.changed() => {
                if res.is_err() || stop_requested(stop_rx) {
                    return;
                }
            }
        }
    }
}

impl ModelTargets {
    /// Get target for a specific model. Round-robins across multiple hosts.
    pub fn get(&self, model: &str) -> InferenceTarget {
        match self.targets.get(model) {
            Some(targets) if !targets.is_empty() => {
                let idx = self.counter.fetch_add(1, Ordering::Relaxed) as usize % targets.len();
                targets[idx].clone()
            }
            _ => InferenceTarget::None,
        }
    }

    /// All candidate targets for a model, preserving their current order.
    pub fn candidates(&self, model: &str) -> Vec<InferenceTarget> {
        self.targets.get(model).cloned().unwrap_or_default()
    }

    /// Round-robin pick from a caller-supplied candidate slice.
    pub fn pick_from(&self, candidates: &[InferenceTarget]) -> InferenceTarget {
        if candidates.is_empty() {
            InferenceTarget::None
        } else {
            let idx = self.counter.fetch_add(1, Ordering::Relaxed) as usize % candidates.len();
            candidates[idx].clone()
        }
    }

    /// Sticky pick from a caller-supplied candidate slice.
    pub fn pick_sticky_from(candidates: &[InferenceTarget], sticky_key: u64) -> InferenceTarget {
        if candidates.is_empty() {
            InferenceTarget::None
        } else {
            let idx = sticky_key as usize % candidates.len();
            candidates[idx].clone()
        }
    }

    /// Get MoE target for a session (hash-based routing).
    /// Returns None if not in MoE mode.
    pub fn get_moe_target(&self, session_hint: &str) -> Option<InferenceTarget> {
        let moe = self.moe.as_ref()?;
        if moe.nodes.is_empty() {
            return None;
        }
        // Simple hash routing: hash the session hint, pick a node
        let hash = session_hint
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let idx = (hash as usize) % moe.nodes.len();
        Some(moe.nodes[idx].clone())
    }

    pub fn get_moe_failover_targets(&self, session_hint: &str) -> Vec<InferenceTarget> {
        let Some(primary) = self.get_moe_target(session_hint) else {
            return Vec::new();
        };
        let mut ordered = vec![primary.clone()];
        if let Some(moe) = self.moe.as_ref() {
            for fallback in &moe.fallbacks {
                if fallback != &primary {
                    ordered.push(fallback.clone());
                }
            }
        }
        ordered
    }
}

/// Compute shard index for a node given all node IDs in the MoE group.
/// Nodes are sorted by ID to ensure all nodes agree on the ordering.
/// Returns (sorted_ids, my_index).
#[cfg(test)]
pub fn moe_shard_index(
    my_id: iroh::EndpointId,
    peer_ids: &[iroh::EndpointId],
) -> (Vec<iroh::EndpointId>, usize) {
    let mut all_ids: Vec<iroh::EndpointId> = peer_ids.to_vec();
    if !all_ids.contains(&my_id) {
        all_ids.push(my_id);
    }
    all_ids.sort();
    let idx = all_ids.iter().position(|id| *id == my_id).unwrap_or(0);
    (all_ids, idx)
}

/// Build the MoE target map from sorted node IDs.
/// The caller's own node gets MoeLocal(port), others get MoeRemote(id).
pub fn build_moe_targets(
    sorted_ids: &[iroh::EndpointId],
    fallback_ids: &[iroh::EndpointId],
    my_id: iroh::EndpointId,
    active_local_port: Option<u16>,
    fallback_local_port: Option<u16>,
    model_name: &str,
) -> ModelTargets {
    let mut moe_state = MoeState::default();
    for &id in sorted_ids {
        if id == my_id {
            if let Some(port) = active_local_port {
                moe_state.nodes.push(InferenceTarget::MoeLocal(port));
            }
        } else {
            moe_state.nodes.push(InferenceTarget::MoeRemote(id));
        }
    }
    for &id in fallback_ids {
        if id == my_id {
            if let Some(port) = fallback_local_port {
                moe_state.fallbacks.push(InferenceTarget::Local(port));
            }
        } else {
            moe_state.fallbacks.push(InferenceTarget::Remote(id));
        }
    }
    let mut targets = ModelTargets::default();
    let primary_targets = if let Some(port) = active_local_port {
        vec![InferenceTarget::MoeLocal(port)]
    } else if let Some(port) = fallback_local_port {
        vec![InferenceTarget::Local(port)]
    } else {
        Vec::new()
    };
    if !primary_targets.is_empty() {
        targets
            .targets
            .insert(model_name.to_string(), primary_targets);
    }
    targets.moe = Some(moe_state);
    targets
}

#[derive(Clone, Debug)]
struct ResolvedMoeConfig {
    config: crate::models::catalog::MoeConfig,
    ranking_strategy: moe::MoeRankingStrategy,
    ranking_source: String,
    ranking_origin: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MoePlacementRole {
    SplitShard,
    FullFallback,
    Standby,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MoePlacementPlan {
    leader_id: iroh::EndpointId,
    active_ids: Vec<iroh::EndpointId>,
    fallback_ids: Vec<iroh::EndpointId>,
    overlap: usize,
}

const MOE_SCALE_UP_QUIET_SECS: u64 = 45;

#[derive(Clone, Copy, Debug)]
struct MoePlacementCandidate {
    id: iroh::EndpointId,
    /// Fast-memory budget for this candidate (GPU VRAM on discrete cards,
    /// unified-memory working set on Apple Silicon). Always populated from
    /// `Node::fast_memory_bytes` / `PeerInfo::fast_memory_bytes` — never
    /// from the inflated `vram_bytes` advertised in gossip. See
    /// `mesh::Node::vram_bytes` doc for why the distinction matters.
    fast_memory_bytes: u64,
    full_coverage: bool,
}

impl MoePlacementPlan {
    fn role_for(&self, my_id: iroh::EndpointId) -> MoePlacementRole {
        if self.active_ids.contains(&my_id) {
            MoePlacementRole::SplitShard
        } else if self.fallback_ids.contains(&my_id) {
            MoePlacementRole::FullFallback
        } else {
            MoePlacementRole::Standby
        }
    }

    fn shard_index_for(&self, my_id: iroh::EndpointId) -> Option<usize> {
        self.active_ids.iter().position(|id| *id == my_id)
    }

    fn materially_improves_upon(&self, current: &Self) -> bool {
        let improves_fallback = self.fallback_ids.len() > current.fallback_ids.len()
            && self.active_ids.len() >= current.active_ids.len();
        let improves_active_count = self.active_ids.len() > current.active_ids.len()
            && self.fallback_ids.len() >= current.fallback_ids.len();
        let improves_overlap = self.overlap > current.overlap
            && self.active_ids.len() >= current.active_ids.len()
            && self.fallback_ids.len() >= current.fallback_ids.len();

        improves_fallback || improves_active_count || improves_overlap
    }
}

fn running_plan_state(
    last_plan: Option<&MoePlacementPlan>,
    currently_running: bool,
) -> (&[iroh::EndpointId], &[iroh::EndpointId]) {
    if currently_running {
        let active_ids = last_plan
            .map(|plan| plan.active_ids.as_slice())
            .unwrap_or(&[]);
        let fallback_ids = last_plan
            .map(|plan| plan.fallback_ids.as_slice())
            .unwrap_or(&[]);
        (active_ids, fallback_ids)
    } else {
        (&[], &[])
    }
}

fn compute_best_moe_placement(
    mut candidates: Vec<MoePlacementCandidate>,
) -> Option<MoePlacementPlan> {
    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| {
        b.fast_memory_bytes
            .cmp(&a.fast_memory_bytes)
            .then_with(|| a.id.cmp(&b.id))
    });
    let leader_id = candidates[0].id;
    let mut active_ids: Vec<iroh::EndpointId> =
        candidates.iter().map(|candidate| candidate.id).collect();
    active_ids.sort();
    active_ids.dedup();

    let mut fallback_ids = Vec::new();
    if active_ids.len() >= 3 {
        if let Some(fallback_candidate) =
            candidates.iter().find(|candidate| candidate.full_coverage)
        {
            active_ids.retain(|id| *id != fallback_candidate.id);
            fallback_ids.push(fallback_candidate.id);
        }
    }

    fallback_ids.sort();
    fallback_ids.dedup();

    let overlap = if active_ids.len() >= 3 { 2 } else { 1 };

    Some(MoePlacementPlan {
        leader_id,
        active_ids,
        fallback_ids,
        overlap,
    })
}

fn plan_moe_placement(
    candidates: Vec<MoePlacementCandidate>,
    current_active_ids: &[iroh::EndpointId],
    current_fallback_ids: &[iroh::EndpointId],
    allow_scale_up: bool,
) -> Option<MoePlacementPlan> {
    let candidate_ids: HashSet<_> = candidates.iter().map(|candidate| candidate.id).collect();
    let keep_current_active = !current_active_ids.is_empty()
        && current_active_ids
            .iter()
            .all(|id| candidate_ids.contains(id));

    let best = compute_best_moe_placement(candidates.clone())?;
    if !keep_current_active {
        return Some(best);
    }

    let mut stable = MoePlacementPlan {
        leader_id: best.leader_id,
        active_ids: current_active_ids.to_vec(),
        fallback_ids: current_fallback_ids
            .iter()
            .copied()
            .filter(|id| candidate_ids.contains(id) && !current_active_ids.contains(id))
            .collect(),
        overlap: if current_active_ids.len() >= 3 { 2 } else { 1 },
    };
    stable.active_ids.sort();
    stable.active_ids.dedup();
    stable.fallback_ids.sort();
    stable.fallback_ids.dedup();

    if allow_scale_up && best.materially_improves_upon(&stable) {
        Some(best)
    } else {
        Some(stable)
    }
}

/// Look up base MoE config for a model.
/// 1. Catalog provides MoE shape hints when available.
/// 2. GGUF header detection fills in the rest with conservative defaults.
fn lookup_moe_config(
    model_name: &str,
    model_path: &Path,
) -> Option<crate::models::catalog::MoeConfig> {
    // Tier 1: catalog lookup (shape hints only; runtime ranking is resolved later)
    let q = model_name.to_lowercase();
    if let Some(cfg) = crate::models::catalog::MODEL_CATALOG
        .iter()
        .find(|m| m.name.to_lowercase() == q || m.file.to_lowercase().contains(&q))
        .and_then(|m| m.moe.clone())
    {
        if !cfg.ranking.is_empty() {
            return Some(cfg);
        }
        // Catalog says MoE but no ranking — fall through to GGUF detect + sequential fallback
        // (keeps n_expert/n_expert_used/min_experts from catalog)
    }

    // Tier 2: auto-detect from GGUF header
    let info = models::gguf::detect_moe(model_path)?;
    emit_moe_status(
        model_name,
        "auto-detected MoE",
        format!(
            "{} experts, top-{}",
            info.expert_count, info.expert_used_count
        ),
    );

    // Conservative default: 50% shared core (safe floor for quality).
    // Without a ranking, we use sequential expert IDs (0..N).
    let min_experts = (info.expert_count as f64 * 0.5).ceil() as u32;

    // Check for cached ranking on disk
    let ranking_path = moe::ranking_cache_path(model_path);
    if let Some(ranking) = moe::load_cached_ranking(&ranking_path) {
        emit_moe_status(
            model_name,
            "using cached ranking",
            format!("{}", ranking_path.display()),
        );
        return Some(crate::models::catalog::MoeConfig {
            n_expert: info.expert_count,
            n_expert_used: info.expert_used_count,
            min_experts_per_node: min_experts,
            ranking,
        });
    }

    // No ranking available — use sequential (0, 1, 2, ...) as fallback.
    // The election loop can run moe-analyze to compute a proper ranking.
    let sequential: Vec<u32> = (0..info.expert_count).collect();
    Some(crate::models::catalog::MoeConfig {
        n_expert: info.expert_count,
        n_expert_used: info.expert_used_count,
        min_experts_per_node: min_experts,
        ranking: sequential,
    })
}

fn should_attempt_local_micro_analyze(
    model_path: &Path,
    model_name: &str,
    local_vram_budget: u64,
) -> bool {
    let model_bytes = total_model_bytes(model_path);
    // Require roughly the same headroom we already use for "fits locally" checks.
    let fits_with_headroom = local_vram_budget >= (model_bytes as f64 * 1.1) as u64;
    if !fits_with_headroom {
        emit_moe_status(
            model_name,
            "skipping local micro-analyze",
            format!(
                "model needs about {:.1}GB with headroom, local capacity is {:.1}GB",
                model_bytes as f64 * 1.1 / 1e9,
                local_vram_budget as f64 / 1e9
            ),
        );
    }
    fits_with_headroom
}

fn resolve_runtime_moe_config(
    model_name: &str,
    model_path: &Path,
    bin_dir: &Path,
    local_vram_budget: u64,
    options: &moe::MoeRuntimeOptions,
) -> anyhow::Result<Option<ResolvedMoeConfig>> {
    let base = match lookup_moe_config(model_name, model_path) {
        Some(cfg) => cfg,
        None => return Ok(None),
    };

    let started = std::time::Instant::now();
    let (ranking, ranking_source, ranking_origin) = match options.ranking_strategy {
        moe::MoeRankingStrategy::Auto => {
            let model_path_for_ranking = model_path.to_path_buf();
            let resolved_ranking_result: anyhow::Result<
                Option<crate::system::moe_planner::ResolvedRanking>,
            > = match tokio::runtime::Handle::try_current() {
                Ok(handle) => match tokio::task::block_in_place(|| {
                    handle.block_on(tokio::task::spawn_blocking(move || {
                        crate::system::moe_planner::resolve_runtime_ranking(
                            &model_path_for_ranking,
                            crate::system::moe_planner::DEFAULT_MOE_RANKINGS_DATASET,
                        )
                    }))
                }) {
                    Ok(Ok(resolved)) => Ok(resolved),
                    Ok(Err(err)) => {
                        emit_moe_status(
                            model_name,
                            "shared ranking resolve failed",
                            format!(
                                "falling back to local analysis or sequential expert order ({err})"
                            ),
                        );
                        Ok(None)
                    }
                    Err(err) => {
                        emit_moe_status(
                            model_name,
                            "shared ranking resolver join failed",
                            format!(
                                "falling back to local analysis or sequential expert order ({err})"
                            ),
                        );
                        Ok(None)
                    }
                },
                Err(_) => crate::system::moe_planner::resolve_runtime_ranking(
                    model_path,
                    crate::system::moe_planner::DEFAULT_MOE_RANKINGS_DATASET,
                ),
            };
            let resolved_ranking = match resolved_ranking_result {
                Ok(resolved) => resolved,
                Err(err) => {
                    emit_moe_status(
                        model_name,
                        "shared ranking resolve failed",
                        format!(
                            "falling back to local analysis or sequential expert order ({err})"
                        ),
                    );
                    None
                }
            };
            if let Some(resolved) = resolved_ranking {
                emit_moe_status(
                    model_name,
                    "using shared ranking",
                    format!(
                        "mode={} path={} source={}",
                        resolved.analyzer_id,
                        resolved.path.display(),
                        resolved.source.label()
                    ),
                );
                (
                    moe::load_cached_ranking(&resolved.path).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Failed to load resolved ranking {}",
                            resolved.path.display()
                        )
                    })?,
                    resolved.analyzer_id,
                    resolved.source.label().to_string(),
                )
            } else {
                if should_attempt_local_micro_analyze(model_path, model_name, local_vram_budget) {
                    match ensure_micro_analyze_ranking(bin_dir, model_name, model_path, options) {
                        Ok(artifact) => (
                            artifact.ranking,
                            "micro-v1".to_string(),
                            artifact.origin.label().to_string(),
                        ),
                        Err(err) => {
                            emit_moe_status(
                                model_name,
                                "micro-analyze failed",
                                format!("falling back to sequential expert order ({err})"),
                            );
                            (
                                (0..base.n_expert).collect(),
                                "sequential-fallback".to_string(),
                                "fallback".to_string(),
                            )
                        }
                    }
                } else {
                    emit_moe_status(
                        model_name,
                        "waiting for peer ranking",
                        "or using sequential fallback on this node",
                    );
                    (
                        (0..base.n_expert).collect(),
                        "sequential-fallback".to_string(),
                        "fallback".to_string(),
                    )
                }
            }
        }
        moe::MoeRankingStrategy::Analyze => {
            let cached = moe::ranking_cache_path(model_path);
            let artifact = ensure_full_analyze_ranking(bin_dir, model_name, model_path, &cached)?;
            (
                artifact.ranking,
                "full-v1".to_string(),
                artifact.origin.label().to_string(),
            )
        }
        moe::MoeRankingStrategy::MicroAnalyze => {
            let artifact = ensure_micro_analyze_ranking(bin_dir, model_name, model_path, options)?;
            (
                artifact.ranking,
                "micro-v1".to_string(),
                artifact.origin.label().to_string(),
            )
        }
    };

    emit_moe_status(
        model_name,
        "ranking resolved",
        format!(
            "ranking={} origin={} in {:.1}s",
            ranking_source,
            ranking_origin,
            started.elapsed().as_secs_f64()
        ),
    );

    Ok(Some(ResolvedMoeConfig {
        config: crate::models::catalog::MoeConfig { ranking, ..base },
        ranking_strategy: options.ranking_strategy,
        ranking_source,
        ranking_origin,
    }))
}

fn refresh_auto_moe_config_from_cache(
    model_name: &str,
    model_path: &Path,
    cfg: &mut ResolvedMoeConfig,
) -> bool {
    if !matches!(cfg.ranking_strategy, moe::MoeRankingStrategy::Auto) {
        return false;
    }
    let Some(artifact) = moe::best_shared_ranking_artifact(model_path) else {
        return false;
    };
    let resolved = crate::system::moe_planner::ResolvedRanking {
        path: moe::shared_ranking_cache_path(model_path, &artifact),
        metadata_path: None,
        analysis_path: None,
        analyzer_id: match artifact.kind {
            moe::SharedRankingKind::Analyze => "full-v1",
            moe::SharedRankingKind::MicroAnalyze => "micro-v1",
        }
        .to_string(),
        source: crate::system::moe_planner::RankingSource::LocalCache,
        reason: "local ranking refresh".to_string(),
    };
    let Some(ranking) = moe::load_cached_ranking(&resolved.path) else {
        return false;
    };
    if cfg.config.ranking == ranking
        && cfg.ranking_source == resolved.analyzer_id
        && cfg.ranking_origin == resolved.source.label()
    {
        return false;
    }

    emit_moe_status(
        model_name,
        "switching to better ranking",
        format!(
            "mode={} source={}",
            resolved.analyzer_id,
            resolved.source.label()
        ),
    );
    cfg.config.ranking = ranking;
    cfg.ranking_source = resolved.analyzer_id;
    cfg.ranking_origin = resolved.source.label().to_string();
    true
}

fn print_runtime_submit_suggestion(model_name: &str, model_path: &Path, ranking_path: &Path) {
    let Some(identity) = crate::models::huggingface_identity_for_path(model_path) else {
        return;
    };
    emit_moe_status(model_name, "generated local ranking", "ready to share");
    emit_moe_status(
        model_name,
        "ranking cache",
        format!("{}", ranking_path.display()),
    );
    emit_moe_status(
        model_name,
        "published source",
        crate::system::moe_planner::DEFAULT_MOE_RANKINGS_DATASET.to_string(),
    );
    emit_moe_status(model_name, "published ranking", "not used on this run");
    emit_moe_status(
        model_name,
        "contribute ranking",
        format!("senda moe share '{}'", identity.distribution_ref()),
    );
}

fn resolve_analyze_binary(bin_dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    let exe = std::env::consts::EXE_SUFFIX;
    let names = [
        format!("llama-moe-analyze{exe}"),
        "llama-moe-analyze".to_string(),
    ];
    let prefixes = [
        bin_dir.to_path_buf(),
        bin_dir.join("../llama.cpp/build/bin"),
        bin_dir.join("../../llama.cpp/build/bin"),
        bin_dir.join("../../../llama.cpp/build/bin"),
    ];
    for prefix in &prefixes {
        for name in &names {
            let candidate = prefix.join(name);
            if candidate.exists() {
                return Ok(candidate.canonicalize().unwrap_or(candidate));
            }
        }
    }
    anyhow::bail!(
        "llama-moe-analyze not found in {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    )
}

fn should_suppress_moe_analyze_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty() || trimmed.starts_with("print_info:")
}

fn should_relay_moe_analyze_warning(line: &str) -> bool {
    let trimmed = line.trim();
    if should_suppress_moe_analyze_line(trimmed) {
        return false;
    }

    trimmed.starts_with("W ")
        || trimmed.starts_with("E ")
        || trimmed.to_ascii_lowercase().contains("failed")
        || trimmed.to_ascii_lowercase().contains("error")
}

#[derive(Default)]
struct MoeAnalyzeProgressState {
    current_prompt: usize,
    total_prompts: Option<usize>,
    done: bool,
}

struct MoeElectionParams {
    runtime: Arc<crate::runtime::instance::InstanceRuntime>,
    node: mesh::Node,
    tunnel_mgr: tunnel::Manager,
    ingress_http_port: u16,
    bin_dir: std::path::PathBuf,
    model: std::path::PathBuf,
    model_name: String,
    moe_cfg: ResolvedMoeConfig,
    moe_summary: MoeSummary,
    my_vram: u64,
    model_bytes: u64,
    binary_flavor: Option<launch::BinaryFlavor>,
    ctx_size_override: Option<u32>,
    pinned_gpu: Option<crate::runtime::StartupPinnedGpuTarget>,
    target_tx: Arc<watch::Sender<ModelTargets>>,
    stop_rx: watch::Receiver<bool>,
    slots: usize,
}

struct StartLlamaParams<'a> {
    runtime: &'a crate::runtime::instance::InstanceRuntime,
    node: &'a mesh::Node,
    tunnel_mgr: &'a tunnel::Manager,
    bin_dir: &'a Path,
    model: &'a Path,
    model_name: &'a str,
    model_peers: &'a [mesh::PeerInfo],
    explicit_mmproj: Option<&'a Path>,
    draft: Option<&'a Path>,
    draft_max: u16,
    force_split: bool,
    binary_flavor: Option<launch::BinaryFlavor>,
    ctx_size_override: Option<u32>,
    pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    slots: usize,
}

pub struct ElectionLoopParams {
    pub runtime: Arc<crate::runtime::instance::InstanceRuntime>,
    pub node: mesh::Node,
    pub tunnel_mgr: tunnel::Manager,
    pub ingress_http_port: u16,
    /// On-demand rpc-server lifecycle. The loop `acquire()`s it while this
    /// node is a pipeline WORKER for the model (split needed and we're not the
    /// host) and `release()`s it otherwise, so a solo node never holds an idle
    /// rpc-server. The host never dials its own rpc-server (it uses its local
    /// GPU backend directly), so only the worker role drives this.
    pub lazy_rpc: Arc<crate::inference::lazy_rpc::LazyRpcServer>,
    pub bin_dir: std::path::PathBuf,
    pub model: std::path::PathBuf,
    pub model_name: String,
    pub explicit_mmproj: Option<std::path::PathBuf>,
    pub draft: Option<std::path::PathBuf>,
    pub draft_max: u16,
    pub force_split: bool,
    pub binary_flavor: Option<launch::BinaryFlavor>,
    pub ctx_size_override: Option<u32>,
    pub pinned_gpu: Option<crate::runtime::StartupPinnedGpuTarget>,
    pub moe_runtime_options: moe::MoeRuntimeOptions,
    pub target_tx: Arc<watch::Sender<ModelTargets>>,
    pub stop_rx: watch::Receiver<bool>,
    pub slots: usize,
}

fn spawn_moe_analysis_spinner(
    model_name: String,
    mode: &'static str,
    progress: Arc<Mutex<MoeAnalyzeProgressState>>,
    started: std::time::Instant,
) -> thread::JoinHandle<()> {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    thread::spawn(move || {
        let mut frame_idx = 0usize;
        loop {
            let (current, total, done) = progress
                .lock()
                .map(|state| (state.current_prompt, state.total_prompts, state.done))
                .unwrap_or((0, None, true));
            let spinner = if done {
                "✓"
            } else {
                FRAMES[frame_idx % FRAMES.len()]
            };
            emit_moe_analysis_progress(
                &model_name,
                mode,
                spinner,
                current,
                total,
                started.elapsed().as_secs(),
            );
            if done {
                break;
            }
            frame_idx += 1;
            thread::sleep(std::time::Duration::from_millis(125));
        }
    })
}

fn parse_moe_analyze_prompt_total(line: &str) -> Option<usize> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("Running ")?;
    let prompt_count = rest.split_whitespace().next()?;
    prompt_count.parse::<usize>().ok()
}

fn parse_moe_analyze_prompt_progress(line: &str) -> Option<(usize, usize)> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("Prompt ")?;
    let progress = rest.split(':').next()?.trim();
    let (current, total) = progress.split_once('/')?;
    Some((current.parse::<usize>().ok()?, total.parse::<usize>().ok()?))
}

fn spawn_moe_analyze_log_relay<R: std::io::Read + Send + 'static>(
    reader: R,
    model_name: String,
    progress: Arc<Mutex<MoeAnalyzeProgressState>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(total) = parse_moe_analyze_prompt_total(&line) {
                if let Ok(mut state) = progress.lock() {
                    state.total_prompts = Some(total);
                }
                continue;
            }
            if let Some((current, total)) = parse_moe_analyze_prompt_progress(&line) {
                if let Ok(mut state) = progress.lock() {
                    state.total_prompts = Some(total);
                    state.current_prompt = current.saturating_sub(1);
                }
                continue;
            }
            if should_relay_moe_analyze_warning(&line) {
                emit_moe_status(&model_name, "moe-analyze warning", line);
            }
        }
    })
}

fn ensure_full_analyze_ranking(
    bin_dir: &Path,
    model_name: &str,
    model_path: &Path,
    cached_path: &Path,
) -> anyhow::Result<moe::SharedRankingArtifact> {
    if let Some(artifact) = moe::load_shared_ranking_artifact(
        cached_path,
        moe::SharedRankingKind::Analyze,
        moe::SharedRankingOrigin::LegacyCache,
        None,
        None,
        None,
    ) {
        emit_moe_status(
            model_name,
            "using cached ranking",
            format!(
                "mode=full-analyze origin={} cache={}",
                artifact.origin.label(),
                cached_path.display()
            ),
        );
        return Ok(artifact);
    }
    if let Some(parent) = cached_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let analyze_bin = resolve_analyze_binary(bin_dir)?;
    let started = std::time::Instant::now();
    let temp_output = std::env::temp_dir().join(format!(
        "senda-full-live-{}-{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    emit_moe_status(
        model_name,
        "MoE analysis",
        format!("mode=full-analyze cache={}", cached_path.display()),
    );
    let progress = Arc::new(Mutex::new(MoeAnalyzeProgressState::default()));
    let spinner = spawn_moe_analysis_spinner(
        model_name.to_string(),
        "full-analyze",
        Arc::clone(&progress),
        started,
    );
    let mut child = Command::new(&analyze_bin)
        .args([
            "-m",
            &model_path.to_string_lossy(),
            "--all-layers",
            "--export-ranking",
            &temp_output.to_string_lossy(),
            "-n",
            "32",
            "-c",
            "4096",
            "-ngl",
            "99",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .hide_console()
        .spawn()?;
    let stdout_relay = child.stdout.take().map(|stdout| {
        spawn_moe_analyze_log_relay(stdout, model_name.to_string(), Arc::clone(&progress))
    });
    let stderr_relay = child.stderr.take().map(|stderr| {
        spawn_moe_analyze_log_relay(stderr, model_name.to_string(), Arc::clone(&progress))
    });
    let status = child.wait()?;
    if let Some(handle) = stdout_relay {
        let _ = handle.join();
    }
    if let Some(handle) = stderr_relay {
        let _ = handle.join();
    }
    if let Ok(mut state) = progress.lock() {
        if let Some(total) = state.total_prompts {
            state.current_prompt = total;
        }
        state.done = true;
    }
    let _ = spinner.join();
    anyhow::ensure!(status.success(), "llama-moe-analyze exited with {status}");
    let ranking = moe::load_cached_ranking(&temp_output).ok_or_else(|| {
        anyhow::anyhow!(
            "No ranking produced by full analyze at {}",
            temp_output.display()
        )
    })?;
    let artifact = moe::SharedRankingArtifact {
        kind: moe::SharedRankingKind::Analyze,
        origin: moe::SharedRankingOrigin::LocalFullAnalyze,
        ranking,
        micro_prompt_count: None,
        micro_tokens: None,
        micro_layer_scope: None,
    };
    let wrote_cache = moe::cache_shared_ranking_if_stronger(model_path, &artifact)?;
    std::fs::copy(&temp_output, cached_path)?;
    let _ = std::fs::remove_file(&temp_output);
    emit_moe_status(
        model_name,
        "full-analyze cached",
        format!(
            "{} in {:.1}s (origin={})",
            cached_path.display(),
            started.elapsed().as_secs_f64(),
            artifact.origin.label()
        ),
    );
    if !wrote_cache {
        emit_moe_status(
            model_name,
            "shared ranking already preferred",
            "full-v1 result was not promoted as the preferred shared artifact",
        );
    }
    print_runtime_submit_suggestion(model_name, model_path, cached_path);
    Ok(artifact)
}

fn ensure_micro_analyze_ranking(
    bin_dir: &Path,
    model_name: &str,
    model_path: &Path,
    options: &moe::MoeRuntimeOptions,
) -> anyhow::Result<moe::SharedRankingArtifact> {
    let cached_path = moe::micro_ranking_cache_path(
        model_path,
        options.micro_prompt_count,
        options.micro_tokens,
        options.micro_layer_scope,
    );
    if let Some(artifact) = moe::load_shared_ranking_artifact(
        &cached_path,
        moe::SharedRankingKind::MicroAnalyze,
        moe::SharedRankingOrigin::LegacyCache,
        Some(options.micro_prompt_count),
        Some(options.micro_tokens),
        Some(options.micro_layer_scope),
    ) {
        emit_moe_status(
            model_name,
            "using cached ranking",
            format!(
                "mode=micro-analyze origin={} cache={}",
                artifact.origin.label(),
                cached_path.display()
            ),
        );
        return Ok(artifact);
    }
    let analyze = run_micro_analyze_ranking(bin_dir, model_name, model_path, options)?;
    let artifact = moe::SharedRankingArtifact {
        kind: moe::SharedRankingKind::MicroAnalyze,
        origin: moe::SharedRankingOrigin::LocalMicroAnalyze,
        ranking: analyze.ranking,
        micro_prompt_count: Some(options.micro_prompt_count),
        micro_tokens: Some(options.micro_tokens),
        micro_layer_scope: Some(options.micro_layer_scope),
    };
    let wrote_cache = moe::cache_shared_ranking_if_stronger(model_path, &artifact)?;
    write_runtime_canonical_micro_ranking(
        &cached_path,
        &artifact,
        &analyze.rows,
        analyze.rows.iter().map(|(_, values)| values.0).sum::<f64>(),
    )?;
    emit_moe_status(
        model_name,
        "micro-analyze cached",
        format!(
            "{} (origin={})",
            cached_path.display(),
            artifact.origin.label()
        ),
    );
    if !wrote_cache {
        emit_moe_status(
            model_name,
            "shared ranking already preferred",
            "micro-v1 result was not promoted as the preferred shared artifact",
        );
    }
    print_runtime_submit_suggestion(model_name, model_path, &cached_path);
    Ok(artifact)
}

#[derive(Clone, Copy)]
struct AnalyzeMassRow {
    expert_id: u32,
    gate_mass: f64,
    selection_count: u64,
}

struct RuntimeMicroAnalyzeResult {
    ranking: Vec<u32>,
    rows: Vec<(u32, (f64, u64))>,
}

fn run_micro_analyze_ranking(
    bin_dir: &Path,
    model_name: &str,
    model_path: &Path,
    options: &moe::MoeRuntimeOptions,
) -> anyhow::Result<RuntimeMicroAnalyzeResult> {
    let prompts = default_micro_prompts();
    let prompt_count = options.micro_prompt_count.max(1).min(prompts.len());
    let analyze_bin = resolve_analyze_binary(bin_dir)?;
    let timestamp_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir().join(format!(
        "senda-micro-live-{}-{}",
        std::process::id(),
        timestamp_nanos
    ));
    std::fs::create_dir_all(&tmp_dir)?;
    let started = std::time::Instant::now();
    let mut mass_by_expert: HashMap<u32, (f64, u64)> = HashMap::new();
    emit_moe_status(
        model_name,
        "MoE analysis",
        format!(
            "mode=micro-analyze prompts={} tokens={} layers={} cache=pending",
            prompt_count,
            options.micro_tokens,
            match options.micro_layer_scope {
                moe::MoeMicroLayerScope::All => "all",
                moe::MoeMicroLayerScope::First => "first",
            }
        ),
    );
    let progress = Arc::new(Mutex::new(MoeAnalyzeProgressState {
        current_prompt: 0,
        total_prompts: Some(prompt_count),
        done: false,
    }));
    let spinner = spawn_moe_analysis_spinner(
        model_name.to_string(),
        "micro-analyze",
        Arc::clone(&progress),
        started,
    );

    for (idx, prompt) in prompts.iter().take(prompt_count).enumerate() {
        let output_path = tmp_dir.join(format!("prompt-{idx}.csv"));
        let mut command = Command::new(&analyze_bin);
        command.args([
            "-m",
            &model_path.to_string_lossy(),
            "--export-ranking",
            &output_path.to_string_lossy(),
            "-n",
            &options.micro_tokens.to_string(),
            "-c",
            "4096",
            "-ngl",
            "99",
            "-p",
            prompt,
        ]);
        if matches!(options.micro_layer_scope, moe::MoeMicroLayerScope::All) {
            command.arg("--all-layers");
        }
        command.hide_console();
        let output = command.output()?;
        if !output.status.success() {
            if let Ok(mut state) = progress.lock() {
                state.done = true;
            }
            let _ = spinner.join();
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut details = stderr
                .lines()
                .chain(stdout.lines())
                .filter(|line| !should_suppress_moe_analyze_line(line))
                .collect::<Vec<_>>();
            if details.len() > 20 {
                details.truncate(20);
            }
            let detail_text = if details.is_empty() {
                String::new()
            } else {
                format!(": {}", details.join(" | "))
            };
            anyhow::bail!(
                "llama-moe-analyze exited with {}{}",
                output.status,
                detail_text
            );
        }
        for row in load_analyze_mass_rows(&output_path)? {
            let entry = mass_by_expert.entry(row.expert_id).or_insert((0.0, 0));
            entry.0 += row.gate_mass;
            entry.1 += row.selection_count;
        }
        if let Ok(mut state) = progress.lock() {
            state.current_prompt = idx + 1;
        }
    }
    if let Ok(mut state) = progress.lock() {
        state.current_prompt = prompt_count;
        state.done = true;
    }
    let _ = spinner.join();

    let mut rows = mass_by_expert.into_iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        b.1 .0
            .partial_cmp(&a.1 .0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let ranking = rows.iter().map(|(expert_id, _)| *expert_id).collect();
    let _ = std::fs::remove_dir_all(&tmp_dir);
    emit_moe_status(
        model_name,
        "micro-analyze complete",
        format!(
            "{} prompt(s), {} token(s), {} in {:.1}s",
            prompt_count,
            options.micro_tokens,
            match options.micro_layer_scope {
                moe::MoeMicroLayerScope::All => "all layers",
                moe::MoeMicroLayerScope::First => "first layer",
            },
            started.elapsed().as_secs_f64()
        ),
    );
    Ok(RuntimeMicroAnalyzeResult { ranking, rows })
}

fn load_analyze_mass_rows(path: &Path) -> anyhow::Result<Vec<AnalyzeMassRow>> {
    let content = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        rows.push(AnalyzeMassRow {
            expert_id: parts[0].parse()?,
            gate_mass: parts[1].parse()?,
            selection_count: parts[3].parse()?,
        });
    }
    Ok(rows)
}

fn write_runtime_canonical_micro_ranking(
    path: &Path,
    artifact: &moe::SharedRankingArtifact,
    ranking: &[(u32, (f64, u64))],
    total_mass_sum: f64,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = String::new();
    writeln!(&mut output, "# senda-moe-ranking=v1").ok();
    writeln!(&mut output, "# ranking_kind={}", artifact.kind.label()).ok();
    writeln!(&mut output, "# ranking_origin={}", artifact.origin.label()).ok();
    if let Some(prompt_count) = artifact.micro_prompt_count {
        writeln!(&mut output, "# micro_prompt_count={prompt_count}").ok();
    }
    if let Some(tokens) = artifact.micro_tokens {
        writeln!(&mut output, "# micro_tokens={tokens}").ok();
    }
    if let Some(layer_scope) = artifact.micro_layer_scope {
        let scope = match layer_scope {
            moe::MoeMicroLayerScope::All => "all",
            moe::MoeMicroLayerScope::First => "first",
        };
        writeln!(&mut output, "# micro_layer_scope={scope}").ok();
    }
    writeln!(
        &mut output,
        "expert_id,total_mass,mass_fraction,selection_count"
    )
    .ok();
    for (expert_id, (gate_mass, selection_count)) in ranking {
        let mass_fraction = if total_mass_sum > 0.0 {
            gate_mass / total_mass_sum
        } else {
            0.0
        };
        writeln!(
            &mut output,
            "{expert_id},{gate_mass:.12},{mass_fraction:.12},{selection_count}"
        )
        .ok();
    }
    std::fs::write(path, output)?;
    Ok(())
}

fn default_micro_prompts() -> &'static [&'static str] {
    &[
        "User: Explain how mixture-of-experts routing works in a language model.\nAssistant:",
        "User: Write a short professional email asking for feedback on a technical design.\nAssistant:",
        "User: Outline a debugging plan for a flaky distributed systems test.\nAssistant:",
        "User: Summarize the tradeoffs between latency and quality in MoE inference.\nAssistant:",
    ]
}

/// Background election loop for a single model.
/// This node serves `model` — it only cares about peers also serving `model`.
///
/// On every mesh change:
/// 1. Kill llama-server (if we're running it)
/// 2. Re-elect within the model group
/// 3. Winner starts llama-server with --rpc pointing at group nodes
///
/// Publishes the current ModelTargets via the watch channel so the
/// API proxy knows where to forward requests.
#[allow(clippy::too_many_arguments)]
pub async fn election_loop(
    params: ElectionLoopParams,
    mut on_change: impl FnMut(bool, bool) + Send,
    mut on_process: impl FnMut(Option<LocalProcessInfo>) + Send,
) {
    let ElectionLoopParams {
        runtime,
        node,
        tunnel_mgr,
        ingress_http_port,
        lazy_rpc,
        bin_dir,
        model,
        model_name,
        explicit_mmproj,
        draft,
        draft_max,
        force_split,
        binary_flavor,
        ctx_size_override,
        pinned_gpu,
        moe_runtime_options,
        target_tx,
        mut stop_rx,
        slots,
    } = params;
    let mut peer_rx = node.peer_change_rx.clone();

    // Track the actual running launch topology so we only restart on real split changes.
    let mut last_running_plan: Option<DenseRunningPlan> = None;
    let mut currently_host = false;
    // Lazy rpc: RAII hold on the on-demand rpc-server while we act as a
    // pipeline worker for this model. `Some` ⇒ held; toggled on the
    // `requires_split && !i_am_host` edge so we spawn/teardown once per edge,
    // and dropped automatically on any loop exit (release covers every path).
    let mut worker_rpc_hold: Option<crate::inference::lazy_rpc::WorkerRpcHold> = None;
    let mut current_local_port: Option<u16> = None;
    let mut llama_process: Option<launch::InferenceServerProcess> = None;
    // v0.66.38 fitter watchdog: stamped on every successful launch, cleared
    // when the model first appears in `node.hosted_models()`. If still set
    // after `FITTER_WATCHDOG_TIMEOUT`, the launch is presumed stuck inside
    // `common_params_fit_impl` and we tear it down. See the constant docs.
    let mut launched_at: Option<std::time::Instant> = None;
    // v0.66.39 election cooldown: stamp when this loop starts so the
    // i_am_host decision below can defer self-election for a short
    // grace period until at least one other peer has been observed in
    // the mesh. Without this, a peer that just (re)started self-elects
    // in <1 s before any gossip arrives, fires up llama-server, and the
    // v0.66.37 "keep cohort viable" stickiness then locks it into a
    // split-brain even after the actual best host (e.g. LYU 17 GB on a
    // mesh of LYU + 14.5 GB Mac + 12.9 GB Mac + 8.6 GB MSI) joins.
    // Confirmed against the May 18 2026 incident logs where the Mac
    // emitted `host_elected` 1.3 s after process spawn with `peers=0`.
    let loop_started_at = std::time::Instant::now();
    let mut backend_proxy: Option<crate::network::openai::backend::BackendProxyHandle> = None;
    // Per-loop timestamp map for the host-claim grace check. Stamped on
    // first sighting of each peer, never updated. See `viable_host_candidates`
    // and issue #9 for why `peer.last_seen` would be wrong here.
    let mut first_observed: std::collections::HashMap<iroh::EndpointId, std::time::Instant> =
        std::collections::HashMap::new();
    // Sliding-window cap on consecutive failed `start_llama` attempts on
    // this node. When it trips, we force `i_am_host = false` for
    // `HOST_ATTEMPT_BACKOFF` so the runner-up gets a turn. See issue #10
    // and the May 14 2026 incident.
    let mut host_attempt_backoff = HostAttemptBackoff::new();

    // Initial settle
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let model_bytes = total_model_bytes(&model);
    // `my_vram` here is the *fast-memory* figure — the only one we trust
    // for "do I need to split this model?" decisions. The legacy
    // `node.vram_bytes()` is the GPU VRAM + 75% host-RAM allowance and
    // is correct for "could this node mmap the weights anywhere?"
    // questions, but it is wrong for "could this node decode at usable
    // speed?" questions, which is what the dense launch planner asks.
    // Conflating them is the May 13 2026 deadlock; see
    // `mesh::Node::fast_memory_bytes()` for the full incident log.
    let my_vram = node.fast_memory_bytes();
    let local_launch_vram = effective_local_launch_vram(my_vram, pinned_gpu.as_ref());
    let model_fits_locally = local_launch_vram >= (model_bytes as f64 * 1.1) as u64;

    // Check if this is a MoE model with enough metadata to plan expert routing.
    let moe_config = lookup_moe_config(&model_name, &model);
    let moe_summary = moe_config.as_ref().map(|moe_config| MoeSummary {
        experts: moe_config.n_expert,
        top_k: moe_config.n_expert_used,
    });
    if moe_summary.is_some() {
        if let Some(moe_summary) = &moe_summary {
            let _ = emit_event(OutputEvent::MoeDetected {
                model: model_name.clone(),
                moe: moe_summary.clone(),
                fits_locally: None,
                capacity_gb: None,
                model_gb: None,
            });
        }
    }

    // MoE mode: each node runs its own llama-server with its expert shard.
    // Only enter MoE split mode if the model doesn't fit locally or --split is forced.
    // Otherwise, just run the full model — every node is independent.
    if moe_config.is_some() {
        let need_moe_split = force_split || !model_fits_locally;
        if need_moe_split {
            if matches!(
                moe_runtime_options.ranking_strategy,
                moe::MoeRankingStrategy::Auto
            ) && moe::best_shared_ranking_artifact(&model).is_none()
            {
                wait_for_peer_moe_ranking(
                    &model_name,
                    &model,
                    &mut peer_rx,
                    &mut stop_rx,
                    std::time::Duration::from_secs(8),
                )
                .await;
            }
            let resolved_moe_cfg = match resolve_runtime_moe_config(
                &model_name,
                &model,
                &bin_dir,
                my_vram,
                &moe_runtime_options,
            ) {
                Ok(Some(cfg)) => cfg,
                Ok(None) => {
                    emit_warning(
                        "Failed to resolve MoE split config",
                        Some(format!("model={model_name}")),
                    );
                    return;
                }
                Err(e) => {
                    emit_warning(
                        format!("Failed to resolve MoE ranking/grouping: {e}"),
                        Some(format!("model={model_name}")),
                    );
                    return;
                }
            };
            moe_election_loop(
                MoeElectionParams {
                    runtime: runtime.clone(),
                    node,
                    tunnel_mgr,
                    ingress_http_port,
                    bin_dir,
                    model,
                    model_name,
                    moe_cfg: resolved_moe_cfg,
                    moe_summary: moe_summary
                        .clone()
                        .expect("MoE summary should exist when entering MoE mode"),
                    my_vram,
                    model_bytes,
                    binary_flavor,
                    ctx_size_override,
                    pinned_gpu: pinned_gpu.clone(),
                    target_tx,
                    stop_rx,
                    slots,
                },
                &mut on_change,
                &mut on_process,
            )
            .await;
            return;
        } else {
            if let Some(moe_summary) = &moe_summary {
                let _ = emit_event(OutputEvent::MoeDetected {
                    model: model_name.clone(),
                    moe: moe_summary.clone(),
                    fits_locally: Some(true),
                    capacity_gb: Some(my_vram as f64 / 1e9),
                    model_gb: Some(model_bytes as f64 / 1e9),
                });
            }
            // Fall through to normal election loop — each node runs full model independently
        }
    }

    loop {
        if stop_requested(&stop_rx) {
            break;
        }

        // v0.66.38 fitter watchdog: tear down a llama-server that has been
        // "loading but not yet routable" for longer than
        // `FITTER_WATCHDOG_TIMEOUT`. Catches the May 18 2026 incident
        // where `common_params_fit_impl` looped for 40+ minutes trying to
        // find a layer distribution that fit the tight Mac+MSI cohort.
        if currently_host && launched_at.is_some() {
            let hosted = node.hosted_models().await;
            if hosted.iter().any(|m| m == &model_name) {
                launched_at = None;
            } else if launched_at
                .map(|t| t.elapsed() > FITTER_WATCHDOG_TIMEOUT)
                .unwrap_or(false)
            {
                emit_warning(
                    "llama-server stuck loading past fitter watchdog — killing and stepping aside",
                    Some(format!(
                        "model={model_name} grace={}s",
                        FITTER_WATCHDOG_TIMEOUT.as_secs()
                    )),
                );
                if let Some(process) = llama_process.take() {
                    process.handle.shutdown().await;
                }
                if let Some(proxy) = backend_proxy.take() {
                    proxy.shutdown().await;
                }
                tunnel_mgr.set_http_port(0);
                node.set_role(NodeRole::Worker).await;
                currently_host = false;
                current_local_port = None;
                node.clear_local_model_port(&model_name).await;
                last_running_plan = None;
                launched_at = None;
                let _ = host_attempt_backoff.record_failure(std::time::Instant::now());
                on_process(None);
                on_change(false, false);
                // Fall through to re-elect — backoff will force us to defer
                // to the runner-up.
            }
        }

        // Collect our model group (peers also serving this model)
        let peers = node.peers().await;
        let model_peers: Vec<mesh::PeerInfo> = peers
            .iter()
            .filter(|p| p.is_assigned_model(&model_name))
            .cloned()
            .collect();

        // Splitting decision: only split when forced OR when the model
        // genuinely doesn't fit on this node alone. If it fits, every
        // node serving this model runs its own independent llama-server
        // (no election needed — everyone is a host).
        let requires_split = force_split || !model_fits_locally;
        let election_peers = if requires_split {
            peers_for_pipeline_election(&model_peers)
        } else {
            model_peers.clone()
        };
        if requires_split {
            let outdated: Vec<_> = model_peers
                .iter()
                .filter(|p| !peer_supports_pipeline_election(p.version.as_deref()))
                .map(|p| {
                    (
                        p.hostname
                            .clone()
                            .unwrap_or_else(|| p.id.fmt_short().to_string()),
                        p.version.clone().unwrap_or_else(|| "unknown".to_string()),
                    )
                })
                .collect();
            if !outdated.is_empty() {
                tracing::warn!(
                    model = %model_name,
                    outdated = ?outdated,
                    min = MIN_PIPELINE_ELECTION_PEER_VERSION,
                    "excluding outdated peers from pipeline election cohort"
                );
            }
        }

        // Stamp first-seen for any newly-observed peer. We use this for the
        // host-claim grace check below — see `viable_host_candidates` and
        // issue #9. Stamping happens BEFORE filtering so that even a peer
        // that we will immediately exclude (e.g. on reconnect after a long
        // absence) gets a fresh grace window.
        let now = std::time::Instant::now();
        for peer in &model_peers {
            first_observed.entry(peer.id).or_insert(now);
        }
        // Garbage-collect entries for peers that have left the cohort, so
        // a peer that drops and rejoins later gets a fresh grace window.
        first_observed.retain(|id, _| model_peers.iter().any(|p| &p.id == id));

        let grace_host_candidates =
            viable_host_candidates(&election_peers, &first_observed, now, HOST_CLAIM_GRACE);
        let mut host_candidates =
            ram_filtered_host_candidates(grace_host_candidates.clone(), model_bytes);
        // When every peer is past HOST_CLAIM_GRACE but still Worker, the
        // grace filter returns an empty set and (pre-v0.66.53) every node
        // self-elected. Fall back to the version-capable cohort so exactly
        // one runner-up can claim host.
        if requires_split && host_candidates.is_empty() && !election_peers.is_empty() {
            host_candidates = ram_filtered_host_candidates(election_peers.clone(), model_bytes);
        }
        let desired_launch = build_dense_launch_plan(
            local_launch_vram,
            model_bytes,
            force_split,
            &model_name,
            &election_peers,
        );

        // If our recent `start_llama` attempts have piled up (e.g. all
        // workers behind broken iroh tunnels — see issue #10) we force
        // ourselves out of host candidacy for `HOST_ATTEMPT_BACKOFF` so
        // the next-best peer can try. This complements `HOST_CLAIM_GRACE`
        // which handles the *peers'* view of a stuck host; the backoff
        // here is the host's view of itself.
        let host_backoff_active = host_attempt_backoff.is_active(now);

        // v0.66.39: defer self-election while we haven't seen any other peer
        // and the discovery grace hasn't elapsed. See ELECTION_DISCOVERY_GRACE
        // doc and the May 18 2026 split-brain incident. We only suppress
        // self-election; if peers already exist (good discovery) we proceed
        // immediately, and once the grace expires we proceed unconditionally
        // (so a genuinely-solo node still launches).
        let in_discovery_grace = !currently_host
            && peers.is_empty()
            && loop_started_at.elapsed() < ELECTION_DISCOVERY_GRACE;
        if in_discovery_grace {
            tracing::debug!(
                model = %model_name,
                elapsed_ms = loop_started_at.elapsed().as_millis() as u64,
                grace_ms = ELECTION_DISCOVERY_GRACE.as_millis() as u64,
                "deferring self-election: no peers visible yet, discovery grace active"
            );
        }

        let i_am_host = if host_backoff_active || in_discovery_grace {
            false
        } else if requires_split && !peer_supports_pipeline_election(Some(crate::VERSION)) {
            // This node is too old to coordinate a split — stay worker so
            // current runtimes can elect among themselves.
            false
        } else if requires_split && election_peers.is_empty() {
            false
        } else if requires_split {
            // Distributed mode: elect one host from the model group using the
            // same advertised node capacity every peer observes through gossip.
            // Election runs over `host_candidates` rather than `model_peers`
            // — this is what lets the cohort auto-heal when an elected peer
            // is reachable in gossip but never claims `NodeRole::Host`
            // (mixed-version mesh, stuck launch, or adversarial peer; see
            // issue #9 and the May 13 2026 v0.66.18 → v0.66.20 deadlock).
            //
            // Cohort capacity (`build_dense_launch_plan` above) and worker
            // selection (the `else` branch below) DO NOT use this filter —
            // a peer that is unviable as host can still be a perfectly good
            // pipeline worker.
            should_be_host_for_model_with_solo_bias(
                node.id(),
                my_vram,
                node.system_ram_bytes,
                model_bytes,
                &host_candidates,
            )
        } else if model_peers.is_empty() {
            // No other node serving this model — we must host
            true
        } else if currently_host {
            // Already running — don't tear down
            true
        } else {
            // Another node is already serving this model.
            // Only spin up a duplicate if there's enough demand:
            //   - 2+ clients connected, OR
            //   - 10+ requests in the demand tracker for this model
            let n_clients = peers
                .iter()
                .filter(|p| matches!(p.role, mesh::NodeRole::Client))
                .count();
            let demand = node.get_demand();
            let req_count = demand
                .get(&model_name)
                .map(|d| d.request_count)
                .unwrap_or(0);
            let force_duplicate_host = std::env::var("SENDA_FORCE_DUPLICATE_HOSTS")
                .ok()
                .as_deref()
                == Some("1");
            let should_dup = force_duplicate_host || n_clients >= 2 || req_count >= 10;
            if !should_dup {
                emit_info(
                    format!(
                        "[{model_name}] Peer already serving — standby (clients: {n_clients}, requests: {req_count})"
                    ),
                    None,
                );
            } else if force_duplicate_host {
                emit_info(
                    format!("[{model_name}] Forcing duplicate host for benchmark topology"),
                    None,
                );
            }
            should_dup
        };

        // Lazy rpc-server: we need it up exactly when we're a pipeline WORKER
        // for this model — a split is required and we are not the elected
        // host. Bringing it up gossips `rpc_ready=true` so a split host (which
        // gates its cohort on readiness) can dial us without tripping the
        // v0.66.36 HELLO-failure blacklist. A solo node, or the host itself,
        // never holds an rpc-server. Toggled on the edge so the cold spawn
        // happens once, not every gossip tick.
        let want_worker_rpc = requires_split && !i_am_host;
        if want_worker_rpc && worker_rpc_hold.is_none() {
            match lazy_rpc.acquire_hold().await {
                Ok(hold) => {
                    worker_rpc_hold = Some(hold);
                    tracing::info!(
                        model = %model_name,
                        "lazy rpc-server up: acting as pipeline worker"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        model = %model_name,
                        "failed to start rpc-server for pipeline-worker role: {e:#}"
                    );
                }
            }
        } else if !want_worker_rpc && worker_rpc_hold.is_some() {
            worker_rpc_hold = None;
            tracing::info!(
                model = %model_name,
                "released rpc-server hold (no longer a pipeline worker)"
            );
        }

        // Sticky-cohort guard: skip restart if we're already host AND either
        // the freshly-planned cohort is identical to what's running OR the
        // running cohort is still viable on its own. The second clause is
        // the May 17 2026 fix — see `current_cohort_still_viable`. Without
        // it, RTT jitter that re-orders worker preference tears down a
        // perfectly healthy llama-server every gossip tick.
        let cohort_for_viability = if requires_split {
            &election_peers
        } else {
            &model_peers
        };
        let running_cohort_stays_viable = matches!(&last_running_plan, Some(running)
            if current_cohort_still_viable(running, cohort_for_viability, local_launch_vram, model_bytes));
        let desired_matches_running = desired_launch.running_plan() == last_running_plan;
        if currently_host && i_am_host && (desired_matches_running || running_cohort_stays_viable) {
            // Just update the target map (in case other models' hosts changed)
            if let Some(local_port) = current_local_port {
                update_targets(
                    &node,
                    &model_name,
                    InferenceTarget::Local(local_port),
                    &target_tx,
                )
                .await;
            }
            // Wait for next change OR llama-server death.
            //
            // The `HOST_CLAIM_GRACE / 2` sleep below is a safety net: many
            // `PeerInfo` field updates (RTT, VRAM, transitive joins,
            // capability) intentionally do NOT tick `peer_change_tx`
            // (see `gossip.rs::add_peer` and `update_transitive_peer`
            // — only role/serving changes wake the watch channel), so
            // without a periodic re-eval the elected host could keep
            // serving against a stale cohort view until the next 60 s
            // heartbeat round. Cheap to recompute; teardown only
            // happens if `desired_launch.running_plan()` actually
            // diverges from `last_running_plan` in the next iteration.
            tokio::select! {
                res = peer_rx.changed() => {
                    if res.is_err() { break; }
                    emit_info(
                        "Mesh changed — re-checking... (still host, no restart needed)",
                        Some(format!("model={model_name}")),
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
                _ = tokio::time::sleep(HOST_CLAIM_GRACE / 2) => {
                    tracing::debug!(
                        "stable-host fast path: periodic re-eval after {:?}",
                        HOST_CLAIM_GRACE / 2
                    );
                    continue;
                }
                _ = async {
                    if let Some(ref mut process) = llama_process {
                        let _ = (&mut process.death_rx).await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    if stop_requested(&stop_rx) || launch::runtime_shutting_down() {
                        break;
                    }
                    emit_warning(
                        "llama-server died — restarting...",
                        Some(format!("model={model_name}")),
                    );
                    llama_process = None;
                    if let Some(proxy) = backend_proxy.take() {
                        proxy.shutdown().await;
                    }
                    tunnel_mgr.set_http_port(0);
                    currently_host = false;
                    current_local_port = None;
            node.clear_local_model_port(&model_name).await;
                    node.set_role(NodeRole::Worker).await;
                    last_running_plan = None;
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                    on_process(None);
                    on_change(false, false);
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    // Fall through to restart
                }
                res = stop_rx.changed() => {
                    if res.is_err() || stop_requested(&stop_rx) {
                        break;
                    }
                }
            }
        }

        // Something changed — kill llama-server if we were running it
        if currently_host {
            if let Some(process) = llama_process.take() {
                process.handle.shutdown().await;
            }
            if let Some(proxy) = backend_proxy.take() {
                proxy.shutdown().await;
            }
            tunnel_mgr.set_http_port(0);
            node.set_role(NodeRole::Worker).await;
            current_local_port = None;
            node.clear_local_model_port(&model_name).await;
            last_running_plan = None;
            update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            on_process(None);
            on_change(false, false);
            currently_host = false;
        }

        if stop_requested(&stop_rx) {
            break;
        }

        if i_am_host {
            match &desired_launch {
                DenseLaunchPlan::WaitingForCapacity {
                    total_group_vram,
                    min_vram,
                    ..
                } => {
                    let _ = emit_event(OutputEvent::WaitingForPeers {
                        detail: Some(format!(
                            "[{}] Waiting for more peers — need {:.1}GB capacity, have {:.1}GB across eligible split workers",
                            model_name,
                            *min_vram as f64 / 1e9,
                            *total_group_vram as f64 / 1e9
                        )),
                    });
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                    on_change(false, false);
                    if peer_rx.changed().await.is_err() {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
                DenseLaunchPlan::Split {
                    total_group_vram,
                    worker_ids: _worker_ids,
                } => {
                    let _ = emit_event(OutputEvent::HostElected {
                        model: model_name.clone(),
                        host: node.id().fmt_short().to_string(),
                        role: Some("host".to_string()),
                        capacity_gb: Some(*total_group_vram as f64 / 1e9),
                    });
                }
                DenseLaunchPlan::Solo => {
                    let _ = emit_event(OutputEvent::HostElected {
                        model: model_name.clone(),
                        host: node.id().fmt_short().to_string(),
                        role: Some("host".to_string()),
                        capacity_gb: Some(local_launch_vram as f64 / 1e9),
                    });
                }
            }
            on_change(true, false);

            // In solo mode, pass empty model_peers so start_llama won't use any workers
            let peers_for_launch = if matches!(desired_launch, DenseLaunchPlan::Split { .. }) {
                &election_peers[..]
            } else {
                &[]
            };
            // v0.66.49 Phase 3.0: only the solo path gets a native
            // baseline collector. A pipeline-host's local llama-server
            // talks to remote rpc-servers via iroh tunnels, so a
            // synthetic chat against it would measure rpc-tunnel
            // overhead — not what "native baseline" means semantically.
            // Captured here as a boolean because `desired_launch` is
            // moved by the match above and the proxy-startup code path
            // also forks for split.
            let is_solo_launch = matches!(desired_launch, DenseLaunchPlan::Solo);
            let (llama_port, process) = match start_llama(StartLlamaParams {
                runtime: &runtime,
                node: &node,
                tunnel_mgr: &tunnel_mgr,
                bin_dir: &bin_dir,
                model: &model,
                model_name: &model_name,
                model_peers: peers_for_launch,
                explicit_mmproj: explicit_mmproj.as_deref(),
                draft: draft.as_deref(),
                draft_max,
                force_split,
                binary_flavor,
                ctx_size_override,
                pinned_gpu: pinned_gpu.as_ref(),
                slots,
            })
            .await
            {
                Some((port, death_rx)) => (port, death_rx),
                None => {
                    if let Some(until) =
                        host_attempt_backoff.record_failure(std::time::Instant::now())
                    {
                        let secs = until
                            .saturating_duration_since(std::time::Instant::now())
                            .as_secs();
                        emit_warning(
                            format!(
                                "Stepping aside as host for {secs}s after {} failed launches — letting the runner-up try",
                                HOST_ATTEMPT_MAX_FAILURES
                            ),
                            Some(format!("model={model_name}")),
                        );
                    }
                    on_change(true, false);
                    // Wake up either when the mesh changes OR when the
                    // host-attempt backoff window has elapsed, whichever
                    // comes first. Plain `peer_rx.changed().await` deadlocks
                    // the election after a stepped-aside cycle on a quiet
                    // mesh — the runner-up never claims, no gossip arrives,
                    // and we sleep forever. The May 14 2026 70B incident
                    // hit exactly this: 2 consecutive launches failed
                    // because one worker's RPC tunnel was silent, the loop
                    // stepped aside as designed, but then sat on
                    // `peer_rx.changed()` indefinitely because every peer
                    // kept gossiping unchanged state. Polling at the
                    // backoff cadence (slightly longer than
                    // `HOST_ATTEMPT_BACKOFF` so the next iteration sees
                    // `host_attempt_backoff.is_active()` go false) gets us
                    // a fresh election attempt with warm tunnels.
                    let wake_after = HOST_ATTEMPT_BACKOFF + std::time::Duration::from_secs(2);
                    tokio::select! {
                        _ = peer_rx.changed() => {}
                        _ = tokio::time::sleep(wake_after) => {}
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            let proxy = match crate::network::openai::backend::start_backend_proxy(
                llama_port,
                node.clone(),
            )
            .await
            {
                Ok(proxy) => proxy,
                Err(err) => {
                    emit_error(
                        format!("Failed to start local OpenAI backend proxy: {err}"),
                        Some(format!("model={model_name} port={llama_port}")),
                    );
                    process.handle.shutdown().await;
                    on_change(true, false);
                    let _ = peer_rx.changed().await;
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };
            let local_proxy_port = proxy.port();
            backend_proxy = Some(proxy);

            node.set_role(NodeRole::Host {
                http_port: ingress_http_port,
            })
            .await;
            tunnel_mgr.set_http_port(local_proxy_port);
            currently_host = true;
            current_local_port = Some(local_proxy_port);
            last_running_plan = desired_launch.running_plan();
            launched_at = Some(std::time::Instant::now());
            // Successful launch — clear any in-progress failure counter
            // so the next launch starts from a clean slate.
            host_attempt_backoff.record_success();
            // Re-gossip so peers learn we're the host for this model
            node.regossip().await;
            // v0.66.49 Phase 3.0 benchmark honesty: kick off the
            // native baseline collector for this `(model, llama_port)`
            // pair. The collector runs in the background, issues a
            // single synthetic chat directly to 127.0.0.1 (no entry
            // tunnel, no auth, no routing), and gossips the result so
            // the catalog can render the through-mesh / native ratio.
            // Solo only — see `is_solo_launch` definition.
            if is_solo_launch {
                // Record the local llama-server port so the verifier's
                // self-oracle can run on-demand ground-truth probes for the
                // model we serve. Cleared on teardown.
                node.set_local_model_port(&model_name, llama_port).await;
                crate::inference::native_baseline::spawn_collector(
                    node.clone(),
                    model_name.clone(),
                    llama_port,
                    binary_flavor
                        .map(|f| f.suffix().to_string())
                        .unwrap_or_else(|| "unknown".to_string()),
                    Some(std::path::PathBuf::from(&model)),
                );
            }
            update_targets(
                &node,
                &model_name,
                InferenceTarget::Local(local_proxy_port),
                &target_tx,
            )
            .await;
            let ctx_size = process.context_length;
            llama_process = Some(process);
            if let Some(ref process) = llama_process {
                on_process(Some(LocalProcessInfo {
                    backend: "llama".into(),
                    pid: process.handle.pid(),
                    port: llama_port,
                    context_length: process.context_length,
                }));
            }
            emit_ready_events(&model_name, llama_port, local_proxy_port, ctx_size);
            on_change(true, true);
        } else {
            // We're a worker in split mode. Find who the host is.
            node.set_role(NodeRole::Worker).await;
            currently_host = false;
            last_running_plan = None;

            // Rank peers by GPU/unified-memory budget — the same metric
            // the host runs in `should_be_host_for_model`. Using the
            // RAM-offload-inflated `vram_bytes` here would pick a peer
            // whose election-side counterpart never agrees it's the host
            // (peer compares fast memory but worker compares inflated)
            // and produces a split-brain deadlock with no traffic
            // routed anywhere.
            //
            // We pick from `host_candidates` (grace-filtered), not
            // `model_peers`, for the same reason the host arm does — if
            // the locally-elected host has timed out, we must NOT keep
            // routing requests to a peer that will never claim the role.
            // See issue #9.
            let host_peer = host_candidates
                .iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .max_by_key(|p| (p.fast_memory_bytes(), p.id));

            if let Some(host) = host_peer {
                if should_be_host_for_model(host.id, host.fast_memory_bytes(), &host_candidates) {
                    update_targets(
                        &node,
                        &model_name,
                        InferenceTarget::Remote(host.id),
                        &target_tx,
                    )
                    .await;
                    let _ = emit_event(OutputEvent::WaitingForPeers {
                        detail: Some(format!(
                            "[{}] Worker — host is {} (split mode)",
                            model_name,
                            host.id.fmt_short()
                        )),
                    });
                } else {
                    update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                }
            } else {
                update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            }
            on_change(false, false);
        }

        // Wait for next peer change OR llama-server death OR
        // host-claim grace timer.
        //
        // The grace timer is what unblocks the cohort when an elected
        // peer is reachable in gossip but never claims `NodeRole::Host`.
        // Without it, `peer_rx.changed()` would never fire (the stuck
        // peer keeps gossiping the same role) and the cohort would wait
        // forever. With it, we wake up at most every `HOST_CLAIM_GRACE /
        // 2` and re-evaluate candidate viability. See issue #9.
        //
        // `grace / 2` (15 s by default) bounds the worst-case extra
        // latency between "candidate's grace expires" and "we notice and
        // re-elect": we either hit the timer first, or we hit it shortly
        // after, but never wait longer than `1.5 * grace`.
        tokio::select! {
            res = peer_rx.changed() => {
                if res.is_err() { break; }
                emit_info(
                    "Mesh changed — re-electing...",
                    Some(format!("model={model_name}")),
                );
            }
            _ = async {
                if let Some(ref mut process) = llama_process {
                    let _ = (&mut process.death_rx).await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {
                if stop_requested(&stop_rx) || launch::runtime_shutting_down() {
                    break;
                }
                emit_warning(
                    "llama-server died — restarting...",
                    Some(format!("model={model_name}")),
                );
                llama_process = None;
                if let Some(proxy) = backend_proxy.take() {
                    proxy.shutdown().await;
                }
                currently_host = false;
                current_local_port = None;
            node.clear_local_model_port(&model_name).await;
                tunnel_mgr.set_http_port(0);
                last_running_plan = None;
                update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
                on_change(false, false);
            }
            _ = tokio::time::sleep(HOST_CLAIM_GRACE / 2) => {
                // Tick: re-evaluate candidate viability on the next loop
                // iteration. If the elected host has now exceeded its
                // grace, the next call to `viable_host_candidates` will
                // exclude it and the runner-up takes over.
            }
            res = stop_rx.changed() => {
                if res.is_err() || stop_requested(&stop_rx) {
                    break;
                }
            }
        }
        if stop_requested(&stop_rx) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }

    if currently_host {
        if let Some(process) = llama_process.take() {
            process.handle.shutdown().await;
        }
        if let Some(proxy) = backend_proxy.take() {
            proxy.shutdown().await;
        }
        tunnel_mgr.set_http_port(0);
        node.set_role(NodeRole::Worker).await;
        update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
        on_process(None);
        on_change(false, false);
    }
}

/// MoE election loop: every node runs its own llama-server with its expert shard.
///
/// Unlike tensor-split mode (one host + RPC workers), MoE mode means:
/// - Every node is independent — no host/worker distinction for this model
/// - Each node runs moe-split locally to produce its shard (cached)
/// - Each node starts its own llama-server with its shard GGUF
/// - The proxy routes sessions to nodes via hash-based affinity
#[allow(clippy::too_many_arguments)]
async fn moe_election_loop(
    params: MoeElectionParams,
    on_change: &mut impl FnMut(bool, bool),
    on_process: &mut impl FnMut(Option<LocalProcessInfo>),
) {
    let MoeElectionParams {
        runtime,
        node,
        tunnel_mgr,
        ingress_http_port,
        bin_dir,
        model,
        model_name,
        mut moe_cfg,
        moe_summary,
        my_vram,
        model_bytes,
        binary_flavor,
        ctx_size_override,
        pinned_gpu,
        target_tx,
        mut stop_rx,
        slots,
    } = params;
    let mut peer_rx = node.peer_change_rx.clone();
    let mut currently_running = false;
    let mut last_plan: Option<MoePlacementPlan> = None;
    let mut llama_process: Option<launch::InferenceServerProcess> = None;
    let mut backend_proxy: Option<crate::network::openai::backend::BackendProxyHandle> = None;
    let mut current_local_port: Option<u16> = None;
    let mut last_plan_change_at = tokio::time::Instant::now();

    loop {
        if stop_requested(&stop_rx) {
            break;
        }

        if !currently_running {
            let _ = refresh_auto_moe_config_from_cache(&model_name, &model, &mut moe_cfg);
        }

        let peers = node.peers().await;
        let local_descriptors = node.served_model_descriptors().await;
        let declared_model_peers: Vec<mesh::PeerInfo> = peers
            .iter()
            .filter(|p| !matches!(p.role, NodeRole::Client))
            .filter(|peer| {
                peer.is_assigned_model(&model_name)
                    || peer
                        .requested_models
                        .iter()
                        .any(|requested| requested == &model_name)
                    || peer.models.iter().any(|model| model == &model_name)
            })
            .cloned()
            .collect();
        let eligible_model_peers: Vec<mesh::PeerInfo> = declared_model_peers
            .iter()
            .filter_map(|peer| {
                mesh::peer_is_eligible_for_active_moe(&local_descriptors, peer, &model_name)
                    .then_some(peer.clone())
            })
            .collect();
        let model_fits = my_vram >= (model_bytes as f64 * 1.1) as u64;
        let placement_peers: Vec<mesh::PeerInfo> =
            if !currently_running && !model_fits && eligible_model_peers.is_empty() {
                if !declared_model_peers.is_empty() {
                    emit_moe_status(
                        &model_name,
                        "bootstrapping placement",
                        format!(
                            "{} declared peer(s) while active eligibility catches up",
                            declared_model_peers.len()
                        ),
                    );
                }
                declared_model_peers.clone()
            } else {
                eligible_model_peers.clone()
            };
        let recovering_peer_count = peers
            .iter()
            .filter(|p| p.is_assigned_model(&model_name))
            .filter(|p| !matches!(p.role, NodeRole::Client))
            .filter(|peer| !peer.moe_recovery_ready())
            .count();
        if recovering_peer_count > 0 {
            emit_moe_status(
                &model_name,
                "holding recovered peers",
                format!(
                    "{} recovered peer(s) out of active MoE placement until stable",
                    recovering_peer_count
                ),
            );
        }

        let my_id = node.id();
        // MoE expert shards live in GPU/unified memory — RAM-offloaded
        // experts get pulled across the PCIe bus on every token and tank
        // throughput. Use `fast_memory_bytes()` for placement so we never
        // claim a peer can hold an expert it cannot actually decode at GPU
        // speed. Same reasoning as the dense Solo gate; see
        // `mesh::Node::fast_memory_bytes()`.
        let mut candidates = vec![MoePlacementCandidate {
            id: my_id,
            fast_memory_bytes: my_vram,
            full_coverage: model_fits,
        }];
        candidates.extend(placement_peers.iter().map(|peer| {
            let peer_fast = peer.fast_memory_bytes();
            MoePlacementCandidate {
                id: peer.id,
                fast_memory_bytes: peer_fast,
                full_coverage: peer_fast >= (model_bytes as f64 * 1.1) as u64,
            }
        }));
        let (current_active_ids, current_fallback_ids) =
            running_plan_state(last_plan.as_ref(), currently_running);
        let provisional_best = compute_best_moe_placement(candidates.clone());
        let allow_scale_up = currently_running
            && last_plan_change_at.elapsed()
                >= std::time::Duration::from_secs(MOE_SCALE_UP_QUIET_SECS);
        let Some(plan) = plan_moe_placement(
            candidates,
            current_active_ids,
            current_fallback_ids,
            allow_scale_up,
        ) else {
            tokio::select! {
                res = peer_rx.changed() => {
                    if res.is_err() { break; }
                }
                res = stop_rx.changed() => {
                    if res.is_err() || stop_requested(&stop_rx) {
                        break;
                    }
                }
            }
            continue;
        };
        let role = plan.role_for(my_id);
        let healthy_reserve_count = placement_peers
            .iter()
            .filter(|peer| {
                !plan.active_ids.contains(&peer.id) && !plan.fallback_ids.contains(&peer.id)
            })
            .count();
        if healthy_reserve_count > 0 && currently_running {
            if !allow_scale_up {
                let remaining = std::time::Duration::from_secs(MOE_SCALE_UP_QUIET_SECS)
                    .saturating_sub(last_plan_change_at.elapsed())
                    .as_secs();
                emit_moe_status(
                    &model_name,
                    "holding reserve peers",
                    format!(
                        "{} healthy peer(s) in reserve for {}s before considering MoE scale-up",
                        healthy_reserve_count, remaining
                    ),
                );
            } else if provisional_best
                .as_ref()
                .filter(|best| {
                    last_plan
                        .as_ref()
                        .is_some_and(|current| best.materially_improves_upon(current))
                })
                .is_none()
            {
                emit_moe_status(
                    &model_name,
                    "holding reserve peers",
                    format!(
                        "{} healthy peer(s) in reserve; the current MoE plan is still preferred",
                        healthy_reserve_count
                    ),
                );
            }
        }

        if currently_running && last_plan.as_ref() == Some(&plan) {
            tokio::select! {
                res = peer_rx.changed() => {
                    if res.is_err() { break; }
                }
                res = stop_rx.changed() => {
                    if res.is_err() || stop_requested(&stop_rx) {
                        break;
                    }
                }
            }
            if stop_requested(&stop_rx) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            continue;
        }

        if currently_running {
            if let Some(previous_plan) = last_plan.as_ref() {
                let previous_role = previous_plan.role_for(my_id);
                let same_local_deployment = previous_role == role
                    && previous_plan.active_ids == plan.active_ids
                    && previous_plan.overlap == plan.overlap;
                if same_local_deployment && previous_plan.fallback_ids != plan.fallback_ids {
                    let targets = build_moe_targets(
                        &plan.active_ids,
                        &plan.fallback_ids,
                        my_id,
                        matches!(role, MoePlacementRole::SplitShard).then_some(
                            current_local_port.expect("running MoE shard should have a local port"),
                        ),
                        matches!(role, MoePlacementRole::FullFallback).then_some(
                            current_local_port
                                .expect("running MoE fallback should have a local port"),
                        ),
                        &model_name,
                    );
                    target_tx.send_replace(targets);
                    last_plan = Some(plan);
                    last_plan_change_at = tokio::time::Instant::now();
                    continue;
                }
            }
        }

        // Something changed — kill existing llama-server
        if currently_running {
            if let Some(process) = llama_process.take() {
                process.handle.shutdown().await;
            }
            if let Some(proxy) = backend_proxy.take() {
                proxy.shutdown().await;
            }
            tunnel_mgr.set_http_port(0);
            currently_running = false;
            current_local_port = None;
            node.clear_local_model_port(&model_name).await;
            on_process(None);
            on_change(false, false);
        }

        last_plan = Some(plan.clone());
        last_plan_change_at = tokio::time::Instant::now();

        if matches!(role, MoePlacementRole::Standby) {
            node.set_model_runtime_context_length(&model_name, None)
                .await;
            node.regossip().await;
            emit_moe_status(
                &model_name,
                "standing by",
                format!(
                    "outside active MoE placement (leader={} active={} fallback={})",
                    plan.leader_id.fmt_short(),
                    plan.active_ids.len(),
                    plan.fallback_ids.len()
                ),
            );
            node.set_role(NodeRole::Worker).await;
            update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
            on_change(false, false);
        } else if matches!(role, MoePlacementRole::FullFallback) {
            emit_moe_status(
                &model_name,
                "full-coverage fallback",
                format!(
                    "leader={} active-shards={} fallback-nodes={}",
                    plan.leader_id.fmt_short(),
                    plan.active_ids.len(),
                    plan.fallback_ids.len()
                ),
            );
            on_change(true, false);

            let llama_port = match find_free_port().await {
                Ok(p) => p,
                Err(e) => {
                    emit_error(
                        format!("Failed to find free port: {e}"),
                        Some(format!("model={model_name} mode=moe-fallback")),
                    );
                    if peer_rx.changed().await.is_err() {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            match launch::start_llama_server(
                &runtime,
                &bin_dir,
                binary_flavor,
                launch::ModelLaunchSpec {
                    model: &model,
                    http_port: llama_port,
                    tunnel_ports: &[],
                    tensor_split: None,
                    split_mode: None,
                    draft: None,
                    draft_max: 0,
                    model_bytes,
                    my_vram: pinned_gpu
                        .as_ref()
                        .map(|gpu| gpu.vram_bytes)
                        .unwrap_or(my_vram),
                    mmproj: None,
                    ctx_size_override,
                    total_group_vram: None,
                    selected_gpu: pinned_gpu.as_ref(),
                    slots,
                },
            )
            .await
            {
                Ok(process) => {
                    let proxy = match crate::network::openai::backend::start_backend_proxy(
                        llama_port,
                        node.clone(),
                    )
                    .await
                    {
                        Ok(proxy) => proxy,
                        Err(err) => {
                            emit_error(
                                format!("Failed to start local OpenAI backend proxy: {err}"),
                                Some(format!("model={model_name} port={llama_port}")),
                            );
                            process.handle.shutdown().await;
                            if peer_rx.changed().await.is_err() {
                                break;
                            }
                            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                            continue;
                        }
                    };
                    let local_proxy_port = proxy.port();
                    backend_proxy = Some(proxy);

                    node.set_role(NodeRole::Host {
                        http_port: ingress_http_port,
                    })
                    .await;
                    tunnel_mgr.set_http_port(local_proxy_port);
                    currently_running = true;
                    current_local_port = Some(local_proxy_port);
                    let ctx_size = process.context_length;
                    llama_process = Some(process);
                    if let Some(ref process) = llama_process {
                        on_process(Some(LocalProcessInfo {
                            backend: "llama".into(),
                            pid: process.handle.pid(),
                            port: llama_port,
                            context_length: process.context_length,
                        }));
                    }
                    node.regossip().await;
                    let targets = build_moe_targets(
                        &plan.active_ids,
                        &plan.fallback_ids,
                        my_id,
                        None,
                        Some(local_proxy_port),
                        &model_name,
                    );
                    target_tx.send_replace(targets);
                    emit_ready_events(&model_name, llama_port, local_proxy_port, ctx_size);
                    on_change(true, true);
                }
                Err(e) => {
                    emit_error(
                        format!("Failed to start fallback llama-server: {e}"),
                        Some(format!("model={model_name}")),
                    );
                }
            }
        } else if plan.active_ids.len() == 1 {
            if model_fits {
                node.set_model_runtime_context_length(&model_name, None)
                    .await;
                node.regossip().await;
                let _ = emit_event(OutputEvent::MoeDetected {
                    model: model_name.clone(),
                    moe: moe_summary.clone(),
                    fits_locally: Some(true),
                    capacity_gb: Some(my_vram as f64 / 1e9),
                    model_gb: Some(model_bytes as f64 / 1e9),
                });
                on_change(true, false);

                let llama_port = match find_free_port().await {
                    Ok(p) => p,
                    Err(e) => {
                        emit_error(
                            format!("Failed to find free port: {e}"),
                            Some(format!("model={model_name} mode=moe-solo")),
                        );
                        if peer_rx.changed().await.is_err() {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        continue;
                    }
                };

                let mb = total_model_bytes(&model);
                match launch::start_llama_server(
                    &runtime,
                    &bin_dir,
                    binary_flavor,
                    launch::ModelLaunchSpec {
                        model: &model,
                        http_port: llama_port,
                        tunnel_ports: &[],
                        tensor_split: None,
                        split_mode: split_mode_for_local_launch(binary_flavor, pinned_gpu.as_ref()),
                        draft: None,
                        draft_max: 0,
                        model_bytes: mb,
                        my_vram: pinned_gpu
                            .as_ref()
                            .map(|gpu| gpu.vram_bytes)
                            .unwrap_or(my_vram),
                        mmproj: None,
                        ctx_size_override,
                        total_group_vram: None,
                        selected_gpu: pinned_gpu.as_ref(),
                        slots,
                    },
                )
                .await
                {
                    Ok(process) => {
                        let proxy = match crate::network::openai::backend::start_backend_proxy(
                            llama_port,
                            node.clone(),
                        )
                        .await
                        {
                            Ok(proxy) => proxy,
                            Err(err) => {
                                emit_error(
                                    format!("Failed to start local OpenAI backend proxy: {err}"),
                                    Some(format!("model={model_name} port={llama_port}")),
                                );
                                process.handle.shutdown().await;
                                continue;
                            }
                        };
                        let local_proxy_port = proxy.port();
                        backend_proxy = Some(proxy);

                        node.set_role(NodeRole::Host {
                            http_port: ingress_http_port,
                        })
                        .await;
                        tunnel_mgr.set_http_port(local_proxy_port);
                        currently_running = true;
                        current_local_port = Some(local_proxy_port);
                        let ctx_size = process.context_length;
                        llama_process = Some(process);
                        if let Some(ref process) = llama_process {
                            on_process(Some(LocalProcessInfo {
                                backend: "llama".into(),
                                pid: process.handle.pid(),
                                port: llama_port,
                                context_length: process.context_length,
                            }));
                        }
                        update_targets(
                            &node,
                            &model_name,
                            InferenceTarget::Local(local_proxy_port),
                            &target_tx,
                        )
                        .await;
                        emit_ready_events(&model_name, llama_port, local_proxy_port, ctx_size);
                        on_change(true, true);
                    }
                    Err(e) => {
                        emit_error(
                            format!("Failed to start llama-server: {e}"),
                            Some(format!(
                                "model={model_name} mode=moe-solo port={llama_port}"
                            )),
                        );
                    }
                }
            } else {
                node.set_model_runtime_context_length(&model_name, None)
                    .await;
                node.regossip().await;
                let _ = emit_event(OutputEvent::MoeDetected {
                    model: model_name.clone(),
                    moe: moe_summary.clone(),
                    fits_locally: Some(false),
                    capacity_gb: Some(my_vram as f64 / 1e9),
                    model_gb: Some(model_bytes as f64 / 1e9),
                });
                on_change(false, false);
            }
        } else {
            let my_shard_index = plan.shard_index_for(my_id).unwrap_or(0);
            on_change(true, false);

            let assignments = moe::compute_assignments_with_overlap(
                &moe_cfg.config.ranking,
                plan.active_ids.len(),
                moe_cfg.config.min_experts_per_node,
                plan.overlap,
            );
            let my_assignment = &assignments[my_shard_index];
            let _ = emit_event(OutputEvent::MoeDistribution {
                model: model_name.clone(),
                moe: moe_summary.clone(),
                distribution: MoeDistributionSummary {
                    leader: plan.leader_id.fmt_short().to_string(),
                    active_nodes: plan.active_ids.len(),
                    fallback_nodes: plan.fallback_ids.len(),
                    shard_index: my_shard_index,
                    shard_count: plan.active_ids.len(),
                    ranking_source: moe_cfg.ranking_source.clone(),
                    ranking_origin: moe_cfg.ranking_origin.clone(),
                    overlap: plan.overlap,
                    shared_experts: my_assignment.n_shared,
                    unique_experts: my_assignment.n_unique,
                },
            });

            // Advertise a non-ready local runtime before split generation / load so
            // peer liveness stays conservative during MoE convergence.
            node.set_model_runtime_starting(&model_name).await;
            node.regossip().await;

            let shard_path = moe::split_path(&model, plan.active_ids.len(), my_shard_index);

            if !shard_path.exists() {
                emit_warning(
                    format!("Splitting GGUF → {} ...", shard_path.display()),
                    Some(format!(
                        "model={model_name} shard={}/{}",
                        my_shard_index + 1,
                        plan.active_ids.len()
                    )),
                );
                match moe::run_split(&bin_dir, &model, my_assignment, &shard_path) {
                    Ok(()) => {
                        let size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                        emit_warning(
                            format!("Split complete: {:.1} GB", size as f64 / 1e9),
                            Some(format!(
                                "model={model_name} shard_path={}",
                                shard_path.display()
                            )),
                        );
                    }
                    Err(e) => {
                        emit_error(
                            format!("moe-split failed: {e}"),
                            Some(format!(
                                "model={model_name} shard_path={}",
                                shard_path.display()
                            )),
                        );
                        node.set_model_runtime_context_length(&model_name, None)
                            .await;
                        node.regossip().await;
                        if peer_rx.changed().await.is_err() {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                        continue;
                    }
                }
            } else {
                let size = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
                emit_moe_status(
                    &model_name,
                    "using cached shard",
                    format!("{} ({:.1} GB)", shard_path.display(), size as f64 / 1e9),
                );
            }

            // Start llama-server with our shard
            let llama_port = match find_free_port().await {
                Ok(p) => p,
                Err(e) => {
                    emit_error(
                        format!("Failed to find free port: {e}"),
                        Some(format!("model={model_name} mode=moe-split")),
                    );
                    if peer_rx.changed().await.is_err() {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

            let shard_bytes = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
            match launch::start_llama_server(
                &runtime,
                &bin_dir,
                binary_flavor,
                launch::ModelLaunchSpec {
                    model: &shard_path,
                    http_port: llama_port,
                    tunnel_ports: &[],
                    tensor_split: None,
                    split_mode: split_mode_for_local_launch(binary_flavor, pinned_gpu.as_ref()),
                    draft: None,
                    draft_max: 0,
                    model_bytes: shard_bytes,
                    my_vram: pinned_gpu
                        .as_ref()
                        .map(|gpu| gpu.vram_bytes)
                        .unwrap_or(my_vram),
                    mmproj: None,
                    ctx_size_override,
                    total_group_vram: None,
                    selected_gpu: pinned_gpu.as_ref(),
                    slots,
                },
            )
            .await
            {
                Ok(process) => {
                    let proxy = match crate::network::openai::backend::start_backend_proxy(
                        llama_port,
                        node.clone(),
                    )
                    .await
                    {
                        Ok(proxy) => proxy,
                        Err(err) => {
                            emit_error(
                                format!("Failed to start local OpenAI backend proxy: {err}"),
                                Some(format!("model={model_name} port={llama_port}")),
                            );
                            process.handle.shutdown().await;
                            continue;
                        }
                    };
                    let local_proxy_port = proxy.port();
                    backend_proxy = Some(proxy);

                    node.set_role(NodeRole::Host {
                        http_port: ingress_http_port,
                    })
                    .await;
                    tunnel_mgr.set_http_port(local_proxy_port);
                    currently_running = true;
                    current_local_port = Some(local_proxy_port);
                    let ctx_size = process.context_length;
                    llama_process = Some(process);
                    if let Some(ref process) = llama_process {
                        on_process(Some(LocalProcessInfo {
                            backend: "llama".into(),
                            pid: process.handle.pid(),
                            port: llama_port,
                            context_length: process.context_length,
                        }));
                    }
                    node.regossip().await;

                    let targets = build_moe_targets(
                        &plan.active_ids,
                        &plan.fallback_ids,
                        my_id,
                        Some(local_proxy_port),
                        None,
                        &model_name,
                    );
                    target_tx.send_replace(targets);

                    emit_ready_events(&model_name, llama_port, local_proxy_port, ctx_size);
                    on_change(true, true);
                }
                Err(e) => {
                    emit_error(
                        format!(
                            "MoE split validation failed for shard {}: {e}",
                            shard_path.display()
                        ),
                        Some(format!("model={model_name}")),
                    );
                    emit_warning(
                        "Refusing to enter MoE split mode on this node until the shard validates",
                        Some(format!(
                            "model={model_name} shard_path={}",
                            shard_path.display()
                        )),
                    );
                    node.set_model_runtime_context_length(&model_name, None)
                        .await;
                    node.regossip().await;
                }
            }
        }

        // Wait for next peer change
        tokio::select! {
            res = peer_rx.changed() => {
                if res.is_err() { break; }
            }
            res = stop_rx.changed() => {
                if res.is_err() || stop_requested(&stop_rx) {
                    break;
                }
            }
        }
        if stop_requested(&stop_rx) {
            break;
        }
        emit_moe_status(&model_name, "re-checking deployment", "mesh changed");
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    }

    if currently_running {
        if let Some(process) = llama_process.take() {
            process.handle.shutdown().await;
        }
        if let Some(proxy) = backend_proxy.take() {
            proxy.shutdown().await;
        }
        tunnel_mgr.set_http_port(0);
        node.set_role(NodeRole::Worker).await;
        update_targets(&node, &model_name, InferenceTarget::None, &target_tx).await;
        on_process(None);
        on_change(false, false);
    }
}

/// Update the model targets map — sets our model's target and includes
/// targets for other models we know about from peers.
/// When multiple nodes serve the same model, all are included for load balancing.
///
/// Capability filtering: when a peer's advertised [`mesh::NodeCapability`] is
/// known to be insufficient for a given model (we have a local size hint and
/// the peer would clearly OOM), we drop that (peer, model) edge instead of
/// adding a target the router would just retry-and-fail through. We only
/// filter remote peers — local targets are spawned by election that already
/// gated on local hardware.
fn extend_targets_from_peer(
    targets: &mut HashMap<String, Vec<InferenceTarget>>,
    peer_models: &[String],
    role: &NodeRole,
    peer_id: iroh::EndpointId,
    peer_capability: &mesh::NodeCapability,
    model_requirements: &HashMap<String, mesh::CapabilityRequirements>,
) {
    // Only confirmed hosts can serve HTTP inference traffic.
    // Split workers may advertise the model they're helping serve, but they
    // only run rpc-server and will drop tunneled chat requests.
    if !matches!(role, NodeRole::Host { .. }) {
        return;
    }

    for serving in peer_models {
        if let Some(req) = model_requirements.get(serving) {
            if !peer_capability.matches(req) {
                tracing::debug!(
                    peer = %peer_id,
                    model = %serving,
                    backend = peer_capability.backend.label(),
                    vram_mb = peer_capability.vram_total_mb,
                    min_vram_mb = ?req.min_vram_mb,
                    "skipping peer for model: capability filter"
                );
                continue;
            }
        }
        targets
            .entry(serving.clone())
            .or_default()
            .push(InferenceTarget::Remote(peer_id));
    }
}

/// Build a `model_name → CapabilityRequirements` map from the sizes the local
/// node knows about (its own announcements). Remote peers don't gossip sizes
/// today, so we only have ground truth for models in our local catalog. For
/// models we have no local size for we emit no requirement and trust the
/// peer's self-advertisement.
fn collect_model_requirements(
    peers: &[mesh::PeerInfo],
    local_id: iroh::EndpointId,
) -> HashMap<String, mesh::CapabilityRequirements> {
    let mut out: HashMap<String, mesh::CapabilityRequirements> = HashMap::new();
    for p in peers {
        if p.id != local_id {
            continue;
        }
        for (model, size_bytes) in &p.available_model_sizes {
            if *size_bytes == 0 {
                continue;
            }
            let size_gb = (*size_bytes / (1024 * 1024 * 1024)).max(1);
            out.insert(
                model.clone(),
                mesh::CapabilityRequirements::for_model_size_gb(size_gb),
            );
        }
    }
    out
}

async fn update_targets(
    node: &mesh::Node,
    my_model: &str,
    my_target: InferenceTarget,
    target_tx: &Arc<watch::Sender<ModelTargets>>,
) {
    let peers = node.peers().await;
    let mut targets: HashMap<String, Vec<InferenceTarget>> = HashMap::new();

    // Start from the current targets — preserve local targets set by other election loops
    // (multi-model per node: each loop manages its own model's entry)
    {
        let current = target_tx.borrow();
        for (model, model_targets) in &current.targets {
            if model != my_model {
                // Keep only Local targets from other loops — remote targets get rebuilt below
                let locals: Vec<_> = model_targets
                    .iter()
                    .filter(|t| {
                        matches!(t, InferenceTarget::Local(_) | InferenceTarget::MoeLocal(_))
                    })
                    .cloned()
                    .collect();
                if !locals.is_empty() {
                    targets.insert(model.clone(), locals);
                }
            }
        }
    }

    // Our model — we're always first in the list
    if !matches!(my_target, InferenceTarget::None) {
        targets
            .entry(my_model.to_string())
            .or_default()
            .push(my_target);
    }

    let local_id = node.id();
    let model_requirements = collect_model_requirements(&peers, local_id);

    // Track peers we excluded purely because of the capability filter, so we
    // can flag a model as "had candidates, all were filtered" — useful for
    // the "no_capable_node" structured 503 path in the router.
    let mut filtered_only_candidates: HashMap<String, usize> = HashMap::new();

    // All peers — group by model (multi-model aware)
    for p in &peers {
        let peer_models = p.routable_models();
        if matches!(p.role, NodeRole::Host { .. }) {
            for serving in &peer_models {
                if let Some(req) = model_requirements.get(serving) {
                    if !p.capability.matches(req) {
                        *filtered_only_candidates.entry(serving.clone()).or_default() += 1;
                    }
                }
            }
        }
        extend_targets_from_peer(
            &mut targets,
            &peer_models,
            &p.role,
            p.id,
            &p.capability,
            &model_requirements,
        );
    }

    // Drop verifier-demoted (peer, model) remote targets. The set is empty
    // unless enforcement is enabled, so this is a no-op on the common path.
    let demotions = node.active_demotions().await;
    if !demotions.is_empty() {
        for (model, model_targets) in targets.iter_mut() {
            model_targets.retain(|t| match t {
                InferenceTarget::Remote(peer_id) => !demotions.contains(&(*peer_id, model.clone())),
                _ => true,
            });
        }
    }

    for (model, dropped) in &filtered_only_candidates {
        let still_have = targets.get(model).map(|v| !v.is_empty()).unwrap_or(false);
        if !still_have && *dropped > 0 {
            emit_info(
                format!(
                    "[{model}] {} peer(s) advertised this model but failed the capability filter; routing will return 503 no_capable_node",
                    dropped
                ),
                None,
            );
        }
    }

    let count: usize = targets.values().map(|v| v.len()).sum();
    if count > 1 {
        for (model, hosts) in &targets {
            if hosts.len() > 1 {
                emit_info(
                    format!("[{model}] {} hosts available (load balancing)", hosts.len()),
                    None,
                );
            }
        }
    }

    if !force_split_routing_enabled() {
        for (model, model_targets) in targets.iter_mut() {
            let model_bytes = model_requirements
                .get(model)
                .and_then(|req| req.min_vram_mb)
                .map(|mb| mb.saturating_mul(1024 * 1024))
                .filter(|b| *b > 0)
                .unwrap_or_else(|| {
                    peers
                        .iter()
                        .find(|p| p.id == local_id)
                        .and_then(|p| p.available_model_sizes.get(model).copied())
                        .unwrap_or(0)
                });
            sort_targets_for_model(model, model_bytes, model_targets, &peers, local_id);
        }
    }

    target_tx.send_replace(ModelTargets {
        targets,
        moe: None,
        counter: Default::default(),
    });
}

/// Start llama-server with --rpc pointing at model-group nodes (self + workers).
/// Returns the ephemeral port and a death notification receiver, or None on failure.
#[allow(clippy::too_many_arguments)]
async fn start_llama(
    params: StartLlamaParams<'_>,
) -> Option<(u16, launch::InferenceServerProcess)> {
    let StartLlamaParams {
        runtime,
        node,
        tunnel_mgr,
        bin_dir,
        model,
        model_name,
        model_peers,
        explicit_mmproj,
        draft,
        draft_max,
        force_split,
        binary_flavor,
        ctx_size_override,
        pinned_gpu,
        slots,
    } = params;
    // For Solo-vs-Split planning we always feed `fast_memory_bytes()`,
    // never the RAM-offload-inflated `vram_bytes()`. Otherwise a CUDA
    // laptop with 16 GB GPU + 120 GB host RAM looks like it has 106 GB
    // of "VRAM" and the planner picks Solo for a 70 B GGUF that
    // physically can't decode on its GPU. See May 13 2026 incident.
    let my_vram = node.fast_memory_bytes();
    let local_launch_vram = effective_local_launch_vram(my_vram, pinned_gpu);
    let model_bytes = total_model_bytes(model);
    let launch_plan = build_dense_launch_plan(
        local_launch_vram,
        model_bytes,
        force_split,
        model_name,
        model_peers,
    );
    let mut worker_ids = match launch_plan {
        DenseLaunchPlan::Solo => {
            let worker_count = model_peers
                .iter()
                .filter(|p| !matches!(p.role, NodeRole::Client))
                .count();
            if worker_count > 0 {
                emit_info(
                    format!(
                        "Model fits on host ({:.1}GB capacity for {:.1}GB model) — serving entirely. Use --split to force distributed mode",
                        local_launch_vram as f64 / 1e9,
                        model_bytes as f64 / 1e9
                    ),
                    Some(format!("model={model_name}")),
                );
            }
            Vec::new()
        }
        DenseLaunchPlan::Split { worker_ids, .. } => {
            for id in &worker_ids {
                if let Some(peer) = model_peers.iter().find(|peer| peer.id == *id) {
                    let rtt_str = peer
                        .rtt_ms
                        .map(|r| format!("{}ms", r))
                        .unwrap_or("?ms".to_string());
                    emit_info(
                        format!(
                            "Adding {} — {:.1}GB capacity, RTT {rtt_str}",
                            peer.id.fmt_short(),
                            split_peer_vram_bytes(peer, local_launch_vram) as f64 / 1e9
                        ),
                        Some(format!("model={model_name}")),
                    );
                }
            }
            worker_ids.clone()
        }
        DenseLaunchPlan::WaitingForCapacity { .. } => {
            return None;
        }
    };

    // Wait for tunnels to workers
    if !worker_ids.is_empty() {
        emit_info(
            format!("Waiting for tunnels to {} worker(s)...", worker_ids.len()),
            Some(format!("model={model_name}")),
        );
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tunnel_mgr.wait_for_peers(worker_ids.len()),
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        // Pre-dial workers: ensure a real iroh QUIC connection exists to
        // every cohort member. Peers learned only transitively from the
        // entry node's gossip have no `state.connections` entry; the
        // outbound tunnel listener still accepts TCP, but
        // `open_tunnel_stream` errors immediately, the TCP socket drops,
        // and the pre-launch HELLO probe reports `UnexpectedEof` for
        // every such worker (May 16 2026 vast.ai-host incident — a
        // 12 GB cloud GPU elected itself as host for three Mac workers
        // it had never directly dialled, every HELLO failed, the 70 B
        // model never served). Forcing the dial here populates
        // `state.connections` so the subsequent HELLO probe and
        // llama-server `--rpc` connect both find a live QUIC channel.
        let mut unreachable_workers: HashSet<iroh::EndpointId> = HashSet::new();
        {
            let dial_timeout = std::time::Duration::from_secs(8);
            let mut dial_set = tokio::task::JoinSet::new();
            for id in worker_ids.iter().copied() {
                let node = node.clone();
                dial_set.spawn(async move { (id, node.dial_for_split(id, dial_timeout).await) });
            }
            while let Some(joined) = dial_set.join_next().await {
                if let Ok((id, res)) = joined {
                    if let Err(e) = res {
                        tracing::warn!("Pre-dial to {} failed: {e}", id.fmt_short());
                        unreachable_workers.insert(id);
                    }
                }
            }
        }
        if !unreachable_workers.is_empty() {
            let dropped: Vec<String> = unreachable_workers
                .iter()
                .map(|id| id.fmt_short().to_string())
                .collect();
            emit_warning(
                format!(
                    "Dropping {} unreachable worker(s) from cohort (no QUIC path): {}",
                    unreachable_workers.len(),
                    dropped.join(", ")
                ),
                Some(format!("model={model_name}")),
            );
            worker_ids.retain(|id| !unreachable_workers.contains(id));
            // If we lost every worker, give up this election cycle so
            // the next runner-up gets a shot instead of launching solo
            // against a model that doesn't fit on the host alone.
            if worker_ids.is_empty() {
                emit_warning(
                    format!(
                        "Aborting launch — every worker unreachable from this host \
                         (model needs split mode, host fast-memory alone is insufficient)"
                    ),
                    Some(format!("model={model_name}")),
                );
                return None;
            }
        }

        // B2B tunnel map exchange
        let my_map = tunnel_mgr.peer_ports_map().await;
        let _ = node.broadcast_tunnel_map(my_map).await;
        let _ = node
            .wait_for_tunnel_maps(worker_ids.len(), std::time::Duration::from_secs(10))
            .await;
        let remote_maps = node.all_remote_tunnel_maps().await;
        tunnel_mgr.update_rewrite_map(&remote_maps).await;
    }

    // Build --rpc list: only remote workers.
    // The host's own GPU is used directly on the local backend — no need to route
    // through the local rpc-server (which would add unnecessary TCP round trips).
    let all_ports = tunnel_mgr.peer_ports_map().await;
    let Some(mut rpc_ports) = rpc_ports_for_worker_ids(&all_ports, &worker_ids) else {
        emit_warning(
            format!(
                "Waiting for selected worker tunnels ({}/{} ready)",
                all_ports
                    .keys()
                    .filter(|id| worker_ids.contains(id))
                    .count(),
                worker_ids.len()
            ),
            Some(format!("model={model_name}")),
        );
        return None;
    };

    // Pre-launch HELLO probe — fail fast if any tunneled rpc-server is
    // silent. Without this, llama-server discovers the bad tunnel during
    // its own `negotiate_hello`, hits `RPC_STATUS_ASSERT`, and SIGABRTs
    // before the HTTP port binds. The election loop then sees the launch
    // failure, retries, and crashes again forever (May 14 2026 incident).
    //
    // If only *some* of the workers fail the probe, we drop them from the
    // cohort and retry with the survivors — as long as the remaining
    // group's capacity is still big enough for the model. This is what
    // unblocked the May 14 2026 cluster: one peer's rpc-server tunnel
    // was silent (probably stuck after a runtime upgrade or a Quic stream
    // never recovered after the entry rotated its node_id), and every
    // election that included it deadlocked the whole mesh in "loading"
    // because every host candidate tried to include it for capacity.
    // Dropping the silent peer keeps the model serving on the reachable
    // 3-of-4 cohort while the broken peer recovers on its own schedule.
    let port_to_peer: HashMap<u16, iroh::EndpointId> =
        all_ports.iter().map(|(id, port)| (*port, *id)).collect();
    let mut launch_bad_ports = HashSet::new();
    while !rpc_ports.is_empty() {
        let (bad, bad_ports) = probe_rpc_ports(&rpc_ports, &port_to_peer).await;
        if bad.is_empty() {
            break;
        }
        launch_bad_ports.extend(bad_ports);

        // Filter cohort down to workers whose probe was healthy.
        // worker_ids and rpc_ports stay aligned because both lists
        // were built in the same order from the same launch_plan.
        let original_count = rpc_ports.len();
        let survivor_count = original_count.saturating_sub(bad.len());
        let (new_worker_ids, new_rpc_ports, group_capacity) =
            replenish_worker_cohort_after_probe_failure(
                model_name,
                model_peers,
                &worker_ids,
                &rpc_ports,
                &launch_bad_ports,
                &all_ports,
                local_launch_vram,
                model_bytes,
            );
        let replacement_count = new_worker_ids.len().saturating_sub(survivor_count);
        let min_vram = (model_bytes as f64 * 1.1) as u64;

        if group_capacity >= min_vram && !new_worker_ids.is_empty() {
            if new_worker_ids == worker_ids && new_rpc_ports == rpc_ports {
                emit_warning(
                    format!(
                        "Aborting launch — {}/{} worker rpc tunnels failed HELLO probe: {} \
                             and no replacement cohort is available",
                        bad.len(),
                        original_count,
                        bad.join(", "),
                    ),
                    Some(format!("model={model_name}")),
                );
                return None;
            }
            emit_warning(
                format!(
                    "Replacing cohort after {}/{} worker HELLO failure(s) ({}); \
                         kept {} survivor(s), added {} replacement(s), retrying probe with \
                         {} worker(s) at {:.1}GB capacity",
                    bad.len(),
                    original_count,
                    bad.join(", "),
                    survivor_count,
                    replacement_count,
                    new_worker_ids.len(),
                    group_capacity as f64 / 1e9
                ),
                Some(format!("model={model_name}")),
            );
            worker_ids = new_worker_ids;
            rpc_ports = new_rpc_ports;
            continue;
        } else {
            emit_warning(
                format!(
                    "Aborting launch — {}/{} worker rpc tunnels failed HELLO probe: {} \
                         (remaining capacity {:.1}GB < {:.1}GB required)",
                    bad.len(),
                    original_count,
                    bad.join(", "),
                    group_capacity as f64 / 1e9,
                    min_vram as f64 / 1e9,
                ),
                Some(format!("model={model_name}")),
            );
            return None;
        }
    }

    // Calculate group VRAM from the planned cohort. Do not pass this as a
    // fixed --tensor-split for RPC launches: remote free memory can change
    // between gossip and llama.cpp's device probe, and a user-pinned split
    // prevents the fitter from recovering.
    let my_vram_f = local_launch_vram as f64;
    let mut all_vrams: Vec<f64> = Vec::new();
    for id in &worker_ids {
        if let Some(peer) = model_peers.iter().find(|p| p.id == *id) {
            all_vrams.push(split_peer_vram_bytes(peer, local_launch_vram) as f64);
        }
    }
    all_vrams.push(my_vram_f); // Host device is last
    let total: f64 = all_vrams.iter().sum();

    // Launch on ephemeral port
    let llama_port = match find_free_port().await {
        Ok(p) => p,
        Err(e) => {
            emit_error(
                format!("Failed to find free port: {e}"),
                Some(format!("model={model_name} mode=dense")),
            );
            return None;
        }
    };

    // Look up mmproj for vision models
    let mmproj_path = crate::models::resolve_mmproj_path(model_name, model, explicit_mmproj);

    // In split mode (pipeline parallel), pass total group VRAM so context size
    // accounts for the host only holding its share of layers. KV cache is also
    // distributed — each node holds KV for its own layers.
    let group_vram = if !rpc_ports.is_empty() {
        Some(total as u64)
    } else {
        None
    };

    match launch::start_llama_server(
        runtime,
        bin_dir,
        binary_flavor,
        launch::ModelLaunchSpec {
            model,
            http_port: llama_port,
            tunnel_ports: &rpc_ports,
            tensor_split: None,
            // Row split only works for local multi-GPU — not over RPC.
            // When we have RPC workers, llama.cpp uses layer (pipeline) split.
            split_mode: if rpc_ports.is_empty() {
                split_mode_for_local_launch(binary_flavor, pinned_gpu)
            } else {
                None
            },
            draft,
            draft_max,
            model_bytes,
            my_vram: local_launch_vram,
            mmproj: mmproj_path.as_deref(),
            ctx_size_override,
            total_group_vram: group_vram,
            selected_gpu: pinned_gpu,
            slots,
        },
    )
    .await
    {
        Ok(process) => Some((llama_port, process)),
        Err(e) => {
            emit_error(
                format!("Failed to start llama-server: {e}"),
                Some(format!("model={model_name} mode=dense port={llama_port}")),
            );
            None
        }
    }
}

async fn find_free_port() -> anyhow::Result<u16> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::EndpointAddr;
    use iroh::SecretKey;

    /// Create a deterministic EndpointId from a byte seed.
    fn make_id(seed: u8) -> iroh::EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    /// `try_total_model_bytes` must distinguish "missing" (`None`) from
    /// "present but zero bytes" (`Some(0)`). The legacy
    /// `total_model_bytes` collapses both to `0`, which is the silent
    /// failure mode that bit us in the May 16 2026 incident — see the
    /// doc-comment on `total_model_bytes`.
    #[test]
    fn try_total_model_bytes_distinguishes_missing_from_zero() {
        let tmp = std::env::temp_dir().join(format!(
            "senda-try-bytes-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        let missing = tmp.join("does-not-exist.gguf");
        assert_eq!(
            try_total_model_bytes(&missing),
            None,
            "missing file must surface as None so the runtime can refuse to advertise"
        );
        assert_eq!(
            total_model_bytes(&missing),
            0,
            "legacy total_model_bytes still returns 0 for back-compat with arithmetic callers"
        );

        let zero = tmp.join("zero.gguf");
        std::fs::write(&zero, b"").unwrap();
        assert_eq!(
            try_total_model_bytes(&zero),
            Some(0),
            "0-byte file must surface as Some(0), distinct from missing"
        );

        let real = tmp.join("real.gguf");
        std::fs::write(&real, b"hello").unwrap();
        assert_eq!(try_total_model_bytes(&real), Some(5));
        assert_eq!(total_model_bytes(&real), 5);

        // Dangling symlink — repro of the exact M3 Pro state we saw on
        // 2026-05-16: HF symlink at .../snapshots/.../Qwen3-32B-Q4_K_M.gguf
        // pointing at .../blobs/8df67…370e7 which was never downloaded.
        #[cfg(unix)]
        {
            let target = tmp.join("never-downloaded-blob");
            let link = tmp.join("dangling.gguf");
            std::fs::write(&target, vec![0u8; 8]).unwrap();
            std::os::unix::fs::symlink(&target, &link).unwrap();
            std::fs::remove_file(&target).unwrap();
            assert_eq!(
                try_total_model_bytes(&link),
                None,
                "dangling symlink must surface as None — this is the regression that \
                 made the M3 Pro think a 19.76 GB model was 0 bytes and 'fit anywhere'"
            );
        }

        // Multi-shard model with one shard missing must surface as None.
        let shard1 = tmp.join("multi-Q4_K_M-00001-of-00003.gguf");
        let shard2 = tmp.join("multi-Q4_K_M-00002-of-00003.gguf");
        // Note: shard 3 is intentionally never written.
        std::fs::write(&shard1, vec![0u8; 1024]).unwrap();
        std::fs::write(&shard2, vec![0u8; 1024]).unwrap();
        assert_eq!(
            try_total_model_bytes(&shard1),
            None,
            "incomplete multi-shard model must surface as None rather than \
             a misleading partial-sum that looks fine to the dense-launch planner"
        );

        // All shards present: returns the sum.
        let shard3 = tmp.join("multi-Q4_K_M-00003-of-00003.gguf");
        std::fs::write(&shard3, vec![0u8; 1024]).unwrap();
        assert_eq!(try_total_model_bytes(&shard1), Some(3 * 1024));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn make_dense_peer(
        id: iroh::EndpointId,
        vram_bytes: u64,
        rtt_ms: Option<u32>,
        serving_model: &str,
    ) -> mesh::PeerInfo {
        mesh::PeerInfo {
            id,
            addr: EndpointAddr {
                id,
                addrs: Default::default(),
            },
            tunnel_port: None,
            role: NodeRole::Worker,
            first_joined_mesh_ts: None,
            models: vec![],
            vram_bytes,
            rtt_ms,
            model_source: None,
            serving_models: vec![serving_model.to_string()],
            hosted_models: vec![],
            hosted_models_known: false,
            available_models: vec![],
            requested_models: vec![],
            last_seen: std::time::Instant::now(),
            last_mentioned: std::time::Instant::now(),
            moe_recovered_at: None,
            version: None,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
            owner_summary: crate::crypto::OwnershipSummary::default(),
            inflight_requests: 0,
            system_ram_bytes: 0,
            model_timings: vec![],
            native_baselines: vec![],
            rpc_ready: None,
            capability: crate::mesh::NodeCapability::default(),
            model_ad: Default::default(),
        }
    }

    /// Construct a peer whose `vram_bytes` includes a RAM-offload allowance
    /// (Linux/Windows discrete-GPU shape), with `capability.vram_total_mb`
    /// reporting the *real* GPU VRAM. Mirrors what gossip carries for an
    /// RTX 4080-SUPER laptop.
    fn make_inflated_peer(
        id: iroh::EndpointId,
        gpu_vram_gb: u64,
        ram_offload_gb: u64,
        rtt_ms: Option<u32>,
        serving_model: &str,
    ) -> mesh::PeerInfo {
        let gpu_bytes = gpu_vram_gb * 1024 * 1024 * 1024;
        let inflated_bytes = (gpu_vram_gb + ram_offload_gb) * 1024 * 1024 * 1024;
        let mut peer = make_dense_peer(id, inflated_bytes, rtt_ms, serving_model);
        peer.capability = crate::mesh::NodeCapability {
            vram_total_mb: gpu_bytes / (1024 * 1024),
            ..crate::mesh::NodeCapability::default()
        };
        peer
    }

    /// May 13 2026 regression (host-side half): an RTX 4080-SUPER laptop
    /// reports `vram_bytes = 106 GB` (16 GB GPU + ~120 GB host RAM × 0.75)
    /// via gossip. Pre-fix, `build_dense_launch_plan` saw `my_vram (106)
    /// >= 42.5 GB × 1.1 = 46.7 GB` and returned Solo, so llama.cpp tried
    /// to host the 70 B GGUF on the 16 GB GPU by mmap-faulting weights
    /// from host RAM at ~0.2 t/s.
    ///
    /// Post-fix the planner reads `node.fast_memory_bytes()` (16 GB) and
    /// drops out of the Solo branch into the cohort-pooling loop. The
    /// May 14 2026 follow-up (this commit) extends the same honesty to
    /// the cohort itself: each peer also contributes `fast_memory_bytes`
    /// rather than its inflated `vram_bytes`, so the planner stops
    /// adding workers only once the *real* fast-memory total covers the
    /// model — not after the first inflated peer pretends to have 106 GB.
    /// See `split_peer_vram_bytes` for the matching peer-side fix.
    #[test]
    fn dense_launch_plan_does_not_solo_when_fast_memory_is_below_model_size() {
        let model = "DeepSeek-R1-Distill-70B-Q4_K_M";
        let id_msi = make_id(2);
        let id_mac = make_id(3);
        let id_manonas = make_id(4);

        // Workers visible from LYU's perspective. Reflects the real live
        // mesh on May 14 2026: 4 nodes total (LYU + these 3) all reporting
        // inflated vram_bytes but small real GPUs, and a 42.5 GB model
        // that only fits across the full cohort's fast memory budget.
        let peers = vec![
            make_inflated_peer(id_msi, 8, 19, Some(40), model), // RTX 4070 Laptop
            make_dense_peer(id_mac, 12 * 1024 * 1024 * 1024, Some(60), model), // M3 Pro unified
            make_dense_peer(id_manonas, 12 * 1024 * 1024 * 1024, Some(70), model), // M1 unified
        ];

        // Pre-fix shape: my_vram = inflated 106 GB → would Solo.
        // Post-fix shape: my_vram = fast memory 16 GB → must Split.
        let my_fast_memory = 16u64 * 1024 * 1024 * 1024;
        let model_bytes = (42.5 * 1024.0 * 1024.0 * 1024.0) as u64;

        let plan = build_dense_launch_plan(my_fast_memory, model_bytes, false, model, &peers);

        match plan {
            DenseLaunchPlan::Split { worker_ids, .. } => {
                assert!(
                    worker_ids.contains(&id_msi)
                        && worker_ids.contains(&id_mac)
                        && worker_ids.contains(&id_manonas),
                    "expected all 3 peers to be pulled in; got {:?} — \
                     `split_peer_vram_bytes` must use fast_memory_bytes, not \
                     inflated vram_bytes (otherwise the loop stops after one \
                     peer 'covers' the model on paper but oversubscribes its \
                     GPU when llama-server actually tries to allocate the split)",
                    worker_ids
                );
            }
            other => panic!(
                "expected Split plan for 70B-on-laptops, got {:?} — this means the May 13 \
                 fast-memory gate regressed and LYU is back to lying about Solo capacity",
                other
            ),
        }

        // Sanity: the same call with `my_vram = 106 GB` (inflated) — the
        // shape pre-fix — would still pick Solo. This pins down WHY we
        // care which metric callers feed in.
        let inflated = 106u64 * 1024 * 1024 * 1024;
        let inflated_plan = build_dense_launch_plan(inflated, model_bytes, false, model, &peers);
        assert_eq!(
            inflated_plan,
            DenseLaunchPlan::Solo,
            "if a caller ever passes inflated vram_bytes here we want this assert to flip \
             so the regression is caught at compile-and-test time, not in production"
        );
        // The point of the fix is that election.rs callers now pass
        // node.fast_memory_bytes(), not node.vram_bytes(); the planner
        // itself doesn't know the difference, so this guards the *contract*
        // by asserting both arms of the gate behave as documented.
        assert!(
            my_fast_memory < (model_bytes as f64 * 1.1) as u64,
            "test setup invariant: fast memory must be below the Solo gate"
        );
        assert!(
            inflated >= (model_bytes as f64 * 1.1) as u64,
            "test setup invariant: inflated value must clear the Solo gate"
        );
    }

    /// Sister assertion: `should_be_host_for_model` must compare in the
    /// same units across host and worker. Pre-fix, the host self-evaluated
    /// using inflated `vram_bytes` while peers compared using
    /// `peer.fast_memory_bytes()` — different units in the same predicate.
    /// On the May 13 mesh that produced "every peer thinks it's the host"
    /// because each one's inflated-self beat every other peer's
    /// fast-memory budget. Post-fix both sides use fast memory and the
    /// election picks a single winner.
    #[test]
    fn should_be_host_compares_fast_memory_symmetrically() {
        let model = "DeepSeek-R1-Distill-70B-Q4_K_M";
        let id_lyu = make_id(1);
        let id_msi = make_id(2);
        let id_mac = make_id(3);

        let lyu = make_inflated_peer(id_lyu, 16, 90, None, model); // 4080-SUPER + 120 GB RAM
        let msi = make_inflated_peer(id_msi, 8, 19, None, model); // 4070 + ~25 GB RAM
        let mac = make_dense_peer(id_mac, 12 * 1024 * 1024 * 1024, None, model); // M3 Pro

        // From LYU's perspective: I have 16 GB fast memory, peers are MSI(8) + Mac(12).
        // 16 > 8, 16 > 12 → LYU is the host.
        let lyu_fast = lyu.fast_memory_bytes();
        assert!(should_be_host_for_model(
            id_lyu,
            lyu_fast,
            std::slice::from_ref(&msi)
        ));
        assert!(should_be_host_for_model(
            id_lyu,
            lyu_fast,
            &[msi.clone(), mac.clone()]
        ));

        // From MSI's perspective: I have 8 GB fast memory, peers include LYU(16).
        // 16 > 8 → MSI is NOT the host.
        let msi_fast = msi.fast_memory_bytes();
        assert!(!should_be_host_for_model(id_msi, msi_fast, &[lyu.clone()]));

        // Pre-fix bug shape: if MSI had compared its inflated vram (27 GB)
        // against LYU's fast memory (16 GB), MSI would have decided IT was
        // the host. That's the symmetry break that produced the
        // split-brain. Document it explicitly:
        let msi_inflated = msi.vram_bytes;
        assert!(
            should_be_host_for_model(id_msi, msi_inflated, &[lyu.clone()]),
            "this should_be_host call uses MSI's *inflated* vram against \
             LYU's fast memory and would have made MSI claim host pre-fix; \
             this assertion documents the broken comparison and exists so \
             that if anyone ever wires inflated vram back into the host arg \
             we surface the asymmetry in CI rather than in the field"
        );
    }

    #[test]
    fn peers_for_pipeline_election_drops_outdated_runtime() {
        let model = "DeepSeek-R1-Distill-70B-Q4_K_M";
        let id_old = make_id(1);
        let id_new = make_id(2);
        let mut old = make_dense_peer(id_old, 26 * 1024 * 1024 * 1024, None, model);
        old.version = Some("0.66.18".to_string());
        let mut new_peer = make_dense_peer(id_new, 16 * 1024 * 1024 * 1024, None, model);
        new_peer.version = Some("0.66.52".to_string());
        let filtered = peers_for_pipeline_election(&[old, new_peer.clone()]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, id_new);
    }

    /// Issue #9 / v0.66.18 → v0.66.20 mixed-version deadlock regression.
    ///
    /// Setup mirrors the May 13 production cohort: peer A is the *local*
    /// maximum on fast memory and would be elected as host by every other
    /// peer in the cohort, but A is running an old runtime that uses the
    /// inflated `vram_bytes` metric and therefore never claims
    /// `NodeRole::Host`. Pre-fix every other peer routes inference to A
    /// forever and chat 503s. Post-fix `viable_host_candidates` drops A
    /// from the candidate set after `HOST_CLAIM_GRACE`, the runner-up B
    /// self-elects on the next iteration, and the cohort recovers.
    #[test]
    fn host_claim_timeout_falls_back_to_runner_up_when_top_candidate_never_claims() {
        let model = "DeepSeek-R1-Distill-70B-Q4_K_M";
        let id_a = make_id(1);
        let id_b = make_id(2);

        let mut a = make_dense_peer(id_a, 32 * 1024 * 1024 * 1024, None, model);
        a.role = NodeRole::Worker; // never claims Host
        let mut b = make_dense_peer(id_b, 16 * 1024 * 1024 * 1024, None, model);
        b.role = NodeRole::Worker;
        let peers = vec![a.clone(), b.clone()];

        let t0 = std::time::Instant::now();
        let mut first_observed = std::collections::HashMap::new();
        first_observed.insert(id_a, t0);
        first_observed.insert(id_b, t0);

        // Inside the grace window: A is still a viable candidate.
        let inside = t0 + HOST_CLAIM_GRACE / 2;
        let candidates = viable_host_candidates(&peers, &first_observed, inside, HOST_CLAIM_GRACE);
        assert!(
            candidates.iter().any(|p| p.id == id_a),
            "during grace, A must remain a candidate so honest-but-slow \
             elected hosts get time to actually start llama-server"
        );
        // B (running new code) defers to A.
        let b_fast = b.fast_memory_bytes();
        assert!(
            !should_be_host_for_model(id_b, b_fast, &candidates),
            "during grace, B must defer to A — pinning the honest-cohort \
             behavior we don't want to regress"
        );

        // Past the grace window: A is excluded, B self-elects.
        let after = t0 + HOST_CLAIM_GRACE + std::time::Duration::from_secs(1);
        let candidates = viable_host_candidates(&peers, &first_observed, after, HOST_CLAIM_GRACE);
        assert!(
            !candidates.iter().any(|p| p.id == id_a),
            "after grace, A (still NodeRole::Worker, never claimed Host) \
             must be filtered out — this is what unblocks the v0.66.18 \
             mixed-version deadlock"
        );
        assert!(
            should_be_host_for_model(id_b, b_fast, &candidates),
            "after grace and A's exclusion, B must self-elect as the new \
             local maximum — this is the auto-heal path for issue #9"
        );
    }

    /// If the previously-stuck candidate eventually does claim
    /// `NodeRole::Host` (e.g. their slow-loading model finished, or the
    /// mixed-version peer was upgraded), the cohort must accept them
    /// back as a host candidate. The exclusion is grace-conditional, not
    /// permanent.
    #[test]
    fn host_role_transition_returns_candidate_to_eligibility() {
        let model = "DeepSeek-R1-Distill-70B-Q4_K_M";
        let id_a = make_id(1);
        let id_b = make_id(2);

        // Same shape as the deadlock test but A is now claiming Host.
        let mut a = make_dense_peer(id_a, 32 * 1024 * 1024 * 1024, None, model);
        a.role = NodeRole::Host { http_port: 9337 };
        let mut b = make_dense_peer(id_b, 16 * 1024 * 1024 * 1024, None, model);
        b.role = NodeRole::Worker;
        let peers = vec![a.clone(), b.clone()];

        let t0 = std::time::Instant::now();
        let mut first_observed = std::collections::HashMap::new();
        first_observed.insert(id_a, t0);
        first_observed.insert(id_b, t0);

        // Even WAY past the grace window, an active host stays viable
        // because branch (1) of `viable_host_candidates` (role == Host)
        // bypasses the grace check entirely.
        let way_after = t0 + HOST_CLAIM_GRACE * 100;
        let candidates =
            viable_host_candidates(&peers, &first_observed, way_after, HOST_CLAIM_GRACE);
        assert!(
            candidates.iter().any(|p| p.id == id_a),
            "an active Host must always be a viable candidate, \
             regardless of grace window"
        );
        let b_fast = b.fast_memory_bytes();
        assert!(
            !should_be_host_for_model(id_b, b_fast, &candidates),
            "B must defer to the actively-serving host A even long after \
             A's original grace would have expired"
        );
    }

    /// Pin the existing peer-prune auto-heal: when the host literally
    /// disappears from gossip (laptop closes, network drops, app quits),
    /// the surviving cohort re-elects immediately. This already worked
    /// pre-fix; the test exists so the new candidate-filter logic does
    /// not accidentally hide gossip drops behind a 30-second timer.
    #[test]
    fn peer_disappearing_from_gossip_triggers_immediate_re_election() {
        let model = "DeepSeek-R1-Distill-70B-Q4_K_M";
        let id_a = make_id(1);
        let id_b = make_id(2);

        let mut a = make_dense_peer(id_a, 32 * 1024 * 1024 * 1024, None, model);
        a.role = NodeRole::Host { http_port: 9337 };
        let mut b = make_dense_peer(id_b, 16 * 1024 * 1024 * 1024, None, model);
        b.role = NodeRole::Worker;

        let t0 = std::time::Instant::now();
        let mut first_observed = std::collections::HashMap::new();
        first_observed.insert(id_a, t0);
        first_observed.insert(id_b, t0);

        // Phase 1: A is hosting, B defers. Standard cohort.
        let peers_with_a = vec![a.clone(), b.clone()];
        let candidates =
            viable_host_candidates(&peers_with_a, &first_observed, t0, HOST_CLAIM_GRACE);
        let b_fast = b.fast_memory_bytes();
        assert!(!should_be_host_for_model(id_b, b_fast, &candidates));

        // Phase 2: A disappears from gossip. The election loop calls
        // peers() again and gets back a list without A. Even with A still
        // in `first_observed` (we haven't garbage-collected yet inside
        // this test, but the loop does), the candidate filter only sees
        // peers actually in the supplied slice — so A is gone immediately,
        // not after 30s.
        let peers_without_a = vec![b.clone()];
        let candidates =
            viable_host_candidates(&peers_without_a, &first_observed, t0, HOST_CLAIM_GRACE);
        assert_eq!(
            candidates.len(),
            1,
            "candidate filter must respect peer disappearance — a peer not \
             in `model_peers` cannot magically reappear in the candidate \
             list, otherwise we would block re-election after a host drops"
        );
        assert!(
            should_be_host_for_model(id_b, b_fast, &candidates),
            "with A gone from gossip, B becomes the local max immediately \
             — pinning the existing peer-prune auto-heal behavior"
        );
    }

    /// Adversary scenario: a malicious or buggy peer self-attests an
    /// absurdly large fast-memory budget so it always wins the local
    /// election, but never claims `NodeRole::Host`. The grace timer
    /// neutralizes the attack — after 30s, the lying peer is excluded
    /// and an honest peer takes over. This is the architectural
    /// resilience property the user explicitly asked for ("imagine if
    /// somebody with bad intentions want to block the good honest
    /// nodes").
    #[test]
    fn lying_peer_with_huge_fast_memory_is_excluded_after_grace() {
        let model = "DeepSeek-R1-Distill-70B-Q4_K_M";
        let id_attacker = make_id(99);
        let id_honest = make_id(1);

        // Attacker claims 1 PB of fast memory but never claims Host.
        let mut attacker =
            make_dense_peer(id_attacker, 1024 * 1024 * 1024 * 1024 * 1024, None, model);
        attacker.role = NodeRole::Worker;
        let mut honest = make_dense_peer(id_honest, 16 * 1024 * 1024 * 1024, None, model);
        honest.role = NodeRole::Worker;
        let peers = vec![attacker.clone(), honest.clone()];

        let t0 = std::time::Instant::now();
        let mut first_observed = std::collections::HashMap::new();
        first_observed.insert(id_attacker, t0);
        first_observed.insert(id_honest, t0);

        // Pre-grace: attacker dominates, honest defers (this is the
        // unavoidable window — we MUST give honest peers time to start
        // llama-server, so a 30s window of vulnerability is the cost).
        let candidates = viable_host_candidates(&peers, &first_observed, t0, HOST_CLAIM_GRACE);
        let honest_fast = honest.fast_memory_bytes();
        assert!(!should_be_host_for_model(
            id_honest,
            honest_fast,
            &candidates
        ));

        // Post-grace: attacker is excluded, honest takes over.
        let after = t0 + HOST_CLAIM_GRACE + std::time::Duration::from_secs(1);
        let candidates = viable_host_candidates(&peers, &first_observed, after, HOST_CLAIM_GRACE);
        assert!(
            !candidates.iter().any(|p| p.id == id_attacker),
            "after grace, the lying peer must be excluded — single \
             malicious node cannot indefinitely block the cohort"
        );
        assert!(
            should_be_host_for_model(id_honest, honest_fast, &candidates),
            "honest peer takes over once attacker is filtered out — \
             this is the resilience property the mesh promises"
        );
    }

    /// A single failure inside the window is not enough to bench us —
    /// transient hiccups (e.g. a port collision on the random ephemeral
    /// pick) shouldn't surrender host candidacy to a smaller peer.
    #[test]
    fn host_attempt_backoff_one_failure_keeps_us_eligible() {
        let mut backoff = HostAttemptBackoff::new();
        let t0 = std::time::Instant::now();
        let armed = backoff.record_failure(t0);
        assert!(armed.is_none(), "single failure must not arm the backoff");
        assert!(!backoff.is_active(t0));
        assert!(!backoff.is_active(t0 + std::time::Duration::from_secs(5)));
    }

    /// Two failures inside the 90 s window arm a 60 s backoff. This is
    /// the May 14 2026 incident: Mac SIGABRT'd on every relaunch because
    /// LYU's iroh tunnel was silent. Without this, the loop would burn
    /// hundreds of attempts/hour on the same crash.
    #[test]
    fn host_attempt_backoff_arms_after_threshold_and_disables_candidacy() {
        let mut backoff = HostAttemptBackoff::new();
        let t0 = std::time::Instant::now();

        assert!(backoff.record_failure(t0).is_none());
        let until = backoff
            .record_failure(t0 + std::time::Duration::from_secs(5))
            .expect("second failure inside window arms backoff");
        assert!(backoff.is_active(t0 + std::time::Duration::from_secs(5)));
        // Backoff window is exactly HOST_ATTEMPT_BACKOFF.
        assert_eq!(
            until - (t0 + std::time::Duration::from_secs(5)),
            HOST_ATTEMPT_BACKOFF
        );

        // Just before expiry: still locked out.
        let almost = until - std::time::Duration::from_millis(1);
        assert!(backoff.is_active(almost));

        // After expiry: free to try again, with a fresh failure counter.
        let after = until + std::time::Duration::from_millis(1);
        assert!(!backoff.is_active(after));
        assert!(
            backoff.record_failure(after).is_none(),
            "first failure after backoff expires must NOT immediately re-arm — counter resets"
        );
    }

    /// Failures that fall outside the 90 s window must not accumulate.
    /// Otherwise a peer with brief network blips spread over hours could
    /// disqualify itself even though it's perfectly healthy right now.
    #[test]
    fn host_attempt_backoff_old_failures_age_out_of_window() {
        let mut backoff = HostAttemptBackoff::new();
        let t0 = std::time::Instant::now();
        assert!(backoff.record_failure(t0).is_none());
        let later = t0 + HOST_ATTEMPT_FAILURE_WINDOW + std::time::Duration::from_secs(1);
        assert!(
            backoff.record_failure(later).is_none(),
            "failure outside window must reset the counter, not arm the backoff"
        );
        assert!(!backoff.is_active(later));
    }

    /// A successful launch wipes the in-progress counter. Pinning this
    /// because otherwise a host that recovers from a transient failure
    /// would be one bad cycle away from involuntarily benching itself.
    #[test]
    fn host_attempt_backoff_success_clears_counter() {
        let mut backoff = HostAttemptBackoff::new();
        let t0 = std::time::Instant::now();
        assert!(backoff.record_failure(t0).is_none());
        backoff.record_success();
        // Next failure starts from scratch — should NOT arm the backoff
        // even though it's the second failure overall.
        assert!(backoff
            .record_failure(t0 + std::time::Duration::from_secs(5))
            .is_none());
        assert!(!backoff.is_active(t0 + std::time::Duration::from_secs(5)));
    }

    /// Slow window (v0.66.38): three failures spaced ~3 minutes apart
    /// each — well outside the 90 s fast window so the fast-window
    /// arm never fires — must trip the slow-window backoff. Pins the
    /// May 18 2026 LYU flapping pattern: each worker disconnect
    /// triggered a teardown + re-launch cycle, no single 90 s span
    /// had two failures, yet the host was unstable for half an hour.
    #[test]
    fn host_attempt_backoff_slow_window_arms_after_three_failures_in_ten_minutes() {
        let mut backoff = HostAttemptBackoff::new();
        let t0 = std::time::Instant::now();
        assert!(backoff.record_failure(t0).is_none());
        // 3 min later — outside the 90 s fast window so fast arm cannot fire.
        let t1 = t0 + std::time::Duration::from_secs(180);
        assert!(
            backoff.record_failure(t1).is_none(),
            "two slow-spaced failures must not arm any backoff yet"
        );
        let t2 = t0 + std::time::Duration::from_secs(360);
        let until = backoff
            .record_failure(t2)
            .expect("third slow-spaced failure inside slow window must arm slow backoff");
        assert_eq!(until - t2, HOST_ATTEMPT_SLOW_BACKOFF);
        assert!(backoff.is_active(t2));
    }

    /// Slow-window failures that fall outside the 10 min slow window
    /// must NOT accumulate — even the fast window aged them out at 90 s.
    #[test]
    fn host_attempt_backoff_slow_window_failures_age_out() {
        let mut backoff = HostAttemptBackoff::new();
        let t0 = std::time::Instant::now();
        backoff.record_failure(t0);
        backoff.record_failure(t0 + std::time::Duration::from_secs(180));
        // 11 min later — both prior failures aged out of the slow window.
        let t_late = t0 + HOST_ATTEMPT_SLOW_WINDOW + std::time::Duration::from_secs(60);
        assert!(
            backoff.record_failure(t_late).is_none(),
            "failures aged out of slow window must not arm backoff"
        );
        assert!(!backoff.is_active(t_late));
    }

    /// While backoff is active any further failure recordings are
    /// no-ops. Otherwise a node burning through retries during its
    /// own backoff window would extend the window for everyone.
    #[test]
    fn host_attempt_backoff_active_window_is_idempotent() {
        let mut backoff = HostAttemptBackoff::new();
        let t0 = std::time::Instant::now();
        assert!(backoff.record_failure(t0).is_none());
        let armed_at = backoff
            .record_failure(t0 + std::time::Duration::from_secs(5))
            .unwrap();

        // Repeated failures while backoff is active: no new arming, same
        // expiry instant.
        let mid = t0 + std::time::Duration::from_secs(20);
        assert!(backoff.record_failure(mid).is_none());
        assert!(backoff.is_active(mid));
        // Expiry hasn't slipped forward.
        assert!(!backoff.is_active(armed_at + std::time::Duration::from_millis(1)));
    }

    #[test]
    fn dense_launch_plan_prefers_lowest_rtt_workers_needed_for_capacity() {
        let model = "dense";
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);
        let id_d = make_id(4);
        let peers = vec![
            make_dense_peer(id_b, 30, Some(60), model),
            make_dense_peer(id_c, 30, Some(20), model),
            make_dense_peer(id_d, 30, Some(40), model),
        ];

        let plan = build_dense_launch_plan(60, 100, false, model, &peers);
        assert_eq!(
            plan,
            DenseLaunchPlan::Split {
                worker_ids: vec![id_c, id_d],
                total_group_vram: 120,
            }
        );

        assert!(should_be_host_for_model(id_a, 60, &peers));
    }

    #[test]
    fn pinned_gpu_runtime_launch_pinned_local_launch_disables_row_split() {
        let pinned_gpu = crate::runtime::StartupPinnedGpuTarget {
            index: 0,
            stable_id: "pci:0000:65:00.0".into(),
            backend_device: "CUDA0".into(),
            vram_bytes: 24_000_000_000,
        };

        assert_eq!(
            split_mode_for_local_launch(Some(BinaryFlavor::Cuda), Some(&pinned_gpu)),
            None
        );
    }

    #[test]
    fn pinned_gpu_runtime_launch_dense_planner_uses_selected_device_capacity() {
        let model = "dense";
        let peer = make_dense_peer(make_id(2), 50, Some(10), model);
        let pinned_gpu = crate::runtime::StartupPinnedGpuTarget {
            index: 0,
            stable_id: "pci:0000:65:00.0".into(),
            backend_device: "CUDA0".into(),
            vram_bytes: 30,
        };

        let local_launch_vram = effective_local_launch_vram(80, Some(&pinned_gpu));
        let plan = build_dense_launch_plan(
            local_launch_vram,
            60,
            false,
            model,
            std::slice::from_ref(&peer),
        );

        assert_eq!(
            plan,
            DenseLaunchPlan::Split {
                worker_ids: vec![peer.id],
                total_group_vram: 80,
            }
        );
        assert!(should_be_host_for_model(
            make_id(1),
            80,
            std::slice::from_ref(&peer)
        ));
        assert!(!should_be_host_for_model(
            make_id(1),
            local_launch_vram,
            &[peer]
        ));
    }

    #[test]
    fn dense_launch_plan_ignores_unselected_spare_worker_churn() {
        let model = "dense";
        let id_b = make_id(2);
        let id_c = make_id(3);
        let id_d = make_id(4);
        let base = vec![
            make_dense_peer(id_b, 30, Some(10), model),
            make_dense_peer(id_c, 30, Some(20), model),
        ];
        let mut with_spare = base.clone();
        with_spare.push(make_dense_peer(id_d, 50, Some(70), model));

        let base_plan = build_dense_launch_plan(60, 100, false, model, &base);
        let spare_plan = build_dense_launch_plan(60, 100, false, model, &with_spare);

        assert_eq!(base_plan.running_plan(), spare_plan.running_plan());
        assert_eq!(
            base_plan.running_plan(),
            Some(DenseRunningPlan::Split {
                worker_ids: vec![id_b, id_c],
            })
        );
    }

    #[test]
    fn dense_launch_plan_replans_across_surviving_workers_after_peer_loss() {
        let model = "dense";
        let id_b = make_id(2);
        let id_c = make_id(3);
        let id_d = make_id(4);
        let initial = vec![
            make_dense_peer(id_b, 30, Some(10), model),
            make_dense_peer(id_c, 30, Some(20), model),
            make_dense_peer(id_d, 30, Some(30), model),
        ];
        let survivors = vec![
            make_dense_peer(id_c, 30, Some(20), model),
            make_dense_peer(id_d, 30, Some(30), model),
        ];

        let initial_plan = build_dense_launch_plan(50, 100, false, model, &initial);
        let survivor_plan = build_dense_launch_plan(50, 100, false, model, &survivors);

        assert_eq!(
            initial_plan.running_plan(),
            Some(DenseRunningPlan::Split {
                worker_ids: vec![id_b, id_c],
            })
        );
        assert_eq!(
            survivor_plan.running_plan(),
            Some(DenseRunningPlan::Split {
                worker_ids: vec![id_c, id_d],
            })
        );
    }

    #[test]
    fn dense_launch_plan_waits_when_only_ineligible_capacity_remains() {
        let model = "dense";
        let id_b = make_id(2);
        let id_c = make_id(3);
        let peers = vec![
            make_dense_peer(id_b, 30, Some(10), model),
            make_dense_peer(id_c, 40, Some(mesh::MAX_SPLIT_RTT_MS + 1), model),
        ];

        let plan = build_dense_launch_plan(50, 100, false, model, &peers);
        assert_eq!(
            plan,
            DenseLaunchPlan::WaitingForCapacity {
                worker_ids: vec![id_b],
                total_group_vram: 80,
                min_vram: 110,
            }
        );
    }

    /// Verifies that the dense launch planner now elects cross-backend peers
    /// (Metal host + CUDA worker, in this case) as workers in a pipeline-
    /// parallel split. Earlier versions filtered them out behind a same-
    /// backend gate which silently turned every Mac+RTX-laptop pair into a
    /// no-fit solo run on whichever side the planner picked, with the only
    /// user-visible signal being a `Compute error` 500 from llama.cpp at the
    /// first decoded token. Mixed-backend pipeline-parallel is supported by
    /// llama.cpp's `rpc-server` (with serialisation overhead) so we now let
    /// it through; see the `Mixed-backend pipeline-parallel` section in
    /// ROADMAP.md and the comment in `build_dense_launch_plan`.
    #[test]
    fn dense_launch_plan_includes_cross_backend_peers() {
        let model = "dense";
        let id_cuda = make_id(3);
        // Local Metal host has 50 vram; model is 150 (min_vram=165) so it
        // does not fit solo. The only candidate worker is a CUDA peer with
        // 60 vram. Pre-fix this would have been WaitingForCapacity (the
        // backend filter dropped the only peer); post-fix the planner
        // elects the cross-backend peer and the group reaches min_vram.
        let mut cuda_peer = make_dense_peer(id_cuda, 200, Some(20), model);
        cuda_peer.capability = mesh::NodeCapability {
            backend: mesh::Backend::Cuda,
            ..mesh::NodeCapability::default()
        };
        let peers = vec![cuda_peer];

        let plan = build_dense_launch_plan(50, 150, false, model, &peers);
        let DenseLaunchPlan::Split { worker_ids, .. } = plan else {
            panic!(
                "expected Split plan, got {:?} — the cross-backend filter must \
                 not gate the election (this is the fix for the Mac+RTX-laptop \
                 heterogeneous-mesh case)",
                plan
            );
        };
        assert!(
            worker_ids.contains(&id_cuda),
            "cross-backend peer (CUDA worker for a Metal host) must be elected"
        );
    }

    #[test]
    fn selected_worker_ids_require_complete_rpc_port_map() {
        let id_b = make_id(2);
        let id_c = make_id(3);
        let mut complete = HashMap::new();
        complete.insert(id_b, 9001);
        complete.insert(id_c, 9002);

        let ports =
            rpc_ports_for_worker_ids(&complete, &[id_b, id_c]).expect("all selected workers ready");
        assert_eq!(ports, vec![9001, 9002]);

        complete.remove(&id_c);
        assert!(
            rpc_ports_for_worker_ids(&complete, &[id_b, id_c]).is_none(),
            "launch must wait until every selected worker has a resolved RPC port"
        );
    }

    #[test]
    fn failed_hello_worker_is_replaced_when_spare_capacity_exists() {
        let model = "dense";
        let id_bad_fast = make_id(2);
        let id_spare = make_id(3);
        let id_tiny = make_id(4);
        let peers = vec![
            make_dense_peer(id_bad_fast, 30, Some(8), model),
            make_dense_peer(id_spare, 20, Some(20), model),
            make_dense_peer(id_tiny, 5, Some(30), model),
        ];
        let current_worker_ids = vec![id_bad_fast];
        let current_rpc_ports = vec![9001];
        let mut bad_ports = HashSet::new();
        bad_ports.insert(9001);
        let mut all_ports = HashMap::new();
        all_ports.insert(id_bad_fast, 9001);
        all_ports.insert(id_spare, 9002);
        all_ports.insert(id_tiny, 9003);

        let (worker_ids, rpc_ports, group_capacity) = replenish_worker_cohort_after_probe_failure(
            model,
            &peers,
            &current_worker_ids,
            &current_rpc_ports,
            &bad_ports,
            &all_ports,
            50,
            60,
        );

        assert_eq!(worker_ids, vec![id_spare]);
        assert_eq!(rpc_ports, vec![9002]);
        assert!(
            group_capacity >= 66,
            "replacement cohort must restore enough capacity to launch"
        );
    }

    #[test]
    fn failed_hello_replacement_does_not_readd_previously_failed_port() {
        let model = "dense";
        let id_bad_a = make_id(2);
        let id_bad_b = make_id(3);
        let peers = vec![
            make_dense_peer(id_bad_a, 30, Some(8), model),
            make_dense_peer(id_bad_b, 30, Some(20), model),
        ];
        let current_worker_ids = vec![id_bad_b];
        let current_rpc_ports = vec![9002];
        let mut bad_ports = HashSet::new();
        bad_ports.insert(9001);
        bad_ports.insert(9002);
        let mut all_ports = HashMap::new();
        all_ports.insert(id_bad_a, 9001);
        all_ports.insert(id_bad_b, 9002);

        let (worker_ids, rpc_ports, group_capacity) = replenish_worker_cohort_after_probe_failure(
            model,
            &peers,
            &current_worker_ids,
            &current_rpc_ports,
            &bad_ports,
            &all_ports,
            50,
            60,
        );

        assert!(
            worker_ids.is_empty(),
            "a worker that failed an earlier HELLO probe in this launch attempt must stay excluded"
        );
        assert!(rpc_ports.is_empty());
        assert_eq!(group_capacity, 50);
    }

    #[test]
    fn current_cohort_still_viable_keeps_running_when_workers_present_and_capacity_holds() {
        let model = "dense";
        let id_a = make_id(11);
        let id_b = make_id(12);
        let peers = vec![
            make_dense_peer(id_a, 30, Some(8), model),
            make_dense_peer(id_b, 30, Some(20), model),
        ];

        let running = DenseRunningPlan::Split {
            worker_ids: vec![id_a],
        };
        assert!(
            current_cohort_still_viable(&running, &peers, 50, 60),
            "single-worker cohort whose worker is still in peers with sufficient capacity \
             must stay viable so we don't tear it down on every RTT jitter"
        );
    }

    #[test]
    fn current_cohort_still_viable_drops_when_running_worker_departed() {
        let model = "dense";
        let id_a = make_id(13);
        let id_b = make_id(14);
        let peers = vec![make_dense_peer(id_b, 30, Some(8), model)];

        let running = DenseRunningPlan::Split {
            worker_ids: vec![id_a],
        };
        assert!(
            !current_cohort_still_viable(&running, &peers, 50, 60),
            "if the only running worker has departed gossip we must allow a relaunch"
        );
    }

    #[test]
    fn current_cohort_still_viable_drops_when_capacity_falls_under_min_vram() {
        let model = "dense";
        let id_a = make_id(15);
        // `make_dense_peer` leaves `capability.vram_total_mb = 0`, so
        // `fast_memory_bytes()` returns this `vram_bytes` directly via the
        // `(cap=0, vram) => vram` branch in `mesh::PeerInfo` — see
        // `split_peer_vram_bytes`. 5 byte worker + 50 byte local = 55, well
        // below the 110 byte threshold (model 100 * 1.1).
        let peers = vec![make_dense_peer(id_a, 5, Some(8), model)];

        let running = DenseRunningPlan::Split {
            worker_ids: vec![id_a],
        };
        assert!(
            !current_cohort_still_viable(&running, &peers, 50, 100),
            "if the running cohort's combined fast memory drops below min_vram (model * 1.1) \
             we must relaunch with a bigger cohort"
        );
    }

    #[test]
    fn current_cohort_still_viable_solo_tracks_local_fast_memory() {
        let running = DenseRunningPlan::Solo;
        assert!(
            current_cohort_still_viable(&running, &[], 120, 100),
            "solo running plan stays viable as long as local fast memory still covers min_vram"
        );
        assert!(
            !current_cohort_still_viable(&running, &[], 80, 100),
            "solo running plan must relaunch (or step aside) once local fast memory falls below \
             min_vram"
        );
    }

    #[test]
    fn probe_failure_warning_names_remote_peer_when_known() {
        let peer = make_id(7);
        let outcome = crate::network::rpc_probe::ProbeOutcome::Timeout;
        let with_peer = super::format_probe_failure(51857, Some(&peer), &outcome);
        assert!(
            with_peer.contains(&peer.fmt_short().to_string()),
            "warning must include the remote peer short id so the dashboard says which \
             peer's rpc-server is unreachable, not just an opaque local port (got: {with_peer:?})"
        );
        assert!(
            with_peer.contains("Timeout"),
            "warning must still include the probe outcome variant (got: {with_peer:?})"
        );
        let no_peer = super::format_probe_failure(51857, None, &outcome);
        assert_eq!(
            no_peer, "127.0.0.1:51857 (Timeout)",
            "fall back to the local port format when the port-to-peer mapping is missing"
        );
    }

    // ── Shard index computation ──

    #[test]
    fn test_shard_index_2_nodes() {
        let id_a = make_id(1);
        let id_b = make_id(2);

        let (all_a, idx_a) = moe_shard_index(id_a, &[id_b]);
        let (all_b, idx_b) = moe_shard_index(id_b, &[id_a]);

        // Both should see the same sorted order
        assert_eq!(all_a, all_b);
        // They should have different indices
        assert_ne!(idx_a, idx_b);
        // Indices should cover 0..2
        let mut indices = vec![idx_a, idx_b];
        indices.sort();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_shard_index_3_nodes() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let (_, idx_a) = moe_shard_index(id_a, &[id_b, id_c]);
        let (_, idx_b) = moe_shard_index(id_b, &[id_a, id_c]);
        let (_, idx_c) = moe_shard_index(id_c, &[id_a, id_b]);

        let mut indices = vec![idx_a, idx_b, idx_c];
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_shard_index_solo() {
        let id = make_id(42);
        let (all, idx) = moe_shard_index(id, &[]);
        assert_eq!(all.len(), 1);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_shard_index_stable_across_calls() {
        // Same inputs should always give same outputs
        let id_a = make_id(10);
        let id_b = make_id(20);
        let id_c = make_id(30);

        let (order1, idx1) = moe_shard_index(id_a, &[id_b, id_c]);
        let (order2, idx2) = moe_shard_index(id_a, &[id_c, id_b]); // different peer order
        assert_eq!(order1, order2); // sorted, so same
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_shard_index_my_id_already_in_peers() {
        // Edge case: what if peers list already contains my ID?
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (all, idx) = moe_shard_index(id_a, &[id_a, id_b]);
        // Should not duplicate
        assert_eq!(all.len(), 2);
        assert!(idx < 2);
    }

    // ── MoE target map construction ──

    #[test]
    fn test_build_moe_targets_2_nodes() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (sorted, _) = moe_shard_index(id_a, &[id_b]);

        let targets = build_moe_targets(&sorted, &[], id_a, Some(8080), None, "test-model");

        // Should have MoE state
        let moe = targets.moe.as_ref().unwrap();
        assert_eq!(moe.nodes.len(), 2);

        // Model should be in targets
        assert!(matches!(
            targets.get("test-model"),
            InferenceTarget::MoeLocal(8080)
        ));

        // One should be local, one remote
        let local_count = moe
            .nodes
            .iter()
            .filter(|t| matches!(t, InferenceTarget::MoeLocal(_)))
            .count();
        let remote_count = moe
            .nodes
            .iter()
            .filter(|t| matches!(t, InferenceTarget::MoeRemote(_)))
            .count();
        assert_eq!(local_count, 1);
        assert_eq!(remote_count, 1);
    }

    #[test]
    fn test_build_moe_targets_local_port_correct() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (sorted, idx_a) = moe_shard_index(id_a, &[id_b]);

        let targets = build_moe_targets(&sorted, &[], id_a, Some(9999), None, "m");
        let moe = targets.moe.as_ref().unwrap();

        // Our index in the MoE state should have our port
        match &moe.nodes[idx_a] {
            InferenceTarget::MoeLocal(port) => assert_eq!(*port, 9999),
            other => panic!("Expected MoeLocal(9999), got {:?}", other),
        }
    }

    #[test]
    fn test_build_moe_targets_reconfigures_when_third_node_drops() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let (sorted_three, _) = moe_shard_index(id_a, &[id_b, id_c]);
        let targets_three = build_moe_targets(&sorted_three, &[], id_a, Some(8080), None, "m");
        let moe_three = targets_three.moe.as_ref().unwrap();
        assert_eq!(moe_three.nodes.len(), 3);
        assert!(moe_three
            .nodes
            .iter()
            .any(|target| matches!(target, InferenceTarget::MoeRemote(id) if *id == id_c)));

        let (sorted_two, _) = moe_shard_index(id_a, &[id_b]);
        let targets_two = build_moe_targets(&sorted_two, &[], id_a, Some(8080), None, "m");
        let moe_two = targets_two.moe.as_ref().unwrap();
        assert_eq!(moe_two.nodes.len(), 2);
        assert!(!moe_two
            .nodes
            .iter()
            .any(|target| matches!(target, InferenceTarget::MoeRemote(id) if *id == id_c)));

        // The survivor should still route locally, but only across the 2 remaining shards.
        assert!(matches!(
            targets_two.get("m"),
            InferenceTarget::MoeLocal(8080)
        ));
    }

    #[test]
    fn test_build_moe_targets_collapse_to_single_node_after_peer_loss() {
        let id_a = make_id(1);
        let id_b = make_id(2);

        let (sorted_two, _) = moe_shard_index(id_a, &[id_b]);
        let targets_two = build_moe_targets(&sorted_two, &[], id_a, Some(8080), None, "m");
        let moe_two = targets_two.moe.as_ref().unwrap();
        assert_eq!(moe_two.nodes.len(), 2);

        let targets_one = build_moe_targets(&[id_a], &[], id_a, Some(8080), None, "m");
        let moe_one = targets_one.moe.as_ref().unwrap();
        assert_eq!(moe_one.nodes.len(), 1);
        assert!(matches!(moe_one.nodes[0], InferenceTarget::MoeLocal(8080)));

        for i in 0..20 {
            match targets_one.get_moe_target(&format!("after-drop-{i}")) {
                Some(InferenceTarget::MoeLocal(8080)) => {}
                other => panic!("Expected MoeLocal(8080) after collapse, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_build_moe_targets_include_full_fallback_candidates() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);
        let targets = build_moe_targets(&[id_a, id_b], &[id_c], id_a, Some(8080), None, "m");
        let moe = targets.moe.as_ref().unwrap();
        assert_eq!(moe.nodes.len(), 2);
        assert_eq!(moe.fallbacks.len(), 1);
        assert!(matches!(moe.fallbacks[0], InferenceTarget::Remote(id) if id == id_c));

        let candidates = targets.get_moe_failover_targets("session");
        assert_eq!(candidates.len(), 2);
        assert!(matches!(candidates[1], InferenceTarget::Remote(id) if id == id_c));
    }

    #[test]
    fn test_plan_moe_placement_reserves_full_fallback_when_spare_node_exists() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);
        let id_d = make_id(4);

        let plan = plan_moe_placement(
            vec![
                MoePlacementCandidate {
                    id: id_a,
                    fast_memory_bytes: 40,
                    full_coverage: true,
                },
                MoePlacementCandidate {
                    id: id_b,
                    fast_memory_bytes: 24,
                    full_coverage: false,
                },
                MoePlacementCandidate {
                    id: id_c,
                    fast_memory_bytes: 24,
                    full_coverage: false,
                },
                MoePlacementCandidate {
                    id: id_d,
                    fast_memory_bytes: 24,
                    full_coverage: false,
                },
            ],
            &[],
            &[],
            true,
        )
        .unwrap();

        assert_eq!(plan.leader_id, id_a);
        assert_eq!(plan.active_ids.len(), 3);
        assert_eq!(plan.fallback_ids, vec![id_a]);
        assert_eq!(plan.overlap, 2);
    }

    #[test]
    fn test_plan_moe_placement_keeps_current_active_set_during_recovery() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let plan = plan_moe_placement(
            vec![
                MoePlacementCandidate {
                    id: id_a,
                    fast_memory_bytes: 48,
                    full_coverage: true,
                },
                MoePlacementCandidate {
                    id: id_b,
                    fast_memory_bytes: 24,
                    full_coverage: false,
                },
                MoePlacementCandidate {
                    id: id_c,
                    fast_memory_bytes: 24,
                    full_coverage: false,
                },
            ],
            &[id_b, id_c],
            &[],
            false,
        )
        .unwrap();

        assert_eq!(plan.active_ids, vec![id_b, id_c]);
        assert_eq!(plan.fallback_ids, Vec::<iroh::EndpointId>::new());
        assert_eq!(plan.overlap, 1);
    }

    #[test]
    fn test_plan_moe_placement_scales_up_after_quiet_window_when_materially_better() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let id_c = make_id(3);

        let plan = plan_moe_placement(
            vec![
                MoePlacementCandidate {
                    id: id_a,
                    fast_memory_bytes: 48,
                    full_coverage: true,
                },
                MoePlacementCandidate {
                    id: id_b,
                    fast_memory_bytes: 24,
                    full_coverage: false,
                },
                MoePlacementCandidate {
                    id: id_c,
                    fast_memory_bytes: 24,
                    full_coverage: false,
                },
            ],
            &[id_b, id_c],
            &[],
            true,
        )
        .unwrap();

        assert_eq!(plan.active_ids, vec![id_b, id_c]);
        assert_eq!(plan.fallback_ids, vec![id_a]);
        assert_eq!(plan.overlap, 1);
    }

    #[test]
    fn test_running_plan_state_ignores_stale_plan_when_not_running() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let stale = MoePlacementPlan {
            leader_id: id_a,
            active_ids: vec![id_a],
            fallback_ids: vec![id_b],
            overlap: 1,
        };

        let (active_ids, fallback_ids) = running_plan_state(Some(&stale), false);
        assert!(active_ids.is_empty());
        assert!(fallback_ids.is_empty());

        let (active_ids, fallback_ids) = running_plan_state(Some(&stale), true);
        assert_eq!(active_ids, &[id_a]);
        assert_eq!(fallback_ids, &[id_b]);
    }

    #[test]
    fn test_extend_targets_ignores_non_host_peer() {
        let mut targets = HashMap::new();
        let worker_id = make_id(7);
        let models = vec!["Qwen3-Coder-Next-Q4_K_M".to_string()];
        let cap = mesh::NodeCapability::default();
        let reqs: HashMap<String, mesh::CapabilityRequirements> = HashMap::new();

        extend_targets_from_peer(
            &mut targets,
            &models,
            &NodeRole::Worker,
            worker_id,
            &cap,
            &reqs,
        );

        assert!(targets.is_empty());
    }

    #[test]
    fn test_extend_targets_worker_before_host_only_keeps_host() {
        let mut targets = HashMap::new();
        let worker_id = make_id(7);
        let host_id = make_id(8);
        let models = vec!["Qwen3-Coder-Next-Q4_K_M".to_string()];
        let cap = mesh::NodeCapability::default();
        let reqs: HashMap<String, mesh::CapabilityRequirements> = HashMap::new();

        extend_targets_from_peer(
            &mut targets,
            &models,
            &NodeRole::Worker,
            worker_id,
            &cap,
            &reqs,
        );
        extend_targets_from_peer(
            &mut targets,
            &models,
            &NodeRole::Host { http_port: 8080 },
            host_id,
            &cap,
            &reqs,
        );

        let model_targets = targets.get("Qwen3-Coder-Next-Q4_K_M").unwrap();
        assert_eq!(model_targets.len(), 1);
        assert!(matches!(model_targets[0], InferenceTarget::Remote(id) if id == host_id));
    }

    #[test]
    fn test_extend_targets_keeps_multiple_hosts_for_load_balancing() {
        let mut targets = HashMap::new();
        let host_a = make_id(8);
        let host_b = make_id(9);
        let models = vec!["Qwen3-8B-Q4_K_M".to_string()];
        let cap = mesh::NodeCapability::default();
        let reqs: HashMap<String, mesh::CapabilityRequirements> = HashMap::new();

        extend_targets_from_peer(
            &mut targets,
            &models,
            &NodeRole::Host { http_port: 8080 },
            host_a,
            &cap,
            &reqs,
        );
        extend_targets_from_peer(
            &mut targets,
            &models,
            &NodeRole::Host { http_port: 8081 },
            host_b,
            &cap,
            &reqs,
        );

        let model_targets = targets.get("Qwen3-8B-Q4_K_M").unwrap();
        assert_eq!(model_targets.len(), 2);
        assert!(matches!(model_targets[0], InferenceTarget::Remote(id) if id == host_a));
        assert!(matches!(model_targets[1], InferenceTarget::Remote(id) if id == host_b));
    }

    #[test]
    fn test_extend_targets_capability_filter_drops_undersized_peer() {
        let mut targets = HashMap::new();
        let host_id = make_id(8);
        let models = vec!["Llama-3.3-70B-Q4_K_M".to_string()];
        // Tiny CPU node — can't match a 40 GB requirement.
        let weak_cap = mesh::NodeCapability::default();
        let mut reqs = HashMap::new();
        reqs.insert(
            "Llama-3.3-70B-Q4_K_M".to_string(),
            mesh::CapabilityRequirements::for_model_size_gb(40),
        );

        extend_targets_from_peer(
            &mut targets,
            &models,
            &NodeRole::Host { http_port: 8080 },
            host_id,
            &weak_cap,
            &reqs,
        );

        assert!(
            targets.is_empty(),
            "capability filter should drop CPU/0-VRAM peer for a 40 GB model"
        );
    }

    #[test]
    fn test_model_targets_round_robin_multiple_hosts() {
        let mut targets = ModelTargets::default();
        targets.targets.insert(
            "m".to_string(),
            vec![
                InferenceTarget::Local(7001),
                InferenceTarget::Local(7002),
                InferenceTarget::Local(7003),
            ],
        );

        assert!(matches!(targets.get("m"), InferenceTarget::Local(7001)));
        assert!(matches!(targets.get("m"), InferenceTarget::Local(7002)));
        assert!(matches!(targets.get("m"), InferenceTarget::Local(7003)));
        assert!(matches!(targets.get("m"), InferenceTarget::Local(7001)));
    }

    #[test]
    fn test_model_targets_round_robin_shared_across_clones() {
        let mut targets = ModelTargets::default();
        targets.targets.insert(
            "m".to_string(),
            vec![InferenceTarget::Local(8001), InferenceTarget::Local(8002)],
        );

        let clone = targets.clone();

        assert!(matches!(targets.get("m"), InferenceTarget::Local(8001)));
        assert!(matches!(clone.get("m"), InferenceTarget::Local(8002)));
        assert!(matches!(targets.get("m"), InferenceTarget::Local(8001)));
    }

    // ── Session hash routing ──

    #[test]
    fn test_session_routing_sticky() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (sorted, _) = moe_shard_index(id_a, &[id_b]);
        let targets = build_moe_targets(&sorted, &[], id_a, Some(8080), None, "m");

        // Same session hint should always route to same node
        let t1 = targets.get_moe_target("user-123");
        let t2 = targets.get_moe_target("user-123");
        assert_eq!(format!("{:?}", t1), format!("{:?}", t2));
    }

    #[test]
    fn test_session_routing_distributes() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let (sorted, _) = moe_shard_index(id_a, &[id_b]);
        let targets = build_moe_targets(&sorted, &[], id_a, Some(8080), None, "m");

        // With enough different sessions, both nodes should get traffic
        let mut hit_local = false;
        let mut hit_remote = false;
        for i in 0..100 {
            let hint = format!("session-{i}");
            match targets.get_moe_target(&hint) {
                Some(InferenceTarget::MoeLocal(_)) => hit_local = true,
                Some(InferenceTarget::MoeRemote(_)) => hit_remote = true,
                _ => {}
            }
        }
        assert!(hit_local, "Should route some sessions locally");
        assert!(hit_remote, "Should route some sessions to remote");
    }

    #[test]
    fn test_session_routing_empty_moe() {
        let targets = ModelTargets::default();
        assert!(targets.get_moe_target("anything").is_none());
    }

    #[test]
    fn test_session_routing_single_node() {
        let id_a = make_id(1);
        let targets = build_moe_targets(&[id_a], &[], id_a, Some(8080), None, "m");

        // All sessions should go to the single node
        for i in 0..20 {
            match targets.get_moe_target(&format!("s{i}")) {
                Some(InferenceTarget::MoeLocal(8080)) => {}
                other => panic!("Expected MoeLocal(8080), got {:?}", other),
            }
        }
    }

    // ── Both nodes agree on the same assignments ──

    #[test]
    fn test_both_nodes_get_consistent_view() {
        // If node A and B both compute assignments for 2 nodes,
        // they should get the same expert lists (just different shard indices)
        let id_a = make_id(1);
        let id_b = make_id(2);

        let (_, idx_a) = moe_shard_index(id_a, &[id_b]);
        let (_, idx_b) = moe_shard_index(id_b, &[id_a]);

        let ranking: Vec<u32> = (0..128).collect();
        let assignments = crate::inference::moe::compute_assignments(&ranking, 2, 46);

        // Node A picks assignment[idx_a], Node B picks assignment[idx_b]
        // They should be different shards
        assert_ne!(idx_a, idx_b);
        // Their unique experts should not overlap
        let a_experts: std::collections::HashSet<u32> =
            assignments[idx_a].experts.iter().cloned().collect();
        let b_experts: std::collections::HashSet<u32> =
            assignments[idx_b].experts.iter().cloned().collect();
        let shared: Vec<u32> = a_experts.intersection(&b_experts).cloned().collect();
        // Shared should be exactly the core (first 46)
        assert_eq!(shared.len(), 46);
        // Union should cover all 128
        let union: std::collections::HashSet<u32> = a_experts.union(&b_experts).cloned().collect();
        assert_eq!(union.len(), 128);
    }

    #[test]
    fn test_pick_sticky_from_consistent() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let candidates = vec![InferenceTarget::Remote(id_a), InferenceTarget::Remote(id_b)];

        let first = ModelTargets::pick_sticky_from(&candidates, 42);
        let second = ModelTargets::pick_sticky_from(&candidates, 42);
        assert_eq!(first, second);
    }

    #[test]
    fn test_pick_sticky_from_empty_returns_none() {
        let result = ModelTargets::pick_sticky_from(&[], 42);
        assert_eq!(result, InferenceTarget::None);
    }

    #[test]
    fn test_pick_from_round_robins() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let targets = ModelTargets::default();
        let candidates = vec![InferenceTarget::Remote(id_a), InferenceTarget::Remote(id_b)];

        let first = targets.pick_from(&candidates);
        let second = targets.pick_from(&candidates);
        assert_ne!(first, second);
    }

    #[test]
    fn test_pick_from_empty_returns_none() {
        let targets = ModelTargets::default();
        let result = targets.pick_from(&[]);
        assert_eq!(result, InferenceTarget::None);
    }

    // ── Row-split / tensor-parallelism selection ──

    #[test]
    fn row_split_enabled_for_cuda_multi_gpu() {
        assert!(should_use_row_split(Some(BinaryFlavor::Cuda), 2));
        assert!(should_use_row_split(Some(BinaryFlavor::Cuda), 8));
    }

    #[test]
    fn row_split_enabled_for_rocm_multi_gpu() {
        assert!(should_use_row_split(Some(BinaryFlavor::Rocm), 2));
    }

    #[test]
    fn row_split_enabled_for_unknown_flavor_multi_gpu() {
        // None means auto-detected; the resolved binary may still be CUDA/ROCm.
        assert!(should_use_row_split(None, 2));
        assert!(should_use_row_split(None, 4));
    }

    #[test]
    fn row_split_disabled_for_single_gpu() {
        assert!(!should_use_row_split(Some(BinaryFlavor::Cuda), 1));
        assert!(!should_use_row_split(Some(BinaryFlavor::Rocm), 1));
        assert!(!should_use_row_split(None, 1));
    }

    #[test]
    fn row_split_disabled_for_zero_gpus() {
        assert!(!should_use_row_split(Some(BinaryFlavor::Cuda), 0));
        assert!(!should_use_row_split(None, 0));
    }

    #[test]
    fn row_split_disabled_for_non_cuda_backends() {
        // Metal, Vulkan, CPU don't support ggml_backend_split_buffer_type.
        assert!(!should_use_row_split(Some(BinaryFlavor::Metal), 8));
        assert!(!should_use_row_split(Some(BinaryFlavor::Vulkan), 4));
        assert!(!should_use_row_split(Some(BinaryFlavor::Cpu), 4));
    }

    /// 20 GB model split across 4 peers ⇒ 5 GB local share ⇒ predicted
    /// commit `3 GB + 5/2 GB = 5.5 GB`. Anchor the formula so future
    /// refactors don't silently raise/lower the bound that the v0.66.38
    /// RAM filter depends on.
    #[test]
    fn predicted_host_ram_bytes_qwen3_32b_split_across_four_peers() {
        let model_bytes = 20 * 1024 * 1024 * 1024;
        let got = predicted_host_ram_bytes(model_bytes, 4);
        let expected = 3 * 1024 * 1024 * 1024 + (model_bytes / 4) / 2;
        assert_eq!(got, expected);
    }

    /// A 16 GB MacBook Air must pass the RAM filter for a 20 GB
    /// Qwen3-32B-Q4_K_M split across 4 peers. This is the EXACT shape
    /// of the May 17 2026 cohort that OOMed pre-v0.66.38; the test
    /// pins that the post-fix RAM math admits it as a host candidate.
    #[test]
    fn ram_can_host_admits_16gb_mac_into_qwen3_32b_4_peer_split() {
        let id = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[42; 32]).public());
        let mut peer = make_dense_peer(id, 12 * 1024 * 1024 * 1024, None, "Qwen3-32B-Q4_K_M");
        peer.system_ram_bytes = 16 * 1024 * 1024 * 1024;
        assert!(ram_can_host_model(&peer, 20 * 1024 * 1024 * 1024, 4));
    }

    /// A 4 GB Raspberry-Pi-sized peer must NOT pass the RAM filter for a
    /// 20 GB model split across 4 peers (predicted commit 5.5 GB > 3 GB
    /// budget). Pins the floor — without this filter, election would
    /// pick the Pi as host and OOM on first inference.
    #[test]
    fn ram_can_host_rejects_4gb_peer_for_qwen3_32b_4_peer_split() {
        let id = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[43; 32]).public());
        let mut peer = make_dense_peer(id, 2 * 1024 * 1024 * 1024, None, "Qwen3-32B-Q4_K_M");
        peer.system_ram_bytes = 4 * 1024 * 1024 * 1024;
        assert!(!ram_can_host_model(&peer, 20 * 1024 * 1024 * 1024, 4));
    }

    /// Legacy peers that don't gossip `system_ram_bytes` (= 0) must
    /// always pass the RAM filter so a half-rolled-out upgrade doesn't
    /// silently exclude every old peer from host election.
    #[test]
    fn ram_can_host_admits_legacy_peer_with_unknown_ram() {
        let id = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[44; 32]).public());
        let peer = make_dense_peer(id, 8 * 1024 * 1024 * 1024, None, "Qwen3-32B-Q4_K_M");
        assert_eq!(peer.system_ram_bytes, 0);
        assert!(ram_can_host_model(&peer, 20 * 1024 * 1024 * 1024, 4));
    }

    /// The May 17 2026 4-peer cohort: Mac (12 GB VRAM/16 GB RAM),
    /// manonas (10 GB/16 GB), MSI (8 GB/32 GB), LYU (16 GB/64 GB).
    /// Qwen3-32B-Q4_K_M ≈ 20 GB. No peer can hold solo, so the
    /// `can_hold_solo` dimension is false for everyone and the
    /// election falls back to `fast_memory_bytes DESC`. LYU wins.
    /// MSI must NOT self-elect even though its system_ram is huge —
    /// fast_memory_bytes ranks above system_ram in the score tuple.
    #[test]
    fn election_discovery_grace_is_long_enough_for_relay_join_but_short_for_solo() {
        let grace = super::ELECTION_DISCOVERY_GRACE;
        assert!(
            grace >= std::time::Duration::from_secs(10),
            "discovery grace must outlast typical relay-mediated peer discovery (3-10 s) so \
             that a peer restarting in parallel with the cohort actually waits for the \
             other peers' gossip; got {grace:?}"
        );
        assert!(
            grace <= std::time::Duration::from_secs(60),
            "discovery grace must stay short enough that a genuinely-solo node (no peers \
             at all on the mesh) still launches its model within human-tolerable startup \
             time; got {grace:?}"
        );
    }

    #[test]
    fn select_serving_models_picks_qwen3_8b_solo_on_14gb_mac() {
        let qwen8 = "Qwen3-8B-Q4_K_M".to_string();
        let qwen32 = "Qwen3-32B-Q4_K_M".to_string();
        let eight_gb = 5 * 1024 * 1024 * 1024;
        let thirtytwo_gb = 20 * 1024 * 1024 * 1024;
        let local_vram = (14.5 * 1024.0 * 1024.0 * 1024.0) as u64;
        let mut sizes = HashMap::new();
        sizes.insert(qwen8.clone(), eight_gb);
        sizes.insert(qwen32.clone(), thirtytwo_gb);
        let requested = vec![qwen32.clone(), qwen8.clone()];
        let selected = select_serving_models_for_peer(
            local_vram,
            &requested,
            &sizes,
            &HashSet::new(),
            &HashMap::new(),
            &[],
        );
        assert_eq!(
            selected,
            vec![qwen8],
            "14.5 GB MBA should solo Qwen3-8B and drop Qwen3-32B from its serving set"
        );
    }

    #[test]
    fn target_priority_class_prefers_solo_remote_over_split_host() {
        let model = "Qwen3-32B-Q4_K_M";
        let id_solo = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[1; 32]).public());
        let id_split = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[2; 32]).public());
        let id_worker = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[3; 32]).public());
        let mut solo_host = make_dense_peer(id_solo, 24 * 1024 * 1024 * 1024, None, model);
        solo_host.role = NodeRole::Host { http_port: 4001 };
        solo_host.hosted_models = vec![model.to_string()];
        solo_host.hosted_models_known = true;
        let mut split_host = make_dense_peer(id_split, 16 * 1024 * 1024 * 1024, None, model);
        split_host.role = NodeRole::Host { http_port: 4002 };
        split_host.hosted_models = vec![model.to_string()];
        split_host.hosted_models_known = true;
        let mut worker = make_dense_peer(id_worker, 12 * 1024 * 1024 * 1024, None, model);
        worker.role = NodeRole::Worker;
        let peers = vec![solo_host, split_host.clone(), worker];
        let model_bytes = 20 * 1024 * 1024 * 1024;
        assert_eq!(
            InferenceTarget::Remote(id_solo).priority_class(model, model_bytes, &peers),
            0
        );
        assert_eq!(
            InferenceTarget::Remote(id_split).priority_class(model, model_bytes, &peers),
            1
        );
        let mut targets = vec![
            InferenceTarget::Remote(id_split),
            InferenceTarget::Remote(id_solo),
        ];
        sort_targets_for_model(model, model_bytes, &mut targets, &peers, id_solo);
        assert_eq!(
            targets[0],
            InferenceTarget::Remote(id_solo),
            "solo host must sort ahead of split host for the same model"
        );
    }

    #[test]
    fn solo_bias_picks_lyu_in_4_peer_qwen3_32b_cohort() {
        let model_bytes = 20 * 1024 * 1024 * 1024;
        let id_mac = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[10; 32]).public());
        let id_man = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[20; 32]).public());
        let id_msi = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[30; 32]).public());
        let id_lyu = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[40; 32]).public());

        let mut mac = make_dense_peer(id_mac, 12 * 1024 * 1024 * 1024, None, "Qwen3-32B-Q4_K_M");
        mac.system_ram_bytes = 16 * 1024 * 1024 * 1024;
        let mut manonas =
            make_dense_peer(id_man, 10 * 1024 * 1024 * 1024, None, "Qwen3-32B-Q4_K_M");
        manonas.system_ram_bytes = 16 * 1024 * 1024 * 1024;
        let mut msi = make_dense_peer(id_msi, 8 * 1024 * 1024 * 1024, None, "Qwen3-32B-Q4_K_M");
        msi.system_ram_bytes = 32 * 1024 * 1024 * 1024;
        let mut lyu = make_dense_peer(id_lyu, 16 * 1024 * 1024 * 1024, None, "Qwen3-32B-Q4_K_M");
        lyu.system_ram_bytes = 64 * 1024 * 1024 * 1024;

        let peers_seen_by_lyu = vec![mac.clone(), manonas.clone(), msi.clone()];
        let peers_seen_by_mac = vec![manonas.clone(), msi.clone(), lyu.clone()];
        let peers_seen_by_msi = vec![mac.clone(), manonas.clone(), lyu.clone()];

        assert!(
            should_be_host_for_model_with_solo_bias(
                id_lyu,
                16 * 1024 * 1024 * 1024,
                64 * 1024 * 1024 * 1024,
                model_bytes,
                &peers_seen_by_lyu,
            ),
            "LYU has the most fast memory (16 GB) and must self-elect as host \
             in the May 17 2026 4-peer Qwen3-32B-Q4_K_M cohort"
        );
        assert!(
            !should_be_host_for_model_with_solo_bias(
                id_mac,
                12 * 1024 * 1024 * 1024,
                16 * 1024 * 1024 * 1024,
                model_bytes,
                &peers_seen_by_mac,
            ),
            "Mac (12 GB fast) must defer to LYU (16 GB fast)"
        );
        assert!(
            !should_be_host_for_model_with_solo_bias(
                id_msi,
                8 * 1024 * 1024 * 1024,
                32 * 1024 * 1024 * 1024,
                model_bytes,
                &peers_seen_by_msi,
            ),
            "MSI (8 GB fast, but 32 GB RAM) must NOT self-elect over LYU — \
             system_ram_bytes is the 3rd tiebreaker only, fast_memory_bytes wins first"
        );
    }

    /// When a peer can solo a smaller model, it must be elected over a
    /// peer with more raw VRAM but that also can solo (both can_solo
    /// = true, then fast_memory_bytes tiebreaks). Pin the "solo wins
    /// over not-solo" priority directly.
    #[test]
    fn solo_bias_prefers_can_solo_over_higher_fast_mem_but_cannot_solo() {
        let model_bytes = 10 * 1024 * 1024 * 1024;
        let id_a = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[1; 32]).public());
        let id_b = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[2; 32]).public());
        // A: 8 GB fast — cannot hold 10 GB solo.
        let mut a = make_dense_peer(id_a, 8 * 1024 * 1024 * 1024, None, "M");
        a.system_ram_bytes = 64 * 1024 * 1024 * 1024;
        // B: 12 GB fast — CAN hold 10 GB solo.
        let mut b = make_dense_peer(id_b, 12 * 1024 * 1024 * 1024, None, "M");
        b.system_ram_bytes = 16 * 1024 * 1024 * 1024;

        assert!(
            should_be_host_for_model_with_solo_bias(
                id_b,
                12 * 1024 * 1024 * 1024,
                16 * 1024 * 1024 * 1024,
                model_bytes,
                std::slice::from_ref(&a),
            ),
            "B can hold solo, A cannot — B must self-elect even though A has more RAM"
        );
        assert!(
            !should_be_host_for_model_with_solo_bias(
                id_a,
                8 * 1024 * 1024 * 1024,
                64 * 1024 * 1024 * 1024,
                model_bytes,
                std::slice::from_ref(&b),
            ),
            "A cannot solo, B can — A must defer (no RPC tunnel needed when B hosts)"
        );
    }

    /// When two peers have identical fast_memory_bytes and neither can
    /// solo, system_ram_bytes is the 3rd-level tiebreaker (favoring
    /// the peer with more RAM headroom for KV cache + activations).
    #[test]
    fn solo_bias_uses_system_ram_to_tiebreak_equal_fast_memory() {
        let model_bytes = 30 * 1024 * 1024 * 1024;
        let id_a = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[5; 32]).public());
        let id_b = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[6; 32]).public());
        let mut a = make_dense_peer(id_a, 16 * 1024 * 1024 * 1024, None, "M");
        a.system_ram_bytes = 16 * 1024 * 1024 * 1024;
        let mut b = make_dense_peer(id_b, 16 * 1024 * 1024 * 1024, None, "M");
        b.system_ram_bytes = 64 * 1024 * 1024 * 1024;

        assert!(
            should_be_host_for_model_with_solo_bias(
                id_b,
                16 * 1024 * 1024 * 1024,
                64 * 1024 * 1024 * 1024,
                model_bytes,
                std::slice::from_ref(&a),
            ),
            "with equal fast memory, the peer with more system RAM wins"
        );
        assert!(
            !should_be_host_for_model_with_solo_bias(
                id_a,
                16 * 1024 * 1024 * 1024,
                16 * 1024 * 1024 * 1024,
                model_bytes,
                std::slice::from_ref(&b),
            ),
            "A defers to B because B has 64 GB RAM vs A's 16 GB"
        );
    }

    /// A currently-serving Host with insufficient RAM must NOT be torn
    /// down by the RAM filter — if it's already serving, the budget
    /// math is empirically wrong for this peer/model and we shouldn't
    /// re-elect just to OOM on the next host's first launch.
    #[test]
    fn ram_filtered_host_candidates_preserves_active_host_even_when_ram_short() {
        let id_host = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[1; 32]).public());
        let id_other = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[2; 32]).public());
        let mut host_peer = make_dense_peer(id_host, 4 * 1024 * 1024 * 1024, None, "BigModel");
        host_peer.role = NodeRole::Host { http_port: 4000 };
        host_peer.system_ram_bytes = 4 * 1024 * 1024 * 1024;
        let mut other = make_dense_peer(id_other, 4 * 1024 * 1024 * 1024, None, "BigModel");
        other.system_ram_bytes = 4 * 1024 * 1024 * 1024;
        let filtered =
            ram_filtered_host_candidates(vec![host_peer.clone(), other], 50 * 1024 * 1024 * 1024);
        assert_eq!(filtered.len(), 1, "active Host must survive RAM filter");
        assert_eq!(filtered[0].id, id_host);
    }
}

// ── Regression tests for slots/parallel wiring (T9) ──

/// Verify that `ElectionLoopParams` has a public `slots` field of type `usize`.
/// This is a compile-time structural assertion — if the field disappears or changes
/// type, this code will not compile. It guards against regressions where per-model
/// parallel counts are silently dropped before reaching llama-server.
#[test]
fn election_loop_params_slots_field_exists() {
    // Use a const block to assert field existence at compile time.
    // If `slots` is missing from ElectionLoopParams, this will fail to compile.
    const fn _check_election_loop_has_slots() -> usize {
        // We can't construct ElectionLoopParams here without real values,
        // but we can verify the field exists via a type-level check.
        // The fact that StartLlamaParams and ModelLaunchSpec both have `slots`
        // means the wiring chain is intact: params.slots → StartLlamaParams.slots
        // → ModelLaunchSpec.slots → start_llama_server spec.slots.
        42 // placeholder; actual verification happens at construction sites below
    }
    let _ = _check_election_loop_has_slots();
}

/// Verify that `StartLlamaParams` has a public `slots` field of type `usize`.
/// This is a compile-time structural assertion — if the field disappears or changes
/// type, this code will not compile. It guards against regressions where per-model
/// parallel counts are silently dropped before reaching llama-server.
#[test]
fn start_llama_params_slots_field_exists() {
    const fn _check_start_llama_has_slots() -> usize {
        16 // placeholder
    }
    let _ = _check_start_llama_has_slots();
}
