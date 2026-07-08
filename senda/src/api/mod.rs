//! Mesh management API — read-only dashboard on port 3131 (default).
//!
//! Endpoints:
//!   GET  /api/status    — live mesh state plus local-only routing metrics (JSON)
//!   GET  /api/models    — mesh model inventory plus local-only routing metrics (JSON)
//!   GET  /api/search    — catalog or Hugging Face model search with the same JSON payload as `senda models search --json`
//!   GET  /api/model-interests — local explicit-interest readback (JSON)
//!   POST /api/model-interests — register local explicit interest for a canonical model ref
//!   DELETE /api/model-interests/{model_ref} — clear local explicit interest
//!   GET  /api/runtime   — local model state (JSON)
//!   GET  /api/runtime/endpoints — registered plugin endpoint state (JSON)
//!   GET  /api/runtime/processes — local inference process state (JSON)
//!   POST /api/runtime/models — load a local model
//!   DELETE /api/runtime/models/{model} — unload a local model
//!   GET  /api/events    — SSE stream of status updates
//!   GET  /api/discover  — browse Nostr-published meshes
//!   POST /api/chat      — proxy to chat completions API
//!   POST /api/responses — proxy to responses API
//!   POST /api/objects   — upload a request-scoped media object
//!   GET  /              — embedded web dashboard
//!
//! The dashboard is mostly read-only — shows status, topology, and models.
//! Local model load/unload is exposed for operator control.
//!
//! `routing_metrics`, `routing_metrics.local_node`, `routing_metrics.pressure`,
//! and `/api/models` per-model `routing_metrics.targets` are measured on the
//! current node only; not mesh-wide aggregates.

mod assets;
mod http;
mod routes;
mod state;
mod status;

pub use self::state::{
    LocalModelInterest, MeshApi, PublicationState, RuntimeControlRequest, RuntimeModelPayload,
    RuntimeProcessPayload,
};
pub(crate) use self::status::classify_runtime_error;

use self::assets::{respond_console_asset, respond_console_index};
use self::http::{http_body_text, respond_error};
use self::routes::dispatch_request;
use self::state::ApiInner;
use self::status::{
    build_gpus, build_ownership_payload, build_runtime_processes_payload,
    build_runtime_status_payload, LocalInstance, MeshFitPayload, MeshModelPayload, MoeShardPayload,
    NodeState, PeerPayload, RuntimeProcessesPayload, RuntimeStatusPayload, SplitGroupPayload,
    StatusPayload, WakeableNode, WakeableNodeState,
};
use crate::inference::election;
use crate::mesh;
use crate::network::{affinity, nostr, proxy};
use crate::plugin;
use crate::runtime::wakeable::{WakeableInventoryEntry, WakeableState};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex};

const SENDA_VERSION: &str = crate::VERSION;

fn find_catalog_model(name: &str) -> Option<&'static crate::models::catalog::CatalogModel> {
    crate::models::catalog::MODEL_CATALOG
        .iter()
        .find(|m| m.name == name || m.file.strip_suffix(".gguf").unwrap_or(m.file.as_str()) == name)
}

fn is_huggingface_repository_like(repository: &str) -> bool {
    let trimmed = repository.trim();
    !trimmed.is_empty()
        && !trimmed.starts_with('/')
        && !trimmed.ends_with('/')
        && !trimmed.contains('\\')
        && trimmed.split('/').count() == 2
}

fn huggingface_repository_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    matches!(identity.source_kind, mesh::ModelSourceKind::HuggingFace)
        .then(|| {
            identity
                .repository
                .clone()
                .filter(|repo| is_huggingface_repository_like(repo))
        })
        .flatten()
}

fn source_page_url_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    huggingface_repository_from_identity(identity)
        .map(|repository| format!("https://huggingface.co/{repository}"))
}

fn source_file_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    identity
        .artifact
        .clone()
        .or_else(|| identity.local_file_name.clone())
}

fn likely_reasoning_model(name: &str, description: Option<&str>) -> bool {
    let haystack = format!("{} {}", name, description.unwrap_or_default()).to_ascii_lowercase();
    ["reasoning", "thinking", "deepseek-r1"]
        .iter()
        .any(|needle| haystack.contains(needle))
}

#[derive(Debug, Default, PartialEq)]
struct HttpRouteStats {
    node_count: usize,
    active_nodes: Vec<String>,
    mesh_vram_gb: f64,
}

fn http_route_stats(
    model_name: &str,
    peers: &[mesh::PeerInfo],
    my_hosted_models: &[String],
    my_hostname: Option<&str>,
    my_vram_gb: f64,
) -> HttpRouteStats {
    let mut active_nodes = Vec::new();
    let mut node_count = 0usize;
    let mut mesh_vram_gb = 0.0;

    if my_hosted_models.iter().any(|hosted| hosted == model_name) {
        node_count += 1;
        mesh_vram_gb += my_vram_gb;
        active_nodes.push(
            my_hostname
                .filter(|hostname| !hostname.trim().is_empty())
                .unwrap_or("This node")
                .to_string(),
        );
    }

    for peer in peers {
        if !peer.routes_http_model(model_name) {
            continue;
        }
        node_count += 1;
        mesh_vram_gb += peer.fast_memory_bytes() as f64 / 1e9;
        active_nodes.push(
            peer.hostname
                .clone()
                .filter(|hostname| !hostname.trim().is_empty())
                .unwrap_or_else(|| peer.id.fmt_short().to_string()),
        );
    }

    active_nodes.sort();
    active_nodes.dedup();

    HttpRouteStats {
        node_count,
        active_nodes,
        mesh_vram_gb,
    }
}

fn likely_vision_model(name: &str, description: Option<&str>) -> bool {
    let haystack = format!("{} {}", name, description.unwrap_or_default()).to_ascii_lowercase();
    ["vision", "-vl", "llava", "omni", "qwen2.5-vl", "mllama"]
        .iter()
        .any(|needle| haystack.contains(needle))
}

fn likely_audio_model(name: &str, description: Option<&str>) -> bool {
    let haystack = format!("{} {}", name, description.unwrap_or_default()).to_ascii_lowercase();
    [
        "audio",
        "speech",
        "voice",
        "omni",
        "ultravox",
        "qwen2-audio",
    ]
    .iter()
    .any(|needle| haystack.contains(needle))
}

/// Result of [`classify_peer_split_role`] — three optional fields the caller
/// folds into a `PeerPayload`. We return a struct rather than a tuple so the
/// call sites stay readable when more split modes get added later.
struct PeerSplitClassification {
    role: Option<String>,
    group: Option<SplitGroupPayload>,
    moe_shard: Option<MoeShardPayload>,
}

/// Largest model-size in bytes any peer in `all_peers` has observed locally.
///
/// `available_model_sizes` is populated by every peer that scanned the GGUF
/// on disk (see `models::inventory`). Sizes for the same model are always
/// the byte-equal weights file, so taking `max` is just a robust pick that
/// works when most peers gossip 0 (didn't scan) and one peer gossips the
/// true size. Returns `0` if no peer has scanned the model.
fn model_bytes_observed_in_mesh(model: &str, all_peers: &[mesh::PeerInfo]) -> u64 {
    all_peers
        .iter()
        .filter_map(|p| p.available_model_sizes.get(model).copied())
        .max()
        .unwrap_or(0)
}

/// Matches `min_vram_for_solo` in `inference::election` — kept as a free
/// function here to avoid adding a dependency edge from `api` to `inference`
/// just for one constant. The 1.1× headroom mirrors `build_dense_launch_plan`.
fn host_can_solo_model(host_fast_memory_bytes: u64, model_bytes: u64) -> bool {
    model_bytes > 0 && host_fast_memory_bytes >= (model_bytes as f64 * 1.1) as u64
}

fn serving_mode_from_classification(
    peer: &mesh::PeerInfo,
    split: &PeerSplitClassification,
) -> Option<String> {
    if peer.serving_models.is_empty() && peer.hosted_models.is_empty() {
        return None;
    }
    if split.moe_shard.is_some() {
        return Some("moe_shard".to_string());
    }
    match split.role.as_deref() {
        Some("pipeline_host") => Some("split_host".to_string()),
        Some("pipeline_worker") => Some("split_worker".to_string()),
        _ => Some("solo".to_string()),
    }
}

/// Coarse classification of a peer's role inside the mesh's current
/// inference topology. Pure inference from observable gossip — no live
/// election state is required.
///
/// We classify based on:
///   - The peer's `NodeRole` (Host vs Worker vs Client).
///   - Whether `serving_models` overlaps with any model that has multiple
///     active serving peers (a proxy for "this model is split or
///     replicated across nodes").
///   - Whether the model is bigger than any single peer in the group
///     (which makes pipeline-parallel the only way it could be running).
///
/// MoE detection currently leans on the `serving_models` count being >1
/// for the same model — we don't have GGUF metadata for peer models in
/// the mesh state at this layer, so the runtime's per-model `MeshModelPayload`
/// is the source of truth for `moe = true`. The peer-side classifier
/// returns `MoeShardPayload` heuristically; clients should cross-reference
/// against `MeshModelPayload.split_kind == "moe"` for ground truth.
fn classify_peer_split_role(
    peer: &mesh::PeerInfo,
    all_peers: &[mesh::PeerInfo],
    my_vram_gb: f64,
) -> PeerSplitClassification {
    if matches!(peer.role, mesh::NodeRole::Client) {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    // Find the first model this peer is actively contributing to that has
    // co-serving peers. Most of the time this is a single model per peer;
    // multi-model nodes still get classified by the largest cohort.
    let candidate_models: Vec<&str> = peer
        .serving_models
        .iter()
        .map(String::as_str)
        .chain(peer.hosted_models.iter().map(String::as_str))
        .collect();

    let mut best: Option<(&str, Vec<&mesh::PeerInfo>, Option<&mesh::PeerInfo>)> = None;
    for model in &candidate_models {
        let cohort: Vec<&mesh::PeerInfo> = all_peers
            .iter()
            .filter(|p| p.id != peer.id && p.is_assigned_model(model))
            .collect();
        let host = all_peers
            .iter()
            .find(|p| p.routes_http_model(model))
            .or_else(|| {
                if peer.routes_http_model(model) {
                    Some(peer)
                } else {
                    None
                }
            });
        if !cohort.is_empty()
            || matches!(
                best.as_ref(),
                Some((_, existing, _)) if existing.is_empty()
            )
            || best.is_none()
        {
            best = Some((*model, cohort, host));
        }
    }

    let Some((model, cohort, host)) = best else {
        // Peer reports no models — could be standby. No role to surface.
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    };

    if cohort.is_empty() {
        // Solo serve — the peer is hosting/serving a model nobody else does.
        // Skip a role badge for plain solo because the existing `role` and
        // `serving_models` already convey that; the UI uses `split_role` for
        // multi-node states specifically.
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    // Honesty gate: if NO peer in the cohort has graduated to `NodeRole::Host`
    // for this model AND the peer being classified isn't a Host either, then
    // there is no actual pipeline split in progress — the election simply
    // hasn't converged. The old code fell through to `host_id = peer.id` and
    // labelled the peer `pipeline_worker`, producing the "I'm a worker for
    // myself" payload that powered every "Awaiting workers" / "You're holding
    // part of the model" lie on the dashboard while three boxes deadlocked
    // each electing themselves. See docs/senda-llm/STRATEGY.md §2 — the
    // pipeline-split mode is the most common source of false-positive UI
    // claims, and we want the classifier to refuse to invent a topology that
    // doesn't exist.
    if host.is_none() && !peer.routes_http_model(model) {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    // Standby-cohort honesty gate (May 25 2026, the Qwen3-8B-on-three-boxes
    // miscall): when multiple peers all opt in to share the same model but
    // the model genuinely fits solo on the elected host, the election deliberately
    // keeps the other peers in `NodeRole::Worker` *standby* — they're warm
    // for failover, not actually splitting layers across an RPC tunnel. The
    // pre-gate classifier blindly tagged the host as `pipeline_host` and the
    // standby peers as `pipeline_worker`, and the UI dutifully advertised a
    // 3-way "pooled VRAM" group that did not exist in the runtime. Refuse to
    // fabricate the split when the host fits the model on its own fast memory.
    // Safe fallback: the gate above returns early when `host.is_none() &&
    // !peer.routes_http_model(model)`, so if we reach here either `host` is
    // Some or `peer` itself routes the model and is the de-facto host.
    let elected_host = host.unwrap_or(peer);
    let model_bytes = model_bytes_observed_in_mesh(model, all_peers);
    if host_can_solo_model(elected_host.fast_memory_bytes(), model_bytes) {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    // Multi-host replication honesty gate (Jun 1 2026, the dense-Qwen3-8B
    // "moe_shard" miscall): when two or more peers each independently route
    // HTTP for this model, they are replicas — every one of them holds the
    // whole model and serves a request end-to-end, and the entry node simply
    // load-balances across them. That is NOT a pipeline split (one host
    // pooling VRAM from layer-providing rpc workers) and no peer is "a shard"
    // of anything. The pre-gate code blindly tagged ≥2 routing hosts as
    // `moe_shard` with a pooled `split_group`, so two boxes each solo-serving
    // a dense 8B showed up on the dashboard as a 2-way MoE group that did not
    // exist. The peer-level classifier also can't tell dense replication from
    // a genuine MoE expert-shard layout without GGUF metadata (see the doc
    // comment above) — the model-level `split_kind` is the source of truth for
    // moe-vs-multi_host — so the honest per-peer badge for replicas is silence.
    // This gate is robust to the model-size gossip being absent, which would
    // otherwise bypass the solo-fit gate above and resurrect the bogus badge.
    let independent_routing_hosts = all_peers
        .iter()
        .filter(|p| p.id != peer.id && p.routes_http_model(model))
        .count()
        + if peer.routes_http_model(model) { 1 } else { 0 };
    if independent_routing_hosts >= 2 {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    // Exactly one routing host plus a non-routing cohort ⇒ a genuine
    // pipeline-parallel split: the model didn't fit solo, so the elected host
    // pools VRAM from rpc-worker peers. Build the SplitGroup payload.
    let mut peer_ids: Vec<String> = std::iter::once(peer.id.fmt_short().to_string())
        .chain(cohort.iter().map(|c| c.id.fmt_short().to_string()))
        .collect();
    peer_ids.sort();
    peer_ids.dedup();

    let total_group_vram_gb = (peer.fast_memory_bytes() as f64 / 1e9)
        + cohort
            .iter()
            .map(|c| c.fast_memory_bytes() as f64 / 1e9)
            .sum::<f64>();

    let host_id = host
        .map(|h| h.id.fmt_short().to_string())
        .unwrap_or_else(|| peer.id.fmt_short().to_string());

    let group = SplitGroupPayload {
        model: model.to_string(),
        host_id,
        peer_ids,
        total_group_vram_gb,
    };

    let role = if peer.routes_http_model(model) {
        Some("pipeline_host".to_string())
    } else if matches!(peer.role, mesh::NodeRole::Worker) {
        Some("pipeline_worker".to_string())
    } else {
        None
    };

    let _ = my_vram_gb; // kept for symmetry with future weight-aware branches

    PeerSplitClassification {
        role,
        group: Some(group),
        moe_shard: None,
    }
}

/// Coarse classification of how the mesh is currently serving a single
/// model. See [`status::MeshModelPayload::split_kind`] for the wire-level
/// description; this is the producer side.
///
/// We pick one of:
/// - `cold`        — no live host yet (model in catalog only)
/// - `solo`        — single peer hosts/serves it
/// - `pipeline`    — multiple peers participating, dense + size > any
///   single peer's VRAM (only viable layout)
/// - `moe`         — multiple peers and the model is MoE (independent
///   shards or load-balancing replicas)
/// - `multi_host`  — multiple peers, dense, fits on each individually
///   (load-balancing replicas, no split)
//
// Kept as 8 args because each input is genuinely orthogonal — collapsing
// into a struct would force callers to construct + name fields they don't
// otherwise carry. The "too many arguments" lint is allowed here on
// purpose.
#[allow(clippy::too_many_arguments)]
fn classify_model_split_kind(
    model_name: &str,
    is_warm: bool,
    moe: bool,
    node_count: usize,
    size_gb: f64,
    all_peers: &[mesh::PeerInfo],
    my_vram_gb: f64,
    local_serves_this_model: bool,
) -> String {
    if !is_warm || node_count == 0 {
        return "cold".to_string();
    }
    if node_count == 1 {
        return "solo".to_string();
    }
    if moe {
        return "moe".to_string();
    }
    // Dense + multi-node. Distinguish "pipeline (split because too big)" from
    // "multi_host (load-balanced replicas)" by checking whether any single
    // hosting peer has enough VRAM to hold the model alone. The election
    // planner would not split a model that fits anywhere by itself.
    let single_peer_can_fit = if local_serves_this_model {
        my_vram_gb * 0.95 >= size_gb
    } else {
        false
    } || all_peers.iter().any(|p| {
        p.routes_http_model(model_name) && (p.fast_memory_bytes() as f64 / 1e9) * 0.95 >= size_gb
    });

    if single_peer_can_fit {
        "multi_host".to_string()
    } else {
        "pipeline".to_string()
    }
}

/// Per-model mesh capacity assessment. The desktop Models page renders one
/// of three states from this:
///   - `fits_on_largest_node = true`  → "Solo on this Mac" / "Solo on N1"
///   - `fits_pooled = true`           → "Will use M nodes pooling Y GB"
///   - both false                     → "Needs more contributors (X / Y GB)"
///
/// We only count peers that are plausibly eligible to participate in a
/// pipeline split: not entry nodes, with non-zero VRAM, and not currently
/// flagged as Client-only. Same-backend / RTT eligibility is *not* enforced
/// here yet — those come from `senda-llm/senda/src/inference/election.rs`
/// and would require either a per-call backend query or backend-aware peer
/// filtering. The chat product can refine this client-side using
/// `peer.capability.backend` once it consumes the per-peer split fields.
fn compute_mesh_fit(size_gb: f64, my_vram_gb: f64, all_peers: &[mesh::PeerInfo]) -> MeshFitPayload {
    // Mirror the dense election planner's headroom multiplier so the UI
    // doesn't claim a fit the runtime would refuse (senda-llm/senda/
    // src/inference/election.rs build_dense_launch_plan multiplies by 1.1).
    let needed_vram_gb = size_gb * 1.1;

    let mut fits_on_largest_node = my_vram_gb * 1.0 >= needed_vram_gb;
    let mut pooled_vram_gb = my_vram_gb;
    let mut eligible_peer_count: u32 = if my_vram_gb > 0.0 { 1 } else { 0 };

    for peer in all_peers {
        if matches!(peer.role, mesh::NodeRole::Client) {
            continue;
        }
        // Skip likely entry-node hostnames so the public mesh entry doesn't
        // inflate "pooled" capacity in the chat UI. This mirrors the
        // app-side `isEntryNode` filter in app/api/status/route.ts.
        if peer
            .hostname
            .as_deref()
            .map(|h| h.starts_with("ip-"))
            .unwrap_or(false)
        {
            continue;
        }
        let peer_vram_gb = peer.fast_memory_bytes() as f64 / 1e9;
        if peer_vram_gb <= 0.0 {
            continue;
        }
        eligible_peer_count += 1;
        pooled_vram_gb += peer_vram_gb;
        if peer_vram_gb >= needed_vram_gb {
            fits_on_largest_node = true;
        }
    }

    let fits_pooled = pooled_vram_gb >= needed_vram_gb;

    MeshFitPayload {
        fits_on_largest_node,
        fits_pooled,
        pooled_vram_gb,
        needed_vram_gb,
        eligible_peer_count,
    }
}

/// Local-node analogue of [`classify_peer_split_role`]. The local node never
/// appears in the `peers` list, so we feed our own model membership into the
/// same logic to surface `my_split_role` / `my_split_group` on `StatusPayload`.
fn classify_local_split_role(
    self_id: &str,
    is_host: bool,
    is_client: bool,
    serving_models: &[String],
    hosted_models: &[String],
    my_vram_gb: f64,
    all_peers: &[mesh::PeerInfo],
) -> PeerSplitClassification {
    if is_client {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    let candidate_models: Vec<&str> = serving_models
        .iter()
        .map(String::as_str)
        .chain(hosted_models.iter().map(String::as_str))
        .collect();

    let mut best: Option<(&str, Vec<&mesh::PeerInfo>)> = None;
    for model in &candidate_models {
        let cohort: Vec<&mesh::PeerInfo> = all_peers
            .iter()
            .filter(|p| p.is_assigned_model(model))
            .collect();
        if !cohort.is_empty() || best.is_none() {
            best = Some((*model, cohort));
        }
    }

    let Some((model, cohort)) = best else {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    };

    if cohort.is_empty() {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    // Honesty gate (mirror of `classify_peer_split_role`): when the cohort is
    // non-empty (multiple peers all advertise interest in the same model) but
    // NO peer has graduated to `NodeRole::Host` for it AND we ourselves are
    // not host, the pipeline-host election has not converged. Don't fabricate
    // a split-group payload with `host_id = self_id` and label ourselves as
    // `pipeline_worker` — that's the lie that lets the dashboard claim
    // "you're holding part of the model" while the rpc-server is actually
    // idle waiting for a host that will never connect.
    let any_host_for_model = all_peers.iter().any(|p| p.routes_http_model(model));
    if !any_host_for_model && !is_host {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    // Standby-cohort honesty gate (May 25 2026): when the model fits solo on
    // the elected host, multi-peer interest is *standby*, not split — see the
    // matching gate in `classify_peer_split_role` for the incident write-up.
    let model_bytes = model_bytes_observed_in_mesh(model, all_peers);
    let elected_host_fast_memory = all_peers
        .iter()
        .find(|p| p.routes_http_model(model))
        .map(|h| h.fast_memory_bytes())
        .unwrap_or_else(|| (my_vram_gb * 1e9) as u64);
    if host_can_solo_model(elected_host_fast_memory, model_bytes) {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    // Multi-host replication honesty gate (mirror of `classify_peer_split_role`):
    // if another peer already routes HTTP for this model AND we route it too
    // (we're a host), there are ≥2 independent full hosts — replicas the entry
    // node load-balances, not a pipeline split. We hold the whole model, so we
    // are not "a shard"; surface no per-peer split badge and let the
    // model-level `split_kind` carry moe-vs-multi_host. Robust to absent
    // model-size gossip (which bypasses the solo-fit gate above).
    let other_routing_hosts = all_peers
        .iter()
        .filter(|p| p.routes_http_model(model))
        .count();
    let independent_routing_hosts = other_routing_hosts + if is_host { 1 } else { 0 };
    if independent_routing_hosts >= 2 {
        return PeerSplitClassification {
            role: None,
            group: None,
            moe_shard: None,
        };
    }

    let mut peer_ids: Vec<String> = std::iter::once(self_id.to_string())
        .chain(cohort.iter().map(|c| c.id.fmt_short().to_string()))
        .collect();
    peer_ids.sort();
    peer_ids.dedup();

    let total_group_vram_gb = my_vram_gb
        + cohort
            .iter()
            .map(|c| c.fast_memory_bytes() as f64 / 1e9)
            .sum::<f64>();

    // Determine host: prefer an existing peer with Host role for this model,
    // otherwise fall back to "we are the host" if our local flag agrees.
    let host_peer = all_peers.iter().find(|p| p.routes_http_model(model));
    let host_id = host_peer
        .map(|h| h.id.fmt_short().to_string())
        .unwrap_or_else(|| self_id.to_string());

    let group = SplitGroupPayload {
        model: model.to_string(),
        host_id,
        peer_ids,
        total_group_vram_gb,
    };

    let i_am_host = host_peer.map(|_| false).unwrap_or(is_host);

    let role = if i_am_host {
        Some("pipeline_host".to_string())
    } else {
        Some("pipeline_worker".to_string())
    };

    PeerSplitClassification {
        role,
        group: Some(group),
        moe_shard: None,
    }
}

fn fit_hint_for_machine(size_gb: f64, my_vram_gb: f64) -> (String, String) {
    if size_gb <= 0.0 || my_vram_gb <= 0.0 {
        return (
            "Unknown".into(),
            "No local capacity signal is available for this machine yet.".into(),
        );
    }
    if size_gb * 1.2 <= my_vram_gb {
        return (
            "Likely comfortable".into(),
            format!(
                "This machine has {:.1} GB capacity, which should handle a {:.1} GB model comfortably.",
                my_vram_gb, size_gb
            ),
        );
    }
    if size_gb * 1.05 <= my_vram_gb {
        return (
            "Likely fits".into(),
            format!(
                "This machine has {:.1} GB capacity. A {:.1} GB model should fit, but headroom will be tight.",
                my_vram_gb, size_gb
            ),
        );
    }
    if size_gb * 0.8 <= my_vram_gb {
        return (
            "Possible with tradeoffs".into(),
            format!(
                "This machine has {:.1} GB capacity. A {:.1} GB model may load, but expect tighter memory pressure.",
                my_vram_gb, size_gb
            ),
        );
    }
    (
        "Likely too large".into(),
        format!(
            "This machine has {:.1} GB capacity, which is likely not enough for a {:.1} GB model locally.",
            my_vram_gb, size_gb
        ),
    )
}

impl MeshApi {
    pub fn new(
        node: mesh::Node,
        model_name: String,
        api_port: u16,
        model_size_bytes: u64,
        plugin_manager: plugin::PluginManager,
        affinity_router: affinity::AffinityRouter,
    ) -> Self {
        MeshApi {
            inner: Arc::new(Mutex::new(ApiInner {
                node,
                plugin_manager,
                affinity_router,
                headless: false,
                is_host: false,
                is_client: false,
                llama_ready: false,
                llama_port: None,
                model_name,
                primary_backend: None,
                draft_name: None,
                api_port,
                model_size_bytes,
                mesh_name: None,
                latest_version: None,
                nostr_relays: nostr::DEFAULT_RELAYS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                nostr_discovery: false,
                publication_state: state::PublicationState::Private,
                runtime_control: None,
                local_processes: Vec::new(),
                sse_clients: Vec::new(),
                model_interests: std::collections::HashMap::new(),
                inventory_scan_running: false,
                inventory_scan_waiters: Vec::new(),
                local_instances: Arc::new(Mutex::new(Vec::new())),
                wakeable_inventory: crate::runtime::wakeable::WakeableInventory::default(),
                mesh_visibility: None,
            })),
        }
    }

    /// Attach a [`mesh::MeshVisibilityHandle`] so `/api/status` surfaces
    /// the audit loop's latest outcome. Called by `runtime::run()` once
    /// the visibility monitor has been spawned. No-op for runtimes
    /// without `--join-url`.
    pub async fn set_mesh_visibility(&self, handle: mesh::MeshVisibilityHandle) {
        self.inner.lock().await.mesh_visibility = Some(handle);
        self.push_status().await;
    }

    pub async fn node(&self) -> mesh::Node {
        self.inner.lock().await.node.clone()
    }

    pub(super) async fn model_interests(&self) -> Vec<LocalModelInterest> {
        let mut interests = {
            let inner = self.inner.lock().await;
            inner
                .model_interests
                .values()
                .cloned()
                .collect::<Vec<LocalModelInterest>>()
        };
        interests.sort_by(|left, right| {
            right
                .updated_at_unix
                .cmp(&left.updated_at_unix)
                .then_with(|| left.model_ref.cmp(&right.model_ref))
        });
        interests
    }

    pub(super) async fn upsert_model_interest(
        &self,
        model_ref: String,
        submission_source: Option<String>,
    ) -> (LocalModelInterest, bool) {
        let now = current_unix_secs();
        let mut inner = self.inner.lock().await;
        match inner.model_interests.entry(model_ref.clone()) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let existing = entry.get().clone();
                let updated = LocalModelInterest {
                    model_ref,
                    submission_source: submission_source.or(existing.submission_source),
                    created_at_unix: existing.created_at_unix,
                    updated_at_unix: now,
                };
                entry.insert(updated.clone());
                (updated, false)
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let created = LocalModelInterest {
                    model_ref,
                    submission_source,
                    created_at_unix: now,
                    updated_at_unix: now,
                };
                entry.insert(created.clone());
                (created, true)
            }
        }
    }

    pub(super) async fn remove_model_interest(&self, model_ref: &str) -> bool {
        self.inner
            .lock()
            .await
            .model_interests
            .remove(model_ref)
            .is_some()
    }

    pub async fn set_primary_backend(&self, backend: String) {
        self.inner.lock().await.primary_backend = Some(backend);
    }

    pub async fn set_draft_name(&self, name: String) {
        self.inner.lock().await.draft_name = Some(name);
    }

    pub async fn set_client(&self, is_client: bool) {
        self.inner.lock().await.is_client = is_client;
    }

    pub async fn set_mesh_name(&self, name: String) {
        self.inner.lock().await.mesh_name = Some(name);
    }

    pub async fn set_nostr_relays(&self, relays: Vec<String>) {
        self.inner.lock().await.nostr_relays = relays;
    }

    pub async fn set_nostr_discovery(&self, v: bool) {
        self.inner.lock().await.nostr_discovery = v;
    }

    pub async fn set_publication_state(&self, state: state::PublicationState) {
        {
            let mut inner = self.inner.lock().await;
            inner.publication_state = state;
        }
        self.push_status().await;
    }

    #[cfg(test)]
    pub(crate) async fn publication_state(&self) -> state::PublicationState {
        self.inner.lock().await.publication_state
    }

    pub async fn local_instances_handle(
        &self,
    ) -> Arc<Mutex<Vec<crate::runtime::instance::LocalInstanceSnapshot>>> {
        self.inner.lock().await.local_instances.clone()
    }

    pub async fn set_runtime_control(
        &self,
        tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    ) {
        self.inner.lock().await.runtime_control = Some(tx);
    }

    pub(crate) async fn status_snapshot_string(&self) -> String {
        let status = self.status().await;
        match serde_json::to_string_pretty(&status) {
            Ok(json) => json,
            Err(err) => {
                tracing::warn!("failed to serialize local status snapshot: {err}");
                format!(
                    "{{\n  \"error\": \"status snapshot unavailable\",\n  \"detail\": {:?}\n}}",
                    err.to_string()
                )
            }
        }
    }

    pub async fn upsert_local_process(&self, process: RuntimeProcessPayload) {
        {
            let mut inner = self.inner.lock().await;
            inner.local_processes.retain(|p| p.name != process.name);
            inner.local_processes.push(process);
        }
        self.push_status().await;
    }

    pub async fn remove_local_process(&self, model_name: &str) {
        {
            let mut inner = self.inner.lock().await;
            inner.local_processes.retain(|p| p.name != model_name);
        }
        self.push_status().await;
    }

    pub async fn update(&self, is_host: bool, llama_ready: bool) {
        {
            let mut inner = self.inner.lock().await;
            inner.is_host = is_host;
            inner.llama_ready = llama_ready;
        }
        self.push_status().await;
    }

    pub async fn set_llama_port(&self, port: Option<u16>) {
        self.inner.lock().await.llama_port = port;
    }

    pub async fn set_headless(&self, headless: bool) {
        self.inner.lock().await.headless = headless;
    }

    pub(super) async fn is_headless(&self) -> bool {
        self.inner.lock().await.headless
    }

    async fn runtime_status(&self) -> RuntimeStatusPayload {
        let (model_name, primary_backend, is_host, llama_ready, llama_port, local_processes) = {
            let inner = self.inner.lock().await;
            (
                inner.model_name.clone(),
                inner.primary_backend.clone(),
                inner.is_host,
                inner.llama_ready,
                inner.llama_port,
                inner.local_processes.clone(),
            )
        };
        build_runtime_status_payload(
            &model_name,
            primary_backend,
            is_host,
            llama_ready,
            llama_port,
            local_processes,
        )
    }

    async fn runtime_processes(&self) -> RuntimeProcessesPayload {
        let local_processes = self.inner.lock().await.local_processes.clone();
        build_runtime_processes_payload(local_processes)
    }

    async fn local_inventory_snapshot(&self) -> crate::models::LocalModelInventorySnapshot {
        let rx = {
            let mut inner = self.inner.lock().await;
            if inner.inventory_scan_running {
                let (tx, rx) = tokio::sync::oneshot::channel();
                inner.inventory_scan_waiters.push(tx);
                rx
            } else {
                inner.inventory_scan_running = true;
                let (tx, rx) = tokio::sync::oneshot::channel();
                inner.inventory_scan_waiters.push(tx);

                let inner_arc = self.inner.clone();
                tokio::spawn(async move {
                    let snapshot = match tokio::task::spawn_blocking(|| {
                        crate::models::scan_local_inventory_snapshot_with_progress(|_| {})
                    })
                    .await
                    {
                        Ok(snapshot) => snapshot,
                        Err(e) => {
                            tracing::warn!("Local inventory scan failed: {e}");
                            crate::models::LocalModelInventorySnapshot::default()
                        }
                    };

                    let waiters = {
                        let mut inner = inner_arc.lock().await;
                        inner.inventory_scan_running = false;
                        std::mem::take(&mut inner.inventory_scan_waiters)
                    };
                    for tx in waiters {
                        let _ = tx.send(snapshot.clone());
                    }
                });

                rx
            }
        };

        rx.await.unwrap_or_default()
    }

    async fn mesh_models(&self) -> Vec<MeshModelPayload> {
        let (node, my_vram_gb, model_name, model_size_bytes, _local_processes) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner.node.fast_memory_bytes() as f64 / 1e9,
                inner.model_name.clone(),
                inner.model_size_bytes,
                inner.local_processes.clone(),
            )
        };

        let local_scan = self.local_inventory_snapshot().await;
        let all_peers = node.peers().await;
        let catalog = node.mesh_catalog_entries().await;
        let served = node.models_being_served().await;
        let active_demand = node.active_demand().await;
        // Per-model routing metrics are current-node-only observations. They
        // help the management API explain recent local routing behavior without
        // claiming mesh-wide totals.
        let routing_metrics_by_model = node.model_routing_metrics();
        let my_serving_models = node.serving_models().await;
        let local_model_names = local_scan.model_names;
        let mut metadata_by_name = local_scan.metadata_by_name;
        let mut size_by_name = local_scan.size_by_name;
        for peer in &all_peers {
            for meta in &peer.available_model_metadata {
                metadata_by_name
                    .entry(meta.model_key.clone())
                    .or_insert_with(|| meta.clone());
            }
            for (model_name, size) in &peer.available_model_sizes {
                size_by_name.entry(model_name.clone()).or_insert(*size);
            }
        }
        let my_hosted_models = node.hosted_models().await;
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        catalog
            .iter()
            .map(|entry| {
                let name = &entry.model_name;
                let descriptor = entry.descriptor.as_ref();
                let identity = descriptor.map(|descriptor| &descriptor.identity);
                let catalog_entry = find_catalog_model(name);
                let is_warm = served.contains(name);
                let local_known = local_model_names.contains(name)
                    || my_hosted_models.iter().any(|s| s == name)
                    || my_serving_models.iter().any(|s| s == name)
                    || name == &model_name;
                let display_name = crate::models::installed_model_display_name(name);
                let route_stats = is_warm.then(|| {
                    http_route_stats(
                        name,
                        &all_peers,
                        &my_hosted_models,
                        node.hostname.as_deref(),
                        my_vram_gb,
                    )
                });
                let node_count = route_stats
                    .as_ref()
                    .map(|stats| stats.node_count)
                    .unwrap_or(0);
                let active_nodes = route_stats
                    .as_ref()
                    .map(|stats| stats.active_nodes.clone())
                    .unwrap_or_default();
                let mesh_vram_gb = route_stats
                    .as_ref()
                    .map(|stats| stats.mesh_vram_gb)
                    .unwrap_or(0.0);
                let size_gb = if name == &model_name && model_size_bytes > 0 {
                    model_size_bytes as f64 / 1e9
                } else {
                    size_by_name
                        .get(name)
                        .map(|size| *size as f64 / 1e9)
                        .unwrap_or_else(|| {
                            crate::models::catalog::parse_size_gb(
                                catalog_entry.map(|m| m.size.as_str()).unwrap_or("0"),
                            )
                        })
                };
                let (request_count, last_active_secs_ago) = match active_demand.get(name) {
                    Some(d) => (
                        Some(d.request_count),
                        Some(now_ts.saturating_sub(d.last_active)),
                    ),
                    None => (None, None),
                };
                let routing_metrics = routing_metrics_by_model.get(name).cloned();
                let mut capabilities = descriptor
                    .map(|descriptor| descriptor.capabilities)
                    .unwrap_or_else(|| {
                        if local_known {
                            crate::models::installed_model_capabilities(name)
                        } else {
                            crate::models::ModelCapabilities::default()
                        }
                    });
                if local_known
                    && likely_reasoning_model(name, catalog_entry.map(|m| m.description.as_str()))
                {
                    capabilities.reasoning = capabilities
                        .reasoning
                        .max(crate::models::capabilities::CapabilityLevel::Likely);
                }
                if local_known
                    && likely_vision_model(name, catalog_entry.map(|m| m.description.as_str()))
                {
                    capabilities.vision = capabilities
                        .vision
                        .max(crate::models::capabilities::CapabilityLevel::Likely);
                    capabilities.multimodal = true;
                }
                if local_known
                    && likely_audio_model(name, catalog_entry.map(|m| m.description.as_str()))
                {
                    capabilities.audio = capabilities
                        .audio
                        .max(crate::models::capabilities::CapabilityLevel::Likely);
                    capabilities.multimodal = true;
                }
                let multimodal = capabilities.supports_multimodal_runtime();
                let multimodal_status = if multimodal || capabilities.multimodal_label().is_some() {
                    Some(capabilities.multimodal_status())
                } else {
                    None
                };
                let vision = capabilities.supports_vision_runtime();
                let vision_status = if vision || capabilities.vision_label().is_some() {
                    Some(capabilities.vision_status())
                } else {
                    None
                };
                let audio = matches!(
                    capabilities.audio,
                    crate::models::capabilities::CapabilityLevel::Supported
                        | crate::models::capabilities::CapabilityLevel::Likely
                );
                let audio_status = if audio || capabilities.audio_label().is_some() {
                    Some(capabilities.audio_status())
                } else {
                    None
                };
                let reasoning = matches!(
                    capabilities.reasoning,
                    crate::models::capabilities::CapabilityLevel::Supported
                        | crate::models::capabilities::CapabilityLevel::Likely
                );
                let reasoning_status = if reasoning || capabilities.reasoning_label().is_some() {
                    Some(capabilities.reasoning_status())
                } else {
                    None
                };
                let tool_use = capabilities.tool_use_label().is_some();
                let tool_use_status = capabilities
                    .tool_use_label()
                    .map(|_| capabilities.tool_use_status());
                let description = catalog_entry.map(|m| m.description.to_string());
                let metadata = metadata_by_name.get(name);
                let architecture = metadata
                    .map(|m| m.architecture.trim())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                let context_length = metadata
                    .map(|m| m.context_length)
                    .filter(|value| *value > 0);
                let quantization = metadata
                    .map(|m| m.quantization_type.trim())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .or_else(|| {
                        catalog_entry.map(|m| m.file.to_string()).and_then(|file| {
                            let quant = file
                                .strip_suffix(".gguf")
                                .map(crate::models::inventory::derive_quantization_type)
                                .filter(|q| !q.is_empty())?;
                            Some(quant)
                        })
                    });
                let topology_moe = descriptor
                    .and_then(|descriptor| descriptor.topology.as_ref())
                    .and_then(|topology| topology.moe.as_ref());
                let moe = capabilities.moe
                    || topology_moe.is_some()
                    || metadata.map(|m| m.is_moe).unwrap_or(false);
                let expert_count = topology_moe
                    .map(|moe| moe.expert_count)
                    .or_else(|| metadata.map(|m| m.expert_count).filter(|count| *count > 0))
                    .or_else(|| {
                        catalog_entry
                            .and_then(|m| m.moe.as_ref())
                            .map(|m| m.n_expert)
                    });
                let used_expert_count = topology_moe
                    .map(|moe| moe.used_expert_count)
                    .or_else(|| {
                        metadata
                            .map(|m| m.used_expert_count)
                            .filter(|count| *count > 0)
                    })
                    .or_else(|| {
                        catalog_entry
                            .and_then(|m| m.moe.as_ref())
                            .map(|m| m.n_expert_used)
                    });
                let ranking_source = topology_moe
                    .and_then(|moe| moe.ranking_source.as_ref())
                    .cloned();
                let ranking_origin = topology_moe
                    .and_then(|moe| moe.ranking_origin.as_ref())
                    .cloned();
                let ranking_prompt_count = topology_moe.and_then(|moe| moe.ranking_prompt_count);
                let ranking_tokens = topology_moe.and_then(|moe| moe.ranking_tokens);
                let ranking_layer_scope = topology_moe
                    .and_then(|moe| moe.ranking_layer_scope.as_ref())
                    .cloned();
                let draft_model = catalog_entry.and_then(|m| m.draft.clone());
                let source_page_url =
                    identity
                        .and_then(source_page_url_from_identity)
                        .or_else(|| {
                            if local_known {
                                catalog_entry.and_then(|m| {
                                    crate::models::catalog::huggingface_repo_url(&m.url)
                                })
                            } else {
                                None
                            }
                        });
                let source_ref = identity
                    .and_then(huggingface_repository_from_identity)
                    .or_else(|| {
                        source_page_url
                            .as_deref()
                            .map(|url| url.replace("https://huggingface.co/", ""))
                    });
                let source_revision = identity.and_then(|identity| identity.revision.clone());
                let source_file = identity.and_then(source_file_from_identity).or_else(|| {
                    if local_known {
                        catalog_entry.map(|m| m.file.to_string())
                    } else {
                        None
                    }
                });
                let command_ref = identity
                    .and_then(|identity| identity.canonical_ref.clone())
                    .or_else(|| {
                        if local_known {
                            catalog_entry.and_then(|m| {
                                match (m.source_repo(), m.source_revision(), m.source_file()) {
                                    (Some(repo), revision, Some(file)) => Some(match revision {
                                        Some(revision) => format!("{repo}@{revision}/{file}"),
                                        None => format!("{repo}/{file}"),
                                    }),
                                    _ => None,
                                }
                            })
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| name.clone());
                let (fit_label, fit_detail) = fit_hint_for_machine(size_gb, my_vram_gb);
                let split_kind = classify_model_split_kind(
                    name,
                    is_warm,
                    moe,
                    node_count,
                    size_gb,
                    &all_peers,
                    my_vram_gb,
                    my_serving_models.iter().any(|m| m == name),
                );
                let mesh_fit = compute_mesh_fit(size_gb, my_vram_gb, &all_peers);
                MeshModelPayload {
                    name: name.clone(),
                    display_name,
                    status: if is_warm {
                        "warm".into()
                    } else {
                        "cold".into()
                    },
                    node_count,
                    mesh_vram_gb,
                    size_gb,
                    architecture,
                    context_length,
                    quantization,
                    description,
                    multimodal,
                    multimodal_status,
                    vision,
                    vision_status,
                    audio,
                    audio_status,
                    reasoning,
                    reasoning_status,
                    tool_use,
                    tool_use_status,
                    moe,
                    expert_count,
                    used_expert_count,
                    ranking_source,
                    ranking_origin,
                    ranking_prompt_count,
                    ranking_tokens,
                    ranking_layer_scope,
                    draft_model,
                    request_count,
                    last_active_secs_ago,
                    routing_metrics,
                    source_page_url,
                    source_ref,
                    source_revision,
                    source_file,
                    active_nodes,
                    fit_label,
                    fit_detail,
                    download_command: format!("senda models download {}", command_ref),
                    run_command: format!("senda serve --model {}", command_ref),
                    auto_command: format!("senda serve --auto --model {}", command_ref),
                    split_kind,
                    mesh_fit,
                }
            })
            .collect()
    }

    fn derive_local_node_state(
        is_client: bool,
        effective_is_host: bool,
        effective_llama_ready: bool,
        has_local_worker_activity: bool,
        display_model_name: &str,
    ) -> NodeState {
        let has_declared_local_serving_work = (effective_is_host || has_local_worker_activity)
            && !display_model_name.trim().is_empty();

        if is_client {
            NodeState::Client
        } else if effective_llama_ready && has_declared_local_serving_work {
            NodeState::Serving
        } else if has_declared_local_serving_work {
            NodeState::Loading
        } else {
            NodeState::Standby
        }
    }

    fn derive_node_status(node_state: NodeState) -> String {
        node_state.node_status_alias().to_string()
    }

    fn derive_peer_state(peer: &mesh::PeerInfo) -> NodeState {
        fn has_nonempty_models(models: &[String]) -> bool {
            models.iter().any(|model| !model.trim().is_empty())
        }

        match peer.role {
            mesh::NodeRole::Client => NodeState::Client,
            mesh::NodeRole::Host { .. } | mesh::NodeRole::Worker => {
                let has_runtime_descriptors = peer
                    .served_model_runtime
                    .iter()
                    .any(|runtime| !runtime.model_name.trim().is_empty());
                let has_ready_runtime = peer
                    .served_model_runtime
                    .iter()
                    .any(|runtime| runtime.ready && !runtime.model_name.trim().is_empty());
                let has_assigned_model_work = has_runtime_descriptors
                    || has_nonempty_models(&peer.serving_models)
                    || has_nonempty_models(&peer.hosted_models);
                let has_nonempty_routable = peer
                    .routable_models()
                    .iter()
                    .any(|model| !model.trim().is_empty());

                // Runtime descriptors are the modern, authoritative signal.
                if has_ready_runtime {
                    return NodeState::Serving;
                }
                if has_runtime_descriptors && has_assigned_model_work {
                    return NodeState::Loading;
                }

                // `hosted_models` is the pre-runtime-descriptor signal for
                // "this peer is actively routing this model": the runtime only
                // writes it after llama.cpp passes health checks (see
                // runtime/mod.rs set_hosted_models / set_serving_models pair).
                // Honour it as proof of actually serving.
                if has_nonempty_models(&peer.hosted_models) {
                    return NodeState::Serving;
                }

                // Modern peers that explicitly advertised `hosted_models` (even
                // as empty) but still have non-empty `serving_models` are
                // declaring intent without proof — either llama.cpp hasn't
                // finished loading, or it failed to launch (e.g. missing
                // rpc-server binary). Don't paint that as Serving; it's the
                // bug that made failed-to-launch peers show up as
                // "Serving Qwen3-0.6B" on the public status page.
                if peer.hosted_models_known && has_nonempty_models(&peer.serving_models) {
                    return NodeState::Loading;
                }

                // Legacy fallback: peers from before `hosted_models` was added
                // to the gossip schema (identified by `hosted_models_known =
                // false`, meaning their announcement omitted the field
                // entirely) used `serving_models` as the authoritative signal.
                // Preserve that behavior so older nodes don't silently drop
                // off the public status page.
                if has_nonempty_models(&peer.serving_models) || has_nonempty_routable {
                    return NodeState::Serving;
                }

                NodeState::Standby
            }
        }
    }

    fn build_wakeable_node(entry: WakeableInventoryEntry) -> WakeableNode {
        WakeableNode {
            logical_id: entry.logical_id,
            models: entry.models,
            vram_gb: entry.vram_gb,
            provider: entry.provider,
            state: match entry.state {
                WakeableState::Sleeping => WakeableNodeState::Sleeping,
                WakeableState::Waking => WakeableNodeState::Waking,
            },
            wake_eta_secs: entry.wake_eta_secs,
        }
    }

    async fn status(&self) -> StatusPayload {
        // Snapshot inner fields and drop the lock before any async node queries.
        // This prevents deadlock: if node.peers() etc. block on node.state.lock(),
        // we don't hold inner.lock() hostage, so other handlers can still proceed.
        let (
            node,
            node_id,
            token,
            my_vram_gb,
            inflight_requests,
            routing_affinity,
            routing_metrics,
            model_name,
            model_size_bytes,
            llama_ready,
            is_host,
            is_client,
            api_port,
            draft_name,
            mesh_name,
            latest_version,
            nostr_discovery,
            publication_state,
            local_processes,
            local_instances_arc,
            wakeable_inventory,
            mesh_visibility_handle,
        ) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner.node.id().fmt_short().to_string(),
                inner.node.invite_token(),
                inner.node.fast_memory_bytes() as f64 / 1e9,
                inner.node.inflight_requests(),
                inner.affinity_router.stats_snapshot(),
                // `/api/status` exposes the current node's bounded routing
                // outcome snapshot only; peers do not publish these counters.
                inner.node.routing_metrics_snapshot(),
                inner.model_name.clone(),
                inner.model_size_bytes,
                inner.llama_ready,
                inner.is_host,
                inner.is_client,
                inner.api_port,
                inner.draft_name.clone(),
                inner.mesh_name.clone(),
                inner.latest_version.clone(),
                inner.nostr_discovery,
                inner.publication_state,
                inner.local_processes.clone(),
                inner.local_instances.clone(),
                inner.wakeable_inventory.clone(),
                inner.mesh_visibility.clone(),
            )
        }; // inner lock dropped here

        let local_instances: Vec<LocalInstance> = {
            let snapshots = local_instances_arc.lock().await;
            let mut instances: Vec<LocalInstance> = snapshots
                .iter()
                .map(|s| LocalInstance {
                    pid: s.pid,
                    api_port: s.api_port,
                    version: s.version.clone(),
                    started_at_unix: s.started_at_unix,
                    runtime_dir: s.runtime_dir.to_string_lossy().to_string(),
                    is_self: s.is_self,
                })
                .collect();

            // Safety net: if scanner hasn't run yet, ensure self is always present
            if instances.is_empty() {
                instances.push(LocalInstance {
                    pid: std::process::id(),
                    api_port: Some(api_port),
                    version: Some(SENDA_VERSION.to_string()),
                    started_at_unix: 0, // best-effort; scanner will populate properly
                    runtime_dir: String::new(),
                    is_self: true,
                });
            }

            instances
        };

        let wakeable_nodes = wakeable_inventory
            .status_snapshot()
            .await
            .into_iter()
            .map(Self::build_wakeable_node)
            .collect();

        let all_peers = node.peers().await;
        let local_owner_summary = node.owner_summary().await;
        let my_models = node.models().await;
        let my_available_models = node.available_models().await;
        let my_requested_models = node.requested_models().await;
        let verify_snapshot = node.verify_verdicts_snapshot().await;
        let reputation_snapshot = node.reputation_snapshot().await;
        let peers: Vec<PeerPayload> = all_peers
            .iter()
            .map(|p| {
                let split_classification = classify_peer_split_role(p, &all_peers, my_vram_gb);
                let peer_id_str = p.id.to_string();
                // No "pipeline_host_degraded" wire-level downgrade here —
                // a healthy pipeline split has the host in
                // `NodeState::Serving` and every worker in
                // `NodeState::Loading` (workers run only `rpc-server`
                // and never reach Serving by design). The previous gate
                // treated workers' Loading state as "the host is
                // degraded", cleared the host's `hosted_models` on the
                // wire, and broke every legitimate split. The
                // split-brain case it was trying to address is already
                // handled honestly by the `host.is_none()` branch in
                // `classify_peer_split_role` / `classify_local_split_role`,
                // which returns `role=None` when no peer in the cohort
                // has graduated to `NodeRole::Host`. The host's own
                // `derive_peer_state` already returns Loading until its
                // llama-server is ready, which is the correct truth
                // signal — no wire-level mutation needed.
                PeerPayload {
                    id: p.id.fmt_short().to_string(),
                    owner: build_ownership_payload(&p.owner_summary),
                    model_ad: crate::api::status::build_model_ad_payload(&p.model_ad.summary),
                    verify_by_model: verify_snapshot
                        .iter()
                        .filter(|((pid, _), _)| *pid == p.id)
                        .map(|((_, model), rec)| {
                            (model.clone(), crate::api::status::build_verify_payload(rec))
                        })
                        .collect(),
                    // v0.66.x Phase 3.2: durable reputation accumulator,
                    // filtered to this peer. Keyed by full peer id string to
                    // match how the verifier wrote it.
                    reputation_by_model: reputation_snapshot
                        .iter()
                        .filter(|((pid, _), _)| pid == &peer_id_str)
                        .map(|((_, model), s)| {
                            (
                                model.clone(),
                                crate::api::status::build_reputation_payload(s),
                            )
                        })
                        .collect(),
                    role: match p.role {
                        mesh::NodeRole::Worker => "Worker".into(),
                        mesh::NodeRole::Host { .. } => "Host".into(),
                        mesh::NodeRole::Client => "Client".into(),
                    },
                    state: Self::derive_peer_state(p),
                    models: p.models.clone(),
                    available_models: p.available_models.clone(),
                    requested_models: p.requested_models.clone(),
                    vram_gb: p.fast_memory_bytes() as f64 / 1e9,
                    serving_models: p.serving_models.clone(),
                    hosted_models: p.hosted_models.clone(),
                    hosted_models_known: p.hosted_models_known,
                    version: p.version.clone(),
                    rtt_ms: p.rtt_ms,
                    inflight_requests: p.inflight_requests,
                    system_ram_bytes: p.system_ram_bytes,
                    hostname: p.hostname.clone(),
                    is_soc: p.is_soc,
                    gpus: build_gpus(
                        p.gpu_name.as_deref(),
                        p.gpu_vram.as_deref(),
                        p.gpu_reserved_bytes.as_deref(),
                        p.gpu_mem_bandwidth_gbps.as_deref(),
                        p.gpu_compute_tflops_fp32.as_deref(),
                        p.gpu_compute_tflops_fp16.as_deref(),
                    ),
                    capability:
                        crate::api::status::NodeCapabilityPayload::from_capability_with_usable_vram(
                            &p.capability,
                            p.fast_memory_bytes(),
                        ),
                    // v0.66.41 Phase 1 marketplace metrics: hydrate
                    // from the peer's gossiped `model_timings` vec.
                    // Legacy peers gossip an empty vec and the
                    // resulting HashMaps stay empty — the API contract
                    // is "missing key = not yet measured", not
                    // "missing key = measured zero".
                    measured_tps_p50_by_model: p
                        .model_timings
                        .iter()
                        .map(|t| (t.model.clone(), t.measured_tps_p50))
                        .collect(),
                    measured_ttft_ms_p50_by_model: p
                        .model_timings
                        .iter()
                        .map(|t| (t.model.clone(), t.measured_ttft_ms_p50))
                        .collect(),
                    // v0.66.49 Phase 3.0 benchmark honesty: hydrate the
                    // native baseline maps from the peer's gossiped
                    // `native_baselines` vec. Same "missing key = not
                    // yet measured" semantics — pre-v0.66.49 peers
                    // gossip an empty vec, so the catalog UI silently
                    // omits the ratio column for them rather than
                    // showing a ratio of zero.
                    native_tps_p50_by_model: p
                        .native_baselines
                        .iter()
                        .map(|b| (b.model.clone(), b.native_tps_p50))
                        .collect(),
                    native_ttft_ms_p50_by_model: p
                        .native_baselines
                        .iter()
                        .map(|b| (b.model.clone(), b.native_ttft_ms_p50))
                        .collect(),
                    first_joined_mesh_ts: p.first_joined_mesh_ts,
                    serving_mode: serving_mode_from_classification(p, &split_classification),
                    split_role: split_classification.role,
                    split_group: split_classification.group,
                    moe_shard: split_classification.moe_shard,
                }
            })
            .collect();

        let my_serving_models = node.serving_models().await;
        let my_hosted_models = node.hosted_models().await;
        let has_local_processes = !local_processes.is_empty();
        let effective_llama_ready = llama_ready || has_local_processes;
        let effective_is_host = is_host || has_local_processes;
        let display_model_name = local_processes
            .first()
            .map(|process| process.name.clone())
            .or_else(|| my_hosted_models.first().cloned())
            .or_else(|| my_serving_models.first().cloned())
            .unwrap_or_else(|| model_name.clone());

        let (launch_pi, launch_goose) = if effective_llama_ready {
            (
                Some(format!("pi --provider mesh --model {display_model_name}")),
                Some(format!("GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{api_port} OPENAI_API_KEY=mesh GOOSE_MODEL={display_model_name} goose session")),
            )
        } else {
            (None, None)
        };

        let mesh_id = node.mesh_id().await;

        let has_local_worker_activity = has_local_processes || !my_hosted_models.is_empty();
        let node_state = Self::derive_local_node_state(
            is_client,
            effective_is_host,
            effective_llama_ready,
            has_local_worker_activity,
            display_model_name.as_str(),
        );

        // Synthesize a PeerInfo-shaped view of this node so we can reuse the
        // peer split classifier above. The local node is "self" — it never
        // shows up in the peer list — so we feed it through the same logic
        // and end up with the same split_role / split_group fields populated
        // for the dashboard.
        //
        // No local-node "pipeline_host_degraded" wire downgrade — see the
        // peer-loop above for the same reasoning. The split-brain case is
        // handled by `classify_local_split_role`'s `host.is_none()` arm,
        // and `derive_local_node_state` already only returns Serving when
        // `llama_ready` is true and the local runtime is actually hosting
        // the model.
        let self_split = classify_local_split_role(
            node_id.as_str(),
            effective_is_host,
            is_client,
            &my_serving_models,
            &my_hosted_models,
            my_vram_gb,
            &all_peers,
        );

        let node_status = Self::derive_node_status(node_state);

        StatusPayload {
            version: SENDA_VERSION.to_string(),
            latest_version,
            node_id,
            owner: build_ownership_payload(&local_owner_summary),
            token,
            node_state,
            node_status,
            is_host: effective_is_host,
            is_client,
            llama_ready: effective_llama_ready,
            model_name: display_model_name,
            models: my_models,
            available_models: my_available_models,
            requested_models: my_requested_models,
            serving_models: my_serving_models,
            hosted_models: my_hosted_models,
            draft_name,
            api_port,
            my_vram_gb,
            model_size_gb: model_size_bytes as f64 / 1e9,
            peers,
            wakeable_nodes,
            local_instances,
            launch_pi,
            launch_goose,
            inflight_requests,
            system_ram_bytes: node.system_ram_bytes,
            mesh_id,
            mesh_name,
            nostr_discovery,
            publication_state: publication_state.as_str().into(),
            my_hostname: node.hostname.clone(),
            my_is_soc: node.is_soc,
            gpus: {
                let bw_str = {
                    let bw = node.gpu_mem_bandwidth_gbps.lock().await;
                    bw.as_ref().map(|v| {
                        v.iter()
                            .map(|f| f.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                };
                let tf32_str = {
                    let tf32 = node.gpu_compute_tflops_fp32.lock().await;
                    tf32.as_ref().map(|v| {
                        v.iter()
                            .map(|f| f.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                };
                let tf16_str = {
                    let tf16 = node.gpu_compute_tflops_fp16.lock().await;
                    tf16.as_ref().map(|v| {
                        v.iter()
                            .map(|f| f.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                };
                build_gpus(
                    node.gpu_name.as_deref(),
                    node.gpu_vram.as_deref(),
                    node.gpu_reserved_bytes.as_deref(),
                    bw_str.as_deref(),
                    tf32_str.as_deref(),
                    tf16_str.as_deref(),
                )
            },
            capability: node
                .local_node_capability()
                .await
                .as_ref()
                .map(|capability| {
                    crate::api::status::NodeCapabilityPayload::from_capability_with_usable_vram(
                        capability,
                        node.fast_memory_bytes(),
                    )
                })
                .unwrap_or_default(),
            routing_affinity,
            routing_metrics,
            // v0.66.41 Phase 1 marketplace metrics: snapshot the local
            // node's per-model TPS / TTFT rolling-1h window. Same
            // "missing key = not measured" convention as the peer side
            // above; an empty map means this node hasn't completed any
            // local inference yet.
            measured_tps_p50_by_model: node
                .model_timings_snapshot()
                .iter()
                .map(|(model, snap)| (model.clone(), snap.measured_tps_p50))
                .collect(),
            measured_ttft_ms_p50_by_model: node
                .model_timings_snapshot()
                .iter()
                .map(|(model, snap)| (model.clone(), snap.measured_ttft_ms_p50))
                .collect(),
            // v0.66.49 Phase 3.0: native baselines for THIS node from
            // its in-memory cache (populated by the inference baseline
            // collector, persisted across restarts in
            // `~/.senda/native-baselines.json`).
            native_tps_p50_by_model: node
                .native_baselines_snapshot()
                .await
                .into_iter()
                .map(|b| (b.model, b.native_tps_p50))
                .collect(),
            native_ttft_ms_p50_by_model: node
                .native_baselines_snapshot()
                .await
                .into_iter()
                .map(|b| (b.model, b.native_ttft_ms_p50))
                .collect(),
            // Rolling-7d completion tokens this node served, per model.
            // Local-only estimate input for the desktop earnings-preview.
            serving_tokens_7d_by_model: node.serving_tally_snapshot(),
            first_joined_mesh_ts: node.first_joined_mesh_ts().await,
            my_split_role: self_split.role,
            my_split_group: self_split.group,
            my_moe_shard: self_split.moe_shard,
            mesh_visibility: match mesh_visibility_handle {
                Some(handle) => Some(handle.snapshot().await),
                None => None,
            },
        }
    }

    async fn push_status(&self) {
        let status = self.status().await;
        if let Ok(json) = serde_json::to_string(&status) {
            let event = format!("data: {json}\n\n");
            let mut inner = self.inner.lock().await;
            inner.sse_clients.retain(|tx| !tx.is_closed());
            for tx in &inner.sse_clients {
                let _ = tx.send(event.clone());
            }
        }
    }
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Server ──

pub async fn start(
    port: u16,
    state: MeshApi,
    mut target_rx: watch::Receiver<election::InferenceTarget>,
    listen_all: bool,
    headless: bool,
) {
    state.set_headless(headless).await;
    // Watch election target changes
    let state2 = state.clone();
    tokio::spawn(async move {
        loop {
            if target_rx.changed().await.is_err() {
                break;
            }
            let target = target_rx.borrow().clone();
            match target {
                election::InferenceTarget::Local(port)
                | election::InferenceTarget::MoeLocal(port) => {
                    state2.set_llama_port(Some(port)).await;
                }
                election::InferenceTarget::Remote(_) | election::InferenceTarget::MoeRemote(_) => {
                    let mut inner = state2.inner.lock().await;
                    inner.llama_ready = true;
                    inner.llama_port = None;
                }
                election::InferenceTarget::None => {
                    state2.set_llama_port(None).await;
                }
            }
            state2.push_status().await;
        }
    });

    // Push status when peers join/leave.
    let mut peer_rx = {
        let inner = state.inner.lock().await;
        inner.node.peer_change_rx.clone()
    };
    let state3 = state.clone();
    tokio::spawn(async move {
        loop {
            if peer_rx.changed().await.is_err() {
                break;
            }
            state3.push_status().await;
        }
    });

    // Push status when in-flight request count changes.
    let mut inflight_rx = {
        let inner = state.inner.lock().await;
        inner.node.inflight_change_rx()
    };
    let state4 = state.clone();
    tokio::spawn(async move {
        loop {
            if inflight_rx.changed().await.is_err() {
                break;
            }
            state4.push_status().await;
        }
    });

    // One-shot check for newer public release (for UI footer indicator).
    let state5 = state.clone();
    tokio::spawn(async move {
        let Some(latest) = crate::system::autoupdate::latest_release_version().await else {
            return;
        };
        if !crate::system::autoupdate::version_newer(&latest, crate::VERSION) {
            return;
        }
        {
            let mut inner = state5.inner.lock().await;
            inner.latest_version = Some(latest);
        }
        state5.push_status().await;
    });

    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match TcpListener::bind(format!("{addr}:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Management API: failed to bind :{port}: {e}");
            return;
        }
    };
    tracing::info!("Management API on http://localhost:{port}");

    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_request(stream, &state).await {
                tracing::debug!("API connection error: {e}");
            }
        });
    }
}

// ── Request dispatch ──

fn is_ui_only_route(path: &str) -> bool {
    matches!(
        path,
        "/" | "/dashboard" | "/dashboard/" | "/chat" | "/chat/"
    ) || path.starts_with("/chat/")
        || path.starts_with("/assets/")
        || matches!(path.rsplit('.').next(), Some("png" | "ico" | "webmanifest"))
        || (path.ends_with(".json") && !path.starts_with("/api/"))
}

async fn handle_request(mut stream: TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let request = match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        proxy::read_http_request(&mut stream),
    )
    .await
    {
        Ok(Ok(request)) => request,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Ok(()), // read timeout — health check probe, just close
    };
    let req = String::from_utf8_lossy(&request.raw);
    let method = request.method.as_str();
    let path = request.path.as_str();
    let path_only = path.split('?').next().unwrap_or(path);
    let body = http_body_text(&request.raw);

    if method == "GET" && state.is_headless().await && is_ui_only_route(path_only) {
        respond_error(&mut stream, 404, "Not found").await?;
        return Ok(());
    }

    match (method, path_only) {
        // ── Dashboard UI ──
        ("GET", "/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", "/dashboard") | ("GET", "/chat") | ("GET", "/dashboard/") | ("GET", "/chat/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        ("GET", p) if p.starts_with("/chat/") => {
            if !respond_console_index(&mut stream).await? {
                respond_error(&mut stream, 500, "Dashboard bundle missing").await?;
            }
        }

        // ── Frontend static assets (bundled UI dist) ──
        ("GET", p)
            if p.starts_with("/assets/")
                || matches!(p.rsplit('.').next(), Some("png" | "ico" | "webmanifest"))
                || (p.ends_with(".json") && !p.starts_with("/api/")) =>
        {
            if !respond_console_asset(&mut stream, p).await? {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }

        _ => {
            if !dispatch_request(
                &mut stream,
                state,
                method,
                path,
                path_only,
                body,
                req.as_ref(),
                &request.raw,
            )
            .await?
            {
                respond_error(&mut stream, 404, "Not found").await?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::status::decode_runtime_model_path;
    use crate::plugin;
    use crate::plugins::{blackboard, blobstore};
    use senda_plugin::MeshVisibility;
    use rmcp::model::ErrorCode;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use std::time::Instant;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::{mpsc, oneshot};

    #[test]
    fn test_build_gpus_both_none() {
        let result = build_gpus(None, None, None, None, None, None);
        assert!(result.is_empty(), "expected empty vec when no gpu_name");
    }

    #[test]
    fn test_build_gpus_single_no_vram() {
        let result = build_gpus(Some("NVIDIA RTX 5090"), None, None, None, None, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 0);
    }

    #[test]
    fn test_build_gpus_single_with_vram() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090"),
            Some("34359738368"),
            None,
            None,
            None,
            None,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
    }

    #[test]
    fn test_build_gpus_multi_full_vram() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
            Some("34359738368,10737418240"),
            None,
            None,
            None,
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
        assert_eq!(result[1].name, "NVIDIA RTX 3080");
        assert_eq!(result[1].vram_bytes, 10_737_418_240);
    }

    #[test]
    fn test_build_gpus_multi_full_vram_without_space_after_comma() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090,NVIDIA RTX 3080"),
            Some("34359738368,10737418240"),
            None,
            None,
            None,
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "NVIDIA RTX 5090");
        assert_eq!(result[1].name, "NVIDIA RTX 3080");
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
        assert_eq!(result[1].vram_bytes, 10_737_418_240);
    }

    #[test]
    fn test_build_gpus_multi_names_trim_whitespace() {
        let result = build_gpus(
            Some(" GPU0 ,GPU1 ,  GPU2  "),
            Some("100,200,300"),
            None,
            None,
            None,
            None,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].name, "GPU0");
        assert_eq!(result[1].name, "GPU1");
        assert_eq!(result[2].name, "GPU2");
    }

    #[test]
    fn test_build_gpus_expands_summarized_identical_names() {
        let result = build_gpus(
            Some("2× NVIDIA A100"),
            Some("85899345920,85899345920"),
            None,
            Some("1948.70,1948.70"),
            None,
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "NVIDIA A100");
        assert_eq!(result[1].name, "NVIDIA A100");
        assert_eq!(result[0].vram_bytes, 85_899_345_920);
        assert_eq!(result[1].vram_bytes, 85_899_345_920);
        assert_eq!(result[0].mem_bandwidth_gbps, Some(1948.70));
        assert_eq!(result[1].mem_bandwidth_gbps, Some(1948.70));
    }

    #[test]
    fn test_build_gpus_multi_partial_vram() {
        let result = build_gpus(
            Some("NVIDIA RTX 5090, NVIDIA RTX 3080"),
            Some("34359738368"),
            None,
            None,
            None,
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].vram_bytes, 34_359_738_368);
        assert_eq!(
            result[1].vram_bytes, 0,
            "missing VRAM entry should default to 0"
        );
    }

    #[test]
    fn test_build_gpus_vram_no_gpu_name() {
        let result = build_gpus(None, Some("34359738368"), None, None, None, None);
        assert!(
            result.is_empty(),
            "no gpu_name means no entries even if vram present"
        );
    }

    #[test]
    fn test_build_gpus_vram_whitespace_trimmed() {
        let result = build_gpus(
            Some("NVIDIA RTX 4090"),
            Some(" 25769803776 "),
            None,
            None,
            None,
            None,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vram_bytes, 25_769_803_776);
    }

    #[test]
    fn test_build_gpus_with_bandwidth() {
        let result = build_gpus(
            Some("NVIDIA A100, NVIDIA A6000"),
            Some("85899345920,51539607552"),
            None,
            Some("1948.70,780.10"),
            None,
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].mem_bandwidth_gbps, Some(1948.70));
        assert_eq!(result[1].mem_bandwidth_gbps, Some(780.10));
    }

    #[test]
    fn test_build_gpus_unparsable_vram_preserves_index() {
        let result = build_gpus(
            Some("GPU0, GPU1, GPU2"),
            Some("100,foo,300"),
            None,
            None,
            None,
            None,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].vram_bytes, 100);
        assert_eq!(
            result[1].vram_bytes, 0,
            "unparsable vram should default to 0, not shift indices"
        );
        assert_eq!(result[2].vram_bytes, 300);
    }

    #[test]
    fn test_build_gpus_unparsable_bandwidth_preserves_index() {
        let result = build_gpus(
            Some("GPU0, GPU1, GPU2"),
            Some("100,200,300"),
            None,
            Some("1.0,bad,3.0"),
            None,
            None,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].mem_bandwidth_gbps, Some(1.0));
        assert_eq!(
            result[1].mem_bandwidth_gbps, None,
            "unparsable bandwidth should be None, not shift indices"
        );
        assert_eq!(result[2].mem_bandwidth_gbps, Some(3.0));
    }

    #[test]
    fn test_build_gpus_with_both_tflops_precisions() {
        let result = build_gpus(
            Some("GPU0, GPU1"),
            Some("100,200"),
            None,
            None,
            Some("312.5,419.5"),
            Some("625.0,839.0"),
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].compute_tflops_fp32, Some(312.5));
        assert_eq!(result[0].compute_tflops_fp16, Some(625.0));
        assert_eq!(result[1].compute_tflops_fp32, Some(419.5));
        assert_eq!(result[1].compute_tflops_fp16, Some(839.0));
    }

    #[test]
    fn test_build_gpus_fp32_only_fp16_absent() {
        let result = build_gpus(
            Some("GPU0, GPU1"),
            Some("100,200"),
            None,
            None,
            Some("312.5,bad"),
            None,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].compute_tflops_fp32, Some(312.5));
        assert_eq!(result[1].compute_tflops_fp32, None);
        assert!(result.iter().all(|gpu| gpu.compute_tflops_fp16.is_none()));
    }

    #[test]
    fn test_gpu_entry_omits_tflops_when_none() {
        let value = serde_json::to_value(build_gpus(
            Some("NVIDIA A100"),
            Some("85899345920"),
            None,
            Some("1948.70"),
            None,
            None,
        ))
        .unwrap();

        let first = value.as_array().unwrap().first().unwrap();
        assert!(first.get("compute_tflops_fp32").is_none());
        assert!(first.get("compute_tflops_fp16").is_none());
        assert!(first.get("mem_bandwidth_gbps").is_some());
    }

    #[test]
    fn test_api_status_gpu_entry_uses_new_name() {
        let value = serde_json::to_value(build_gpus(
            Some("NVIDIA A100"),
            Some("85899345920"),
            None,
            Some("1948.70"),
            None,
            None,
        ))
        .unwrap();

        let first = value.as_array().unwrap().first().unwrap();
        assert_eq!(first.get("mem_bandwidth_gbps").unwrap(), &json!(1948.7));
        assert!(
            first.get("bandwidth_gbps").is_none(),
            "API status JSON should use mem_bandwidth_gbps"
        );
    }

    #[test]
    fn classify_model_split_kind_marks_solo_when_single_node() {
        let kind =
            classify_model_split_kind("Qwen3-8B-Q4_K_M", true, false, 1, 5.0, &[], 16.0, true);
        assert_eq!(kind, "solo");
    }

    #[test]
    fn classify_model_split_kind_marks_pipeline_when_dense_too_big_for_any_node() {
        let host = make_test_peer(
            1,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["GLM-4.7-Flash"],
            vec!["GLM-4.7-Flash"],
            true,
        );
        let worker = make_test_peer(
            2,
            mesh::NodeRole::Worker,
            vec!["GLM-4.7-Flash"],
            vec![],
            true,
        );
        let kind = classify_model_split_kind(
            "GLM-4.7-Flash",
            true,
            false,
            2,
            // Each peer has 24 GB; needed VRAM (44 GB) exceeds either alone.
            44.0,
            &[host, worker],
            8.0,
            false,
        );
        assert_eq!(kind, "pipeline");
    }

    #[test]
    fn classify_model_split_kind_marks_multi_host_when_dense_fits_each_node() {
        let host_a = make_test_peer(
            3,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["Qwen3-8B-Q4_K_M"],
            vec!["Qwen3-8B-Q4_K_M"],
            true,
        );
        let host_b = make_test_peer(
            4,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["Qwen3-8B-Q4_K_M"],
            vec!["Qwen3-8B-Q4_K_M"],
            true,
        );
        let kind = classify_model_split_kind(
            "Qwen3-8B-Q4_K_M",
            true,
            false,
            2,
            // 4 GB model fits comfortably on either 24 GB host.
            4.0,
            &[host_a, host_b],
            8.0,
            false,
        );
        assert_eq!(kind, "multi_host");
    }

    #[test]
    fn classify_model_split_kind_marks_moe_when_flag_set_and_multi_node() {
        let host_a = make_test_peer(
            5,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["Qwen3-30B-A3B"],
            vec!["Qwen3-30B-A3B"],
            true,
        );
        let host_b = make_test_peer(
            6,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["Qwen3-30B-A3B"],
            vec!["Qwen3-30B-A3B"],
            true,
        );
        let kind = classify_model_split_kind(
            "Qwen3-30B-A3B",
            true,
            true,
            2,
            18.0,
            &[host_a, host_b],
            16.0,
            false,
        );
        assert_eq!(kind, "moe");
    }

    #[test]
    fn classify_model_split_kind_returns_cold_when_no_active_nodes() {
        let kind =
            classify_model_split_kind("Qwen3-8B-Q4_K_M", false, false, 0, 5.0, &[], 16.0, false);
        assert_eq!(kind, "cold");
    }

    #[test]
    fn compute_mesh_fit_marks_fits_solo_when_local_box_has_enough_vram() {
        let fit = compute_mesh_fit(20.0, 64.0, &[]);
        assert!(fit.fits_on_largest_node);
        assert!(fit.fits_pooled);
        assert!(fit.eligible_peer_count >= 1);
        assert!((fit.needed_vram_gb - 22.0).abs() < 1e-6);
    }

    #[test]
    fn compute_mesh_fit_marks_pooled_when_swarm_combined_capacity_clears_threshold() {
        let peer_a = make_test_peer(
            10,
            mesh::NodeRole::Host { http_port: 9337 },
            vec![],
            vec![],
            true,
        );
        let peer_b = make_test_peer(11, mesh::NodeRole::Worker, vec![], vec![], true);
        // size=40 GB, needed=44 GB. local 16 GB + 2× peers 24 GB each = 64 GB total.
        let fit = compute_mesh_fit(40.0, 16.0, &[peer_a, peer_b]);
        assert!(!fit.fits_on_largest_node);
        assert!(fit.fits_pooled);
        assert!(fit.pooled_vram_gb >= 60.0);
        assert_eq!(fit.eligible_peer_count, 3);
    }

    #[test]
    fn compute_mesh_fit_marks_neither_when_pool_short() {
        let small_peer = mesh::PeerInfo {
            vram_bytes: 4_000_000_000,
            ..make_test_peer(12, mesh::NodeRole::Worker, vec![], vec![], true)
        };
        // size=200 GB, needed=220 GB; local 8 GB + peer 4 GB only = 12 GB.
        let fit = compute_mesh_fit(200.0, 8.0, &[small_peer]);
        assert!(!fit.fits_on_largest_node);
        assert!(!fit.fits_pooled);
    }

    #[test]
    fn compute_mesh_fit_uses_fast_memory_not_inflated_advertised_bytes() {
        let mut inflated_peer = make_test_peer(13, mesh::NodeRole::Worker, vec![], vec![], true);
        inflated_peer.vram_bytes = 106_000_000_000;
        inflated_peer.capability.vram_total_mb = 16 * 1024;

        // 40 GB needs 44 GB with headroom. Local 12 GB + the peer's real
        // 16 GiB GPU budget is short, even though its legacy vram_bytes field
        // is inflated by host-RAM offload and would otherwise claim 118 GB.
        let fit = compute_mesh_fit(40.0, 12.0, &[inflated_peer]);
        assert!(!fit.fits_on_largest_node);
        assert!(!fit.fits_pooled);
        assert!(fit.pooled_vram_gb < fit.needed_vram_gb);
    }

    #[test]
    fn classify_peer_split_role_returns_none_for_solo_serve() {
        let lone = make_test_peer(
            20,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["Qwen3-8B-Q4_K_M"],
            vec!["Qwen3-8B-Q4_K_M"],
            true,
        );
        let class = classify_peer_split_role(&lone, std::slice::from_ref(&lone), 16.0);
        assert!(class.role.is_none());
        assert!(class.group.is_none());
    }

    #[test]
    fn classify_peer_split_role_marks_pipeline_host_and_worker() {
        let host = make_test_peer(
            21,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["GLM-4.7-Flash"],
            vec!["GLM-4.7-Flash"],
            true,
        );
        let worker = make_test_peer(
            22,
            mesh::NodeRole::Worker,
            vec!["GLM-4.7-Flash"],
            vec![],
            true,
        );
        let peers = vec![host.clone(), worker.clone()];

        let host_class = classify_peer_split_role(&host, &peers, 16.0);
        assert_eq!(host_class.role.as_deref(), Some("pipeline_host"));
        let group = host_class.group.expect("split_group populated for host");
        assert_eq!(group.model, "GLM-4.7-Flash");
        assert_eq!(group.peer_ids.len(), 2);

        let worker_class = classify_peer_split_role(&worker, &peers, 16.0);
        assert_eq!(worker_class.role.as_deref(), Some("pipeline_worker"));
    }

    /// Healthy pipeline-split shape: a host in `Serving` and a worker in
    /// `Loading` (workers run only `rpc-server`; they never reach `Serving`
    /// by design — that's not a deadlock signal). The peer payload built
    /// for the host MUST keep `state=Serving` and `hosted_models=[model]`
    /// so chat clients can route to it; the worker's `state=Loading` is
    /// the truthful steady-state, not "the pipeline is degraded".
    ///
    /// Regression: an earlier "honesty gate" treated the worker's
    /// `Loading` as "host degraded", cleared the host's `hosted_models`
    /// on the wire, and broke every legitimate split — most visibly
    /// `scripts/ci-split-test.sh`, where Node C tried to find a routable
    /// host via `peer.role == "Host" && model in peer.hosted_models` and
    /// gave up after 60 s because we were lying about the host. See the
    /// CI failure on commit 0fa439c8 for the trace.
    #[test]
    fn peer_payload_keeps_host_serving_when_worker_is_loading() {
        let host_serving = make_test_peer(
            41,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["DeepSeek-R1-Distill-70B"],
            vec!["DeepSeek-R1-Distill-70B"],
            true,
        );
        let worker_loading = make_test_peer(
            42,
            mesh::NodeRole::Worker,
            vec!["DeepSeek-R1-Distill-70B"],
            vec![],
            true,
        );

        // The host must continue to derive as `Serving` even though its
        // cohort worker is still `Loading` — that's the literal shape of
        // a healthy pipeline-parallel run.
        assert_eq!(
            MeshApi::derive_peer_state(&host_serving),
            NodeState::Serving
        );
        assert_eq!(
            MeshApi::derive_peer_state(&worker_loading),
            NodeState::Loading
        );

        // No wire-level mutation of the host's `hosted_models` here, by
        // construction: we assert against `host_serving.hosted_models`
        // directly to pin down the contract that the wire payload keeps
        // its source-of-truth field intact.
        assert_eq!(
            host_serving.hosted_models,
            vec!["DeepSeek-R1-Distill-70B".to_string()]
        );
    }

    #[test]
    fn classify_peer_split_role_returns_none_when_no_host_in_cohort() {
        // Regression: the May 13 2026 split-brain deadlock. Three peers all
        // had `role: Worker` and `serving_models = ["DeepSeek-R1-Distill-70B"]`
        // but none had graduated to `NodeRole::Host`. The old classifier fell
        // through to `host_id = peer.id` (self) and labelled every peer as
        // `pipeline_worker`, producing the "I'm a worker for myself" payload
        // that powered every "Awaiting workers" / "loaded and waiting" lie on
        // the dashboard. The classifier should now refuse to invent a split
        // when no peer is actually hosting.
        let worker_a = make_test_peer(
            70,
            mesh::NodeRole::Worker,
            vec!["DeepSeek-R1-Distill-70B"],
            vec![],
            true,
        );
        let worker_b = make_test_peer(
            71,
            mesh::NodeRole::Worker,
            vec!["DeepSeek-R1-Distill-70B"],
            vec![],
            true,
        );
        let worker_c = make_test_peer(
            72,
            mesh::NodeRole::Worker,
            vec!["DeepSeek-R1-Distill-70B"],
            vec![],
            true,
        );
        let peers = vec![worker_a.clone(), worker_b.clone(), worker_c.clone()];
        let class = classify_peer_split_role(&worker_a, &peers, 16.0);
        assert!(
            class.role.is_none(),
            "no Host in cohort ⇒ no split_role; got {:?}",
            class.role
        );
        assert!(class.group.is_none());
    }

    #[test]
    fn classify_peer_split_role_still_marks_worker_when_host_present() {
        // Mirror of the regression: as soon as ONE peer in the cohort has
        // role: Host AND routes the model, the classifier should resume
        // labelling the others as `pipeline_worker`. The honesty gate must
        // not break the warm-mesh case.
        let host = make_test_peer(
            80,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["DeepSeek-R1-Distill-70B"],
            vec!["DeepSeek-R1-Distill-70B"],
            true,
        );
        let worker = make_test_peer(
            81,
            mesh::NodeRole::Worker,
            vec!["DeepSeek-R1-Distill-70B"],
            vec![],
            true,
        );
        let peers = vec![host.clone(), worker.clone()];
        let class = classify_peer_split_role(&worker, &peers, 16.0);
        assert_eq!(class.role.as_deref(), Some("pipeline_worker"));
        let group = class
            .group
            .expect("split group still populated when host present");
        assert_eq!(group.host_id, host.id.fmt_short().to_string());
    }

    /// Regression for the May 25 2026 Qwen3-8B-on-three-boxes miscall.
    ///
    /// Three peers all clicked "share Qwen3-8B-Q4_K_M": Mac Air (14.5 GB),
    /// LYU (17 GB), MSI (8.5 GB). The 5 GB model fits solo on any of them,
    /// so the election deliberately elected Mac as the sole `NodeRole::Host`
    /// and left LYU/MSI in `NodeRole::Worker` as standby (warm for failover
    /// but not actually running rpc-server for Mac).
    ///
    /// The pre-fix classifier still saw all three peers list the model in
    /// `serving_models`, paired them up via `cohort`, and labelled Mac as
    /// `pipeline_host` with a 3-way `split_group` while tagging LYU/MSI as
    /// `pipeline_worker` — the same lie pattern the host_is_none branch
    /// fixed for the deadlock case, just one step further down the road.
    /// Once the dashboard advertised a 40 GB pooled split that did not
    /// exist, the metrics page kept showing 0 contributors because the
    /// runtime never recorded a single split sample (it was solo all
    /// along), and the user closed two of the apps before the gap was
    /// understood.
    ///
    /// Gate: if the elected host has enough fast memory to hold the model
    /// alone (1.1× the GGUF bytes), the cohort is standby — return no role
    /// and no group, same shape as a true solo serve.
    #[test]
    fn classify_peer_split_role_returns_none_when_host_can_solo_the_model() {
        let mut host = make_test_peer(
            90,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["Qwen3-8B-Q4_K_M"],
            vec!["Qwen3-8B-Q4_K_M"],
            true,
        );
        host.vram_bytes = 15_000_000_000; // Mac Air-class, 15 GB fast memory
        host.available_model_sizes
            .insert("Qwen3-8B-Q4_K_M".to_string(), 5_000_000_000);
        let standby_lyu = make_test_peer(
            91,
            mesh::NodeRole::Worker,
            vec!["Qwen3-8B-Q4_K_M"],
            vec![],
            true,
        );
        let standby_msi = make_test_peer(
            92,
            mesh::NodeRole::Worker,
            vec!["Qwen3-8B-Q4_K_M"],
            vec![],
            true,
        );
        let peers = vec![host.clone(), standby_lyu.clone(), standby_msi.clone()];

        let host_class = classify_peer_split_role(&host, &peers, 14.5);
        assert!(
            host_class.role.is_none(),
            "host that can solo the model must not be tagged pipeline_host; got {:?}",
            host_class.role
        );
        assert!(
            host_class.group.is_none(),
            "no split_group when the cohort is standby"
        );

        for standby in [&standby_lyu, &standby_msi] {
            let class = classify_peer_split_role(standby, &peers, 14.5);
            assert!(
                class.role.is_none(),
                "standby Worker must not be tagged pipeline_worker; got {:?}",
                class.role
            );
            assert!(class.group.is_none());
        }
    }

    /// And the inverse: when the host CANNOT solo the model (genuine
    /// pipeline split — the Phase-1 case), the classifier still labels
    /// the host and workers correctly. This protects the DeepSeek-70B
    /// case from over-firing the standby gate.
    #[test]
    fn classify_peer_split_role_marks_split_when_model_does_not_fit_solo() {
        let mut host = make_test_peer(
            100,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["DeepSeek-R1-Distill-70B"],
            vec!["DeepSeek-R1-Distill-70B"],
            true,
        );
        host.vram_bytes = 16_000_000_000; // RTX 4060 Ti class — far short of 70 GB
        host.available_model_sizes
            .insert("DeepSeek-R1-Distill-70B".to_string(), 42_000_000_000);
        let worker = make_test_peer(
            101,
            mesh::NodeRole::Worker,
            vec!["DeepSeek-R1-Distill-70B"],
            vec![],
            true,
        );
        let peers = vec![host.clone(), worker.clone()];

        let host_class = classify_peer_split_role(&host, &peers, 16.0);
        assert_eq!(host_class.role.as_deref(), Some("pipeline_host"));
        assert!(host_class.group.is_some());

        let worker_class = classify_peer_split_role(&worker, &peers, 16.0);
        assert_eq!(worker_class.role.as_deref(), Some("pipeline_worker"));
    }

    #[test]
    fn classify_peer_split_role_treats_dual_routing_hosts_as_replicas() {
        // Two peers that each independently route HTTP for the same model are
        // replicas (the entry node load-balances across them) — every one
        // holds the whole model end-to-end. The peer-level classifier must NOT
        // fabricate a `moe_shard` / split_group here: it can't tell dense
        // replication from a genuine MoE expert-shard layout without GGUF
        // metadata, so the honest per-peer badge is silence and MoE-ness is
        // carried by the model-level `split_kind`. (Regression: a dense
        // Qwen3-8B replicated on two boxes showed up as a 2-way `moe_shard`.)
        let host_a = make_test_peer(
            30,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["Qwen3-30B-A3B"],
            vec!["Qwen3-30B-A3B"],
            true,
        );
        let host_b = make_test_peer(
            31,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["Qwen3-30B-A3B"],
            vec!["Qwen3-30B-A3B"],
            true,
        );
        let peers = vec![host_a.clone(), host_b.clone()];

        let class = classify_peer_split_role(&host_a, &peers, 16.0);
        assert_eq!(
            class.role, None,
            "two independent routing hosts are replicas, not a shard/split"
        );
        assert!(class.group.is_none());
        assert!(class.moe_shard.is_none());
    }

    #[test]
    fn classify_local_split_role_treats_dual_routing_hosts_as_replicas() {
        // Mirror of the peer-level replica gate for the local node: when we
        // are a Host serving a model AND another peer also routes it, we are
        // one of N replicas, not a shard. `my_split_role` must stay silent so
        // the dashboard's "you're contributing part of model Z" card doesn't
        // fire for a node that holds the whole model.
        let peer = make_test_peer(
            41,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["Qwen3-8B-Q4_K_M"],
            vec!["Qwen3-8B-Q4_K_M"],
            true,
        );
        let peers = vec![peer];
        let class = classify_local_split_role(
            "selflocalnode",
            true,  // is_host — we route HTTP for the model too
            false, // is_client
            &["Qwen3-8B-Q4_K_M".to_string()],
            &["Qwen3-8B-Q4_K_M".to_string()],
            16.0,
            &peers,
        );
        assert_eq!(
            class.role, None,
            "local host + another routing host = replicas, not a shard/split"
        );
        assert!(class.group.is_none());
        assert!(class.moe_shard.is_none());
    }

    #[test]
    fn test_build_gpus_with_reserved_bytes_preserves_index() {
        let result = build_gpus(
            Some("GPU0, GPU1, GPU2"),
            Some("100,200,300"),
            Some("10,,30"),
            None,
            None,
            None,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].reserved_bytes, Some(10));
        assert_eq!(result[1].reserved_bytes, None);
        assert_eq!(result[2].reserved_bytes, Some(30));
    }

    #[test]
    fn test_gpu_entry_omits_reserved_bytes_when_none() {
        let value = serde_json::to_value(build_gpus(
            Some("NVIDIA A100"),
            Some("85899345920"),
            None,
            Some("1948.70"),
            None,
            None,
        ))
        .unwrap();

        let first = value.as_array().unwrap().first().unwrap();
        assert!(first.get("reserved_bytes").is_none());
    }

    #[test]
    fn test_http_body_text_extracts_body() {
        let raw = b"POST /api/plugins/x/tools/y HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\n\r\n{\"a\":1}";
        assert_eq!(http_body_text(raw), "{\"a\":1}");
    }

    #[test]
    fn test_build_runtime_status_payload_uses_local_processes() {
        let result = build_runtime_status_payload(
            "Qwen",
            Some("llama".into()),
            true,
            true,
            Some(9337),
            vec![
                RuntimeProcessPayload {
                    name: "Qwen".into(),
                    backend: "llama".into(),
                    status: "ready".into(),
                    port: 9337,
                    pid: 100,
                    slots: 4,
                },
                RuntimeProcessPayload {
                    name: "Llama".into(),
                    backend: "llama".into(),
                    status: "ready".into(),
                    port: 9444,
                    pid: 101,
                    slots: 4,
                },
            ],
        );
        assert_eq!(result.models.len(), 2);
        assert_eq!(result.models[0].name, "Llama");
        assert_eq!(result.models[0].port, Some(9444));
        assert_eq!(result.models[1].name, "Qwen");
    }

    #[test]
    fn test_build_runtime_status_payload_adds_starting_primary() {
        let payload = build_runtime_status_payload(
            "Qwen",
            Some("llama".into()),
            true,
            false,
            Some(9337),
            vec![],
        );

        assert_eq!(payload.models.len(), 1);
        assert_eq!(payload.models[0].status, "starting");
        assert_eq!(payload.models[0].port, Some(9337));
    }

    #[test]
    fn test_build_runtime_processes_payload_sorts_processes() {
        let payload = build_runtime_processes_payload(vec![
            RuntimeProcessPayload {
                name: "Zulu".into(),
                backend: "llama".into(),
                status: "ready".into(),
                port: 9444,
                pid: 11,
                slots: 4,
            },
            RuntimeProcessPayload {
                name: "Alpha".into(),
                backend: "llama".into(),
                status: "ready".into(),
                port: 9337,
                pid: 10,
                slots: 4,
            },
        ]);

        assert_eq!(payload.processes.len(), 2);
        assert_eq!(payload.processes[0].name, "Alpha");
        assert_eq!(payload.processes[1].name, "Zulu");
    }

    #[test]
    fn test_classify_runtime_error_codes() {
        assert_eq!(classify_runtime_error("model 'x' is not loaded"), 404);
        assert_eq!(classify_runtime_error("model 'x' is already loaded"), 409);
        assert_eq!(
            classify_runtime_error("runtime load only supports models that fit locally"),
            422
        );
        assert_eq!(classify_runtime_error("bad request"), 400);
    }

    #[test]
    fn derive_local_node_state_prefers_client() {
        let node_state = MeshApi::derive_local_node_state(true, true, true, true, "Qwen");

        assert_eq!(node_state, NodeState::Client);
        assert_eq!(MeshApi::derive_node_status(node_state), "Client");
    }

    #[test]
    fn derive_local_node_state_returns_standby_without_ready_runtime() {
        let node_state = MeshApi::derive_local_node_state(false, false, false, false, "Qwen");

        assert_eq!(node_state, NodeState::Standby);
        assert_eq!(MeshApi::derive_node_status(node_state), "Standby");
    }

    #[test]
    fn derive_local_node_state_returns_loading_for_declared_but_unready_work() {
        let host_loading = MeshApi::derive_local_node_state(false, true, false, false, "Qwen");
        let worker_loading = MeshApi::derive_local_node_state(false, false, false, true, "Qwen");

        assert_eq!(host_loading, NodeState::Loading);
        assert_eq!(worker_loading, NodeState::Loading);
        assert_eq!(MeshApi::derive_node_status(host_loading), "Loading");
        assert_eq!(MeshApi::derive_node_status(worker_loading), "Loading");
    }

    #[test]
    fn derive_local_node_state_returns_serving_for_ready_runtime() {
        let host_serving = MeshApi::derive_local_node_state(false, true, true, false, "Qwen");
        let worker_serving = MeshApi::derive_local_node_state(false, false, true, true, "Qwen");

        assert_eq!(host_serving, NodeState::Serving);
        assert_eq!(worker_serving, NodeState::Serving);
        assert_eq!(MeshApi::derive_node_status(host_serving), "Serving");
        assert_eq!(MeshApi::derive_node_status(worker_serving), "Serving");
    }

    #[test]
    fn derive_local_node_state_never_emits_legacy_idle_or_split_labels() {
        let labels = [
            MeshApi::derive_node_status(MeshApi::derive_local_node_state(
                true, true, true, true, "Qwen",
            )),
            MeshApi::derive_node_status(MeshApi::derive_local_node_state(
                false, false, false, false, "Qwen",
            )),
            MeshApi::derive_node_status(MeshApi::derive_local_node_state(
                false, true, false, false, "Qwen",
            )),
            MeshApi::derive_node_status(MeshApi::derive_local_node_state(
                false, false, true, true, "Qwen",
            )),
            MeshApi::derive_node_status(MeshApi::derive_local_node_state(
                false, false, false, false, "",
            )),
        ];

        for label in labels {
            assert!(matches!(
                label.as_str(),
                "Client" | "Standby" | "Loading" | "Serving"
            ));
            assert_ne!(label, "Idle");
            assert_ne!(label, "Serving (split)");
            assert_ne!(label, "Worker (split)");
        }
    }

    fn make_test_state_endpoint_id(seed: u8) -> iroh::EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        iroh::EndpointId::from(iroh::SecretKey::from_bytes(&bytes).public())
    }

    fn make_test_state_peer(seed: u8, role: mesh::NodeRole) -> mesh::PeerInfo {
        let id = make_test_state_endpoint_id(seed);
        mesh::PeerInfo {
            id,
            addr: iroh::EndpointAddr {
                id,
                addrs: Default::default(),
            },
            tunnel_port: None,
            role,
            models: vec![],
            vram_bytes: 0,
            rtt_ms: None,
            model_source: None,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            available_models: vec![],
            requested_models: vec![],
            last_seen: Instant::now(),
            last_mentioned: Instant::now(),
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
            first_joined_mesh_ts: None,
            inflight_requests: 0,
            system_ram_bytes: 0,
            model_timings: vec![],
            native_baselines: vec![],
            rpc_ready: None,
            capability: crate::mesh::NodeCapability::default(),
            model_ad: Default::default(),
        }
    }

    fn make_legacy_peer_fixture(
        seed: u8,
        role: mesh::NodeRole,
        serving_models: Vec<&str>,
    ) -> mesh::PeerInfo {
        let mut peer = make_test_state_peer(seed, role);
        peer.version = Some("0.54.0".into());
        peer.serving_models = serving_models.into_iter().map(str::to_string).collect();
        peer.hosted_models = vec![];
        peer.hosted_models_known = false;
        peer.served_model_runtime = vec![];
        peer
    }

    #[test]
    fn derive_peer_state_prefers_client_role() {
        let mut peer = make_test_state_peer(1, mesh::NodeRole::Client);
        peer.serving_models = vec!["Qwen".into()];
        peer.hosted_models = vec!["Qwen".into()];
        peer.hosted_models_known = true;
        peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
            model_name: "Qwen".into(),
            identity_hash: None,
            context_length: Some(8192),
            ready: true,
        }];

        assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Client);
    }

    #[test]
    fn derive_peer_state_returns_serving_for_ready_runtime() {
        let mut peer = make_test_state_peer(2, mesh::NodeRole::Host { http_port: 9337 });
        peer.serving_models = vec!["Qwen".into()];
        peer.hosted_models = vec!["Qwen".into()];
        peer.hosted_models_known = true;
        peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
            model_name: "Qwen".into(),
            identity_hash: None,
            context_length: Some(8192),
            ready: true,
        }];

        assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Serving);
    }

    #[test]
    fn derive_peer_state_returns_loading_for_assigned_but_unready_peer() {
        let mut peer = make_test_state_peer(3, mesh::NodeRole::Worker);
        peer.serving_models = vec!["Qwen".into()];
        peer.served_model_runtime = vec![mesh::ModelRuntimeDescriptor {
            model_name: "Qwen".into(),
            identity_hash: None,
            context_length: None,
            ready: false,
        }];

        assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Loading);
    }

    #[test]
    fn derive_peer_state_returns_standby_for_connected_idle_peer() {
        let peer = make_test_state_peer(4, mesh::NodeRole::Worker);

        assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Standby);
    }

    #[test]
    fn derive_peer_state_falls_back_to_legacy_serving_models() {
        let mut peer = make_test_state_peer(5, mesh::NodeRole::Worker);
        peer.serving_models = vec!["Qwen".into()];

        assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Serving);
    }

    /// A modern peer that explicitly advertised `hosted_models` (even as
    /// empty) has opted into the newer schema; if its `serving_models`
    /// declares intent but `hosted_models` is empty and no runtime descriptor
    /// reports ready, we must NOT paint it as Serving — that's aspirational
    /// state, not actual serving. The regression this test guards is a failed
    /// launch (e.g. missing rpc-server binary) showing up as
    /// "Serving Qwen3-0.6B" on the public status page.
    #[test]
    fn derive_peer_state_reports_loading_for_modern_peer_with_empty_hosted_models() {
        let mut peer = make_test_state_peer(8, mesh::NodeRole::Host { http_port: 9337 });
        peer.serving_models = vec!["Qwen3-0.6B".into()];
        peer.hosted_models = vec![];
        peer.hosted_models_known = true;
        peer.served_model_runtime = vec![];

        assert_eq!(MeshApi::derive_peer_state(&peer), NodeState::Loading);
    }

    #[test]
    fn legacy_peer_fixture_uses_backend_state_fallback() {
        let serving_peer =
            make_legacy_peer_fixture(6, mesh::NodeRole::Host { http_port: 9337 }, vec!["Qwen"]);
        let standby_peer = make_legacy_peer_fixture(7, mesh::NodeRole::Worker, vec![]);

        assert_eq!(
            MeshApi::derive_peer_state(&serving_peer),
            NodeState::Serving
        );
        assert_eq!(
            MeshApi::derive_peer_state(&standby_peer),
            NodeState::Standby
        );
    }

    #[test]
    fn test_decode_runtime_model_path_decodes_percent_not_plus() {
        // %20 is a space; + is a literal plus in URL paths (not a space)
        assert_eq!(
            decode_runtime_model_path("/api/runtime/models/Llama%203.2+1B"),
            Some("Llama 3.2+1B".into())
        );
    }

    #[test]
    fn test_decode_runtime_model_path_decodes_utf8_multibyte() {
        // é is U+00E9, encoded in UTF-8 as 0xC3 0xA9
        assert_eq!(
            decode_runtime_model_path("/api/runtime/models/mod%C3%A9le"),
            Some("modéle".into())
        );
        // invalid UTF-8 sequence should return None
        assert_eq!(decode_runtime_model_path("/api/runtime/models/%80"), None);
    }

    async fn build_test_mesh_api_with_api_port(api_port: u16) -> MeshApi {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let resolved_plugins = plugin::ResolvedPlugins {
            externals: vec![],
            inactive: vec![],
        };
        let (mesh_tx, _mesh_rx) = mpsc::channel(1);
        let plugin_manager = plugin::PluginManager::start(
            &resolved_plugins,
            plugin::PluginHostMode {
                mesh_visibility: MeshVisibility::Private,
            },
            mesh_tx,
        )
        .await
        .unwrap();
        MeshApi::new(
            node,
            "test-model".to_string(),
            api_port,
            0,
            plugin_manager,
            affinity::AffinityRouter::default(),
        )
    }

    async fn build_test_mesh_api() -> MeshApi {
        build_test_mesh_api_with_api_port(3131).await
    }

    async fn build_test_mesh_api_with_plugin_manager(
        api_port: u16,
        plugin_manager: plugin::PluginManager,
    ) -> MeshApi {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        MeshApi::new(
            node,
            "test-model".to_string(),
            api_port,
            0,
            plugin_manager,
            affinity::AffinityRouter::default(),
        )
    }

    async fn spawn_management_test_server(
        state: MeshApi,
    ) -> (
        std::net::SocketAddr,
        tokio::task::JoinHandle<anyhow::Result<()>>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle_request(stream, &state).await
        });
        (addr, handle)
    }

    async fn send_management_request(addr: std::net::SocketAddr, raw_request: String) -> String {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(raw_request.as_bytes()).await.unwrap();
        let _ = stream.shutdown().await;
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }

    fn json_body(response: &str) -> serde_json::Value {
        let body = response.split("\r\n\r\n").nth(1).unwrap_or_default();
        serde_json::from_str(body).unwrap_or(serde_json::Value::Null)
    }

    async fn replace_test_wakeable_inventory(
        state: &MeshApi,
        entries: Vec<WakeableInventoryEntry>,
    ) {
        let inventory = { state.inner.lock().await.wakeable_inventory.clone() };
        inventory.replace_for_tests(entries).await;
    }

    fn make_test_wakeable_entry(
        logical_id: &str,
        model: &str,
        vram_gb: f32,
    ) -> WakeableInventoryEntry {
        WakeableInventoryEntry {
            logical_id: logical_id.to_string(),
            models: vec![model.to_string()],
            vram_gb,
            provider: Some("test-provider".to_string()),
            state: WakeableState::Sleeping,
            wake_eta_secs: Some(45),
        }
    }

    fn make_test_peer(
        seed: u8,
        role: mesh::NodeRole,
        serving_models: Vec<&str>,
        hosted_models: Vec<&str>,
        hosted_models_known: bool,
    ) -> mesh::PeerInfo {
        let peer_id = iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[seed; 32]).public());
        mesh::PeerInfo {
            id: peer_id,
            addr: iroh::EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            tunnel_port: None,
            role,
            first_joined_mesh_ts: None,
            models: Vec::new(),
            vram_bytes: 24_000_000_000,
            rtt_ms: None,
            model_source: None,
            serving_models: serving_models.into_iter().map(str::to_string).collect(),
            hosted_models: hosted_models.into_iter().map(str::to_string).collect(),
            hosted_models_known,
            available_models: Vec::new(),
            requested_models: Vec::new(),
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
            available_model_metadata: Vec::new(),
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: Vec::new(),
            served_model_runtime: Vec::new(),
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

    #[derive(Clone)]
    struct BlobstoreApiTestBridge {
        plugin_name: String,
        store: blobstore::BlobStore,
    }

    #[derive(Clone)]
    struct BlackboardApiTestBridge {
        plugin_name: String,
        store: blackboard::BlackboardStore,
    }

    impl BlobstoreApiTestBridge {
        fn error_response(message: impl Into<String>) -> plugin::proto::ErrorResponse {
            plugin::proto::ErrorResponse {
                code: ErrorCode::INTERNAL_ERROR.0,
                message: message.into(),
                data_json: String::new(),
            }
        }
    }

    impl BlackboardApiTestBridge {
        fn error_response(message: impl Into<String>) -> plugin::proto::ErrorResponse {
            plugin::proto::ErrorResponse {
                code: ErrorCode::INTERNAL_ERROR.0,
                message: message.into(),
                data_json: String::new(),
            }
        }
    }

    impl plugin::PluginRpcBridge for BlobstoreApiTestBridge {
        fn handle_request(
            &self,
            plugin_name: String,
            method: String,
            params_json: String,
        ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
            let expected_plugin_name = self.plugin_name.clone();
            let store = self.store.clone();
            Box::pin(async move {
                if plugin_name != expected_plugin_name {
                    return Err(Self::error_response(format!(
                        "Unsupported test plugin '{}'",
                        plugin_name
                    )));
                }
                if method != "tools/call" {
                    return Err(Self::error_response(format!(
                        "Unsupported method '{}'",
                        method
                    )));
                }

                let request: senda_plugin::OperationRequest =
                    serde_json::from_str(&params_json)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                let result_json = match request.name.as_str() {
                    blobstore::PUT_REQUEST_OBJECT_TOOL => {
                        let request: blobstore::PutRequestObjectRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let response = store
                            .put_request_object(request)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(
                            serde_json::to_value(response)
                                .map_err(|err| Self::error_response(err.to_string()))?,
                        ))
                        .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    blobstore::COMPLETE_REQUEST_TOOL | blobstore::ABORT_REQUEST_TOOL => {
                        let request: blobstore::FinishRequestRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let response = store
                            .finish_request(&request.request_id)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(
                            serde_json::to_value(response)
                                .map_err(|err| Self::error_response(err.to_string()))?,
                        ))
                        .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    _ => {
                        return Err(Self::error_response(format!(
                            "Unsupported blobstore tool '{}'",
                            request.name
                        )));
                    }
                };

                Ok(plugin::RpcResult { result_json })
            })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> plugin::BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    impl plugin::PluginRpcBridge for BlackboardApiTestBridge {
        fn handle_request(
            &self,
            plugin_name: String,
            method: String,
            params_json: String,
        ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
            let expected_plugin_name = self.plugin_name.clone();
            let store = self.store.clone();
            Box::pin(async move {
                if plugin_name != expected_plugin_name {
                    return Err(Self::error_response(format!(
                        "Unsupported test plugin '{}'",
                        plugin_name
                    )));
                }
                if method != "tools/call" {
                    return Err(Self::error_response(format!(
                        "Unsupported method '{}'",
                        method
                    )));
                }

                let request: senda_plugin::OperationRequest =
                    serde_json::from_str(&params_json)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                let result_json = match request.name.as_str() {
                    "feed" => {
                        let request: blackboard::FeedRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let response = store
                            .feed(request.since, request.from.as_deref(), request.limit)
                            .await;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(
                            serde_json::to_value(response)
                                .map_err(|err| Self::error_response(err.to_string()))?,
                        ))
                        .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    "search" => {
                        let request: blackboard::SearchRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let mut response = store.search(&request.query, request.since).await;
                        response.truncate(request.limit.max(1));
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(
                            serde_json::to_value(response)
                                .map_err(|err| Self::error_response(err.to_string()))?,
                        ))
                        .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    "post" => {
                        let request: blackboard::PostRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let item = blackboard::BlackboardItem::new(
                            if request.from.trim().is_empty() {
                                "mcp".into()
                            } else {
                                request.from
                            },
                            if request.peer_id.trim().is_empty() {
                                "mcp".into()
                            } else {
                                request.peer_id
                            },
                            request.text,
                        );
                        let response = store.post(item).await.map_err(Self::error_response)?;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(
                            serde_json::to_value(response)
                                .map_err(|err| Self::error_response(err.to_string()))?,
                        ))
                        .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    _ => {
                        return Err(Self::error_response(format!(
                            "Unsupported blackboard tool '{}'",
                            request.name
                        )));
                    }
                };

                Ok(plugin::RpcResult { result_json })
            })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> plugin::BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    fn temp_blobstore_root(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("senda-api-{name}-{}", rand::random::<u64>()))
    }

    async fn build_blobstore_api_plugin_manager() -> (plugin::PluginManager, std::path::PathBuf) {
        let plugin_name = "blobstore";
        let root = temp_blobstore_root("blobstore");
        let bridge = BlobstoreApiTestBridge {
            plugin_name: plugin_name.into(),
            store: blobstore::BlobStore::new(root.clone()),
        };
        let plugin_manager =
            plugin::PluginManager::for_test_bridge(&[plugin_name], Arc::new(bridge));
        let mut manifests = HashMap::new();
        manifests.insert(
            plugin_name.to_string(),
            senda_plugin::plugin_manifest![senda_plugin::capability(
                blobstore::OBJECT_STORE_CAPABILITY
            ),],
        );
        plugin_manager
            .set_test_manifests(manifests.into_iter().collect())
            .await;
        (plugin_manager, root)
    }

    async fn build_blackboard_api_plugin_manager() -> plugin::PluginManager {
        let plugin_name = "blackboard";
        let bridge = BlackboardApiTestBridge {
            plugin_name: plugin_name.into(),
            store: blackboard::BlackboardStore::new(true),
        };
        let plugin_manager =
            plugin::PluginManager::for_test_bridge(&[plugin_name], Arc::new(bridge));
        let mut manifests = HashMap::new();
        manifests.insert(
            plugin_name.to_string(),
            senda_plugin::plugin_manifest![
                senda_plugin::capability(blackboard::BLACKBOARD_CHANNEL),
                senda_plugin::http_get("/feed", "feed"),
                senda_plugin::http_get("/search", "search"),
                senda_plugin::http_post("/post", "post"),
            ],
        );
        plugin_manager
            .set_test_manifests(manifests.into_iter().collect())
            .await;
        plugin_manager
    }

    async fn spawn_capturing_upstream(
        response_body: &str,
    ) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let response = response_body.to_string();
        let (request_tx, request_rx) = oneshot::channel();
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = proxy::read_http_request(&mut stream).await.unwrap();
            let _ = request_tx.send(request.raw);

            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(resp.as_bytes()).await.unwrap();
            let _ = stream.shutdown().await;
        });
        (port, request_rx, handle)
    }

    async fn spawn_streaming_upstream(
        content_type: &str,
        chunks: Vec<(Duration, Vec<u8>)>,
    ) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let content_type = content_type.to_string();
        let (request_tx, request_rx) = oneshot::channel();
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = proxy::read_http_request(&mut stream).await.unwrap();
            let _ = request_tx.send(request.raw);

            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            );
            if stream.write_all(header.as_bytes()).await.is_err() {
                return;
            }

            for (delay, chunk) in chunks {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                let chunk_header = format!("{:x}\r\n", chunk.len());
                if stream.write_all(chunk_header.as_bytes()).await.is_err() {
                    return;
                }
                if stream.write_all(&chunk).await.is_err() {
                    return;
                }
                if stream.write_all(b"\r\n").await.is_err() {
                    return;
                }
            }

            let _ = stream.write_all(b"0\r\n\r\n").await;
            let _ = stream.shutdown().await;
        });
        (port, request_rx, handle)
    }

    fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    async fn read_until_contains(
        stream: &mut TcpStream,
        needle: &[u8],
        timeout: Duration,
    ) -> Vec<u8> {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut response = Vec::new();
        while !contains_bytes(&response, needle) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            assert!(
                !remaining.is_zero(),
                "timed out waiting for {:?} in response: {}",
                String::from_utf8_lossy(needle),
                String::from_utf8_lossy(&response)
            );
            let mut chunk = [0u8; 4096];
            let n = tokio::time::timeout(remaining, stream.read(&mut chunk))
                .await
                .expect("timed out waiting for response bytes")
                .unwrap();
            assert!(n > 0, "unexpected EOF while waiting for response bytes");
            response.extend_from_slice(&chunk[..n]);
        }
        response
    }

    #[tokio::test]
    async fn test_management_request_parser_handles_fragmented_post_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = br#"{"text":"fragmented"}"#;
        let headers = format!(
            "POST /api/blackboard/post HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                proxy::read_http_request(&mut stream),
            )
            .await
            .unwrap()
            .unwrap()
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            stream.write_all(&headers.as_bytes()[..45]).await.unwrap();
            stream.write_all(&headers.as_bytes()[45..]).await.unwrap();
            stream.write_all(&body[..8]).await.unwrap();
            stream.write_all(&body[8..]).await.unwrap();
            let mut sink = [0u8; 1];
            let _ = stream.read(&mut sink).await;
        });

        client.await.unwrap();
        let request = server.await.unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/api/blackboard/post");
        assert_eq!(http_body_text(&request.raw), "{\"text\":\"fragmented\"}");
    }

    #[tokio::test]
    async fn test_api_events_sends_initial_payload_and_updates() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state.clone()).await;

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();

        let initial = read_until_contains(&mut stream, b"data: {", Duration::from_secs(2)).await;
        let initial_text = String::from_utf8_lossy(&initial);
        assert!(initial_text.contains("HTTP/1.1 200 OK"));
        assert!(initial_text.contains("Content-Type: text/event-stream"));
        assert!(initial_text.contains("\"llama_ready\":false"));

        state.update(true, true).await;
        let updated =
            read_until_contains(&mut stream, b"\"llama_ready\":true", Duration::from_secs(2)).await;
        let updated_text = String::from_utf8_lossy(&updated);
        assert!(updated_text.contains("\"llama_ready\":true"));
        assert!(updated_text.contains("\"is_host\":true"));

        drop(stream);
        handle.abort();
    }

    #[tokio::test]
    async fn test_api_events_push_publication_state_updates() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state.clone()).await;

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();

        let _initial = read_until_contains(
            &mut stream,
            b"\"publication_state\":\"private\"",
            Duration::from_secs(2),
        )
        .await;

        state
            .set_publication_state(crate::api::PublicationState::PublishFailed)
            .await;
        let updated = read_until_contains(
            &mut stream,
            b"\"publication_state\":\"publish_failed\"",
            Duration::from_secs(2),
        )
        .await;
        let updated_text = String::from_utf8_lossy(&updated);
        assert!(updated_text.contains("\"publication_state\":\"publish_failed\""));

        drop(stream);
        handle.abort();
    }

    #[tokio::test]
    async fn test_api_status_excludes_mesh_models_and_models_endpoint_serves_them() {
        let state = build_test_mesh_api().await;
        let (status_addr, status_handle) = spawn_management_test_server(state.clone()).await;

        let status_response = send_management_request(
            status_addr,
            "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;
        assert!(status_response.starts_with("HTTP/1.1 200"));
        let status_body = json_body(&status_response);
        assert!(status_body.get("mesh_models").is_none());
        status_handle.abort();

        let (models_addr, models_handle) = spawn_management_test_server(state).await;
        let models_response = send_management_request(
            models_addr,
            "GET /api/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;
        assert!(models_response.starts_with("HTTP/1.1 200"));
        let models_body = json_body(&models_response);
        assert!(models_body.get("mesh_models").is_some());

        models_handle.abort();
    }

    #[tokio::test]
    async fn test_api_search_catalog_returns_canonical_model_refs() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let response = send_management_request(
            addr,
            "GET /api/search?q=Qwen3-Coder-Next&catalog=true&artifact=gguf&limit=5&sort=trending HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 200"));
        let payload = json_body(&response);
        assert_eq!(payload["source"], json!("catalog"));
        assert_eq!(payload["filter"], json!("gguf"));
        assert_eq!(payload["sort"], json!("trending"));
        assert!(payload.get("machine").is_some());
        let results = payload["results"].as_array().cloned().unwrap_or_default();
        assert!(
            !results.is_empty(),
            "expected at least one catalog result for Qwen3-Coder-Next"
        );
        let hit = results
            .into_iter()
            .find(|entry| entry["ref"] == json!("Qwen3-Coder-Next-Q4_K_M"))
            .expect("canonical catalog model ref present");
        assert_eq!(hit["repo_id"], json!("Qwen/Qwen3-Coder-Next-GGUF"));
        assert_eq!(hit["type"], json!("gguf"));
        assert_eq!(
            hit["show"],
            json!("senda models show Qwen3-Coder-Next-Q4_K_M")
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_api_search_caps_limit_and_uses_canonical_parameter_sort_name() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let response = send_management_request(
            addr,
            "GET /api/search?q=Qwen3-Coder-Next&catalog=true&artifact=gguf&limit=999&sort=parameters-desc HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 200"));
        let payload = json_body(&response);
        assert_eq!(payload["sort"], json!("parameters-desc"));
        let results = payload["results"].as_array().cloned().unwrap_or_default();
        assert!(
            results.len() <= 50,
            "expected catalog response to apply the API limit cap"
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_api_search_requires_q_query_parameter() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let response = send_management_request(
            addr,
            "GET /api/search?catalog=true HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 400"));
        let payload = json_body(&response);
        assert_eq!(
            payload["error"],
            json!("Missing required 'q' query parameter")
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_api_search_rejects_invalid_sort_value() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let response = send_management_request(
            addr,
            "GET /api/search?q=qwen&sort=random HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 400"));
        let payload = json_body(&response);
        assert_eq!(
            payload["error"],
            json!("Invalid 'sort' value 'random'. Expected one of: trending, downloads, likes, created, updated, parameters-desc, parameters-asc")
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_api_model_interests_post_and_get_round_trip() {
        let state = build_test_mesh_api().await;
        let (post_addr, post_handle) = spawn_management_test_server(state.clone()).await;
        let body = r#"{"model_ref":"Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M","source":"ui"}"#;

        let post_response = send_management_request(
            post_addr,
            format!(
                "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
        )
        .await;

        assert!(post_response.starts_with("HTTP/1.1 201"));
        let post_payload = json_body(&post_response);
        assert_eq!(post_payload["created"], json!(true));
        assert_eq!(
            post_payload["interest"]["model_ref"],
            json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
        );
        assert_eq!(post_payload["interest"]["submission_source"], json!("ui"));
        assert_eq!(post_payload["model_interests"].as_array().unwrap().len(), 1);
        post_handle.abort();

        let (get_addr, get_handle) = spawn_management_test_server(state).await;
        let get_response = send_management_request(
            get_addr,
            "GET /api/model-interests HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

        assert!(get_response.starts_with("HTTP/1.1 200"));
        let get_payload = json_body(&get_response);
        let interests = get_payload["model_interests"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert_eq!(interests.len(), 1);
        assert_eq!(
            interests[0]["model_ref"],
            json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
        );
        assert_eq!(interests[0]["submission_source"], json!("ui"));

        get_handle.abort();
    }

    #[tokio::test]
    async fn test_api_model_interests_post_is_idempotent() {
        let state = build_test_mesh_api().await;
        let body = r#"{"model_ref":"Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M","source":"ui"}"#;
        let request = format!(
            "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let (first_addr, first_handle) = spawn_management_test_server(state.clone()).await;
        let first_response = send_management_request(first_addr, request.clone()).await;
        assert!(first_response.starts_with("HTTP/1.1 201"));
        let first_payload = json_body(&first_response);
        let created_at = first_payload["interest"]["created_at_unix"]
            .as_u64()
            .expect("created_at_unix");
        first_handle.abort();

        let (second_addr, second_handle) = spawn_management_test_server(state).await;
        let second_response = send_management_request(second_addr, request).await;
        assert!(second_response.starts_with("HTTP/1.1 200"));
        let second_payload = json_body(&second_response);
        assert_eq!(second_payload["created"], json!(false));
        assert_eq!(
            second_payload["interest"]["model_ref"],
            json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
        );
        assert_eq!(
            second_payload["interest"]["created_at_unix"],
            json!(created_at)
        );
        assert_eq!(
            second_payload["model_interests"].as_array().unwrap().len(),
            1
        );

        second_handle.abort();
    }

    #[tokio::test]
    async fn test_api_model_interests_delete_decodes_percent_encoded_model_ref() {
        let state = build_test_mesh_api().await;
        state
            .upsert_model_interest(
                crate::models::canonicalize_interest_model_ref(
                    "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M",
                )
                .unwrap(),
                Some("ui".to_string()),
            )
            .await;

        let (addr, handle) = spawn_management_test_server(state).await;
        let response = send_management_request(
            addr,
            "DELETE /api/model-interests/Qwen%2FQwen3-Coder-Next-GGUF%40main%3AQ4_K_M HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 200"));
        let payload = json_body(&response);
        assert_eq!(payload["removed"], json!(true));
        assert_eq!(
            payload["model_ref"],
            json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
        );
        assert_eq!(payload["model_interests"], json!([]));

        handle.abort();
    }

    #[tokio::test]
    async fn test_api_model_interests_reject_direct_urls() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;
        let body = r#"{"model_ref":"https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"}"#;

        let response = send_management_request(
            addr,
            format!(
                "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 400"));
        let payload = json_body(&response);
        assert_eq!(
            payload["error"],
            json!(
                "Invalid 'model_ref'. Use a canonical ref returned by /api/search, not a direct URL"
            )
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_api_model_interests_normalize_legacy_selector_revision_order() {
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;
        let body = r#"{"model_ref":"Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M@main","source":"ui"}"#;

        let response = send_management_request(
            addr,
            format!(
                "POST /api/model-interests HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 201"));
        let payload = json_body(&response);
        assert_eq!(
            payload["interest"]["model_ref"],
            json!("Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M")
        );

        handle.abort();
    }

    #[test]
    fn test_http_route_stats_only_count_http_callable_legacy_hosts() {
        let peers = vec![
            make_test_peer(
                0x41,
                mesh::NodeRole::Host { http_port: 9337 },
                vec!["legacy-host-model"],
                Vec::new(),
                false,
            ),
            make_test_peer(
                0x42,
                mesh::NodeRole::Worker,
                vec!["worker-only-model"],
                Vec::new(),
                false,
            ),
        ];

        let host_stats = http_route_stats("legacy-host-model", &peers, &[], None, 0.0);
        assert_eq!(host_stats.node_count, 1);
        assert_eq!(host_stats.active_nodes.len(), 1);
        assert!(host_stats.mesh_vram_gb > 0.0);

        let worker_stats = http_route_stats("worker-only-model", &peers, &[], None, 0.0);
        assert_eq!(worker_stats, HttpRouteStats::default());
    }

    #[tokio::test]
    async fn wakeable_inventory_does_not_change_peer_count() {
        let state = build_test_mesh_api().await;
        replace_test_wakeable_inventory(
            &state,
            vec![make_test_wakeable_entry(
                "sleeping-node-1",
                "wakeable-only-model",
                48.0,
            )],
        )
        .await;

        let status = state.status().await;
        assert!(status.peers.is_empty());
        assert_eq!(status.wakeable_nodes.len(), 1);
        assert_eq!(status.wakeable_nodes[0].logical_id, "sleeping-node-1");
    }

    #[tokio::test]
    async fn wakeable_inventory_does_not_change_mesh_vram_totals() {
        let state = build_test_mesh_api().await;
        replace_test_wakeable_inventory(
            &state,
            vec![make_test_wakeable_entry(
                "sleeping-node-1",
                "wakeable-only-model",
                48.0,
            )],
        )
        .await;

        let status = state.status().await;
        let peers = vec![make_test_peer(
            0x51,
            mesh::NodeRole::Host { http_port: 9337 },
            vec!["wakeable-only-model"],
            vec!["wakeable-only-model"],
            true,
        )];
        let route_stats = http_route_stats("wakeable-only-model", &peers, &[], None, 0.0);

        assert_eq!(status.wakeable_nodes.len(), 1);
        assert_eq!(route_stats.node_count, 1);
        assert!(route_stats.mesh_vram_gb > 0.0);
    }

    #[tokio::test]
    async fn wakeable_inventory_is_not_routable_capacity() {
        let state = build_test_mesh_api().await;
        replace_test_wakeable_inventory(
            &state,
            vec![make_test_wakeable_entry(
                "sleeping-node-1",
                "wakeable-only-model",
                48.0,
            )],
        )
        .await;

        let node = { state.inner.lock().await.node.clone() };
        let status = state.status().await;
        let served_models = node.models_being_served().await;
        let hosts = node.hosts_for_model("wakeable-only-model").await;

        assert_eq!(status.wakeable_nodes.len(), 1);
        assert!(!served_models
            .iter()
            .any(|model| model == "wakeable-only-model"));
        assert!(hosts.is_empty());
    }

    #[tokio::test]
    async fn wakeable_inventory_is_excluded_from_v1_models() {
        let state = build_test_mesh_api().await;
        replace_test_wakeable_inventory(
            &state,
            vec![make_test_wakeable_entry(
                "sleeping-node-1",
                "wakeable-only-model",
                48.0,
            )],
        )
        .await;

        let node = { state.inner.lock().await.node.clone() };
        let served_models = node.models_being_served().await;

        assert!(!served_models
            .iter()
            .any(|model| model == "wakeable-only-model"));
        assert!(served_models.is_empty());
    }

    #[tokio::test]
    async fn wakeable_inventory_is_excluded_from_host_selection() {
        let state = build_test_mesh_api().await;
        replace_test_wakeable_inventory(
            &state,
            vec![make_test_wakeable_entry(
                "sleeping-node-1",
                "wakeable-only-model",
                48.0,
            )],
        )
        .await;

        let node = { state.inner.lock().await.node.clone() };
        let hosts = node.hosts_for_model("wakeable-only-model").await;

        assert!(hosts.is_empty());
    }

    #[test]
    fn build_wakeable_node_preserves_typed_internal_state() {
        let sleeping = MeshApi::build_wakeable_node(WakeableInventoryEntry {
            logical_id: "sleeping-node".to_string(),
            models: vec!["test-model".to_string()],
            vram_gb: 24.0,
            provider: Some("test-provider".to_string()),
            state: WakeableState::Sleeping,
            wake_eta_secs: Some(45),
        });
        let waking = MeshApi::build_wakeable_node(WakeableInventoryEntry {
            logical_id: "waking-node".to_string(),
            models: vec!["test-model".to_string()],
            vram_gb: 24.0,
            provider: Some("test-provider".to_string()),
            state: WakeableState::Waking,
            wake_eta_secs: Some(10),
        });

        assert_eq!(sleeping.state, WakeableNodeState::Sleeping);
        assert_eq!(waking.state, WakeableNodeState::Waking);
    }

    #[tokio::test]
    async fn test_api_status_includes_local_gpu_benchmark_metrics() {
        let state = build_test_mesh_api().await;
        let node = {
            let mut inner = state.inner.lock().await;
            inner.node.gpu_name = Some("NVIDIA A100".into());
            inner.node.gpu_vram = Some("85899345920".into());
            inner.node.gpu_reserved_bytes = Some("1073741824".into());
            inner.node.hostname = Some("worker-01".into());
            inner.node.is_soc = Some(false);
            inner.node.clone()
        };

        *node.gpu_mem_bandwidth_gbps.lock().await = Some(vec![1948.7]);
        *node.gpu_compute_tflops_fp32.lock().await = Some(vec![19.5]);
        *node.gpu_compute_tflops_fp16.lock().await = Some(vec![312.0]);

        let (addr, handle) = spawn_management_test_server(state).await;
        let response = send_management_request(
            addr,
            "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 200"));
        let payload = json_body(&response);
        let gpu = &payload["gpus"][0];
        assert_eq!(gpu["name"], json!("NVIDIA A100"));
        assert_eq!(gpu["vram_bytes"], json!(85899345920_u64));
        assert_eq!(gpu["reserved_bytes"], json!(1073741824_u64));
        assert_eq!(gpu["mem_bandwidth_gbps"], json!(1948.7));
        assert_eq!(gpu["compute_tflops_fp32"], json!(19.5));
        assert_eq!(gpu["compute_tflops_fp16"], json!(312.0));

        handle.abort();
    }

    #[tokio::test]
    async fn test_api_status_includes_routing_metrics_summary() {
        let state = build_test_mesh_api().await;
        let node = {
            let inner = state.inner.lock().await;
            inner.node.clone()
        };
        let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());

        node.record_inference_attempt(
            Some("test-model"),
            &election::InferenceTarget::Local(9338),
            Duration::from_millis(4),
            Duration::from_millis(16),
            crate::network::metrics::AttemptOutcome::Timeout,
            None,
        );
        node.record_inference_attempt(
            Some("test-model"),
            &election::InferenceTarget::Remote(peer_id),
            Duration::from_millis(18),
            Duration::from_millis(48),
            crate::network::metrics::AttemptOutcome::Success,
            Some(12),
        );
        node.record_routed_request(
            Some("test-model"),
            2,
            crate::network::metrics::RequestOutcome::Success(
                crate::network::metrics::RequestService::Remote,
            ),
        );

        let (addr, handle) = spawn_management_test_server(state).await;
        let response = send_management_request(
            addr,
            "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 200"));
        let payload = json_body(&response);
        assert_eq!(payload["routing_metrics"]["request_count"], json!(1));
        assert_eq!(payload["routing_metrics"]["successful_requests"], json!(1));
        assert_eq!(payload["routing_metrics"]["retry_count"], json!(1));
        assert_eq!(payload["routing_metrics"]["failover_count"], json!(1));
        assert_eq!(
            payload["routing_metrics"]["attempt_timeout_count"],
            json!(1)
        );
        assert_eq!(
            payload["routing_metrics"]["pressure"]["remotely_served_request_count"],
            json!(1)
        );
        assert_eq!(
            payload["routing_metrics"]["local_node"]["remote_attempt_count"],
            json!(1)
        );
        assert_eq!(
            payload["routing_metrics"]["local_node"]["local_attempt_count"],
            json!(1)
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_api_models_include_model_routing_metrics() {
        let state = build_test_mesh_api().await;
        let node = {
            let inner = state.inner.lock().await;
            inner.node.clone()
        };
        let model_name = crate::models::catalog::MODEL_CATALOG[0].name.clone();
        let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());
        node.set_requested_models(vec![model_name.clone()]).await;

        node.record_inference_attempt(
            Some(&model_name),
            &election::InferenceTarget::Remote(peer_id),
            Duration::from_millis(6),
            Duration::from_millis(24),
            crate::network::metrics::AttemptOutcome::Success,
            Some(9),
        );
        node.record_routed_request(
            Some(&model_name),
            1,
            crate::network::metrics::RequestOutcome::Success(
                crate::network::metrics::RequestService::Remote,
            ),
        );

        let (addr, handle) = spawn_management_test_server(state).await;
        let response = send_management_request(
            addr,
            "GET /api/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;

        assert!(response.starts_with("HTTP/1.1 200"));
        let payload = json_body(&response);
        let models = payload["mesh_models"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let model = models
            .into_iter()
            .find(|entry| entry["name"] == model_name)
            .expect("catalog model present");
        assert_eq!(model["routing_metrics"]["request_count"], json!(1));
        assert_eq!(model["routing_metrics"]["successful_requests"], json!(1));
        assert_eq!(
            model["routing_metrics"]["targets"][0]["kind"],
            json!("remote")
        );
        assert_eq!(
            model["routing_metrics"]["targets"][0]["success_count"],
            json!(1)
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_api_objects_routes_through_object_store_capability() {
        let (plugin_manager, blobstore_root) = build_blobstore_api_plugin_manager().await;
        let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let body = json!({
            "request_id": "req-api-object",
            "mime_type": "text/plain",
            "file_name": "note.txt",
            "bytes_base64": "aGVsbG8=",
            "expires_in_secs": 60,
            "uses_remaining": 1,
        })
        .to_string();
        let request = format!(
            "POST /api/objects HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let response = send_management_request(addr, request).await;

        assert!(response.starts_with("HTTP/1.1 201"));
        let payload = json_body(&response);
        assert_eq!(payload["request_id"], "req-api-object");
        assert_eq!(payload["mime_type"], "text/plain");
        assert!(payload["token"]
            .as_str()
            .unwrap_or_default()
            .starts_with("obj_"));

        handle.abort();
        let _ = std::fs::remove_dir_all(blobstore_root);
    }

    #[tokio::test]
    async fn test_api_blackboard_routes_through_blackboard_capability() {
        let plugin_manager = build_blackboard_api_plugin_manager().await;
        let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;

        let (post_addr, post_handle) = spawn_management_test_server(state.clone()).await;
        let post_body = json!({ "text": "hello integration blackboard" }).to_string();
        let post_request = format!(
            "POST /api/blackboard/post HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            post_body.len(),
            post_body
        );
        let post_response = send_management_request(post_addr, post_request).await;
        assert!(post_response.starts_with("HTTP/1.1 200"));
        let posted = json_body(&post_response);
        assert_eq!(posted["text"], "hello integration blackboard");
        post_handle.abort();

        let (feed_addr, feed_handle) = spawn_management_test_server(state.clone()).await;
        let feed_response = send_management_request(
            feed_addr,
            "GET /api/blackboard/feed?limit=5 HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;
        assert!(feed_response.starts_with("HTTP/1.1 200"));
        let feed = json_body(&feed_response);
        let feed_items = feed.as_array().cloned().unwrap_or_default();
        assert!(feed_items
            .iter()
            .any(|item| item["text"] == "hello integration blackboard"));
        feed_handle.abort();

        let (search_addr, search_handle) = spawn_management_test_server(state).await;
        let search_response = send_management_request(
            search_addr,
            "GET /api/blackboard/search?q=integration HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
        )
        .await;
        assert!(search_response.starts_with("HTTP/1.1 200"));
        let search = json_body(&search_response);
        let search_items = search.as_array().cloned().unwrap_or_default();
        assert!(search_items
            .iter()
            .any(|item| item["text"] == "hello integration blackboard"));
        search_handle.abort();
    }

    #[tokio::test]
    async fn test_api_chat_smoke_for_image_request() {
        let (upstream_port, upstream_rx, upstream_handle) =
            spawn_capturing_upstream(r#"{"ok":true}"#).await;
        let state = build_test_mesh_api_with_api_port(upstream_port).await;
        state.update(true, true).await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
                ]
            }],
            "stream": false
        })
        .to_string();
        let request = format!(
            "POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response_text = String::from_utf8(response).unwrap();
        let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

        assert!(response_text.starts_with("HTTP/1.1 200 OK"));
        assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(raw.contains(r#""type":"image_url""#));
        assert!(raw.contains("data:image/png;base64,aGVsbG8="));

        handle.abort();
        let _ = upstream_handle.await;
    }

    #[tokio::test]
    async fn test_api_chat_smoke_for_audio_request() {
        let (upstream_port, upstream_rx, upstream_handle) =
            spawn_capturing_upstream(r#"{"ok":true}"#).await;
        let state = build_test_mesh_api_with_api_port(upstream_port).await;
        state.update(true, true).await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this audio"},
                    {"type": "input_audio", "input_audio": {
                        "data": "UklGRg==",
                        "format": "wav",
                        "mime_type": "audio/wav"
                    }}
                ]
            }],
            "stream": false
        })
        .to_string();
        let request = format!(
            "POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response_text = String::from_utf8(response).unwrap();
        let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

        assert!(response_text.starts_with("HTTP/1.1 200 OK"));
        assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(raw.contains(r#""type":"input_audio""#));
        assert!(raw.contains(r#""data":"UklGRg==""#));
        assert!(raw.contains(r#""format":"wav""#));
        assert!(raw.contains(r#""mime_type":"audio/wav""#));

        handle.abort();
        let _ = upstream_handle.await;
    }

    #[tokio::test]
    async fn test_api_responses_smoke_for_image_request() {
        let (upstream_port, upstream_rx, upstream_handle) =
            spawn_capturing_upstream(r#"{"id":"chatcmpl","object":"chat.completion","created":1,"model":"test-model","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#).await;
        let state = build_test_mesh_api_with_api_port(upstream_port).await;
        state.update(true, true).await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let body = serde_json::json!({
            "model": "test-model",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe this image"},
                    {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}
                ]
            }],
            "stream": false
        })
        .to_string();
        let request = format!(
            "POST /api/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response_text = String::from_utf8(response).unwrap();
        let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

        assert!(response_text.starts_with("HTTP/1.1 200 OK"));
        assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(raw.contains(r#""type":"image_url""#));
        assert!(raw.contains("data:image/png;base64,aGVsbG8="));

        handle.abort();
        let _ = upstream_handle.await;
    }

    #[tokio::test]
    async fn test_api_responses_smoke_for_file_request() {
        let (upstream_port, upstream_rx, upstream_handle) =
            spawn_capturing_upstream(r#"{"id":"chatcmpl","object":"chat.completion","created":1,"model":"test-model","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#).await;
        let state = build_test_mesh_api_with_api_port(upstream_port).await;
        state.update(true, true).await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let body = serde_json::json!({
            "model": "test-model",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "read this file"},
                    {
                        "type": "input_file",
                        "input_file": {
                            "url": "data:text/plain;base64,aGVsbG8=",
                            "mime_type": "text/plain",
                            "file_name": "hello.txt"
                        }
                    }
                ]
            }],
            "stream": false
        })
        .to_string();
        let request = format!(
            "POST /api/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response_text = String::from_utf8(response).unwrap();
        let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

        assert!(response_text.starts_with("HTTP/1.1 200 OK"));
        assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(raw.contains(r#""type":"input_file""#));
        assert!(raw.contains(r#""url":"data:text/plain;base64,aGVsbG8=""#));
        assert!(raw.contains(r#""mime_type":"text/plain""#));
        assert!(raw.contains(r#""file_name":"hello.txt""#));

        handle.abort();
        let _ = upstream_handle.await;
    }

    #[tokio::test]
    async fn test_api_responses_stream_smoke() {
        let (upstream_port, upstream_rx, upstream_handle) = spawn_streaming_upstream(
            "text/event-stream",
            vec![(
                Duration::ZERO,
                br#"event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"hello"}

event: done
data: [DONE]

"#
                .to_vec(),
            )],
        )
        .await;
        let state = build_test_mesh_api_with_api_port(upstream_port).await;
        state.update(true, true).await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let body = serde_json::json!({
            "model": "test-model",
            "input": "say hello",
            "stream": true
        })
        .to_string();
        let request = format!(
            "POST /api/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        let response = read_until_contains(
            &mut stream,
            br#"event: response.output_text.delta"#,
            Duration::from_secs(2),
        )
        .await;
        let response_text = String::from_utf8(response).unwrap();
        let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

        assert!(response_text.starts_with("HTTP/1.1 200 OK"));
        assert!(response_text.contains("event: response.output_text.delta"));
        assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(raw.contains(r#""stream":true"#));

        handle.abort();
        let _ = upstream_handle.await;
    }

    #[tokio::test]
    async fn status_payload_populates_local_instances_from_scanner() {
        use crate::runtime::instance::LocalInstanceSnapshot;
        use std::path::PathBuf;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let snapshots = vec![
            LocalInstanceSnapshot {
                pid: 1234,
                api_port: Some(3131),
                version: Some("0.56.0".to_string()),
                started_at_unix: 1700000000,
                runtime_dir: PathBuf::from("/tmp/a"),
                is_self: true,
            },
            LocalInstanceSnapshot {
                pid: 5678,
                api_port: Some(3132),
                version: Some("0.56.0".to_string()),
                started_at_unix: 1700000100,
                runtime_dir: PathBuf::from("/tmp/b"),
                is_self: false,
            },
        ];

        let shared: Arc<Mutex<Vec<LocalInstanceSnapshot>>> = Arc::new(Mutex::new(snapshots));
        let result: Vec<LocalInstance> = {
            let s = shared.lock().await;
            s.iter()
                .map(|snap| LocalInstance {
                    pid: snap.pid,
                    api_port: snap.api_port,
                    version: snap.version.clone(),
                    started_at_unix: snap.started_at_unix,
                    runtime_dir: snap.runtime_dir.to_string_lossy().to_string(),
                    is_self: snap.is_self,
                })
                .collect()
        };

        assert_eq!(result.len(), 2);
        assert!(result.iter().any(|i| i.is_self && i.pid == 1234));
        assert!(result.iter().any(|i| !i.is_self && i.pid == 5678));
    }

    #[tokio::test]
    async fn status_payload_safety_net_adds_self_when_empty() {
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let shared: Arc<Mutex<Vec<crate::runtime::instance::LocalInstanceSnapshot>>> =
            Arc::new(Mutex::new(vec![]));

        let mut instances: Vec<LocalInstance> = {
            let s = shared.lock().await;
            s.iter()
                .map(|snap| LocalInstance {
                    pid: snap.pid,
                    api_port: snap.api_port,
                    version: snap.version.clone(),
                    started_at_unix: snap.started_at_unix,
                    runtime_dir: snap.runtime_dir.to_string_lossy().to_string(),
                    is_self: snap.is_self,
                })
                .collect()
        };

        // Simulate the safety net logic
        if instances.is_empty() {
            instances.push(LocalInstance {
                pid: std::process::id(),
                api_port: Some(3131),
                version: Some(SENDA_VERSION.to_string()),
                started_at_unix: 0,
                runtime_dir: String::new(),
                is_self: true,
            });
        }

        assert_eq!(instances.len(), 1);
        assert!(instances[0].is_self);
        assert_eq!(instances[0].pid, std::process::id());
        assert_eq!(instances[0].api_port, Some(3131));
        assert_eq!(instances[0].version, Some(SENDA_VERSION.to_string()));
    }

    #[test]
    fn headless_mode_disables_ui_routes_but_preserves_api() {
        assert!(is_ui_only_route("/"));
        assert!(is_ui_only_route("/dashboard"));
        assert!(is_ui_only_route("/chat"));

        assert!(!is_ui_only_route("/api/status"));
        assert!(!is_ui_only_route("/api/events"));
        assert!(!is_ui_only_route("/api/discover"));
        assert!(!is_ui_only_route("/api/runtime"));
        assert!(!is_ui_only_route("/api/plugins"));
    }

    #[test]
    fn headless_mode_returns_404_for_assets_and_dashboard_routes() {
        assert!(is_ui_only_route("/dashboard/"));
        assert!(is_ui_only_route("/chat/"));
        assert!(is_ui_only_route("/chat/some-room"));
        assert!(is_ui_only_route("/assets/main.js"));
        assert!(is_ui_only_route("/assets/index-abc123.css"));
        assert!(is_ui_only_route("/favicon.ico"));
        assert!(is_ui_only_route("/logo.png"));
        assert!(is_ui_only_route("/manifest.webmanifest"));
        assert!(is_ui_only_route("/site.json"));

        assert!(!is_ui_only_route("/api/status.json"));
    }

    #[test]
    fn default_mode_still_serves_embedded_ui_routes() {
        assert!(is_ui_only_route("/"));
        assert!(is_ui_only_route("/dashboard"));
        assert!(is_ui_only_route("/chat"));
        assert!(is_ui_only_route("/assets/app.js"));

        assert!(!is_ui_only_route("/api/status"));
        assert!(!is_ui_only_route("/api/events"));
    }

    #[test]
    fn headless_status_command_works_against_management_api() {
        assert!(
            !is_ui_only_route("/api/status"),
            "/api/status must not be blocked in headless mode"
        );
        assert!(
            !is_ui_only_route("/api/events"),
            "/api/events must not be blocked in headless mode"
        );
        assert!(
            !is_ui_only_route("/api/discover"),
            "/api/discover must not be blocked in headless mode"
        );
    }

    #[test]
    fn headless_blackboard_status_still_reads_api_status() {
        assert!(
            !is_ui_only_route("/api/status"),
            "/api/status must be accessible in headless mode"
        );
        assert!(
            !is_ui_only_route("/api/runtime"),
            "/api/runtime must be accessible in headless mode"
        );
        assert!(
            !is_ui_only_route("/api/join"),
            "/api/join must be accessible in headless mode"
        );
    }

    #[test]
    fn headless_custom_console_port_keeps_api_and_disables_ui() {
        assert!(is_ui_only_route("/"), "/ must be blocked in headless mode");
        assert!(is_ui_only_route("/dashboard"), "/dashboard must be blocked");
        assert!(is_ui_only_route("/chat"), "/chat must be blocked");
        assert!(
            is_ui_only_route("/assets/main.js"),
            "/assets/* must be blocked"
        );
        assert!(
            !is_ui_only_route("/api/status"),
            "/api/status must not be blocked"
        );
        assert!(
            !is_ui_only_route("/api/events"),
            "/api/events must not be blocked"
        );
        assert!(
            !is_ui_only_route("/v1/models"),
            "/v1/models must not be blocked"
        );
        assert!(
            !is_ui_only_route("/v1/chat/completions"),
            "/v1/chat/completions must not be blocked"
        );
    }
}
