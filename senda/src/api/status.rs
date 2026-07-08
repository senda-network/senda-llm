use super::{RuntimeModelPayload, RuntimeProcessPayload};
use crate::crypto::{ModelAdStatus, ModelAdSummary, OwnershipStatus, OwnershipSummary};
use crate::network::{affinity, metrics};
use crate::system::hardware::expand_gpu_names;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum NodeState {
    Client,
    Standby,
    Loading,
    Serving,
}

impl NodeState {
    pub(super) const fn node_status_alias(self) -> &'static str {
        match self {
            Self::Client => "Client",
            Self::Standby => "Standby",
            Self::Loading => "Loading",
            Self::Serving => "Serving",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum WakeableNodeState {
    Sleeping,
    Waking,
}

#[derive(Serialize)]
pub(super) struct RuntimeStatusPayload {
    pub(super) models: Vec<RuntimeModelPayload>,
}

#[derive(Serialize)]
pub(super) struct RuntimeProcessesPayload {
    pub(super) processes: Vec<RuntimeProcessPayload>,
}

#[derive(Serialize)]
pub(super) struct GpuEntry {
    pub(super) name: String,
    pub(super) vram_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reserved_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mem_bandwidth_gbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) compute_tflops_fp32: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) compute_tflops_fp16: Option<f64>,
}

fn inferred_gpu_name_count(gpu_name: Option<&str>) -> usize {
    let Some(raw) = gpu_name.map(str::trim) else {
        return 0;
    };
    if raw.is_empty() {
        return 0;
    }

    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.split_once('×')
                .or_else(|| part.split_once('x'))
                .or_else(|| part.split_once('X'))
                .and_then(|(count, _)| count.trim().parse::<usize>().ok())
                .filter(|&count| count > 0)
                .unwrap_or(1)
        })
        .sum()
}

pub(super) fn build_gpus(
    gpu_name: Option<&str>,
    gpu_vram: Option<&str>,
    gpu_reserved_bytes: Option<&str>,
    gpu_mem_bandwidth: Option<&str>,
    gpu_compute_tflops_fp32: Option<&str>,
    gpu_compute_tflops_fp16: Option<&str>,
) -> Vec<GpuEntry> {
    let vrams: Vec<Option<u64>> = gpu_vram
        .map(|s| s.split(',').map(|v| v.trim().parse::<u64>().ok()).collect())
        .unwrap_or_default();
    let reserved: Vec<Option<u64>> = gpu_reserved_bytes
        .map(|s| s.split(',').map(|v| v.trim().parse::<u64>().ok()).collect())
        .unwrap_or_default();
    let bandwidths: Vec<Option<f64>> = gpu_mem_bandwidth
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    let compute_fp32: Vec<Option<f64>> = gpu_compute_tflops_fp32
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    let compute_fp16: Vec<Option<f64>> = gpu_compute_tflops_fp16
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    let expected_count = [
        vrams.len(),
        reserved.len(),
        bandwidths.len(),
        compute_fp32.len(),
        compute_fp16.len(),
        inferred_gpu_name_count(gpu_name),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let names = expand_gpu_names(gpu_name, expected_count)
        .into_iter()
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>();
    if names.is_empty() {
        return vec![];
    }
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| GpuEntry {
            name,
            vram_bytes: vrams.get(i).copied().flatten().unwrap_or(0),
            reserved_bytes: reserved.get(i).copied().flatten(),
            mem_bandwidth_gbps: bandwidths.get(i).copied().flatten(),
            compute_tflops_fp32: compute_fp32.get(i).copied().flatten(),
            compute_tflops_fp16: compute_fp16.get(i).copied().flatten(),
        })
        .collect()
}

#[derive(Serialize)]
pub(super) struct StatusPayload {
    pub(super) version: String,
    pub(super) latest_version: Option<String>,
    pub(super) node_id: String,
    pub(super) owner: OwnershipPayload,
    pub(super) token: String,
    pub(super) node_state: NodeState,
    pub(super) node_status: String,
    pub(super) is_host: bool,
    pub(super) is_client: bool,
    pub(super) llama_ready: bool,
    pub(super) model_name: String,
    pub(super) models: Vec<String>,
    pub(super) available_models: Vec<String>,
    pub(super) requested_models: Vec<String>,
    pub(super) serving_models: Vec<String>,
    pub(super) hosted_models: Vec<String>,
    pub(super) draft_name: Option<String>,
    pub(super) api_port: u16,
    pub(super) my_vram_gb: f64,
    pub(super) model_size_gb: f64,
    pub(super) peers: Vec<PeerPayload>,
    pub(super) wakeable_nodes: Vec<WakeableNode>,
    pub(super) local_instances: Vec<LocalInstance>,
    pub(super) launch_pi: Option<String>,
    pub(super) launch_goose: Option<String>,
    pub(super) inflight_requests: u64,
    /// Total system RAM on the local node (v0.66.38+). `0` when detection
    /// fails on the platform; UI should render as "—" rather than "0 GB".
    pub(super) system_ram_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mesh_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mesh_name: Option<String>,
    pub(super) nostr_discovery: bool,
    /// Best-effort publication state per Issue #240: private | public | publish_failed.
    pub(super) publication_state: String,
    pub(super) my_hostname: Option<String>,
    pub(super) my_is_soc: Option<bool>,
    pub(super) gpus: Vec<GpuEntry>,
    pub(super) capability: NodeCapabilityPayload,
    pub(super) routing_affinity: affinity::AffinityStatsSnapshot,
    /// Local-only routing outcome and current-node pressure snapshot measured on
    /// this node only; not mesh-wide aggregates.
    pub(super) routing_metrics: metrics::RoutingMetricsStatusSnapshot,
    /// v0.66.41 Phase 1 marketplace metrics: per-model median tokens/sec
    /// over the last hour of successful local-inference completions on
    /// this node. Empty when nothing has been served locally yet — UI
    /// should treat missing keys as "not yet measured", not "measured 0".
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) measured_tps_p50_by_model: std::collections::HashMap<String, f64>,
    /// v0.66.41 Phase 1 marketplace metrics: per-model median time-to-first-token
    /// (milliseconds) over the last hour of successful local-inference
    /// completions on this node. Same "missing = not measured" semantics
    /// as `measured_tps_p50_by_model`.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) measured_ttft_ms_p50_by_model: std::collections::HashMap<String, u64>,
    /// v0.66.49 Phase 3.0 benchmark honesty: per-model native TPS measured
    /// by issuing a synthetic chat directly to 127.0.0.1:llama_port,
    /// bypassing the entry tunnel, auth gateway, and routing layer. Empty
    /// when no baseline has been collected yet. Paired with
    /// `measured_tps_p50_by_model` lets the UI render the mesh overhead
    /// ratio per model.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) native_tps_p50_by_model: std::collections::HashMap<String, f64>,
    /// v0.66.49 Phase 3.0: per-model native TTFT (ms). Same semantics as
    /// `native_tps_p50_by_model`.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) native_ttft_ms_p50_by_model: std::collections::HashMap<String, u64>,
    /// Per-model completion tokens THIS node served over a rolling 7-day
    /// window (local-only, disk-persisted; never gossiped). Empty when
    /// nothing has been served. Feeds the desktop dashboard's
    /// "estimated earnings this week" preview — an illustrative estimate
    /// against a placeholder rate card, not a ledger or a payout.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) serving_tokens_7d_by_model: std::collections::HashMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) first_joined_mesh_ts: Option<u64>,
    /// This node's current split-role classification, mirroring the same
    /// field on `PeerPayload`. The desktop dashboard reads `my_split_role`
    /// to render the live "you're contributing layers X-Y of model Z"
    /// card.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) my_split_role: Option<String>,
    /// Pipeline-split group this node currently belongs to, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) my_split_group: Option<SplitGroupPayload>,
    /// MoE shard this node currently runs, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) my_moe_shard: Option<MoeShardPayload>,
    /// Current state of the mesh-visibility audit loop. Present only
    /// when this runtime was started with `--join-url` (i.e. it has a
    /// parent entry to verify against). The desktop UI and the public
    /// status page consume this to render the truth — "we say we're
    /// serving but the entry can't see us" — instead of the optimistic
    /// local-only `node_state` flag.
    ///
    /// See `mesh::visibility` for the audit semantics and auto-heal
    /// thresholds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mesh_visibility: Option<crate::mesh::MeshVisibilitySnapshot>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct WakeableNode {
    pub(super) logical_id: String,
    pub(super) models: Vec<String>,
    pub(super) vram_gb: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) provider: Option<String>,
    pub(super) state: WakeableNodeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) wake_eta_secs: Option<u32>,
}

#[derive(Serialize)]
pub(super) struct PeerPayload {
    pub(super) id: String,
    pub(super) owner: OwnershipPayload,
    /// v0.66.x Phase 3.1: verdict of verifying this peer's owner-signed model
    /// advertisement (the trust-sensitive subset of its performance claims)
    /// against the local trust store. `verified=false` with `status=unsigned`
    /// for legacy peers and peers with no owner key.
    pub(super) model_ad: ModelAdPayload,
    /// v0.66.x Phase 3.2: latest sample-and-verify verdict per model the local
    /// verifier probed on this peer, keyed by model id. `match` means the
    /// peer's live logits reproduced an independently-generated reference;
    /// `mismatch` means they diverged. Empty for peers nobody has probed yet
    /// (only the entry node runs the verifier). Observe-only — present
    /// regardless of whether enforcement is enabled.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) verify_by_model: std::collections::HashMap<String, VerifyPayload>,
    /// v0.66.x Phase 3.2: persistent reputation score per model the local
    /// verifier has probed on this peer, keyed by model id. Accumulated EWMA of
    /// sample-and-verify verdicts (survives restarts), unlike the hour-bounded
    /// `verify_by_model`. Empty for peers nobody has probed yet (only the entry
    /// node runs the verifier). Observe-only — does not gate routing.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) reputation_by_model: std::collections::HashMap<String, ReputationPayload>,
    pub(super) role: String,
    pub(super) state: NodeState,
    pub(super) models: Vec<String>,
    pub(super) available_models: Vec<String>,
    pub(super) requested_models: Vec<String>,
    pub(super) vram_gb: f64,
    pub(super) serving_models: Vec<String>,
    pub(super) hosted_models: Vec<String>,
    pub(super) hosted_models_known: bool,
    pub(super) version: Option<String>,
    pub(super) rtt_ms: Option<u32>,
    pub(super) inflight_requests: u64,
    /// Total system RAM in bytes (v0.66.38+). `0` means the peer is on a
    /// legacy build that doesn't gossip this field; the desktop / website
    /// should render it as "—" rather than "0 GB".
    pub(super) system_ram_bytes: u64,
    pub(super) hostname: Option<String>,
    pub(super) is_soc: Option<bool>,
    pub(super) gpus: Vec<GpuEntry>,
    pub(super) capability: NodeCapabilityPayload,
    /// v0.66.41 Phase 1 marketplace metrics: per-model median tokens/sec
    /// gossiped by this peer over the last hour of its successful
    /// local-inference completions. Empty for legacy peers (<= v0.66.40)
    /// and for peers with no recent local serving — UI treats missing
    /// keys as "not yet measured", not "measured 0".
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) measured_tps_p50_by_model: std::collections::HashMap<String, f64>,
    /// v0.66.41 Phase 1 marketplace metrics: per-model median time-to-first-token
    /// (milliseconds) gossiped by this peer. Same semantics as
    /// `measured_tps_p50_by_model`.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) measured_ttft_ms_p50_by_model: std::collections::HashMap<String, u64>,
    /// v0.66.49 Phase 3.0 benchmark honesty: per-model native TPS gossiped
    /// by this peer (measured by issuing a synthetic chat against its own
    /// llama-server, no mesh involvement). Empty for legacy peers
    /// (<= v0.66.48) and for peers that haven't completed a baseline run
    /// yet. Paired with `measured_tps_p50_by_model` lets the catalog
    /// render a per-`(peer, model)` mesh overhead ratio.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) native_tps_p50_by_model: std::collections::HashMap<String, f64>,
    /// v0.66.49 Phase 3.0: per-model native TTFT (ms). Same semantics as
    /// `native_tps_p50_by_model`.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub(super) native_ttft_ms_p50_by_model: std::collections::HashMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) first_joined_mesh_ts: Option<u64>,
    /// Coarse classification of how this peer is currently participating in
    /// inference for one of the mesh's models. The desktop UI reads this to
    /// render role badges ("Pipeline host of X", "Layer worker", "MoE shard")
    /// so users can see when the mesh is operating as a collective rather
    /// than a bag of independent servers.
    ///
    /// Inferred from `role` + `serving_models` + the model's mesh-wide
    /// fan-out — see [`crate::api::mod::classify_peer_split_role`]. The
    /// runtime does NOT (yet) propagate live MoE shard membership or
    /// per-layer assignment over gossip; clients that want layer ranges or
    /// expert counts should treat those fields as best-effort hints.
    /// Phase 2: coarse serving topology for marketplace UI (`solo` default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) serving_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) split_role: Option<String>,
    /// When this peer is participating in pipeline-parallel inference for a
    /// model, identifies the elected host and the peer set splitting it.
    /// Always `None` for solo serves and standby peers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) split_group: Option<SplitGroupPayload>,
    /// When this peer is one of N independent MoE shard nodes for a model,
    /// names the model and gives the rough shard-of-N count. Per-expert
    /// indices are intentionally omitted — they're an implementation detail
    /// and not stable across runtime restarts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) moe_shard: Option<MoeShardPayload>,
}

/// Pipeline-parallel split group membership. Surfaced on `PeerPayload` and on
/// the local-node fields of `StatusPayload` whenever an election has placed
/// this peer in a multi-node serving group.
#[derive(Clone, Debug, Serialize)]
pub(super) struct SplitGroupPayload {
    /// Model being split across the group (e.g. "GLM-4.7-Flash-Q4_K_M").
    pub(super) model: String,
    /// Short id of the elected host node — the one running `llama-server`
    /// and coordinating layer workers via RPC. Matches `peer.id` formatting.
    pub(super) host_id: String,
    /// Short ids of all peers in the group (host + workers). The local node
    /// is included if it's a member.
    pub(super) peer_ids: Vec<String>,
    /// Combined VRAM (GB) advertised by every group member. The chat product
    /// uses this in copy like "split across 3 nodes pooling 72 GB".
    pub(super) total_group_vram_gb: f64,
}

/// MoE expert-shard membership for a peer. Each shard node runs an
/// independent `llama-server` with its own slice of the expert tensor;
/// there's no cross-node traffic during inference.
#[derive(Clone, Debug, Serialize)]
pub(super) struct MoeShardPayload {
    /// Model being sharded (e.g. "Qwen3-30B-A3B-Q4_K_M").
    pub(super) model: String,
    /// Total number of MoE shard nodes in the active deployment.
    pub(super) total_shards: u32,
}

/// Wire-friendly view of [`crate::mesh::NodeCapability`] for the status API.
/// Coarse on purpose — UI clients render a single capability badge per node.
#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct NodeCapabilityPayload {
    /// "metal" | "cuda" | "rocm" | "vulkan" | "cpu"
    pub(super) backend: String,
    /// "apple" | "nvidia" | "amd" | "intel" | "none"
    pub(super) vendor: String,
    /// "lo" | "mid" | "hi" | "pro"
    pub(super) compute_class: String,
    pub(super) vram_total_mb: u64,
    pub(super) vram_free_mb: u64,
    /// In whole GB. 0 means "unknown / fall back to vram_total_mb".
    pub(super) can_serve_max_gb: u64,
    pub(super) loaded_models: Vec<String>,
}

impl NodeCapabilityPayload {
    pub(super) fn from_capability(c: &crate::mesh::NodeCapability) -> Self {
        Self {
            backend: c.backend.label().to_string(),
            vendor: c.vendor.label().to_string(),
            compute_class: c.compute_class.label().to_string(),
            vram_total_mb: c.vram_total_mb,
            vram_free_mb: c.vram_free_mb,
            can_serve_max_gb: c.can_serve_max_gb,
            loaded_models: c.loaded_models.clone(),
        }
    }

    pub(super) fn from_capability_with_usable_vram(
        c: &crate::mesh::NodeCapability,
        usable_vram_bytes: u64,
    ) -> Self {
        let mut payload = Self::from_capability(c);
        if usable_vram_bytes > 0 {
            let usable_vram_mb = usable_vram_bytes / (1024 * 1024);
            payload.vram_total_mb = if payload.vram_total_mb > 0 {
                payload.vram_total_mb.min(usable_vram_mb)
            } else {
                usable_vram_mb
            };
            payload.vram_free_mb = payload.vram_free_mb.min(payload.vram_total_mb);
        }
        payload
    }
}

#[derive(Serialize)]
pub(super) struct OwnershipPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) owner_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cert_id: Option<String>,
    pub(super) status: String,
    pub(super) verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) expires_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) node_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) hostname_hint: Option<String>,
}

pub(super) fn build_ownership_payload(summary: &OwnershipSummary) -> OwnershipPayload {
    OwnershipPayload {
        owner_id: summary.owner_id.clone(),
        cert_id: summary.cert_id.clone(),
        status: match summary.status {
            OwnershipStatus::Verified => "verified",
            OwnershipStatus::Unsigned => "unsigned",
            OwnershipStatus::Expired => "expired",
            OwnershipStatus::InvalidSignature => "invalid_signature",
            OwnershipStatus::MismatchedNodeId => "mismatched_node_id",
            OwnershipStatus::RevokedOwner => "revoked_owner",
            OwnershipStatus::RevokedCert => "revoked_cert",
            OwnershipStatus::RevokedNodeId => "revoked_node_id",
            OwnershipStatus::UnsupportedProtocol => "unsupported_protocol",
            OwnershipStatus::UntrustedOwner => "untrusted_owner",
        }
        .to_string(),
        verified: summary.verified,
        expires_at_unix_ms: summary.expires_at_unix_ms,
        node_label: summary.node_label.clone(),
        hostname_hint: summary.hostname_hint.clone(),
    }
}

/// v0.66.x Phase 3.1: serialized verdict of verifying a peer's owner-signed
/// model advertisement against the local trust store. `verified` is the field
/// the UI keys on; `status` carries the specific reason when it's false.
#[derive(Serialize, Default)]
pub(super) struct ModelAdPayload {
    pub(super) status: String,
    pub(super) verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) owner_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) issued_at_unix_ms: Option<u64>,
    #[serde(default)]
    pub(super) model_count: usize,
}

/// v0.66.x Phase 3.2: serialized sample-and-verify verdict for one `(peer,
/// model)`. `verdict` is `match` | `mismatch` | `inconclusive`; the UI keys on
/// `match` for the "independently verified" badge and `mismatch` for a
/// warning. `agreement` is the prefix-token agreement fraction [0,1].
#[derive(Serialize)]
pub(super) struct VerifyPayload {
    pub(super) verdict: String,
    pub(super) agreement: f64,
    pub(super) compared_tokens: usize,
    pub(super) mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reason: Option<String>,
    pub(super) checked_at_unix_secs: u64,
}

/// v0.66.x Phase 3.2: serialized reputation score for one `(peer, model)`. The
/// UI keys on `grade` (`trusted` | `watch` | `unproven`) for the chip and shows
/// `score` / `samples` in the tooltip. Unlike `verify_by_model` (hour-bounded,
/// latest-only), this is the persistent EWMA accumulator that survives entry
/// restarts. Entry-local; present only for `(peer, model)` pairs the entry
/// verifier has probed. Observe-only — does not gate routing.
#[derive(Serialize)]
pub(super) struct ReputationPayload {
    pub(super) grade: String,
    pub(super) score: f64,
    pub(super) samples: u64,
    pub(super) matches: u64,
    pub(super) mismatches: u64,
    pub(super) last_verdict: String,
    pub(super) updated_at_unix_secs: u64,
}

pub(super) fn build_reputation_payload(
    s: &crate::inference::reputation::ReputationScore,
) -> ReputationPayload {
    ReputationPayload {
        grade: crate::inference::reputation::grade(s).as_str().to_string(),
        score: s.score,
        samples: s.samples,
        matches: s.matches,
        mismatches: s.mismatches,
        last_verdict: s.last_verdict.clone(),
        updated_at_unix_secs: s.updated_at_unix_secs,
    }
}

pub(super) fn build_verify_payload(rec: &crate::mesh::VerifyVerdictRecord) -> VerifyPayload {
    VerifyPayload {
        verdict: rec.verdict.clone(),
        agreement: rec.agreement,
        compared_tokens: rec.compared_tokens,
        mode: rec.mode.clone(),
        reason: rec.reason.clone(),
        checked_at_unix_secs: rec.checked_at_unix_secs,
    }
}

pub(super) fn build_model_ad_payload(summary: &ModelAdSummary) -> ModelAdPayload {
    ModelAdPayload {
        status: match summary.status {
            ModelAdStatus::Verified => "verified",
            ModelAdStatus::Unsigned => "unsigned",
            ModelAdStatus::InvalidSignature => "invalid_signature",
            ModelAdStatus::MismatchedNodeId => "mismatched_node_id",
            ModelAdStatus::Stale => "stale",
            ModelAdStatus::RevokedOwner => "revoked_owner",
            ModelAdStatus::UntrustedOwner => "untrusted_owner",
            ModelAdStatus::UnsupportedProtocol => "unsupported_protocol",
        }
        .to_string(),
        verified: summary.verified,
        owner_id: summary.owner_id.clone(),
        issued_at_unix_ms: summary.issued_at_unix_ms,
        model_count: summary.model_count,
    }
}

#[derive(Serialize)]
pub(super) struct LocalInstance {
    pub(super) pid: u32,
    pub(super) api_port: Option<u16>,
    pub(super) version: Option<String>,
    pub(super) started_at_unix: i64,
    pub(super) runtime_dir: String,
    pub(super) is_self: bool,
}

#[derive(Serialize)]
pub(super) struct MeshModelPayload {
    pub(super) name: String,
    pub(super) display_name: String,
    pub(super) status: String,
    pub(super) node_count: usize,
    pub(super) mesh_vram_gb: f64,
    pub(super) size_gb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) description: Option<String>,
    pub(super) multimodal: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) multimodal_status: Option<&'static str>,
    pub(super) vision: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) vision_status: Option<&'static str>,
    pub(super) audio: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) audio_status: Option<&'static str>,
    pub(super) reasoning: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reasoning_status: Option<&'static str>,
    pub(super) tool_use: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_use_status: Option<&'static str>,
    pub(super) moe: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) expert_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) used_expert_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_origin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_prompt_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) ranking_layer_scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) draft_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) request_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) last_active_secs_ago: Option<u64>,
    /// Local-only per-model routing outcome snapshot measured on the current
    /// node only; not mesh-wide aggregates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) routing_metrics: Option<metrics::ModelRoutingMetricsSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_page_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_file: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(super) active_nodes: Vec<String>,
    pub(super) fit_label: String,
    pub(super) fit_detail: String,
    pub(super) download_command: String,
    pub(super) run_command: String,
    pub(super) auto_command: String,
    /// Coarse classification of HOW the mesh is currently serving this
    /// model. One of:
    /// - `"solo"`        — one node, the model fits there alone
    /// - `"pipeline"`    — multiple nodes hosting the same dense model
    ///   together (layer-split via llama.cpp RPC)
    /// - `"moe"`         — multiple independent MoE shard nodes
    /// - `"multi_host"`  — multiple nodes serving the same model
    ///   independently for load balancing (no split)
    /// - `"cold"`        — no live host yet (model is in catalog only)
    ///
    /// Surfaced in the chat product so users can see, per model, when the
    /// mesh is "behaving as one big computer" vs running redundant copies.
    pub(super) split_kind: String,
    /// Structured "does the mesh fit this model" answer for the desktop
    /// Models page. Replaces the user-facing single-machine `fit_label`
    /// check that used to say "Won't fit on this Mac" against any model
    /// the local box couldn't hold solo.
    pub(super) mesh_fit: MeshFitPayload,
}

/// Mesh-wide capacity assessment for a single model. Tells the desktop UI
/// which of three states to render: "solo on this Mac" / "fits on the mesh
/// (pooled across N nodes)" / "needs more contributors (M GB short)".
///
/// The runtime computes this once per model from the live peer set, so the
/// app doesn't have to reimplement RTT eligibility or the ~10% headroom
/// multiplier the election planner uses. Backends are not filtered: as of
/// v0.66.13, cross-backend pipeline splits (e.g. Metal host + CUDA worker)
/// are eligible — see `build_dense_launch_plan` and the ROADMAP entry on
/// "Mixed-backend pipeline-parallel".
#[derive(Clone, Debug, Serialize)]
pub(super) struct MeshFitPayload {
    /// True when at least one peer (or this node) has enough free VRAM to
    /// hold the model alone.
    pub(super) fits_on_largest_node: bool,
    /// True when the eligible peer pool's combined VRAM clears the model's
    /// load requirement (size × 1.1). Implies the runtime would elect a
    /// pipeline-split group rather than refusing the request.
    pub(super) fits_pooled: bool,
    /// Combined VRAM (GB) of all eligible peers — RTT under the split cap,
    /// capability above the model size threshold; backend no longer gated.
    /// Equals `mesh_vram_gb` for already-warm models; pre-computed for
    /// cold models so the UI can show "M of N GB so far" without doing
    /// the math itself.
    pub(super) pooled_vram_gb: f64,
    /// Approximate VRAM (GB) required to load this model with headroom.
    /// `size_gb × 1.1` matches the dense election planner's threshold.
    pub(super) needed_vram_gb: f64,
    /// Number of peers (excluding entry nodes) that are currently eligible
    /// to participate in a split for this model.
    pub(super) eligible_peer_count: u32,
}

pub(super) fn build_runtime_status_payload(
    model_name: &str,
    primary_backend: Option<String>,
    is_host: bool,
    llama_ready: bool,
    llama_port: Option<u16>,
    mut local_processes: Vec<RuntimeProcessPayload>,
) -> RuntimeStatusPayload {
    local_processes.sort_by_key(|process| process.name.to_lowercase());

    let mut models: Vec<RuntimeModelPayload> = local_processes
        .into_iter()
        .map(|process| RuntimeModelPayload {
            name: process.name,
            backend: process.backend,
            status: process.status,
            port: Some(process.port),
        })
        .collect();

    let has_model_process = models.iter().any(|model| model.name == model_name);
    if is_host && !llama_ready && !has_model_process && !model_name.is_empty() {
        models.insert(
            0,
            RuntimeModelPayload {
                name: model_name.to_string(),
                backend: primary_backend.unwrap_or_else(|| "unknown".into()),
                status: "starting".into(),
                port: llama_port,
            },
        );
    }

    RuntimeStatusPayload { models }
}

pub(super) fn build_runtime_processes_payload(
    mut local_processes: Vec<RuntimeProcessPayload>,
) -> RuntimeProcessesPayload {
    local_processes.sort_by_key(|process| process.name.to_lowercase());
    RuntimeProcessesPayload {
        processes: local_processes,
    }
}

pub(crate) fn classify_runtime_error(msg: &str) -> u16 {
    if msg.contains("not loaded") {
        404
    } else if msg.contains("already loaded") {
        409
    } else if msg.contains("fit locally") || msg.contains("runtime load only supports") {
        422
    } else {
        400
    }
}

pub(super) fn decode_runtime_model_path(path: &str) -> Option<String> {
    let raw = path.strip_prefix("/api/runtime/models/")?;
    if raw.is_empty() {
        return None;
    }

    let bytes = raw.as_bytes();
    let mut decoded: Vec<u8> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = bytes[i + 1] as char;
                let lo = bytes[i + 2] as char;
                let hex = [hi, lo].iter().collect::<String>();
                if let Ok(value) = u8::from_str_radix(&hex, 16) {
                    decoded.push(value);
                    i += 3;
                    continue;
                } else {
                    return None;
                }
            }
            b'+' => decoded.push(b'+'),
            b => decoded.push(b),
        }
        i += 1;
    }
    String::from_utf8(decoded).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_owner_payload() -> OwnershipPayload {
        OwnershipPayload {
            owner_id: None,
            cert_id: None,
            status: "unsigned".to_string(),
            verified: false,
            expires_at_unix_ms: None,
            node_label: None,
            hostname_hint: None,
        }
    }

    #[test]
    fn test_peer_payload_serializes_version_field() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            model_ad: Default::default(),
            verify_by_model: Default::default(),
            reputation_by_model: Default::default(),
            role: "Worker".to_string(),
            state: NodeState::Standby,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            version: Some("0.56.0".to_string()),
            rtt_ms: None,
            inflight_requests: 0,
            system_ram_bytes: 0,
            hostname: None,
            is_soc: None,
            gpus: vec![],
            capability: NodeCapabilityPayload::default(),
            measured_tps_p50_by_model: std::collections::HashMap::new(),
            measured_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            native_tps_p50_by_model: std::collections::HashMap::new(),
            native_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            first_joined_mesh_ts: None,
            serving_mode: None,
            split_role: None,
            split_group: None,
            moe_shard: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(json.contains("\"version\":\"0.56.0\""));
    }

    #[test]
    fn test_peer_payload_serializes_null_version() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            model_ad: Default::default(),
            verify_by_model: Default::default(),
            reputation_by_model: Default::default(),
            role: "Worker".to_string(),
            state: NodeState::Standby,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            version: None,
            rtt_ms: None,
            inflight_requests: 0,
            system_ram_bytes: 0,
            hostname: None,
            is_soc: None,
            gpus: vec![],
            capability: NodeCapabilityPayload::default(),
            measured_tps_p50_by_model: std::collections::HashMap::new(),
            measured_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            native_tps_p50_by_model: std::collections::HashMap::new(),
            native_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            first_joined_mesh_ts: None,
            serving_mode: None,
            split_role: None,
            split_group: None,
            moe_shard: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(json.contains("\"version\":null"));
    }

    #[test]
    fn test_status_payload_has_local_instances_field() {
        let instances: Vec<LocalInstance> = vec![];
        let json = serde_json::to_string(&instances).expect("serialization failed");
        assert_eq!(json, "[]");
    }

    #[test]
    fn status_payload_serializes_node_state_and_node_status_alias() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            token: "token-1".to_string(),
            node_state: NodeState::Loading,
            node_status: NodeState::Loading.node_status_alias().to_string(),
            is_host: true,
            is_client: false,
            llama_ready: false,
            model_name: "Qwen".to_string(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 0.0,
            model_size_gb: 0.0,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            system_ram_bytes: 0,
            mesh_id: None,
            mesh_name: None,
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            capability: NodeCapabilityPayload::default(),
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            measured_tps_p50_by_model: std::collections::HashMap::new(),
            measured_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            native_tps_p50_by_model: std::collections::HashMap::new(),
            native_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            serving_tokens_7d_by_model: std::collections::HashMap::new(),
            first_joined_mesh_ts: None,
            my_split_role: None,
            my_split_group: None,
            my_moe_shard: None,
            mesh_visibility: None,
        };

        let json = serde_json::to_string(&status).expect("serialization failed");
        assert!(json.contains("\"node_state\":\"loading\""));
        assert!(json.contains("\"node_status\":\"Loading\""));
    }

    #[test]
    fn status_payload_keeps_node_status_for_compatibility() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            token: "token-1".to_string(),
            node_state: NodeState::Serving,
            node_status: NodeState::Serving.node_status_alias().to_string(),
            is_host: true,
            is_client: false,
            llama_ready: true,
            model_name: "Qwen".to_string(),
            models: vec!["Qwen".to_string()],
            available_models: vec!["Qwen".to_string()],
            requested_models: vec!["Qwen".to_string()],
            serving_models: vec!["Qwen".to_string()],
            hosted_models: vec!["Qwen".to_string()],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 24.0,
            model_size_gb: 4.0,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            system_ram_bytes: 0,
            mesh_id: None,
            mesh_name: None,
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            capability: NodeCapabilityPayload::default(),
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            measured_tps_p50_by_model: std::collections::HashMap::new(),
            measured_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            native_tps_p50_by_model: std::collections::HashMap::new(),
            native_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            serving_tokens_7d_by_model: std::collections::HashMap::new(),
            first_joined_mesh_ts: None,
            my_split_role: None,
            my_split_group: None,
            my_moe_shard: None,
            mesh_visibility: None,
        };

        let json = serde_json::to_string(&status).expect("serialization failed");
        assert!(json.contains("\"node_state\":\"serving\""));
        assert!(json.contains("\"node_status\":\"Serving\""));
    }

    #[test]
    fn status_payload_serializes_wakeable_nodes_separately() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            token: "token-1".to_string(),
            node_state: NodeState::Standby,
            node_status: NodeState::Standby.node_status_alias().to_string(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            model_name: String::new(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 0.0,
            model_size_gb: 0.0,
            peers: vec![],
            wakeable_nodes: vec![WakeableNode {
                logical_id: "provider-node-1".to_string(),
                models: vec!["Qwen".to_string()],
                vram_gb: 24.0,
                provider: Some("lightsail".to_string()),
                state: WakeableNodeState::Sleeping,
                wake_eta_secs: Some(90),
            }],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            system_ram_bytes: 0,
            mesh_id: None,
            mesh_name: None,
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            capability: NodeCapabilityPayload::default(),
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            measured_tps_p50_by_model: std::collections::HashMap::new(),
            measured_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            native_tps_p50_by_model: std::collections::HashMap::new(),
            native_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            serving_tokens_7d_by_model: std::collections::HashMap::new(),
            first_joined_mesh_ts: None,
            my_split_role: None,
            my_split_group: None,
            my_moe_shard: None,
            mesh_visibility: None,
        };

        let json = serde_json::to_value(&status).expect("serialization failed");
        assert_eq!(json["peers"], serde_json::json!([]));
        assert_eq!(json["wakeable_nodes"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["wakeable_nodes"][0]["state"], "sleeping");
        assert_eq!(json["wakeable_nodes"][0]["logical_id"], "provider-node-1");
    }

    #[test]
    fn status_payload_defaults_to_empty_wakeable_inventory() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            token: "token-1".to_string(),
            node_state: NodeState::Standby,
            node_status: NodeState::Standby.node_status_alias().to_string(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            model_name: String::new(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 0.0,
            model_size_gb: 0.0,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            system_ram_bytes: 0,
            mesh_id: None,
            mesh_name: None,
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            capability: NodeCapabilityPayload::default(),
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            measured_tps_p50_by_model: std::collections::HashMap::new(),
            measured_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            native_tps_p50_by_model: std::collections::HashMap::new(),
            native_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            serving_tokens_7d_by_model: std::collections::HashMap::new(),
            first_joined_mesh_ts: None,
            my_split_role: None,
            my_split_group: None,
            my_moe_shard: None,
            mesh_visibility: None,
        };

        let json = serde_json::to_value(&status).expect("serialization failed");
        assert_eq!(json["wakeable_nodes"], serde_json::json!([]));
        assert_eq!(json["peers"], serde_json::json!([]));
    }

    #[test]
    fn peer_status_serializes_state_without_mutating_role() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            model_ad: Default::default(),
            verify_by_model: Default::default(),
            reputation_by_model: Default::default(),
            role: "Host".to_string(),
            state: NodeState::Serving,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec!["Qwen".to_string()],
            hosted_models: vec!["Qwen".to_string()],
            hosted_models_known: true,
            version: Some("0.60.2".to_string()),
            rtt_ms: Some(12),
            inflight_requests: 0,
            system_ram_bytes: 0,
            hostname: Some("peer.local".to_string()),
            is_soc: Some(false),
            gpus: vec![],
            capability: NodeCapabilityPayload::default(),
            measured_tps_p50_by_model: std::collections::HashMap::new(),
            measured_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            native_tps_p50_by_model: std::collections::HashMap::new(),
            native_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            first_joined_mesh_ts: None,
            serving_mode: None,
            split_role: None,
            split_group: None,
            moe_shard: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(json.contains("\"role\":\"Host\""));
        assert!(json.contains("\"state\":\"serving\""));
    }

    /// v0.66.41 Phase 1: empty per-model timing maps must be SKIPPED in
    /// the JSON, not emitted as `"measured_tps_p50_by_model": {}`. The
    /// frontend Catalog view interprets a present-but-empty map the
    /// same as a populated one ("we measured zero throughput"), so the
    /// missing-entirely path is the only way to communicate "this peer
    /// has no measurements yet".
    #[test]
    fn peer_payload_omits_empty_model_timing_maps() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            model_ad: Default::default(),
            verify_by_model: Default::default(),
            reputation_by_model: Default::default(),
            role: "Host".to_string(),
            state: NodeState::Serving,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: true,
            version: None,
            rtt_ms: None,
            inflight_requests: 0,
            system_ram_bytes: 0,
            hostname: None,
            is_soc: None,
            gpus: vec![],
            capability: NodeCapabilityPayload::default(),
            measured_tps_p50_by_model: std::collections::HashMap::new(),
            measured_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            native_tps_p50_by_model: std::collections::HashMap::new(),
            native_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            first_joined_mesh_ts: None,
            serving_mode: None,
            split_role: None,
            split_group: None,
            moe_shard: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(
            !json.contains("measured_tps_p50_by_model"),
            "empty timing map must be omitted from JSON; got {json}"
        );
        assert!(
            !json.contains("measured_ttft_ms_p50_by_model"),
            "empty TTFT map must be omitted from JSON; got {json}"
        );
        assert!(
            !json.contains("native_tps_p50_by_model"),
            "empty native baseline map must be omitted from JSON; got {json}"
        );
        assert!(
            !json.contains("native_ttft_ms_p50_by_model"),
            "empty native TTFT map must be omitted from JSON; got {json}"
        );
    }

    /// Populated per-model timing maps must serialize as JSON objects so
    /// the frontend Catalog view can render per-model rows.
    #[test]
    fn peer_payload_emits_populated_model_timings() {
        let mut tps = std::collections::HashMap::new();
        tps.insert("Qwen3-32B-Q4_K_M".to_string(), 14.5);
        let mut ttft = std::collections::HashMap::new();
        ttft.insert("Qwen3-32B-Q4_K_M".to_string(), 320u64);
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            model_ad: Default::default(),
            verify_by_model: Default::default(),
            reputation_by_model: Default::default(),
            role: "Host".to_string(),
            state: NodeState::Serving,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: true,
            version: None,
            rtt_ms: None,
            inflight_requests: 0,
            system_ram_bytes: 0,
            hostname: None,
            is_soc: None,
            gpus: vec![],
            capability: NodeCapabilityPayload::default(),
            measured_tps_p50_by_model: tps,
            measured_ttft_ms_p50_by_model: ttft,
            native_tps_p50_by_model: std::collections::HashMap::new(),
            native_ttft_ms_p50_by_model: std::collections::HashMap::new(),
            first_joined_mesh_ts: None,
            serving_mode: None,
            split_role: None,
            split_group: None,
            moe_shard: None,
        };

        let json = serde_json::to_value(&peer).expect("serialization failed");
        assert_eq!(json["measured_tps_p50_by_model"]["Qwen3-32B-Q4_K_M"], 14.5);
        assert_eq!(
            json["measured_ttft_ms_p50_by_model"]["Qwen3-32B-Q4_K_M"],
            320
        );
    }

    #[test]
    fn test_local_instance_serializes_is_self() {
        let instance = LocalInstance {
            pid: 1234,
            api_port: Some(3131),
            version: Some("0.56.0".to_string()),
            started_at_unix: 1700000000,
            runtime_dir: "/home/user/.senda/runtime/1234".to_string(),
            is_self: true,
        };

        let json = serde_json::to_string(&instance).expect("serialization failed");
        assert!(json.contains("\"is_self\":true"));
    }
}
