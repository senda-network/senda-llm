use crate::cli::output::{emit_event, OutputEvent};
use crate::mesh::{Node, NodeRole, PeerInfo};
use std::path::PathBuf;
use std::time::Duration;

const REEMIT_AFTER: Duration = Duration::from_secs(15 * 60);
const POLL_AFTER_READY_CHECK: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StuckReason {
    ModelTooLargeForSoloAndNoPartner {
        model_bytes: u64,
        local_fast_bytes: u64,
        total_peer_fast_bytes: u64,
    },
    WaitingForSplitPartner {
        model_bytes: Option<u64>,
        total_peer_fast_bytes: u64,
    },
    /// Every otherwise-viable host candidate failed the RAM-aware filter
    /// (predicted host-peak RAM > 0.75 * system_ram_bytes). Surfaced so
    /// operators can see "your peers have GPUs but not enough RAM to
    /// hold even the local-share weights" instead of a generic
    /// `NoHostElected`. Added in v0.66.38 alongside `ram_can_host_model`.
    NoHostCandidateFitsInRam {
        model_bytes: u64,
        peer_count: usize,
    },
    NoHostElected,
    /// Cohort includes peers below [`election::MIN_PIPELINE_ELECTION_PEER_VERSION`].
    OutdatedPeersBlockingElection {
        min_version: String,
        peers: Vec<(String, String)>,
    },
    LlamaServerStartFailed {
        detail: String,
    },
}

impl StuckReason {
    pub(crate) fn message(&self, model: &str) -> String {
        match self {
            Self::ModelTooLargeForSoloAndNoPartner {
                model_bytes,
                local_fast_bytes,
                total_peer_fast_bytes,
            } => format!(
                "`{model}` has been loading too long: model is {:.1} GB, local fast memory is {:.1} GB, and visible split partners contribute {:.1} GB. No eligible split cohort is ready.",
                *model_bytes as f64 / 1e9,
                *local_fast_bytes as f64 / 1e9,
                *total_peer_fast_bytes as f64 / 1e9,
            ),
            Self::WaitingForSplitPartner {
                model_bytes,
                total_peer_fast_bytes,
            } => format!(
                "`{model}` has been loading too long: waiting for a split partner/cohort. Model size: {}; visible peer fast memory: {:.1} GB.",
                model_bytes
                    .map(|b| format!("{:.1} GB", b as f64 / 1e9))
                    .unwrap_or_else(|| "unknown".to_string()),
                *total_peer_fast_bytes as f64 / 1e9,
            ),
            Self::NoHostCandidateFitsInRam {
                model_bytes,
                peer_count,
            } => format!(
                "`{model}` has been loading too long: none of the {peer_count} candidate peers has enough system RAM to host this model ({:.1} GB). Free RAM on at least one peer or split across more machines.",
                *model_bytes as f64 / 1e9,
            ),
            Self::NoHostElected => format!(
                "`{model}` has been loading too long: no peer has claimed host. Check peer connectivity and host election logs."
            ),
            Self::OutdatedPeersBlockingElection { min_version, peers } => {
                let list = peers
                    .iter()
                    .map(|(host, ver)| format!("{host} (v{ver})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "`{model}` has been loading too long: outdated peer(s) in the cohort block pipeline election (need senda >= {min_version}): {list}. Upgrade those machines or wait for auto-update; current runtimes ignore them for host picks."
                )
            }
            Self::LlamaServerStartFailed { detail } => format!(
                "`{model}` has been loading too long: llama-server startup appears to be failing repeatedly. {detail}"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LoadingPeerSnapshot {
    pub role: NodeRole,
    pub fast_memory_bytes: u64,
    pub system_ram_bytes: u64,
    pub serving_models: Vec<String>,
    pub hosted_models: Vec<String>,
    pub version: Option<String>,
    pub hostname: Option<String>,
}

impl From<&PeerInfo> for LoadingPeerSnapshot {
    fn from(peer: &PeerInfo) -> Self {
        Self {
            role: peer.role.clone(),
            fast_memory_bytes: peer.fast_memory_bytes(),
            system_ram_bytes: peer.system_ram_bytes,
            serving_models: peer.serving_models.clone(),
            hosted_models: peer.hosted_models.clone(),
            version: peer.version.clone(),
            hostname: peer.hostname.clone(),
        }
    }
}

pub(crate) fn diagnose_stuck_loading(
    model: &str,
    model_bytes: Option<u64>,
    local_fast_bytes: u64,
    local_system_ram_bytes: u64,
    peers: &[LoadingPeerSnapshot],
    llama_start_failure: Option<String>,
) -> StuckReason {
    if let Some(detail) = llama_start_failure {
        return StuckReason::LlamaServerStartFailed { detail };
    }

    if peers.iter().any(|p| {
        matches!(p.role, NodeRole::Host { .. }) && p.hosted_models.iter().any(|m| m == model)
    }) {
        let total_peer_fast_bytes = peers.iter().map(|p| p.fast_memory_bytes).sum();
        return StuckReason::WaitingForSplitPartner {
            model_bytes,
            total_peer_fast_bytes,
        };
    }

    let cohort_peers: Vec<&LoadingPeerSnapshot> = peers
        .iter()
        .filter(|p| p.serving_models.iter().any(|m| m == model))
        .collect();
    let total_peer_fast_bytes: u64 = cohort_peers.iter().map(|p| p.fast_memory_bytes).sum();

    if let Some(bytes) = model_bytes {
        let needed = (bytes as f64 * 1.1) as u64;
        if needed > local_fast_bytes
            && local_fast_bytes.saturating_add(total_peer_fast_bytes) < needed
        {
            return StuckReason::ModelTooLargeForSoloAndNoPartner {
                model_bytes: bytes,
                local_fast_bytes,
                total_peer_fast_bytes,
            };
        }
        // RAM-aware diagnostic: if every cohort peer (including self) has
        // VRAM but not enough system RAM to host its predicted share, the
        // split will OOM the host. Surface that explicitly rather than
        // falling through to the generic NoHostElected / WaitingForSplit.
        // Self only contributes if its system_ram is known and gpu vram
        // is enough to participate.
        let cohort_size = cohort_peers.len() + 1;
        let any_peer_fits_ram = cohort_peers
            .iter()
            .filter(|p| p.system_ram_bytes > 0)
            .any(|p| {
                let predicted =
                    crate::inference::election::predicted_host_ram_bytes(bytes, cohort_size);
                predicted <= (p.system_ram_bytes as f64 * 0.75) as u64
            });
        let local_fits_ram = local_system_ram_bytes > 0 && {
            let predicted =
                crate::inference::election::predicted_host_ram_bytes(bytes, cohort_size);
            predicted <= (local_system_ram_bytes as f64 * 0.75) as u64
        };
        let any_known_ram =
            local_system_ram_bytes > 0 || cohort_peers.iter().any(|p| p.system_ram_bytes > 0);
        if any_known_ram && !any_peer_fits_ram && !local_fits_ram {
            return StuckReason::NoHostCandidateFitsInRam {
                model_bytes: bytes,
                peer_count: cohort_peers.len(),
            };
        }
        if needed > local_fast_bytes {
            return StuckReason::WaitingForSplitPartner {
                model_bytes,
                total_peer_fast_bytes,
            };
        }
    }

    let outdated: Vec<(String, String)> = cohort_peers
        .iter()
        .filter(|p| {
            !crate::inference::election::peer_supports_pipeline_election(p.version.as_deref())
        })
        .map(|p| {
            (
                p.hostname.clone().unwrap_or_else(|| "peer".to_string()),
                p.version.clone().unwrap_or_else(|| "unknown".to_string()),
            )
        })
        .collect();
    if !outdated.is_empty() {
        return StuckReason::OutdatedPeersBlockingElection {
            min_version: crate::inference::election::MIN_PIPELINE_ELECTION_PEER_VERSION.to_string(),
            peers: outdated,
        };
    }

    StuckReason::NoHostElected
}

pub(crate) fn spawn_loading_watchdog(
    node: Node,
    model_name: String,
    model_path: PathBuf,
    grace: Duration,
) {
    tokio::spawn(async move {
        tokio::time::sleep(grace).await;
        let mut last_emitted: Option<(StuckReason, std::time::Instant)> = None;
        loop {
            if node
                .models_being_served_routable()
                .await
                .iter()
                .any(|m| m == &model_name)
            {
                return;
            }

            let peers: Vec<LoadingPeerSnapshot> = node
                .peers()
                .await
                .iter()
                .map(LoadingPeerSnapshot::from)
                .collect();
            let reason = diagnose_stuck_loading(
                &model_name,
                crate::inference::election::try_total_model_bytes(&model_path),
                node.fast_memory_bytes(),
                node.system_ram_bytes,
                &peers,
                None,
            );

            let should_emit = last_emitted
                .as_ref()
                .map(|(prev, when)| prev != &reason || when.elapsed() >= REEMIT_AFTER)
                .unwrap_or(true);

            if should_emit {
                let _ = emit_event(OutputEvent::Error {
                    message: reason.message(&model_name),
                    context: Some(format!("model={model_name}")),
                });
                tracing::error!(
                    model = %model_name,
                    reason = ?reason,
                    "model remained non-routable past loading watchdog grace"
                );
                last_emitted = Some((reason, std::time::Instant::now()));
            }

            tokio::time::sleep(POLL_AFTER_READY_CHECK).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(
        role: NodeRole,
        fast_gb: u64,
        serving: &[&str],
        hosted: &[&str],
    ) -> LoadingPeerSnapshot {
        LoadingPeerSnapshot {
            role,
            fast_memory_bytes: fast_gb * 1024 * 1024 * 1024,
            system_ram_bytes: 0,
            serving_models: serving.iter().map(|s| s.to_string()).collect(),
            hosted_models: hosted.iter().map(|s| s.to_string()).collect(),
            version: Some(crate::VERSION.to_string()),
            hostname: None,
        }
    }

    fn peer_with_ram(
        role: NodeRole,
        fast_gb: u64,
        ram_gb: u64,
        serving: &[&str],
        hosted: &[&str],
    ) -> LoadingPeerSnapshot {
        let mut p = peer(role, fast_gb, serving, hosted);
        p.system_ram_bytes = ram_gb * 1024 * 1024 * 1024;
        p
    }

    #[test]
    fn diagnoses_model_too_large_for_solo_and_no_partner() {
        let reason = diagnose_stuck_loading(
            "Qwen3-32B-Q4_K_M",
            Some(20 * 1024 * 1024 * 1024),
            8 * 1024 * 1024 * 1024,
            0,
            &[],
            None,
        );
        assert!(matches!(
            reason,
            StuckReason::ModelTooLargeForSoloAndNoPartner { .. }
        ));
    }

    #[test]
    fn diagnoses_waiting_for_split_partner_when_pool_could_fit() {
        let model = "Qwen3-32B-Q4_K_M";
        let reason = diagnose_stuck_loading(
            model,
            Some(20 * 1024 * 1024 * 1024),
            8 * 1024 * 1024 * 1024,
            0,
            &[peer(NodeRole::Worker, 24, &[model], &[])],
            None,
        );
        assert!(matches!(reason, StuckReason::WaitingForSplitPartner { .. }));
    }

    #[test]
    fn diagnoses_no_host_elected_when_model_should_fit_locally() {
        let reason = diagnose_stuck_loading(
            "Qwen3-8B-Q4_K_M",
            Some(5 * 1024 * 1024 * 1024),
            16 * 1024 * 1024 * 1024,
            0,
            &[],
            None,
        );
        assert_eq!(reason, StuckReason::NoHostElected);
    }

    #[test]
    fn diagnoses_llama_start_failure_first() {
        let reason = diagnose_stuck_loading(
            "Qwen3-8B-Q4_K_M",
            Some(5 * 1024 * 1024 * 1024),
            16 * 1024 * 1024 * 1024,
            0,
            &[],
            Some("last child exited with code 1".to_string()),
        );
        assert_eq!(
            reason,
            StuckReason::LlamaServerStartFailed {
                detail: "last child exited with code 1".to_string()
            }
        );
    }

    /// 20 GB model split across 4 peers ⇒ each peer's local share is 5 GB,
    /// predicted host RAM commit is `3 GB + 5/2 = 5.5 GB`. With 16 GB Macs
    /// (12 GB budget) every peer fits — should NOT trip the RAM filter.
    #[test]
    fn ram_filter_does_not_trip_when_cohort_can_split_thinly() {
        let model = "Qwen3-32B-Q4_K_M";
        let reason = diagnose_stuck_loading(
            model,
            Some(20 * 1024 * 1024 * 1024),
            8 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
            &[
                peer_with_ram(NodeRole::Worker, 8, 16, &[model], &[]),
                peer_with_ram(NodeRole::Worker, 8, 16, &[model], &[]),
                peer_with_ram(NodeRole::Worker, 8, 16, &[model], &[]),
            ],
            None,
        );
        assert!(
            !matches!(reason, StuckReason::NoHostCandidateFitsInRam { .. }),
            "20 GB model in a 4-peer cohort with 16 GB Macs should not trip the RAM filter; got {:?}",
            reason
        );
    }

    /// 20 GB model split across 2 peers each with 4 GB RAM ⇒ predicted
    /// commit `3 GB + 5 GB = 8 GB` per host, budget `4 * 0.75 = 3 GB`. The
    /// cohort has plenty of VRAM (8 + 24 = 32 GB > 22 GB needed) so the
    /// "model too large" guard passes — but both peers fail the RAM
    /// budget and we surface NoHostCandidateFitsInRam.
    #[test]
    fn diagnoses_no_host_candidate_fits_in_ram_for_huge_model_on_tiny_ram_peers() {
        let model = "Qwen3-32B-Q4_K_M";
        let reason = diagnose_stuck_loading(
            model,
            Some(20 * 1024 * 1024 * 1024),
            8 * 1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
            &[peer_with_ram(NodeRole::Worker, 24, 4, &[model], &[])],
            None,
        );
        assert!(
            matches!(
                reason,
                StuckReason::NoHostCandidateFitsInRam {
                    model_bytes,
                    peer_count: 1,
                } if model_bytes == 20 * 1024 * 1024 * 1024
            ),
            "got {:?}",
            reason
        );
    }
}
