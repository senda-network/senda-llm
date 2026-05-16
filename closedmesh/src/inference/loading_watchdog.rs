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
    NoHostElected,
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
            Self::NoHostElected => format!(
                "`{model}` has been loading too long: no peer has claimed host. Check peer connectivity and host election logs."
            ),
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
    pub serving_models: Vec<String>,
    pub hosted_models: Vec<String>,
}

impl From<&PeerInfo> for LoadingPeerSnapshot {
    fn from(peer: &PeerInfo) -> Self {
        Self {
            role: peer.role.clone(),
            fast_memory_bytes: peer.fast_memory_bytes(),
            serving_models: peer.serving_models.clone(),
            hosted_models: peer.hosted_models.clone(),
        }
    }
}

pub(crate) fn diagnose_stuck_loading(
    model: &str,
    model_bytes: Option<u64>,
    local_fast_bytes: u64,
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

    let total_peer_fast_bytes: u64 = peers
        .iter()
        .filter(|p| p.serving_models.iter().any(|m| m == model))
        .map(|p| p.fast_memory_bytes)
        .sum();

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
        if needed > local_fast_bytes {
            return StuckReason::WaitingForSplitPartner {
                model_bytes,
                total_peer_fast_bytes,
            };
        }
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
            serving_models: serving.iter().map(|s| s.to_string()).collect(),
            hosted_models: hosted.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn diagnoses_model_too_large_for_solo_and_no_partner() {
        let reason = diagnose_stuck_loading(
            "Qwen3-32B-Q4_K_M",
            Some(20 * 1024 * 1024 * 1024),
            8 * 1024 * 1024 * 1024,
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
}
