use crate::cli::output::{emit_event, OutputEvent};
use crate::mesh::Node;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

const FAILURE_WINDOW: Duration = Duration::from_secs(60);
const REFRESH_BACKOFF: Duration = Duration::from_secs(30);
const FAILURE_THRESHOLD: u32 = 3;

static RECOVERY: LazyLock<Mutex<EntryTunnelHealth>> =
    LazyLock::new(|| Mutex::new(EntryTunnelHealth::default()));

#[derive(Debug)]
pub(crate) struct EntryTunnelHealth {
    join_url: Option<String>,
    consecutive_failures: u32,
    first_failure_at: Option<Instant>,
    last_refresh_at: Option<Instant>,
}

impl Default for EntryTunnelHealth {
    fn default() -> Self {
        Self {
            join_url: None,
            consecutive_failures: 0,
            first_failure_at: None,
            last_refresh_at: None,
        }
    }
}

impl EntryTunnelHealth {
    fn set_join_url(&mut self, join_url: Option<String>) {
        self.join_url = join_url;
        self.consecutive_failures = 0;
        self.first_failure_at = None;
        self.last_refresh_at = None;
    }

    fn observe_failure(&mut self, now: Instant) -> Option<String> {
        let join_url = self.join_url.clone()?;
        if self
            .first_failure_at
            .is_none_or(|first| now.duration_since(first) > FAILURE_WINDOW)
        {
            self.first_failure_at = Some(now);
            self.consecutive_failures = 0;
        }
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        if self.consecutive_failures < FAILURE_THRESHOLD {
            return None;
        }
        if self
            .last_refresh_at
            .is_some_and(|last| now.duration_since(last) < REFRESH_BACKOFF)
        {
            return None;
        }
        self.last_refresh_at = Some(now);
        self.consecutive_failures = 0;
        self.first_failure_at = None;
        Some(join_url)
    }

    fn observe_success(&mut self) {
        self.consecutive_failures = 0;
        self.first_failure_at = None;
    }
}

pub(crate) fn configure_join_url(join_url: Option<String>) {
    if let Ok(mut state) = RECOVERY.lock() {
        state.set_join_url(join_url);
    }
}

pub(crate) fn record_tunnel_success() {
    if let Ok(mut state) = RECOVERY.lock() {
        state.observe_success();
    }
}

pub(crate) fn record_tunnel_failure(node: Node) {
    let join_url = {
        let Ok(mut state) = RECOVERY.lock() else {
            return;
        };
        state.observe_failure(Instant::now())
    };
    let Some(join_url) = join_url else {
        return;
    };

    tokio::spawn(async move {
        let _ = emit_event(OutputEvent::Warning {
            message: format!(
                "Multiple mesh tunnel failures detected; refreshing entry join token from {join_url}"
            ),
            context: Some("entry-recovery".into()),
        });
        match super::fetch_join_url_token(&join_url).await {
            Ok(token) => {
                if let Err(e) = node.join(&token).await {
                    tracing::warn!("entry recovery rejoin failed after token refresh: {e}");
                } else {
                    tracing::info!("entry recovery refreshed join token and rejoined");
                }
            }
            Err(e) => {
                tracing::warn!("entry recovery failed to refresh join-url {join_url}: {e:#}");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn third_failure_within_window_triggers_refresh() {
        let mut health = EntryTunnelHealth::default();
        health.set_join_url(Some("https://mesh.example.test/api/status".into()));
        let t0 = Instant::now();
        assert!(health.observe_failure(t0).is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(10))
            .is_none());
        assert_eq!(
            health.observe_failure(t0 + Duration::from_secs(20)),
            Some("https://mesh.example.test/api/status".to_string())
        );
    }

    #[test]
    fn refresh_is_rate_limited() {
        let mut health = EntryTunnelHealth::default();
        health.set_join_url(Some("https://mesh.example.test/api/status".into()));
        let t0 = Instant::now();
        assert!(health.observe_failure(t0).is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(1))
            .is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(2))
            .is_some());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(3))
            .is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(4))
            .is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(5))
            .is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(30))
            .is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(31))
            .is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(32))
            .is_some());
    }

    #[test]
    fn success_resets_failure_counter() {
        let mut health = EntryTunnelHealth::default();
        health.set_join_url(Some("https://mesh.example.test/api/status".into()));
        let t0 = Instant::now();
        assert!(health.observe_failure(t0).is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(1))
            .is_none());
        health.observe_success();
        assert!(health
            .observe_failure(t0 + Duration::from_secs(2))
            .is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(3))
            .is_none());
        assert!(health
            .observe_failure(t0 + Duration::from_secs(4))
            .is_some());
    }
}
