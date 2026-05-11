#![recursion_limit = "256"]

mod api;
mod cli;
pub mod crypto;
mod inference;
mod mesh;
mod models;
mod network;
mod plugin;
mod plugins;
mod process_util;
mod protocol;
mod runtime;
mod system;

pub mod proto {
    pub mod node {
        include!(concat!(env!("OUT_DIR"), "/meshllm.node.v1.rs"));
    }
}

pub(crate) use plugins::blackboard;

use anyhow::Result;
use std::time::Duration;

pub const VERSION: &str = "0.66.15-pre1";

/// Migrate legacy data directories to `~/.closedmesh/`.
///
/// Runs once on startup. If `~/.closedmesh/` does not yet exist but a legacy
/// directory does, rename it in place so existing keys, models, and configs
/// survive the rebrand. The chain is `~/.closedmesh` -> `~/.forgemesh` ->
/// `~/.closedmesh`; we pick the newest legacy dir present.
fn migrate_legacy_dir() {
    let Some(home) = dirs::home_dir() else { return };
    let target = home.join(".closedmesh");
    if target.exists() {
        return;
    }
    for legacy in [".forgemesh", ".closedmesh"] {
        let candidate = home.join(legacy);
        if candidate.exists() {
            let _ = std::fs::rename(&candidate, &target);
            return;
        }
    }
}

pub async fn run() -> Result<()> {
    migrate_legacy_dir();
    runtime::run().await
}

pub async fn run_main() -> i32 {
    match run().await {
        Ok(()) => 0,
        Err(err) => {
            let _ = cli::output::emit_fatal_error(&err);
            tokio::time::sleep(Duration::from_millis(50)).await;
            1
        }
    }
}
