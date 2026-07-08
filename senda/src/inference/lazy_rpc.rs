//! On-demand ("lazy") `rpc-server` lifecycle.
//!
//! A solo node never joins a pipeline split, so an always-on `rpc-server`
//! only pins an idle CUDA context + ~1.5 GB VRAM for work that never happens
//! (proven non-perf — see `internal/RESILIENCE.md`, v0.66.69/v0.66.70). This
//! type brings the process up *only* on the pipeline-worker edge and tears it
//! down once it goes idle.
//!
//! Two correctness constraints shape the design:
//!
//! 1. **HELLO-blacklist (v0.66.36):** a split host blacklists a worker for the
//!    launch attempt if its RPC probe fails. So a worker must have its
//!    `rpc-server` *listening* before any host dials it. We solve this by
//!    gossiping an `rpc_ready` bit (flipped here via [`Node::set_local_rpc_ready`])
//!    so a host only ever dials a warm worker.
//! 2. **Refcounting:** several per-model election loops share one node-wide
//!    `rpc-server`. Each pipeline-worker edge takes a [`WorkerRpcHold`]; the
//!    process stays up while any hold is live and is eligible for idle teardown
//!    only once every hold — across all model loops — is dropped.
//!
//! The tunnel's inbound RPC handler is repointed via
//! [`tunnel::Manager::set_rpc_port`] as the server comes up (real port) / goes
//! down (port 0, which makes the handler drop RPC streams while cold).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::Mutex;

use crate::inference::launch::{self, BinaryFlavor, RpcServerHandle};
use crate::mesh::Node;
use crate::network::tunnel;
use crate::runtime::instance::InstanceRuntime;

/// How long the `rpc-server` may sit with zero holders before idle teardown.
/// Checked by the runtime's 30 s tick loop, so the effective teardown latency
/// is this value rounded up to the next tick.
pub const RPC_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Everything needed to (re)spawn the `rpc-server` on demand. Captured once at
/// construction so [`LazyRpcServer::acquire`] can spawn without the caller
/// re-threading runtime/bin/device context.
pub struct LazyRpcParams {
    pub runtime: Arc<InstanceRuntime>,
    pub bin_dir: PathBuf,
    pub flavor: Option<BinaryFlavor>,
    pub device: Option<String>,
    pub gguf: Option<PathBuf>,
}

struct Inner {
    handle: Option<RpcServerHandle>,
    /// Live [`WorkerRpcHold`]s across all per-model election loops.
    holders: usize,
    /// When `holders` last hit zero; `None` while held. Drives idle teardown.
    idle_since: Option<Instant>,
}

/// Refcounted, idle-timed owner of the node's single on-demand `rpc-server`.
/// Construct once per process; share the `Arc` to every election loop.
pub struct LazyRpcServer {
    params: LazyRpcParams,
    tunnel: tunnel::Manager,
    node: Node,
    idle_timeout: Duration,
    inner: Mutex<Inner>,
}

impl LazyRpcServer {
    pub fn new(
        params: LazyRpcParams,
        tunnel: tunnel::Manager,
        node: Node,
        idle_timeout: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            params,
            tunnel,
            node,
            idle_timeout,
            inner: Mutex::new(Inner {
                handle: None,
                holders: 0,
                idle_since: None,
            }),
        })
    }

    /// Take a hold, spawning the `rpc-server` if this is the first one. The
    /// lock is held across the spawn so two concurrent acquires can't
    /// double-spawn; on spawn failure the holder count is left untouched (the
    /// `?` early-returns before the increment), so a failed acquire can't
    /// wedge idle teardown.
    async fn acquire(self: &Arc<Self>) -> Result<()> {
        let mut inner = self.inner.lock().await;
        let mut just_spawned = false;
        if inner.handle.is_none() {
            let handle = launch::start_rpc_server(
                &self.params.runtime,
                &self.params.bin_dir,
                self.params.flavor,
                self.params.device.as_deref(),
                self.params.gguf.as_deref(),
            )
            .await?;
            // Extract the port before moving the handle into `inner` to keep the
            // immutable borrow from overlapping the mutable store (E0502).
            let port = handle.port;
            inner.handle = Some(handle);
            self.tunnel.set_rpc_port(port);
            just_spawned = true;
        }
        inner.holders += 1;
        inner.idle_since = None;
        drop(inner);
        // Gossip readiness only on the cold→warm transition, and outside the
        // lock (it triggers a regossip we don't want to serialize under it).
        if just_spawned {
            self.node.set_local_rpc_ready(true).await;
        }
        Ok(())
    }

    /// Drop one hold. When the count reaches zero, start the idle clock; the
    /// process is not torn down here — [`tick`](Self::tick) does that once the
    /// idle timeout elapses, so a brief worker→host→worker flap doesn't pay a
    /// cold respawn.
    async fn release(self: &Arc<Self>) {
        let mut inner = self.inner.lock().await;
        if inner.holders > 0 {
            inner.holders -= 1;
        }
        if inner.holders == 0 {
            inner.idle_since = Some(Instant::now());
        }
    }

    /// Idle-teardown poll. Tear the `rpc-server` down once it has had zero
    /// holders for longer than `idle_timeout`. Safe to call on a fixed
    /// interval regardless of state.
    pub async fn tick(self: &Arc<Self>) {
        let mut inner = self.inner.lock().await;
        if inner.holders != 0 {
            return;
        }
        let idle_long_enough = inner
            .idle_since
            .map(|since| since.elapsed() >= self.idle_timeout)
            .unwrap_or(false);
        if !idle_long_enough {
            return;
        }
        let Some(handle) = inner.handle.take() else {
            return;
        };
        inner.idle_since = None;
        drop(inner);
        self.tunnel.set_rpc_port(0);
        self.node.set_local_rpc_ready(false).await;
        handle.shutdown().await;
    }

    /// Unconditional teardown for process shutdown. Drops any holders and the
    /// running `rpc-server` immediately.
    pub async fn shutdown(self: &Arc<Self>) {
        let mut inner = self.inner.lock().await;
        inner.holders = 0;
        inner.idle_since = None;
        let handle = inner.handle.take();
        drop(inner);
        if let Some(handle) = handle {
            self.tunnel.set_rpc_port(0);
            self.node.set_local_rpc_ready(false).await;
            handle.shutdown().await;
        }
    }

    /// [`acquire`](Self::acquire) paired with an RAII guard that releases on
    /// drop. This is the safe form for the election loop, whose many early
    /// `return` paths (stop, peer-death, watchdog, model unload) would
    /// otherwise leak a hold.
    pub async fn acquire_hold(self: &Arc<Self>) -> Result<WorkerRpcHold> {
        self.acquire().await?;
        Ok(WorkerRpcHold {
            server: self.clone(),
        })
    }
}

/// RAII hold returned by [`LazyRpcServer::acquire_hold`]. Dropping it releases
/// the hold (best-effort async release scheduled on the current runtime), so
/// the `rpc-server` becomes eligible for idle teardown once every holder —
/// across all per-model election loops — is gone.
pub struct WorkerRpcHold {
    server: Arc<LazyRpcServer>,
}

impl Drop for WorkerRpcHold {
    fn drop(&mut self) {
        let server = self.server.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move { server.release().await });
        }
    }
}
