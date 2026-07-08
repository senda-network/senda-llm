//! Pre-launch handshake probe for llama.cpp's RPC backend.
//!
//! Why this exists: when we launch `llama-server` in split mode with
//! `--rpc HOST:PORT,...`, llama.cpp parses every endpoint up front and
//! calls `negotiate_hello` on each one. If a single target is silent or
//! returns garbage during the 8-byte HELLO response, llama.cpp aborts
//! the *whole* process with `SIGABRT` from `RPC_STATUS_ASSERT` before
//! the HTTP server ever binds. The runtime then sees the launch fail,
//! decides we're still the host, and retries forever. See
//! `~/.senda/runtime/<pid>/logs/llama-server-*.log` for the
//! telltale `recv failed (bytes_recv=0, size_to_recv=8)` backtrace.
//!
//! This module performs the same handshake that `negotiate_hello`
//! would, but with a tight per-target deadline. The election loop
//! probes every tunnel port before invoking llama-server. If any
//! target fails the probe we abort the launch attempt, which lets the
//! host-attempt backoff (see `election.rs`) demote us to Worker so the
//! runner-up gets a chance, instead of looping on the same SIGABRT.
//!
//! Protocol constants are pinned to the patched llama.cpp shipped in
//! `.deps/llama.cpp` (RPC v4.x). They live here as plain literals
//! because the Rust runtime doesn't link against ggml-rpc — see
//! `third_party/llama.cpp/upstream.txt` for the pinned commit.

use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// `RPC_CMD_HELLO` — must match `ggml-rpc.cpp`'s
/// `static_assert(RPC_CMD_HELLO == 14)`.
const RPC_CMD_HELLO: u8 = 14;

/// `RPC_CONN_CAPS_SIZE` — see `transport.h`. The HELLO request /
/// response both carry a fixed-width capability blob of this many
/// bytes; we send a zero-filled blob (no RDMA, no special transport).
const RPC_CONN_CAPS_SIZE: usize = 24;

/// `sizeof(rpc_msg_hello_req)` — exactly `RPC_CONN_CAPS_SIZE` bytes.
const HELLO_REQ_BYTES: u64 = RPC_CONN_CAPS_SIZE as u64;

/// `sizeof(rpc_msg_hello_rsp)` — 4 bytes of version (major, minor,
/// patch, padding) followed by the conn_caps blob.
const HELLO_RSP_BYTES: u64 = 4 + RPC_CONN_CAPS_SIZE as u64;

/// Major protocol version we speak — see `ggml-rpc.h`.
/// Mismatched majors are unrecoverable; we treat them as probe
/// failures so llama-server doesn't abort post-launch.
const RPC_PROTO_MAJOR_VERSION: u8 = 4;

/// Default per-target deadline. 3 s is generous for an iroh tunnel
/// that's actually working (local relays are ~5 ms RTT, distant ones
/// rarely above 200 ms) and short enough that probing two stuck
/// tunnels still keeps the host-claim cycle under 10 s.
pub const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Result of probing a single RPC endpoint.
#[derive(Debug, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// Endpoint responded with a HELLO reply whose major version
    /// matches ours. Safe to hand to `llama-server --rpc`.
    Healthy,
    /// TCP connect failed.
    ConnectFailed,
    /// Connected but write/read timed out (deadlock or silent drop —
    /// the iroh-tunnel-without-bytes case from May 14 2026).
    Timeout,
    /// Read returned EOF before the full response — the remote
    /// rpc-server crashed or refused our HELLO.
    UnexpectedEof,
    /// Response framing was wrong (size header didn't match what the
    /// pinned llama.cpp expects).
    MalformedResponse,
    /// Server speaks a different major protocol version. Surfacing
    /// this distinctly makes mixed-version diagnosis easier.
    VersionMismatch { major: u8 },
}

impl ProbeOutcome {
    pub fn is_healthy(&self) -> bool {
        matches!(self, ProbeOutcome::Healthy)
    }
}

/// Performs the same client-side HELLO that `negotiate_hello` runs in
/// `llama.cpp`'s `ggml-rpc.cpp`, with a hard `timeout` for the whole
/// exchange.
///
/// Sends `[cmd=14][input_size=24 (LE u64)][24 zero bytes]` and expects
/// `[response_size=28 (LE u64)][major][minor][patch][_padding][24 caps
/// bytes]` back.
pub async fn probe_hello(addr: SocketAddr, timeout: Duration) -> ProbeOutcome {
    match tokio::time::timeout(timeout, probe_inner(addr)).await {
        Ok(outcome) => outcome,
        Err(_) => ProbeOutcome::Timeout,
    }
}

async fn probe_inner(addr: SocketAddr) -> ProbeOutcome {
    let mut sock = match TcpStream::connect(addr).await {
        Ok(s) => s,
        Err(_) => return ProbeOutcome::ConnectFailed,
    };

    // Request: cmd byte + input_size + input
    let mut req = Vec::with_capacity(1 + 8 + RPC_CONN_CAPS_SIZE);
    req.push(RPC_CMD_HELLO);
    req.extend_from_slice(&HELLO_REQ_BYTES.to_le_bytes());
    req.extend_from_slice(&[0u8; RPC_CONN_CAPS_SIZE]);

    if sock.write_all(&req).await.is_err() {
        return ProbeOutcome::UnexpectedEof;
    }

    // Response framing: 8-byte size followed by `size` bytes of payload.
    // The pinned rpc-server always replies with exactly HELLO_RSP_BYTES,
    // so anything else means we're talking to a non-llama.cpp endpoint
    // (or a tunnel that's silently dropping bytes).
    let mut size_buf = [0u8; 8];
    if sock.read_exact(&mut size_buf).await.is_err() {
        return ProbeOutcome::UnexpectedEof;
    }
    let rsp_size = u64::from_le_bytes(size_buf);
    if rsp_size != HELLO_RSP_BYTES {
        return ProbeOutcome::MalformedResponse;
    }

    let mut rsp_buf = [0u8; HELLO_RSP_BYTES as usize];
    if sock.read_exact(&mut rsp_buf).await.is_err() {
        return ProbeOutcome::UnexpectedEof;
    }
    let major = rsp_buf[0];
    if major != RPC_PROTO_MAJOR_VERSION {
        return ProbeOutcome::VersionMismatch { major };
    }
    ProbeOutcome::Healthy
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    /// Minimal in-process rpc-server stand-in. `behavior` decides
    /// what the fake server does after accepting the connection so
    /// each test can pin a specific failure mode.
    enum FakeBehavior {
        Healthy,
        SilentAccept,
        WrongMajor(u8),
        ShortResponse,
        SizeMismatch,
    }

    async fn spawn_fake(behavior: FakeBehavior) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = match listener.accept().await {
                Ok(p) => p,
                Err(_) => return,
            };
            // Read full HELLO request before responding so the client
            // can't race us. 1 + 8 + 24 = 33 bytes.
            let mut req = [0u8; 1 + 8 + RPC_CONN_CAPS_SIZE];
            if sock.read_exact(&mut req).await.is_err() {
                return;
            }
            match behavior {
                FakeBehavior::Healthy => {
                    let _ = sock.write_all(&HELLO_RSP_BYTES.to_le_bytes()).await;
                    let mut payload = [0u8; HELLO_RSP_BYTES as usize];
                    payload[0] = RPC_PROTO_MAJOR_VERSION;
                    let _ = sock.write_all(&payload).await;
                }
                FakeBehavior::SilentAccept => {
                    // Accept the bytes, then hold the socket open
                    // without writing — the iroh-tunnel-without-bytes
                    // case. `probe_hello` should hit its deadline.
                    tokio::time::sleep(Duration::from_secs(60)).await;
                }
                FakeBehavior::WrongMajor(major) => {
                    let _ = sock.write_all(&HELLO_RSP_BYTES.to_le_bytes()).await;
                    let mut payload = [0u8; HELLO_RSP_BYTES as usize];
                    payload[0] = major;
                    let _ = sock.write_all(&payload).await;
                }
                FakeBehavior::ShortResponse => {
                    let _ = sock.write_all(&HELLO_RSP_BYTES.to_le_bytes()).await;
                    // Send only 3 bytes of the 28-byte payload, then drop.
                    let _ = sock.write_all(&[RPC_PROTO_MAJOR_VERSION, 0, 0]).await;
                }
                FakeBehavior::SizeMismatch => {
                    // Advertise a totally different size — what an
                    // unpatched / older rpc-server would send.
                    let _ = sock.write_all(&7u64.to_le_bytes()).await;
                    let _ = sock.write_all(&[0u8; 7]).await;
                }
            }
        });
        addr
    }

    #[tokio::test]
    async fn healthy_server_returns_healthy() {
        let addr = spawn_fake(FakeBehavior::Healthy).await;
        let outcome = probe_hello(addr, Duration::from_secs(2)).await;
        assert_eq!(outcome, ProbeOutcome::Healthy);
        assert!(outcome.is_healthy());
    }

    #[tokio::test]
    async fn silent_tunnel_returns_timeout() {
        // The bug we're actually trying to catch: tunnel listener
        // accepts the connection, bytes go nowhere, no reply comes.
        // `probe_hello` must surface this as `Timeout` rather than
        // hanging the election loop indefinitely.
        let addr = spawn_fake(FakeBehavior::SilentAccept).await;
        let outcome = probe_hello(addr, Duration::from_millis(200)).await;
        assert_eq!(outcome, ProbeOutcome::Timeout);
        assert!(!outcome.is_healthy());
    }

    #[tokio::test]
    async fn unreachable_address_returns_connect_failed() {
        // 127.0.0.1:1 is virtually guaranteed to refuse on darwin/linux.
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let outcome = probe_hello(addr, Duration::from_millis(500)).await;
        assert!(
            matches!(outcome, ProbeOutcome::ConnectFailed | ProbeOutcome::Timeout),
            "expected connect failure, got {outcome:?}"
        );
        assert!(!outcome.is_healthy());
    }

    #[tokio::test]
    async fn major_mismatch_is_distinct() {
        let addr = spawn_fake(FakeBehavior::WrongMajor(3)).await;
        let outcome = probe_hello(addr, Duration::from_secs(2)).await;
        assert_eq!(outcome, ProbeOutcome::VersionMismatch { major: 3 });
    }

    #[tokio::test]
    async fn truncated_response_is_unexpected_eof() {
        let addr = spawn_fake(FakeBehavior::ShortResponse).await;
        let outcome = probe_hello(addr, Duration::from_secs(2)).await;
        assert_eq!(outcome, ProbeOutcome::UnexpectedEof);
    }

    #[tokio::test]
    async fn wrong_response_size_is_malformed() {
        let addr = spawn_fake(FakeBehavior::SizeMismatch).await;
        let outcome = probe_hello(addr, Duration::from_secs(2)).await;
        assert_eq!(outcome, ProbeOutcome::MalformedResponse);
    }
}
