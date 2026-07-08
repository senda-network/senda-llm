//! Gossip protocol: peer announcement exchange, transitive peer tracking,
//! and peer list management (add/remove/update).

use super::*;

pub fn backfill_legacy_descriptors(ann: &mut PeerAnnouncement) {
    if ann.served_model_descriptors.is_empty() {
        let primary_model_name = ann
            .serving_models
            .first()
            .map(String::as_str)
            .unwrap_or_default()
            .to_string();
        ann.served_model_descriptors = infer_remote_served_descriptors(
            &primary_model_name,
            &ann.serving_models,
            ann.model_source.as_deref(),
        );
    }
}

pub(super) fn peer_meaningfully_changed(old: &PeerInfo, new: &PeerInfo) -> bool {
    old.addr != new.addr
        || old.role != new.role
        || old.first_joined_mesh_ts != new.first_joined_mesh_ts
        || old.models != new.models
        || old.vram_bytes != new.vram_bytes
        || old.rtt_ms != new.rtt_ms
        || old.model_source != new.model_source
        || old.serving_models != new.serving_models
        || old.hosted_models_known != new.hosted_models_known
        || old.hosted_models != new.hosted_models
        || old.available_models != new.available_models
        || old.requested_models != new.requested_models
        || old.served_model_descriptors != new.served_model_descriptors
        || old.served_model_runtime != new.served_model_runtime
        || old.version != new.version
        || old.owner_summary != new.owner_summary
        || old.gpu_reserved_bytes != new.gpu_reserved_bytes
        || old.rpc_ready != new.rpc_ready
}

fn merge_first_joined_mesh_ts(existing: &mut Option<u64>, incoming: Option<u64>) {
    match (*existing, incoming) {
        (None, Some(v)) => *existing = Some(v),
        (Some(_), None) => {}
        (Some(a), Some(b)) => *existing = Some(a.min(b)),
        (None, None) => {}
    }
}

pub(super) fn apply_transitive_ann(
    existing: &mut PeerInfo,
    addr: &EndpointAddr,
    ann: &PeerAnnouncement,
) -> bool {
    let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
    let serving_changed = existing.serving_models != ann.serving_models
        || existing.hosted_models != ann_hosted_models
        || existing.hosted_models_known != ann.hosted_models.is_some();
    existing.serving_models = ann.serving_models.clone();
    existing.hosted_models = ann_hosted_models;
    existing.hosted_models_known = ann.hosted_models.is_some();
    existing.role = ann.role.clone();
    merge_first_joined_mesh_ts(&mut existing.first_joined_mesh_ts, ann.first_joined_mesh_ts);
    existing.vram_bytes = ann.vram_bytes;
    // Only advance addr if the transitive announcement is at least as path-rich,
    // so a direct peer's richer address is not overwritten by a weaker transitive one.
    if !addr.addrs.is_empty() && addr.addrs.len() >= existing.addr.addrs.len() {
        existing.addr = addr.clone();
    }
    if ann.version.is_some() {
        existing.version = ann.version.clone();
    }
    if ann.gpu_name.is_some() {
        existing.gpu_name = ann.gpu_name.clone();
    }
    if ann.hostname.is_some() {
        existing.hostname = ann.hostname.clone();
    }
    if ann.is_soc.is_some() {
        existing.is_soc = ann.is_soc;
    }
    if ann.gpu_vram.is_some() {
        existing.gpu_vram = ann.gpu_vram.clone();
    }
    if ann.gpu_reserved_bytes.is_some() {
        existing.gpu_reserved_bytes = ann.gpu_reserved_bytes.clone();
    }
    if ann.gpu_mem_bandwidth_gbps.is_some() {
        existing.gpu_mem_bandwidth_gbps = ann.gpu_mem_bandwidth_gbps.clone();
    }
    if ann.gpu_compute_tflops_fp32.is_some() {
        existing.gpu_compute_tflops_fp32 = ann.gpu_compute_tflops_fp32.clone();
    }
    if ann.gpu_compute_tflops_fp16.is_some() {
        existing.gpu_compute_tflops_fp16 = ann.gpu_compute_tflops_fp16.clone();
    }
    if let Some(cap) = ann.capability.clone() {
        existing.capability = cap;
    } else {
        // Older peer didn't advertise capability — re-derive from whatever
        // legacy GPU fields we have so the router still has something useful.
        existing.capability = backfill_capability_from_legacy(
            existing.gpu_name.as_deref(),
            existing.gpu_vram.as_deref(),
            existing.is_soc,
            &existing.serving_models,
        );
    }
    existing.models = ann.models.clone();
    existing.available_models.clear();
    existing.requested_models = ann.requested_models.clone();
    existing.owner_attestation = ann.owner_attestation.clone();
    existing.inflight_requests = ann.inflight_requests;
    // rpc_ready: owner-authoritative + dynamic. Guard against a legacy relay
    // (which can't see the field, so sends None) stomping a known value — same
    // discipline as the v0.66.50 empty-transitive-ann fix.
    if ann.rpc_ready.is_some() {
        existing.rpc_ready = ann.rpc_ready;
    }
    if ann.model_source.is_some() {
        existing.model_source = ann.model_source.clone();
    }
    existing.served_model_descriptors = ann.served_model_descriptors.clone();
    existing.served_model_runtime = ann.served_model_runtime.clone();
    if ann.experts_summary.is_some() {
        existing.experts_summary = ann.experts_summary.clone();
    }
    // v0.66.50: refresh measured per-model TPS/TTFT for transitively-learned
    // peers, but DO NOT overwrite a non-empty existing value with an empty
    // ann. In a mixed-version mesh the May 20 2026 incident showed that a
    // legacy v0.66.48 worker's transitive view of a v0.66.49 host has
    // `model_timings: []` and `native_baselines: []` (default empty for an
    // unknown proto field). When that legacy worker gossips its world to
    // the entry, an unconditional `existing.X = ann.X.clone()` blanks the
    // direct-gossip data the host published moments earlier. The end-state
    // flapped per gossip tick: catalog rendered the efficiency ratio for
    // ~one cycle, then dropped it again.
    //
    // The "metrics from a host learned through a relay would never reach
    // the entry" concern that motivated the unconditional overwrite still
    // works: as long as some peer in the chain has the data populated, the
    // refresh below applies. We only refuse to overwrite *full* with
    // *empty* — full→full and empty→full both still work.
    if !ann.model_timings.is_empty() || existing.model_timings.is_empty() {
        existing.model_timings = ann.model_timings.clone();
    }
    if !ann.native_baselines.is_empty() || existing.native_baselines.is_empty() {
        existing.native_baselines = ann.native_baselines.clone();
    }
    serving_changed
}

impl Node {
    /// Open a gossip stream on an existing connection to exchange peer info.
    pub(super) async fn initiate_gossip(&self, conn: Connection, remote: EndpointId) -> Result<()> {
        // Timeout only the gossip round-trip. A misbehaving peer may accept the
        // QUIC connection and even the bi-stream but never send a gossip response,
        // blocking the join path indefinitely and preventing fallback to other
        // candidates.
        match tokio::time::timeout(
            PEER_CONNECT_AND_GOSSIP_TIMEOUT,
            self.gossip_round_trip(&conn, remote),
        )
        .await
        {
            Ok(Ok((their_announcements, rtt_ms))) => {
                self.apply_gossip_announcements(remote, rtt_ms, &their_announcements, true)
                    .await
            }
            Ok(Err(e)) => Err(e),
            Err(_) => anyhow::bail!(
                "gossip exchange with {} timed out ({}s)",
                remote.fmt_short(),
                PEER_CONNECT_AND_GOSSIP_TIMEOUT.as_secs()
            ),
        }
    }

    pub(super) async fn initiate_gossip_inner(
        &self,
        conn: Connection,
        remote: EndpointId,
        discover_peers: bool,
    ) -> Result<()> {
        let (their_announcements, rtt_ms) = self.gossip_round_trip(&conn, remote).await?;
        self.apply_gossip_announcements(remote, rtt_ms, &their_announcements, discover_peers)
            .await
    }

    async fn gossip_round_trip(
        &self,
        conn: &Connection,
        remote: EndpointId,
    ) -> Result<(Vec<(EndpointAddr, PeerAnnouncement)>, u32)> {
        let protocol = connection_protocol(conn);
        let t0 = std::time::Instant::now();
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_GOSSIP]).await?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let buf = read_len_prefixed(&mut recv).await?;
        // Measure the round-trip AFTER the response has been read. Capturing
        // it earlier (between t0 and send.finish()) was timing only the
        // local QUIC send buffer, which is microseconds on most paths and
        // rounds to 0 ms when cast to u32 — feeding `update_peer_rtt` the
        // spurious "0 ms" value that flipped the split-eligibility gate
        // back on every gossip cycle and produced the
        // `RTT improved (140ms → 0ms) — re-electing for split` log storm
        // in the entry-node journal. Reading the response first means
        // `t0.elapsed()` includes the actual peer-side processing + return
        // trip, which is what the split RTT threshold is meant to gate on.
        let rtt_ms = t0.elapsed().as_millis() as u32;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;

        let _ = recv.read_to_end(0).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Ok((their_announcements, rtt_ms))
    }

    async fn apply_gossip_announcements(
        &self,
        remote: EndpointId,
        rtt_ms: u32,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        discover_peers: bool,
    ) -> Result<()> {
        for (addr, ann) in their_announcements {
            let peer_id = addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            if peer_id == remote {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(remote, addr.clone(), ann).await;
                // Defensive `> 0` gate matching the other two `update_peer_rtt`
                // callsites (path-info recheck below + `handle_gossip_stream`).
                // A 0 ms reading isn't physically meaningful over the network
                // and should never feed the split-eligibility gate; the
                // `gossip_round_trip` measurement was the lone hole that let
                // those phantom "0 ms" values through and caused continuous
                // re-elections. Belt-and-braces.
                if rtt_ms > 0 {
                    self.update_peer_rtt(remote, rtt_ms).await;
                }
            } else {
                self.update_transitive_peer(peer_id, addr, ann).await;
            }
        }

        // Also check the connection's actual path info — the gossip round-trip
        // time above may reflect relay latency even if a direct path is now active.
        {
            let conn = self.state.lock().await.connections.get(&remote).cloned();
            if let Some(conn) = conn {
                let mut paths = conn.paths();
                let path_list = iroh::Watcher::get(&mut paths);
                for path_info in path_list {
                    if path_info.is_selected() {
                        let path_rtt_ms = match path_info.rtt() {
                            Some(rtt) => rtt.as_millis() as u32,
                            None => continue,
                        };
                        let path_type = if path_info.is_ip() { "direct" } else { "relay" };
                        if path_rtt_ms > 0 && path_rtt_ms < rtt_ms {
                            // Per-cycle path telemetry: keep at debug so it does
                            // not flood the default (info) log stream — this
                            // fires on every gossip tick per peer and, under
                            // `--log-format json` to a file, grew stdout.log to
                            // multiple GB. The user-visible signal is the
                            // change-gated "RTT improved — re-electing" line
                            // emitted from `update_peer_rtt`.
                            tracing::debug!(
                                "Peer {} RTT: {}ms ({}) [path info]",
                                remote.fmt_short(),
                                path_rtt_ms,
                                path_type
                            );
                            self.update_peer_rtt(remote, path_rtt_ms).await;
                        }
                        break;
                    }
                }
            }
        }

        if discover_peers {
            let my_role = self.role.lock().await.clone();
            for (addr, ann) in their_announcements {
                let peer_id = addr.id;
                if peer_id == self.endpoint.id() {
                    continue;
                }
                // Clients skip connecting to other clients
                if matches!(my_role, super::NodeRole::Client)
                    && matches!(ann.role, super::NodeRole::Client)
                {
                    continue;
                }
                let has_conn = self.state.lock().await.connections.contains_key(&peer_id);
                if !has_conn {
                    if let Err(e) = Box::pin(self.connect_to_peer(addr.clone())).await {
                        tracing::debug!(
                            "Could not connect to discovered peer {}: {e}",
                            peer_id.fmt_short()
                        );
                    }
                }
            }
        }

        Ok(())
    }

    pub(super) async fn handle_gossip_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        tracing::info!("Inbound gossip from {}", remote.fmt_short());

        {
            let mut state = self.state.lock().await;
            if state.dead_peers.remove(&remote) {
                super::emit_mesh_info(format!(
                    "🔄 Dead peer {} is gossiping — clearing dead status",
                    remote.fmt_short()
                ));
            }
        }

        let buf = read_len_prefixed(&mut recv).await?;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let _ = recv.read_to_end(0).await;

        for (addr, ann) in &their_announcements {
            let peer_id = addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            if peer_id == remote {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                self.add_peer(remote, addr.clone(), ann).await;
            } else {
                self.update_transitive_peer(peer_id, addr, ann).await;
            }
        }

        {
            let conn = self.state.lock().await.connections.get(&remote).cloned();
            if let Some(conn) = conn {
                let mut paths = conn.paths();
                let path_list = iroh::Watcher::get(&mut paths);
                for path_info in path_list {
                    if path_info.is_selected() {
                        let rtt_ms = match path_info.rtt() {
                            Some(rtt) => rtt.as_millis() as u32,
                            None => continue,
                        };
                        let path_type = if path_info.is_ip() { "direct" } else { "relay" };
                        if rtt_ms > 0 {
                            // Per-cycle path telemetry — debug only (see the
                            // matching note in `merge_gossip`). update_peer_rtt
                            // still carries the change-gated, user-visible logs.
                            tracing::debug!(
                                "Peer {} RTT: {}ms ({})",
                                remote.fmt_short(),
                                rtt_ms,
                                path_type
                            );
                            self.update_peer_rtt(remote, rtt_ms).await;
                        }
                        break;
                    }
                }
            }
        }

        let my_role = self.role.lock().await.clone();
        for (addr, ann) in their_announcements {
            let peer_id = addr.id;
            if peer_id == self.endpoint.id() {
                continue;
            }
            // Clients should only connect to hosts/workers — not other clients.
            // This avoids O(N²) client-to-client connections in large meshes.
            if matches!(my_role, super::NodeRole::Client)
                && matches!(ann.role, super::NodeRole::Client)
            {
                continue;
            }
            let already_known = self.state.lock().await.peers.contains_key(&peer_id);
            if !already_known {
                if let Err(e) = Box::pin(self.connect_to_peer(addr)).await {
                    tracing::warn!("Failed to discover peer: {e}");
                }
            }
        }

        Ok(())
    }
    pub(super) async fn remove_peer(&self, id: EndpointId) {
        let mut state = self.state.lock().await;
        // Always clear any rejection-tracking entry so the map stays bounded.
        state.policy_rejected_peers.remove(&id);
        if let Some(peer) = state.peers.remove(&id) {
            tracing::info!(
                "Peer removed: {} (total: {})",
                id.fmt_short(),
                state.peers.len()
            );
            let count = state.peers.len();
            drop(state);
            let _ = self.peer_change_tx.send(count);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerDown,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    pub(super) async fn add_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
    ) {
        let imported_ranking = import_remote_moe_rankings(&ann.served_model_descriptors);
        let trust_store = self.trust_store.lock().await.clone();
        let owner_summary = verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        );
        // Phase 3.1: verify the owner-signed model advertisement against this
        // exact peer id + the local trust store. Advisory for now (surfaced on
        // /api/status); does not gate peer acceptance.
        let model_ad_summary = verify_model_advertisement(
            ann.model_advertisement.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
            DEFAULT_MODEL_AD_TTL_MS,
        );
        if !policy_accepts_peer(self.trust_policy, &owner_summary) {
            let mut state = self.state.lock().await;
            let last_status = state.policy_rejected_peers.get(&id).cloned();
            if last_status.as_ref() != Some(&owner_summary.status) {
                tracing::warn!(
                    "Rejecting peer {} due to owner policy: {:?}",
                    id.fmt_short(),
                    owner_summary.status
                );
                state
                    .policy_rejected_peers
                    .insert(id, owner_summary.status.clone());
            }
            if state.peers.remove(&id).is_some() {
                let _ = self.peer_change_tx.send(state.peers.len());
            }
            return;
        }
        let mut state = self.state.lock().await;
        // Peer accepted — clear any prior rejection record so future rejections log again.
        state.policy_rejected_peers.remove(&id);
        if id == self.endpoint.id() {
            return;
        }
        let now = std::time::Instant::now();
        // If this peer was previously dead, clear it — add_peer is only called
        // after a successful gossip exchange, which is proof of life.
        let recovered = state.dead_peers.remove(&id);
        if recovered {
            super::emit_mesh_info(format!(
                "🔄 Peer {} back from the dead (successful gossip)",
                id.fmt_short()
            ));
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let role_changed = existing.role != ann.role;
            let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
            let serving_changed = existing.serving_models != ann.serving_models
                || existing.hosted_models != ann_hosted_models
                || existing.hosted_models_known != ann.hosted_models.is_some();
            if role_changed {
                tracing::info!(
                    "Peer {} role updated: {:?} → {:?}",
                    id.fmt_short(),
                    existing.role,
                    ann.role
                );
                existing.role = ann.role.clone();
            }
            // Update addr if the new one has more info
            if !addr.addrs.is_empty() {
                existing.addr = addr;
            }
            existing.models = ann.models.clone();
            merge_first_joined_mesh_ts(
                &mut existing.first_joined_mesh_ts,
                ann.first_joined_mesh_ts,
            );
            existing.vram_bytes = ann.vram_bytes;
            if ann.model_source.is_some() {
                existing.model_source = ann.model_source.clone();
            }
            existing.serving_models = ann.serving_models.clone();
            existing.hosted_models = ann_hosted_models;
            existing.hosted_models_known = ann.hosted_models.is_some();
            existing.available_models.clear();
            existing.requested_models = ann.requested_models.clone();
            existing.last_seen = now;
            if recovered {
                existing.moe_recovered_at = Some(now);
            }
            existing.owner_attestation = ann.owner_attestation.clone();
            existing.owner_summary = owner_summary.clone();
            existing.model_ad = ModelAdState {
                advertisement: ann.model_advertisement.clone(),
                summary: model_ad_summary.clone(),
            };
            existing.inflight_requests = ann.inflight_requests;
            // Direct gossip is authoritative; a new peer always reports
            // Some(..), a legacy peer always None — both are the truth.
            existing.rpc_ready = ann.rpc_ready;
            existing.served_model_descriptors = ann.served_model_descriptors.clone();
            existing.served_model_runtime = ann.served_model_runtime.clone();
            if ann.version.is_some() {
                existing.version = ann.version.clone();
            }
            existing.gpu_name = ann.gpu_name.clone();
            existing.hostname = ann.hostname.clone();
            existing.is_soc = ann.is_soc;
            existing.gpu_vram = ann.gpu_vram.clone();
            existing.gpu_reserved_bytes = ann.gpu_reserved_bytes.clone();
            existing.gpu_mem_bandwidth_gbps = ann.gpu_mem_bandwidth_gbps.clone();
            existing.gpu_compute_tflops_fp32 = ann.gpu_compute_tflops_fp32.clone();
            existing.gpu_compute_tflops_fp16 = ann.gpu_compute_tflops_fp16.clone();
            if ann.experts_summary.is_some() {
                existing.experts_summary = ann.experts_summary.clone();
            }
            // Refresh the peer's normalized capability advertisement, mirroring
            // what `apply_transitive_ann` already does for indirectly-mentioned
            // peers. Without this update, the `capability` snapshot a peer sent
            // on its very first gossip exchange was retained for the lifetime
            // of the entry — so a host that came up with empty `serving_models`
            // (the runtime sets that field once it commits to bringing a model
            // up, which can happen seconds AFTER the first announcement) would
            // be advertised across the mesh forever as `loaded_models: []`,
            // even after `hosted_models` had flipped on. That mismatch was the
            // root cause of the public /status page rendering Mac as "Loading"
            // while it was genuinely serving Qwen3-30B as a pipeline_host —
            // the dashboard's `hasLoadedModel` predicate keys on
            // `capability.loaded_models`, not the top-level `hosted_models`,
            // so the staleness silently lied. Backfilling from legacy GPU
            // fields (mirroring `apply_transitive_ann`) preserves behavior for
            // older peers that don't include the field in their announcement.
            if let Some(cap) = ann.capability.clone() {
                existing.capability = cap;
            } else {
                existing.capability = backfill_capability_from_legacy(
                    existing.gpu_name.as_deref(),
                    existing.gpu_vram.as_deref(),
                    existing.is_soc,
                    &existing.serving_models,
                );
            }
            // v0.66.50 (May 20 2026): refresh measured per-model TPS/TTFT
            // and Phase 3.0 native baselines from the latest announcement,
            // but DO NOT overwrite a non-empty existing value with an empty
            // ann. See the matching block in `apply_transitive_ann` for the
            // mixed-version-mesh incident that motivated the empty-guard;
            // the same flap can hit the direct-gossip path when the entry
            // re-handshakes with a legacy peer who happens to have the
            // host's stale view in its connection-warm cache.
            if !ann.model_timings.is_empty() || existing.model_timings.is_empty() {
                existing.model_timings = ann.model_timings.clone();
            }
            if !ann.native_baselines.is_empty() || existing.native_baselines.is_empty() {
                existing.native_baselines = ann.native_baselines.clone();
            }
            let updated_peer = existing.clone();
            let changed = peer_meaningfully_changed(&old_peer, &updated_peer)
                || old_peer.gpu_name != updated_peer.gpu_name
                || old_peer.hostname != updated_peer.hostname
                || old_peer.is_soc != updated_peer.is_soc
                || old_peer.gpu_vram != updated_peer.gpu_vram
                || old_peer.gpu_reserved_bytes != updated_peer.gpu_reserved_bytes
                || old_peer.gpu_mem_bandwidth_gbps != updated_peer.gpu_mem_bandwidth_gbps
                || old_peer.gpu_compute_tflops_fp32 != updated_peer.gpu_compute_tflops_fp32
                || old_peer.gpu_compute_tflops_fp16 != updated_peer.gpu_compute_tflops_fp16;
            if role_changed || serving_changed {
                let count = state.peers.len();
                drop(state);
                let _ = self.peer_change_tx.send(count);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            } else {
                drop(state);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            }
            if imported_ranking {
                self.refresh_served_model_descriptors().await;
            }
            return;
        }
        tracing::info!(
            "Peer added: {} role={:?} vram={:.1}GB assigned={:?} catalog={:?} (total: {})",
            id.fmt_short(),
            ann.role,
            ann.vram_bytes as f64 / 1e9,
            ann.serving_models.first(),
            ann.available_models,
            state.peers.len() + 1
        );
        let mut peer = PeerInfo::from_announcement(id, addr, ann, owner_summary, model_ad_summary);
        if recovered {
            peer.moe_recovered_at = Some(now);
        }
        state.peers.insert(id, peer.clone());
        let count = state.peers.len();
        drop(state);
        let _ = self.peer_change_tx.send(count);
        self.emit_plugin_mesh_event(
            crate::plugin::proto::mesh_event::Kind::PeerUp,
            Some(&peer),
            String::new(),
        )
        .await;
        if imported_ranking {
            self.refresh_served_model_descriptors().await;
        }
    }

    /// Update a peer learned transitively through gossip (not directly connected).
    /// Updates assigned/hosted state so models_being_served() includes their models.
    /// Refreshes `last_mentioned` (not `last_seen`) so the peer survives pruning
    /// and gossip propagation as long as a bridge peer keeps mentioning it, but
    /// PeerDown silencing uses only `last_seen` (direct proof-of-life).
    ///
    /// Tick semantics for `peer_change_tx`:
    ///   * Existing transitive peer, serving changed → tick (election cares).
    ///   * Existing transitive peer, RTT / VRAM / capability noise → no tick.
    ///   * **New** transitive peer enters the active set → tick (added
    ///     May 16 2026 — without this, an elected host could keep
    ///     serving against a stale cohort view for up to one heartbeat
    ///     round when a candidate worker arrived via gossip without
    ///     also touching serving fields; Phase A3's stable-host re-eval
    ///     timer is the backstop for any case the tick still misses).
    ///
    /// The `tokio::sync::watch` channel auto-coalesces concurrent sends,
    /// so a single gossip wave that mentions N new transitive peers
    /// only wakes downstream receivers once.
    pub(super) async fn update_transitive_peer(
        &self,
        id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
    ) {
        let imported_ranking = import_remote_moe_rankings(&ann.served_model_descriptors);
        let trust_store = self.trust_store.lock().await.clone();
        let owner_summary = verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        );
        // Phase 3.1: verify the relayed advertisement against this peer id. The
        // signature covers the originating node id, so a forged/misattached
        // advertisement is caught even when it arrives via a third-party relay.
        let model_ad_summary = verify_model_advertisement(
            ann.model_advertisement.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
            DEFAULT_MODEL_AD_TTL_MS,
        );
        if !policy_accepts_peer(self.trust_policy, &owner_summary) {
            let mut state = self.state.lock().await;
            if state.peers.remove(&id).is_some() {
                let _ = self.peer_change_tx.send(state.peers.len());
            }
            return;
        }
        let mut state = self.state.lock().await;
        if id == self.endpoint.id() {
            return;
        }
        if state.dead_peers.contains(&id) {
            return;
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let serving_changed = apply_transitive_ann(existing, addr, ann);
            existing.owner_summary = owner_summary;
            existing.model_ad = ModelAdState {
                advertisement: ann.model_advertisement.clone(),
                summary: model_ad_summary,
            };
            // Refresh last_mentioned: the bridge peer vouches for this peer
            // being alive (collect_announcements already filters stale peers).
            // We update last_mentioned (not last_seen) so that PeerDown
            // silencing and collect_announcements use only direct proof-of-life,
            // while the prune decision considers both timestamps.
            existing.last_mentioned = std::time::Instant::now();
            let updated_peer = existing.clone();
            let changed = peer_meaningfully_changed(&old_peer, &updated_peer);
            if serving_changed {
                let count = state.peers.len();
                drop(state);
                let _ = self.peer_change_tx.send(count);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            } else {
                drop(state);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            }
            if imported_ranking {
                self.refresh_served_model_descriptors().await;
            }
        } else {
            // New transitive peer — not directly verified, so set last_seen to
            // epoch (not "now") to avoid incorrectly silencing PeerDown reports.
            // last_mentioned = now keeps the peer alive for the prune window.
            let mut peer =
                PeerInfo::from_announcement(id, addr.clone(), ann, owner_summary, model_ad_summary);
            // Mark as never directly seen — only transitively mentioned.
            // `checked_sub` instead of `-`: on Windows `Instant` is anchored to
            // QueryPerformanceCounter ticks since system boot, so within the
            // first ~6 min after boot the bare subtraction panics ("overflow
            // when subtracting duration from instant"). Saw this on the May
            // 18 2026 LYU panic-loop after a Windows Update reboot. Falling
            // back to "now" makes the new transitive peer trivially stale-
            // eligible for the next prune, which is the same direction the
            // original code intended (just safer arithmetic).
            peer.last_seen = std::time::Instant::now()
                .checked_sub(std::time::Duration::from_secs(PEER_STALE_SECS * 2))
                .unwrap_or_else(std::time::Instant::now);
            state.peers.insert(id, peer.clone());
            let count = state.peers.len();
            drop(state);
            let _ = self.peer_change_tx.send(count);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUp,
                Some(&peer),
                String::new(),
            )
            .await;
            if imported_ranking {
                self.refresh_served_model_descriptors().await;
            }
        }
    }

    /// Phase 3.1: build an owner-signed advertisement over the node's current
    /// per-model metric snapshot. Returns `None` when the node has no owner key
    /// or when nothing has been measured yet (an empty advertisement carries no
    /// trust value). The signed claims are exactly the `model_timings` /
    /// `native_baselines` subset that a malicious relay could otherwise forge,
    /// merged per model so each model produces one claim.
    fn build_self_model_advertisement(
        &self,
        timings: &[ModelTimingEntry],
        baselines: &[NativeBaselineEntry],
    ) -> Option<crate::crypto::SignedModelAdvertisement> {
        let keypair = self.owner_keypair.as_ref().as_ref()?;
        let mut by_model: std::collections::BTreeMap<String, crate::crypto::ModelClaim> =
            std::collections::BTreeMap::new();
        for t in timings {
            let claim = by_model.entry(t.model.clone()).or_default();
            claim.model = t.model.clone();
            claim.measured_tps_p50 = Some(t.measured_tps_p50);
            claim.measured_ttft_ms_p50 = Some(t.measured_ttft_ms_p50 as f64);
            claim.samples_in_window = t.samples_in_window as u32;
        }
        for b in baselines {
            let claim = by_model.entry(b.model.clone()).or_default();
            claim.model = b.model.clone();
            claim.native_tps_p50 = Some(b.native_tps_p50);
            claim.native_ttft_ms_p50 = Some(b.native_ttft_ms_p50 as f64);
            claim.native_backend = Some(b.backend.clone());
            claim.native_logit_fingerprint = b
                .logit_fingerprint
                .as_ref()
                .map(|fp| fp.output_sha256.clone());
        }
        let models: Vec<crate::crypto::ModelClaim> = by_model.into_values().collect();
        if models.is_empty() {
            return None;
        }
        match crate::crypto::sign_model_advertisement(
            keypair,
            self.endpoint.id().as_bytes(),
            models,
        ) {
            Ok(ad) => Some(ad),
            Err(err) => {
                tracing::warn!("Phase 3.1: failed to sign model advertisement: {err}");
                None
            }
        }
    }

    pub(super) async fn collect_announcements(&self) -> Vec<PeerAnnouncement> {
        // Snapshot all locks independently — never hold multiple locks simultaneously.
        let my_role = self.role.lock().await.clone();
        let my_models = self.models.lock().await.clone();
        let my_source = self.model_source.lock().await.clone();
        let my_serving_models = self.serving_models.lock().await.clone();
        let my_served_model_descriptors = self.served_model_descriptors.lock().await.clone();
        let my_model_runtime_descriptors = self.model_runtime_descriptors.lock().await.clone();
        let my_hosted_models = self.hosted_models.lock().await.clone();
        let my_available = self.available_models.lock().await.clone();
        let my_requested = self.requested_models.lock().await.clone();
        let my_mesh_id = self.mesh_id.lock().await.clone();
        let my_owner_attestation = self.owner_attestation.lock().await.clone();
        let my_demand = self.get_demand();
        // See `peer.last_seen` assignment above for the `checked_sub` rationale
        // (May 18 2026 Windows panic-loop). Right after boot the cutoff
        // collapses to "now", which means the gossip filter pass-through
        // (`p.last_seen >= stale_cutoff`) admits everything — correct, since
        // nothing has had time to go stale yet.
        let stale_cutoff = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(PEER_STALE_SECS))
            .unwrap_or_else(std::time::Instant::now);
        // Gossip wire encoding strips available_model_metadata and available_model_sizes,
        // and remote ingest ignores them. Avoid an expensive scan_local_inventory_snapshot()
        // on the hot gossip path.
        let my_model_metadata: Vec<_> = Vec::new();
        let my_model_sizes: HashMap<_, _> = HashMap::new();
        let mut announcements: Vec<PeerAnnouncement> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .filter(|p| p.last_seen >= stale_cutoff || p.last_mentioned >= stale_cutoff)
                .map(|p| PeerAnnouncement {
                    addr: p.addr.clone(),
                    role: p.role.clone(),
                    first_joined_mesh_ts: p.first_joined_mesh_ts,
                    models: p.models.clone(),
                    vram_bytes: p.vram_bytes,
                    model_source: p.model_source.clone(),
                    serving_models: p.serving_models.clone(),
                    hosted_models: p.hosted_models_known.then(|| p.hosted_models.clone()),
                    available_models: p.available_models.clone(),
                    requested_models: p.requested_models.clone(),
                    version: p.version.clone(),
                    model_demand: HashMap::new(),
                    mesh_id: None,
                    gpu_name: p.gpu_name.clone(),
                    hostname: p.hostname.clone(),
                    is_soc: p.is_soc,
                    gpu_vram: p.gpu_vram.clone(),
                    gpu_reserved_bytes: p.gpu_reserved_bytes.clone(),
                    gpu_mem_bandwidth_gbps: p.gpu_mem_bandwidth_gbps.clone(),
                    gpu_compute_tflops_fp32: p.gpu_compute_tflops_fp32.clone(),
                    gpu_compute_tflops_fp16: p.gpu_compute_tflops_fp16.clone(),
                    available_model_metadata: p.available_model_metadata.clone(),
                    experts_summary: p.experts_summary.clone(),
                    available_model_sizes: p.available_model_sizes.clone(),
                    served_model_descriptors: p.served_model_descriptors.clone(),
                    served_model_runtime: p.served_model_runtime.clone(),
                    owner_attestation: p.owner_attestation.clone(),
                    inflight_requests: p.inflight_requests,
                    system_ram_bytes: p.system_ram_bytes,
                    // Relay the peer's own rpc readiness unchanged — only the
                    // peer knows whether its lazy rpc-server is up.
                    rpc_ready: p.rpc_ready,
                    // v0.66.41 Phase 1: relay the peer's gossiped model
                    // timings unchanged. We only re-announce what they
                    // told us; we never substitute our own measurements
                    // of their performance (those would be WAN-tainted).
                    model_timings: p.model_timings.clone(),
                    // v0.66.49 Phase 3.0: native baselines relay
                    // unchanged for the same reason — only the peer's
                    // own llama-server can measure its native
                    // performance, anything we add here would be noise.
                    native_baselines: p.native_baselines.clone(),
                    capability: Some(p.capability.clone()),
                    // Phase 3.1: relay the peer's own owner-signed advertisement
                    // unchanged — the signature covers its node id + metrics, so
                    // it survives this transitive hop and we never re-sign for
                    // another owner.
                    model_advertisement: p.model_ad.advertisement.clone(),
                })
                .collect()
        };
        let my_first_joined_mesh_ts = *self.first_joined_mesh_ts.lock().await;
        // v0.66.41 Phase 1 marketplace metrics: snapshot the local node's
        // per-model TPS / TTFT rolling-1h window. Empty when this node hasn't
        // served any model locally yet — see `ModelTimingEntry` for why empty
        // != "measured zero".
        let my_model_timings: Vec<ModelTimingEntry> = self
            .model_timings_snapshot()
            .into_iter()
            .map(|(model, snap)| ModelTimingEntry {
                model,
                measured_tps_p50: snap.measured_tps_p50,
                measured_ttft_ms_p50: snap.measured_ttft_ms_p50,
                samples_in_window: snap.samples_in_window,
            })
            .collect();
        // v0.66.49 Phase 3.0: native baselines measured directly against the
        // local llama-server (no mesh involved).
        let my_native_baselines = self.native_baselines_snapshot().await;
        // v0.66.x Phase 3.1: sign this exact metric snapshot under the owner
        // key so peers can reject relayed forgeries. `None` when the node has
        // no owner key or nothing measured yet.
        let my_model_advertisement =
            self.build_self_model_advertisement(&my_model_timings, &my_native_baselines);
        announcements.push(PeerAnnouncement {
            addr: self.endpoint.addr(),
            role: my_role,
            first_joined_mesh_ts: my_first_joined_mesh_ts,
            models: my_models,
            vram_bytes: self.vram_bytes,
            model_source: my_source,
            serving_models: my_serving_models,
            hosted_models: Some(my_hosted_models),
            available_models: my_available,
            requested_models: my_requested,
            version: Some(crate::VERSION.to_string()),
            model_demand: my_demand,
            mesh_id: my_mesh_id,
            gpu_name: if self.enumerate_host {
                self.gpu_name.clone()
            } else {
                None
            },
            hostname: if self.enumerate_host {
                self.hostname.clone()
            } else {
                None
            },
            is_soc: self.is_soc,
            gpu_vram: if self.enumerate_host {
                self.gpu_vram.clone()
            } else {
                None
            },
            gpu_reserved_bytes: if self.enumerate_host {
                self.gpu_reserved_bytes.clone()
            } else {
                None
            },
            gpu_mem_bandwidth_gbps: self.gpu_mem_bandwidth_gbps.lock().await.as_ref().map(|v| {
                v.iter()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(",")
            }),
            gpu_compute_tflops_fp32: self.gpu_compute_tflops_fp32.lock().await.as_ref().map(|v| {
                v.iter()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(",")
            }),
            gpu_compute_tflops_fp16: self.gpu_compute_tflops_fp16.lock().await.as_ref().map(|v| {
                v.iter()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(",")
            }),
            available_model_metadata: my_model_metadata,
            experts_summary: None,
            available_model_sizes: my_model_sizes,
            served_model_descriptors: my_served_model_descriptors,
            served_model_runtime: my_model_runtime_descriptors,
            owner_attestation: my_owner_attestation,
            inflight_requests: self.inflight_requests(),
            system_ram_bytes: self.system_ram_bytes,
            // Lazy rpc-server readiness. Self is never legacy, so always
            // Some(..): split hosts can gate worker selection on it.
            rpc_ready: Some(self.local_rpc_ready()),
            model_timings: my_model_timings,
            native_baselines: my_native_baselines,
            capability: self.local_node_capability().await,
            // Phase 3.1: owner-signed attestation over the snapshot above.
            model_advertisement: my_model_advertisement,
        });
        announcements
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::OwnershipSummary;
    use iroh::SecretKey;
    use std::collections::HashMap;

    fn test_endpoint_id(seed: u8) -> EndpointId {
        EndpointId::from(SecretKey::from_bytes(&[seed; 32]).public())
    }

    fn test_addr(seed: u8) -> EndpointAddr {
        EndpointAddr {
            id: test_endpoint_id(seed),
            addrs: Default::default(),
        }
    }

    fn test_announcement(ts: Option<u64>) -> PeerAnnouncement {
        PeerAnnouncement {
            addr: test_addr(0x11),
            role: NodeRole::Worker,
            first_joined_mesh_ts: ts,
            models: vec![],
            vram_bytes: 0,
            model_source: None,
            serving_models: vec![],
            hosted_models: None,
            available_models: vec![],
            requested_models: vec![],
            version: None,
            model_demand: HashMap::new(),
            mesh_id: None,
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
            inflight_requests: 0,
            system_ram_bytes: 0,
            model_timings: vec![],
            native_baselines: vec![],
            rpc_ready: None,
            capability: None,
            model_advertisement: None,
        }
    }

    fn test_peer(ts: Option<u64>) -> PeerInfo {
        PeerInfo::from_announcement(
            test_endpoint_id(0x22),
            test_addr(0x22),
            &test_announcement(ts),
            OwnershipSummary::default(),
            ModelAdSummary::default(),
        )
    }

    #[test]
    fn test_merge_none_to_some() {
        let mut existing = test_peer(None);
        let ann = test_announcement(Some(100));

        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    #[test]
    fn test_merge_some_to_none_keeps_existing() {
        let mut existing = test_peer(Some(100));
        let ann = test_announcement(None);

        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    #[test]
    fn test_merge_earlier_incoming_wins() {
        let mut existing = test_peer(Some(200));
        let ann = test_announcement(Some(100));

        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    #[test]
    fn test_merge_later_incoming_loses() {
        let mut existing = test_peer(Some(100));
        let ann = test_announcement(Some(200));

        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    #[test]
    fn test_merge_equal_values_unchanged() {
        let mut existing = test_peer(Some(100));
        let ann = test_announcement(Some(100));

        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);

        assert_eq!(existing.first_joined_mesh_ts, Some(100));
    }

    /// Regression for v0.66.46: before this fix, `apply_transitive_ann`
    /// silently skipped `model_timings`, so a peer's measured TPS/TTFT
    /// never propagated past its first announcement. Result: the entry
    /// node's `/api/status` (the dashboard's only feed) always showed
    /// `measured_tps_p50_by_model: {}` for every host, even when local
    /// `/api/status` on the host itself reported real samples. We assert
    /// here that the latest announcement always wins, including the
    /// transition from "some samples" back to empty (matters because
    /// the snapshot is omitted when no recent samples exist, and we
    /// don't want stale timings lingering in the UI).
    #[test]
    fn test_apply_transitive_ann_refreshes_model_timings() {
        let mut existing = test_peer(Some(100));
        existing.model_timings = vec![crate::mesh::ModelTimingEntry {
            model: "stale-model".to_string(),
            measured_tps_p50: 1.0,
            measured_ttft_ms_p50: 1,
            samples_in_window: 1,
        }];

        let mut ann = test_announcement(Some(100));
        ann.model_timings = vec![crate::mesh::ModelTimingEntry {
            model: "Qwen3-32B-Q4_K_M".to_string(),
            measured_tps_p50: 12.5,
            measured_ttft_ms_p50: 850,
            samples_in_window: 4,
        }];

        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);

        assert_eq!(existing.model_timings.len(), 1);
        assert_eq!(existing.model_timings[0].model, "Qwen3-32B-Q4_K_M");
        assert!((existing.model_timings[0].measured_tps_p50 - 12.5).abs() < 1e-9);
        assert_eq!(existing.model_timings[0].measured_ttft_ms_p50, 850);

        // v0.66.50 (May 20 2026): empty in a transitive ann must NOT
        // overwrite a non-empty existing — that flap was the live
        // defect that hid the catalog ratio in mixed v0.66.48/v0.66.49
        // meshes. Full→empty only fires from a direct gossip path
        // (`add_peer`), and even there guarded by the same `is_empty`
        // check; the source-of-truth for "this host has no recent
        // samples" must not be a legacy peer's stale relayed view.
        ann.model_timings = vec![];
        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);
        assert_eq!(
            existing.model_timings.len(),
            1,
            "empty transitive ann must not stomp non-empty existing model_timings"
        );
    }

    /// Regression for v0.66.49 Phase 3.0: same defect class as
    /// `test_apply_transitive_ann_refreshes_model_timings` above —
    /// `apply_transitive_ann` MUST overwrite `native_baselines` on
    /// every announcement, not just the first. The Phase 1 post-mortem
    /// rule that motivated this test (any new `PeerAnnouncement` field
    /// gets a refresh assertion, not just a `from_announcement` smoke
    /// check) is the only reason the catalog ratio renders correctly
    /// for transitively-learned peers; without it, a peer learned via
    /// relay would publish its baseline once and then stay frozen on
    /// the entry's `/api/status` forever.
    #[test]
    fn test_apply_transitive_ann_refreshes_native_baselines() {
        let mut existing = test_peer(Some(100));
        existing.native_baselines = vec![crate::mesh::NativeBaselineEntry {
            model: "stale-model".to_string(),
            native_tps_p50: 1.0,
            native_ttft_ms_p50: 1,
            measured_at_unix_secs: 1,
            samples: 1,
            backend: "metal".to_string(),
            logit_fingerprint: None,
        }];

        let mut ann = test_announcement(Some(100));
        ann.native_baselines = vec![crate::mesh::NativeBaselineEntry {
            model: "Qwen3-8B-Q4_K_M".to_string(),
            native_tps_p50: 28.5,
            native_ttft_ms_p50: 320,
            measured_at_unix_secs: 1_747_700_000,
            samples: 1,
            backend: "metal".to_string(),
            logit_fingerprint: None,
        }];

        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);

        assert_eq!(existing.native_baselines.len(), 1);
        assert_eq!(existing.native_baselines[0].model, "Qwen3-8B-Q4_K_M");
        assert!((existing.native_baselines[0].native_tps_p50 - 28.5).abs() < 1e-9);
        assert_eq!(existing.native_baselines[0].native_ttft_ms_p50, 320);

        // v0.66.50 (May 20 2026): same empty-transitive guard as
        // `test_apply_transitive_ann_refreshes_model_timings`. A legacy
        // v0.66.48 worker's view of a v0.66.49 host has
        // `native_baselines: []` (default for an unknown proto field);
        // when that worker gossips its world to the entry we must not
        // let it blank the host's freshly-published baseline.
        ann.native_baselines = vec![];
        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);
        assert_eq!(
            existing.native_baselines.len(),
            1,
            "empty transitive ann must not stomp non-empty existing native_baselines"
        );
    }

    #[test]
    fn test_meaningfully_changed_first_joined_mesh_ts() {
        let old_peer = test_peer(Some(100));
        let new_peer = test_peer(Some(200));

        assert!(peer_meaningfully_changed(&old_peer, &new_peer));
    }

    /// `rpc_ready` (v0.66.70 lazy rpc-server) is owner-authoritative and
    /// *dynamic* — a worker flips it false→true the instant it brings its
    /// on-demand rpc-server up, and back to false on idle teardown. A split
    /// host gates its launch plan on this bit, so a stomp here would either
    /// strand a ready worker out of the candidate set or, worse, dial a still
    /// -cold one and trip the HELLO-blacklist. Unlike `model_timings` /
    /// `native_baselines` (guarded by `is_empty`), this is an `Option<bool>`,
    /// so the legacy-relay guard is `ann.rpc_ready.is_some()`: a legacy peer
    /// that can't see the field sends `None` and must not blank a known value.
    #[test]
    fn test_apply_transitive_ann_guards_rpc_ready() {
        let mut existing = test_peer(Some(100));
        existing.rpc_ready = Some(true);

        // Legacy relay (field invisible -> None) must NOT stomp the known bit.
        let mut ann = test_announcement(Some(100));
        ann.rpc_ready = None;
        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);
        assert_eq!(
            existing.rpc_ready,
            Some(true),
            "None in a transitive ann (legacy relay) must not blank a known rpc_ready"
        );

        // A current peer relaying the owner's real state (teardown) wins.
        ann.rpc_ready = Some(false);
        apply_transitive_ann(&mut existing, &test_addr(0x33), &ann);
        assert_eq!(
            existing.rpc_ready,
            Some(false),
            "Some(false) is the owner's authoritative teardown signal and must apply"
        );
    }

    /// A worker bringing its lazy rpc-server up/down must trigger regossip so
    /// the split host re-evaluates its candidate set promptly — the flip is the
    /// whole point of the gossiped bit.
    #[test]
    fn test_meaningfully_changed_rpc_ready() {
        let mut old_peer = test_peer(Some(100));
        let mut new_peer = test_peer(Some(100));
        old_peer.rpc_ready = Some(false);
        new_peer.rpc_ready = Some(true);

        assert!(peer_meaningfully_changed(&old_peer, &new_peer));
    }
}
