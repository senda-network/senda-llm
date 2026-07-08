//! Phase 3.1 — owner-signed model advertisements.
//!
//! A peer's advertised per-model performance claims (`measured_tps_p50`,
//! `native_tps_p50`, the logit fingerprint, etc.) are gossiped across the mesh
//! and relayed transitively by other peers. Without a signature, a malicious
//! relay can attach a valid node-ownership attestation to *fabricated* metrics
//! for some other peer, and routing has no way to tell. This module binds those
//! claims to the advertising owner's Ed25519 key so they survive transitive
//! relay unforgeable.
//!
//! Design notes:
//! - Signed with the **owner** key (not the iroh node key): marketplace honesty
//!   is an owner-accountability property, and revocation already flows through
//!   the existing trust store (`TrustStore::revoked_owners`).
//! - The `node_endpoint_id` is part of the signed payload (same pattern as
//!   `NodeOwnershipClaim`) so a relay cannot reattach a valid advertisement to a
//!   different node.
//! - `issued_at_unix_ms` + a TTL gives replay protection: metrics change over
//!   time, so a node re-signs its current snapshot every gossip round and
//!   verifiers reject advertisements older than the TTL window.
//! - The authoritative claims are embedded in the signed structure rather than
//!   reconstructed from the parallel unsigned announcement fields, so there is
//!   no float-reconstruction ambiguity at verification time.

use ed25519_dalek::Signer;
use serde::{Deserialize, Serialize};

use super::error::CryptoError;
use super::keys::{owner_id_from_verifying_key, OwnerKeypair};
use super::ownership::{TrustPolicy, TrustStore};

pub const MODEL_AD_VERSION: u32 = 1;

/// How long a signed advertisement is considered fresh. A node re-signs its
/// current metric snapshot on every gossip round (well under this window), so a
/// relay cannot replay a stale, more-favorable advertisement indefinitely.
pub const DEFAULT_MODEL_AD_TTL_MS: u64 = 5 * 60 * 1000;

const SIGNING_DOMAIN_TAG: &[u8] = b"senda-model-ad-v1:";

/// One per-model performance claim, exactly the trust-sensitive subset of a
/// peer announcement that a relay could otherwise forge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ModelClaim {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measured_tps_p50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measured_ttft_ms_p50: Option<f64>,
    #[serde(default)]
    pub samples_in_window: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_tps_p50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_ttft_ms_p50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_logit_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelAdvertisementClaim {
    pub version: u32,
    pub owner_id: String,
    pub owner_sign_public_key: String,
    pub node_endpoint_id: String,
    pub issued_at_unix_ms: u64,
    pub models: Vec<ModelClaim>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SignedModelAdvertisement {
    pub claim: ModelAdvertisementClaim,
    pub signature: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelAdStatus {
    /// Signature valid, node id matches, fresh, owner not revoked/untrusted.
    Verified,
    /// No advertisement present (e.g. peer predates 3.1, or has no owner key).
    #[default]
    Unsigned,
    InvalidSignature,
    MismatchedNodeId,
    /// `issued_at` is in the future or older than the TTL window.
    Stale,
    RevokedOwner,
    UntrustedOwner,
    UnsupportedProtocol,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ModelAdSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner_id: Option<String>,
    pub status: ModelAdStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issued_at_unix_ms: Option<u64>,
    /// Number of models covered by a verified advertisement.
    #[serde(default)]
    pub model_count: usize,
    pub verified: bool,
}

impl ModelAdSummary {
    pub fn is_verified(&self) -> bool {
        self.verified && self.status == ModelAdStatus::Verified
    }
}

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn decode_hex_32(label: &str, value: &str) -> Result<[u8; 32], CryptoError> {
    let decoded = hex::decode(value).map_err(|e| CryptoError::InvalidKeyMaterial {
        reason: format!("bad {label} hex: {e}"),
    })?;
    decoded
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: format!("{label} must be 32 bytes"),
        })
}

fn write_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn write_optional_string(buf: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            buf.push(1);
            write_string(buf, value);
        }
        None => buf.push(0),
    }
}

fn write_optional_f64(buf: &mut Vec<u8>, value: Option<f64>) {
    match value {
        // Encode the IEEE-754 bit pattern so the signed bytes are exact and
        // survive protobuf `double` round-trips unchanged.
        Some(value) => {
            buf.push(1);
            buf.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        None => buf.push(0),
    }
}

/// Deterministic byte encoding of a claim. Models are sorted by name so the
/// signed bytes do not depend on announcement vec ordering.
fn canonical_claim_bytes(claim: &ModelAdvertisementClaim) -> Result<Vec<u8>, CryptoError> {
    let owner_sign_public_key =
        decode_hex_32("owner_sign_public_key", &claim.owner_sign_public_key)?;
    let node_endpoint_id = decode_hex_32("node_endpoint_id", &claim.node_endpoint_id)?;

    let mut buf = Vec::with_capacity(256 + claim.models.len() * 64);
    buf.extend_from_slice(SIGNING_DOMAIN_TAG);
    buf.extend_from_slice(&claim.version.to_le_bytes());
    write_string(&mut buf, &claim.owner_id);
    buf.extend_from_slice(&owner_sign_public_key);
    buf.extend_from_slice(&node_endpoint_id);
    buf.extend_from_slice(&claim.issued_at_unix_ms.to_le_bytes());

    let mut models: Vec<&ModelClaim> = claim.models.iter().collect();
    models.sort_by(|a, b| a.model.cmp(&b.model));
    buf.extend_from_slice(&(models.len() as u64).to_le_bytes());
    for m in models {
        write_string(&mut buf, &m.model);
        write_optional_string(&mut buf, m.quant.as_deref());
        write_optional_f64(&mut buf, m.measured_tps_p50);
        write_optional_f64(&mut buf, m.measured_ttft_ms_p50);
        buf.extend_from_slice(&m.samples_in_window.to_le_bytes());
        write_optional_f64(&mut buf, m.native_tps_p50);
        write_optional_f64(&mut buf, m.native_ttft_ms_p50);
        write_optional_string(&mut buf, m.native_backend.as_deref());
        write_optional_string(&mut buf, m.native_logit_fingerprint.as_deref());
    }
    Ok(buf)
}

/// Sign the node's current model claims under the owner key. Call once per
/// gossip round with a fresh `issued_at` and the current metric snapshot.
pub fn sign_model_advertisement(
    owner: &OwnerKeypair,
    node_endpoint_id: &[u8; 32],
    models: Vec<ModelClaim>,
) -> Result<SignedModelAdvertisement, CryptoError> {
    let claim = ModelAdvertisementClaim {
        version: MODEL_AD_VERSION,
        owner_id: owner.owner_id(),
        owner_sign_public_key: hex::encode(owner.verifying_key().as_bytes()),
        node_endpoint_id: hex::encode(node_endpoint_id),
        issued_at_unix_ms: current_time_unix_ms(),
        models,
    };
    let bytes = canonical_claim_bytes(&claim)?;
    let signature = owner.signing.sign(&bytes);
    Ok(SignedModelAdvertisement {
        claim,
        signature: hex::encode(signature.to_bytes()),
    })
}

/// Verify a signed advertisement against the node it claims to describe and the
/// local trust store. `ttl_ms` bounds how old `issued_at` may be (replay guard).
pub fn verify_model_advertisement(
    advertisement: Option<&SignedModelAdvertisement>,
    actual_node_endpoint_id: &[u8; 32],
    trust_store: &TrustStore,
    policy: TrustPolicy,
    now_unix_ms: u64,
    ttl_ms: u64,
) -> ModelAdSummary {
    let Some(advertisement) = advertisement else {
        return ModelAdSummary::default();
    };
    let claim = &advertisement.claim;

    let mut summary = ModelAdSummary {
        owner_id: Some(claim.owner_id.clone()),
        issued_at_unix_ms: Some(claim.issued_at_unix_ms),
        model_count: claim.models.len(),
        ..ModelAdSummary::default()
    };

    if claim.version != MODEL_AD_VERSION {
        summary.status = ModelAdStatus::UnsupportedProtocol;
        return summary;
    }

    let owner_sign_public_key =
        match decode_hex_32("owner_sign_public_key", &claim.owner_sign_public_key) {
            Ok(bytes) => bytes,
            Err(_) => {
                summary.status = ModelAdStatus::InvalidSignature;
                return summary;
            }
        };
    let node_endpoint_id = match decode_hex_32("node_endpoint_id", &claim.node_endpoint_id) {
        Ok(bytes) => bytes,
        Err(_) => {
            summary.status = ModelAdStatus::InvalidSignature;
            return summary;
        }
    };

    if node_endpoint_id != *actual_node_endpoint_id {
        summary.status = ModelAdStatus::MismatchedNodeId;
        return summary;
    }

    let verifying_key = match ed25519_dalek::VerifyingKey::from_bytes(&owner_sign_public_key) {
        Ok(value) => value,
        Err(_) => {
            summary.status = ModelAdStatus::InvalidSignature;
            return summary;
        }
    };
    if owner_id_from_verifying_key(&verifying_key) != claim.owner_id {
        summary.status = ModelAdStatus::InvalidSignature;
        return summary;
    }

    let canonical = match canonical_claim_bytes(claim) {
        Ok(value) => value,
        Err(_) => {
            summary.status = ModelAdStatus::InvalidSignature;
            return summary;
        }
    };
    let sig_bytes = match hex::decode(&advertisement.signature) {
        Ok(value) => value,
        Err(_) => {
            summary.status = ModelAdStatus::InvalidSignature;
            return summary;
        }
    };
    let sig_bytes: [u8; 64] = match sig_bytes.try_into() {
        Ok(value) => value,
        Err(_) => {
            summary.status = ModelAdStatus::InvalidSignature;
            return summary;
        }
    };
    let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    if verifying_key.verify_strict(&canonical, &signature).is_err() {
        summary.status = ModelAdStatus::InvalidSignature;
        return summary;
    }

    // Freshness: reject advertisements issued in the future (small clock skew
    // tolerance) or older than the TTL window.
    let skew_tolerance_ms = 60_000;
    if claim.issued_at_unix_ms > now_unix_ms.saturating_add(skew_tolerance_ms)
        || now_unix_ms.saturating_sub(claim.issued_at_unix_ms) > ttl_ms
    {
        summary.status = ModelAdStatus::Stale;
        return summary;
    }

    if trust_store
        .revoked_owners
        .iter()
        .any(|entry| entry.owner_id == claim.owner_id)
    {
        summary.status = ModelAdStatus::RevokedOwner;
        return summary;
    }

    if matches!(policy, TrustPolicy::Allowlist)
        && !trust_store
            .trusted_owners
            .iter()
            .any(|entry| entry.owner_id == claim.owner_id)
    {
        summary.status = ModelAdStatus::UntrustedOwner;
        return summary;
    }

    summary.status = ModelAdStatus::Verified;
    summary.verified = true;
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::ownership::{RevokedOwner, TrustedOwner};

    fn sample_models() -> Vec<ModelClaim> {
        vec![
            ModelClaim {
                model: "Qwen2.5-32B-Instruct-Q4_K_M".into(),
                quant: Some("Q4_K_M".into()),
                measured_tps_p50: Some(37.148),
                measured_ttft_ms_p50: Some(115.0),
                samples_in_window: 24,
                native_tps_p50: Some(37.18),
                native_ttft_ms_p50: Some(40.0),
                native_backend: Some("cuda".into()),
                native_logit_fingerprint: Some("abcd1234".into()),
            },
            ModelClaim {
                model: "Qwen2.5-0.5B-Instruct-Q4_K_M".into(),
                quant: Some("Q4_K_M".into()),
                measured_tps_p50: Some(180.0),
                measured_ttft_ms_p50: Some(20.0),
                samples_in_window: 5,
                native_tps_p50: None,
                native_ttft_ms_p50: None,
                native_backend: None,
                native_logit_fingerprint: None,
            },
        ]
    }

    #[test]
    fn sign_then_verify_roundtrips() {
        let owner = OwnerKeypair::generate();
        let node_id = [7u8; 32];
        let ad = sign_model_advertisement(&owner, &node_id, sample_models()).unwrap();
        let now = ad.claim.issued_at_unix_ms;
        let summary = verify_model_advertisement(
            Some(&ad),
            &node_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert!(summary.is_verified(), "status was {:?}", summary.status);
        assert_eq!(summary.model_count, 2);
        assert_eq!(summary.owner_id.as_deref(), Some(owner.owner_id().as_str()));
    }

    #[test]
    fn none_advertisement_is_unsigned() {
        let summary = verify_model_advertisement(
            None,
            &[0u8; 32],
            &TrustStore::default(),
            TrustPolicy::Off,
            current_time_unix_ms(),
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert_eq!(summary.status, ModelAdStatus::Unsigned);
        assert!(!summary.verified);
    }

    #[test]
    fn tampered_metric_fails_verification() {
        let owner = OwnerKeypair::generate();
        let node_id = [7u8; 32];
        let mut ad = sign_model_advertisement(&owner, &node_id, sample_models()).unwrap();
        // A relay inflates the advertised throughput but keeps the signature.
        ad.claim.models[0].measured_tps_p50 = Some(999.0);
        let now = ad.claim.issued_at_unix_ms;
        let summary = verify_model_advertisement(
            Some(&ad),
            &node_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert_eq!(summary.status, ModelAdStatus::InvalidSignature);
    }

    #[test]
    fn model_order_does_not_affect_verification() {
        let owner = OwnerKeypair::generate();
        let node_id = [7u8; 32];
        let ad = sign_model_advertisement(&owner, &node_id, sample_models()).unwrap();
        let mut reordered = ad.clone();
        reordered.claim.models.reverse();
        let now = ad.claim.issued_at_unix_ms;
        let summary = verify_model_advertisement(
            Some(&reordered),
            &node_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert!(summary.is_verified(), "status was {:?}", summary.status);
    }

    #[test]
    fn wrong_node_id_is_rejected() {
        let owner = OwnerKeypair::generate();
        let ad = sign_model_advertisement(&owner, &[7u8; 32], sample_models()).unwrap();
        let now = ad.claim.issued_at_unix_ms;
        let summary = verify_model_advertisement(
            Some(&ad),
            &[9u8; 32], // relay reattached the ad to a different node
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert_eq!(summary.status, ModelAdStatus::MismatchedNodeId);
    }

    #[test]
    fn stale_advertisement_is_rejected() {
        let owner = OwnerKeypair::generate();
        let node_id = [7u8; 32];
        let ad = sign_model_advertisement(&owner, &node_id, sample_models()).unwrap();
        let now = ad.claim.issued_at_unix_ms + DEFAULT_MODEL_AD_TTL_MS + 1;
        let summary = verify_model_advertisement(
            Some(&ad),
            &node_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert_eq!(summary.status, ModelAdStatus::Stale);
    }

    #[test]
    fn future_advertisement_beyond_skew_is_rejected() {
        let owner = OwnerKeypair::generate();
        let node_id = [7u8; 32];
        let ad = sign_model_advertisement(&owner, &node_id, sample_models()).unwrap();
        let now = ad.claim.issued_at_unix_ms.saturating_sub(10 * 60 * 1000);
        let summary = verify_model_advertisement(
            Some(&ad),
            &node_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert_eq!(summary.status, ModelAdStatus::Stale);
    }

    #[test]
    fn revoked_owner_is_rejected() {
        let owner = OwnerKeypair::generate();
        let node_id = [7u8; 32];
        let ad = sign_model_advertisement(&owner, &node_id, sample_models()).unwrap();
        let now = ad.claim.issued_at_unix_ms;
        let trust = TrustStore {
            revoked_owners: vec![RevokedOwner {
                owner_id: owner.owner_id(),
                reason: Some("test".into()),
            }],
            ..TrustStore::default()
        };
        let summary = verify_model_advertisement(
            Some(&ad),
            &node_id,
            &trust,
            TrustPolicy::Off,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert_eq!(summary.status, ModelAdStatus::RevokedOwner);
    }

    #[test]
    fn allowlist_policy_rejects_unknown_owner() {
        let owner = OwnerKeypair::generate();
        let node_id = [7u8; 32];
        let ad = sign_model_advertisement(&owner, &node_id, sample_models()).unwrap();
        let now = ad.claim.issued_at_unix_ms;

        let empty = TrustStore {
            policy: TrustPolicy::Allowlist,
            ..TrustStore::default()
        };
        let summary = verify_model_advertisement(
            Some(&ad),
            &node_id,
            &empty,
            TrustPolicy::Allowlist,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert_eq!(summary.status, ModelAdStatus::UntrustedOwner);

        let allowed = TrustStore {
            policy: TrustPolicy::Allowlist,
            trusted_owners: vec![TrustedOwner {
                owner_id: owner.owner_id(),
                label: None,
            }],
            ..TrustStore::default()
        };
        let summary = verify_model_advertisement(
            Some(&ad),
            &node_id,
            &allowed,
            TrustPolicy::Allowlist,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert!(summary.is_verified(), "status was {:?}", summary.status);
    }

    #[test]
    fn forged_owner_id_mismatch_is_rejected() {
        let owner = OwnerKeypair::generate();
        let node_id = [7u8; 32];
        let mut ad = sign_model_advertisement(&owner, &node_id, sample_models()).unwrap();
        // Claim a different owner_id than the embedded public key derives to.
        ad.claim.owner_id = "0".repeat(64);
        let now = ad.claim.issued_at_unix_ms;
        let summary = verify_model_advertisement(
            Some(&ad),
            &node_id,
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
            DEFAULT_MODEL_AD_TTL_MS,
        );
        assert_eq!(summary.status, ModelAdStatus::InvalidSignature);
    }
}
