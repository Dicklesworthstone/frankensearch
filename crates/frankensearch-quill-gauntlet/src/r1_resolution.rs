//! R1 boundary resolution artifact (bd-r1-exact-repair-or-residual-1i4j4).
//!
//! The R1 epic demanded one canonical, content-addressed decision for the
//! Long John Silver divergence surface: either the divergence mechanism was
//! repaired to exact public score-bit parity (`ExactRepair`), or the trace
//! proved semantically equivalent physical-layout association and a narrow,
//! falsifiable hypothesis is handed to the downstream V8 schema node
//! (`ControlledResidual`). Neither variant is a waiver:
//!
//! * `ExactRepair` binds the causal patch revisions plus focused and broad
//!   exact replay receipts — evidence that the repaired engines agree
//!   bit-for-bit, not a promise;
//! * `ControlledResidual` binds the precise aggregate-only hypothesis and
//!   the six falsification arms the V8 producer must execute; it authorizes
//!   only that controlled experiment and hides no divergence.
//!
//! The resolution is bound to the V7 replay freeze
//! (bd-campaign-report-v7-replay-freeze-715tp), the implementing source
//! revision, and the corpus/query/snapshot/trace identities the decision
//! was derived from, so a stored resolution can be re-verified
//! byte-for-byte and every stale or foreign binding rejects. AST fold
//! compatibility, a single default-layout 1-ULP observation, and generic
//! epsilon remain inadmissible by construction: they cannot inhabit either
//! variant's evidence fields.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

/// Schema tag of the resolution artifact.
pub const R1_BOUNDARY_RESOLUTION_SCHEMA_V1: &str = "quill-gauntlet-r1-boundary-resolution-v1";

/// Domain separator for [`R1BoundaryResolutionV1::digest_sha256`].
const R1_BOUNDARY_RESOLUTION_HASH_DOMAIN: &[u8] =
    b"frankensearch/quill-gauntlet/r1-boundary-resolution/v1\0";

/// Upper bound on replay receipts one resolution may carry per breadth.
pub const MAX_REPLAY_RECEIPTS: usize = 64;

/// Upper bound on causal patch revisions one resolution may carry.
pub const MAX_CAUSAL_PATCH_REVISIONS: usize = 16;

/// Upper bound on the residual hypothesis text, in bytes.
pub const MAX_HYPOTHESIS_BYTES: usize = 4_096;

/// Exactly six falsification arms, per the V8 six-arm producer contract.
pub const RESIDUAL_FALSIFICATION_ARMS: usize = 6;

/// Why a stored resolution is not a valid R1 boundary decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum R1ResolutionError {
    #[error("resolution schema is not {R1_BOUNDARY_RESOLUTION_SCHEMA_V1}")]
    Schema,
    #[error("a provenance digest is unset")]
    UnsetDigest,
    #[error("the source revision is not a lowercase 40-hex git revision")]
    SourceRevision,
    #[error("a causal patch revision is not a lowercase 40-hex git revision")]
    CausalPatchRevision,
    #[error("an exact repair binds no causal patch revisions")]
    EmptyCausalPatch,
    #[error("more than {MAX_CAUSAL_PATCH_REVISIONS} causal patch revisions")]
    TooManyCausalPatches,
    #[error("causal patch revisions repeat")]
    DuplicateCausalPatch,
    #[error("an exact repair binds no focused replay receipts")]
    EmptyFocusedReplay,
    #[error("an exact repair binds no broad replay receipts")]
    EmptyBroadReplay,
    #[error("more than {MAX_REPLAY_RECEIPTS} replay receipts in one breadth")]
    TooManyReplayReceipts,
    #[error("a replay receipt digest is unset")]
    UnsetReplayReceipt,
    #[error("a controlled residual carries an empty hypothesis")]
    EmptyHypothesis,
    #[error("the residual hypothesis exceeds {MAX_HYPOTHESIS_BYTES} bytes")]
    HypothesisTooLarge,
    #[error("a falsification arm digest is unset")]
    UnsetFalsificationArm,
    #[error("falsification arm digests repeat")]
    DuplicateFalsificationArm,
}

fn is_lowercase_40_hex(revision: &str) -> bool {
    revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_unset(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

/// Bound provenance: which V7 freeze, source, corpus, query set, snapshot,
/// and trace receipts the resolution was derived from. Digests only —
/// nothing here carries raw queries, documents, or paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct R1ResolutionProvenanceV1 {
    /// Digest of the immutable CampaignReport V7 replay-freeze bytes
    /// (bd-campaign-report-v7-replay-freeze-715tp fixture).
    pub v7_freeze_sha256: [u8; 32],
    /// Implementing source revision the decision was taken at
    /// (lowercase 40-hex git revision).
    pub source_revision: String,
    /// Digest of the corpus manifest the divergence surface lives on.
    pub corpus_manifest_sha256: [u8; 32],
    /// Digest of the query set exercising the divergence surface.
    pub query_set_sha256: [u8; 32],
    /// Digest of the physical snapshot the observations were taken under.
    pub snapshot_sha256: [u8; 32],
    /// Digest of the scorer-trace receipt set the mechanism analysis
    /// consumed (bd-r1-preconstruction-scorer-trace-rtnwu).
    pub trace_receipt_sha256: [u8; 32],
}

impl R1ResolutionProvenanceV1 {
    fn validate(&self) -> Result<(), R1ResolutionError> {
        for digest in [
            &self.v7_freeze_sha256,
            &self.corpus_manifest_sha256,
            &self.query_set_sha256,
            &self.snapshot_sha256,
            &self.trace_receipt_sha256,
        ] {
            if is_unset(digest) {
                return Err(R1ResolutionError::UnsetDigest);
            }
        }
        if !is_lowercase_40_hex(&self.source_revision) {
            return Err(R1ResolutionError::SourceRevision);
        }
        Ok(())
    }
}

/// The decided branch, with the evidence that branch is required to bind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum R1BoundaryVariantV1 {
    /// The divergence mechanism was repaired; both engines agree on exact
    /// public score bits across focused and broad replay.
    ExactRepair {
        /// The causal patch: every git revision whose change is part of
        /// the repair, lowercase 40-hex, unique, in landing order.
        causal_patch_revisions: Vec<String>,
        /// Digests of the focused exact replay receipts (the named
        /// divergence surface replayed bit-for-bit).
        focused_replay_receipt_sha256s: Vec<[u8; 32]>,
        /// Digests of the broad exact replay receipts (full campaign and
        /// library sweeps at the repaired revision).
        broad_replay_receipt_sha256s: Vec<[u8; 32]>,
    },
    /// The trace proved semantically equivalent physical-layout
    /// association; only the named controlled experiment is authorized.
    ControlledResidual {
        /// The precise aggregate-only hypothesis, in full.
        hypothesis: String,
        /// Digests of the six falsification arm specifications the V8
        /// producer must execute.
        falsification_arm_sha256s: [[u8; 32]; RESIDUAL_FALSIFICATION_ARMS],
    },
}

impl R1BoundaryVariantV1 {
    fn validate(&self) -> Result<(), R1ResolutionError> {
        match self {
            Self::ExactRepair {
                causal_patch_revisions,
                focused_replay_receipt_sha256s,
                broad_replay_receipt_sha256s,
            } => {
                if causal_patch_revisions.is_empty() {
                    return Err(R1ResolutionError::EmptyCausalPatch);
                }
                if causal_patch_revisions.len() > MAX_CAUSAL_PATCH_REVISIONS {
                    return Err(R1ResolutionError::TooManyCausalPatches);
                }
                for revision in causal_patch_revisions {
                    if !is_lowercase_40_hex(revision) {
                        return Err(R1ResolutionError::CausalPatchRevision);
                    }
                }
                let mut unique = causal_patch_revisions.clone();
                unique.sort_unstable();
                unique.dedup();
                if unique.len() != causal_patch_revisions.len() {
                    return Err(R1ResolutionError::DuplicateCausalPatch);
                }
                if focused_replay_receipt_sha256s.is_empty() {
                    return Err(R1ResolutionError::EmptyFocusedReplay);
                }
                if broad_replay_receipt_sha256s.is_empty() {
                    return Err(R1ResolutionError::EmptyBroadReplay);
                }
                for breadth in [focused_replay_receipt_sha256s, broad_replay_receipt_sha256s] {
                    if breadth.len() > MAX_REPLAY_RECEIPTS {
                        return Err(R1ResolutionError::TooManyReplayReceipts);
                    }
                    for digest in breadth {
                        if is_unset(digest) {
                            return Err(R1ResolutionError::UnsetReplayReceipt);
                        }
                    }
                }
                Ok(())
            }
            Self::ControlledResidual {
                hypothesis,
                falsification_arm_sha256s,
            } => {
                if hypothesis.trim().is_empty() {
                    return Err(R1ResolutionError::EmptyHypothesis);
                }
                if hypothesis.len() > MAX_HYPOTHESIS_BYTES {
                    return Err(R1ResolutionError::HypothesisTooLarge);
                }
                for digest in falsification_arm_sha256s {
                    if is_unset(digest) {
                        return Err(R1ResolutionError::UnsetFalsificationArm);
                    }
                }
                let mut unique = falsification_arm_sha256s.to_vec();
                unique.sort_unstable();
                unique.dedup();
                if unique.len() != falsification_arm_sha256s.len() {
                    return Err(R1ResolutionError::DuplicateFalsificationArm);
                }
                Ok(())
            }
        }
    }
}

/// The canonical R1 boundary resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct R1BoundaryResolutionV1 {
    pub schema: String,
    pub variant: R1BoundaryVariantV1,
    pub provenance: R1ResolutionProvenanceV1,
}

impl R1BoundaryResolutionV1 {
    /// Build and validate a resolution.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant.
    pub fn new(
        variant: R1BoundaryVariantV1,
        provenance: R1ResolutionProvenanceV1,
    ) -> Result<Self, R1ResolutionError> {
        let resolution = Self {
            schema: R1_BOUNDARY_RESOLUTION_SCHEMA_V1.to_owned(),
            variant,
            provenance,
        };
        resolution.validate()?;
        Ok(resolution)
    }

    /// Validate every invariant of the stored resolution.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant.
    pub fn validate(&self) -> Result<(), R1ResolutionError> {
        if self.schema != R1_BOUNDARY_RESOLUTION_SCHEMA_V1 {
            return Err(R1ResolutionError::Schema);
        }
        self.variant.validate()?;
        self.provenance.validate()
    }

    /// Canonical JSON bytes of a validated resolution.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant, or [`R1ResolutionError::Schema`]
    /// if the resolution cannot be serialized — never an empty byte string
    /// that would hash to a plausible-looking digest.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, R1ResolutionError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| R1ResolutionError::Schema)
    }

    /// Domain-separated content address of a validated resolution.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant.
    pub fn digest_sha256(&self) -> Result<[u8; 32], R1ResolutionError> {
        let bytes = self.canonical_bytes()?;
        let mut hasher = Sha256::new();
        hasher.update(R1_BOUNDARY_RESOLUTION_HASH_DOMAIN);
        hasher.update(&bytes);
        Ok(hasher.finalize().into())
    }

    /// Decode and validate a stored resolution.
    ///
    /// # Errors
    ///
    /// Returns [`R1ResolutionError::Schema`] for undecodable bytes or the
    /// first violated invariant.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, R1ResolutionError> {
        let resolution: Self =
            serde_json::from_slice(bytes).map_err(|_| R1ResolutionError::Schema)?;
        resolution.validate()?;
        Ok(resolution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn provenance() -> R1ResolutionProvenanceV1 {
        R1ResolutionProvenanceV1 {
            v7_freeze_sha256: digest(0x11),
            source_revision: "a".repeat(40),
            corpus_manifest_sha256: digest(0x22),
            query_set_sha256: digest(0x33),
            snapshot_sha256: digest(0x44),
            trace_receipt_sha256: digest(0x55),
        }
    }

    fn exact_repair() -> R1BoundaryVariantV1 {
        R1BoundaryVariantV1::ExactRepair {
            causal_patch_revisions: vec!["b".repeat(40), "c".repeat(40)],
            focused_replay_receipt_sha256s: vec![digest(0x66)],
            broad_replay_receipt_sha256s: vec![digest(0x77), digest(0x78)],
        }
    }

    fn controlled_residual() -> R1BoundaryVariantV1 {
        R1BoundaryVariantV1::ControlledResidual {
            hypothesis: "aggregate-only physical-layout association variance on the \
                         named shape; leaf, membership, rank, count, cutoff, metadata, \
                         snippet, explain, and error surfaces are exact"
                .to_owned(),
            falsification_arm_sha256s: [
                digest(0x81),
                digest(0x82),
                digest(0x83),
                digest(0x84),
                digest(0x85),
                digest(0x86),
            ],
        }
    }

    #[test]
    fn exact_repair_round_trips_twice_with_stable_digest() {
        let resolution =
            R1BoundaryResolutionV1::new(exact_repair(), provenance()).expect("valid resolution");
        let bytes = resolution.canonical_bytes().expect("canonical bytes");
        let first_digest = resolution.digest_sha256().expect("digest");

        let first_reload =
            R1BoundaryResolutionV1::from_canonical_bytes(&bytes).expect("first fresh reload");
        assert_eq!(first_reload, resolution);
        let reload_bytes = first_reload.canonical_bytes().expect("reload bytes");
        let second_reload = R1BoundaryResolutionV1::from_canonical_bytes(&reload_bytes)
            .expect("second fresh reload");
        assert_eq!(second_reload, resolution);
        assert_eq!(
            second_reload.digest_sha256().expect("reload digest"),
            first_digest
        );
    }

    #[test]
    fn controlled_residual_round_trips_twice_with_stable_digest() {
        let resolution = R1BoundaryResolutionV1::new(controlled_residual(), provenance())
            .expect("valid resolution");
        let bytes = resolution.canonical_bytes().expect("canonical bytes");
        let reload =
            R1BoundaryResolutionV1::from_canonical_bytes(&bytes).expect("first fresh reload");
        let reload_bytes = reload.canonical_bytes().expect("reload bytes");
        let second =
            R1BoundaryResolutionV1::from_canonical_bytes(&reload_bytes).expect("second reload");
        assert_eq!(second, resolution);
        assert_eq!(
            second.digest_sha256().expect("digest"),
            resolution.digest_sha256().expect("digest")
        );
    }

    #[test]
    fn digest_is_domain_separated_from_raw_bytes() {
        let resolution =
            R1BoundaryResolutionV1::new(exact_repair(), provenance()).expect("valid resolution");
        let bytes = resolution.canonical_bytes().expect("canonical bytes");
        let raw: [u8; 32] = Sha256::digest(&bytes).into();
        assert_ne!(
            resolution.digest_sha256().expect("digest"),
            raw,
            "the content address must be domain-separated"
        );
    }

    #[test]
    fn schema_mutation_rejects() {
        let mut resolution =
            R1BoundaryResolutionV1::new(exact_repair(), provenance()).expect("valid resolution");
        resolution.schema = "quill-gauntlet-r1-boundary-resolution-v0".to_owned();
        assert_eq!(resolution.validate(), Err(R1ResolutionError::Schema));
    }

    #[test]
    fn every_unset_provenance_digest_rejects() {
        for field in 0..5_usize {
            let mut provenance = provenance();
            match field {
                0 => provenance.v7_freeze_sha256 = [0; 32],
                1 => provenance.corpus_manifest_sha256 = [0; 32],
                2 => provenance.query_set_sha256 = [0; 32],
                3 => provenance.snapshot_sha256 = [0; 32],
                _ => provenance.trace_receipt_sha256 = [0; 32],
            }
            assert_eq!(
                R1BoundaryResolutionV1::new(exact_repair(), provenance).unwrap_err(),
                R1ResolutionError::UnsetDigest,
                "field {field} must reject when unset"
            );
        }
    }

    #[test]
    fn source_revision_mutations_reject() {
        for bad in ["", "abc", &"A".repeat(40), &"g".repeat(40), &"a".repeat(41)] {
            let mut provenance = provenance();
            provenance.source_revision = (*bad).to_owned();
            assert_eq!(
                R1BoundaryResolutionV1::new(exact_repair(), provenance).unwrap_err(),
                R1ResolutionError::SourceRevision,
                "revision {bad:?} must reject"
            );
        }
    }

    #[test]
    fn exact_repair_evidence_mutations_reject() {
        let empty_patch = R1BoundaryVariantV1::ExactRepair {
            causal_patch_revisions: vec![],
            focused_replay_receipt_sha256s: vec![digest(0x66)],
            broad_replay_receipt_sha256s: vec![digest(0x77)],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(empty_patch, provenance()).unwrap_err(),
            R1ResolutionError::EmptyCausalPatch
        );

        let duplicate_patch = R1BoundaryVariantV1::ExactRepair {
            causal_patch_revisions: vec!["b".repeat(40), "b".repeat(40)],
            focused_replay_receipt_sha256s: vec![digest(0x66)],
            broad_replay_receipt_sha256s: vec![digest(0x77)],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(duplicate_patch, provenance()).unwrap_err(),
            R1ResolutionError::DuplicateCausalPatch
        );

        let bad_patch = R1BoundaryVariantV1::ExactRepair {
            causal_patch_revisions: vec!["B".repeat(40)],
            focused_replay_receipt_sha256s: vec![digest(0x66)],
            broad_replay_receipt_sha256s: vec![digest(0x77)],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(bad_patch, provenance()).unwrap_err(),
            R1ResolutionError::CausalPatchRevision
        );

        let empty_focused = R1BoundaryVariantV1::ExactRepair {
            causal_patch_revisions: vec!["b".repeat(40)],
            focused_replay_receipt_sha256s: vec![],
            broad_replay_receipt_sha256s: vec![digest(0x77)],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(empty_focused, provenance()).unwrap_err(),
            R1ResolutionError::EmptyFocusedReplay
        );

        let empty_broad = R1BoundaryVariantV1::ExactRepair {
            causal_patch_revisions: vec!["b".repeat(40)],
            focused_replay_receipt_sha256s: vec![digest(0x66)],
            broad_replay_receipt_sha256s: vec![],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(empty_broad, provenance()).unwrap_err(),
            R1ResolutionError::EmptyBroadReplay
        );

        let unset_receipt = R1BoundaryVariantV1::ExactRepair {
            causal_patch_revisions: vec!["b".repeat(40)],
            focused_replay_receipt_sha256s: vec![[0; 32]],
            broad_replay_receipt_sha256s: vec![digest(0x77)],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(unset_receipt, provenance()).unwrap_err(),
            R1ResolutionError::UnsetReplayReceipt
        );

        let too_many = R1BoundaryVariantV1::ExactRepair {
            causal_patch_revisions: vec!["b".repeat(40)],
            focused_replay_receipt_sha256s: vec![digest(0x66); MAX_REPLAY_RECEIPTS + 1],
            broad_replay_receipt_sha256s: vec![digest(0x77)],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(too_many, provenance()).unwrap_err(),
            R1ResolutionError::TooManyReplayReceipts
        );
    }

    #[test]
    fn controlled_residual_mutations_reject() {
        let empty_hypothesis = R1BoundaryVariantV1::ControlledResidual {
            hypothesis: "   ".to_owned(),
            falsification_arm_sha256s: [
                digest(0x81),
                digest(0x82),
                digest(0x83),
                digest(0x84),
                digest(0x85),
                digest(0x86),
            ],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(empty_hypothesis, provenance()).unwrap_err(),
            R1ResolutionError::EmptyHypothesis
        );

        let oversized = R1BoundaryVariantV1::ControlledResidual {
            hypothesis: "h".repeat(MAX_HYPOTHESIS_BYTES + 1),
            falsification_arm_sha256s: [
                digest(0x81),
                digest(0x82),
                digest(0x83),
                digest(0x84),
                digest(0x85),
                digest(0x86),
            ],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(oversized, provenance()).unwrap_err(),
            R1ResolutionError::HypothesisTooLarge
        );

        let unset_arm = R1BoundaryVariantV1::ControlledResidual {
            hypothesis: "hypothesis".to_owned(),
            falsification_arm_sha256s: [
                digest(0x81),
                [0; 32],
                digest(0x83),
                digest(0x84),
                digest(0x85),
                digest(0x86),
            ],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(unset_arm, provenance()).unwrap_err(),
            R1ResolutionError::UnsetFalsificationArm
        );

        let duplicate_arm = R1BoundaryVariantV1::ControlledResidual {
            hypothesis: "hypothesis".to_owned(),
            falsification_arm_sha256s: [
                digest(0x81),
                digest(0x81),
                digest(0x83),
                digest(0x84),
                digest(0x85),
                digest(0x86),
            ],
        };
        assert_eq!(
            R1BoundaryResolutionV1::new(duplicate_arm, provenance()).unwrap_err(),
            R1ResolutionError::DuplicateFalsificationArm
        );
    }

    #[test]
    fn unknown_fields_and_variant_tamper_reject_at_decode() {
        let resolution =
            R1BoundaryResolutionV1::new(exact_repair(), provenance()).expect("valid resolution");
        let bytes = resolution.canonical_bytes().expect("canonical bytes");
        let mut value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("canonical json parses");

        let mut with_extra = value.clone();
        with_extra["waiver"] = serde_json::json!(true);
        let extra_bytes = serde_json::to_vec(&with_extra).expect("tampered json");
        assert_eq!(
            R1BoundaryResolutionV1::from_canonical_bytes(&extra_bytes).unwrap_err(),
            R1ResolutionError::Schema,
            "unknown fields must reject at decode"
        );

        value["variant"]["kind"] = serde_json::json!("controlled_residual");
        let flipped_bytes = serde_json::to_vec(&value).expect("flipped json");
        assert_eq!(
            R1BoundaryResolutionV1::from_canonical_bytes(&flipped_bytes).unwrap_err(),
            R1ResolutionError::Schema,
            "a variant flip without that variant's evidence must reject at decode"
        );
    }
}
