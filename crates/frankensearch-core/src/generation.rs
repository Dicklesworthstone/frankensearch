//! Generation manifest schema and validator for Native Mode distributed search.
//!
//! A *generation* represents a complete, consistent snapshot of all search artifacts
//! (vector indices, lexical segments, embedder metadata) built from a contiguous window
//! of document commits. Replicas atomically activate a generation to serve queries,
//! ensuring no mixed-generation reads within a single request.
//!
//! The [`GenerationManifest`] captures everything needed to replicate, verify, and
//! activate a generation on any node. The `ManifestValidator` enforces structural
//! and semantic invariants before activation is permitted.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::SearchError;

// ---------------------------------------------------------------------------
// Commit range
// ---------------------------------------------------------------------------

/// Contiguous range of commit sequence numbers that produced this generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitRange {
    /// First commit (inclusive) in the window.
    pub low: u64,
    /// Last commit (inclusive) in the window.
    pub high: u64,
}

impl CommitRange {
    /// Number of commits covered by this range.
    #[must_use]
    pub const fn len(&self) -> u64 {
        if self.high < self.low {
            return 0;
        }
        self.high - self.low + 1
    }

    /// Whether the range is empty (high < low after wrapping / invalid state).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.high < self.low
    }
}

/// Vector quantization format used in FSVI artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationFormat {
    /// IEEE 754 single-precision (32-bit).
    F32,
    /// IEEE 754 half-precision (16-bit). Default for frankensearch.
    F16,
    /// Signed 8-bit integer with per-vector scale factor.
    Int8,
    /// Signed 4-bit integer packed two per byte.
    Int4,
}

// ---------------------------------------------------------------------------
// Embedding identity contracts
// ---------------------------------------------------------------------------

/// Current schema for the mathematical embedding-space identity.
pub const EMBEDDING_SPACE_IDENTITY_SCHEMA_V1: u16 = 1;
/// Current schema for producer attestations.
pub const EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1: u16 = 1;
/// Current schema for outer embedding-input contracts.
pub const EMBEDDING_INPUT_CONTRACT_SCHEMA_V1: u16 = 1;
/// Current schema for vector-storage identities.
pub const VECTOR_STORAGE_IDENTITY_SCHEMA_V1: u16 = 1;
/// Current schema for explicit foreign-producer conformance certificates.
pub const FOREIGN_PRODUCER_CONFORMANCE_CERTIFICATE_SCHEMA_V1: u16 = 1;
/// Current schema for immutable artifact-generation identities.
pub const ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1: u16 = 1;

const MAX_IDENTITY_FIELD_BYTES: usize = 4_096;

/// Immutable identity of one published artifact generation.
///
/// `sequence` is a full-width monotone counter within one publication lineage;
/// it replaces the wrap-prone `u8` compaction generation in FSVI v1. The
/// caller-supplied 128-bit nonce distinguishes independently built generations
/// that happen to use the same sequence number. The nonce is identity material,
/// not a credential.
///
/// Equality is exact. Neither the sequence nor the nonce alone is a sufficient
/// generation identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactGenerationIdentityV1 {
    /// [`ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1`].
    pub schema_version: u16,
    /// Monotone generation sequence within the publisher's lineage.
    pub sequence: u64,
    /// Unique identity material for this build/publication attempt.
    pub nonce: [u8; 16],
}

impl ArtifactGenerationIdentityV1 {
    /// Construct and validate one artifact-generation identity.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the nonce is all zeroes.
    pub fn new(sequence: u64, nonce: [u8; 16]) -> Result<Self, SearchError> {
        let identity = Self {
            schema_version: ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1,
            sequence,
            nonce,
        };
        identity.validate()?;
        Ok(identity)
    }

    /// Validate the schema and reject the reserved all-zero nonce.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an unknown schema or an
    /// all-zero nonce.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "artifact_generation.schema_version",
            self.schema_version,
            ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1,
        )?;
        // ubs:ignore — this serialized generation-uniqueness nonce is public, not an authenticator.
        if self.nonce == [0; 16] {
            return Err(identity_error(
                "artifact_generation.nonce",
                "redacted-zero-nonce",
                "must contain unique non-zero generation identity material",
            ));
        }
        Ok(())
    }

    /// Canonical domain-separated bytes used by persisted FSVI and WAL
    /// bindings.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.artifact-generation-identity.v1");
        encoder.u16(self.schema_version);
        encoder.u64(self.sequence);
        encoder.bytes(&self.nonce);
        encoder.finish()
    }

    /// Lowercase SHA-256 of [`Self::canonical_bytes`].
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }
}

// ---------------------------------------------------------------------------
// Immutable generation authority
// ---------------------------------------------------------------------------

/// Schema carried by immutable generation-authority references and slot frames.
pub const GENERATION_AUTHORITY_SCHEMA_V1: u16 = 1;
/// Schema for the deterministic anti-rollback floor reference provider.
pub const ANTI_ROLLBACK_FLOOR_SCHEMA_V1: u16 = 1;
/// Exact byte size of one physical `AUTHORITY` slot frame.
pub const GENERATION_AUTHORITY_SLOT_BYTES_V1: usize = 4_096;
/// Maximum canonical activation-manifest size accepted before decoding.
pub const GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V1: usize = 4_096;
/// Exact byte size of one physical `LOCK` owner or attempt frame.
pub const GENERATION_LOCK_FRAME_BYTES_V1: usize = 4_096;

const AUTHORITY_SLOT_DIGEST_BYTES: usize = 32;
const AUTHORITY_SLOT_BODY_BYTES: usize =
    GENERATION_AUTHORITY_SLOT_BYTES_V1 - AUTHORITY_SLOT_DIGEST_BYTES;
const AUTHORITY_SLOT_MAGIC_V1: [u8; 8] = *b"FSAUTH01";
const AUTHORITY_SLOT_HEADER_BYTES: usize = 131;
const AUTHORITY_REF_MAGIC_V1: [u8; 9] = *b"FSAUTHREF";
const AUTHORITY_REF_BYTES_V1: usize = 108;
const LOCK_FRAME_DIGEST_BYTES: usize = 32;
const LOCK_FRAME_BODY_BYTES: usize = GENERATION_LOCK_FRAME_BYTES_V1 - LOCK_FRAME_DIGEST_BYTES;
const LOCK_FRAME_MAGIC_V1: [u8; 8] = *b"FSLOCK01";
const LOCK_FRAME_HEADER_BYTES: usize = 104;

/// A bounded reason why an authority reference, physical slot, or slot pair was
/// rejected. The error deliberately contains no paths, opaque object contents,
/// or unbounded caller data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GenerationAuthorityErrorV1 {
    /// A required fixed-format field had an unsupported or reserved value.
    #[error("generation authority has an invalid {field}")]
    InvalidField {
        /// Stable field identifier.
        field: &'static str,
    },
    /// A fixed-size frame was truncated or extended.
    #[error("generation authority slot has an invalid length")]
    InvalidSlotLength,
    /// The frame checksum did not authenticate its exact body bytes.
    #[error("generation authority slot checksum mismatch")]
    ChecksumMismatch,
    /// A frame was copied into the wrong physical slot.
    #[error("generation authority slot index mismatch")]
    SlotIndexMismatch,
    /// A frame was copied between authority roots.
    #[error("generation authority root mismatch")]
    RootMismatch,
    /// Canonical zero padding contained data.
    #[error("generation authority slot contains non-canonical padding")]
    NonCanonicalPadding,
    /// Both slots represented an equal-sequence fork.
    #[error("generation authority slots contain an equal-sequence fork")]
    EqualSequenceFork,
    /// A duplicate authority was not the permitted sequence-one genesis form.
    #[error("generation authority slots contain a non-genesis duplicate")]
    NonGenesisDuplicate,
    /// Both resolver inputs claimed the same physical authority slot.
    #[error("generation authority resolver received one physical slot twice")]
    DuplicatePhysicalSlot,
    /// A non-genesis authority was stored in the wrong physical slot.
    #[error("generation authority slot does not match its sequence parity")]
    SlotSequenceParity,
    /// The newer slot skipped a sequence or lacked an exact predecessor link.
    #[error("generation authority slots lack a consecutive predecessor link")]
    BrokenPredecessorLink,
    /// A lock frame belonged to a different authority root than the slots.
    #[error("generation authority lock root does not match the resolved authority")]
    LockRootMismatch,
    /// An owner lock did not attest the resolved authority head.
    #[error("generation authority lock does not attest the resolved authority")]
    LockAuthorityMismatch,
    /// A valid publication attempt remains unresolved and must be reconciled.
    #[error("generation authority publication attempt remains unresolved")]
    UnresolvedAttempt,
    /// The immutable authority counter cannot advance beyond `u64::MAX`.
    #[error("generation authority sequence is exhausted")]
    SequenceExhausted,
    /// A selected authority is missing or predates the required external floor.
    #[error("generation authority does not satisfy the required external floor")]
    AuthorityBelowFloor,
    /// A required-external profile was selected without an injected floor.
    #[error("generation authority required-external profile has no external floor")]
    ExternalFloorRequired,
    /// A mutation was attempted through an inspection-only profile.
    #[error("generation authority read-only profile forbids mutation")]
    ReadOnlyProfile,
    /// An anti-rollback floor compare-and-advance did not name the current
    /// exact record.
    #[error("generation authority anti-rollback floor compare-and-advance conflicted")]
    FloorCompareAndAdvanceConflict,
    /// An anti-rollback floor update did not advance its authority sequence.
    #[error("generation authority anti-rollback floor did not advance")]
    FloorSequenceRegression,
    /// An idempotency key was reused for a different floor operation.
    #[error("generation authority anti-rollback floor idempotency key conflicted")]
    FloorIdempotencyConflict,
    /// The deterministic reference provider's CAS version was exhausted.
    #[error("generation authority anti-rollback floor CAS version is exhausted")]
    FloorVersionExhausted,
    /// The deterministic reference provider cannot safely continue after a
    /// poisoned lock.
    #[error("generation authority anti-rollback floor store is unavailable")]
    FloorStoreUnavailable,
    /// The immutable activation manifest did not match its stored self-seal.
    #[error("generation activation manifest self-seal mismatch")]
    ManifestSelfSealMismatch,
    /// An externally addressed manifest did not match its authority reference.
    #[error("generation activation manifest does not match its authority reference")]
    ManifestReferenceMismatch,
}

/// Immutable pointer to one activation-manifest object.
///
/// This is intentionally distinct from [`ArtifactGenerationIdentityV1`]: the
/// latter identifies one artifact build, while this counter is the monotone
/// activation authority for a root. Object IDs are random opaque bytes, never a
/// filename or display name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AuthorityRefV1 {
    /// [`GENERATION_AUTHORITY_SCHEMA_V1`].
    pub schema_version: u16,
    /// Monotone activation sequence, beginning at one.
    pub sequence: u64,
    /// Opaque immutable activation-manifest object identifier.
    pub object_id: [u8; 16],
    /// Exact activation-manifest byte length.
    pub manifest_len: u64,
    /// SHA-256 of the exact activation-manifest bytes.
    pub manifest_sha256: [u8; 32],
    /// SHA-256 fingerprint of the immediately preceding authority reference.
    /// Genesis has no predecessor.
    pub predecessor: Option<[u8; 32]>,
}

impl AuthorityRefV1 {
    /// Construct and validate one immutable activation authority.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error when a reserved identity value is used or
    /// the sequence/predecessor relation is not canonical.
    pub fn new(
        sequence: u64,
        object_id: [u8; 16],
        manifest_len: u64,
        manifest_sha256: [u8; 32],
        predecessor: Option<[u8; 32]>,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let reference = Self {
            schema_version: GENERATION_AUTHORITY_SCHEMA_V1,
            sequence,
            object_id,
            manifest_len,
            manifest_sha256,
            predecessor,
        };
        reference.validate()?;
        Ok(reference)
    }

    /// Validate the fixed, forward-safe authority-reference contract.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error for every invalid field.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.schema_version != GENERATION_AUTHORITY_SCHEMA_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.schema_version",
            });
        }
        if self.sequence == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.sequence",
            });
        }
        if self.object_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.object_id",
            });
        }
        if self.manifest_len == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.manifest_len",
            });
        }
        if self.manifest_sha256 == [0; 32] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.manifest_sha256",
            });
        }
        if self.sequence == 1 && self.predecessor.is_some() {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.predecessor",
            });
        }
        if self.sequence > 1 && self.predecessor.is_none() {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.predecessor",
            });
        }
        if self.predecessor == Some([0; 32]) {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.predecessor",
            });
        }
        Ok(())
    }

    /// Return the next authority sequence without allowing counter rollover.
    ///
    /// # Errors
    ///
    /// Returns a typed exhaustion error for the terminal `u64::MAX` authority
    /// rather than permitting an ABA-like wrap to an earlier sequence.
    pub fn next_sequence(&self) -> Result<u64, GenerationAuthorityErrorV1> {
        self.validate()?;
        self.sequence
            .checked_add(1)
            .ok_or(GenerationAuthorityErrorV1::SequenceExhausted)
    }

    /// Canonical bytes used to link consecutive authority references.
    #[must_use]
    pub fn canonical_bytes(&self) -> [u8; AUTHORITY_REF_BYTES_V1] {
        let mut bytes = [0_u8; AUTHORITY_REF_BYTES_V1];
        bytes[..9].copy_from_slice(&AUTHORITY_REF_MAGIC_V1);
        bytes[9..11].copy_from_slice(&self.schema_version.to_be_bytes());
        bytes[11..19].copy_from_slice(&self.sequence.to_be_bytes());
        bytes[19..35].copy_from_slice(&self.object_id);
        bytes[35..43].copy_from_slice(&self.manifest_len.to_be_bytes());
        bytes[43..75].copy_from_slice(&self.manifest_sha256);
        bytes[75] = u8::from(self.predecessor.is_some());
        if let Some(predecessor) = self.predecessor {
            bytes[76..].copy_from_slice(&predecessor);
        }
        bytes
    }

    /// Decode the exact fixed canonical representation of one authority
    /// reference without allocating from caller-controlled lengths.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed, future, or non-canonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() != AUTHORITY_REF_BYTES_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.canonical_bytes",
            });
        }
        if !bytes[..9].eq(AUTHORITY_REF_MAGIC_V1.as_slice()) {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.magic",
            });
        }
        let predecessor = match bytes[75] {
            0 => {
                if !bytes[76..].iter().all(|byte| byte.eq(&0)) {
                    return Err(GenerationAuthorityErrorV1::NonCanonicalPadding);
                }
                None
            }
            1 => Some(bytes[76..].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.predecessor",
                }
            })?),
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.predecessor_present",
                });
            }
        };
        let reference = Self {
            schema_version: u16::from_be_bytes([bytes[9], bytes[10]]),
            sequence: u64::from_be_bytes(bytes[11..19].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.sequence",
                }
            })?),
            object_id: bytes[19..35].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.object_id",
                }
            })?,
            manifest_len: u64::from_be_bytes(bytes[35..43].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.manifest_len",
                }
            })?),
            manifest_sha256: bytes[43..75].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_ref.manifest_sha256",
                }
            })?,
            predecessor,
        };
        reference.validate()?;
        if !reference.canonical_bytes().eq(bytes) {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.canonical_bytes",
            });
        }
        Ok(reference)
    }

    /// SHA-256 fingerprint of this exact canonical reference.
    #[must_use]
    pub fn fingerprint(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_bytes()).into()
    }
}

/// A decoded, authenticated physical authority slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthoritySlotV1 {
    /// Physical slot selected by the publisher (`0` or `1`).
    pub slot_index: u8,
    /// Root identity bound into the frame.
    pub root_id: [u8; 16],
    /// Immutable authority reference carried by the frame.
    pub authority: AuthorityRefV1,
}

/// An externally retained immutable authority that bounds acceptable recovery.
///
/// It is pure evidence, not a filesystem pointer: consumers may require an
/// exact floor or its immediately predecessor-linked successor without
/// silently accepting a stale or unprovable authority head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AuthorityFloorV1 {
    /// Root identity that prevents a floor from being replayed across roots.
    pub root_id: [u8; 16],
    /// Exact externally retained authority reference.
    pub authority: AuthorityRefV1,
}

impl AuthorityFloorV1 {
    /// Construct and validate one bounded external authority floor.
    ///
    /// # Errors
    ///
    /// Returns a typed error for reserved root or authority values.
    pub fn new(
        root_id: [u8; 16],
        authority: AuthorityRefV1,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let floor = Self { root_id, authority };
        floor.validate()?;
        Ok(floor)
    }

    fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.root_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_floor.root_id",
            });
        }
        self.authority.validate()
    }
}

/// Explicit security posture for opening one generation root.
///
/// There is deliberately no `Default` implementation: a caller must choose a
/// posture instead of silently falling back from an externally anchored root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationRootSecurityProfileV1 {
    /// Require a consumer-owned, externally retained authority floor before a
    /// root can be selected for use.
    RequiredExternal,
    /// Permit owner-controlled local recovery while reporting that it cannot
    /// detect a hostile whole-store rollback.
    CooperativeLocal,
    /// Permit inspection only; publication, repair, rollback, and garbage
    /// collection must be rejected by the caller before mutation.
    ReadOnlyUnanchored,
}

impl GenerationRootSecurityProfileV1 {
    /// Whether this profile authorizes a caller to mutate a generation root.
    #[must_use]
    pub const fn permits_mutation(self) -> bool {
        !matches!(self, Self::ReadOnlyUnanchored)
    }

    /// Reject a mutation through an inspection-only profile.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationAuthorityErrorV1::ReadOnlyProfile`] when this is
    /// [`Self::ReadOnlyUnanchored`].
    pub const fn require_mutation_authorized(self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.permits_mutation() {
            Ok(())
        } else {
            Err(GenerationAuthorityErrorV1::ReadOnlyProfile)
        }
    }
}

/// Versioned, digest-bound result of one successful anti-rollback floor CAS.
///
/// The record is intentionally independent of any generation-root path: its
/// caller owns storage and replication of this receipt outside the replaceable
/// root it anchors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AntiRollbackFloorRecordV1 {
    /// [`ANTI_ROLLBACK_FLOOR_SCHEMA_V1`].
    pub schema_version: u16,
    /// Generation root anchored by this record.
    pub root_id: [u8; 16],
    /// Exact externally retained authority floor.
    pub authority: AuthorityRefV1,
    /// Monotone successful compare-and-advance version for this root.
    pub cas_version: u64,
    /// SHA-256 over the canonical record fields above.
    pub record_sha256: [u8; 32],
}

impl AntiRollbackFloorRecordV1 {
    fn new(floor: AuthorityFloorV1, cas_version: u64) -> Result<Self, GenerationAuthorityErrorV1> {
        floor.validate()?;
        if cas_version == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.cas_version",
            });
        }
        let mut record = Self {
            schema_version: ANTI_ROLLBACK_FLOOR_SCHEMA_V1,
            root_id: floor.root_id,
            authority: floor.authority,
            cas_version,
            record_sha256: [0; 32],
        };
        record.record_sha256 = record.computed_record_sha256();
        Ok(record)
    }

    fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.schema_version != ANTI_ROLLBACK_FLOOR_SCHEMA_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.schema_version",
            });
        }
        AuthorityFloorV1::new(self.root_id, self.authority)?;
        if self.cas_version == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.cas_version",
            });
        }
        if self.record_sha256 != self.computed_record_sha256() {
            return Err(GenerationAuthorityErrorV1::ChecksumMismatch);
        }
        Ok(())
    }

    fn computed_record_sha256(&self) -> [u8; 32] {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.anti-rollback-floor-record.v1");
        encoder.u16(self.schema_version);
        encoder.bytes(&self.root_id);
        encoder.bytes(&self.authority.canonical_bytes());
        encoder.u64(self.cas_version);
        Sha256::digest(encoder.finish()).into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AntiRollbackFloorRequestV1 {
    root_id: [u8; 16],
    expected_record_sha256: Option<[u8; 32]>,
    next_authority: AuthorityRefV1,
    result: AntiRollbackFloorRecordV1,
}

#[derive(Default)]
struct AntiRollbackFloorStoreStateV1 {
    records: BTreeMap<[u8; 16], AntiRollbackFloorRecordV1>,
    requests: BTreeMap<[u8; 16], AntiRollbackFloorRequestV1>,
}

/// Deterministic, linearizable in-memory reference provider for anti-rollback
/// floor conformance tests and consumer integration tests.
///
/// This provider is deliberately not a crash-durable authority. Production
/// callers must inject a consumer-owned provider outside the generation root;
/// this implementation makes the exact load/CAS/idempotency contract testable
/// without selecting a storage backend.
#[derive(Default)]
pub struct InMemoryAntiRollbackFloorStoreV1 {
    state: std::sync::Mutex<AntiRollbackFloorStoreStateV1>,
}

impl InMemoryAntiRollbackFloorStoreV1 {
    /// Construct an empty deterministic reference provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the exact current floor record for one root.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationAuthorityErrorV1::FloorStoreUnavailable`] when a
    /// prior panic poisoned the reference-provider lock.
    pub fn load(
        &self,
        root_id: [u8; 16],
    ) -> Result<Option<AntiRollbackFloorRecordV1>, GenerationAuthorityErrorV1> {
        if root_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.root_id",
            });
        }
        let state = self
            .state
            .lock()
            .map_err(|_| GenerationAuthorityErrorV1::FloorStoreUnavailable)?;
        Ok(state.records.get(&root_id).copied())
    }

    /// Atomically compare and advance one root's anti-rollback floor.
    ///
    /// `expected` must be the exact record returned by a prior [`Self::load`]
    /// call (or `None` for a previously unanchored root). Repeating the same
    /// operation with the same non-zero `idempotency_key` returns the original
    /// successful receipt; reusing that key for another operation fails closed.
    ///
    /// # Errors
    ///
    /// Returns a typed conflict, regression, idempotency, exhaustion, or store
    /// availability error without changing the recorded floor.
    pub fn compare_and_advance(
        &self,
        expected: Option<AntiRollbackFloorRecordV1>,
        next: AuthorityFloorV1,
        idempotency_key: [u8; 16],
    ) -> Result<AntiRollbackFloorRecordV1, GenerationAuthorityErrorV1> {
        next.validate()?;
        if idempotency_key == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.idempotency_key",
            });
        }
        if let Some(expected) = expected {
            expected.validate()?;
            if expected.root_id != next.root_id {
                return Err(GenerationAuthorityErrorV1::RootMismatch);
            }
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| GenerationAuthorityErrorV1::FloorStoreUnavailable)?;
        let expected_record_sha256 = expected.map(|record| record.record_sha256);
        if let Some(previous) = state.requests.get(&idempotency_key) {
            if previous.root_id == next.root_id
                && previous.expected_record_sha256 == expected_record_sha256
                && previous.next_authority == next.authority
            {
                return Ok(previous.result);
            }
            return Err(GenerationAuthorityErrorV1::FloorIdempotencyConflict);
        }

        let current = state.records.get(&next.root_id).copied();
        if current != expected {
            return Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict);
        }
        if current.is_some_and(|record| next.authority.sequence <= record.authority.sequence) {
            return Err(GenerationAuthorityErrorV1::FloorSequenceRegression);
        }
        let cas_version = current
            .map(|record| record.cas_version)
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(GenerationAuthorityErrorV1::FloorVersionExhausted)?;
        let result = AntiRollbackFloorRecordV1::new(next, cas_version)?;
        state.records.insert(next.root_id, result);
        state.requests.insert(
            idempotency_key,
            AntiRollbackFloorRequestV1 {
                root_id: next.root_id,
                expected_record_sha256,
                next_authority: next.authority,
                result,
            },
        );
        Ok(result)
    }
}

/// Kind of fixed `LOCK` frame. Owner and attempt remain distinct on disk so an
/// interrupted attempt cannot be silently mistaken for authority ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationLockFrameKindV1 {
    /// Current writer ownership record.
    Owner,
    /// In-progress publication attempt record.
    Attempt,
}

impl GenerationLockFrameKindV1 {
    const fn tag(self) -> u8 {
        match self {
            Self::Owner => 1,
            Self::Attempt => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, GenerationAuthorityErrorV1> {
        match tag {
            1 => Ok(Self::Owner),
            2 => Ok(Self::Attempt),
            _ => Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.kind",
            }),
        }
    }
}

/// Bounded owner/attempt evidence for the fixed generation-root `LOCK` path.
///
/// It is an immutable byte contract only: taking a filesystem lock or deciding
/// whether an attempt may publish belongs to the publisher lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationLockFrameV1 {
    /// Owner or attempt frame classification.
    pub kind: GenerationLockFrameKindV1,
    /// Root ID that prevents cross-root frame replay.
    pub root_id: [u8; 16],
    /// Opaque writer identity.
    pub writer_id: [u8; 16],
    /// Opaque immutable publication attempt identity.
    pub attempt_id: [u8; 16],
    /// Monotone writer fence, beginning at one.
    pub fence: u64,
    /// Fingerprint of the authority reference the lock state observed.
    pub authority_fingerprint: [u8; 32],
}

impl GenerationLockFrameV1 {
    /// Construct and validate a bounded lock owner/attempt frame.
    ///
    /// # Errors
    ///
    /// Returns a typed error when any identity or fence is reserved.
    pub fn new(
        kind: GenerationLockFrameKindV1,
        root_id: [u8; 16],
        writer_id: [u8; 16],
        attempt_id: [u8; 16],
        fence: u64,
        authority_fingerprint: [u8; 32],
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let frame = Self {
            kind,
            root_id,
            writer_id,
            attempt_id,
            fence,
            authority_fingerprint,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Validate the fixed lock-frame facts.
    ///
    /// # Errors
    ///
    /// Returns a typed error when any fact is reserved or cannot identify a
    /// specific publication attempt.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.root_id == [0; 16]
            || self.writer_id == [0; 16]
            || self.attempt_id == [0; 16]
            || self.fence == 0
            || self.authority_fingerprint == [0; 32]
        {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.frame",
            });
        }
        Ok(())
    }

    /// Encode this owner/attempt frame into its fixed canonical form.
    ///
    /// # Errors
    ///
    /// Returns a typed error instead of serializing a reserved frame.
    pub fn encode(
        self,
    ) -> Result<[u8; GENERATION_LOCK_FRAME_BYTES_V1], GenerationAuthorityErrorV1> {
        self.validate()?;
        let mut bytes = [0_u8; GENERATION_LOCK_FRAME_BYTES_V1];
        bytes[..8].copy_from_slice(&LOCK_FRAME_MAGIC_V1);
        bytes[8..10].copy_from_slice(&GENERATION_AUTHORITY_SCHEMA_V1.to_be_bytes());
        bytes[10] = self.kind.tag();
        bytes[16..32].copy_from_slice(&self.root_id);
        bytes[32..48].copy_from_slice(&self.writer_id);
        bytes[48..64].copy_from_slice(&self.attempt_id);
        bytes[64..72].copy_from_slice(&self.fence.to_be_bytes());
        bytes[72..104].copy_from_slice(&self.authority_fingerprint);
        let digest = Sha256::digest(&bytes[..LOCK_FRAME_BODY_BYTES]);
        bytes[LOCK_FRAME_BODY_BYTES..].copy_from_slice(&digest);
        Ok(bytes)
    }

    /// Parse and authenticate a fixed `LOCK` frame for the expected root.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed, copied, tampered, or noncanonical
    /// owner/attempt frames.
    pub fn from_authenticated_bytes(
        bytes: &[u8],
        expected_root_id: [u8; 16],
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() != GENERATION_LOCK_FRAME_BYTES_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidSlotLength);
        }
        if !bytes[..8].eq(LOCK_FRAME_MAGIC_V1.as_slice())
            || !bytes[8..10].eq(GENERATION_AUTHORITY_SCHEMA_V1.to_be_bytes().as_slice())
            || !bytes[11..16].iter().all(|byte| byte.eq(&0))
        {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.header",
            });
        }
        if !bytes[16..32].eq(expected_root_id.as_slice()) {
            return Err(GenerationAuthorityErrorV1::RootMismatch);
        }
        let expected_digest = Sha256::digest(&bytes[..LOCK_FRAME_BODY_BYTES]);
        if !bytes[LOCK_FRAME_BODY_BYTES..].eq(&expected_digest[..]) {
            return Err(GenerationAuthorityErrorV1::ChecksumMismatch);
        }
        if bytes[LOCK_FRAME_HEADER_BYTES..LOCK_FRAME_BODY_BYTES]
            .iter()
            .any(|byte| !byte.eq(&0))
        {
            return Err(GenerationAuthorityErrorV1::NonCanonicalPadding);
        }
        let mut writer_id = [0_u8; 16];
        writer_id.copy_from_slice(&bytes[32..48]);
        let mut attempt_id = [0_u8; 16];
        attempt_id.copy_from_slice(&bytes[48..64]);
        let mut authority_fingerprint = [0_u8; 32];
        authority_fingerprint.copy_from_slice(&bytes[72..104]);
        Self::new(
            GenerationLockFrameKindV1::from_tag(bytes[10])?,
            expected_root_id,
            writer_id,
            attempt_id,
            u64::from_be_bytes(bytes[64..72].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "generation_lock.fence",
                }
            })?),
            authority_fingerprint,
        )
    }
}

impl AuthoritySlotV1 {
    /// Construct one physical authority-slot frame.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error when the physical slot/root or its
    /// authority reference is invalid.
    pub fn new(
        slot_index: u8,
        root_id: [u8; 16],
        authority: AuthorityRefV1,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let slot = Self {
            slot_index,
            root_id,
            authority,
        };
        slot.validate()?;
        Ok(slot)
    }

    /// Validate this in-memory physical slot before encoding or resolving it.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error for invalid physical binding data.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.slot_index > 1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.slot_index",
            });
        }
        if self.root_id == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.root_id",
            });
        }
        self.authority.validate()
    }

    /// Encode this slot into its fixed 4096-byte canonical representation.
    ///
    /// # Errors
    ///
    /// Returns a bounded typed error instead of serializing an invalid slot.
    pub fn encode(
        self,
    ) -> Result<[u8; GENERATION_AUTHORITY_SLOT_BYTES_V1], GenerationAuthorityErrorV1> {
        self.validate()?;
        let mut bytes = [0_u8; GENERATION_AUTHORITY_SLOT_BYTES_V1];
        bytes[..8].copy_from_slice(&AUTHORITY_SLOT_MAGIC_V1);
        bytes[8..10].copy_from_slice(&GENERATION_AUTHORITY_SCHEMA_V1.to_be_bytes());
        bytes[10] = self.slot_index;
        bytes[16..32].copy_from_slice(&self.root_id);
        bytes[32..34].copy_from_slice(&self.authority.schema_version.to_be_bytes());
        bytes[34..42].copy_from_slice(&self.authority.sequence.to_be_bytes());
        bytes[42..58].copy_from_slice(&self.authority.object_id);
        bytes[58..66].copy_from_slice(&self.authority.manifest_len.to_be_bytes());
        bytes[66..98].copy_from_slice(&self.authority.manifest_sha256);
        bytes[98] = u8::from(self.authority.predecessor.is_some());
        if let Some(predecessor) = self.authority.predecessor {
            bytes[99..131].copy_from_slice(&predecessor);
        }
        let digest = Sha256::digest(&bytes[..AUTHORITY_SLOT_BODY_BYTES]);
        bytes[AUTHORITY_SLOT_BODY_BYTES..].copy_from_slice(&digest);
        Ok(bytes)
    }

    /// Decode and authenticate one physical slot for the expected root and
    /// physical index.
    ///
    /// # Errors
    ///
    /// Returns a typed error for checksum, root, index, canonical-padding, or
    /// authority-reference violations.
    pub fn from_authenticated_bytes(
        bytes: &[u8],
        expected_slot_index: u8,
        expected_root_id: [u8; 16],
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() != GENERATION_AUTHORITY_SLOT_BYTES_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidSlotLength);
        }
        if expected_slot_index > 1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.expected_slot_index",
            });
        }
        if !bytes[..8].eq(AUTHORITY_SLOT_MAGIC_V1.as_slice())
            || !bytes[8..10].eq(GENERATION_AUTHORITY_SCHEMA_V1.to_be_bytes().as_slice())
            || !bytes[11..16].iter().all(|byte| byte.eq(&0))
        {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.header",
            });
        }
        if !bytes[10].eq(&expected_slot_index) {
            return Err(GenerationAuthorityErrorV1::SlotIndexMismatch);
        }
        if !bytes[16..32].eq(expected_root_id.as_slice()) {
            return Err(GenerationAuthorityErrorV1::RootMismatch);
        }
        let expected_digest = Sha256::digest(&bytes[..AUTHORITY_SLOT_BODY_BYTES]);
        if !bytes[AUTHORITY_SLOT_BODY_BYTES..].eq(&expected_digest[..]) {
            return Err(GenerationAuthorityErrorV1::ChecksumMismatch);
        }
        if bytes[AUTHORITY_SLOT_HEADER_BYTES..AUTHORITY_SLOT_BODY_BYTES]
            .iter()
            .any(|byte| !byte.eq(&0))
        {
            return Err(GenerationAuthorityErrorV1::NonCanonicalPadding);
        }

        let mut object_id = [0_u8; 16];
        object_id.copy_from_slice(&bytes[42..58]);
        let mut manifest_sha256 = [0_u8; 32];
        manifest_sha256.copy_from_slice(&bytes[66..98]);
        let predecessor = match bytes[98] {
            0 => {
                if !bytes[99..131].iter().all(|byte| byte.eq(&0)) {
                    return Err(GenerationAuthorityErrorV1::NonCanonicalPadding);
                }
                None
            }
            1 => {
                let mut predecessor = [0_u8; 32];
                predecessor.copy_from_slice(&bytes[99..131]);
                Some(predecessor)
            }
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_slot.predecessor_present",
                });
            }
        };
        let authority = AuthorityRefV1 {
            schema_version: u16::from_be_bytes([bytes[32], bytes[33]]),
            sequence: u64::from_be_bytes(bytes[34..42].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_slot.sequence",
                }
            })?),
            object_id,
            manifest_len: u64::from_be_bytes(bytes[58..66].try_into().map_err(|_| {
                GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_slot.manifest_len",
                }
            })?),
            manifest_sha256,
            predecessor,
        };
        authority.validate()?;
        Self::new(expected_slot_index, expected_root_id, authority)
    }
}

/// Resolve the two physical authority slots without performing I/O.
///
/// A single surviving valid slot remains usable. Two different slots must be
/// exactly consecutive and predecessor-linked; an equal reference is legal
/// only for the duplicated sequence-one genesis record.
///
/// # Errors
///
/// Returns a typed error instead of selecting a structurally credible fork or
/// skipping a missing authority history edge.
pub fn resolve_authority_slots_v1(
    first: Option<AuthoritySlotV1>,
    second: Option<AuthoritySlotV1>,
) -> Result<Option<AuthoritySlotV1>, GenerationAuthorityErrorV1> {
    if let Some(slot) = first {
        slot.validate()?;
    }
    if let Some(slot) = second {
        slot.validate()?;
    }
    match (first, second) {
        (None, None) => Ok(None),
        (Some(slot), None) | (None, Some(slot)) => {
            if !slot_matches_authority_sequence(slot) {
                return Err(GenerationAuthorityErrorV1::SlotSequenceParity);
            }
            Ok(Some(slot))
        }
        (Some(first), Some(second)) => {
            // ubs:ignore — slot indices and root IDs are public structural identities.
            if first.root_id != second.root_id {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "authority_slot.pair",
                });
            }
            // ubs:ignore — physical slot indices are public structural identities.
            if first.slot_index == second.slot_index {
                return Err(GenerationAuthorityErrorV1::DuplicatePhysicalSlot);
            }
            // ubs:ignore — authority sequences are public monotone counters.
            if first.authority.sequence == second.authority.sequence {
                // ubs:ignore — authority references carry only public immutable identities.
                if first.authority != second.authority {
                    return Err(GenerationAuthorityErrorV1::EqualSequenceFork);
                }
                if first.authority.sequence != 1 {
                    return Err(GenerationAuthorityErrorV1::NonGenesisDuplicate);
                }
                return Ok(Some(first));
            }

            let (older, newer) = if first.authority.sequence < second.authority.sequence {
                (first, second)
            } else {
                (second, first)
            };
            if !slot_matches_authority_sequence(older) || !slot_matches_authority_sequence(newer) {
                return Err(GenerationAuthorityErrorV1::SlotSequenceParity);
            }
            // ubs:ignore — sequence/predecessor fingerprints are public history identities.
            if newer.authority.sequence != older.authority.sequence.saturating_add(1)
                // ubs:ignore — predecessor fingerprints are public immutable history identities.
                || newer.authority.predecessor != Some(older.authority.fingerprint())
            {
                return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
            }
            Ok(Some(newer))
        }
    }
}

/// Decode and resolve the two physical authority frames without filesystem
/// access.
///
/// A present but malformed frame is never converted to `None`: callers must
/// reconcile that corruption explicitly rather than treating it as one-slot
/// survival. Only an absent physical frame may enter the survival path.
///
/// # Errors
///
/// Returns the exact bounded slot decode error for malformed, copied, or
/// cross-root frames before attempting pair resolution.
pub fn resolve_authority_slot_frames_v1(
    first: Option<&[u8]>,
    second: Option<&[u8]>,
    expected_root_id: [u8; 16],
) -> Result<Option<AuthoritySlotV1>, GenerationAuthorityErrorV1> {
    let first = first
        .map(|bytes| AuthoritySlotV1::from_authenticated_bytes(bytes, 0, expected_root_id))
        .transpose()?;
    let second = second
        .map(|bytes| AuthoritySlotV1::from_authenticated_bytes(bytes, 1, expected_root_id))
        .transpose()?;
    resolve_authority_slots_v1(first, second)
}

/// Resolve two authority slots against one exact externally retained floor.
///
/// The selected authority must be the exact floor or its immediate
/// predecessor-linked successor. A later but unprovable head is not selected;
/// callers must supply its intervening authority evidence explicitly.
///
/// # Errors
///
/// Returns a typed error for missing/stale authorities, cross-root floors,
/// equal-sequence divergence, gaps, or sequence exhaustion.
pub fn resolve_authority_slots_at_floor_v1(
    first: Option<AuthoritySlotV1>,
    second: Option<AuthoritySlotV1>,
    floor: AuthorityFloorV1,
) -> Result<AuthoritySlotV1, GenerationAuthorityErrorV1> {
    let resolved = resolve_authority_slots_v1(first, second)?
        .ok_or(GenerationAuthorityErrorV1::AuthorityBelowFloor)?;
    resolve_selected_authority_at_floor_v1(resolved, floor)
}

fn resolve_selected_authority_at_floor_v1(
    resolved: AuthoritySlotV1,
    floor: AuthorityFloorV1,
) -> Result<AuthoritySlotV1, GenerationAuthorityErrorV1> {
    floor.validate()?;
    // ubs:ignore — root IDs are public structural identities.
    if resolved.root_id != floor.root_id {
        return Err(GenerationAuthorityErrorV1::RootMismatch);
    }
    if resolved.authority.sequence < floor.authority.sequence {
        return Err(GenerationAuthorityErrorV1::AuthorityBelowFloor);
    }
    // ubs:ignore — authority sequences are public monotone counters.
    if resolved.authority.sequence == floor.authority.sequence {
        // ubs:ignore — authority references are public immutable identities.
        if resolved.authority != floor.authority {
            return Err(GenerationAuthorityErrorV1::EqualSequenceFork);
        }
        return Ok(resolved);
    }
    let successor = floor
        .authority
        .next_sequence()
        .map_err(|_| GenerationAuthorityErrorV1::SequenceExhausted)?;
    // ubs:ignore — authority sequences are public monotone counters.
    if resolved.authority.sequence != successor {
        return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
    }
    // ubs:ignore — predecessor fingerprints are public immutable history identities.
    if resolved.authority.predecessor != Some(floor.authority.fingerprint()) {
        return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
    }
    Ok(resolved)
}

/// Resolve authority slots while refusing to silently step around a valid
/// owner/attempt `LOCK` state.
///
/// A persisted attempt is evidence of an outcome the caller has not yet
/// reconciled, so it fails closed even if an older authority pair resolves.
/// The helper performs no filesystem I/O and does not mutate either frame.
///
/// # Errors
///
/// Returns a bounded typed error when frame kind, root binding, owner binding,
/// or a pending attempt prevents a safe selection.
pub fn resolve_authority_slots_with_locks_v1(
    first: Option<AuthoritySlotV1>,
    second: Option<AuthoritySlotV1>,
    owner: Option<GenerationLockFrameV1>,
    attempt: Option<GenerationLockFrameV1>,
) -> Result<Option<AuthoritySlotV1>, GenerationAuthorityErrorV1> {
    let resolved = resolve_authority_slots_v1(first, second)?;
    if let Some(owner) = owner {
        owner.validate()?;
        // ubs:ignore — lock kinds are public fixed-format state tags.
        if owner.kind != GenerationLockFrameKindV1::Owner {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.owner.kind",
            });
        }
    }
    if let Some(attempt) = attempt {
        attempt.validate()?;
        // ubs:ignore — lock kinds are public fixed-format state tags.
        if attempt.kind != GenerationLockFrameKindV1::Attempt {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.attempt.kind",
            });
        }
    }
    if let (Some(owner), Some(attempt)) = (owner, attempt) {
        // ubs:ignore — root IDs are public structural identities.
        if owner.root_id != attempt.root_id {
            return Err(GenerationAuthorityErrorV1::LockRootMismatch);
        }
    }
    if let Some(slot) = resolved {
        // ubs:ignore — root IDs are public structural identities.
        if owner.is_some_and(|frame| frame.root_id != slot.root_id) {
            return Err(GenerationAuthorityErrorV1::LockRootMismatch);
        }
        // ubs:ignore — root IDs are public structural identities.
        if attempt.is_some_and(|frame| frame.root_id != slot.root_id) {
            return Err(GenerationAuthorityErrorV1::LockRootMismatch);
        }
    }
    if attempt.is_some() {
        return Err(GenerationAuthorityErrorV1::UnresolvedAttempt);
    }
    if let Some(owner) = owner {
        let resolved = resolved.ok_or(GenerationAuthorityErrorV1::LockAuthorityMismatch)?;
        // ubs:ignore — authority fingerprints are public immutable-integrity identities.
        if owner.authority_fingerprint != resolved.authority.fingerprint() {
            return Err(GenerationAuthorityErrorV1::LockAuthorityMismatch);
        }
    }
    Ok(resolved)
}

/// Resolve authority slots under one explicit generation-root security profile.
///
/// The lock-aware resolver always runs first, so an unresolved publication
/// attempt cannot be bypassed by selecting a weaker profile. Required external
/// roots additionally require an injected retained floor; local and inspection
/// profiles intentionally remain unanchored and are distinguishable to callers.
///
/// # Errors
///
/// Returns a typed error when lock reconciliation fails, an external floor is
/// absent or not satisfied, or the underlying slots are malformed.
pub fn resolve_authority_slots_with_profile_v1(
    first: Option<AuthoritySlotV1>,
    second: Option<AuthoritySlotV1>,
    owner: Option<GenerationLockFrameV1>,
    attempt: Option<GenerationLockFrameV1>,
    profile: GenerationRootSecurityProfileV1,
    external_floor: Option<AuthorityFloorV1>,
) -> Result<Option<AuthoritySlotV1>, GenerationAuthorityErrorV1> {
    let resolved = resolve_authority_slots_with_locks_v1(first, second, owner, attempt)?;
    match profile {
        GenerationRootSecurityProfileV1::RequiredExternal => {
            let floor = external_floor.ok_or(GenerationAuthorityErrorV1::ExternalFloorRequired)?;
            let resolved = resolved.ok_or(GenerationAuthorityErrorV1::AuthorityBelowFloor)?;
            resolve_selected_authority_at_floor_v1(resolved, floor).map(Some)
        }
        GenerationRootSecurityProfileV1::CooperativeLocal
        | GenerationRootSecurityProfileV1::ReadOnlyUnanchored => Ok(resolved),
    }
}

fn slot_matches_authority_sequence(slot: AuthoritySlotV1) -> bool {
    if slot.authority.sequence == 1 {
        return true;
    }
    let expected_slot = (slot.authority.sequence & 1) as u8;
    slot.slot_index == expected_slot
}

/// The transition that created an immutable activation manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationAuthorityActionV1 {
    /// Select a newly built generation.
    Activate,
    /// Select a previously published generation under a newer authority.
    Rollback,
    /// Publish a repaired replacement for a damaged generation.
    Repair,
    /// Publish a schema-preserving migrated generation.
    Migrate,
}

impl GenerationAuthorityActionV1 {
    const fn tag(self) -> u8 {
        match self {
            Self::Activate => 1,
            Self::Rollback => 2,
            Self::Repair => 3,
            Self::Migrate => 4,
        }
    }
}

/// Immutable receipt for one exact generation component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationComponentReceiptV1 {
    /// Exact component byte length.
    pub byte_len: u64,
    /// SHA-256 of the exact component bytes.
    pub sha256: [u8; 32],
}

impl GenerationComponentReceiptV1 {
    /// Validate the bounded immutable component receipt.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the receipt cannot identify real bytes.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.byte_len == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.component.byte_len",
            });
        }
        if self.sha256 == [0; 32] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.component.sha256",
            });
        }
        Ok(())
    }

    fn encode(self, encoder: &mut CanonicalEncoder) {
        encoder.u64(self.byte_len);
        encoder.bytes(&self.sha256);
    }
}

/// The four component receipts that make one generation selectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationComponentReceiptsV1 {
    /// Vector-index component receipt.
    pub vector: GenerationComponentReceiptV1,
    /// Lexical-index component receipt.
    pub lexical: GenerationComponentReceiptV1,
    /// ANN component receipt.
    pub ann: GenerationComponentReceiptV1,
    /// Metadata component receipt.
    pub metadata: GenerationComponentReceiptV1,
}

impl GenerationComponentReceiptsV1 {
    fn validate(self) -> Result<(), GenerationAuthorityErrorV1> {
        self.vector.validate()?;
        self.lexical.validate()?;
        self.ann.validate()?;
        self.metadata.validate()
    }

    fn encode(self, encoder: &mut CanonicalEncoder) {
        self.vector.encode(encoder);
        self.lexical.encode(encoder);
        self.ann.encode(encoder);
        self.metadata.encode(encoder);
    }
}

/// Immutable payload addressed by [`AuthorityRefV1`].
///
/// It is pure data: publishing the payload, selecting its object ID, and any
/// filesystem action belong to the generation-root publisher lane. The
/// self-seal detects an in-memory or decoded field substitution before a
/// caller uses the externally addressed manifest digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationManifestV1 {
    /// [`GENERATION_AUTHORITY_SCHEMA_V1`].
    pub schema_version: u16,
    /// Authority sequence this manifest is eligible to serve.
    pub authority_sequence: u64,
    /// Exact preceding authority reference. Genesis has none.
    pub predecessor: Option<AuthorityRefV1>,
    /// Transition that produced this generation selection.
    pub action: GenerationAuthorityActionV1,
    /// Artifact-generation identity selected by this authority transition.
    pub generation: ArtifactGenerationIdentityV1,
    /// Immutable writer-fence witness.
    pub writer_fence_sha256: [u8; 32],
    /// Immutable source-checkpoint witness.
    pub source_checkpoint_sha256: [u8; 32],
    /// Canonical document-set witness.
    pub document_set_sha256: [u8; 32],
    /// Exact vector, lexical, ANN, and metadata component receipts.
    pub components: GenerationComponentReceiptsV1,
    /// SHA-256 self-seal over every preceding field.
    pub self_seal_sha256: [u8; 32],
}

impl ActivationManifestV1 {
    /// Construct and self-seal a canonical immutable activation manifest.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a field or predecessor relation is invalid.
    pub fn new(
        authority_sequence: u64,
        predecessor: Option<AuthorityRefV1>,
        action: GenerationAuthorityActionV1,
        generation: ArtifactGenerationIdentityV1,
        writer_fence_sha256: [u8; 32],
        source_checkpoint_sha256: [u8; 32],
        document_set_sha256: [u8; 32],
        components: GenerationComponentReceiptsV1,
    ) -> Result<Self, GenerationAuthorityErrorV1> {
        let mut manifest = Self {
            schema_version: GENERATION_AUTHORITY_SCHEMA_V1,
            authority_sequence,
            predecessor,
            action,
            generation,
            writer_fence_sha256,
            source_checkpoint_sha256,
            document_set_sha256,
            components,
            self_seal_sha256: [0; 32],
        };
        manifest.validate_unsealed()?;
        manifest.self_seal_sha256 = manifest.computed_self_seal();
        Ok(manifest)
    }

    /// Validate every field and the self-seal without performing I/O.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid field, broken predecessor relation,
    /// or mismatched self-seal.
    pub fn validate(&self) -> Result<(), GenerationAuthorityErrorV1> {
        self.validate_unsealed()?;
        // ubs:ignore — the self-seal is public immutable-integrity evidence, not a secret.
        if self.self_seal_sha256 != self.computed_self_seal() {
            return Err(GenerationAuthorityErrorV1::ManifestSelfSealMismatch);
        }
        Ok(())
    }

    /// Canonical bytes excluding the self-seal field.
    #[must_use]
    pub fn canonical_unsealed_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.activation-manifest.v1");
        encoder.u16(self.schema_version);
        encoder.u64(self.authority_sequence);
        encoder.u8(self.action.tag());
        encoder.option(self.predecessor.as_ref(), |predecessor, encoder| {
            encoder.u16(predecessor.schema_version);
            encoder.u64(predecessor.sequence);
            encoder.bytes(&predecessor.object_id);
            encoder.u64(predecessor.manifest_len);
            encoder.bytes(&predecessor.manifest_sha256);
            encoder.option(predecessor.predecessor.as_ref(), |ancestor, encoder| {
                encoder.bytes(ancestor);
            });
        });
        encoder.u16(self.generation.schema_version);
        encoder.u64(self.generation.sequence);
        encoder.bytes(&self.generation.nonce);
        encoder.bytes(&self.writer_fence_sha256);
        encoder.bytes(&self.source_checkpoint_sha256);
        encoder.bytes(&self.document_set_sha256);
        self.components.encode(&mut encoder);
        encoder.finish()
    }

    /// Canonical bytes including the validated self-seal.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = self.canonical_unsealed_bytes();
        bytes.extend_from_slice(&self.self_seal_sha256);
        bytes
    }

    /// Parse, validate, and re-encode one bounded canonical activation
    /// manifest. The input must match [`Self::canonical_bytes`] byte-for-byte.
    ///
    /// # Errors
    ///
    /// Returns a typed error for truncated, extended, future, malformed, or
    /// non-canonical manifest bytes without allocating based on input lengths.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, GenerationAuthorityErrorV1> {
        if bytes.len() < 32 || bytes.len() > GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes",
            });
        }
        let unsealed_len = bytes.len() - 32;
        let (unsealed, seal) = bytes.split_at(unsealed_len);
        let mut decoder = CanonicalDecoder::new(unsealed);
        if decoder.bytes("activation_manifest.domain", 64)?
            != b"frankensearch.activation-manifest.v1"
        {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.domain",
            });
        }
        let schema_version = decoder.u16("activation_manifest.schema_version")?;
        let authority_sequence = decoder.u64("activation_manifest.authority_sequence")?;
        let action = match decoder.u8("activation_manifest.action")? {
            1 => GenerationAuthorityActionV1::Activate,
            2 => GenerationAuthorityActionV1::Rollback,
            3 => GenerationAuthorityActionV1::Repair,
            4 => GenerationAuthorityActionV1::Migrate,
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.action",
                });
            }
        };
        let predecessor = match decoder.u8("activation_manifest.predecessor.present")? {
            0 => None,
            1 => Some(AuthorityRefV1 {
                schema_version: decoder.u16("activation_manifest.predecessor.schema_version")?,
                sequence: decoder.u64("activation_manifest.predecessor.sequence")?,
                object_id: decoder.fixed_bytes("activation_manifest.predecessor.object_id")?,
                manifest_len: decoder.u64("activation_manifest.predecessor.manifest_len")?,
                manifest_sha256: decoder
                    .fixed_bytes("activation_manifest.predecessor.manifest_sha256")?,
                predecessor: match decoder.u8("activation_manifest.predecessor.ancestor.present")? {
                    0 => None,
                    1 => Some(decoder.fixed_bytes("activation_manifest.predecessor.ancestor")?),
                    _ => {
                        return Err(GenerationAuthorityErrorV1::InvalidField {
                            field: "activation_manifest.predecessor.ancestor.present",
                        });
                    }
                },
            }),
            _ => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.predecessor.present",
                });
            }
        };
        let generation = ArtifactGenerationIdentityV1 {
            schema_version: decoder.u16("activation_manifest.generation.schema_version")?,
            sequence: decoder.u64("activation_manifest.generation.sequence")?,
            nonce: decoder.fixed_bytes("activation_manifest.generation.nonce")?,
        };
        let writer_fence_sha256 = decoder.fixed_bytes("activation_manifest.writer_fence_sha256")?;
        let source_checkpoint_sha256 =
            decoder.fixed_bytes("activation_manifest.source_checkpoint_sha256")?;
        let document_set_sha256 = decoder.fixed_bytes("activation_manifest.document_set_sha256")?;
        let components = GenerationComponentReceiptsV1 {
            vector: decoder.component("activation_manifest.components.vector")?,
            lexical: decoder.component("activation_manifest.components.lexical")?,
            ann: decoder.component("activation_manifest.components.ann")?,
            metadata: decoder.component("activation_manifest.components.metadata")?,
        };
        decoder.finish()?;
        let mut self_seal_sha256 = [0_u8; 32];
        self_seal_sha256.copy_from_slice(seal);
        let manifest = Self {
            schema_version,
            authority_sequence,
            predecessor,
            action,
            generation,
            writer_fence_sha256,
            source_checkpoint_sha256,
            document_set_sha256,
            components,
            self_seal_sha256,
        };
        manifest.validate()?;
        if !manifest.canonical_bytes().eq(bytes) {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes",
            });
        }
        Ok(manifest)
    }

    /// Exact length and SHA-256 used by the external [`AuthorityRefV1`].
    #[must_use]
    pub fn object_receipt(&self) -> (u64, [u8; 32]) {
        let bytes = self.canonical_bytes();
        (
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            Sha256::digest(bytes).into(),
        )
    }

    fn validate_unsealed(&self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.schema_version != GENERATION_AUTHORITY_SCHEMA_V1 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.schema_version",
            });
        }
        if self.authority_sequence == 0 {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.authority_sequence",
            });
        }
        match (self.authority_sequence, self.predecessor) {
            (1, None) => {}
            (1, Some(_)) => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.predecessor",
                });
            }
            (_, None) => {
                return Err(GenerationAuthorityErrorV1::InvalidField {
                    field: "activation_manifest.predecessor",
                });
            }
            (sequence, Some(predecessor)) => {
                predecessor.validate()?;
                if predecessor.sequence.checked_add(1) != Some(sequence) {
                    return Err(GenerationAuthorityErrorV1::BrokenPredecessorLink);
                }
            }
        }
        self.generation
            .validate()
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.generation",
            })?;
        for (field, digest) in [
            (
                "activation_manifest.writer_fence_sha256",
                self.writer_fence_sha256,
            ),
            (
                "activation_manifest.source_checkpoint_sha256",
                self.source_checkpoint_sha256,
            ),
            (
                "activation_manifest.document_set_sha256",
                self.document_set_sha256,
            ),
        ] {
            // ubs:ignore — component digests are public immutable-integrity identities.
            if digest == [0; 32] {
                return Err(GenerationAuthorityErrorV1::InvalidField { field });
            }
        }
        self.components.validate()
    }

    fn computed_self_seal(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_unsealed_bytes()).into()
    }
}

/// Verify that an immutable activation manifest is the exact object named by
/// one authority reference.
///
/// The caller supplies bytes/object storage outside this pure helper. This
/// check binds the authority sequence, predecessor fingerprint, exact object
/// length, and exact object digest before a consumer can select the manifest.
///
/// # Errors
///
/// Returns a typed error for malformed inputs or any reference/manifest
/// mismatch. It never falls back to a different manifest.
pub fn verify_authority_manifest_reference_v1(
    authority: &AuthorityRefV1,
    manifest: &ActivationManifestV1,
) -> Result<(), GenerationAuthorityErrorV1> {
    authority.validate()?;
    manifest.validate()?;
    // ubs:ignore — authority sequence/predecessor fingerprints are public integrity identities.
    if authority.sequence != manifest.authority_sequence
        || authority.predecessor
            != manifest
                .predecessor
                .map(|predecessor| predecessor.fingerprint())
    {
        return Err(GenerationAuthorityErrorV1::ManifestReferenceMismatch);
    }
    let (manifest_len, manifest_sha256) = manifest.object_receipt();
    // ubs:ignore — manifest lengths and digests are public immutable object identities.
    if authority.manifest_len != manifest_len || authority.manifest_sha256 != manifest_sha256 {
        return Err(GenerationAuthorityErrorV1::ManifestReferenceMismatch);
    }
    Ok(())
}

/// One immutable, role-tagged artifact participating in a vector space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingArtifactIdentityV1 {
    /// Stable semantic role such as `weights`, `tokenizer`, `vocabulary`, or `config`.
    pub role: String,
    /// Lowercase SHA-256 of the exact artifact bytes.
    pub sha256: String,
    /// Exact artifact size in bytes.
    pub size: u64,
}

/// Whether an embedding space is semantic or an explicit non-semantic control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingSpaceKindV1 {
    /// A learned vector space pinned by immutable model artifacts.
    Semantic,
    /// A deterministic test/control space that must never imply semantic availability.
    HashControl,
}

/// Complete algorithm contract for a deterministic hash control.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HashControlProfileV1 {
    /// Algorithm family, for example `fnv1a-feature-hash`.
    pub algorithm: String,
    /// Immutable algorithm/protocol revision.
    pub algorithm_revision: String,
    /// Seed or key identifier. This is configuration, not secret key material.
    pub seed: u64,
    /// Exact feature extraction rules.
    pub feature_rules: String,
    /// Exact tokenization rules.
    pub tokenization_rules: String,
    /// Exact signed-bucket rules.
    pub signing_rules: String,
    /// Exact output normalization rules.
    pub normalization_rules: String,
}

/// Structured Matryoshka/projection ancestry for a derived vector space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProjectionV1 {
    /// Fingerprint of the complete parent vector space.
    pub parent_space_fingerprint: String,
    /// Dimension before projection.
    pub source_dimension: u32,
    /// Dimension after projection.
    pub output_dimension: u32,
    /// Exact deterministic projection/truncation rule.
    pub projection_rule: String,
    /// Exact post-projection normalization rule.
    pub renormalization_rule: String,
}

/// Mathematical identity of the complete input-to-vector map.
///
/// Equality is exact. Human names, directory names, filenames, and dimension-only
/// matches never establish compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingSpaceIdentityV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Semantic model/control identifier, never a display label.
    pub logical_model_id: String,
    /// Immutable upstream model or algorithm revision.
    pub immutable_revision: String,
    /// Semantic versus explicit hash-control classification.
    pub kind: EmbeddingSpaceKindV1,
    /// Fingerprint of the verified artifact-space contract: immutable
    /// role-tagged artifacts plus all model semantics, excluding producer and
    /// distribution metadata.
    ///
    /// Hash controls use the canonical hash-profile fingerprint here.
    pub artifact_manifest_fingerprint: String,
    /// Artifact identities. Canonical encoding sorts them by role.
    pub artifacts: Vec<EmbeddingArtifactIdentityV1>,
    /// Exact tokenizer identity.
    pub tokenizer_fingerprint: String,
    /// Exact vocabulary identity.
    pub vocabulary_fingerprint: String,
    /// Exact model configuration identity.
    pub model_config_fingerprint: String,
    /// Model-internal preprocessing contract.
    pub model_preprocessing: String,
    /// Sequence length, truncation, and padding contract.
    pub sequence_policy: String,
    /// Query instruction applied inside the model contract.
    pub query_instruction: String,
    /// Document instruction applied inside the model contract.
    pub document_instruction: String,
    /// Pooling rule.
    pub pooling: String,
    /// Output normalization rule.
    pub output_normalization: String,
    /// Output dimension.
    pub dimension: u32,
    /// Fingerprint of the separately versioned outer input contract.
    pub input_contract_fingerprint: String,
    /// Required for hash controls and forbidden for semantic spaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_control: Option<HashControlProfileV1>,
    /// Structured ancestry for MRL or another deterministic projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<EmbeddingProjectionV1>,
}

impl EmbeddingSpaceIdentityV1 {
    /// Validate all required fields and cross-field invariants.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for an unknown schema, malformed digest,
    /// duplicate role, incomplete semantic space, or inconsistent projection.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "embedding_space_identity.schema_version",
            self.schema_version,
            EMBEDDING_SPACE_IDENTITY_SCHEMA_V1,
        )?;
        validate_identity_text("logical_model_id", &self.logical_model_id)?;
        validate_identity_text("immutable_revision", &self.immutable_revision)?;
        validate_sha256(
            "artifact_manifest_fingerprint",
            &self.artifact_manifest_fingerprint,
        )?;
        validate_sha256("tokenizer_fingerprint", &self.tokenizer_fingerprint)?;
        validate_sha256("vocabulary_fingerprint", &self.vocabulary_fingerprint)?;
        validate_sha256("model_config_fingerprint", &self.model_config_fingerprint)?;
        validate_identity_text("model_preprocessing", &self.model_preprocessing)?;
        validate_identity_text("sequence_policy", &self.sequence_policy)?;
        validate_optional_identity_text("query_instruction", &self.query_instruction)?;
        validate_optional_identity_text("document_instruction", &self.document_instruction)?;
        validate_identity_text("pooling", &self.pooling)?;
        validate_identity_text("output_normalization", &self.output_normalization)?;
        validate_sha256(
            "input_contract_fingerprint",
            &self.input_contract_fingerprint,
        )?;
        if self.dimension == 0 {
            return Err(identity_error(
                "dimension",
                "0",
                "must be greater than zero",
            ));
        }

        let mut roles = BTreeSet::new();
        for artifact in &self.artifacts {
            validate_identity_text("artifacts[].role", &artifact.role)?;
            validate_sha256("artifacts[].sha256", &artifact.sha256)?;
            if artifact.size == 0 {
                return Err(identity_error(
                    "artifacts[].size",
                    "0",
                    "must be greater than zero",
                ));
            }
            if !roles.insert(artifact.role.as_str()) {
                return Err(identity_error(
                    "artifacts[].role",
                    &artifact.role,
                    "duplicate artifact role",
                ));
            }
        }

        match (self.kind, &self.hash_control) {
            (EmbeddingSpaceKindV1::Semantic, None) => {
                if self.artifacts.is_empty() {
                    return Err(identity_error(
                        "artifacts",
                        "[]",
                        "semantic spaces require immutable artifacts",
                    ));
                }
            }
            (EmbeddingSpaceKindV1::Semantic, Some(_)) => {
                return Err(identity_error(
                    "hash_control",
                    "present",
                    "semantic spaces cannot carry a hash-control profile",
                ));
            }
            (EmbeddingSpaceKindV1::HashControl, Some(profile)) => {
                profile.validate()?;
                // ubs:ignore — manifest fingerprints are public compatibility identities, not secrets.
                if self.artifact_manifest_fingerprint != profile.fingerprint() {
                    return Err(identity_error(
                        "artifact_manifest_fingerprint",
                        &self.artifact_manifest_fingerprint,
                        "must equal the canonical hash-control profile fingerprint",
                    ));
                }
                if !self.artifacts.is_empty() {
                    return Err(identity_error(
                        "artifacts",
                        "present",
                        "hash controls bind rules, not learned model artifacts",
                    ));
                }
            }
            (EmbeddingSpaceKindV1::HashControl, None) => {
                return Err(identity_error(
                    "hash_control",
                    "absent",
                    "hash-control spaces require the complete algorithm profile",
                ));
            }
        }

        if let Some(projection) = &self.projection {
            projection.validate()?;
            if projection.output_dimension != self.dimension {
                return Err(identity_error(
                    "projection.output_dimension",
                    &projection.output_dimension.to_string(),
                    "must equal the space output dimension",
                ));
            }
        }
        Ok(())
    }

    /// Canonical domain-separated, length-prefixed bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.embedding-space.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.logical_model_id);
        encoder.text(&self.immutable_revision);
        encoder.u8(match self.kind {
            EmbeddingSpaceKindV1::Semantic => 1,
            EmbeddingSpaceKindV1::HashControl => 2,
        });
        encoder.text(&self.artifact_manifest_fingerprint);

        let mut artifacts = self.artifacts.iter().collect::<Vec<_>>();
        artifacts.sort_by(|left, right| left.role.cmp(&right.role));
        encoder.usize(artifacts.len());
        for artifact in artifacts {
            encoder.text(&artifact.role);
            encoder.text(&artifact.sha256);
            encoder.u64(artifact.size);
        }

        encoder.text(&self.tokenizer_fingerprint);
        encoder.text(&self.vocabulary_fingerprint);
        encoder.text(&self.model_config_fingerprint);
        encoder.text(&self.model_preprocessing);
        encoder.text(&self.sequence_policy);
        encoder.text(&self.query_instruction);
        encoder.text(&self.document_instruction);
        encoder.text(&self.pooling);
        encoder.text(&self.output_normalization);
        encoder.u32(self.dimension);
        encoder.text(&self.input_contract_fingerprint);
        encoder.option(self.hash_control.as_ref(), HashControlProfileV1::encode);
        encoder.option(self.projection.as_ref(), EmbeddingProjectionV1::encode);
        encoder.finish()
    }

    /// Lowercase SHA-256 fingerprint of [`canonical_bytes`](Self::canonical_bytes).
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }

    /// Produce a structurally derived MRL/projection identity.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the target is not a strict non-zero reduction.
    pub fn derive_projection(
        &self,
        target_dimension: u32,
        projection_rule: &str,
        renormalization_rule: &str,
    ) -> Result<Self, SearchError> {
        self.validate()?;
        if target_dimension == 0 || target_dimension >= self.dimension {
            return Err(identity_error(
                "projection.output_dimension",
                &target_dimension.to_string(),
                "must be between 1 and parent dimension - 1",
            ));
        }
        validate_identity_text("projection_rule", projection_rule)?;
        validate_identity_text("renormalization_rule", renormalization_rule)?;

        let mut derived = self.clone();
        derived.dimension = target_dimension;
        renormalization_rule.clone_into(&mut derived.output_normalization);
        derived.projection = Some(EmbeddingProjectionV1 {
            parent_space_fingerprint: self.fingerprint(),
            source_dimension: self.dimension,
            output_dimension: target_dimension,
            projection_rule: projection_rule.to_owned(),
            renormalization_rule: renormalization_rule.to_owned(),
        });
        derived.validate()?;
        Ok(derived)
    }
}

impl HashControlProfileV1 {
    fn validate(&self) -> Result<(), SearchError> {
        validate_identity_text("hash_control.algorithm", &self.algorithm)?;
        validate_identity_text("hash_control.algorithm_revision", &self.algorithm_revision)?;
        validate_identity_text("hash_control.feature_rules", &self.feature_rules)?;
        validate_identity_text("hash_control.tokenization_rules", &self.tokenization_rules)?;
        validate_identity_text("hash_control.signing_rules", &self.signing_rules)?;
        validate_identity_text(
            "hash_control.normalization_rules",
            &self.normalization_rules,
        )
    }

    /// Canonical domain-separated, length-prefixed profile bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.hash-control-profile.v1");
        self.encode(&mut encoder);
        encoder.finish()
    }

    /// Lowercase SHA-256 of the complete deterministic hash-control profile.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }

    fn encode(&self, encoder: &mut CanonicalEncoder) {
        encoder.text(&self.algorithm);
        encoder.text(&self.algorithm_revision);
        encoder.u64(self.seed);
        encoder.text(&self.feature_rules);
        encoder.text(&self.tokenization_rules);
        encoder.text(&self.signing_rules);
        encoder.text(&self.normalization_rules);
    }
}

impl EmbeddingProjectionV1 {
    fn validate(&self) -> Result<(), SearchError> {
        validate_sha256(
            "projection.parent_space_fingerprint",
            &self.parent_space_fingerprint,
        )?;
        if self.source_dimension == 0
            || self.output_dimension == 0
            || self.output_dimension >= self.source_dimension
        {
            return Err(identity_error(
                "projection.dimension",
                &format!("{}->{}", self.source_dimension, self.output_dimension),
                "must be a strict non-zero reduction",
            ));
        }
        validate_identity_text("projection.projection_rule", &self.projection_rule)?;
        validate_identity_text(
            "projection.renormalization_rule",
            &self.renormalization_rule,
        )
    }

    fn encode(&self, encoder: &mut CanonicalEncoder) {
        encoder.text(&self.parent_space_fingerprint);
        encoder.u32(self.source_dimension);
        encoder.u32(self.output_dimension);
        encoder.text(&self.projection_rule);
        encoder.text(&self.renormalization_rule);
    }
}

/// Outer content-selection and canonicalization contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingInputContractV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Outer canonicalization before model-specific preprocessing.
    pub canonicalization: String,
    /// Which document/query content is selected.
    pub content_selection: String,
    /// Chunking and overlap semantics.
    pub chunking: String,
    /// Query instruction semantics.
    pub query_instruction: String,
    /// Document instruction semantics.
    pub document_instruction: String,
    /// Document-id and aggregation semantics.
    pub doc_id_semantics: String,
}

impl EmbeddingInputContractV1 {
    /// Validate the complete input contract.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for an unknown schema or an empty required field.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "embedding_input_contract.schema_version",
            self.schema_version,
            EMBEDDING_INPUT_CONTRACT_SCHEMA_V1,
        )?;
        validate_identity_text("canonicalization", &self.canonicalization)?;
        validate_identity_text("content_selection", &self.content_selection)?;
        validate_identity_text("chunking", &self.chunking)?;
        validate_optional_identity_text("query_instruction", &self.query_instruction)?;
        validate_optional_identity_text("document_instruction", &self.document_instruction)?;
        validate_identity_text("doc_id_semantics", &self.doc_id_semantics)
    }

    /// Canonical domain-separated, length-prefixed bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.embedding-input.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.canonicalization);
        encoder.text(&self.content_selection);
        encoder.text(&self.chunking);
        encoder.text(&self.query_instruction);
        encoder.text(&self.document_instruction);
        encoder.text(&self.doc_id_semantics);
        encoder.finish()
    }

    /// Lowercase SHA-256 of the canonical input-contract bytes.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }
}

/// Golden-vector certificate pinning one implementation to a vector space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenVectorCertificateV1 {
    /// Digest of the ordered, redacted golden input corpus.
    pub corpus_sha256: String,
    /// Digest of exact output f32 bit patterns.
    pub vectors_sha256: String,
    /// Number of golden vectors.
    pub vector_count: u32,
    /// Dimension of every golden vector.
    pub dimension: u32,
}

impl GoldenVectorCertificateV1 {
    /// Fingerprint an ordered, non-empty conformance text corpus.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the corpus is empty.
    pub fn corpus_fingerprint(texts: &[&str]) -> Result<String, SearchError> {
        if texts.is_empty() {
            return Err(identity_error(
                "golden.corpus_shape",
                "0 texts",
                "the golden corpus must be non-empty",
            ));
        }
        let mut corpus = CanonicalEncoder::new(b"frankensearch.golden-corpus.v1");
        corpus.usize(texts.len());
        for text in texts {
            corpus.text(text);
        }
        Ok(sha256_hex(&corpus.finish()))
    }

    /// Build a certificate from an ordered text corpus and exact `f32` output bits.
    ///
    /// Text and vector bytes are encoded in separate domain-separated,
    /// length-prefixed transcripts. Floating-point values are recorded with
    /// [`f32::to_bits`], preserving signed zero and NaN payloads.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the corpus is empty, the text/vector counts
    /// differ, vector dimensions are inconsistent, or the shape exceeds `u32`.
    pub fn from_exact_f32(texts: &[&str], vectors: &[Vec<f32>]) -> Result<Self, SearchError> {
        if texts.is_empty() || texts.len() != vectors.len() {
            return Err(identity_error(
                "golden.corpus_shape",
                &format!("{} texts/{} vectors", texts.len(), vectors.len()),
                "the non-empty text and vector counts must match",
            ));
        }
        let dimension = vectors.first().map_or(0, Vec::len);
        if dimension == 0 || vectors.iter().any(|vector| vector.len() != dimension) {
            return Err(identity_error(
                "golden.vector_shape",
                &format!("{} vectors", vectors.len()),
                "all golden vectors must have one identical non-zero dimension",
            ));
        }
        let vector_count = u32::try_from(vectors.len()).map_err(|_| {
            identity_error(
                "golden.vector_count",
                "out-of-range",
                "golden vector count must fit in u32",
            )
        })?;
        let dimension = u32::try_from(dimension).map_err(|_| {
            identity_error(
                "golden.dimension",
                "out-of-range",
                "golden vector dimension must fit in u32",
            )
        })?;

        let mut outputs = CanonicalEncoder::new(b"frankensearch.golden-f32-vectors.v1");
        outputs.u32(vector_count);
        outputs.u32(dimension);
        for vector in vectors {
            for value in vector {
                outputs.u32(value.to_bits());
            }
        }

        Ok(Self {
            corpus_sha256: Self::corpus_fingerprint(texts)?,
            vectors_sha256: sha256_hex(&outputs.finish()),
            vector_count,
            dimension,
        })
    }

    /// Verify an ordered corpus and exact output bits against this certificate.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the certificate is malformed or any corpus,
    /// shape, or output bit differs. The error never includes input or vector data.
    pub fn verify_exact_f32(
        &self,
        texts: &[&str],
        vectors: &[Vec<f32>],
    ) -> Result<(), SearchError> {
        self.validate()?;
        let observed = Self::from_exact_f32(texts, vectors)?;
        if observed != *self {
            return Err(identity_error(
                "golden.conformance",
                "mismatch",
                "ordered corpus, vector shape, or exact f32 output bits drifted",
            ));
        }
        Ok(())
    }

    /// Validate digest shape and non-zero certificate dimensions.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for malformed hashes or an empty certificate.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_sha256("golden.corpus_sha256", &self.corpus_sha256)?;
        validate_sha256("golden.vectors_sha256", &self.vectors_sha256)?;
        if self.vector_count == 0 || self.dimension == 0 {
            return Err(identity_error(
                "golden.shape",
                &format!("{}x{}", self.vector_count, self.dimension),
                "vector count and dimension must be non-zero",
            ));
        }
        Ok(())
    }

    fn encode(&self, encoder: &mut CanonicalEncoder) {
        encoder.text(&self.corpus_sha256);
        encoder.text(&self.vectors_sha256);
        encoder.u32(self.vector_count);
        encoder.u32(self.dimension);
    }
}

/// Attestation for the implementation that produced vectors in a space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProducerAttestationV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Backend family such as `model2vec-native`, `fastembed-onnx`, or `remote-api`.
    pub backend: String,
    /// Immutable backend implementation revision.
    pub implementation_revision: String,
    /// Wire/inference protocol revision.
    pub protocol_revision: String,
    /// Numeric execution profile, including dtype and deterministic tolerances.
    pub numeric_profile: String,
    /// Fingerprint of the complete provenance manifest. Local/native learned
    /// models bind the full frozen `ModelArtifactManifestV1` here, including
    /// provider, repository, license, distribution metadata, execution
    /// contract, and golden certificate. Hash controls bind their canonical
    /// algorithm profile.
    pub provenance_manifest_fingerprint: String,
    /// Fingerprint of the exact mathematical vector space.
    pub space_fingerprint: String,
    /// Pinned conformance-vector certificate.
    pub golden_vectors: GoldenVectorCertificateV1,
}

impl EmbeddingProducerAttestationV1 {
    /// Validate the complete producer attestation.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for an unknown schema, empty field, malformed
    /// fingerprint, or incomplete golden-vector certificate.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "embedding_producer_attestation.schema_version",
            self.schema_version,
            EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
        )?;
        validate_identity_text("backend", &self.backend)?;
        validate_identity_text("implementation_revision", &self.implementation_revision)?;
        validate_identity_text("protocol_revision", &self.protocol_revision)?;
        validate_identity_text("numeric_profile", &self.numeric_profile)?;
        validate_sha256(
            "provenance_manifest_fingerprint",
            &self.provenance_manifest_fingerprint,
        )?;
        validate_sha256("space_fingerprint", &self.space_fingerprint)?;
        self.golden_vectors.validate()
    }

    /// Canonical domain-separated, length-prefixed bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.embedding-producer.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.backend);
        encoder.text(&self.implementation_revision);
        encoder.text(&self.protocol_revision);
        encoder.text(&self.numeric_profile);
        encoder.text(&self.provenance_manifest_fingerprint);
        encoder.text(&self.space_fingerprint);
        self.golden_vectors.encode(&mut encoder);
        encoder.finish()
    }

    /// Lowercase SHA-256 of the canonical producer-attestation bytes.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }
}

/// Physical vector encoding identity, deliberately separate from vector-space identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorStorageIdentityV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Storage format and format revision, for example `fsvi-v2`.
    pub format: String,
    /// Quantization/storage scalar format.
    pub quantization: QuantizationFormat,
    /// Byte order.
    pub endianness: String,
    /// Stored-vector normalization assumption.
    pub vector_normalization: String,
    /// Stored vector dimension.
    pub dimension: u32,
}

impl VectorStorageIdentityV1 {
    /// Validate the storage contract.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` for an unknown schema, empty field, or zero dimension.
    pub fn validate(&self) -> Result<(), SearchError> {
        validate_schema(
            "vector_storage_identity.schema_version",
            self.schema_version,
            VECTOR_STORAGE_IDENTITY_SCHEMA_V1,
        )?;
        validate_identity_text("format", &self.format)?;
        validate_identity_text("endianness", &self.endianness)?;
        validate_identity_text("vector_normalization", &self.vector_normalization)?;
        if self.dimension == 0 {
            return Err(identity_error(
                "storage.dimension",
                "0",
                "must be greater than zero",
            ));
        }
        if self.format.starts_with("in-memory-") && self.quantization != QuantizationFormat::F32 {
            return Err(identity_error(
                "storage.quantization",
                &format!("{:?}", self.quantization),
                "in-memory Vec<f32> formats require F32 values",
            ));
        }
        if self.format.starts_with("in-memory-")
            && !matches!(
                self.endianness.as_str(),
                "native-f32-values" | "native-test-only"
            )
        {
            return Err(identity_error(
                "storage.endianness",
                &self.endianness,
                "in-memory formats require an explicit native-value contract",
            ));
        }
        if self.format.starts_with("fsvi-") && self.endianness != "little-endian" {
            return Err(identity_error(
                "storage.endianness",
                &self.endianness,
                "FSVI storage is canonically little-endian",
            ));
        }
        Ok(())
    }

    /// Canonical domain-separated, length-prefixed bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.vector-storage.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.format);
        encoder.u8(match self.quantization {
            QuantizationFormat::F32 => 1,
            QuantizationFormat::F16 => 2,
            QuantizationFormat::Int8 => 3,
            QuantizationFormat::Int4 => 4,
        });
        encoder.text(&self.endianness);
        encoder.text(&self.vector_normalization);
        encoder.u32(self.dimension);
        encoder.finish()
    }

    /// Lowercase SHA-256 of the canonical storage bytes.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }
}

/// Complete, independently versioned embedding identity bundle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingIdentityBundleV1 {
    /// Mathematical input-to-vector map.
    pub space: EmbeddingSpaceIdentityV1,
    /// Implementation/protocol attestation.
    pub producer: EmbeddingProducerAttestationV1,
    /// Outer content-selection/canonicalization contract.
    pub input: EmbeddingInputContractV1,
    /// Physical vector encoding identity.
    pub storage: VectorStorageIdentityV1,
}

/// Persistable exact bytes and fingerprint for one complete identity bundle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenEmbeddingIdentityBundleV1 {
    /// Structured identity contracts.
    pub identity: EmbeddingIdentityBundleV1,
    /// Exact domain-separated canonical bytes.
    pub canonical_bytes: Vec<u8>,
    /// Lowercase SHA-256 of `canonical_bytes`.
    pub fingerprint: String,
}

/// Why producer compatibility could not be established.
///
/// Variants deliberately carry no caller-controlled text. Callers may log the
/// stable variant name together with bounded witness fingerprints without ever
/// exposing conformance inputs, vectors, model paths, or policy material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum ProducerCompatibilityErrorV1 {
    /// The reference bundle failed its complete identity validation.
    #[error("reference embedding identity is invalid")]
    InvalidReferenceIdentity,
    /// The candidate bundle failed its complete identity validation.
    #[error("candidate embedding identity is invalid")]
    InvalidCandidateIdentity,
    /// The bundles describe different mathematical embedding spaces.
    #[error("producer comparison crossed mathematical embedding spaces")]
    SpaceMismatch,
    /// Exact producer identity did not match, so certified evidence is required.
    #[error("foreign producer requires an explicit conformance certificate")]
    CertificateRequired,
    /// The certified path was invoked for an already-exact producer.
    #[error("foreign-producer certificate is forbidden for an exact producer")]
    CertificateForbiddenForExactProducer,
    /// The ordered fixture or one of its exact vector sets was malformed.
    #[error("golden conformance fixture is malformed")]
    GoldenFixtureInvalid,
    /// Reference and candidate outputs or their embedded certificates differ.
    #[error("golden conformance vectors do not match exactly")]
    GoldenVectorMismatch,
    /// The raw certificate has an unknown schema or malformed field.
    #[error("foreign-producer certificate is malformed")]
    CertificateMalformed,
    /// The canonical raw certificate does not match the independently pinned digest.
    #[error("foreign-producer certificate fingerprint is not trusted")]
    CertificateFingerprintMismatch,
    /// The certificate names a policy other than the independently pinned policy.
    #[error("foreign-producer certificate policy does not match trusted policy")]
    PolicyMismatch,
    /// One or both direction-sensitive producer bindings are wrong.
    #[error("foreign-producer certificate does not bind the exact producer pair")]
    ProducerBindingMismatch,
    /// The certificate does not bind the independently verified golden fixture.
    #[error("foreign-producer certificate does not bind the trusted fixture")]
    FixtureBindingMismatch,
    /// The certificate is not active yet at the trusted evaluation time.
    #[error("foreign-producer certificate is not yet valid")]
    CertificateNotYetValid,
    /// The certificate is expired at the trusted evaluation time.
    #[error("foreign-producer certificate is expired")]
    CertificateExpired,
    /// The certificate revision is outside the trusted policy window.
    #[error("foreign-producer certificate revision is outside trusted policy")]
    CertificateRevisionOutsidePolicy,
}

/// Authority-free fixture receipt derived from both producers' actual outputs.
///
/// This type is intentionally serializable but not deserializable. Untrusted
/// bytes can never regain the authority established by executing both producers
/// over the same ordered inputs and comparing their exact `f32` output bits.
///
/// ```compile_fail
/// use frankensearch_core::generation::VerifiedGoldenConformanceManifestV1;
/// let _: VerifiedGoldenConformanceManifestV1 = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct VerifiedGoldenConformanceManifestV1 {
    certificate: GoldenVectorCertificateV1,
    canonical_bytes: Vec<u8>,
    fingerprint: String,
}

/// Raw, non-authoritative certificate claiming that two distinct producers are
/// bit-exact on one verified fixture.
///
/// Deserialization only recovers an untrusted claim. The sole authority-bearing
/// promotion path is
/// [`EmbeddingIdentityBundleV1::verify_certified_foreign_producer_with`], which
/// compares every field with independently trusted identity, policy, time, and
/// fixture state.
///
/// Raw certificate bytes alone cannot become either trust context or witness:
///
/// ```compile_fail
/// use frankensearch_core::generation::{
///     ForeignProducerConformanceCertificateV1, ProducerCompatibilityWitnessV1,
///     TrustedProducerConformanceContextV1,
/// };
/// let raw: ForeignProducerConformanceCertificateV1 = serde_json::from_str("{}").unwrap();
/// let _: TrustedProducerConformanceContextV1<'_> = raw.clone().into();
/// let _: ProducerCompatibilityWitnessV1 = raw.into();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignProducerConformanceCertificateV1 {
    schema_version: u16,
    reference_producer_fingerprint: String,
    candidate_producer_fingerprint: String,
    space_fingerprint: String,
    golden_fixture_fingerprint: String,
    policy_fingerprint: String,
    certificate_revision: u64,
    not_before_unix_seconds: u64,
    expires_at_unix_seconds: u64,
}

/// Independently sourced authority for validating one raw foreign-producer
/// certificate.
///
/// The pinned certificate and policy fingerprints, evaluation time, revision
/// window, and verified fixture must come from retained owners or sealed local
/// policy. Never populate this context from the certificate being validated.
/// This type is intentionally neither serializable nor deserializable.
///
/// ```compile_fail
/// use frankensearch_core::generation::TrustedProducerConformanceContextV1;
/// let _: TrustedProducerConformanceContextV1<'_> = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TrustedProducerConformanceContextV1<'a> {
    policy_fingerprint: &'a str,
    certificate_fingerprint: &'a str,
    fixture: &'a VerifiedGoldenConformanceManifestV1,
    evaluation_time_unix_seconds: u64,
    minimum_certificate_revision: u64,
    maximum_certificate_revision: u64,
}

/// How an opaque producer-compatibility witness was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProducerCompatibilityKindV1 {
    /// Both validated bundles carry the exact same producer identity.
    Exact,
    /// A distinct producer was promoted through an independently trusted certificate.
    Certified,
}

/// Opaque proof that one candidate producer is admissible relative to one
/// reference producer.
///
/// Fields are private and the type is not deserializable, so a caller cannot
/// manufacture runtime authority from JSON or plausible digest strings.
///
/// ```compile_fail
/// use frankensearch_core::generation::ProducerCompatibilityWitnessV1;
/// let _: ProducerCompatibilityWitnessV1 = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct ProducerCompatibilityWitnessV1 {
    kind: ProducerCompatibilityKindV1,
    space_fingerprint: String,
    reference_producer_fingerprint: String,
    candidate_producer_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    certificate_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    golden_fixture_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    certificate_revision: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at_unix_seconds: Option<u64>,
}

impl VerifiedGoldenConformanceManifestV1 {
    /// Execute the fixture comparison boundary over already-produced exact vectors.
    ///
    /// # Errors
    ///
    /// Returns [`ProducerCompatibilityErrorV1::GoldenFixtureInvalid`] for malformed
    /// corpus/vector shapes, or [`ProducerCompatibilityErrorV1::GoldenVectorMismatch`]
    /// when the producers differ in any ordered input, shape, or exact `f32` bit.
    pub fn from_exact_pair_f32(
        texts: &[&str],
        reference_vectors: &[Vec<f32>],
        candidate_vectors: &[Vec<f32>],
    ) -> Result<Self, ProducerCompatibilityErrorV1> {
        let reference = GoldenVectorCertificateV1::from_exact_f32(texts, reference_vectors)
            .map_err(|_| ProducerCompatibilityErrorV1::GoldenFixtureInvalid)?;
        let candidate = GoldenVectorCertificateV1::from_exact_f32(texts, candidate_vectors)
            .map_err(|_| ProducerCompatibilityErrorV1::GoldenFixtureInvalid)?;
        if reference != candidate {
            return Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch);
        }
        let canonical_bytes = Self::canonical_bytes_for(&reference);
        Ok(Self {
            certificate: reference,
            fingerprint: sha256_hex(&canonical_bytes),
            canonical_bytes,
        })
    }

    /// Exact common golden-vector certificate produced by the fixture run.
    #[must_use]
    pub const fn certificate(&self) -> &GoldenVectorCertificateV1 {
        &self.certificate
    }

    /// Domain-separated SHA-256 of the frozen fixture manifest.
    #[must_use]
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Exact canonical bytes bound by [`Self::fingerprint`].
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    fn canonical_bytes_for(certificate: &GoldenVectorCertificateV1) -> Vec<u8> {
        let mut encoder =
            CanonicalEncoder::new(b"frankensearch.verified-golden-conformance-manifest.v1");
        certificate.encode(&mut encoder);
        encoder.finish()
    }

    fn validate(&self) -> Result<(), ProducerCompatibilityErrorV1> {
        self.certificate
            .validate()
            .map_err(|_| ProducerCompatibilityErrorV1::GoldenFixtureInvalid)?;
        let canonical_bytes = Self::canonical_bytes_for(&self.certificate);
        // ubs:ignore — these canonical certificate bytes are public integrity evidence, not secrets.
        if self.canonical_bytes != canonical_bytes
            // ubs:ignore — this fingerprint is public certificate-integrity evidence, not a secret.
            || self.fingerprint != sha256_hex(&canonical_bytes)
        {
            return Err(ProducerCompatibilityErrorV1::GoldenFixtureInvalid);
        }
        Ok(())
    }
}

impl ForeignProducerConformanceCertificateV1 {
    /// Create an authority-free raw receipt for a verified producer pair.
    ///
    /// The returned value remains untrusted until promoted against a
    /// [`TrustedProducerConformanceContextV1`] that independently pins its
    /// canonical fingerprint. Computing that fingerprint from this returned
    /// value and immediately feeding it back as the supposed independent pin is
    /// self-assertion, not certification.
    ///
    /// # Errors
    ///
    /// Returns a typed producer-compatibility error for invalid identities,
    /// cross-space or exact-producer use, fixture disagreement, malformed policy,
    /// zero revision, or an empty validity interval.
    pub fn new_untrusted_receipt_from_verified_pair(
        reference: &EmbeddingIdentityBundleV1,
        candidate: &EmbeddingIdentityBundleV1,
        fixture: &VerifiedGoldenConformanceManifestV1,
        policy_fingerprint: &str,
        certificate_revision: u64,
        not_before_unix_seconds: u64,
        expires_at_unix_seconds: u64,
    ) -> Result<Self, ProducerCompatibilityErrorV1> {
        validate_reference_and_candidate(reference, candidate)?;
        // ubs:ignore — embedding-space fingerprints are public compatibility identities.
        if reference.space.fingerprint() != candidate.space.fingerprint() {
            return Err(ProducerCompatibilityErrorV1::SpaceMismatch);
        }
        let reference_producer_fingerprint = reference.producer.fingerprint();
        let candidate_producer_fingerprint = candidate.producer.fingerprint();
        if reference_producer_fingerprint == candidate_producer_fingerprint {
            return Err(ProducerCompatibilityErrorV1::CertificateForbiddenForExactProducer);
        }
        fixture.validate()?;
        if fixture.certificate != reference.producer.golden_vectors
            || fixture.certificate != candidate.producer.golden_vectors
        {
            return Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch);
        }
        if validate_sha256(
            "producer_conformance.policy_fingerprint",
            policy_fingerprint,
        )
        .is_err()
            || certificate_revision == 0
            || not_before_unix_seconds >= expires_at_unix_seconds
        {
            return Err(ProducerCompatibilityErrorV1::CertificateMalformed);
        }
        Ok(Self {
            schema_version: FOREIGN_PRODUCER_CONFORMANCE_CERTIFICATE_SCHEMA_V1,
            reference_producer_fingerprint,
            candidate_producer_fingerprint,
            space_fingerprint: reference.space.fingerprint(),
            golden_fixture_fingerprint: fixture.fingerprint.clone(),
            policy_fingerprint: policy_fingerprint.to_owned(),
            certificate_revision,
            not_before_unix_seconds,
            expires_at_unix_seconds,
        })
    }

    /// Domain-separated canonical bytes for policy pinning.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder =
            CanonicalEncoder::new(b"frankensearch.foreign-producer-conformance-certificate.v1");
        encoder.u16(self.schema_version);
        encoder.text(&self.reference_producer_fingerprint);
        encoder.text(&self.candidate_producer_fingerprint);
        encoder.text(&self.space_fingerprint);
        encoder.text(&self.golden_fixture_fingerprint);
        encoder.text(&self.policy_fingerprint);
        encoder.u64(self.certificate_revision);
        encoder.u64(self.not_before_unix_seconds);
        encoder.u64(self.expires_at_unix_seconds);
        encoder.finish()
    }

    /// Lowercase SHA-256 of [`Self::canonical_bytes`].
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }

    /// Certificate revision bound by the canonical fingerprint.
    #[must_use]
    pub const fn certificate_revision(&self) -> u64 {
        self.certificate_revision
    }

    /// Exclusive expiry bound in Unix seconds.
    #[must_use]
    pub const fn expires_at_unix_seconds(&self) -> u64 {
        self.expires_at_unix_seconds
    }

    fn validate(&self) -> Result<(), ProducerCompatibilityErrorV1> {
        if self.schema_version != FOREIGN_PRODUCER_CONFORMANCE_CERTIFICATE_SCHEMA_V1
            || self.certificate_revision == 0
            || self.not_before_unix_seconds >= self.expires_at_unix_seconds
        {
            return Err(ProducerCompatibilityErrorV1::CertificateMalformed);
        }
        for (field, fingerprint) in [
            (
                "producer_conformance.reference_producer_fingerprint",
                &self.reference_producer_fingerprint,
            ),
            (
                "producer_conformance.candidate_producer_fingerprint",
                &self.candidate_producer_fingerprint,
            ),
            (
                "producer_conformance.space_fingerprint",
                &self.space_fingerprint,
            ),
            (
                "producer_conformance.golden_fixture_fingerprint",
                &self.golden_fixture_fingerprint,
            ),
            (
                "producer_conformance.policy_fingerprint",
                &self.policy_fingerprint,
            ),
        ] {
            if validate_sha256(field, fingerprint).is_err() {
                return Err(ProducerCompatibilityErrorV1::CertificateMalformed);
            }
        }
        Ok(())
    }
}

impl<'a> TrustedProducerConformanceContextV1<'a> {
    /// Bind independently trusted policy, certificate, fixture, time, and revision state.
    ///
    /// The C1 foundation deliberately provides no production policy registry or
    /// consumer. Applications must source these pins from an owner-controlled
    /// channel independent of the raw certificate; this constructor does not
    /// authenticate caller authority and must never be populated from fields or
    /// digests learned only from the certificate under review.
    ///
    /// # Errors
    ///
    /// Returns [`ProducerCompatibilityErrorV1::CertificateMalformed`] for malformed
    /// pinned digests or an invalid revision window, and propagates fixture defects.
    pub fn from_independent_policy(
        policy_fingerprint: &'a str,
        certificate_fingerprint: &'a str,
        fixture: &'a VerifiedGoldenConformanceManifestV1,
        evaluation_time_unix_seconds: u64,
        minimum_certificate_revision: u64,
        maximum_certificate_revision: u64,
    ) -> Result<Self, ProducerCompatibilityErrorV1> {
        if validate_sha256(
            "trusted_producer_conformance.policy_fingerprint",
            policy_fingerprint,
        )
        .is_err()
            || validate_sha256(
                "trusted_producer_conformance.certificate_fingerprint",
                certificate_fingerprint,
            )
            .is_err()
            || minimum_certificate_revision == 0
            || minimum_certificate_revision > maximum_certificate_revision
        {
            return Err(ProducerCompatibilityErrorV1::CertificateMalformed);
        }
        fixture.validate()?;
        Ok(Self {
            policy_fingerprint,
            certificate_fingerprint,
            fixture,
            evaluation_time_unix_seconds,
            minimum_certificate_revision,
            maximum_certificate_revision,
        })
    }
}

impl ProducerCompatibilityWitnessV1 {
    /// Exact versus explicitly certified producer compatibility.
    #[must_use]
    pub const fn kind(&self) -> ProducerCompatibilityKindV1 {
        self.kind
    }

    /// Mathematical space shared by the producer pair.
    #[must_use]
    pub fn space_fingerprint(&self) -> &str {
        &self.space_fingerprint
    }

    /// Reference producer bound into the witness.
    #[must_use]
    pub fn reference_producer_fingerprint(&self) -> &str {
        &self.reference_producer_fingerprint
    }

    /// Candidate producer bound into the witness.
    #[must_use]
    pub fn candidate_producer_fingerprint(&self) -> &str {
        &self.candidate_producer_fingerprint
    }

    /// Canonical certificate fingerprint for certified witnesses.
    #[must_use]
    pub fn certificate_fingerprint(&self) -> Option<&str> {
        self.certificate_fingerprint.as_deref()
    }
}

fn validate_reference_and_candidate(
    reference: &EmbeddingIdentityBundleV1,
    candidate: &EmbeddingIdentityBundleV1,
) -> Result<(), ProducerCompatibilityErrorV1> {
    reference
        .validate()
        .map_err(|_| ProducerCompatibilityErrorV1::InvalidReferenceIdentity)?;
    candidate
        .validate()
        .map_err(|_| ProducerCompatibilityErrorV1::InvalidCandidateIdentity)
}

impl EmbeddingIdentityBundleV1 {
    /// Validate every component and all cross-component bindings.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when any component is malformed or a fingerprint
    /// or dimension does not bind the other bundled contracts exactly.
    pub fn validate(&self) -> Result<(), SearchError> {
        self.space.validate()?;
        self.producer.validate()?;
        self.input.validate()?;
        self.storage.validate()?;
        let space_fingerprint = self.space.fingerprint();
        if self.producer.space_fingerprint != space_fingerprint {
            return Err(identity_error(
                "producer.space_fingerprint",
                &self.producer.space_fingerprint,
                "does not bind the bundled space identity",
            ));
        }
        if self.space.kind == EmbeddingSpaceKindV1::HashControl
            && self.producer.provenance_manifest_fingerprint
                != self.space.artifact_manifest_fingerprint
        {
            return Err(identity_error(
                "producer.provenance_manifest_fingerprint",
                &self.producer.provenance_manifest_fingerprint,
                "hash controls must bind the canonical hash-control profile",
            ));
        }
        let input_fingerprint = self.input.fingerprint();
        if self.space.input_contract_fingerprint != input_fingerprint {
            return Err(identity_error(
                "space.input_contract_fingerprint",
                &self.space.input_contract_fingerprint,
                "does not bind the bundled input contract",
            ));
        }
        if self.storage.dimension != self.space.dimension {
            return Err(identity_error(
                "storage.dimension",
                &self.storage.dimension.to_string(),
                "does not match the bundled space dimension",
            ));
        }
        if self.storage.vector_normalization != self.space.output_normalization {
            return Err(identity_error(
                "storage.vector_normalization",
                &self.storage.vector_normalization,
                "does not match the bundled space output normalization",
            ));
        }
        if self.producer.golden_vectors.dimension != self.space.dimension {
            return Err(identity_error(
                "producer.golden_vectors.dimension",
                &self.producer.golden_vectors.dimension.to_string(),
                "does not match the bundled space dimension",
            ));
        }
        Ok(())
    }

    /// Canonical domain-separated bytes containing the four component fingerprints.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::new(b"frankensearch.embedding-bundle.v1");
        encoder.text(&self.space.fingerprint());
        encoder.text(&self.producer.fingerprint());
        encoder.text(&self.input.fingerprint());
        encoder.text(&self.storage.fingerprint());
        encoder.finish()
    }

    /// Lowercase SHA-256 of the complete identity bundle.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }

    /// Prove that a candidate carries the exact same producer identity.
    ///
    /// Storage identity is deliberately excluded: the same producer may persist
    /// an identical mathematical space in a different physical encoding. A
    /// same-space producer with any implementation, protocol, numeric, provenance,
    /// or golden-certificate drift must use the explicitly certified path.
    ///
    /// # Errors
    ///
    /// Returns a closed [`ProducerCompatibilityErrorV1`] when either bundle is
    /// invalid, the mathematical spaces differ, or the producer fingerprints are
    /// not identical.
    pub fn verify_exact_producer_with(
        &self,
        candidate: &Self,
    ) -> Result<ProducerCompatibilityWitnessV1, ProducerCompatibilityErrorV1> {
        validate_reference_and_candidate(self, candidate)?;
        let space_fingerprint = self.space.fingerprint();
        // ubs:ignore — embedding-space fingerprints are public compatibility identities.
        if space_fingerprint != candidate.space.fingerprint() {
            return Err(ProducerCompatibilityErrorV1::SpaceMismatch);
        }
        let reference_producer_fingerprint = self.producer.fingerprint();
        let candidate_producer_fingerprint = candidate.producer.fingerprint();
        if reference_producer_fingerprint != candidate_producer_fingerprint {
            return Err(ProducerCompatibilityErrorV1::CertificateRequired);
        }
        Ok(ProducerCompatibilityWitnessV1 {
            kind: ProducerCompatibilityKindV1::Exact,
            space_fingerprint,
            reference_producer_fingerprint,
            candidate_producer_fingerprint,
            certificate_fingerprint: None,
            policy_fingerprint: None,
            golden_fixture_fingerprint: None,
            certificate_revision: None,
            expires_at_unix_seconds: None,
        })
    }

    /// Prove that a distinct producer is admissible under an explicitly trusted
    /// conformance certificate.
    ///
    /// The certificate is direction-sensitive and binds the exact producer pair,
    /// mathematical space, independently executed golden fixture, policy,
    /// revision, and validity interval. The trusted context must independently
    /// pin the certificate's canonical fingerprint; copying fields from the raw
    /// certificate into the context does not establish a trustworthy boundary.
    ///
    /// # Errors
    ///
    /// Returns a closed [`ProducerCompatibilityErrorV1`] for invalid identities,
    /// cross-space use, exact-producer misuse, malformed or untrusted certificate
    /// state, binding drift, fixture disagreement, expiry, or revision-policy
    /// violations.
    pub fn verify_certified_foreign_producer_with(
        &self,
        candidate: &Self,
        certificate: &ForeignProducerConformanceCertificateV1,
        trusted: TrustedProducerConformanceContextV1<'_>,
    ) -> Result<ProducerCompatibilityWitnessV1, ProducerCompatibilityErrorV1> {
        validate_reference_and_candidate(self, candidate)?;
        let space_fingerprint = self.space.fingerprint();
        // ubs:ignore — space fingerprints are public compatibility identities, not secrets.
        if space_fingerprint != candidate.space.fingerprint() {
            return Err(ProducerCompatibilityErrorV1::SpaceMismatch);
        }
        let reference_producer_fingerprint = self.producer.fingerprint();
        let candidate_producer_fingerprint = candidate.producer.fingerprint();
        if reference_producer_fingerprint == candidate_producer_fingerprint {
            return Err(ProducerCompatibilityErrorV1::CertificateForbiddenForExactProducer);
        }

        certificate.validate()?;
        trusted.fixture.validate()?;
        let certificate_fingerprint = certificate.fingerprint();
        // ubs:ignore — certificate fingerprints are public integrity identities, not authenticators.
        if certificate_fingerprint != trusted.certificate_fingerprint {
            return Err(ProducerCompatibilityErrorV1::CertificateFingerprintMismatch);
        }
        // ubs:ignore — policy fingerprints are public conformance identities, not secrets.
        if certificate.policy_fingerprint != trusted.policy_fingerprint {
            return Err(ProducerCompatibilityErrorV1::PolicyMismatch);
        }
        if certificate.reference_producer_fingerprint != reference_producer_fingerprint
            || certificate.candidate_producer_fingerprint != candidate_producer_fingerprint
            || certificate.space_fingerprint != space_fingerprint
        {
            return Err(ProducerCompatibilityErrorV1::ProducerBindingMismatch);
        }
        // ubs:ignore — fixture fingerprints are public conformance identities, not secrets.
        if certificate.golden_fixture_fingerprint != trusted.fixture.fingerprint
            // ubs:ignore — golden-vector certificates are public conformance evidence.
            || trusted.fixture.certificate != self.producer.golden_vectors
            // ubs:ignore — golden-vector certificates are public conformance evidence.
            || trusted.fixture.certificate != candidate.producer.golden_vectors
        {
            return Err(ProducerCompatibilityErrorV1::FixtureBindingMismatch);
        }
        if certificate.certificate_revision < trusted.minimum_certificate_revision
            || certificate.certificate_revision > trusted.maximum_certificate_revision
        {
            return Err(ProducerCompatibilityErrorV1::CertificateRevisionOutsidePolicy);
        }
        if trusted.evaluation_time_unix_seconds < certificate.not_before_unix_seconds {
            return Err(ProducerCompatibilityErrorV1::CertificateNotYetValid);
        }
        if trusted.evaluation_time_unix_seconds >= certificate.expires_at_unix_seconds {
            return Err(ProducerCompatibilityErrorV1::CertificateExpired);
        }

        Ok(ProducerCompatibilityWitnessV1 {
            kind: ProducerCompatibilityKindV1::Certified,
            space_fingerprint,
            reference_producer_fingerprint,
            candidate_producer_fingerprint,
            certificate_fingerprint: Some(certificate_fingerprint),
            policy_fingerprint: Some(certificate.policy_fingerprint.clone()),
            golden_fixture_fingerprint: Some(certificate.golden_fixture_fingerprint.clone()),
            certificate_revision: Some(certificate.certificate_revision),
            expires_at_unix_seconds: Some(certificate.expires_at_unix_seconds),
        })
    }

    /// Validate and freeze the exact bytes that persistence must store.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when any component or cross-binding is invalid.
    pub fn freeze(&self) -> Result<FrozenEmbeddingIdentityBundleV1, SearchError> {
        self.validate()?;
        let canonical_bytes = self.canonical_bytes();
        Ok(FrozenEmbeddingIdentityBundleV1 {
            identity: self.clone(),
            fingerprint: sha256_hex(&canonical_bytes),
            canonical_bytes,
        })
    }

    /// Derive a complete identity bundle for a deterministic MRL/projection wrapper.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if the parent is invalid or the target is not a
    /// strict non-zero reduction.
    pub fn derive_projection(
        &self,
        target_dimension: u32,
        projection_rule: &str,
        renormalization_rule: &str,
    ) -> Result<Self, SearchError> {
        self.validate()?;
        let mut derived = self.clone();
        derived.space = self.space.derive_projection(
            target_dimension,
            projection_rule,
            renormalization_rule,
        )?;
        let parent_producer_fingerprint = self.producer.fingerprint();
        derived.producer.implementation_revision = format!(
            "frankensearch-identity-projection-wrapper-v1:parent={parent_producer_fingerprint}"
        );
        "deterministic-identity-projection-v1".clone_into(&mut derived.producer.protocol_revision);
        derived.producer.numeric_profile =
            format!("projection-and-renormalization-f32-v1:parent={parent_producer_fingerprint}");
        derived.producer.space_fingerprint = derived.space.fingerprint();
        derived.producer.golden_vectors.dimension = target_dimension;
        let mut certificate = CanonicalEncoder::new(b"frankensearch.projected-golden.v1");
        certificate.text(&self.producer.golden_vectors.vectors_sha256);
        certificate.u32(target_dimension);
        certificate.text(projection_rule);
        certificate.text(renormalization_rule);
        derived.producer.golden_vectors.vectors_sha256 = sha256_hex(&certificate.finish());
        derived.storage.dimension = target_dimension;
        renormalization_rule.clone_into(&mut derived.storage.vector_normalization);
        derived.validate()?;
        Ok(derived)
    }

    /// Construct an explicitly synthetic identity for tests and examples.
    ///
    /// The constructor is intentionally named and tagged as synthetic so it cannot
    /// be mistaken for verified semantic availability.
    #[must_use]
    pub fn explicit_test_model(model_id: &str, dimension: u32) -> Self {
        let seed_digest = sha256_hex(model_id.as_bytes());
        let input = EmbeddingInputContractV1 {
            schema_version: EMBEDDING_INPUT_CONTRACT_SCHEMA_V1,
            canonicalization: "explicit-test-identity-v1".to_owned(),
            content_selection: "caller-provided-test-text".to_owned(),
            chunking: "none".to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            doc_id_semantics: "test-only-no-document-binding".to_owned(),
        };
        let profile = HashControlProfileV1 {
            algorithm: "explicit-test-vector-source".to_owned(),
            algorithm_revision: "v1".to_owned(),
            seed: 0,
            feature_rules: model_id.to_owned(),
            tokenization_rules: "test-defined".to_owned(),
            signing_rules: "test-defined".to_owned(),
            normalization_rules: "test-defined".to_owned(),
        };
        let profile_fingerprint = profile.fingerprint();
        let space = EmbeddingSpaceIdentityV1 {
            schema_version: EMBEDDING_SPACE_IDENTITY_SCHEMA_V1,
            logical_model_id: model_id.to_owned(),
            immutable_revision: "explicit-test-v1".to_owned(),
            kind: EmbeddingSpaceKindV1::HashControl,
            artifact_manifest_fingerprint: profile_fingerprint.clone(),
            artifacts: Vec::new(),
            tokenizer_fingerprint: seed_digest.clone(),
            vocabulary_fingerprint: seed_digest.clone(),
            model_config_fingerprint: seed_digest.clone(),
            model_preprocessing: "test-defined".to_owned(),
            sequence_policy: "test-defined".to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            pooling: "test-defined".to_owned(),
            output_normalization: "test-defined".to_owned(),
            dimension,
            input_contract_fingerprint: input.fingerprint(),
            hash_control: Some(profile),
            projection: None,
        };
        let producer = EmbeddingProducerAttestationV1 {
            schema_version: EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
            backend: "explicit-test-backend".to_owned(),
            implementation_revision: "v1".to_owned(),
            protocol_revision: "in-process-v1".to_owned(),
            numeric_profile: "test-defined-f32".to_owned(),
            provenance_manifest_fingerprint: profile_fingerprint,
            space_fingerprint: space.fingerprint(),
            golden_vectors: GoldenVectorCertificateV1 {
                corpus_sha256: seed_digest.clone(),
                vectors_sha256: seed_digest,
                vector_count: 1,
                dimension,
            },
        };
        Self {
            space,
            producer,
            input,
            storage: VectorStorageIdentityV1 {
                schema_version: VECTOR_STORAGE_IDENTITY_SCHEMA_V1,
                format: "in-memory-test-vector-v1".to_owned(),
                quantization: QuantizationFormat::F32,
                endianness: "native-test-only".to_owned(),
                vector_normalization: "test-defined".to_owned(),
                dimension,
            },
        }
    }
}

impl FrozenEmbeddingIdentityBundleV1 {
    #[cfg(test)]
    pub(crate) fn explicit_test_model(model_id: &str, dimension: u32) -> Self {
        EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension)
            .freeze()
            .expect("explicit test identity must be valid")
    }

    /// Reject unknown schemas, noncanonical bytes, and bytes/digest disagreement.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the structured identity, canonical bytes, or
    /// stored digest disagree.
    pub fn validate(&self) -> Result<(), SearchError> {
        self.identity.validate()?;
        validate_sha256("frozen_bundle.fingerprint", &self.fingerprint)?;
        let canonical_bytes = self.identity.canonical_bytes();
        // ubs:ignore — these canonical authority bytes are public integrity evidence.
        if self.canonical_bytes != canonical_bytes {
            return Err(identity_error(
                "frozen_bundle.canonical_bytes",
                "redacted",
                "stored bytes disagree with canonical structured identity",
            ));
        }
        let fingerprint = sha256_hex(&canonical_bytes);
        // ubs:ignore — this fingerprint is public authority-integrity evidence, not a secret.
        if self.fingerprint != fingerprint {
            return Err(identity_error(
                "frozen_bundle.fingerprint",
                &self.fingerprint,
                "stored digest disagrees with canonical bytes",
            ));
        }
        Ok(())
    }
}

/// Frozen complete identity stored by generation manifests.
///
/// The former display-name/weights/dimension-only structure is deliberately
/// absorbed into exact canonical bytes plus their digest so no ambiguous
/// parallel revision type remains.
pub type EmbedderRevision = FrozenEmbeddingIdentityBundleV1;

#[derive(Debug)]
struct CanonicalEncoder {
    bytes: Vec<u8>,
}

impl CanonicalEncoder {
    fn new(domain: &[u8]) -> Self {
        let mut encoder = Self { bytes: Vec::new() };
        encoder.bytes(domain);
        encoder
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(u64::try_from(value).unwrap_or(u64::MAX));
    }

    fn bytes(&mut self, value: &[u8]) {
        self.usize(value.len());
        self.bytes.extend_from_slice(value);
    }

    fn text(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn option<T>(&mut self, value: Option<&T>, encode: impl FnOnce(&T, &mut Self)) {
        match value {
            Some(value) => {
                self.u8(1);
                encode(value, self);
            }
            None => self.u8(0),
        }
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Bounds-first reader for the fixed subset of canonical fields used by the
/// activation-manifest codec. It never allocates from a decoded length.
struct CanonicalDecoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> CanonicalDecoder<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn u8(&mut self, field: &'static str) -> Result<u8, GenerationAuthorityErrorV1> {
        Ok(self.take(1, field)?[0])
    }

    fn u16(&mut self, field: &'static str) -> Result<u16, GenerationAuthorityErrorV1> {
        Ok(u16::from_be_bytes(self.array(field)?))
    }

    fn u64(&mut self, field: &'static str) -> Result<u64, GenerationAuthorityErrorV1> {
        Ok(u64::from_be_bytes(self.array(field)?))
    }

    fn array<const N: usize>(
        &mut self,
        field: &'static str,
    ) -> Result<[u8; N], GenerationAuthorityErrorV1> {
        let mut result = [0_u8; N];
        result.copy_from_slice(self.take(N, field)?);
        Ok(result)
    }

    fn fixed_bytes<const N: usize>(
        &mut self,
        field: &'static str,
    ) -> Result<[u8; N], GenerationAuthorityErrorV1> {
        let bytes = self.bytes(field, N)?;
        if bytes.len() != N {
            return Err(GenerationAuthorityErrorV1::InvalidField { field });
        }
        let mut result = [0_u8; N];
        result.copy_from_slice(bytes);
        Ok(result)
    }

    fn bytes(
        &mut self,
        field: &'static str,
        maximum_len: usize,
    ) -> Result<&'a [u8], GenerationAuthorityErrorV1> {
        let length = usize::try_from(self.u64(field)?)
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField { field })?;
        if length > maximum_len {
            return Err(GenerationAuthorityErrorV1::InvalidField { field });
        }
        self.take(length, field)
    }

    fn component(
        &mut self,
        field: &'static str,
    ) -> Result<GenerationComponentReceiptV1, GenerationAuthorityErrorV1> {
        let byte_len = self.u64(field)?;
        let sha256 = self.fixed_bytes(field)?;
        Ok(GenerationComponentReceiptV1 { byte_len, sha256 })
    }

    fn finish(self) -> Result<(), GenerationAuthorityErrorV1> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes",
            })
        }
    }

    fn take(
        &mut self,
        len: usize,
        field: &'static str,
    ) -> Result<&'a [u8], GenerationAuthorityErrorV1> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(GenerationAuthorityErrorV1::InvalidField { field })?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(GenerationAuthorityErrorV1::InvalidField { field })?;
        self.offset = end;
        Ok(value)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

fn validate_schema(field: &str, actual: u16, expected: u16) -> Result<(), SearchError> {
    if actual == expected {
        return Ok(());
    }
    Err(identity_error(
        field,
        &actual.to_string(),
        &format!("unsupported schema; expected {expected}"),
    ))
}

fn validate_identity_text(field: &str, value: &str) -> Result<(), SearchError> {
    if value.len() > MAX_IDENTITY_FIELD_BYTES {
        return Err(identity_error(
            field,
            "redacted-oversized",
            "field exceeds the bounded identity size",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(identity_error(
            field,
            "redacted-control-character",
            "field must not contain control characters",
        ));
    }
    if value.trim().is_empty() {
        return Err(identity_error(field, value, "must not be empty"));
    }
    Ok(())
}

fn validate_optional_identity_text(field: &str, value: &str) -> Result<(), SearchError> {
    if value.len() > MAX_IDENTITY_FIELD_BYTES {
        return Err(identity_error(
            field,
            "redacted-oversized",
            "field exceeds the bounded identity size",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(identity_error(
            field,
            "redacted-control-character",
            "field must not contain control characters",
        ));
    }
    Ok(())
}

fn validate_sha256(field: &str, value: &str) -> Result<(), SearchError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(());
    }
    Err(identity_error(
        field,
        "redacted-invalid-sha256",
        "must be lowercase 64-character SHA-256",
    ))
}

fn identity_error(field: &str, value: &str, reason: &str) -> SearchError {
    let bounded_value = if value.len() <= 128 {
        value.to_owned()
    } else {
        format!("sha256:{}", sha256_hex(value.as_bytes()))
    };
    SearchError::InvalidConfig {
        field: format!("embedding_identity.{field}"),
        value: bounded_value,
        reason: reason.to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Artifact descriptors
// ---------------------------------------------------------------------------

/// Descriptor for a single FSVI vector index shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorArtifact {
    /// Relative path within the generation directory (e.g. `"vectors/shard_0.fsvi"`).
    pub path: String,
    /// Byte size of the artifact file.
    pub size_bytes: u64,
    /// Hex-encoded checksum (SHA-256) of the file contents.
    pub checksum: String,
    /// Number of vectors stored in this shard.
    pub vector_count: u64,
    /// Vector dimensionality.
    pub dimension: u32,
    /// Which embedder tier produced these vectors.
    pub embedder_tier: EmbedderTierTag,
}

/// Tag identifying which tier of the two-tier system produced an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedderTierTag {
    /// Fast tier (e.g. potion-128M, ~0.57ms).
    Fast,
    /// Quality tier (e.g. MiniLM-L6-v2, ~128ms).
    Quality,
}

/// Descriptor for a Tantivy lexical index segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalArtifact {
    /// Relative path within the generation directory (e.g. `"lexical/segment_0"`).
    pub path: String,
    /// Byte size of all files in the segment directory.
    pub size_bytes: u64,
    /// Hex-encoded checksum (SHA-256) of the concatenated segment files.
    pub checksum: String,
    /// Number of documents indexed in this segment.
    pub document_count: u64,
}

/// Metadata for `RaptorQ` repair symbols protecting an artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairDescriptor {
    /// Path of the protected artifact (matches a `VectorArtifact.path` or `LexicalArtifact.path`).
    pub protected_artifact: String,
    /// Path to the `.fec` sidecar file containing repair symbols.
    pub sidecar_path: String,
    /// Number of source symbols the artifact was split into.
    pub source_symbols: u32,
    /// Number of repair symbols generated.
    pub repair_symbols: u32,
    /// Overhead ratio (`repair_symbols` / `source_symbols`).
    pub overhead_ratio: f64,
}

// ---------------------------------------------------------------------------
// Activation invariants
// ---------------------------------------------------------------------------

/// A predicate that must hold before a generation can be activated for serving.
///
/// Activation invariants enforce all-or-nothing readiness: every invariant must
/// pass, or the generation is rejected and the previous generation continues serving.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationInvariant {
    /// Machine-readable invariant identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// The kind of check this invariant represents.
    pub kind: InvariantKind,
}

/// Classification of activation invariant checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvariantKind {
    /// All listed artifacts must be present and pass checksum verification.
    AllArtifactsVerified,
    /// Embedder revision must match the node's runtime embedder.
    EmbedderRevisionMatch,
    /// Total vector count must match document count (no missing embeddings).
    VectorCountConsistency {
        /// Expected total vectors across all shards.
        expected_total: u64,
    },
    /// Generation must cover a commit range that is contiguous with the previous
    /// activated generation (no gaps in commit history).
    CommitContinuity {
        /// The `high` value of the previous generation's commit range.
        previous_high: u64,
    },
    /// Custom predicate supplied by the deployment.
    Custom {
        /// Name of the custom check.
        check_name: String,
    },
}

// ---------------------------------------------------------------------------
// Generation manifest
// ---------------------------------------------------------------------------

/// Complete manifest for a search generation.
///
/// This is the unit of replication and activation in Native Mode distributed search.
/// A node fetches the manifest, verifies all artifacts, checks invariants, and
/// atomically swaps the active generation pointer on success.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationManifest {
    /// Exact schema version for fail-closed parsing.
    pub schema_version: u32,
    /// Unique identifier for this generation (content-derived or monotonic).
    pub generation_id: String,
    /// Hex-encoded hash (SHA-256) of the canonical serialized manifest body,
    /// computed with this field set to the empty string.
    pub manifest_hash: String,
    /// Commit range that produced this generation.
    pub commit_range: CommitRange,
    /// Timestamp (Unix millis) when generation build started.
    pub build_started_at: u64,
    /// Timestamp (Unix millis) when generation build completed.
    pub build_completed_at: u64,
    /// Embedder revisions used (keyed by tier tag stringified).
    pub embedders: BTreeMap<String, EmbedderRevision>,
    /// Vector index artifacts in this generation.
    pub vector_artifacts: Vec<VectorArtifact>,
    /// Lexical index artifacts in this generation.
    pub lexical_artifacts: Vec<LexicalArtifact>,
    /// Repair symbol descriptors for durability.
    pub repair_descriptors: Vec<RepairDescriptor>,
    /// Activation invariants that must all pass before serving.
    pub activation_invariants: Vec<ActivationInvariant>,
    /// Total document count across all artifacts.
    pub total_documents: u64,
    /// Optional free-form metadata (deployment tags, build host, etc.).
    pub metadata: BTreeMap<String, String>,
}

/// Current schema version for [`GenerationManifest`].
///
/// Version 2 replaces the ambiguous display-name/weights/dimension embedder
/// descriptor with the complete frozen embedding-identity bundle.
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Result of validating a [`GenerationManifest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationResult {
    /// Individual validation findings (empty means valid).
    pub findings: Vec<ValidationFinding>,
}

impl ValidationResult {
    /// Whether the manifest passes all validation checks.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.findings
            .iter()
            .all(|f| f.severity != FindingSeverity::Error)
    }

    /// Collect only error-severity findings.
    #[must_use]
    pub fn errors(&self) -> Vec<&ValidationFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Error)
            .collect()
    }

    /// Collect only warning-severity findings.
    #[must_use]
    pub fn warnings(&self) -> Vec<&ValidationFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Warning)
            .collect()
    }
}

/// A single validation finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationFinding {
    /// Which check produced this finding.
    pub check: &'static str,
    /// Severity of the finding.
    pub severity: FindingSeverity,
    /// Human-readable description.
    pub message: String,
}

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingSeverity {
    /// Informational, does not block activation.
    Info,
    /// Suspicious but not blocking.
    Warning,
    /// Blocks activation.
    Error,
}

/// Validates a [`GenerationManifest`] for structural and semantic correctness.
///
/// Returns a [`ValidationResult`] with all findings. Call [`ValidationResult::is_valid`]
/// to check whether the manifest is safe to activate.
#[must_use]
pub fn validate_manifest(manifest: &GenerationManifest) -> ValidationResult {
    let mut findings = Vec::new();

    check_schema_version(manifest, &mut findings);
    check_generation_id(manifest, &mut findings);
    check_manifest_hash(manifest, &mut findings);
    check_commit_range(manifest, &mut findings);
    check_timestamps(manifest, &mut findings);
    check_embedders(manifest, &mut findings);
    check_vector_artifacts(manifest, &mut findings);
    check_lexical_artifacts(manifest, &mut findings);
    check_repair_descriptors(manifest, &mut findings);
    check_activation_invariants(manifest, &mut findings);
    check_document_count_consistency(manifest, &mut findings);

    ValidationResult { findings }
}

/// Computes the canonical manifest hash for a generation manifest.
///
/// The canonical hash is SHA-256 over JSON serialization of the manifest with
/// `manifest_hash` cleared.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if serialization fails.
pub fn compute_manifest_hash(manifest: &GenerationManifest) -> crate::SearchResult<String> {
    let mut canonical = manifest.clone();
    canonical.manifest_hash.clear();
    let serialized =
        serde_json::to_vec(&canonical).map_err(|source| SearchError::SubsystemError {
            subsystem: "generation_manifest",
            source: Box::new(source),
        })?;
    Ok(lower_hex(Sha256::digest(serialized)))
}

fn lower_hex(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

/// Convert a validation result into a `SearchResult`, producing an error
/// if any error-severity findings exist.
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` when validation fails.
pub fn require_valid(result: &ValidationResult) -> crate::SearchResult<()> {
    if result.is_valid() {
        return Ok(());
    }
    let messages: Vec<String> = result.errors().iter().map(|f| f.message.clone()).collect();
    Err(SearchError::InvalidConfig {
        field: "generation_manifest".into(),
        value: String::new(),
        reason: messages.join("; "),
    })
}

// ---------------------------------------------------------------------------
// Individual checks
// ---------------------------------------------------------------------------

fn check_schema_version(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.schema_version != MANIFEST_SCHEMA_VERSION {
        f.push(ValidationFinding {
            check: "schema_version",
            severity: FindingSeverity::Error,
            message: format!(
                "schema_version {} is unsupported; expected exactly {}",
                m.schema_version, MANIFEST_SCHEMA_VERSION
            ),
        });
    }
}

fn check_generation_id(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.generation_id.is_empty() {
        f.push(ValidationFinding {
            check: "generation_id",
            severity: FindingSeverity::Error,
            message: "generation_id must not be empty".into(),
        });
    }
}

fn check_manifest_hash(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.manifest_hash.is_empty() {
        f.push(ValidationFinding {
            check: "manifest_hash",
            severity: FindingSeverity::Error,
            message: "manifest_hash must not be empty".into(),
        });
        return;
    }
    if !is_valid_sha256_hex(&m.manifest_hash) {
        f.push(ValidationFinding {
            check: "manifest_hash",
            severity: FindingSeverity::Error,
            message: "manifest_hash must be 64 lowercase/uppercase hex chars".into(),
        });
        return;
    }

    match compute_manifest_hash(m) {
        Ok(expected) => {
            if !m.manifest_hash.eq_ignore_ascii_case(&expected) {
                f.push(ValidationFinding {
                    check: "manifest_hash",
                    severity: FindingSeverity::Error,
                    message: format!(
                        "manifest_hash does not match canonical manifest body (expected {expected})"
                    ),
                });
            }
        }
        Err(err) => {
            f.push(ValidationFinding {
                check: "manifest_hash",
                severity: FindingSeverity::Error,
                message: format!("failed to recompute manifest_hash: {err}"),
            });
        }
    }
}

fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|c| c.is_ascii_hexdigit())
}

fn check_commit_range(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.commit_range.is_empty() {
        f.push(ValidationFinding {
            check: "commit_range",
            severity: FindingSeverity::Error,
            message: format!(
                "commit_range is invalid: high ({}) < low ({})",
                m.commit_range.high, m.commit_range.low
            ),
        });
    }
}

fn check_timestamps(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.build_started_at == 0 {
        f.push(ValidationFinding {
            check: "build_started_at",
            severity: FindingSeverity::Error,
            message: "build_started_at must be a positive Unix timestamp".into(),
        });
    }
    if m.build_completed_at == 0 {
        f.push(ValidationFinding {
            check: "build_completed_at",
            severity: FindingSeverity::Error,
            message: "build_completed_at must be a positive Unix timestamp".into(),
        });
    }
    if m.build_completed_at < m.build_started_at {
        f.push(ValidationFinding {
            check: "build_timestamps",
            severity: FindingSeverity::Error,
            message: format!(
                "build_completed_at ({}) is before build_started_at ({})",
                m.build_completed_at, m.build_started_at
            ),
        });
    }
}

fn check_embedders(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.embedders.is_empty() {
        f.push(ValidationFinding {
            check: "embedders",
            severity: FindingSeverity::Error,
            message: "at least one embedder revision must be specified".into(),
        });
    }
    for (key, rev) in &m.embedders {
        if let Err(error) = validate_identity_text("embedder_tier", key) {
            f.push(ValidationFinding {
                check: "embedder_tier",
                severity: FindingSeverity::Error,
                message: format!("embedder tier label is invalid: {error}"),
            });
            continue;
        }
        if let Err(error) = rev.validate() {
            f.push(ValidationFinding {
                check: "embedder_identity",
                severity: FindingSeverity::Error,
                message: format!("embedder '{key}' has invalid identity: {error}"),
            });
        }
    }
}

fn check_vector_artifacts(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    for (i, art) in m.vector_artifacts.iter().enumerate() {
        if art.path.is_empty() {
            f.push(ValidationFinding {
                check: "vector_artifact_path",
                severity: FindingSeverity::Error,
                message: format!("vector_artifacts[{i}] has empty path"),
            });
        }
        if art.checksum.is_empty() {
            f.push(ValidationFinding {
                check: "vector_artifact_checksum",
                severity: FindingSeverity::Error,
                message: format!("vector_artifacts[{i}] '{}' has empty checksum", art.path),
            });
        }
        if art.dimension == 0 {
            f.push(ValidationFinding {
                check: "vector_artifact_dimension",
                severity: FindingSeverity::Error,
                message: format!("vector_artifacts[{i}] '{}' has dimension 0", art.path),
            });
        }
    }

    // Check for duplicate paths.
    let mut seen = std::collections::HashSet::new();
    for art in &m.vector_artifacts {
        if !seen.insert(&art.path) {
            f.push(ValidationFinding {
                check: "vector_artifact_duplicate",
                severity: FindingSeverity::Error,
                message: format!("duplicate vector artifact path: '{}'", art.path),
            });
        }
    }
}

fn check_lexical_artifacts(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    for (i, art) in m.lexical_artifacts.iter().enumerate() {
        if art.path.is_empty() {
            f.push(ValidationFinding {
                check: "lexical_artifact_path",
                severity: FindingSeverity::Error,
                message: format!("lexical_artifacts[{i}] has empty path"),
            });
        }
        if art.checksum.is_empty() {
            f.push(ValidationFinding {
                check: "lexical_artifact_checksum",
                severity: FindingSeverity::Error,
                message: format!("lexical_artifacts[{i}] '{}' has empty checksum", art.path),
            });
        }
    }

    let mut seen = std::collections::HashSet::new();
    for art in &m.lexical_artifacts {
        if !seen.insert(&art.path) {
            f.push(ValidationFinding {
                check: "lexical_artifact_duplicate",
                severity: FindingSeverity::Error,
                message: format!("duplicate lexical artifact path: '{}'", art.path),
            });
        }
    }
}

fn check_repair_descriptors(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    let all_artifact_paths: std::collections::HashSet<&str> = m
        .vector_artifacts
        .iter()
        .map(|a| a.path.as_str())
        .chain(m.lexical_artifacts.iter().map(|a| a.path.as_str()))
        .collect();

    for (i, rd) in m.repair_descriptors.iter().enumerate() {
        if !all_artifact_paths.contains(rd.protected_artifact.as_str()) {
            f.push(ValidationFinding {
                check: "repair_descriptor_target",
                severity: FindingSeverity::Error,
                message: format!(
                    "repair_descriptors[{i}] references unknown artifact '{}'",
                    rd.protected_artifact
                ),
            });
        }
        if rd.source_symbols == 0 {
            f.push(ValidationFinding {
                check: "repair_descriptor_symbols",
                severity: FindingSeverity::Error,
                message: format!(
                    "repair_descriptors[{i}] for '{}' has 0 source symbols",
                    rd.protected_artifact
                ),
            });
        }
        if rd.overhead_ratio.is_nan() || rd.overhead_ratio < 0.0 || rd.overhead_ratio > 10.0 {
            f.push(ValidationFinding {
                check: "repair_descriptor_overhead",
                severity: FindingSeverity::Warning,
                message: format!(
                    "repair_descriptors[{i}] overhead ratio {} is outside expected range [0, 10]",
                    rd.overhead_ratio
                ),
            });
        }
    }
}

fn check_activation_invariants(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    let mut seen_ids = std::collections::HashSet::new();
    for inv in &m.activation_invariants {
        if inv.id.is_empty() {
            f.push(ValidationFinding {
                check: "invariant_id",
                severity: FindingSeverity::Error,
                message: "activation invariant has empty id".into(),
            });
        }
        if !seen_ids.insert(&inv.id) {
            f.push(ValidationFinding {
                check: "invariant_duplicate",
                severity: FindingSeverity::Error,
                message: format!("duplicate activation invariant id: '{}'", inv.id),
            });
        }
    }
}

fn check_document_count_consistency(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    let vector_total: u64 = m.vector_artifacts.iter().map(|a| a.vector_count).sum();
    let lexical_total: u64 = m.lexical_artifacts.iter().map(|a| a.document_count).sum();

    // Vector count should match declared total (per tier, so may be 2x for two-tier).
    // We only warn if there's a gross mismatch.
    if m.total_documents == 0 && (!m.vector_artifacts.is_empty() || !m.lexical_artifacts.is_empty())
    {
        f.push(ValidationFinding {
            check: "total_documents",
            severity: FindingSeverity::Error,
            message: "total_documents is 0 but artifacts are present".into(),
        });
    }

    if !m.lexical_artifacts.is_empty() && lexical_total != m.total_documents {
        f.push(ValidationFinding {
            check: "lexical_document_count",
            severity: FindingSeverity::Warning,
            message: format!(
                "lexical document count ({lexical_total}) != total_documents ({})",
                m.total_documents
            ),
        });
    }

    // For two-tier, vector_total may be 2 * total_documents (fast + quality tier).
    // Flag only if it doesn't match any reasonable multiple.
    if !m.vector_artifacts.is_empty()
        && vector_total != m.total_documents
        && vector_total != m.total_documents * 2
    {
        f.push(ValidationFinding {
            check: "vector_count_consistency",
            severity: FindingSeverity::Warning,
            message: format!(
                "vector count ({vector_total}) doesn't match total_documents ({}) or 2x (two-tier)",
                m.total_documents
            ),
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn authority_reference(sequence: u64, predecessor: Option<[u8; 32]>) -> AuthorityRefV1 {
        AuthorityRefV1::new(
            sequence,
            [u8::try_from(sequence).expect("small test sequence"); 16],
            4_096 + sequence,
            [u8::try_from(sequence + 16).expect("small test digest"); 32],
            predecessor,
        )
        .expect("valid test authority reference")
    }

    fn authority_slot(slot_index: u8, authority: AuthorityRefV1) -> AuthoritySlotV1 {
        AuthoritySlotV1 {
            slot_index,
            root_id: [0x5a; 16],
            authority,
        }
    }

    fn lock_frame(kind: GenerationLockFrameKindV1) -> GenerationLockFrameV1 {
        GenerationLockFrameV1::new(kind, [0x71; 16], [0x72; 16], [0x73; 16], 1, [0x74; 32])
            .expect("valid test lock frame")
    }

    fn authority_lock(
        kind: GenerationLockFrameKindV1,
        authority: AuthorityRefV1,
    ) -> GenerationLockFrameV1 {
        GenerationLockFrameV1::new(
            kind,
            [0x5a; 16],
            [0x72; 16],
            [0x73; 16],
            1,
            authority.fingerprint(),
        )
        .expect("valid authority-bound test lock frame")
    }

    fn component_receipts() -> GenerationComponentReceiptsV1 {
        GenerationComponentReceiptsV1 {
            vector: GenerationComponentReceiptV1 {
                byte_len: 101,
                sha256: [0x11; 32],
            },
            lexical: GenerationComponentReceiptV1 {
                byte_len: 102,
                sha256: [0x12; 32],
            },
            ann: GenerationComponentReceiptV1 {
                byte_len: 103,
                sha256: [0x13; 32],
            },
            metadata: GenerationComponentReceiptV1 {
                byte_len: 104,
                sha256: [0x14; 32],
            },
        }
    }

    fn activation_manifest(
        authority_sequence: u64,
        predecessor: Option<AuthorityRefV1>,
    ) -> ActivationManifestV1 {
        ActivationManifestV1::new(
            authority_sequence,
            predecessor,
            GenerationAuthorityActionV1::Activate,
            ArtifactGenerationIdentityV1::new(7, [0x21; 16]).expect("test generation"),
            [0x31; 32],
            [0x32; 32],
            [0x33; 32],
            component_receipts(),
        )
        .expect("valid activation manifest")
    }

    #[test]
    fn authority_slot_round_trips_and_authenticates_every_byte() {
        let authority = authority_reference(1, None);
        let slot = authority_slot(0, authority);
        let bytes = slot.encode().expect("encode slot");
        let decoded =
            AuthoritySlotV1::from_authenticated_bytes(&bytes, 0, [0x5a; 16]).expect("decode slot");
        assert_eq!(decoded, slot);

        let mut tampered = bytes;
        tampered[200] ^= 1;
        assert_eq!(
            AuthoritySlotV1::from_authenticated_bytes(&tampered, 0, [0x5a; 16]),
            Err(GenerationAuthorityErrorV1::ChecksumMismatch),
            "a byte outside the structured header must still be authenticated"
        );
    }

    #[test]
    fn authority_slot_rejects_every_single_byte_mutation() {
        let encoded = authority_slot(0, authority_reference(1, None))
            .encode()
            .expect("encode slot");
        for byte_index in 0..GENERATION_AUTHORITY_SLOT_BYTES_V1 {
            let mut mutated = encoded;
            mutated[byte_index] ^= 0x80;
            assert!(
                AuthoritySlotV1::from_authenticated_bytes(&mutated, 0, [0x5a; 16]).is_err(),
                "single-byte mutation at offset {byte_index} must never decode"
            );
        }
    }

    #[test]
    fn authority_slot_rejects_recomputed_noncanonical_padding_and_copied_slot() {
        let slot = authority_slot(0, authority_reference(1, None));
        let mut noncanonical = slot.encode().expect("encode slot");
        noncanonical[200] = 1;
        let digest = Sha256::digest(&noncanonical[..AUTHORITY_SLOT_BODY_BYTES]);
        noncanonical[AUTHORITY_SLOT_BODY_BYTES..].copy_from_slice(&digest);
        assert_eq!(
            AuthoritySlotV1::from_authenticated_bytes(&noncanonical, 0, [0x5a; 16]),
            Err(GenerationAuthorityErrorV1::NonCanonicalPadding),
            "a valid checksum does not permit noncanonical bytes"
        );
        assert_eq!(
            AuthoritySlotV1::from_authenticated_bytes(
                &slot.encode().expect("encode slot"),
                1,
                [0x5a; 16],
            ),
            Err(GenerationAuthorityErrorV1::SlotIndexMismatch),
            "a frame copied between physical slots must fail closed"
        );
    }

    #[test]
    fn fixed_authority_frames_reject_resealed_future_schemas() {
        let slot = authority_slot(0, authority_reference(1, None));
        let mut future_slot = slot.encode().expect("encode authority slot");
        future_slot[8..10].copy_from_slice(&(GENERATION_AUTHORITY_SCHEMA_V1 + 1).to_be_bytes());
        let slot_digest = Sha256::digest(&future_slot[..AUTHORITY_SLOT_BODY_BYTES]);
        future_slot[AUTHORITY_SLOT_BODY_BYTES..].copy_from_slice(&slot_digest);
        assert_eq!(
            AuthoritySlotV1::from_authenticated_bytes(&future_slot, 0, [0x5a; 16]),
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.header"
            }),
            "a future AUTHORITY frame is rejected even with a valid checksum"
        );

        let mut future_lock = lock_frame(GenerationLockFrameKindV1::Owner)
            .encode()
            .expect("encode owner lock frame");
        future_lock[8..10].copy_from_slice(&(GENERATION_AUTHORITY_SCHEMA_V1 + 1).to_be_bytes());
        let lock_digest = Sha256::digest(&future_lock[..LOCK_FRAME_BODY_BYTES]);
        future_lock[LOCK_FRAME_BODY_BYTES..].copy_from_slice(&lock_digest);
        assert_eq!(
            GenerationLockFrameV1::from_authenticated_bytes(&future_lock, [0x71; 16]),
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.header"
            }),
            "a future LOCK frame is rejected even with a valid checksum"
        );
    }

    #[test]
    fn lock_owner_and_attempt_frames_are_distinct_authenticated_contracts() {
        let owner = lock_frame(GenerationLockFrameKindV1::Owner);
        let owner_bytes = owner.encode().expect("encode owner lock frame");
        assert_eq!(
            GenerationLockFrameV1::from_authenticated_bytes(&owner_bytes, [0x71; 16])
                .expect("decode owner lock frame"),
            owner
        );

        let attempt = lock_frame(GenerationLockFrameKindV1::Attempt);
        assert_ne!(
            attempt.encode().expect("encode attempt lock frame"),
            owner_bytes,
            "an attempt must never serialize as an owner record"
        );

        let mut tampered = owner_bytes;
        tampered[300] ^= 1;
        assert_eq!(
            GenerationLockFrameV1::from_authenticated_bytes(&tampered, [0x71; 16]),
            Err(GenerationAuthorityErrorV1::ChecksumMismatch),
            "padding bytes remain authenticated"
        );
        assert_eq!(
            GenerationLockFrameV1::from_authenticated_bytes(&owner_bytes, [0x75; 16]),
            Err(GenerationAuthorityErrorV1::RootMismatch),
            "a lock frame copied across roots must fail closed"
        );
    }

    #[test]
    fn lock_frame_rejects_every_single_byte_mutation() {
        let encoded = lock_frame(GenerationLockFrameKindV1::Owner)
            .encode()
            .expect("encode owner lock frame");
        for byte_index in 0..GENERATION_LOCK_FRAME_BYTES_V1 {
            let mut mutated = encoded;
            mutated[byte_index] ^= 0x80;
            assert!(
                GenerationLockFrameV1::from_authenticated_bytes(&mutated, [0x71; 16]).is_err(),
                "single-byte mutation at offset {byte_index} must never decode"
            );
        }
    }

    #[test]
    fn lock_aware_resolver_requires_owner_agreement_and_attempt_reconciliation() {
        let genesis = authority_reference(1, None);
        let successor = authority_reference(2, Some(genesis.fingerprint()));
        let first = authority_slot(1, genesis);
        let second = authority_slot(0, successor);
        let owner = authority_lock(GenerationLockFrameKindV1::Owner, successor);
        assert_eq!(
            resolve_authority_slots_with_locks_v1(Some(first), Some(second), Some(owner), None),
            Ok(Some(second)),
            "a matching owner attests the resolved committed head"
        );

        let attempt = authority_lock(GenerationLockFrameKindV1::Attempt, successor);
        assert_eq!(
            resolve_authority_slots_with_locks_v1(
                Some(first),
                Some(second),
                Some(owner),
                Some(attempt),
            ),
            Err(GenerationAuthorityErrorV1::UnresolvedAttempt),
            "an authenticated attempt blocks fallback until publisher reconciliation"
        );

        let wrong_authority = GenerationLockFrameV1::new(
            GenerationLockFrameKindV1::Owner,
            [0x5a; 16],
            [0x72; 16],
            [0x73; 16],
            1,
            [0x91; 32],
        )
        .expect("well-formed mismatched owner lock");
        assert_eq!(
            resolve_authority_slots_with_locks_v1(
                Some(first),
                Some(second),
                Some(wrong_authority),
                None,
            ),
            Err(GenerationAuthorityErrorV1::LockAuthorityMismatch),
            "a valid lock for another authority must not select this head"
        );
    }

    #[test]
    fn lock_aware_resolver_rejects_wrong_roles_and_cross_root_evidence() {
        let genesis = authority_reference(1, None);
        let successor = authority_reference(2, Some(genesis.fingerprint()));
        let first = authority_slot(1, genesis);
        let second = authority_slot(0, successor);
        let owner = authority_lock(GenerationLockFrameKindV1::Owner, successor);
        assert_eq!(
            resolve_authority_slots_with_locks_v1(Some(first), Some(second), None, Some(owner)),
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_lock.attempt.kind"
            }),
            "owner frames cannot be reinterpreted as unresolved attempts"
        );

        let foreign_attempt = GenerationLockFrameV1::new(
            GenerationLockFrameKindV1::Attempt,
            [0x6b; 16],
            [0x72; 16],
            [0x73; 16],
            1,
            successor.fingerprint(),
        )
        .expect("well-formed foreign-root attempt");
        assert_eq!(
            resolve_authority_slots_with_locks_v1(
                Some(first),
                Some(second),
                None,
                Some(foreign_attempt),
            ),
            Err(GenerationAuthorityErrorV1::LockRootMismatch),
            "attempt evidence from another root cannot block or select this authority"
        );
    }

    #[test]
    fn authority_resolver_requires_an_exact_consecutive_predecessor() {
        let first = authority_reference(1, None);
        let second = authority_reference(2, Some(first.fingerprint()));
        let resolved = resolve_authority_slots_v1(
            Some(authority_slot(1, first)),
            Some(authority_slot(0, second)),
        )
        .expect("linked authority slots resolve")
        .expect("at least one authority");
        assert_eq!(resolved.authority, second);

        let unlinked = authority_reference(2, Some([0x11; 32]));
        assert_eq!(
            resolve_authority_slots_v1(
                Some(authority_slot(1, first)),
                Some(authority_slot(0, unlinked)),
            ),
            Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
            "a newer authority without the exact predecessor fingerprint is not selectable"
        );
    }

    #[test]
    fn authority_resolver_rejects_non_genesis_slot_parity_violation() {
        let genesis = authority_reference(1, None);
        let successor = authority_reference(2, Some(genesis.fingerprint()));
        assert_eq!(
            resolve_authority_slots_v1(
                Some(authority_slot(0, genesis)),
                Some(authority_slot(1, successor)),
            ),
            Err(GenerationAuthorityErrorV1::SlotSequenceParity),
            "a non-genesis authority copied into the opposite physical slot must fail"
        );
    }

    #[test]
    fn authority_resolver_rejects_repeated_physical_slots_and_lone_parity_tears() {
        let genesis = authority_reference(1, None);
        let successor = authority_reference(2, Some(genesis.fingerprint()));
        assert_eq!(
            resolve_authority_slots_v1(
                Some(authority_slot(0, genesis)),
                Some(authority_slot(0, successor)),
            ),
            Err(GenerationAuthorityErrorV1::DuplicatePhysicalSlot),
            "two inputs claiming one physical slot cannot represent a recoverable pair"
        );
        assert_eq!(
            resolve_authority_slots_v1(Some(authority_slot(1, successor)), None),
            Err(GenerationAuthorityErrorV1::SlotSequenceParity),
            "one surviving non-genesis slot must still bind its publication parity"
        );
        assert_eq!(
            resolve_authority_slots_v1(
                Some(authority_slot(0, successor)),
                Some(authority_slot(1, genesis)),
            ),
            Ok(Some(authority_slot(0, successor))),
            "resolver input order never changes the selected consecutive head"
        );
    }

    #[test]
    fn authority_resolver_accepts_consecutive_heads_across_slot_orientations() {
        let mut older = authority_reference(1, None);
        let mut older_slot_index = 1;
        for sequence in 2..=32 {
            let object_byte = u8::try_from(sequence).expect("bounded test sequence");
            let newer = AuthorityRefV1::new(
                sequence,
                [object_byte; 16],
                4_096 + sequence,
                [object_byte.wrapping_add(16); 32],
                Some(older.fingerprint()),
            )
            .expect("consecutive authority reference");
            let newer_slot_index = u8::try_from(sequence & 1).expect("slot parity fits u8");
            let older_slot = authority_slot(older_slot_index, older);
            let newer_slot = authority_slot(newer_slot_index, newer);
            assert_eq!(
                resolve_authority_slots_v1(Some(older_slot), Some(newer_slot)),
                Ok(Some(newer_slot)),
                "sequence {sequence} resolves in old/new input order"
            );
            assert_eq!(
                resolve_authority_slots_v1(Some(newer_slot), Some(older_slot)),
                Ok(Some(newer_slot)),
                "sequence {sequence} resolves in new/old input order"
            );
            older = newer;
            older_slot_index = newer_slot_index;
        }
    }

    #[test]
    fn raw_authority_frame_resolver_preserves_corruption_as_an_error() {
        let genesis = authority_reference(1, None);
        let successor = authority_reference(2, Some(genesis.fingerprint()));
        let first = authority_slot(0, successor)
            .encode()
            .expect("encode successor frame");
        let second = authority_slot(1, genesis)
            .encode()
            .expect("encode genesis frame");
        assert_eq!(
            resolve_authority_slot_frames_v1(Some(&first), Some(&second), [0x5a; 16]),
            Ok(Some(authority_slot(0, successor))),
            "two valid raw physical frames resolve their linked head"
        );
        assert_eq!(
            resolve_authority_slot_frames_v1(Some(&first), None, [0x5a; 16]),
            Ok(Some(authority_slot(0, successor))),
            "only a genuinely absent second frame permits first-slot survival"
        );

        let mut corrupt_second = second;
        corrupt_second[200] ^= 1;
        assert_eq!(
            resolve_authority_slot_frames_v1(Some(&first), Some(&corrupt_second), [0x5a; 16]),
            Err(GenerationAuthorityErrorV1::ChecksumMismatch),
            "a present corrupt frame must not be silently treated as absent"
        );
    }

    #[test]
    fn raw_authority_frame_resolver_rejects_length_and_order_tears() {
        let genesis = authority_reference(1, None);
        let successor = authority_reference(2, Some(genesis.fingerprint()));
        let first = authority_slot(0, successor)
            .encode()
            .expect("encode successor frame");
        let second = authority_slot(1, genesis)
            .encode()
            .expect("encode genesis frame");

        assert_eq!(
            resolve_authority_slot_frames_v1(Some(&first[..first.len() - 1]), None, [0x5a; 16]),
            Err(GenerationAuthorityErrorV1::InvalidSlotLength),
            "a truncated present frame is not one-slot survival"
        );
        let mut extended = first.to_vec();
        extended.push(0);
        assert_eq!(
            resolve_authority_slot_frames_v1(Some(&extended), None, [0x5a; 16]),
            Err(GenerationAuthorityErrorV1::InvalidSlotLength),
            "an extended present frame is not one-slot survival"
        );
        assert_eq!(
            resolve_authority_slot_frames_v1(Some(&second), Some(&first), [0x5a; 16]),
            Err(GenerationAuthorityErrorV1::SlotIndexMismatch),
            "reordered physical frames cannot swap old and new authorities"
        );
    }

    #[test]
    fn authority_floor_resolver_rejects_stale_forked_and_gapped_heads() {
        let genesis = authority_reference(1, None);
        let floor = AuthorityFloorV1::new([0x5a; 16], genesis).expect("valid authority floor");
        assert_eq!(
            resolve_authority_slots_at_floor_v1(Some(authority_slot(0, genesis)), None, floor),
            Ok(authority_slot(0, genesis)),
            "the exact retained authority is selectable"
        );

        let successor = authority_reference(2, Some(genesis.fingerprint()));
        assert_eq!(
            resolve_authority_slots_at_floor_v1(Some(authority_slot(0, successor)), None, floor),
            Ok(authority_slot(0, successor)),
            "the immediate predecessor-linked successor is selectable"
        );

        let fork = AuthorityRefV1::new(1, [0x91; 16], 4_097, [0x92; 32], None)
            .expect("well-formed forked authority");
        assert_eq!(
            resolve_authority_slots_at_floor_v1(Some(authority_slot(1, fork)), None, floor),
            Err(GenerationAuthorityErrorV1::EqualSequenceFork),
            "equal floor sequence with another immutable object is a fork"
        );

        let gap = AuthorityRefV1::new(3, [0x93; 16], 4_099, [0x94; 32], Some([0x95; 32]))
            .expect("well-formed but unprovable authority gap");
        assert_eq!(
            resolve_authority_slots_at_floor_v1(Some(authority_slot(1, gap)), None, floor),
            Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
            "a head beyond the immediate anchored successor requires more evidence"
        );

        let mut foreign_root = authority_slot(0, successor);
        foreign_root.root_id = [0x96; 16];
        assert_eq!(
            resolve_authority_slots_at_floor_v1(Some(foreign_root), None, floor),
            Err(GenerationAuthorityErrorV1::RootMismatch),
            "an externally retained floor cannot be replayed across roots"
        );
    }

    #[test]
    fn required_external_profile_never_silently_downgrades_to_local() {
        let genesis = authority_reference(1, None);
        let slot = authority_slot(0, genesis);
        assert_eq!(
            resolve_authority_slots_with_profile_v1(
                Some(slot),
                None,
                None,
                None,
                GenerationRootSecurityProfileV1::RequiredExternal,
                None,
            ),
            Err(GenerationAuthorityErrorV1::ExternalFloorRequired),
            "a valid local authority alone is insufficient for required-external admission"
        );

        let floor = AuthorityFloorV1::new([0x5a; 16], genesis).expect("valid external floor");
        assert_eq!(
            resolve_authority_slots_with_profile_v1(
                Some(slot),
                None,
                None,
                None,
                GenerationRootSecurityProfileV1::RequiredExternal,
                Some(floor),
            ),
            Ok(Some(slot)),
            "the exact retained external authority admits the root"
        );
    }

    #[test]
    fn local_and_inspection_profiles_keep_lock_reconciliation_mandatory() {
        let genesis = authority_reference(1, None);
        let slot = authority_slot(0, genesis);
        let attempt = authority_lock(GenerationLockFrameKindV1::Attempt, genesis);
        for profile in [
            GenerationRootSecurityProfileV1::CooperativeLocal,
            GenerationRootSecurityProfileV1::ReadOnlyUnanchored,
        ] {
            assert_eq!(
                resolve_authority_slots_with_profile_v1(
                    Some(slot),
                    None,
                    None,
                    Some(attempt),
                    profile,
                    None,
                ),
                Err(GenerationAuthorityErrorV1::UnresolvedAttempt),
                "{profile:?} cannot bypass an unresolved publication attempt"
            );
        }
    }

    #[test]
    fn root_security_profiles_explicitly_separate_mutation_authority() {
        assert!(GenerationRootSecurityProfileV1::RequiredExternal.permits_mutation());
        assert!(GenerationRootSecurityProfileV1::CooperativeLocal.permits_mutation());
        assert!(!GenerationRootSecurityProfileV1::ReadOnlyUnanchored.permits_mutation());
        assert_eq!(
            GenerationRootSecurityProfileV1::ReadOnlyUnanchored.require_mutation_authorized(),
            Err(GenerationAuthorityErrorV1::ReadOnlyProfile),
            "inspection-only admission cannot be reused for mutation"
        );
    }

    #[test]
    fn in_memory_anti_rollback_floor_is_exact_monotone_and_idempotent() {
        let store = InMemoryAntiRollbackFloorStoreV1::new();
        let genesis = authority_reference(1, None);
        let genesis_floor =
            AuthorityFloorV1::new([0x5a; 16], genesis).expect("valid genesis floor");
        let first = store
            .compare_and_advance(None, genesis_floor, [0x41; 16])
            .expect("first floor advance");
        assert_eq!(first.cas_version, 1);
        assert_eq!(store.load([0x5a; 16]), Ok(Some(first)));
        assert_eq!(
            store.compare_and_advance(None, genesis_floor, [0x41; 16]),
            Ok(first),
            "repeating the exact request returns its original receipt"
        );

        let successor = authority_reference(2, Some(genesis.fingerprint()));
        let successor_floor =
            AuthorityFloorV1::new([0x5a; 16], successor).expect("valid successor floor");
        let second = store
            .compare_and_advance(Some(first), successor_floor, [0x42; 16])
            .expect("monotone successor advance");
        assert_eq!(second.cas_version, 2);
        assert_eq!(store.load([0x5a; 16]), Ok(Some(second)));

        assert_eq!(
            store.compare_and_advance(Some(first), successor_floor, [0x43; 16]),
            Err(GenerationAuthorityErrorV1::FloorCompareAndAdvanceConflict),
            "a stale exact CAS receipt cannot overwrite a newer floor"
        );
        assert_eq!(
            store.compare_and_advance(Some(second), genesis_floor, [0x44; 16]),
            Err(GenerationAuthorityErrorV1::FloorSequenceRegression),
            "a valid older authority cannot roll the retained floor backward"
        );
        assert_eq!(
            store.compare_and_advance(Some(second), successor_floor, [0x41; 16]),
            Err(GenerationAuthorityErrorV1::FloorIdempotencyConflict),
            "a completed idempotency key cannot be rebound to another CAS"
        );
        assert_eq!(store.load([0x5a; 16]), Ok(Some(second)));
    }

    #[test]
    fn in_memory_anti_rollback_floor_rejects_invalid_root_and_request_identity() {
        let store = InMemoryAntiRollbackFloorStoreV1::new();
        let authority = authority_reference(1, None);
        let floor = AuthorityFloorV1::new([0x5a; 16], authority).expect("valid floor");
        assert_eq!(
            store.load([0; 16]),
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.root_id"
            })
        );
        assert_eq!(
            store.compare_and_advance(None, floor, [0; 16]),
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.idempotency_key"
            })
        );
        assert_eq!(store.load([0x5a; 16]), Ok(None));
    }

    #[test]
    fn authority_reference_fingerprint_uses_its_full_domain_separator() {
        let genesis = authority_reference(1, None);
        let bytes = genesis.canonical_bytes();
        assert_eq!(&bytes[..9], b"FSAUTHREF");
        assert_eq!(bytes.len(), 108);
        assert_ne!(genesis.fingerprint(), [0; 32]);
    }

    #[test]
    fn authority_reference_codec_round_trips_and_rejects_noncanonical_forms() {
        let genesis = authority_reference(1, None);
        let successor = authority_reference(2, Some(genesis.fingerprint()));
        assert_eq!(
            AuthorityRefV1::from_canonical_bytes(&successor.canonical_bytes())
                .expect("decode successor authority reference"),
            successor
        );

        let mut noncanonical_genesis = genesis.canonical_bytes();
        noncanonical_genesis[76] = 1;
        assert_eq!(
            AuthorityRefV1::from_canonical_bytes(&noncanonical_genesis),
            Err(GenerationAuthorityErrorV1::NonCanonicalPadding),
            "absent predecessors have a single all-zero representation"
        );

        let mut future_schema = successor.canonical_bytes();
        future_schema[9..11].copy_from_slice(&(GENERATION_AUTHORITY_SCHEMA_V1 + 1).to_be_bytes());
        assert_eq!(
            AuthorityRefV1::from_canonical_bytes(&future_schema),
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_ref.schema_version"
            }),
            "a self-consistent but unknown authority schema fails closed"
        );
        assert!(
            AuthorityRefV1::from_canonical_bytes(&successor.canonical_bytes()[..99]).is_err(),
            "truncated authority references never decode"
        );
    }

    #[test]
    fn authority_reference_never_rolls_over_its_sequence() {
        assert_eq!(authority_reference(1, None).next_sequence(), Ok(2));
        let terminal =
            AuthorityRefV1::new(u64::MAX, [0x81; 16], 4_096, [0x82; 32], Some([0x83; 32]))
                .expect("terminal authority reference remains decodable");
        assert_eq!(
            terminal.next_sequence(),
            Err(GenerationAuthorityErrorV1::SequenceExhausted),
            "terminal authority state must not wrap into an earlier generation"
        );
    }

    #[test]
    fn authority_resolver_accepts_only_duplicate_genesis() {
        let genesis = authority_reference(1, None);
        assert_eq!(
            resolve_authority_slots_v1(
                Some(authority_slot(0, genesis)),
                Some(authority_slot(1, genesis)),
            ),
            Ok(Some(authority_slot(0, genesis)))
        );

        let second = authority_reference(2, Some(genesis.fingerprint()));
        assert_eq!(
            resolve_authority_slots_v1(
                Some(authority_slot(0, second)),
                Some(authority_slot(1, second)),
            ),
            Err(GenerationAuthorityErrorV1::NonGenesisDuplicate)
        );
    }

    #[test]
    fn authority_resolver_rejects_forks_gaps_and_mixed_roots_but_keeps_one_slot_survival() {
        let genesis = authority_reference(1, None);
        assert_eq!(
            resolve_authority_slots_v1(Some(authority_slot(0, genesis)), None),
            Ok(Some(authority_slot(0, genesis))),
            "one authenticated surviving slot remains selectable"
        );

        let fork = AuthorityRefV1::new(1, [0x61; 16], 4_097, [0x62; 32], None)
            .expect("valid conflicting genesis-shaped reference");
        assert_eq!(
            resolve_authority_slots_v1(
                Some(authority_slot(0, genesis)),
                Some(authority_slot(1, fork)),
            ),
            Err(GenerationAuthorityErrorV1::EqualSequenceFork),
            "equal sequence with distinct identities is a fork, never a tie-break"
        );

        let skipped = AuthorityRefV1::new(3, [0x63; 16], 4_099, [0x64; 32], Some([0x65; 32]))
            .expect("well-formed but nonconsecutive authority reference");
        assert_eq!(
            resolve_authority_slots_v1(
                Some(authority_slot(0, genesis)),
                Some(authority_slot(1, skipped)),
            ),
            Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
            "a structurally credible gap must not silently select the newest slot"
        );

        let mut other_root = authority_slot(1, genesis);
        other_root.root_id = [0x66; 16];
        assert_eq!(
            resolve_authority_slots_v1(Some(authority_slot(0, genesis)), Some(other_root)),
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "authority_slot.pair"
            }),
            "two slots from different roots cannot be combined"
        );
    }

    #[test]
    fn activation_manifest_self_seals_all_component_and_transition_witnesses() {
        let manifest = activation_manifest(1, None);
        manifest.validate().expect("self-sealed manifest validates");
        let receipt = manifest.object_receipt();
        assert!(receipt.0 > 0);
        assert_ne!(receipt.1, [0; 32]);

        let mut substituted = manifest.clone();
        substituted.components.metadata.sha256[0] ^= 1;
        assert_eq!(
            substituted.validate(),
            Err(GenerationAuthorityErrorV1::ManifestSelfSealMismatch),
            "a component substitution must invalidate the activation-manifest self-seal"
        );
    }

    #[test]
    fn activation_manifest_codec_round_trips_and_rejects_noncanonical_boundaries() {
        let manifest = activation_manifest(1, None);
        let bytes = manifest.canonical_bytes();
        assert_eq!(
            ActivationManifestV1::from_canonical_bytes(&bytes).expect("canonical decode"),
            manifest,
            "canonical decode must reproduce the exact structured manifest"
        );

        let mut extended = bytes.clone();
        extended.push(0);
        assert!(
            ActivationManifestV1::from_canonical_bytes(&extended).is_err(),
            "trailing bytes must not become a second representation"
        );
        assert!(
            ActivationManifestV1::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err(),
            "a truncated self-seal must fail before selecting the manifest"
        );
        let oversized = vec![0; GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V1 + 1];
        assert_eq!(
            ActivationManifestV1::from_canonical_bytes(&oversized),
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.canonical_bytes"
            }),
            "the manifest ceiling is enforced before decode"
        );

        let mut future = manifest;
        future.schema_version = GENERATION_AUTHORITY_SCHEMA_V1 + 1;
        future.self_seal_sha256 = future.computed_self_seal();
        assert_eq!(
            ActivationManifestV1::from_canonical_bytes(&future.canonical_bytes()),
            Err(GenerationAuthorityErrorV1::InvalidField {
                field: "activation_manifest.schema_version"
            }),
            "a self-consistent future schema still fails closed"
        );
    }

    #[test]
    fn activation_manifest_rejects_every_single_byte_mutation() {
        let encoded = activation_manifest(1, None).canonical_bytes();
        for byte_index in 0..encoded.len() {
            let mut mutated = encoded.clone();
            mutated[byte_index] ^= 0x80;
            assert!(
                ActivationManifestV1::from_canonical_bytes(&mutated).is_err(),
                "single-byte mutation at offset {byte_index} must never decode"
            );
        }
    }

    #[test]
    fn activation_manifest_requires_the_exact_preceding_authority_sequence() {
        let genesis = activation_manifest(1, None);
        let (manifest_len, manifest_sha256) = genesis.object_receipt();
        let predecessor = AuthorityRefV1::new(1, [0x41; 16], manifest_len, manifest_sha256, None)
            .expect("genesis authority reference");
        let successor = activation_manifest(2, Some(predecessor));
        successor
            .validate()
            .expect("consecutive predecessor validates");

        assert_eq!(
            ActivationManifestV1::new(
                3,
                Some(predecessor),
                GenerationAuthorityActionV1::Activate,
                ArtifactGenerationIdentityV1::new(7, [0x21; 16]).expect("test generation"),
                [0x31; 32],
                [0x32; 32],
                [0x33; 32],
                component_receipts(),
            ),
            Err(GenerationAuthorityErrorV1::BrokenPredecessorLink),
            "an activation manifest may not skip an authority sequence"
        );
    }

    #[test]
    fn rollback_uses_a_new_authority_without_reusing_artifact_identity() {
        let genesis = activation_manifest(1, None);
        let (genesis_len, genesis_sha256) = genesis.object_receipt();
        let first = AuthorityRefV1::new(1, [0x41; 16], genesis_len, genesis_sha256, None)
            .expect("first authority reference");
        let second = AuthorityRefV1::new(
            2,
            [0x42; 16],
            genesis_len,
            genesis_sha256,
            Some(first.fingerprint()),
        )
        .expect("second authority reference");
        let rollback = ActivationManifestV1::new(
            3,
            Some(second),
            GenerationAuthorityActionV1::Rollback,
            ArtifactGenerationIdentityV1::new(7, [0x21; 16]).expect("reselected generation"),
            [0x31; 32],
            [0x32; 32],
            [0x33; 32],
            component_receipts(),
        )
        .expect("higher-authority rollback is canonical");
        assert_eq!(rollback.authority_sequence, 3);
        assert_eq!(rollback.generation.sequence, 7);
        assert_eq!(rollback.action, GenerationAuthorityActionV1::Rollback);
        rollback.validate().expect("rollback self-seal validates");
    }

    #[test]
    fn authority_reference_requires_the_exact_self_sealed_manifest_object() {
        let manifest = activation_manifest(1, None);
        let (manifest_len, manifest_sha256) = manifest.object_receipt();
        let authority = AuthorityRefV1::new(1, [0x51; 16], manifest_len, manifest_sha256, None)
            .expect("authority names test manifest");
        verify_authority_manifest_reference_v1(&authority, &manifest)
            .expect("exact authority manifest binding");

        let wrong_length =
            AuthorityRefV1::new(1, [0x51; 16], manifest_len + 1, manifest_sha256, None)
                .expect("otherwise valid authority reference");
        assert_eq!(
            verify_authority_manifest_reference_v1(&wrong_length, &manifest),
            Err(GenerationAuthorityErrorV1::ManifestReferenceMismatch),
            "a length-mismatched external object must not be selected"
        );
    }

    fn sample_embedder() -> EmbedderRevision {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("potion-128M", 256);
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F16;
        identity.storage.endianness = "little-endian".to_owned();
        identity.freeze().unwrap()
    }

    fn sample_semantic_identity() -> EmbeddingIdentityBundleV1 {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("semantic-test", 8);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "1".repeat(64);
        identity.space.artifacts = vec![
            EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "2".repeat(64),
                size: 10,
            },
            EmbeddingArtifactIdentityV1 {
                role: "tokenizer".to_owned(),
                sha256: "3".repeat(64),
                size: 20,
            },
        ];
        identity.space.tokenizer_fingerprint = "3".repeat(64);
        identity.space.vocabulary_fingerprint = "4".repeat(64);
        identity.space.model_config_fingerprint = "5".repeat(64);
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().expect("sample semantic identity");
        identity
    }

    fn sample_foreign_producer_pair() -> (
        EmbeddingIdentityBundleV1,
        EmbeddingIdentityBundleV1,
        VerifiedGoldenConformanceManifestV1,
    ) {
        let texts = ["query: alpha", "document: beta"];
        let vectors = vec![
            vec![0.0, -0.0, 1.0, -1.0, 0.25, 0.5, 0.75, 1.25],
            vec![2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5],
        ];
        let fixture =
            VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(&texts, &vectors, &vectors)
                .expect("exact fixture");

        let mut reference = sample_semantic_identity();
        reference.producer.golden_vectors = fixture.certificate().clone();
        reference.validate().expect("reference identity");

        let mut candidate = reference.clone();
        candidate.producer.backend = "alternate-native-kernel".to_owned();
        candidate.producer.implementation_revision = "alternate-implementation-v2".to_owned();
        candidate.producer.protocol_revision = "alternate-protocol-v3".to_owned();
        candidate.producer.numeric_profile = "deterministic-f32-alternate-v1".to_owned();
        candidate.producer.provenance_manifest_fingerprint = "6".repeat(64);
        candidate.validate().expect("candidate identity");
        assert_ne!(
            reference.producer.fingerprint(),
            candidate.producer.fingerprint()
        );

        (reference, candidate, fixture)
    }

    #[test]
    fn artifact_generation_identity_is_full_width_and_round_trips() {
        let identity =
            ArtifactGenerationIdentityV1::new(u64::MAX, [0xa5; 16]).expect("valid generation");
        identity.validate().expect("generation validates");

        let encoded = serde_json::to_vec(&identity).expect("serialize generation");
        let decoded: ArtifactGenerationIdentityV1 =
            serde_json::from_slice(&encoded).expect("deserialize generation");

        assert_eq!(decoded, identity);
        assert_eq!(decoded.sequence, u64::MAX);
        assert_eq!(decoded.fingerprint().len(), 64);
        assert_eq!(decoded.fingerprint(), identity.fingerprint());
    }

    #[test]
    fn artifact_generation_identity_rejects_reserved_or_unknown_values() {
        assert!(ArtifactGenerationIdentityV1::new(0, [0; 16]).is_err());

        let mut unknown_schema =
            ArtifactGenerationIdentityV1::new(0, [1; 16]).expect("valid generation");
        unknown_schema.schema_version = ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1 + 1;
        assert!(unknown_schema.validate().is_err());

        let injected = serde_json::json!({
            "schema_version": ARTIFACT_GENERATION_IDENTITY_SCHEMA_V1,
            "sequence": 7,
            "nonce": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "unexpected": "field"
        });
        assert!(
            serde_json::from_value::<ArtifactGenerationIdentityV1>(injected).is_err(),
            "unknown generation fields must fail closed"
        );
    }

    #[test]
    fn every_artifact_generation_field_changes_the_fingerprint() {
        let base = ArtifactGenerationIdentityV1::new(7, [1; 16]).expect("valid base generation");

        let mut changed_sequence = base;
        changed_sequence.sequence += 1;
        assert_ne!(base.fingerprint(), changed_sequence.fingerprint());

        let mut changed_nonce = base;
        changed_nonce.nonce[15] ^= 1;
        assert_ne!(base.fingerprint(), changed_nonce.fingerprint());

        let mut changed_schema = base;
        changed_schema.schema_version += 1;
        assert_ne!(base.fingerprint(), changed_schema.fingerprint());
        assert!(changed_schema.validate().is_err());
    }

    #[test]
    fn golden_certificate_binds_order_shape_and_exact_f32_bits() {
        let texts = ["", "Unicode café", "signed zero"];
        let vectors = vec![
            vec![0.0, -0.0],
            vec![1.0, f32::from_bits(0x7fc0_0042)],
            vec![-3.5, f32::INFINITY],
        ];
        let certificate = GoldenVectorCertificateV1::from_exact_f32(&texts, &vectors).unwrap();
        certificate.verify_exact_f32(&texts, &vectors).unwrap();

        let mut reordered_texts = texts;
        reordered_texts.swap(0, 1);
        assert!(
            certificate
                .verify_exact_f32(&reordered_texts, &vectors)
                .is_err()
        );

        let mut changed_bits = vectors.clone();
        changed_bits[0][0] = -0.0;
        assert!(certificate.verify_exact_f32(&texts, &changed_bits).is_err());

        let inconsistent = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(GoldenVectorCertificateV1::from_exact_f32(&texts[..2], &inconsistent).is_err());
    }

    #[test]
    fn exact_producer_witness_allows_storage_drift_but_rejects_foreign_producers() {
        let (reference, foreign, _) = sample_foreign_producer_pair();
        assert_eq!(
            reference.verify_exact_producer_with(&foreign),
            Err(ProducerCompatibilityErrorV1::CertificateRequired)
        );

        let mut alternate_storage = reference.clone();
        alternate_storage.storage.format = "fsvi-v2".to_owned();
        alternate_storage.storage.quantization = QuantizationFormat::F16;
        alternate_storage.storage.endianness = "little-endian".to_owned();
        alternate_storage
            .validate()
            .expect("valid alternate storage");
        assert_ne!(reference.fingerprint(), alternate_storage.fingerprint());

        let witness = reference
            .verify_exact_producer_with(&alternate_storage)
            .expect("exact producer remains compatible across storage encodings");
        assert_eq!(witness.kind(), ProducerCompatibilityKindV1::Exact);
        assert_eq!(
            witness.reference_producer_fingerprint(),
            witness.candidate_producer_fingerprint()
        );
        assert!(witness.certificate_fingerprint().is_none());
    }

    #[test]
    fn certified_foreign_producer_requires_pinned_directional_evidence() {
        let (reference, candidate, fixture) = sample_foreign_producer_pair();
        let policy_fingerprint = "9".repeat(64);
        let certificate =
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &reference,
                &candidate,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            )
            .expect("certificate receipt");
        let certificate_fingerprint = certificate.fingerprint();
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &certificate_fingerprint,
            &fixture,
            150,
            7,
            7,
        )
        .expect("trusted context");

        let witness = reference
            .verify_certified_foreign_producer_with(&candidate, &certificate, trusted)
            .expect("trusted foreign producer");
        assert_eq!(witness.kind(), ProducerCompatibilityKindV1::Certified);
        assert_eq!(
            witness.certificate_fingerprint(),
            Some(certificate_fingerprint.as_str())
        );
        assert_eq!(
            witness.reference_producer_fingerprint(),
            reference.producer.fingerprint()
        );
        assert_eq!(
            witness.candidate_producer_fingerprint(),
            candidate.producer.fingerprint()
        );

        let serialized = serde_json::to_string(&witness).expect("serialize opaque witness");
        assert!(!serialized.contains("alternate-native-kernel"));
        assert!(!serialized.contains("query: alpha"));
        assert!(!serialized.contains("document: beta"));
        assert!(!serialized.contains("canonical_bytes"));
    }

    #[test]
    fn every_foreign_certificate_field_is_bound_by_the_pinned_fingerprint() {
        let (reference, candidate, fixture) = sample_foreign_producer_pair();
        let policy_fingerprint = "9".repeat(64);
        let certificate =
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &reference,
                &candidate,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            )
            .unwrap();
        let pinned_fingerprint = certificate.fingerprint();

        let mut unknown_schema = certificate.clone();
        unknown_schema.schema_version += 1;
        let mut changed_reference = certificate.clone();
        changed_reference.reference_producer_fingerprint = "a".repeat(64);
        let mut changed_candidate = certificate.clone();
        changed_candidate.candidate_producer_fingerprint = "b".repeat(64);
        let mut changed_space = certificate.clone();
        changed_space.space_fingerprint = "c".repeat(64);
        let mut changed_fixture = certificate.clone();
        changed_fixture.golden_fixture_fingerprint = "d".repeat(64);
        let mut changed_policy = certificate.clone();
        changed_policy.policy_fingerprint = "e".repeat(64);
        let mut changed_revision = certificate.clone();
        changed_revision.certificate_revision += 1;
        let mut changed_start = certificate.clone();
        changed_start.not_before_unix_seconds += 1;
        let mut changed_expiry = certificate.clone();
        changed_expiry.expires_at_unix_seconds -= 1;

        let malformed_context = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &pinned_fingerprint,
            &fixture,
            150,
            1,
            u64::MAX,
        )
        .unwrap();
        assert_eq!(
            reference.verify_certified_foreign_producer_with(
                &candidate,
                &unknown_schema,
                malformed_context,
            ),
            Err(ProducerCompatibilityErrorV1::CertificateMalformed)
        );

        for (label, mutated) in [
            ("reference producer", changed_reference),
            ("candidate producer", changed_candidate),
            ("space", changed_space),
            ("fixture", changed_fixture),
            ("policy", changed_policy),
            ("revision", changed_revision),
            ("not-before", changed_start),
            ("expiry", changed_expiry),
        ] {
            let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
                &policy_fingerprint,
                &pinned_fingerprint,
                &fixture,
                150,
                1,
                u64::MAX,
            )
            .unwrap();
            assert_eq!(
                reference.verify_certified_foreign_producer_with(&candidate, &mutated, trusted,),
                Err(ProducerCompatibilityErrorV1::CertificateFingerprintMismatch),
                "mutated {label} must invalidate the pinned certificate"
            );
        }
    }

    #[test]
    fn foreign_certificate_rejects_policy_pair_fixture_and_bundle_substitution() {
        let (reference, candidate, fixture) = sample_foreign_producer_pair();
        let policy_fingerprint = "9".repeat(64);
        let certificate =
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &reference,
                &candidate,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            )
            .unwrap();

        let mut wrong_policy = certificate.clone();
        wrong_policy.policy_fingerprint = "8".repeat(64);
        let wrong_policy_fingerprint = wrong_policy.fingerprint();
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &wrong_policy_fingerprint,
            &fixture,
            150,
            7,
            7,
        )
        .unwrap();
        assert_eq!(
            reference.verify_certified_foreign_producer_with(&candidate, &wrong_policy, trusted,),
            Err(ProducerCompatibilityErrorV1::PolicyMismatch)
        );

        let swapped =
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &candidate,
                &reference,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            )
            .unwrap();
        let swapped_fingerprint = swapped.fingerprint();
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &swapped_fingerprint,
            &fixture,
            150,
            7,
            7,
        )
        .unwrap();
        assert_eq!(
            reference.verify_certified_foreign_producer_with(&candidate, &swapped, trusted),
            Err(ProducerCompatibilityErrorV1::ProducerBindingMismatch)
        );

        for (label, reference_slot, substituted_fingerprint) in [
            (
                "reference storage fingerprint",
                true,
                reference.storage.fingerprint(),
            ),
            (
                "candidate storage fingerprint",
                false,
                candidate.storage.fingerprint(),
            ),
            (
                "reference full-bundle fingerprint",
                true,
                reference.fingerprint(),
            ),
            (
                "candidate full-bundle fingerprint",
                false,
                candidate.fingerprint(),
            ),
        ] {
            let mut substitution = certificate.clone();
            if reference_slot {
                substitution.reference_producer_fingerprint = substituted_fingerprint;
            } else {
                substitution.candidate_producer_fingerprint = substituted_fingerprint;
            }
            let substitution_fingerprint = substitution.fingerprint();
            let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
                &policy_fingerprint,
                &substitution_fingerprint,
                &fixture,
                150,
                7,
                7,
            )
            .unwrap();
            assert_eq!(
                reference.verify_certified_foreign_producer_with(
                    &candidate,
                    &substitution,
                    trusted,
                ),
                Err(ProducerCompatibilityErrorV1::ProducerBindingMismatch),
                "{label} must never stand in for a producer fingerprint"
            );
        }

        let mut fixture_substitution = certificate.clone();
        fixture_substitution.golden_fixture_fingerprint = "a".repeat(64);
        let fixture_substitution_fingerprint = fixture_substitution.fingerprint();
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &fixture_substitution_fingerprint,
            &fixture,
            150,
            7,
            7,
        )
        .unwrap();
        assert_eq!(
            reference.verify_certified_foreign_producer_with(
                &candidate,
                &fixture_substitution,
                trusted,
            ),
            Err(ProducerCompatibilityErrorV1::FixtureBindingMismatch)
        );

        let mut fabricated_candidate = candidate.clone();
        fabricated_candidate.producer.golden_vectors.vectors_sha256 = "f".repeat(64);
        fabricated_candidate.validate().unwrap();
        assert_eq!(
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &reference,
                &fabricated_candidate,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            ),
            Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch)
        );

        let mut fabricated_reference = reference.clone();
        let mut fabricated_candidate = candidate;
        fabricated_reference.producer.golden_vectors.vectors_sha256 = "e".repeat(64);
        fabricated_candidate.producer.golden_vectors.vectors_sha256 = "e".repeat(64);
        fabricated_reference.validate().unwrap();
        fabricated_candidate.validate().unwrap();
        assert_eq!(
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &fabricated_reference,
                &fabricated_candidate,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            ),
            Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch),
            "two matching self-asserted summaries cannot replace the independently executed fixture"
        );
    }

    #[test]
    fn foreign_certificate_enforces_time_revision_and_full_width_boundaries() {
        let (reference, candidate, fixture) = sample_foreign_producer_pair();
        let policy_fingerprint = "9".repeat(64);
        let certificate =
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &reference,
                &candidate,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            )
            .unwrap();
        let certificate_fingerprint = certificate.fingerprint();

        for (time, expected) in [
            (
                99,
                Err(ProducerCompatibilityErrorV1::CertificateNotYetValid),
            ),
            (100, Ok(ProducerCompatibilityKindV1::Certified)),
            (199, Ok(ProducerCompatibilityKindV1::Certified)),
            (200, Err(ProducerCompatibilityErrorV1::CertificateExpired)),
        ] {
            let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
                &policy_fingerprint,
                &certificate_fingerprint,
                &fixture,
                time,
                7,
                7,
            )
            .unwrap();
            let observed = reference
                .verify_certified_foreign_producer_with(&candidate, &certificate, trusted)
                .map(|witness| witness.kind());
            assert_eq!(observed, expected, "evaluation time {time}");
        }

        for (minimum_revision, maximum_revision) in [(8, 8), (1, 6)] {
            let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
                &policy_fingerprint,
                &certificate_fingerprint,
                &fixture,
                150,
                minimum_revision,
                maximum_revision,
            )
            .unwrap();
            assert_eq!(
                reference
                    .verify_certified_foreign_producer_with(&candidate, &certificate, trusted,),
                Err(ProducerCompatibilityErrorV1::CertificateRevisionOutsidePolicy)
            );
        }

        let full_width =
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &reference,
                &candidate,
                &fixture,
                &policy_fingerprint,
                u64::MAX,
                u64::MAX - 2,
                u64::MAX,
            )
            .unwrap();
        let full_width_fingerprint = full_width.fingerprint();
        let trusted = TrustedProducerConformanceContextV1::from_independent_policy(
            &policy_fingerprint,
            &full_width_fingerprint,
            &fixture,
            u64::MAX - 1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        assert_eq!(
            reference
                .verify_certified_foreign_producer_with(&candidate, &full_width, trusted)
                .unwrap()
                .kind(),
            ProducerCompatibilityKindV1::Certified
        );
    }

    #[test]
    fn producer_compatibility_rejects_input_tokenizer_and_invalid_identity_drift() {
        let (reference, candidate, fixture) = sample_foreign_producer_pair();
        let policy_fingerprint = "9".repeat(64);

        let mut changed_input = candidate.clone();
        changed_input.input.canonicalization.push_str("-drift");
        changed_input.space.input_contract_fingerprint = changed_input.input.fingerprint();
        changed_input.producer.space_fingerprint = changed_input.space.fingerprint();
        changed_input.validate().unwrap();
        assert_eq!(
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &reference,
                &changed_input,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            ),
            Err(ProducerCompatibilityErrorV1::SpaceMismatch)
        );

        let mut changed_tokenizer = candidate.clone();
        changed_tokenizer.space.tokenizer_fingerprint = "7".repeat(64);
        changed_tokenizer.producer.space_fingerprint = changed_tokenizer.space.fingerprint();
        changed_tokenizer.validate().unwrap();
        assert_eq!(
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &reference,
                &changed_tokenizer,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            ),
            Err(ProducerCompatibilityErrorV1::SpaceMismatch)
        );

        let mut invalid_reference = reference.clone();
        invalid_reference.storage.dimension = 0;
        assert_eq!(
            invalid_reference.verify_exact_producer_with(&reference),
            Err(ProducerCompatibilityErrorV1::InvalidReferenceIdentity)
        );
        let mut invalid_candidate = candidate;
        invalid_candidate.storage.dimension = 0;
        assert_eq!(
            reference.verify_exact_producer_with(&invalid_candidate),
            Err(ProducerCompatibilityErrorV1::InvalidCandidateIdentity)
        );
    }

    #[test]
    fn foreign_certificate_and_verified_fixture_fail_closed_on_untrusted_bytes() {
        let (reference, candidate, fixture) = sample_foreign_producer_pair();
        let policy_fingerprint = "9".repeat(64);
        let certificate =
            ForeignProducerConformanceCertificateV1::new_untrusted_receipt_from_verified_pair(
                &reference,
                &candidate,
                &fixture,
                &policy_fingerprint,
                7,
                100,
                200,
            )
            .unwrap();

        let mut unknown = serde_json::to_value(&certificate).unwrap();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("unexpected".to_owned(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<ForeignProducerConformanceCertificateV1>(unknown).is_err()
        );

        let raw = serde_json::to_string(&certificate).unwrap();
        let duplicate_schema = raw.replacen('{', "{\"schema_version\":1,", 1);
        assert!(
            serde_json::from_str::<ForeignProducerConformanceCertificateV1>(&duplicate_schema)
                .is_err()
        );

        let texts = ["query: alpha", "document: beta"];
        let reference_vectors = vec![
            vec![0.0, -0.0, 1.0, -1.0, 0.25, 0.5, 0.75, 1.25],
            vec![2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5],
        ];
        let mut changed_bits = reference_vectors.clone();
        changed_bits[0][0] = -0.0;
        assert_eq!(
            VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(
                &texts,
                &reference_vectors,
                &changed_bits,
            ),
            Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch)
        );

        let nan_texts = ["NaN payload"];
        let reference_nan = vec![vec![f32::from_bits(0x7fc0_0041)]];
        let candidate_nan = vec![vec![f32::from_bits(0x7fc0_0042)]];
        assert_eq!(
            VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(
                &nan_texts,
                &reference_nan,
                &candidate_nan,
            ),
            Err(ProducerCompatibilityErrorV1::GoldenVectorMismatch),
            "NaN payload bits are part of exact producer conformance"
        );
        let malformed = vec![vec![1.0], vec![]];
        assert_eq!(
            VerifiedGoldenConformanceManifestV1::from_exact_pair_f32(
                &texts,
                &reference_vectors,
                &malformed,
            ),
            Err(ProducerCompatibilityErrorV1::GoldenFixtureInvalid)
        );
    }

    #[test]
    fn every_identity_field_participates_in_the_bundle_fingerprint() {
        let base = EmbeddingIdentityBundleV1::explicit_test_model("mutation-matrix", 256);
        let base_fingerprint = base.fingerprint();
        macro_rules! changed {
            ($label:literal, $mutate:expr) => {{
                let mut candidate = base.clone();
                $mutate(&mut candidate);
                assert_ne!(base_fingerprint, candidate.fingerprint(), $label);
            }};
        }

        changed!("space schema", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .schema_version +=
            1);
        changed!("logical model", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .logical_model_id
            .push('x'));
        changed!("immutable revision", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .immutable_revision
            .push('x'));
        changed!("space kind", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .kind =
            EmbeddingSpaceKindV1::Semantic);
        changed!(
            "manifest fingerprint",
            |v: &mut EmbeddingIdentityBundleV1| v.space.artifact_manifest_fingerprint =
                "a".repeat(64)
        );
        changed!("artifact", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .artifacts
            .push(EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "b".repeat(64),
                size: 7,
            }));
        changed!("tokenizer", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .tokenizer_fingerprint =
            "a".repeat(64));
        changed!("vocabulary", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .vocabulary_fingerprint =
            "b".repeat(64));
        changed!("model config", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .model_config_fingerprint =
            "c".repeat(64));
        changed!(
            "model preprocessing",
            |v: &mut EmbeddingIdentityBundleV1| v.space.model_preprocessing.push('x')
        );
        changed!("sequence policy", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .sequence_policy
            .push('x'));
        changed!("query instruction", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .query_instruction
            .push('x'));
        changed!(
            "document instruction",
            |v: &mut EmbeddingIdentityBundleV1| v.space.document_instruction.push('x')
        );
        changed!("pooling", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .pooling
            .push('x'));
        changed!("normalization", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .output_normalization
            .push('x'));
        changed!("space dimension", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .dimension +=
            1);
        changed!("input binding", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .input_contract_fingerprint =
            "d".repeat(64));
        changed!("hash algorithm", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .hash_control
            .as_mut()
            .unwrap()
            .algorithm
            .push('x'));
        changed!("hash revision", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .hash_control
            .as_mut()
            .unwrap()
            .algorithm_revision
            .push('x'));
        changed!("hash seed", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .hash_control
            .as_mut()
            .unwrap()
            .seed +=
            1);
        changed!("hash features", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .hash_control
            .as_mut()
            .unwrap()
            .feature_rules
            .push('x'));
        changed!("hash tokenization", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .hash_control
            .as_mut()
            .unwrap()
            .tokenization_rules
            .push('x'));
        changed!("hash signing", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .hash_control
            .as_mut()
            .unwrap()
            .signing_rules
            .push('x'));
        changed!("hash normalization", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .hash_control
            .as_mut()
            .unwrap()
            .normalization_rules
            .push('x'));
        changed!("projection", |v: &mut EmbeddingIdentityBundleV1| v
            .space
            .projection =
            Some(EmbeddingProjectionV1 {
                parent_space_fingerprint: "e".repeat(64),
                source_dimension: 512,
                output_dimension: 256,
                projection_rule: "prefix".to_owned(),
                renormalization_rule: "l2".to_owned(),
            }));

        let artifact_base = sample_semantic_identity();
        let artifact_fingerprint = artifact_base.fingerprint();
        macro_rules! changed_artifact_field {
            ($label:literal, $mutate:expr) => {{
                let mut candidate = artifact_base.clone();
                $mutate(&mut candidate.space.artifacts[0]);
                assert_ne!(artifact_fingerprint, candidate.fingerprint(), $label);
            }};
        }
        changed_artifact_field!(
            "artifact role",
            |artifact: &mut EmbeddingArtifactIdentityV1| {
                artifact.role.push('x');
            }
        );
        changed_artifact_field!(
            "artifact digest",
            |artifact: &mut EmbeddingArtifactIdentityV1| {
                artifact.sha256 = "a".repeat(64);
            }
        );
        changed_artifact_field!(
            "artifact size",
            |artifact: &mut EmbeddingArtifactIdentityV1| {
                artifact.size += 1;
            }
        );

        let projection_base = sample_semantic_identity()
            .derive_projection(4, "prefix-truncate-v1", "l2-f32-after-prefix-v1")
            .unwrap();
        let projection_fingerprint = projection_base.fingerprint();
        macro_rules! changed_projection_field {
            ($label:literal, $mutate:expr) => {{
                let mut candidate = projection_base.clone();
                $mutate(candidate.space.projection.as_mut().unwrap());
                assert_ne!(projection_fingerprint, candidate.fingerprint(), $label);
            }};
        }
        changed_projection_field!(
            "projection parent",
            |projection: &mut EmbeddingProjectionV1| {
                projection.parent_space_fingerprint = "a".repeat(64);
            }
        );
        changed_projection_field!(
            "projection source dimension",
            |projection: &mut EmbeddingProjectionV1| {
                projection.source_dimension += 1;
            }
        );
        changed_projection_field!(
            "projection output dimension",
            |projection: &mut EmbeddingProjectionV1| {
                projection.output_dimension += 1;
            }
        );
        changed_projection_field!(
            "projection rule",
            |projection: &mut EmbeddingProjectionV1| {
                projection.projection_rule.push('x');
            }
        );
        changed_projection_field!(
            "projection renormalization",
            |projection: &mut EmbeddingProjectionV1| {
                projection.renormalization_rule.push('x');
            }
        );

        changed!("producer schema", |v: &mut EmbeddingIdentityBundleV1| v
            .producer
            .schema_version +=
            1);
        changed!("backend", |v: &mut EmbeddingIdentityBundleV1| v
            .producer
            .backend
            .push('x'));
        changed!("implementation", |v: &mut EmbeddingIdentityBundleV1| v
            .producer
            .implementation_revision
            .push('x'));
        changed!("protocol", |v: &mut EmbeddingIdentityBundleV1| v
            .producer
            .protocol_revision
            .push('x'));
        changed!("numeric profile", |v: &mut EmbeddingIdentityBundleV1| v
            .producer
            .numeric_profile
            .push('x'));
        changed!(
            "producer provenance manifest",
            |v: &mut EmbeddingIdentityBundleV1| v.producer.provenance_manifest_fingerprint =
                "c".repeat(64)
        );
        changed!(
            "producer space binding",
            |v: &mut EmbeddingIdentityBundleV1| v.producer.space_fingerprint = "f".repeat(64)
        );
        changed!("golden corpus", |v: &mut EmbeddingIdentityBundleV1| v
            .producer
            .golden_vectors
            .corpus_sha256 =
            "a".repeat(64));
        changed!("golden vectors", |v: &mut EmbeddingIdentityBundleV1| v
            .producer
            .golden_vectors
            .vectors_sha256 =
            "b".repeat(64));
        changed!("golden count", |v: &mut EmbeddingIdentityBundleV1| v
            .producer
            .golden_vectors
            .vector_count +=
            1);
        changed!("golden dimension", |v: &mut EmbeddingIdentityBundleV1| v
            .producer
            .golden_vectors
            .dimension +=
            1);

        changed!("input schema", |v: &mut EmbeddingIdentityBundleV1| v
            .input
            .schema_version +=
            1);
        changed!("canonicalization", |v: &mut EmbeddingIdentityBundleV1| v
            .input
            .canonicalization
            .push('x'));
        changed!("content selection", |v: &mut EmbeddingIdentityBundleV1| v
            .input
            .content_selection
            .push('x'));
        changed!("chunking", |v: &mut EmbeddingIdentityBundleV1| v
            .input
            .chunking
            .push('x'));
        changed!(
            "outer query instruction",
            |v: &mut EmbeddingIdentityBundleV1| v.input.query_instruction.push('x')
        );
        changed!(
            "outer document instruction",
            |v: &mut EmbeddingIdentityBundleV1| v.input.document_instruction.push('x')
        );
        changed!(
            "document id semantics",
            |v: &mut EmbeddingIdentityBundleV1| v.input.doc_id_semantics.push('x')
        );

        changed!("storage schema", |v: &mut EmbeddingIdentityBundleV1| v
            .storage
            .schema_version +=
            1);
        changed!("storage format", |v: &mut EmbeddingIdentityBundleV1| v
            .storage
            .format
            .push('x'));
        changed!("quantization", |v: &mut EmbeddingIdentityBundleV1| v
            .storage
            .quantization =
            QuantizationFormat::Int8);
        changed!("endianness", |v: &mut EmbeddingIdentityBundleV1| v
            .storage
            .endianness
            .push('x'));
        changed!(
            "storage normalization",
            |v: &mut EmbeddingIdentityBundleV1| v.storage.vector_normalization.push('x')
        );
        changed!("storage dimension", |v: &mut EmbeddingIdentityBundleV1| {
            v.storage.dimension += 1;
        });
    }

    #[test]
    fn artifact_role_order_is_canonical_but_duplicate_roles_fail() {
        let left = sample_semantic_identity();
        let mut right = left.clone();
        right.space.artifacts.reverse();
        right.producer.space_fingerprint = right.space.fingerprint();
        right.validate().unwrap();
        assert_eq!(left.space.fingerprint(), right.space.fingerprint());
        assert_eq!(left.fingerprint(), right.fingerprint());
        assert_eq!(
            left.verify_exact_producer_with(&right).unwrap().kind(),
            ProducerCompatibilityKindV1::Exact
        );

        let mut duplicate = left;
        duplicate
            .space
            .artifacts
            .push(duplicate.space.artifacts[0].clone());
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn hash_control_requires_its_canonical_profile_fingerprint() {
        let mut identity =
            EmbeddingIdentityBundleV1::explicit_test_model("hash-profile-binding", 32);
        identity.space.artifact_manifest_fingerprint = "0".repeat(64);
        identity.producer.space_fingerprint = identity.space.fingerprint();
        assert!(identity.validate().is_err());

        let mut identity =
            EmbeddingIdentityBundleV1::explicit_test_model("hash-producer-binding", 32);
        identity.producer.provenance_manifest_fingerprint = "0".repeat(64);
        assert!(identity.validate().is_err());
    }

    #[test]
    fn storage_identity_rejects_cross_field_encoding_contradictions() {
        let mut quantized_memory =
            EmbeddingIdentityBundleV1::explicit_test_model("quantized-memory", 32);
        quantized_memory.storage.quantization = QuantizationFormat::F16;
        assert!(quantized_memory.validate().is_err());

        let mut byte_ordered_memory =
            EmbeddingIdentityBundleV1::explicit_test_model("byte-ordered-memory", 32);
        byte_ordered_memory.storage.endianness = "little-endian".to_owned();
        assert!(byte_ordered_memory.validate().is_err());

        let mut native_fsvi = EmbeddingIdentityBundleV1::explicit_test_model("native-fsvi", 32);
        native_fsvi.storage.format = "fsvi-v2".to_owned();
        native_fsvi.storage.endianness = "native-f32-values".to_owned();
        assert!(native_fsvi.validate().is_err());
    }

    #[test]
    fn frozen_bundle_rejects_noncanonical_bytes_digest_drift_and_unknown_schema() {
        let identity = sample_semantic_identity();
        let frozen = identity.freeze().unwrap();
        frozen.validate().unwrap();

        let mut bad_bytes = frozen.clone();
        bad_bytes.canonical_bytes.push(0);
        assert!(bad_bytes.validate().is_err());

        let mut bad_digest = frozen.clone();
        bad_digest.fingerprint = "0".repeat(64);
        assert!(bad_digest.validate().is_err());

        let mut injected_digest = frozen.clone();
        injected_digest.fingerprint = "digest\nforged-log-line".to_owned();
        let error = injected_digest.validate().unwrap_err();
        assert!(error.to_string().contains("redacted-invalid-sha256"));
        assert!(!error.to_string().contains("forged-log-line"));

        let mut unknown_space = identity.clone();
        unknown_space.space.schema_version += 1;
        assert!(unknown_space.validate().is_err());
        let mut unknown_producer = identity.clone();
        unknown_producer.producer.schema_version += 1;
        assert!(unknown_producer.validate().is_err());
        let mut unknown_input = identity.clone();
        unknown_input.input.schema_version += 1;
        assert!(unknown_input.validate().is_err());
        let mut unknown_storage = identity;
        unknown_storage.storage.schema_version += 1;
        assert!(unknown_storage.validate().is_err());

        let mut normalization_drift = sample_semantic_identity();
        normalization_drift
            .storage
            .vector_normalization
            .push_str("-drift");
        assert!(normalization_drift.validate().is_err());

        let mut oversized_instruction = sample_semantic_identity();
        oversized_instruction.space.query_instruction = "x".repeat(MAX_IDENTITY_FIELD_BYTES + 1);
        assert!(oversized_instruction.validate().is_err());

        let mut unknown_field = serde_json::to_value(sample_semantic_identity()).unwrap();
        unknown_field["space"]["future_unregistered_field"] = serde_json::json!(true);
        assert!(
            serde_json::from_value::<EmbeddingIdentityBundleV1>(unknown_field).is_err(),
            "versioned identities must reject unknown fields instead of silently dropping them"
        );

        let mut log_injection =
            EmbeddingIdentityBundleV1::explicit_test_model("control-character", 32);
        log_injection.space.logical_model_id = "safe\nforged-log-line".to_owned();
        let error = log_injection.validate().unwrap_err();
        assert!(error.to_string().contains("control characters"));
        assert!(!error.to_string().contains("forged-log-line"));

        let mut digest_injection =
            EmbeddingIdentityBundleV1::explicit_test_model("digest-control-character", 32);
        digest_injection.space.tokenizer_fingerprint = "digest\nforged-log-line".to_owned();
        let error = digest_injection.validate().unwrap_err();
        assert!(error.to_string().contains("redacted-invalid-sha256"));
        assert!(!error.to_string().contains("forged-log-line"));
    }

    #[test]
    fn generation_manifest_persists_and_revalidates_identity_bytes_and_digest() {
        let manifest = valid_manifest();
        let encoded = serde_json::to_value(&manifest).unwrap();
        let persisted = &encoded["embedders"]["fast"];
        assert!(
            persisted["canonical_bytes"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
        );
        assert_eq!(persisted["fingerprint"].as_str().map(str::len), Some(64));

        let mut tampered = manifest;
        tampered
            .embedders
            .get_mut("fast")
            .unwrap()
            .canonical_bytes
            .push(0);
        assert!(!validate_manifest(&tampered).is_valid());
    }

    #[test]
    fn mrl_identity_is_structurally_derived_and_cross_bound() {
        let parent = sample_semantic_identity();
        let child = parent
            .derive_projection(4, "prefix-truncate-v1", "l2-f32-after-prefix-v1")
            .unwrap();
        child.validate().unwrap();
        let projection = child.space.projection.as_ref().unwrap();
        assert_eq!(
            projection.parent_space_fingerprint,
            parent.space.fingerprint()
        );
        assert_eq!(projection.source_dimension, 8);
        assert_eq!(projection.output_dimension, 4);
        assert_eq!(child.storage.dimension, 4);
        assert_eq!(child.space.output_normalization, "l2-f32-after-prefix-v1");
        assert_eq!(child.storage.vector_normalization, "l2-f32-after-prefix-v1");
        assert_eq!(child.producer.space_fingerprint, child.space.fingerprint());
        assert!(
            child
                .producer
                .implementation_revision
                .starts_with("frankensearch-identity-projection-wrapper-v1:parent=")
        );
        assert_eq!(
            child.producer.protocol_revision,
            "deterministic-identity-projection-v1"
        );
        assert_ne!(parent.fingerprint(), child.fingerprint());
    }

    fn sample_vector_artifact(path: &str, count: u64) -> VectorArtifact {
        VectorArtifact {
            path: path.into(),
            size_bytes: 1024,
            checksum: "deadbeef".into(),
            vector_count: count,
            dimension: 256,
            embedder_tier: EmbedderTierTag::Fast,
        }
    }

    fn sample_lexical_artifact(path: &str, count: u64) -> LexicalArtifact {
        LexicalArtifact {
            path: path.into(),
            size_bytes: 2048,
            checksum: "cafebabe".into(),
            document_count: count,
        }
    }

    fn valid_manifest() -> GenerationManifest {
        let mut embedders = BTreeMap::new();
        embedders.insert("fast".into(), sample_embedder());

        let mut manifest = GenerationManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            generation_id: "gen-001".into(),
            manifest_hash: String::new(),
            commit_range: CommitRange { low: 1, high: 100 },
            build_started_at: 1_700_000_000_000,
            build_completed_at: 1_700_000_060_000,
            embedders,
            vector_artifacts: vec![sample_vector_artifact("vectors/shard_0.fsvi", 100)],
            lexical_artifacts: vec![sample_lexical_artifact("lexical/segment_0", 100)],
            repair_descriptors: vec![RepairDescriptor {
                protected_artifact: "vectors/shard_0.fsvi".into(),
                sidecar_path: "vectors/shard_0.fsvi.fec".into(),
                source_symbols: 64,
                repair_symbols: 13,
                overhead_ratio: 0.2,
            }],
            activation_invariants: vec![
                ActivationInvariant {
                    id: "all_artifacts".into(),
                    description: "All artifacts verified".into(),
                    kind: InvariantKind::AllArtifactsVerified,
                },
                ActivationInvariant {
                    id: "embedder_match".into(),
                    description: "Embedder revision matches runtime".into(),
                    kind: InvariantKind::EmbedderRevisionMatch,
                },
            ],
            total_documents: 100,
            metadata: BTreeMap::new(),
        };
        manifest.manifest_hash = compute_manifest_hash(&manifest).expect("hash");
        manifest
    }

    fn refresh_manifest_hash(manifest: &mut GenerationManifest) {
        manifest.manifest_hash = compute_manifest_hash(manifest).expect("hash");
    }

    #[test]
    fn valid_manifest_passes() {
        let m = valid_manifest();
        let r = validate_manifest(&m);
        assert!(r.is_valid(), "findings: {:#?}", r.findings);
        assert!(r.errors().is_empty());
    }

    #[test]
    fn legacy_schema_version_is_error() {
        let mut m = valid_manifest();
        m.schema_version = MANIFEST_SCHEMA_VERSION - 1;
        refresh_manifest_hash(&mut m);
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "schema_version"));
    }

    #[test]
    fn future_schema_version_is_error() {
        let mut m = valid_manifest();
        m.schema_version = MANIFEST_SCHEMA_VERSION + 1;
        refresh_manifest_hash(&mut m);
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "schema_version"));
    }

    #[test]
    fn empty_generation_id_is_error() {
        let mut m = valid_manifest();
        m.generation_id = String::new();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "generation_id"));
    }

    #[test]
    fn empty_manifest_hash_is_error() {
        let mut m = valid_manifest();
        m.manifest_hash.clear();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "manifest_hash"));
    }

    #[test]
    fn malformed_manifest_hash_is_error() {
        let mut m = valid_manifest();
        m.manifest_hash = "not-a-sha256".into();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "manifest_hash"));
    }

    #[test]
    fn mismatched_manifest_hash_is_error() {
        let mut m = valid_manifest();
        m.manifest_hash = "0".repeat(64);
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "manifest_hash"
                    && f.message.contains("does not match canonical"))
        );
    }

    #[test]
    fn manifest_hash_match_is_case_insensitive() {
        let mut m = valid_manifest();
        m.manifest_hash = m.manifest_hash.to_uppercase();
        let r = validate_manifest(&m);
        assert!(r.is_valid());
    }

    #[test]
    fn invalid_commit_range_is_error() {
        let mut m = valid_manifest();
        m.commit_range = CommitRange { low: 50, high: 10 };
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "commit_range"));
    }

    #[test]
    fn zero_timestamps_are_errors() {
        let mut m = valid_manifest();
        m.build_started_at = 0;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "build_started_at"));
    }

    #[test]
    fn completed_before_started_is_error() {
        let mut m = valid_manifest();
        m.build_completed_at = m.build_started_at - 1;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "build_timestamps"));
    }

    #[test]
    fn no_embedders_is_error() {
        let mut m = valid_manifest();
        m.embedders.clear();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "embedders"));
    }

    #[test]
    fn embedder_empty_fields_are_errors() {
        let mut m = valid_manifest();
        let mut invalid = EmbedderRevision::explicit_test_model("bad", 256);
        invalid.identity.space.logical_model_id.clear();
        invalid.identity.space.dimension = 0;
        m.embedders.insert("bad".into(), invalid);
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        let errors = r.errors();
        assert!(errors.iter().any(|f| f.check == "embedder_identity"));
    }

    #[test]
    fn duplicate_vector_artifact_paths_is_error() {
        let mut m = valid_manifest();
        m.vector_artifacts
            .push(sample_vector_artifact("vectors/shard_0.fsvi", 100));
        m.total_documents = 200;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "vector_artifact_duplicate")
        );
    }

    #[test]
    fn empty_artifact_path_is_error() {
        let mut m = valid_manifest();
        m.vector_artifacts.push(sample_vector_artifact("", 10));
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "vector_artifact_path"));
    }

    #[test]
    fn empty_artifact_checksum_is_error() {
        let mut m = valid_manifest();
        m.vector_artifacts[0].checksum = String::new();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "vector_artifact_checksum")
        );
    }

    #[test]
    fn repair_descriptor_unknown_artifact_is_error() {
        let mut m = valid_manifest();
        m.repair_descriptors.push(RepairDescriptor {
            protected_artifact: "nonexistent.fsvi".into(),
            sidecar_path: "nonexistent.fsvi.fec".into(),
            source_symbols: 10,
            repair_symbols: 2,
            overhead_ratio: 0.2,
        });
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "repair_descriptor_target")
        );
    }

    #[test]
    fn repair_descriptor_zero_source_symbols_is_error() {
        let mut m = valid_manifest();
        m.repair_descriptors[0].source_symbols = 0;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "repair_descriptor_symbols")
        );
    }

    #[test]
    fn extreme_repair_overhead_is_warning() {
        let mut m = valid_manifest();
        m.repair_descriptors[0].overhead_ratio = 15.0;
        refresh_manifest_hash(&mut m);
        let r = validate_manifest(&m);
        assert!(r.is_valid());
        assert!(
            r.warnings()
                .iter()
                .any(|f| f.check == "repair_descriptor_overhead")
        );
    }

    #[test]
    fn duplicate_invariant_id_is_error() {
        let mut m = valid_manifest();
        m.activation_invariants
            .push(m.activation_invariants[0].clone());
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "invariant_duplicate"));
    }

    #[test]
    fn zero_total_documents_with_artifacts_is_error() {
        let mut m = valid_manifest();
        m.total_documents = 0;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "total_documents"));
    }

    #[test]
    fn lexical_count_mismatch_is_warning() {
        let mut m = valid_manifest();
        m.lexical_artifacts[0].document_count = 50; // != total_documents (100)
        refresh_manifest_hash(&mut m);
        let r = validate_manifest(&m);
        assert!(r.is_valid());
        assert!(
            r.warnings()
                .iter()
                .any(|f| f.check == "lexical_document_count")
        );
    }

    #[test]
    fn two_tier_vector_count_accepted() {
        let mut m = valid_manifest();
        // Fast tier: 100 vectors + Quality tier: 100 vectors = 200 total, 100 docs
        m.vector_artifacts = vec![
            sample_vector_artifact("vectors/fast.fsvi", 100),
            sample_vector_artifact("vectors/quality.fsvi", 100),
        ];
        m.vector_artifacts[1].embedder_tier = EmbedderTierTag::Quality;
        // Update repair descriptor to reference the new fast tier artifact.
        m.repair_descriptors[0].protected_artifact = "vectors/fast.fsvi".into();
        refresh_manifest_hash(&mut m);
        let r = validate_manifest(&m);
        assert!(r.is_valid(), "findings: {:#?}", r.findings);
    }

    #[test]
    fn serde_roundtrip() {
        let m = valid_manifest();
        let json = serde_json::to_string_pretty(&m).expect("serialize");
        let deserialized: GenerationManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(m, deserialized);
    }

    #[test]
    fn commit_range_len_and_empty() {
        let range = CommitRange { low: 5, high: 10 };
        assert_eq!(range.len(), 6);
        assert!(!range.is_empty());

        let empty = CommitRange { low: 10, high: 5 };
        assert!(empty.is_empty());
    }

    #[test]
    fn single_commit_range() {
        let range = CommitRange { low: 42, high: 42 };
        assert_eq!(range.len(), 1);
        assert!(!range.is_empty());
    }

    #[test]
    fn require_valid_passes_for_valid_manifest() {
        let m = valid_manifest();
        let r = validate_manifest(&m);
        assert!(require_valid(&r).is_ok());
    }

    #[test]
    fn require_valid_fails_for_invalid_manifest() {
        let mut m = valid_manifest();
        m.generation_id = String::new();
        let r = validate_manifest(&m);
        let err = require_valid(&r).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn empty_manifest_collects_multiple_errors() {
        let m = GenerationManifest {
            schema_version: 0,
            generation_id: String::new(),
            manifest_hash: String::new(),
            commit_range: CommitRange { low: 10, high: 5 },
            build_started_at: 0,
            build_completed_at: 0,
            embedders: BTreeMap::new(),
            vector_artifacts: vec![],
            lexical_artifacts: vec![],
            repair_descriptors: vec![],
            activation_invariants: vec![],
            total_documents: 0,
            metadata: BTreeMap::new(),
        };
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        // Should find at least: schema_version, generation_id, commit_range,
        // build_started_at, build_completed_at, embedders
        assert!(r.errors().len() >= 5, "found {} errors", r.errors().len());
    }

    #[test]
    fn metadata_is_preserved() {
        let mut m = valid_manifest();
        m.metadata.insert("build_host".into(), "node-7".into());
        m.metadata.insert("deployment".into(), "production".into());
        let json = serde_json::to_string(&m).expect("serialize");
        let deserialized: GenerationManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.metadata.get("build_host").unwrap(), "node-7");
    }

    #[test]
    fn invariant_kinds_serialize() {
        let kinds = vec![
            InvariantKind::AllArtifactsVerified,
            InvariantKind::EmbedderRevisionMatch,
            InvariantKind::VectorCountConsistency {
                expected_total: 500,
            },
            InvariantKind::CommitContinuity { previous_high: 99 },
            InvariantKind::Custom {
                check_name: "custom_check".into(),
            },
        ];
        for kind in &kinds {
            let json = serde_json::to_string(kind).expect("serialize");
            let back: InvariantKind = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(kind, &back);
        }
    }

    #[test]
    fn quantization_format_serialize() {
        for fmt in &[
            QuantizationFormat::F32,
            QuantizationFormat::F16,
            QuantizationFormat::Int8,
            QuantizationFormat::Int4,
        ] {
            let json = serde_json::to_string(fmt).expect("serialize");
            let back: QuantizationFormat = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(fmt, &back);
        }
    }
}
