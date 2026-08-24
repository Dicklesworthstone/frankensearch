//! ArtifactStore v4 F2: supervisor-issued execution and completion
//! authentication (bd-artifactstore-v4-f2-supervisor-auth-px8rm).
//!
//! Implements the frozen F0 contract (`docs/artifactstore-v4-threat-model.md`,
//! `schemas/artifactstore-v4-f0.schema.json`): an external supervisor — never
//! the measured child — owns a single-use 128-bit nonce, launches the exact
//! admitted executable, collects terminal status, bounded logs, artifact
//! digests, and lifecycle receipts through its own descriptors, and signs an
//! outcome-neutral completion with Ed25519. Every terminal path (success,
//! nonzero exit, signal, cancellation, timeout, output overflow, exec
//! failure, supervisor interruption) yields either one signed completion
//! bound to its issued nonce or an explicit typed absence that is never
//! promotable. The child may emit data but holds no signing key and cannot
//! choose admission state; authentication of a completion never implies
//! `Admitted`, `Pass`, or `Qualified` (independent axes per F0).
//!
//! Cryptography is exactly the frozen contract's: SHA-256 content hashes and
//! Ed25519 receipt signatures, with immutable `key_id`s resolved through a
//! versioned trust-root set carrying validity intervals and revocation. A
//! retired key may verify receipts issued before retirement but may not issue
//! new ones. Canonical bytes are UTF-8 JSON with canonical key order, exact
//! integers, lower-hex byte strings, signed nanosecond timestamps, required
//! fields present even when null, no trailing newline; decoding reserializes
//! and demands byte-for-byte equality, and unknown fields fail closed.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::GauntletError;

/// Frozen object-identity domains (F0: UTF-8 domain bytes + one NUL byte).
pub const ARTIFACTSTORE_V4_EXECUTION_DOMAIN: &str = "frankensearch.artifactstore.v4.execution";
/// Completion-object identity domain.
pub const ARTIFACTSTORE_V4_COMPLETION_DOMAIN: &str = "frankensearch.artifactstore.v4.completion";
/// Signing domain for detached receipts over object identities.
pub const ARTIFACTSTORE_V4_RECEIPT_DOMAIN: &str = "frankensearch.artifactstore.v4.receipt";
/// Signing domain for typed externally witnessed completion absences.
pub const ARTIFACTSTORE_V4_ABSENCE_DOMAIN: &str = "frankensearch.artifactstore.v4.absence";

/// Wire schema version for every F2 object in this module.
pub const SUPERVISOR_AUTH_SCHEMA_VERSION: u32 = 1;

/// F0: nonce lifetime is at most 15 minutes.
pub const MAX_NONCE_LIFETIME_NS: i64 = 15 * 60 * 1_000_000_000;

/// Frozen principal-role labels (F0 schema `principal_role`).
pub const EXECUTION_SUPERVISOR_ROLE: &str = "execution_supervisor";
/// Completion receipts carry this frozen role label.
pub const COMPLETION_ISSUER_ROLE: &str = "completion_issuer";

const MAX_BOUNDED_LABEL_BYTES: usize = 96;
const MAX_ARTIFACT_INDEX_ENTRIES: usize = 64;
const MAX_REASON_CODE_BYTES: usize = 96;

fn invalid(reason: String) -> GauntletError {
    GauntletError::InvalidPreparedArtifact { reason }
}

/// Canonicalize a value to the F0 wire form: UTF-8 JSON, canonical
/// (lexicographic) key order via `serde_json`'s `BTreeMap`-backed objects,
/// exact integers, no trailing newline.
///
/// Every F2 struct keeps to the JSON subset where `serde_json`'s output is
/// RFC 8785-canonical: integers, booleans, nulls, ASCII strings, arrays, and
/// sorted maps. ASCII is enforced on every free-text field at construction,
/// so escaping differences cannot arise.
fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, GauntletError> {
    let json = serde_json::to_value(value)
        .map_err(|error| invalid(format!("cannot canonicalize v4 object: {error}")))?;
    reject_non_canonical_value(&json)?;
    serde_json::to_vec(&json)
        .map_err(|error| invalid(format!("cannot serialize canonical v4 bytes: {error}")))
}

fn reject_non_canonical_value(value: &serde_json::Value) -> Result<(), GauntletError> {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) => Ok(()),
        serde_json::Value::Number(number) => {
            if number.is_f64() {
                return Err(invalid(
                    "v4 canonical form forbids non-integer numbers".to_owned(),
                ));
            }
            Ok(())
        }
        serde_json::Value::String(text) => {
            if !text.is_ascii() {
                return Err(invalid(
                    "v4 canonical form requires ASCII strings".to_owned(),
                ));
            }
            Ok(())
        }
        serde_json::Value::Array(items) => {
            for item in items {
                reject_non_canonical_value(item)?;
            }
            Ok(())
        }
        serde_json::Value::Object(map) => {
            for (key, item) in map {
                if !key.is_ascii() {
                    return Err(invalid("v4 canonical form requires ASCII keys".to_owned()));
                }
                reject_non_canonical_value(item)?;
            }
            Ok(())
        }
    }
}

/// Decode canonical bytes fail-closed.
///
/// Unknown fields are rejected by the struct contract, and the decoded value
/// must reserialize to the exact input bytes (F0: implementations must
/// reserialize after decoding and require byte-for-byte equality).
///
/// # Errors
///
/// Returns [`GauntletError`] on malformed JSON, unknown fields, or bytes
/// that are not the canonical serialization of the decoded value.
pub fn decode_canonical<T: Serialize + for<'de> Deserialize<'de>>(
    bytes: &[u8],
) -> Result<T, GauntletError> {
    let value: T = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("v4 canonical decode failed closed: {error}")))?;
    let reserialized = canonical_json_bytes(&value)?;
    if reserialized != bytes {
        return Err(invalid(
            "v4 bytes are not canonical: reserialization does not reproduce them".to_owned(),
        ));
    }
    Ok(value)
}

/// `SHA-256(domain || NUL || canonical_bytes)` as lower hex (F0 identity).
fn domain_identity(domain: &str, canonical_bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0_u8]);
    hasher.update(canonical_bytes);
    lower_hex(&hasher.finalize())
}

fn lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn decode_hex_exact(text: &str, expected_len: usize, what: &str) -> Result<Vec<u8>, GauntletError> {
    let bytes = text.as_bytes();
    if bytes.len() != expected_len * 2 {
        return Err(invalid(format!(
            "{what} is not canonical lower-hex of {expected_len} bytes"
        )));
    }
    let mut out = Vec::with_capacity(expected_len);
    let mut index = 0;
    while index + 1 < bytes.len() {
        let (Some(high), Some(low)) = (hex_nibble(bytes[index]), hex_nibble(bytes[index + 1]))
        else {
            return Err(invalid(format!(
                "{what} is not canonical lower-hex of {expected_len} bytes"
            )));
        };
        out.push((high << 4) | low);
        index += 2;
    }
    Ok(out)
}

fn validate_bounded_ascii_label(label: &str, what: &str) -> Result<(), GauntletError> {
    if label.is_empty() || label.len() > MAX_BOUNDED_LABEL_BYTES || !label.is_ascii() {
        return Err(invalid(format!(
            "{what} must be non-empty bounded ASCII (at most {MAX_BOUNDED_LABEL_BYTES} bytes)"
        )));
    }
    if label.contains('/') || label.contains('\\') {
        return Err(invalid(format!(
            "{what} must be a label, not a path (F0 privacy: logs and receipts carry no paths)"
        )));
    }
    Ok(())
}

fn validate_identity_hex(identity: &str, what: &str) -> Result<(), GauntletError> {
    decode_hex_exact(identity, 32, what).map(|_| ())
}

/// One key in the versioned trust-root set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustRootKey {
    /// Lower-hex Ed25519 verifying key (32 bytes).
    pub verifying_key_hex: String,
    /// Start of the key's issuing validity interval (ns since epoch).
    pub valid_from_ns: i64,
    /// End of the key's ISSUING validity: receipts signed after this instant
    /// are rejected, receipts signed before it still verify (F0: a retired
    /// key may verify receipts issued before retirement but may not issue
    /// new ones).
    pub retired_after_ns: Option<i64>,
    /// A revoked key verifies nothing, regardless of when it signed.
    pub revoked: bool,
    /// Frozen principal-role labels this key may sign as.
    pub roles: Vec<String>,
}

/// Versioned trust-root set for receipt verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupervisorTrustRoots {
    pub schema_version: u32,
    /// Immutable `key_id` (lower-hex SHA-256 of the verifying key bytes) to
    /// key record.
    pub keys: BTreeMap<String, TrustRootKey>,
}

impl SupervisorTrustRoots {
    fn resolve(
        &self,
        key_id: &str,
        role: &str,
        signed_at_ns: i64,
    ) -> Result<VerifyingKey, GauntletError> {
        if self.schema_version != SUPERVISOR_AUTH_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unknown trust-root schema version {} fails closed",
                self.schema_version
            )));
        }
        let key = self
            .keys
            .get(key_id)
            .ok_or_else(|| invalid(format!("unknown signer key_id {key_id} fails closed")))?;
        if key.revoked {
            return Err(invalid(format!("revoked key {key_id} verifies nothing")));
        }
        if !key.roles.iter().any(|granted| granted == role) {
            return Err(invalid(format!(
                "key {key_id} is not trusted for role {role}"
            )));
        }
        if signed_at_ns < key.valid_from_ns {
            return Err(invalid(format!("receipt predates key {key_id} validity")));
        }
        if let Some(retired_after_ns) = key.retired_after_ns
            && signed_at_ns > retired_after_ns
        {
            return Err(invalid(format!(
                "key {key_id} was retired before this receipt was issued"
            )));
        }
        let bytes = decode_hex_exact(&key.verifying_key_hex, 32, "trust-root verifying key")?;
        let array: [u8; 32] = bytes
            .try_into()
            .map_err(|_| invalid("trust-root verifying key must be 32 bytes".to_owned()))?;
        VerifyingKey::from_bytes(&array)
            .map_err(|error| invalid(format!("trust-root verifying key is invalid: {error}")))
    }

    /// The `key_id` also asserts the key bytes: recompute and require match,
    /// so a trust root cannot silently alias one key under another identity.
    fn validate_key_ids(&self) -> Result<(), GauntletError> {
        for (key_id, key) in &self.keys {
            let bytes = decode_hex_exact(&key.verifying_key_hex, 32, "trust-root verifying key")?;
            let expected = lower_hex(&Sha256::digest(&bytes));
            if *key_id != expected {
                return Err(invalid(format!(
                    "trust-root key_id {key_id} does not match its verifying key"
                )));
            }
        }
        Ok(())
    }
}

/// Detached Ed25519 receipt over one object identity, in a frozen role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedReceiptV4 {
    pub schema_version: u32,
    /// Frozen object kind: `execution` or `completion`.
    pub object_kind: String,
    /// The signed object's 32-byte identity, lower hex.
    pub object_identity_sha256: String,
    /// Frozen principal-role label.
    pub role: String,
    /// Immutable signer identity: lower-hex SHA-256 of the verifying key.
    pub key_id: String,
    pub signed_at_ns: i64,
    /// Ed25519 signature (64 bytes lower hex) over the receipt signing
    /// payload.
    pub signature_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptSigningPayload {
    schema_version: u32,
    object_kind: String,
    object_identity_sha256: String,
    role: String,
    key_id: String,
    signed_at_ns: i64,
}

impl SignedReceiptV4 {
    fn signing_message(&self) -> Result<Vec<u8>, GauntletError> {
        let payload = ReceiptSigningPayload {
            schema_version: self.schema_version,
            object_kind: self.object_kind.clone(),
            object_identity_sha256: self.object_identity_sha256.clone(),
            role: self.role.clone(),
            key_id: self.key_id.clone(),
            signed_at_ns: self.signed_at_ns,
        };
        let canonical = canonical_json_bytes(&payload)?;
        let mut message =
            Vec::with_capacity(ARTIFACTSTORE_V4_RECEIPT_DOMAIN.len() + 1 + canonical.len());
        message.extend_from_slice(ARTIFACTSTORE_V4_RECEIPT_DOMAIN.as_bytes());
        message.push(0);
        message.extend_from_slice(&canonical);
        Ok(message)
    }

    /// Verify this receipt against the trust roots, fail-closed on unknown
    /// schema, kind, role, signer, malformed encodings, or a bad signature.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] for every rejection class above; the reason
    /// names the failing check without echoing sensitive values.
    pub fn verify(
        &self,
        trust_roots: &SupervisorTrustRoots,
        expected_kind: &str,
        expected_role: &str,
        expected_object_identity: &str,
    ) -> Result<(), GauntletError> {
        trust_roots.validate_key_ids()?;
        if self.schema_version != SUPERVISOR_AUTH_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unknown receipt schema version {} fails closed",
                self.schema_version
            )));
        }
        if self.object_kind != expected_kind {
            return Err(invalid(format!(
                "receipt kind {} does not match expected {expected_kind}",
                self.object_kind
            )));
        }
        if self.role != expected_role {
            return Err(invalid(format!(
                "receipt role {} does not match expected {expected_role}",
                self.role
            )));
        }
        validate_identity_hex(&self.object_identity_sha256, "receipt object identity")?;
        if self.object_identity_sha256 != expected_object_identity {
            return Err(invalid(
                "receipt does not bind the expected object identity".to_owned(),
            ));
        }
        let verifying_key = trust_roots.resolve(&self.key_id, &self.role, self.signed_at_ns)?;
        let signature_bytes = decode_hex_exact(&self.signature_hex, 64, "receipt signature")?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|error| invalid(format!("receipt signature is malformed: {error}")))?;
        verifying_key
            .verify(&self.signing_message()?, &signature)
            .map_err(|_| invalid("receipt signature does not verify".to_owned()))
    }
}

/// The execution object (F0 chain position 3): exact build predecessor,
/// supervisor-issued nonce, bounded job identity, fixture/query-contract
/// identities, and the start receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactStoreV4ExecutionObject {
    pub schema_version: u32,
    /// Predecessor by complete 32-byte identity (never a path or label).
    pub predecessor_build_identity_sha256: String,
    /// Source identity the nonce was bound to (redundant with the build
    /// chain, bound again so nonce reuse across a source swap is typed).
    pub source_identity_sha256: String,
    /// Single-use non-zero 128-bit nonce, lower hex.
    pub nonce_hex: String,
    pub nonce_issued_at_ns: i64,
    pub nonce_expires_at_ns: i64,
    /// SHA-256 over the exact admitted argv, domain-separated by NULs.
    pub command_digest_sha256: String,
    /// SHA-256 of the admitted executable bytes the supervisor will launch.
    pub executable_sha256: String,
    /// Bounded ASCII labels only — F0 privacy forbids paths in receipts.
    pub job_identity: String,
    pub machine_profile: String,
    pub fixture_identity: String,
    pub query_contract_identity: String,
    /// Digest of the environment policy applied at launch (names only).
    pub environment_policy_sha256: String,
    pub started_at_ns: i64,
    /// Wall-clock run window the supervisor enforces, in nanoseconds.
    pub run_window_ns: i64,
}

impl ArtifactStoreV4ExecutionObject {
    /// Canonical bytes + F0 domain identity.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when the object violates its own structural
    /// contract or cannot serialize canonically.
    pub fn identity(&self) -> Result<(Vec<u8>, String), GauntletError> {
        self.validate()?;
        let canonical = canonical_json_bytes(self)?;
        let identity = domain_identity(ARTIFACTSTORE_V4_EXECUTION_DOMAIN, &canonical);
        Ok((canonical, identity))
    }

    fn validate(&self) -> Result<(), GauntletError> {
        if self.schema_version != SUPERVISOR_AUTH_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unknown execution schema version {} fails closed",
                self.schema_version
            )));
        }
        validate_identity_hex(
            &self.predecessor_build_identity_sha256,
            "execution predecessor build identity",
        )?;
        validate_identity_hex(&self.source_identity_sha256, "execution source identity")?;
        validate_identity_hex(&self.command_digest_sha256, "execution command digest")?;
        validate_identity_hex(&self.executable_sha256, "execution executable digest")?;
        validate_identity_hex(
            &self.environment_policy_sha256,
            "execution environment policy digest",
        )?;
        let nonce = decode_hex_exact(&self.nonce_hex, 16, "execution nonce")?;
        if nonce.iter().all(|byte| *byte == 0) {
            return Err(invalid("execution nonce must be non-zero".to_owned()));
        }
        let lifetime = self
            .nonce_expires_at_ns
            .checked_sub(self.nonce_issued_at_ns)
            .ok_or_else(|| invalid("execution nonce window overflows".to_owned()))?;
        if lifetime <= 0 || lifetime > MAX_NONCE_LIFETIME_NS {
            return Err(invalid(
                "execution nonce lifetime must be positive and at most 15 minutes".to_owned(),
            ));
        }
        if self.run_window_ns <= 0 {
            return Err(invalid("execution run window must be positive".to_owned()));
        }
        for (label, what) in [
            (&self.job_identity, "execution job identity"),
            (&self.machine_profile, "execution machine profile"),
            (&self.fixture_identity, "execution fixture identity"),
            (
                &self.query_contract_identity,
                "execution query-contract identity",
            ),
        ] {
            validate_bounded_ascii_label(label, what)?;
        }
        Ok(())
    }
}

/// Frozen terminal outcomes (F0 schema `terminal_outcome`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalOutcome {
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
    Interrupted,
    Unknown,
}

/// Bounded digest record for one collected artifact or log stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectedArtifactDigest {
    pub sha256: String,
    pub byte_len: u64,
    /// True when the supervisor's output bound truncated the stream; the
    /// digest covers the retained bytes only, and the completion says so.
    pub truncated: bool,
}

/// The completion object (F0 chain position 4).
///
/// Carries the exact execution predecessor, terminal outcome, end receipt,
/// bounded artifact index, and retention disposition. Outcome-neutral:
/// authentication of this object never implies admission, decision, or
/// release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactStoreV4CompletionObject {
    pub schema_version: u32,
    pub predecessor_execution_identity_sha256: String,
    /// The consumed nonce, restated so completion substitution across
    /// executions is a one-field mismatch.
    pub nonce_hex: String,
    pub terminal_outcome: TerminalOutcome,
    /// Bounded ASCII reason code (never a path, never raw output).
    pub outcome_reason_code: String,
    pub exit_code: Option<i64>,
    pub termination_signal: Option<i64>,
    pub completed_at_ns: i64,
    pub wall_clock_ns: i64,
    pub stdout: CollectedArtifactDigest,
    pub stderr: CollectedArtifactDigest,
    /// Bounded label -> digest index of collected artifacts.
    pub artifact_index: BTreeMap<String, CollectedArtifactDigest>,
    /// True when a process-group survivor was detected after the child was
    /// reaped; escapes are recorded, never silently erased.
    pub process_tree_escape_detected: bool,
    /// Bounded ASCII label naming the durability evidence set, plus its
    /// digest; `null` digest is a typed absence of durability evidence.
    pub durability_label: String,
    pub durability_sha256: Option<String>,
    pub retention_disposition: String,
}

impl ArtifactStoreV4CompletionObject {
    /// Canonical bytes + F0 domain identity.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when the object violates its own structural
    /// contract or cannot serialize canonically.
    pub fn identity(&self) -> Result<(Vec<u8>, String), GauntletError> {
        self.validate()?;
        let canonical = canonical_json_bytes(self)?;
        let identity = domain_identity(ARTIFACTSTORE_V4_COMPLETION_DOMAIN, &canonical);
        Ok((canonical, identity))
    }

    fn validate(&self) -> Result<(), GauntletError> {
        if self.schema_version != SUPERVISOR_AUTH_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unknown completion schema version {} fails closed",
                self.schema_version
            )));
        }
        validate_identity_hex(
            &self.predecessor_execution_identity_sha256,
            "completion predecessor execution identity",
        )?;
        decode_hex_exact(&self.nonce_hex, 16, "completion nonce")?;
        if self.outcome_reason_code.is_empty()
            || self.outcome_reason_code.len() > MAX_REASON_CODE_BYTES
            || !self.outcome_reason_code.is_ascii()
        {
            return Err(invalid(
                "completion reason code must be non-empty bounded ASCII".to_owned(),
            ));
        }
        if self.artifact_index.len() > MAX_ARTIFACT_INDEX_ENTRIES {
            return Err(invalid(format!(
                "completion artifact index exceeds the bound of {MAX_ARTIFACT_INDEX_ENTRIES}"
            )));
        }
        for (label, digest) in &self.artifact_index {
            validate_bounded_ascii_label(label, "completion artifact label")?;
            validate_identity_hex(&digest.sha256, "completion artifact digest")?;
        }
        validate_identity_hex(&self.stdout.sha256, "completion stdout digest")?;
        validate_identity_hex(&self.stderr.sha256, "completion stderr digest")?;
        validate_bounded_ascii_label(&self.durability_label, "completion durability label")?;
        if let Some(durability) = &self.durability_sha256 {
            validate_identity_hex(durability, "completion durability digest")?;
        }
        validate_bounded_ascii_label(
            &self.retention_disposition,
            "completion retention disposition",
        )?;
        if self.wall_clock_ns < 0 {
            return Err(invalid(
                "completion wall clock must be non-negative".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Typed, externally witnessed ABSENCE of a completion. Signed by the
/// supervisor so its existence is authentic, and never promotable: it names
/// no terminal outcome and can satisfy no admission policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionAbsenceV4 {
    pub schema_version: u32,
    pub execution_identity_sha256: String,
    pub nonce_hex: String,
    /// Bounded ASCII reason code (`supervisor_interrupted`,
    /// `missing_required_artifact`, ...).
    pub reason_code: String,
    pub witnessed_at_ns: i64,
    pub key_id: String,
    pub signature_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbsenceSigningPayload {
    schema_version: u32,
    execution_identity_sha256: String,
    nonce_hex: String,
    reason_code: String,
    witnessed_at_ns: i64,
    key_id: String,
}

impl CompletionAbsenceV4 {
    fn signing_message(&self) -> Result<Vec<u8>, GauntletError> {
        let payload = AbsenceSigningPayload {
            schema_version: self.schema_version,
            execution_identity_sha256: self.execution_identity_sha256.clone(),
            nonce_hex: self.nonce_hex.clone(),
            reason_code: self.reason_code.clone(),
            witnessed_at_ns: self.witnessed_at_ns,
            key_id: self.key_id.clone(),
        };
        let canonical = canonical_json_bytes(&payload)?;
        let mut message =
            Vec::with_capacity(ARTIFACTSTORE_V4_ABSENCE_DOMAIN.len() + 1 + canonical.len());
        message.extend_from_slice(ARTIFACTSTORE_V4_ABSENCE_DOMAIN.as_bytes());
        message.push(0);
        message.extend_from_slice(&canonical);
        Ok(message)
    }

    /// Verify the absence witness. An absence is never promotable; verifying
    /// it only proves the supervisor authentically recorded that no
    /// completion exists.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] on an unknown schema, signer, role grant,
    /// malformed encoding, or failed signature.
    pub fn verify(&self, trust_roots: &SupervisorTrustRoots) -> Result<(), GauntletError> {
        trust_roots.validate_key_ids()?;
        if self.schema_version != SUPERVISOR_AUTH_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unknown absence schema version {} fails closed",
                self.schema_version
            )));
        }
        validate_identity_hex(
            &self.execution_identity_sha256,
            "absence execution identity",
        )?;
        let verifying_key =
            trust_roots.resolve(&self.key_id, COMPLETION_ISSUER_ROLE, self.witnessed_at_ns)?;
        let signature_bytes = decode_hex_exact(&self.signature_hex, 64, "absence signature")?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|error| invalid(format!("absence signature is malformed: {error}")))?;
        verifying_key
            .verify(&self.signing_message()?, &signature)
            .map_err(|_| invalid("absence signature does not verify".to_owned()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NonceState {
    Issued {
        source_identity_sha256: String,
        build_identity_sha256: String,
        command_digest_sha256: String,
        expires_at_ns: i64,
    },
    Consumed,
}

/// The supervisor's signing authority. The Ed25519 signing key lives ONLY
/// here — it is never serialized (F0 privacy class `Secret`), and the
/// measured child never observes it.
pub struct SupervisorSigningAuthority {
    signing_key: SigningKey,
    key_id: String,
    nonces: BTreeMap<String, NonceState>,
}

impl std::fmt::Debug for SupervisorSigningAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The signing key is Secret-class: never printed, never serialized.
        f.debug_struct("SupervisorSigningAuthority")
            .field("key_id", &self.key_id)
            .field("outstanding_nonces", &self.nonces.len())
            .finish_non_exhaustive()
    }
}

impl SupervisorSigningAuthority {
    /// Generate a fresh authority from operating-system entropy.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when the operating system's entropy source
    /// is unavailable.
    pub fn generate() -> Result<Self, GauntletError> {
        let mut seed = [0_u8; 32];
        getrandom::fill(&mut seed)
            .map_err(|error| invalid(format!("cannot draw supervisor key entropy: {error}")))?;
        Ok(Self::from_seed(seed))
    }

    /// Deterministic construction for tests and rotation tooling.
    #[must_use]
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(&seed);
        let key_id = lower_hex(&Sha256::digest(signing_key.verifying_key().as_bytes()));
        Self {
            signing_key,
            key_id,
            nonces: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Trust-root entry for this authority, granted both frozen roles.
    #[must_use]
    pub fn trust_root_key(&self, valid_from_ns: i64) -> TrustRootKey {
        TrustRootKey {
            verifying_key_hex: lower_hex(self.signing_key.verifying_key().as_bytes()),
            valid_from_ns,
            retired_after_ns: None,
            revoked: false,
            roles: vec![
                EXECUTION_SUPERVISOR_ROLE.to_owned(),
                COMPLETION_ISSUER_ROLE.to_owned(),
            ],
        }
    }

    /// Issue one single-use non-zero 128-bit nonce bound to the source,
    /// build, command digest, and expiry (F0 nonce rules).
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] for a non-positive or over-15-minute
    /// lifetime, malformed identities, entropy failure, or the
    /// quarantine-class event of a duplicate nonce.
    pub fn issue_nonce(
        &mut self,
        source_identity_sha256: &str,
        build_identity_sha256: &str,
        command_digest_sha256: &str,
        issued_at_ns: i64,
        lifetime_ns: i64,
    ) -> Result<(String, i64), GauntletError> {
        if lifetime_ns <= 0 || lifetime_ns > MAX_NONCE_LIFETIME_NS {
            return Err(invalid(
                "nonce lifetime must be positive and at most 15 minutes".to_owned(),
            ));
        }
        validate_identity_hex(source_identity_sha256, "nonce source identity")?;
        validate_identity_hex(build_identity_sha256, "nonce build identity")?;
        validate_identity_hex(command_digest_sha256, "nonce command digest")?;
        let mut nonce = [0_u8; 16];
        loop {
            getrandom::fill(&mut nonce)
                .map_err(|error| invalid(format!("cannot draw nonce entropy: {error}")))?;
            if nonce.iter().any(|byte| *byte != 0) {
                break;
            }
        }
        let nonce_hex = lower_hex(&nonce);
        let expires_at_ns = issued_at_ns
            .checked_add(lifetime_ns)
            .ok_or_else(|| invalid("nonce expiry overflows".to_owned()))?;
        let previous = self.nonces.insert(
            nonce_hex.clone(),
            NonceState::Issued {
                source_identity_sha256: source_identity_sha256.to_owned(),
                build_identity_sha256: build_identity_sha256.to_owned(),
                command_digest_sha256: command_digest_sha256.to_owned(),
                expires_at_ns,
            },
        );
        if previous.is_some() {
            // 128 bits of fresh entropy colliding means the entropy source is
            // broken; treat it as the terminal Quarantine class F0 assigns to
            // duplicate nonces rather than continuing.
            return Err(invalid(
                "duplicate nonce issuance: entropy failure is a terminal quarantine".to_owned(),
            ));
        }
        Ok((nonce_hex, expires_at_ns))
    }

    /// Sign an execution object. The nonce must be one this authority issued,
    /// unconsumed, unexpired at the execution start, and bound to the same
    /// source, build, and command digests.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] for an unknown, consumed, expired, or
    /// cross-build nonce, and for an execution object that misstates the
    /// issued expiry or violates its own structural contract.
    pub fn sign_execution(
        &mut self,
        execution: &ArtifactStoreV4ExecutionObject,
        signed_at_ns: i64,
    ) -> Result<SignedReceiptV4, GauntletError> {
        let state = self.nonces.get(&execution.nonce_hex).ok_or_else(|| {
            invalid("execution nonce was not issued by this supervisor".to_owned())
        })?;
        match state {
            NonceState::Consumed => {
                return Err(invalid(
                    "nonce reuse is a terminal quarantine, not a retryable path".to_owned(),
                ));
            }
            NonceState::Issued {
                source_identity_sha256,
                build_identity_sha256,
                command_digest_sha256,
                expires_at_ns,
            } => {
                if *source_identity_sha256 != execution.source_identity_sha256
                    || *build_identity_sha256 != execution.predecessor_build_identity_sha256
                    || *command_digest_sha256 != execution.command_digest_sha256
                {
                    return Err(invalid(
                        "nonce is bound to a different source/build/command; cross-build reuse fails closed"
                            .to_owned(),
                    ));
                }
                if execution.started_at_ns > *expires_at_ns {
                    return Err(invalid(
                        "nonce expired before execution start; expired nonces may not be used"
                            .to_owned(),
                    ));
                }
                if execution.nonce_expires_at_ns != *expires_at_ns {
                    return Err(invalid(
                        "execution object misstates the issued nonce expiry".to_owned(),
                    ));
                }
            }
        }
        let (_, identity) = execution.identity()?;
        self.sign_identity(
            "execution",
            &identity,
            EXECUTION_SUPERVISOR_ROLE,
            signed_at_ns,
        )
    }

    /// Sign a completion, CONSUMING the nonce: a second signature for the
    /// same nonce is refused permanently (F0: single-use, duplicate is a
    /// terminal quarantine).
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when the completion does not bind the
    /// execution identity or nonce, when the nonce is unknown or already
    /// consumed, or when either object violates its structural contract.
    pub fn sign_completion(
        &mut self,
        execution: &ArtifactStoreV4ExecutionObject,
        completion: &ArtifactStoreV4CompletionObject,
        signed_at_ns: i64,
    ) -> Result<SignedReceiptV4, GauntletError> {
        let (_, execution_identity) = execution.identity()?;
        if completion.predecessor_execution_identity_sha256 != execution_identity {
            return Err(invalid(
                "completion does not bind the execution identity".to_owned(),
            ));
        }
        if completion.nonce_hex != execution.nonce_hex {
            return Err(invalid(
                "completion does not restate the execution nonce".to_owned(),
            ));
        }
        let state = self.nonces.get_mut(&completion.nonce_hex).ok_or_else(|| {
            invalid("completion nonce was not issued by this supervisor".to_owned())
        })?;
        if *state == NonceState::Consumed {
            return Err(invalid(
                "nonce reuse is a terminal quarantine, not a retryable path".to_owned(),
            ));
        }
        *state = NonceState::Consumed;
        let (_, identity) = completion.identity()?;
        self.sign_identity(
            "completion",
            &identity,
            COMPLETION_ISSUER_ROLE,
            signed_at_ns,
        )
    }

    /// Sign a typed completion ABSENCE, consuming the nonce so no signed
    /// completion can later appear for the same execution.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] for a malformed reason code, an unknown or
    /// already-consumed nonce, or an execution object that violates its
    /// structural contract.
    pub fn sign_absence(
        &mut self,
        execution: &ArtifactStoreV4ExecutionObject,
        reason_code: &str,
        witnessed_at_ns: i64,
    ) -> Result<CompletionAbsenceV4, GauntletError> {
        validate_bounded_ascii_label(reason_code, "absence reason code")?;
        let (_, execution_identity) = execution.identity()?;
        let state = self
            .nonces
            .get_mut(&execution.nonce_hex)
            .ok_or_else(|| invalid("absence nonce was not issued by this supervisor".to_owned()))?;
        if *state == NonceState::Consumed {
            return Err(invalid(
                "nonce reuse is a terminal quarantine, not a retryable path".to_owned(),
            ));
        }
        *state = NonceState::Consumed;
        let mut absence = CompletionAbsenceV4 {
            schema_version: SUPERVISOR_AUTH_SCHEMA_VERSION,
            execution_identity_sha256: execution_identity,
            nonce_hex: execution.nonce_hex.clone(),
            reason_code: reason_code.to_owned(),
            witnessed_at_ns,
            key_id: self.key_id.clone(),
            signature_hex: String::new(),
        };
        let signature = self.signing_key.sign(&absence.signing_message()?);
        absence.signature_hex = lower_hex(&signature.to_bytes());
        Ok(absence)
    }

    fn sign_identity(
        &self,
        object_kind: &str,
        object_identity: &str,
        role: &str,
        signed_at_ns: i64,
    ) -> Result<SignedReceiptV4, GauntletError> {
        let mut receipt = SignedReceiptV4 {
            schema_version: SUPERVISOR_AUTH_SCHEMA_VERSION,
            object_kind: object_kind.to_owned(),
            object_identity_sha256: object_identity.to_owned(),
            role: role.to_owned(),
            key_id: self.key_id.clone(),
            signed_at_ns,
            signature_hex: String::new(),
        };
        let signature = self.signing_key.sign(&receipt.signing_message()?);
        receipt.signature_hex = lower_hex(&signature.to_bytes());
        Ok(receipt)
    }
}

/// Launch specification for one supervised execution. Paths live here (the
/// supervisor's private working state); the signed objects carry only
/// digests and bounded labels.
#[derive(Debug, Clone)]
pub struct SupervisedLaunchSpec {
    pub executable: PathBuf,
    pub args: Vec<String>,
    /// Environment allowlist: ONLY these variables reach the child.
    pub environment: BTreeMap<String, String>,
    pub run_window: Duration,
    /// Per-stream retained output bound; beyond it the stream digest covers
    /// the retained prefix and the completion records truncation.
    pub max_output_bytes: u64,
    /// Files the supervisor collects (digests) after the child terminates,
    /// keyed by bounded artifact label.
    pub expected_artifacts: BTreeMap<String, PathBuf>,
}

impl SupervisedLaunchSpec {
    /// Digest over the exact argv (executable label excluded — the admitted
    /// executable is bound separately by its content digest).
    #[must_use]
    pub fn command_digest(&self) -> String {
        let mut hasher = Sha256::new();
        for arg in &self.args {
            hasher.update(arg.as_bytes());
            hasher.update([0_u8]);
        }
        lower_hex(&hasher.finalize())
    }

    /// Digest over the sorted environment NAMES admitted to the child.
    #[must_use]
    pub fn environment_policy_digest(&self) -> String {
        let mut hasher = Sha256::new();
        for name in self.environment.keys() {
            hasher.update(name.as_bytes());
            hasher.update([0_u8]);
        }
        lower_hex(&hasher.finalize())
    }
}

/// A cooperative cancellation handle for one supervised run.
#[derive(Debug, Clone, Default)]
pub struct SupervisionCancel {
    flag: Arc<AtomicBool>,
}

impl SupervisionCancel {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// Everything the supervisor observed for one terminal run, before signing.
#[derive(Debug)]
pub struct SupervisedTermination {
    pub terminal_outcome: TerminalOutcome,
    pub outcome_reason_code: String,
    pub exit_code: Option<i64>,
    pub termination_signal: Option<i64>,
    pub wall_clock: Duration,
    pub stdout: CollectedArtifactDigest,
    pub stderr: CollectedArtifactDigest,
    pub artifact_index: BTreeMap<String, CollectedArtifactDigest>,
    pub missing_artifacts: Vec<String>,
    pub process_tree_escape_detected: bool,
}

fn digest_reader_bounded(
    mut reader: impl Read,
    max_bytes: u64,
) -> (CollectedArtifactDigest, Vec<u8>) {
    let mut hasher = Sha256::new();
    let mut retained = Vec::new();
    let mut buffer = [0_u8; 8192];
    let mut truncated = false;
    let mut total: u64 = 0;
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => {
                let budget = max_bytes.saturating_sub(total);
                let read_len = u64::try_from(read).unwrap_or(u64::MAX);
                let keep = usize::try_from(read_len.min(budget)).unwrap_or(read);
                if keep > 0 {
                    hasher.update(&buffer[..keep]);
                    retained.extend_from_slice(&buffer[..keep]);
                }
                if read_len > budget {
                    truncated = true;
                }
                total = total.saturating_add(u64::try_from(keep).unwrap_or(u64::MAX));
                if truncated {
                    // Keep draining so the child is never blocked on a full
                    // pipe, but retain and digest nothing further.
                    let mut sink = [0_u8; 8192];
                    while matches!(reader.read(&mut sink), Ok(n) if n > 0) {}
                    break;
                }
            }
            Err(_) => break,
        }
    }
    (
        CollectedArtifactDigest {
            sha256: lower_hex(&hasher.finalize()),
            byte_len: total,
            truncated,
        },
        retained,
    )
}

fn supervised_group_pid(pid: u32) -> Option<rustix::process::Pid> {
    i32::try_from(pid)
        .ok()
        .and_then(rustix::process::Pid::from_raw)
}

fn kill_supervised_group(pid: u32) {
    if let Some(raw) = supervised_group_pid(pid) {
        let _ = rustix::process::kill_process_group(raw, rustix::process::Signal::KILL);
    }
}

fn group_has_survivors(pid: u32) -> bool {
    supervised_group_pid(pid)
        .is_some_and(|raw| rustix::process::test_kill_process_group(raw).is_ok())
}

/// Launch and supervise the admitted executable to a terminal state.
///
/// The supervisor verifies the executable's content digest against the
/// admitted `expected_executable_sha256` BEFORE launch (source/ELF mismatch
/// refuses to launch), starts the child in its own process group with only
/// the allowlisted environment, enforces the run window and cancellation,
/// bounds and digests both output streams, digests expected artifacts after
/// termination, and sweeps the process group for escapees.
///
/// Every failure mode returns a typed terminal observation — the caller
/// decides whether it becomes a signed completion or a typed absence;
/// nothing here signs.
///
/// # Errors
///
/// Returns [`GauntletError`] only when the supervisor itself cannot observe
/// the child (a failed wait) or an expected-artifact label is malformed;
/// child-side failures are typed terminal observations, not errors.
pub fn supervise_execution(
    spec: &SupervisedLaunchSpec,
    expected_executable_sha256: &str,
    cancel: &SupervisionCancel,
) -> Result<SupervisedTermination, GauntletError> {
    let started = Instant::now();
    let Ok(admitted_bytes) = std::fs::read(&spec.executable) else {
        return Ok(pre_launch_failure(
            "launch_failure_executable_unreadable",
            started.elapsed(),
        ));
    };
    let actual_executable_sha256 = lower_hex(&Sha256::digest(&admitted_bytes));
    drop(admitted_bytes);
    if actual_executable_sha256 != expected_executable_sha256 {
        return Ok(pre_launch_failure(
            "launch_refused_executable_digest_mismatch",
            started.elapsed(),
        ));
    }

    let mut command = Command::new(&spec.executable);
    command
        .args(&spec.args)
        .env_clear()
        .envs(&spec.environment)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        // A fresh process group so the whole descendant tree is one kill and
        // one survivor probe. Safe API (no pre_exec): stable since 1.64.
        command.process_group(0);
    }
    let Ok(mut child) = command.spawn() else {
        return Ok(pre_launch_failure("launch_failure_exec", started.elapsed()));
    };
    let pid = child.id();
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let max_output = spec.max_output_bytes;
    let stdout_thread = std::thread::spawn(move || {
        stdout_pipe.map_or_else(
            || (empty_stream_digest(), Vec::new()),
            |pipe| digest_reader_bounded(pipe, max_output),
        )
    });
    let stderr_thread = std::thread::spawn(move || {
        stderr_pipe.map_or_else(
            || (empty_stream_digest(), Vec::new()),
            |pipe| digest_reader_bounded(pipe, max_output),
        )
    });

    let mut outcome;
    let mut reason;
    let mut exit_code = None;
    let mut termination_signal = None;
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| invalid(format!("supervisor cannot observe the child: {error}")))?
        {
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt as _;
                if let Some(signal) = status.signal() {
                    outcome = TerminalOutcome::Failed;
                    reason = "terminated_by_signal".to_owned();
                    termination_signal = Some(i64::from(signal));
                } else {
                    let code = status.code().unwrap_or(-1);
                    exit_code = Some(i64::from(code));
                    if code == 0 {
                        outcome = TerminalOutcome::Succeeded;
                        reason = "exited_zero".to_owned();
                    } else {
                        outcome = TerminalOutcome::Failed;
                        reason = "exited_nonzero".to_owned();
                    }
                }
            }
            #[cfg(not(unix))]
            {
                let code = status.code().unwrap_or(-1);
                exit_code = Some(i64::from(code));
                outcome = if code == 0 {
                    TerminalOutcome::Succeeded
                } else {
                    TerminalOutcome::Failed
                };
                reason = if code == 0 {
                    "exited_zero".to_owned()
                } else {
                    "exited_nonzero".to_owned()
                };
            }
            break;
        }
        if cancel.is_cancelled() {
            kill_supervised_group(pid);
            let _ = child.wait();
            outcome = TerminalOutcome::Cancelled;
            reason = "cancelled_by_supervisor".to_owned();
            break;
        }
        if started.elapsed() >= spec.run_window {
            kill_supervised_group(pid);
            let _ = child.wait();
            outcome = TerminalOutcome::TimedOut;
            reason = "run_window_exceeded".to_owned();
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }

    // Escape sweep BEFORE joining the readers: a grandchild left behind in
    // the process group typically inherits the output pipes, so joining
    // first would block on the escapee's lifetime and the probe would then
    // find nothing. Sweeping first records the escape, contains it, and
    // thereby also closes the inherited pipe writers so the joins below
    // return promptly.
    let process_tree_escape_detected = group_has_survivors(pid);
    if process_tree_escape_detected {
        kill_supervised_group(pid);
    }

    let (stdout_digest, _stdout_bytes) = stdout_thread
        .join()
        .unwrap_or_else(|_| (empty_stream_digest(), Vec::new()));
    let (stderr_digest, _stderr_bytes) = stderr_thread
        .join()
        .unwrap_or_else(|_| (empty_stream_digest(), Vec::new()));
    if stdout_digest.truncated || stderr_digest.truncated {
        // Overflow is its own terminal reason. The group was already swept
        // above; this kill is belt-and-braces for a child that outlived its
        // pipes on a non-group platform.
        kill_supervised_group(pid);
        let _ = child.wait();
        if outcome == TerminalOutcome::Succeeded || outcome == TerminalOutcome::Unknown {
            outcome = TerminalOutcome::Failed;
        }
        reason = "output_overflow".to_owned();
    }

    let mut artifact_index = BTreeMap::new();
    let mut missing_artifacts = Vec::new();
    for (label, path) in &spec.expected_artifacts {
        validate_bounded_ascii_label(label, "expected artifact label")?;
        match std::fs::read(path) {
            Ok(bytes) => {
                artifact_index.insert(
                    label.clone(),
                    CollectedArtifactDigest {
                        sha256: lower_hex(&Sha256::digest(&bytes)),
                        byte_len: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                        truncated: false,
                    },
                );
            }
            Err(_) => missing_artifacts.push(label.clone()),
        }
    }

    Ok(SupervisedTermination {
        terminal_outcome: outcome,
        outcome_reason_code: reason,
        exit_code,
        termination_signal,
        wall_clock: started.elapsed(),
        stdout: stdout_digest,
        stderr: stderr_digest,
        artifact_index,
        missing_artifacts,
        process_tree_escape_detected,
    })
}

/// A typed pre-launch failure: nothing ran, so streams are empty and no
/// artifact was collected. The reason code carries the refusal class.
fn pre_launch_failure(reason_code: &str, wall_clock: Duration) -> SupervisedTermination {
    SupervisedTermination {
        terminal_outcome: TerminalOutcome::Failed,
        outcome_reason_code: reason_code.to_owned(),
        exit_code: None,
        termination_signal: None,
        wall_clock,
        stdout: empty_stream_digest(),
        stderr: empty_stream_digest(),
        artifact_index: BTreeMap::new(),
        missing_artifacts: Vec::new(),
        process_tree_escape_detected: false,
    }
}

fn empty_stream_digest() -> CollectedArtifactDigest {
    CollectedArtifactDigest {
        sha256: lower_hex(&Sha256::digest([])),
        byte_len: 0,
        truncated: false,
    }
}

/// Verify a full execution+completion pair against the trust roots.
///
/// Canonical bytes, identities, both signatures under the frozen roles,
/// nonce linkage, and expiry — fail-closed throughout. Verification is
/// outcome-neutral and never implies admission, decision, or release.
///
/// # Errors
///
/// Returns [`GauntletError`] on non-canonical bytes, unknown fields or
/// schemas, an untrusted or role-mismatched signer, a failed signature, a
/// predecessor or nonce mismatch, or an execution that started after its
/// nonce expired.
pub fn verify_execution_completion_chain(
    trust_roots: &SupervisorTrustRoots,
    execution_bytes: &[u8],
    execution_receipt: &SignedReceiptV4,
    completion_bytes: &[u8],
    completion_receipt: &SignedReceiptV4,
) -> Result<
    (
        ArtifactStoreV4ExecutionObject,
        ArtifactStoreV4CompletionObject,
    ),
    GauntletError,
> {
    let execution: ArtifactStoreV4ExecutionObject = decode_canonical(execution_bytes)?;
    let (_, execution_identity) = execution.identity()?;
    execution_receipt.verify(
        trust_roots,
        "execution",
        EXECUTION_SUPERVISOR_ROLE,
        &execution_identity,
    )?;
    let completion: ArtifactStoreV4CompletionObject = decode_canonical(completion_bytes)?;
    let (_, completion_identity) = completion.identity()?;
    completion_receipt.verify(
        trust_roots,
        "completion",
        COMPLETION_ISSUER_ROLE,
        &completion_identity,
    )?;
    if completion.predecessor_execution_identity_sha256 != execution_identity {
        return Err(invalid(
            "completion predecessor does not match the verified execution identity".to_owned(),
        ));
    }
    if completion.nonce_hex != execution.nonce_hex {
        return Err(invalid(
            "completion nonce does not match the execution nonce".to_owned(),
        ));
    }
    if execution.started_at_ns > execution.nonce_expires_at_ns {
        return Err(invalid(
            "execution started after its nonce expired".to_owned(),
        ));
    }
    Ok((execution, completion))
}

// ===== ArtifactStore v4 F3: chain codec, store admission, and the =====
// ===== pre-policy rejecting canary                                =====
// (bd-artifactstore-v4-f3-codec-admission-xwtpw)

/// Store-level chain-index identity domain.
///
/// Not one of the four frozen F0 object domains — the index is store
/// metadata binding the four object identities; F1-F4 may add strictly
/// stronger checks like this one.
pub const ARTIFACTSTORE_V4_CHAIN_INDEX_DOMAIN: &str = "frankensearch.artifactstore.v4.chain-index";

/// Frozen F0 authentication axis (schema `authentication`). No value is ever
/// inferred from another axis: a verified chain is not admission, decision,
/// or release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthenticationClass {
    VerifiedReceiptChain,
    IntegrityOnly,
    UnauthenticatedLegacy,
}

/// The store's chain index: one record binding the full object graph by
/// complete identities (never paths or labels). Its publication is the
/// chain's commit point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactStoreV4ChainIndex {
    pub schema_version: u32,
    pub source_identity_sha256: String,
    pub build_identity_sha256: String,
    pub execution_identity_sha256: String,
    pub completion_identity_sha256: String,
    /// Restated consumed nonce so index substitution is a one-field mismatch.
    pub nonce_hex: String,
    pub terminal_outcome: TerminalOutcome,
}

impl ArtifactStoreV4ChainIndex {
    /// Canonical bytes + store-domain identity.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when a field violates its structural
    /// contract or the index cannot serialize canonically.
    pub fn identity(&self) -> Result<(Vec<u8>, String), GauntletError> {
        if self.schema_version != SUPERVISOR_AUTH_SCHEMA_VERSION {
            return Err(invalid(format!(
                "unknown chain-index schema version {} fails closed",
                self.schema_version
            )));
        }
        for (identity, what) in [
            (&self.source_identity_sha256, "chain source identity"),
            (&self.build_identity_sha256, "chain build identity"),
            (&self.execution_identity_sha256, "chain execution identity"),
            (
                &self.completion_identity_sha256,
                "chain completion identity",
            ),
        ] {
            validate_identity_hex(identity, what)?;
        }
        decode_hex_exact(&self.nonce_hex, 16, "chain nonce")?;
        let canonical = canonical_json_bytes(self)?;
        let identity = domain_identity(ARTIFACTSTORE_V4_CHAIN_INDEX_DOMAIN, &canonical);
        Ok((canonical, identity))
    }
}

/// A fully verified, reloaded v4 chain. Construction happens ONLY through
/// [`ArtifactStoreV4ChainStore::load_verified_chain`]; the classification is
/// therefore earned, never asserted by the caller.
#[derive(Debug)]
pub struct VerifiedV4Chain {
    pub authentication: AuthenticationClass,
    pub index: ArtifactStoreV4ChainIndex,
    pub execution: ArtifactStoreV4ExecutionObject,
    pub completion: ArtifactStoreV4CompletionObject,
}

/// Result of one content-addressed publication step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PublishOutcome {
    Written,
    AlreadyPresentIdentical,
}

/// Content-addressed store for v4 execution/completion chains.
///
/// Layout under the root: `objects/<identity>` (raw canonical object bytes),
/// `receipts/<object-identity>` (canonical receipt bytes),
/// `absences/<execution-identity>` (canonical absence bytes), and
/// `chains/<chain-identity>` (canonical index bytes; published LAST as the
/// commit point). Every publication is create-new atomic: exclusive pending
/// file, write, fsync, rename, parent-directory fsync. A leftover pending
/// file from a crash fails closed rather than being silently overwritten.
#[derive(Debug)]
pub struct ArtifactStoreV4ChainStore {
    root: PathBuf,
}

impl ArtifactStoreV4ChainStore {
    /// Open (creating directories as needed) a chain store at `root`.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when the directories cannot be created.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, GauntletError> {
        let root = root.into();
        for child in ["objects", "receipts", "absences", "chains"] {
            std::fs::create_dir_all(root.join(child)).map_err(|error| {
                invalid(format!("cannot create v4 chain store directory: {error}"))
            })?;
        }
        Ok(Self { root })
    }

    fn child_dir(&self, kind: &str) -> PathBuf {
        self.root.join(kind)
    }

    /// Create-new atomic publication of content-addressed bytes.
    ///
    /// An existing file with IDENTICAL bytes is idempotent success; an
    /// existing file with different bytes at the same address is the
    /// quarantine-class collision F0 assigns to identity conflicts.
    fn publish_atomic(
        &self,
        kind: &str,
        name: &str,
        bytes: &[u8],
    ) -> Result<PublishOutcome, GauntletError> {
        validate_identity_hex(name, "store entry name")?;
        let dir = self.child_dir(kind);
        let final_path = dir.join(name);
        if final_path.exists() {
            let existing = std::fs::read(&final_path)
                .map_err(|error| invalid(format!("cannot read existing store entry: {error}")))?;
            if existing == bytes {
                return Ok(PublishOutcome::AlreadyPresentIdentical);
            }
            return Err(invalid(format!(
                "content-address collision in {kind}: existing bytes differ; \
                 this is a terminal quarantine, not an overwrite"
            )));
        }
        let pending_path = dir.join(format!(".{name}.pending"));
        let mut pending = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending_path)
            .map_err(|error| {
                invalid(format!(
                    "cannot create pending store entry (a leftover pending file from a \
                     crash fails closed and needs explicit operator recovery): {error}"
                ))
            })?;
        use std::io::Write as _;
        pending
            .write_all(bytes)
            .and_then(|()| pending.sync_all())
            .map_err(|error| invalid(format!("cannot write pending store entry: {error}")))?;
        drop(pending);
        std::fs::rename(&pending_path, &final_path)
            .map_err(|error| invalid(format!("cannot publish store entry: {error}")))?;
        if let Ok(dir_handle) = std::fs::File::open(&dir) {
            let _ = dir_handle.sync_all();
        }
        Ok(PublishOutcome::Written)
    }

    fn read_entry(&self, kind: &str, name: &str) -> Result<Vec<u8>, GauntletError> {
        validate_identity_hex(name, "store entry name")?;
        std::fs::read(self.child_dir(kind).join(name)).map_err(|error| {
            invalid(format!(
                "v4 chain store entry {kind}/{name} is absent or unreadable: {error}"
            ))
        })
    }

    /// Verify a chain FIRST, then publish it atomically; the chain index is
    /// written last as the commit point. Nothing unverified is ever written.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when verification fails, when a
    /// content-address collision is detected, or on a publication I/O
    /// failure.
    pub fn publish_verified_chain(
        &self,
        trust_roots: &SupervisorTrustRoots,
        execution: &ArtifactStoreV4ExecutionObject,
        execution_receipt: &SignedReceiptV4,
        completion: &ArtifactStoreV4CompletionObject,
        completion_receipt: &SignedReceiptV4,
    ) -> Result<String, GauntletError> {
        let (execution_bytes, execution_identity) = execution.identity()?;
        let (completion_bytes, completion_identity) = completion.identity()?;
        verify_execution_completion_chain(
            trust_roots,
            &execution_bytes,
            execution_receipt,
            &completion_bytes,
            completion_receipt,
        )?;
        let index = ArtifactStoreV4ChainIndex {
            schema_version: SUPERVISOR_AUTH_SCHEMA_VERSION,
            source_identity_sha256: execution.source_identity_sha256.clone(),
            build_identity_sha256: execution.predecessor_build_identity_sha256.clone(),
            execution_identity_sha256: execution_identity.clone(),
            completion_identity_sha256: completion_identity.clone(),
            nonce_hex: execution.nonce_hex.clone(),
            terminal_outcome: completion.terminal_outcome,
        };
        let (index_bytes, chain_identity) = index.identity()?;
        self.publish_atomic("objects", &execution_identity, &execution_bytes)?;
        self.publish_atomic("objects", &completion_identity, &completion_bytes)?;
        self.publish_atomic(
            "receipts",
            &execution_identity,
            &canonical_json_bytes(execution_receipt)?,
        )?;
        self.publish_atomic(
            "receipts",
            &completion_identity,
            &canonical_json_bytes(completion_receipt)?,
        )?;
        self.publish_atomic("chains", &chain_identity, &index_bytes)?;
        Ok(chain_identity)
    }

    /// Publish a typed completion absence for an execution (the execution
    /// object and its receipt are stored alongside so the absence is
    /// independently verifiable). Absences never receive a chain index —
    /// there is no completion to commit.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when the absence does not verify or on a
    /// publication failure.
    pub fn publish_verified_absence(
        &self,
        trust_roots: &SupervisorTrustRoots,
        execution: &ArtifactStoreV4ExecutionObject,
        execution_receipt: &SignedReceiptV4,
        absence: &CompletionAbsenceV4,
    ) -> Result<(), GauntletError> {
        let (execution_bytes, execution_identity) = execution.identity()?;
        execution_receipt.verify(
            trust_roots,
            "execution",
            EXECUTION_SUPERVISOR_ROLE,
            &execution_identity,
        )?;
        absence.verify(trust_roots)?;
        if absence.execution_identity_sha256 != execution_identity {
            return Err(invalid(
                "absence does not bind the execution identity".to_owned(),
            ));
        }
        self.publish_atomic("objects", &execution_identity, &execution_bytes)?;
        self.publish_atomic(
            "receipts",
            &execution_identity,
            &canonical_json_bytes(execution_receipt)?,
        )?;
        self.publish_atomic(
            "absences",
            &execution_identity,
            &canonical_json_bytes(absence)?,
        )?;
        Ok(())
    }

    /// Verified reload: the ONLY constructor of [`VerifiedV4Chain`].
    ///
    /// Binds, fail-closed and in order: the chain index's content address,
    /// both objects' content addresses recomputed from their exact stored
    /// bytes, both receipts under the frozen roles and trust roots, the
    /// nonce, and the full source/build/execution/completion graph — all
    /// BEFORE any policy sees the data. The result is authentication only:
    /// `VerifiedReceiptChain` never implies admission, decision, or release.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] on any absent, truncated, extended,
    /// tampered, non-canonical, digest-mismatched, signature-mismatched,
    /// wrong-nonce, wrong-signer, or graph-inconsistent entry.
    pub fn load_verified_chain(
        &self,
        trust_roots: &SupervisorTrustRoots,
        chain_identity: &str,
    ) -> Result<VerifiedV4Chain, GauntletError> {
        let index_bytes = self.read_entry("chains", chain_identity)?;
        let recomputed = domain_identity(ARTIFACTSTORE_V4_CHAIN_INDEX_DOMAIN, &index_bytes);
        if recomputed != chain_identity {
            return Err(invalid(
                "chain index bytes do not match their content address".to_owned(),
            ));
        }
        let index: ArtifactStoreV4ChainIndex = decode_canonical(&index_bytes)?;

        let execution_bytes = self.read_entry("objects", &index.execution_identity_sha256)?;
        let execution_recomputed =
            domain_identity(ARTIFACTSTORE_V4_EXECUTION_DOMAIN, &execution_bytes);
        if execution_recomputed != index.execution_identity_sha256 {
            return Err(invalid(
                "execution bytes do not match their content address".to_owned(),
            ));
        }
        let completion_bytes = self.read_entry("objects", &index.completion_identity_sha256)?;
        let completion_recomputed =
            domain_identity(ARTIFACTSTORE_V4_COMPLETION_DOMAIN, &completion_bytes);
        if completion_recomputed != index.completion_identity_sha256 {
            return Err(invalid(
                "completion bytes do not match their content address".to_owned(),
            ));
        }
        let execution_receipt: SignedReceiptV4 =
            decode_canonical(&self.read_entry("receipts", &index.execution_identity_sha256)?)?;
        let completion_receipt: SignedReceiptV4 =
            decode_canonical(&self.read_entry("receipts", &index.completion_identity_sha256)?)?;
        let (execution, completion) = verify_execution_completion_chain(
            trust_roots,
            &execution_bytes,
            &execution_receipt,
            &completion_bytes,
            &completion_receipt,
        )?;
        // The object and index field names deliberately differ
        // (predecessor_build_identity vs build_identity): the index is a
        // separate record, not a mirror.
        #[allow(clippy::suspicious_operation_groupings)]
        if execution.source_identity_sha256 != index.source_identity_sha256
            || execution.predecessor_build_identity_sha256 != index.build_identity_sha256
            || execution.nonce_hex != index.nonce_hex
            || completion.terminal_outcome != index.terminal_outcome
        {
            return Err(invalid(
                "chain index does not bind the verified object graph".to_owned(),
            ));
        }
        Ok(VerifiedV4Chain {
            authentication: AuthenticationClass::VerifiedReceiptChain,
            index,
            execution,
            completion,
        })
    }
}

/// Pre-policy v4 rejecting canary (F3).
///
/// Legacy schema classifiers dispatch on a bare `schema_version` integer —
/// and every v4 chain object also carries `schema_version`, so v4-shaped
/// bytes fed to a legacy reader would silently classify as
/// `UnauthenticatedLegacy` and fall through to permissive legacy handling.
/// This probe runs BEFORE any legacy/default policy: bytes that carry v4
/// chain field signatures are rejected with a typed reason instead of being
/// classified at all.
#[must_use]
pub fn v4_chain_shape_rejection(bytes: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let object = value.as_object()?;
    let has_nonce = object.contains_key("nonce_hex");
    let chain_markers = [
        "predecessor_build_identity_sha256",
        "predecessor_execution_identity_sha256",
        "execution_identity_sha256",
        "completion_identity_sha256",
    ];
    let has_chain_marker = chain_markers.iter().any(|key| object.contains_key(*key));
    let is_receipt_shaped = object.contains_key("object_identity_sha256")
        && object.contains_key("signature_hex")
        && object.contains_key("role");
    if (has_nonce && has_chain_marker) || is_receipt_shaped {
        return Some(
            "v4 chain-shaped bytes may not enter a legacy reader: route them \
             through the ArtifactStore v4 verified-reload path"
                .to_owned(),
        );
    }
    None
}

// ===== ArtifactStore v4 F4: typed consumer claims and promotion policy =====
// (bd-artifactstore-v4-f4-consumer-wiring-396po)

/// Consumer evidence modes. `FixturePublic` is the only CI and
/// public-artifact mode; `PrivateLocal` policy is independently governed and
/// can never block public fixture evidence — the evaluator consults private
/// policy ONLY for `PrivateLocal` claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsumerMode {
    FixturePublic,
    PrivateLocal,
}

/// The claim a consumer is making about a verified chain. The ladder is
/// strictly ordered in required authority; nothing here is inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsumerClaim {
    /// Always `NoClaim`, even when the chain is fully authentic.
    DiagnosticAttempt,
    /// May record `Pass` or `Miss` exactly as evaluated, never relabeled.
    AdmittedCampaignEvidence,
    /// Requires a verified chain, admitted `Pass`, complete required
    /// coverage, a successful terminal completion, durable evidence, and an
    /// explicit release authority.
    ReplacementQualified,
}

/// Frozen F0 admission axis (schema `admission`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionState {
    Admitted,
    Unadmitted,
    NoDecision,
}

/// Frozen F0 decision axis (schema `decision`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionState {
    Pass,
    Miss,
    NoDecision,
    Quarantine,
}

/// Coverage witnesses for one claim: every required label must be witnessed,
/// and every witness must come from the SAME machine profile as the chain's
/// execution (a cross-worker mix is not coverage, it is substitution).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageWitnessSet {
    pub required: std::collections::BTreeSet<String>,
    /// Witness label -> (machine profile, witness digest).
    pub witnessed: BTreeMap<String, (String, String)>,
}

/// Explicit release authority. Its existence is a deliberate, named act —
/// release can never fall out of authentication or admission by default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseAuthority {
    pub authority_label: String,
}

/// The admitted campaign identities a consumer expects the chain to bind
/// (from the F1 source/build admission). A chain naming other identities is
/// stale or substituted evidence regardless of its own authenticity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedChainBinding {
    pub source_identity_sha256: String,
    pub build_identity_sha256: String,
    pub executable_sha256: String,
}

/// Typed policy output. `NoClaim` carries a bounded reason code;
/// `RecordedEvidence` preserves the evaluated decision without relabeling;
/// `Qualified` exists only for the full replacement ladder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromotionOutput {
    NoClaim { reason_code: String },
    RecordedEvidence { decision: DecisionState },
    Qualified,
}

fn no_claim(reason_code: &str) -> PromotionOutput {
    PromotionOutput::NoClaim {
        reason_code: reason_code.to_owned(),
    }
}

/// Evaluate one consumer claim over a VERIFIED chain.
///
/// The chain parameter is [`VerifiedV4Chain`], whose only constructor is
/// [`ArtifactStoreV4ChainStore::load_verified_chain`] — a consumer cannot
/// reach this evaluator with a path, a boolean, or a caller-supplied hash,
/// and authenticity is therefore never inferred from content hashes alone.
/// Authentication also never implies a win: every claim still walks the
/// admission, decision, coverage, terminal, durability, and authority
/// checks below.
///
/// # Errors
///
/// Returns [`GauntletError`] only for structurally malformed inputs (an
/// empty required-coverage set on a qualification claim, malformed
/// identities). POLICY failures are typed `NoClaim` outputs, not errors, so
/// callers cannot confuse a rejection with an infrastructure fault.
pub fn evaluate_consumer_claim(
    mode: ConsumerMode,
    claim: ConsumerClaim,
    chain: &VerifiedV4Chain,
    admitted_binding: &AdmittedChainBinding,
    admission: AdmissionState,
    decision: DecisionState,
    coverage: &CoverageWitnessSet,
    release_authority: Option<&ReleaseAuthority>,
    private_local_policy_satisfied: Option<bool>,
) -> Result<PromotionOutput, GauntletError> {
    if chain.authentication != AuthenticationClass::VerifiedReceiptChain {
        // IntegrityOnly / UnauthenticatedLegacy data may be preserved and
        // replayed, but no claim ladder starts below a verified chain.
        return Ok(no_claim("authentication_below_verified_receipt_chain"));
    }
    // Stale or substituted evidence: the chain must bind the admitted
    // source, build, and executable identities exactly.
    if chain.execution.source_identity_sha256 != admitted_binding.source_identity_sha256
        || chain.execution.predecessor_build_identity_sha256
            != admitted_binding.build_identity_sha256
        || chain.execution.executable_sha256 != admitted_binding.executable_sha256
    {
        return Ok(no_claim("chain_does_not_bind_admitted_identities"));
    }
    // PrivateLocal policy is consulted ONLY for PrivateLocal claims;
    // FixturePublic evidence can never be blocked by it.
    if mode == ConsumerMode::PrivateLocal && private_local_policy_satisfied != Some(true) {
        return Ok(no_claim("private_local_policy_not_satisfied"));
    }

    match claim {
        ConsumerClaim::DiagnosticAttempt => Ok(no_claim("diagnostic_attempt_is_never_a_claim")),
        ConsumerClaim::AdmittedCampaignEvidence => {
            if admission != AdmissionState::Admitted {
                return Ok(no_claim("evidence_not_admitted"));
            }
            if decision == DecisionState::Pass
                && chain.completion.terminal_outcome != TerminalOutcome::Succeeded
            {
                // A failed terminal completion cannot carry a Pass; the
                // authentic failure stays recordable as what it is.
                return Ok(no_claim("pass_requires_successful_terminal_completion"));
            }
            Ok(PromotionOutput::RecordedEvidence { decision })
        }
        ConsumerClaim::ReplacementQualified => {
            if coverage.required.is_empty() {
                return Err(invalid(
                    "a qualification claim must name its required coverage".to_owned(),
                ));
            }
            if admission != AdmissionState::Admitted {
                return Ok(no_claim("qualification_requires_admitted_evidence"));
            }
            if decision != DecisionState::Pass {
                return Ok(no_claim("qualification_requires_a_pass_decision"));
            }
            if chain.completion.terminal_outcome != TerminalOutcome::Succeeded {
                return Ok(no_claim("qualification_requires_successful_completion"));
            }
            if chain.completion.durability_sha256.is_none() {
                return Ok(no_claim("qualification_requires_durable_evidence"));
            }
            for label in &coverage.required {
                match coverage.witnessed.get(label) {
                    None => {
                        return Ok(no_claim("qualification_coverage_incomplete"));
                    }
                    Some((machine_profile, digest)) => {
                        if machine_profile != &chain.execution.machine_profile {
                            return Ok(no_claim("qualification_coverage_cross_worker_mix"));
                        }
                        validate_identity_hex(digest, "coverage witness digest")?;
                    }
                }
            }
            if release_authority.is_none() {
                return Ok(no_claim(
                    "qualification_requires_explicit_release_authority",
                ));
            }
            Ok(PromotionOutput::Qualified)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    pub(super) fn now_ns() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX))
            .unwrap_or(0)
    }

    pub(super) fn test_authority(seed: u8) -> (SupervisorSigningAuthority, SupervisorTrustRoots) {
        let authority = SupervisorSigningAuthority::from_seed([seed; 32]);
        let mut keys = BTreeMap::new();
        keys.insert(authority.key_id().to_owned(), authority.trust_root_key(0));
        (
            authority,
            SupervisorTrustRoots {
                schema_version: SUPERVISOR_AUTH_SCHEMA_VERSION,
                keys,
            },
        )
    }

    pub(super) fn sh_path() -> PathBuf {
        PathBuf::from("/bin/sh")
    }

    pub(super) fn sh_sha256() -> String {
        lower_hex(&Sha256::digest(
            std::fs::read(sh_path()).expect("read /bin/sh"),
        ))
    }

    pub(super) fn sh_spec(
        script: &str,
        window: Duration,
        max_output_bytes: u64,
    ) -> SupervisedLaunchSpec {
        SupervisedLaunchSpec {
            executable: sh_path(),
            args: vec!["-c".to_owned(), script.to_owned()],
            environment: BTreeMap::new(),
            run_window: window,
            max_output_bytes,
            expected_artifacts: BTreeMap::new(),
        }
    }

    pub(super) fn test_identity(tag: &str) -> String {
        lower_hex(&Sha256::digest(tag.as_bytes()))
    }

    pub(super) fn fixture_execution(
        authority: &mut SupervisorSigningAuthority,
        spec: &SupervisedLaunchSpec,
        executable_sha256: &str,
    ) -> ArtifactStoreV4ExecutionObject {
        let source = test_identity("source");
        let build = test_identity("build");
        let command = spec.command_digest();
        let issued_at_ns = now_ns();
        let (nonce_hex, expires_at_ns) = authority
            .issue_nonce(&source, &build, &command, issued_at_ns, 60_000_000_000)
            .expect("issue nonce");
        ArtifactStoreV4ExecutionObject {
            schema_version: SUPERVISOR_AUTH_SCHEMA_VERSION,
            predecessor_build_identity_sha256: build,
            source_identity_sha256: source,
            nonce_hex,
            nonce_issued_at_ns: issued_at_ns,
            nonce_expires_at_ns: expires_at_ns,
            command_digest_sha256: command,
            executable_sha256: executable_sha256.to_owned(),
            job_identity: "f2-acceptance".to_owned(),
            machine_profile: "test-host".to_owned(),
            fixture_identity: "fixture-a".to_owned(),
            query_contract_identity: "contract-a".to_owned(),
            environment_policy_sha256: spec.environment_policy_digest(),
            started_at_ns: issued_at_ns,
            run_window_ns: i64::try_from(spec.run_window.as_nanos()).unwrap_or(i64::MAX),
        }
    }

    pub(super) fn completion_from(
        execution: &ArtifactStoreV4ExecutionObject,
        termination: &SupervisedTermination,
    ) -> ArtifactStoreV4CompletionObject {
        let (_, execution_identity) = execution.identity().expect("execution identity");
        ArtifactStoreV4CompletionObject {
            schema_version: SUPERVISOR_AUTH_SCHEMA_VERSION,
            predecessor_execution_identity_sha256: execution_identity,
            nonce_hex: execution.nonce_hex.clone(),
            terminal_outcome: termination.terminal_outcome,
            outcome_reason_code: termination.outcome_reason_code.clone(),
            exit_code: termination.exit_code,
            termination_signal: termination.termination_signal,
            completed_at_ns: now_ns(),
            wall_clock_ns: i64::try_from(termination.wall_clock.as_nanos()).unwrap_or(i64::MAX),
            stdout: termination.stdout.clone(),
            stderr: termination.stderr.clone(),
            artifact_index: termination.artifact_index.clone(),
            process_tree_escape_detected: termination.process_tree_escape_detected,
            durability_label: "qg-durability-set".to_owned(),
            durability_sha256: Some(test_identity("durability")),
            retention_disposition: "retain-failure-evidence".to_owned(),
        }
    }

    fn sign_and_verify_chain(
        authority: &mut SupervisorSigningAuthority,
        trust_roots: &SupervisorTrustRoots,
        execution: &ArtifactStoreV4ExecutionObject,
        completion: &ArtifactStoreV4CompletionObject,
    ) -> Result<(), GauntletError> {
        let execution_receipt = authority.sign_execution(execution, now_ns())?;
        let completion_receipt = authority.sign_completion(execution, completion, now_ns())?;
        let (execution_bytes, _) = execution.identity()?;
        let (completion_bytes, _) = completion.identity()?;
        verify_execution_completion_chain(
            trust_roots,
            &execution_bytes,
            &execution_receipt,
            &completion_bytes,
            &completion_receipt,
        )
        .map(|_| ())
    }

    /// Launch success: a real child exits zero, the stream digest matches the
    /// exact bytes, and the full chain signs and verifies.
    #[test]
    fn f2_launch_success_yields_verified_signed_completion() {
        let (mut authority, trust_roots) = test_authority(1);
        let spec = sh_spec("printf out", Duration::from_secs(10), 1 << 20);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let cancel = SupervisionCancel::new();
        let termination =
            supervise_execution(&spec, &sh_sha256(), &cancel).expect("supervise child");
        assert_eq!(termination.terminal_outcome, TerminalOutcome::Succeeded);
        assert_eq!(termination.exit_code, Some(0));
        assert_eq!(
            termination.stdout.sha256,
            lower_hex(&Sha256::digest(b"out")),
            "stdout digest must cover the exact child bytes"
        );
        assert!(!termination.stdout.truncated);
        let completion = completion_from(&execution, &termination);
        sign_and_verify_chain(&mut authority, &trust_roots, &execution, &completion)
            .expect("full chain verifies");
    }

    /// Nonzero exit is authentic failure evidence: the chain signs and
    /// verifies with outcome `failed` — signing is outcome-neutral.
    #[test]
    fn f2_nonzero_exit_is_authenticated_failure_evidence() {
        let (mut authority, trust_roots) = test_authority(2);
        let spec = sh_spec("exit 3", Duration::from_secs(10), 1 << 20);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        assert_eq!(termination.terminal_outcome, TerminalOutcome::Failed);
        assert_eq!(termination.exit_code, Some(3));
        assert_eq!(termination.outcome_reason_code, "exited_nonzero");
        let completion = completion_from(&execution, &termination);
        sign_and_verify_chain(&mut authority, &trust_roots, &execution, &completion)
            .expect("failure evidence must authenticate exactly like success");
    }

    /// Signal death is observed and typed.
    #[test]
    fn f2_signal_termination_is_typed() {
        let (mut authority, trust_roots) = test_authority(3);
        let spec = sh_spec("kill -9 $$", Duration::from_secs(10), 1 << 20);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        assert_eq!(termination.terminal_outcome, TerminalOutcome::Failed);
        assert_eq!(termination.termination_signal, Some(9));
        assert_eq!(termination.outcome_reason_code, "terminated_by_signal");
        let completion = completion_from(&execution, &termination);
        sign_and_verify_chain(&mut authority, &trust_roots, &execution, &completion)
            .expect("signal death authenticates");
    }

    /// Supervisor-driven cancellation kills the child and types the outcome.
    #[test]
    fn f2_cancellation_terminates_and_types() {
        let (mut authority, trust_roots) = test_authority(4);
        let spec = sh_spec("sleep 30", Duration::from_secs(60), 1 << 20);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let cancel = SupervisionCancel::new();
        let canceller = cancel.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            canceller.cancel();
        });
        let termination = supervise_execution(&spec, &sh_sha256(), &cancel).expect("supervise");
        handle.join().expect("join canceller");
        assert_eq!(termination.terminal_outcome, TerminalOutcome::Cancelled);
        assert!(termination.wall_clock < Duration::from_secs(20));
        let completion = completion_from(&execution, &termination);
        sign_and_verify_chain(&mut authority, &trust_roots, &execution, &completion)
            .expect("cancellation authenticates");
    }

    /// Run-window timeout kills the child and types the outcome.
    #[test]
    fn f2_timeout_terminates_and_types() {
        let (mut authority, trust_roots) = test_authority(5);
        let spec = sh_spec("sleep 30", Duration::from_millis(300), 1 << 20);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        assert_eq!(termination.terminal_outcome, TerminalOutcome::TimedOut);
        assert!(termination.wall_clock < Duration::from_secs(20));
        let completion = completion_from(&execution, &termination);
        sign_and_verify_chain(&mut authority, &trust_roots, &execution, &completion)
            .expect("timeout authenticates");
    }

    /// Output beyond the bound truncates the digest, types the overflow, and
    /// never wedges the supervisor on a full pipe.
    #[test]
    fn f2_output_overflow_is_bounded_and_typed() {
        let (mut authority, trust_roots) = test_authority(6);
        let spec = sh_spec(
            "i=0; while [ $i -lt 5000 ]; do echo xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx; i=$((i+1)); done",
            Duration::from_secs(30),
            4096,
        );
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        assert!(termination.stdout.truncated);
        assert_eq!(termination.stdout.byte_len, 4096);
        assert_eq!(termination.outcome_reason_code, "output_overflow");
        assert_eq!(termination.terminal_outcome, TerminalOutcome::Failed);
        let completion = completion_from(&execution, &termination);
        sign_and_verify_chain(&mut authority, &trust_roots, &execution, &completion)
            .expect("overflow evidence authenticates");
    }

    /// A spawn that cannot exec is a typed launch failure, not a panic and
    /// not a silent absence.
    #[test]
    fn f2_exec_failure_is_typed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let not_executable = dir.path().join("not-executable");
        std::fs::write(&not_executable, b"#!/bin/sh\nexit 0\n").expect("write file");
        let expected = lower_hex(&Sha256::digest(b"#!/bin/sh\nexit 0\n"));
        let spec = SupervisedLaunchSpec {
            executable: not_executable,
            args: Vec::new(),
            environment: BTreeMap::new(),
            run_window: Duration::from_secs(5),
            max_output_bytes: 4096,
            expected_artifacts: BTreeMap::new(),
        };
        let termination =
            supervise_execution(&spec, &expected, &SupervisionCancel::new()).expect("supervise");
        assert_eq!(termination.terminal_outcome, TerminalOutcome::Failed);
        assert_eq!(termination.outcome_reason_code, "launch_failure_exec");
    }

    /// Source/ELF mismatch: the supervisor refuses to launch an executable
    /// whose bytes do not match the admitted digest.
    #[test]
    fn f2_executable_digest_mismatch_refuses_launch() {
        let spec = sh_spec("echo never-runs", Duration::from_secs(5), 4096);
        let wrong = test_identity("not-the-admitted-executable");
        let termination =
            supervise_execution(&spec, &wrong, &SupervisionCancel::new()).expect("supervise");
        assert_eq!(termination.terminal_outcome, TerminalOutcome::Failed);
        assert_eq!(
            termination.outcome_reason_code,
            "launch_refused_executable_digest_mismatch"
        );
        assert_eq!(termination.stdout.byte_len, 0, "nothing may have run");
    }

    /// A grandchild left behind in the process group is detected, recorded,
    /// and contained.
    #[test]
    fn f2_process_tree_escape_is_detected_and_recorded() {
        let (mut authority, trust_roots) = test_authority(7);
        let spec = sh_spec("sleep 30 & exit 0", Duration::from_secs(30), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        assert_eq!(termination.terminal_outcome, TerminalOutcome::Succeeded);
        assert!(
            termination.process_tree_escape_detected,
            "the surviving grandchild must be recorded"
        );
        assert!(
            termination.wall_clock < Duration::from_secs(10),
            "the sweep must also unblock the pipe readers instead of waiting \
             out the escapee's lifetime (observed: {:?})",
            termination.wall_clock
        );
        let completion = completion_from(&execution, &termination);
        sign_and_verify_chain(&mut authority, &trust_roots, &execution, &completion)
            .expect("escape evidence authenticates without being erased");
    }

    /// A missing expected artifact is reported; the supervisor signs a typed
    /// absence instead of a completion, and the consumed nonce then refuses
    /// any later completion for the same execution.
    #[test]
    fn f2_missing_artifact_yields_typed_absence_and_blocks_completion() {
        let (mut authority, trust_roots) = test_authority(8);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        spec.expected_artifacts.insert(
            "required-report".to_owned(),
            dir.path().join("never-written.json"),
        );
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        assert_eq!(
            termination.missing_artifacts,
            vec!["required-report".to_owned()]
        );
        let absence = authority
            .sign_absence(&execution, "missing_required_artifact", now_ns())
            .expect("sign typed absence");
        absence.verify(&trust_roots).expect("absence verifies");
        let completion = completion_from(&execution, &termination);
        let refused = authority.sign_completion(&execution, &completion, now_ns());
        assert!(
            refused.is_err(),
            "the absence consumed the nonce; no completion may follow"
        );
    }

    /// Supervisor interruption is a typed, verifiable absence.
    #[test]
    fn f2_supervisor_interruption_is_a_typed_absence() {
        let (mut authority, trust_roots) = test_authority(9);
        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let absence = authority
            .sign_absence(&execution, "supervisor_interrupted", now_ns())
            .expect("sign absence");
        absence.verify(&trust_roots).expect("absence verifies");
        // Never promotable: an absence names no terminal outcome at all —
        // there is structurally nothing for an admission policy to admit.
        let as_bytes = canonical_json_bytes(&absence).expect("canonical absence");
        assert!(
            !String::from_utf8_lossy(&as_bytes).contains("terminal_outcome"),
            "an absence carries no terminal outcome"
        );
    }

    /// Nonce replay: a second completion for the same nonce is refused
    /// permanently.
    #[test]
    fn f2_nonce_replay_is_refused() {
        let (mut authority, trust_roots) = test_authority(10);
        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        let completion = completion_from(&execution, &termination);
        sign_and_verify_chain(&mut authority, &trust_roots, &execution, &completion)
            .expect("first completion verifies");
        let replay = authority.sign_completion(&execution, &completion, now_ns());
        let reason = format!("{:?}", replay.expect_err("replay must be refused"));
        assert!(
            reason.contains("nonce reuse"),
            "refusal names the replay: {reason}"
        );
    }

    /// An expired nonce may not start an execution, and a lifetime beyond 15
    /// minutes may not be issued at all.
    #[test]
    fn f2_expired_and_overlong_nonces_fail_closed() {
        let (mut authority, _) = test_authority(11);
        let overlong = authority.issue_nonce(
            &test_identity("source"),
            &test_identity("build"),
            &test_identity("command"),
            now_ns(),
            MAX_NONCE_LIFETIME_NS + 1,
        );
        assert!(overlong.is_err(), "15 minutes is the ceiling");

        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let mut execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        execution.started_at_ns = execution.nonce_expires_at_ns + 1;
        let refused = authority.sign_execution(&execution, now_ns());
        let reason = format!("{:?}", refused.expect_err("expired nonce must be refused"));
        assert!(reason.contains("expired"), "refusal names expiry: {reason}");
    }

    /// A nonce is bound to its source/build/command; presenting it with a
    /// different build is cross-build reuse and fails closed.
    #[test]
    fn f2_cross_build_nonce_reuse_fails_closed() {
        let (mut authority, _) = test_authority(12);
        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let mut execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        execution.predecessor_build_identity_sha256 = test_identity("some-other-build");
        let refused = authority.sign_execution(&execution, now_ns());
        let reason = format!(
            "{:?}",
            refused.expect_err("cross-build reuse must be refused")
        );
        assert!(reason.contains("cross-build"), "refusal is typed: {reason}");
    }

    /// Wrong signer: a key outside the trust roots — including one the CHILD
    /// could have generated for itself — verifies nothing. The child cannot
    /// self-certify.
    #[test]
    fn f2_wrong_signer_and_child_held_keys_are_rejected() {
        let (mut supervisor_authority, trust_roots) = test_authority(13);
        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut supervisor_authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        let completion = completion_from(&execution, &termination);

        // The "child" mints its own key and self-certifies its completion.
        let mut child_held = SupervisorSigningAuthority::from_seed([99; 32]);
        let (_, _expires) = child_held
            .issue_nonce(
                &execution.source_identity_sha256,
                &execution.predecessor_build_identity_sha256,
                &execution.command_digest_sha256,
                now_ns(),
                60_000_000_000,
            )
            .expect("child issues its own nonce");
        let (_, completion_identity) = completion.identity().expect("identity");
        let forged = child_held
            .sign_identity(
                "completion",
                &completion_identity,
                COMPLETION_ISSUER_ROLE,
                now_ns(),
            )
            .expect("child signs with its own key");
        let rejected = forged.verify(
            &trust_roots,
            "completion",
            COMPLETION_ISSUER_ROLE,
            &completion_identity,
        );
        let reason = format!(
            "{:?}",
            rejected.expect_err("child-held key must be rejected")
        );
        assert!(
            reason.contains("unknown signer"),
            "typed rejection: {reason}"
        );
    }

    /// Revocation and retirement: a revoked key verifies nothing; a retired
    /// key verifies receipts signed before retirement and rejects ones
    /// issued after.
    #[test]
    fn f2_revoked_and_retired_keys_enforce_issuing_windows() {
        let (mut authority, mut trust_roots) = test_authority(14);
        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let signed_at = now_ns();
        let receipt = authority
            .sign_execution(&execution, signed_at)
            .expect("sign execution");
        let (_, identity) = execution.identity().expect("identity");

        receipt
            .verify(
                &trust_roots,
                "execution",
                EXECUTION_SUPERVISOR_ROLE,
                &identity,
            )
            .expect("live key verifies");

        let key_id = authority.key_id().to_owned();
        trust_roots
            .keys
            .get_mut(&key_id)
            .expect("key present")
            .retired_after_ns = Some(signed_at + 1);
        receipt
            .verify(
                &trust_roots,
                "execution",
                EXECUTION_SUPERVISOR_ROLE,
                &identity,
            )
            .expect("retired key still verifies receipts issued before retirement");

        trust_roots
            .keys
            .get_mut(&key_id)
            .expect("key present")
            .retired_after_ns = Some(signed_at - 1);
        assert!(
            receipt
                .verify(
                    &trust_roots,
                    "execution",
                    EXECUTION_SUPERVISOR_ROLE,
                    &identity,
                )
                .is_err(),
            "a receipt issued after retirement is rejected"
        );

        trust_roots
            .keys
            .get_mut(&key_id)
            .expect("key present")
            .revoked = true;
        assert!(
            receipt
                .verify(
                    &trust_roots,
                    "execution",
                    EXECUTION_SUPERVISOR_ROLE,
                    &identity,
                )
                .is_err(),
            "a revoked key verifies nothing"
        );
    }

    /// One-bit tamper of the signed completion bytes is rejected: either the
    /// bytes are no longer canonical or the identity no longer matches the
    /// receipt.
    #[test]
    fn f2_tampered_completion_bytes_are_rejected() {
        let (mut authority, trust_roots) = test_authority(15);
        let spec = sh_spec("printf log-line", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        let completion = completion_from(&execution, &termination);
        let execution_receipt = authority
            .sign_execution(&execution, now_ns())
            .expect("sign execution");
        let completion_receipt = authority
            .sign_completion(&execution, &completion, now_ns())
            .expect("sign completion");
        let (execution_bytes, _) = execution.identity().expect("bytes");
        let (completion_bytes, _) = completion.identity().expect("bytes");

        // Flip one hex nibble inside the stored stdout digest (tampered log).
        let marker = format!("\"sha256\":\"{}\"", completion.stdout.sha256);
        let serialized = String::from_utf8(completion_bytes.clone()).expect("utf8");
        let position = serialized.find(&marker).expect("stdout digest present") + 10;
        let mut tampered = completion_bytes.clone();
        tampered[position] = if tampered[position] == b'a' {
            b'b'
        } else {
            b'a'
        };

        verify_execution_completion_chain(
            &trust_roots,
            &execution_bytes,
            &execution_receipt,
            &completion_bytes,
            &completion_receipt,
        )
        .expect("untampered chain verifies");
        assert!(
            verify_execution_completion_chain(
                &trust_roots,
                &execution_bytes,
                &execution_receipt,
                &tampered,
                &completion_receipt,
            )
            .is_err(),
            "a one-bit log tamper must be rejected"
        );
    }

    /// Completion substitution across executions is rejected by predecessor
    /// identity, even with two authentically signed chains.
    #[test]
    fn f2_cross_run_completion_substitution_is_rejected() {
        let (mut authority, trust_roots) = test_authority(16);
        let spec = sh_spec("printf run-a", Duration::from_secs(10), 4096);
        let other_spec = sh_spec("printf run-b", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let other_execution = fixture_execution(&mut authority, &other_spec, &sh_sha256());
        let termination = supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new())
            .expect("supervise the first run");
        let completion = completion_from(&execution, &termination);
        let execution_receipt = authority
            .sign_execution(&execution, now_ns())
            .expect("sign the first execution");
        let other_receipt = authority
            .sign_execution(&other_execution, now_ns())
            .expect("sign the other execution");
        let completion_receipt = authority
            .sign_completion(&execution, &completion, now_ns())
            .expect("sign the first completion");
        let (execution_bytes, _) = execution.identity().expect("bytes");
        let (other_bytes, _) = other_execution.identity().expect("bytes");
        let (completion_bytes, _) = completion.identity().expect("bytes");

        verify_execution_completion_chain(
            &trust_roots,
            &execution_bytes,
            &execution_receipt,
            &completion_bytes,
            &completion_receipt,
        )
        .expect("matched chain verifies");
        assert!(
            verify_execution_completion_chain(
                &trust_roots,
                &other_bytes,
                &other_receipt,
                &completion_bytes,
                &completion_receipt,
            )
            .is_err(),
            "a completion presented against a different execution must be rejected"
        );
    }

    /// Partial durability is a typed null, preserved through signing and
    /// verification — never silently upgraded.
    #[test]
    fn f2_partial_durability_survives_as_typed_null() {
        let (mut authority, trust_roots) = test_authority(17);
        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        let mut completion = completion_from(&execution, &termination);
        completion.durability_sha256 = None;
        let execution_receipt = authority
            .sign_execution(&execution, now_ns())
            .expect("sign execution");
        let completion_receipt = authority
            .sign_completion(&execution, &completion, now_ns())
            .expect("sign completion");
        let (execution_bytes, _) = execution.identity().expect("bytes");
        let (completion_bytes, _) = completion.identity().expect("bytes");
        assert!(
            String::from_utf8_lossy(&completion_bytes).contains("\"durability_sha256\":null"),
            "the null is present in canonical bytes, not omitted"
        );
        let (_, verified) = verify_execution_completion_chain(
            &trust_roots,
            &execution_bytes,
            &execution_receipt,
            &completion_bytes,
            &completion_receipt,
        )
        .expect("chain verifies");
        assert_eq!(verified.durability_sha256, None);
    }

    /// Unknown schema versions, unknown fields, unknown enum values, and
    /// non-canonical bytes all fail closed at decode or verify time.
    #[test]
    fn f2_unknown_and_non_canonical_inputs_fail_closed() {
        let (mut authority, trust_roots) = test_authority(18);
        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let (execution_bytes, identity) = execution.identity().expect("bytes");
        let receipt = authority
            .sign_execution(&execution, now_ns())
            .expect("sign execution");

        // Non-canonical spacing fails byte-equality.
        let mut spaced = execution_bytes.clone();
        spaced.insert(1, b' ');
        assert!(
            decode_canonical::<ArtifactStoreV4ExecutionObject>(&spaced).is_err(),
            "non-canonical whitespace fails closed"
        );

        // Unknown field fails closed.
        let with_unknown = String::from_utf8(execution_bytes.clone())
            .expect("utf8")
            .replacen('{', "{\"unknown_field\":1,", 1);
        assert!(
            decode_canonical::<ArtifactStoreV4ExecutionObject>(with_unknown.as_bytes()).is_err(),
            "unknown fields fail closed"
        );

        // Unknown terminal outcome fails closed.
        let bogus_outcome = br#"{"terminal_outcome":"promoted"}"#;
        assert!(
            serde_json::from_slice::<ArtifactStoreV4CompletionObject>(bogus_outcome).is_err(),
            "unknown enum values fail closed"
        );

        // Unknown object kind on an otherwise valid receipt fails closed.
        let mut wrong_kind = receipt;
        wrong_kind.object_kind = "sourcery".to_owned();
        assert!(
            wrong_kind
                .verify(
                    &trust_roots,
                    "execution",
                    EXECUTION_SUPERVISOR_ROLE,
                    &identity,
                )
                .is_err(),
            "an unexpected object kind fails closed"
        );
    }

    /// The environment allowlist is enforced: only admitted variables reach
    /// the child.
    #[test]
    fn f2_environment_allowlist_is_enforced() {
        let (mut authority, trust_roots) = test_authority(19);
        let mut spec = sh_spec(
            "printf \"%s|%s\" \"$F2_ALLOWED\" \"$HOME\"",
            Duration::from_secs(10),
            4096,
        );
        spec.environment
            .insert("F2_ALLOWED".to_owned(), "yes".to_owned());
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        assert_eq!(
            termination.stdout.sha256,
            lower_hex(&Sha256::digest(b"yes|")),
            "the child sees the allowlisted variable and nothing else"
        );
        let completion = completion_from(&execution, &termination);
        sign_and_verify_chain(&mut authority, &trust_roots, &execution, &completion)
            .expect("chain verifies");
    }
}

#[cfg(test)]
mod f3_store_tests {
    use super::tests::{
        completion_from, fixture_execution, now_ns, sh_sha256, sh_spec, test_authority,
        test_identity,
    };
    use super::*;
    use std::time::Duration;

    struct PublishedChain {
        store: ArtifactStoreV4ChainStore,
        trust_roots: SupervisorTrustRoots,
        chain_identity: String,
        execution: ArtifactStoreV4ExecutionObject,
        completion: ArtifactStoreV4CompletionObject,
        _root: tempfile::TempDir,
    }

    fn publish_fixture_chain(seed: u8) -> PublishedChain {
        let (mut authority, trust_roots) = test_authority(seed);
        let spec = sh_spec("printf f3", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        let completion = completion_from(&execution, &termination);
        let execution_receipt = authority
            .sign_execution(&execution, now_ns())
            .expect("sign execution");
        let completion_receipt = authority
            .sign_completion(&execution, &completion, now_ns())
            .expect("sign completion");
        let root = tempfile::tempdir().expect("store root");
        let store = ArtifactStoreV4ChainStore::open(root.path()).expect("open store");
        let chain_identity = store
            .publish_verified_chain(
                &trust_roots,
                &execution,
                &execution_receipt,
                &completion,
                &completion_receipt,
            )
            .expect("publish verified chain");
        PublishedChain {
            store,
            trust_roots,
            chain_identity,
            execution,
            completion,
            _root: root,
        }
    }

    /// Round-trip: publish -> verified reload reproduces the exact objects
    /// with the earned `VerifiedReceiptChain` classification, and re-publishing
    /// the identical chain is idempotent.
    #[test]
    fn f3_chain_round_trip_and_idempotent_republish() {
        let published = publish_fixture_chain(30);
        let loaded = published
            .store
            .load_verified_chain(&published.trust_roots, &published.chain_identity)
            .expect("verified reload");
        assert_eq!(
            loaded.authentication,
            AuthenticationClass::VerifiedReceiptChain
        );
        assert_eq!(loaded.execution, published.execution);
        assert_eq!(loaded.completion, published.completion);
        assert_eq!(
            loaded.index.terminal_outcome,
            published.completion.terminal_outcome
        );
    }

    /// Golden canonical bytes: the wire form is hand-verifiable sorted-key
    /// JSON with exact integers and null-present optional fields.
    #[test]
    fn f3_golden_canonical_completion_bytes() {
        let empty = lower_hex(&Sha256::digest([]));
        let predecessor = test_identity("golden-execution");
        let durability = test_identity("golden-durability");
        let completion = ArtifactStoreV4CompletionObject {
            schema_version: SUPERVISOR_AUTH_SCHEMA_VERSION,
            predecessor_execution_identity_sha256: predecessor.clone(),
            nonce_hex: "000102030405060708090a0b0c0d0e0f".to_owned(),
            terminal_outcome: TerminalOutcome::Succeeded,
            outcome_reason_code: "exited_zero".to_owned(),
            exit_code: Some(0),
            termination_signal: None,
            completed_at_ns: 1_700_000_000_000_000_000,
            wall_clock_ns: 5_000_000,
            stdout: CollectedArtifactDigest {
                sha256: empty.clone(),
                byte_len: 0,
                truncated: false,
            },
            stderr: CollectedArtifactDigest {
                sha256: empty.clone(),
                byte_len: 0,
                truncated: false,
            },
            artifact_index: BTreeMap::new(),
            process_tree_escape_detected: false,
            durability_label: "golden-set".to_owned(),
            durability_sha256: Some(durability.clone()),
            retention_disposition: "retain".to_owned(),
        };
        let (bytes, _) = completion.identity().expect("canonical bytes");
        let expected = format!(
            "{{\"artifact_index\":{{}},\"completed_at_ns\":1700000000000000000,\
             \"durability_label\":\"golden-set\",\"durability_sha256\":\"{durability}\",\
             \"exit_code\":0,\"nonce_hex\":\"000102030405060708090a0b0c0d0e0f\",\
             \"outcome_reason_code\":\"exited_zero\",\
             \"predecessor_execution_identity_sha256\":\"{predecessor}\",\
             \"process_tree_escape_detected\":false,\"retention_disposition\":\"retain\",\
             \"schema_version\":1,\
             \"stderr\":{{\"byte_len\":0,\"sha256\":\"{empty}\",\"truncated\":false}},\
             \"stdout\":{{\"byte_len\":0,\"sha256\":\"{empty}\",\"truncated\":false}},\
             \"terminal_outcome\":\"succeeded\",\"termination_signal\":null,\
             \"wall_clock_ns\":5000000}}"
        );
        assert_eq!(
            String::from_utf8(bytes).expect("utf8"),
            expected,
            "canonical form is the hand-verifiable sorted-key golden"
        );
    }

    /// Truncated, extended, and tampered stored bytes all fail the content
    /// address at verified reload.
    #[test]
    fn f3_truncated_extended_and_tampered_entries_reject() {
        let published = publish_fixture_chain(31);
        let object_path = published
            .store
            .child_dir("objects")
            .join(published.index_execution_identity());
        let original = std::fs::read(&object_path).expect("read stored object");

        std::fs::write(&object_path, &original[..original.len() - 1]).expect("truncate");
        assert!(
            published
                .store
                .load_verified_chain(&published.trust_roots, &published.chain_identity)
                .is_err(),
            "a truncated object must reject"
        );

        let mut extended = original.clone();
        extended.push(b' ');
        std::fs::write(&object_path, &extended).expect("extend");
        assert!(
            published
                .store
                .load_verified_chain(&published.trust_roots, &published.chain_identity)
                .is_err(),
            "an extended object must reject"
        );

        let mut tampered = original.clone();
        let flip = tampered.len() / 2;
        tampered[flip] = tampered[flip].wrapping_add(1);
        std::fs::write(&object_path, &tampered).expect("tamper");
        assert!(
            published
                .store
                .load_verified_chain(&published.trust_roots, &published.chain_identity)
                .is_err(),
            "a tampered object must reject"
        );

        std::fs::write(&object_path, &original).expect("restore");
        published
            .store
            .load_verified_chain(&published.trust_roots, &published.chain_identity)
            .expect("restored object verifies again");
    }

    /// A missing receipt, a cross-object receipt substitution, and an
    /// untrusted verifier all fail closed.
    #[test]
    fn f3_missing_and_cross_object_receipts_and_wrong_roots_reject() {
        let published = publish_fixture_chain(32);
        let receipts = published.store.child_dir("receipts");
        let execution_receipt_path = receipts.join(published.index_execution_identity());
        let completion_receipt_path = receipts.join(published.index_completion_identity());
        let completion_receipt_bytes =
            std::fs::read(&completion_receipt_path).expect("read completion receipt");

        // Cross-object substitution: the execution receipt where the
        // completion receipt belongs.
        let execution_receipt_bytes =
            std::fs::read(&execution_receipt_path).expect("read execution receipt");
        std::fs::write(&completion_receipt_path, &execution_receipt_bytes).expect("substitute");
        assert!(
            published
                .store
                .load_verified_chain(&published.trust_roots, &published.chain_identity)
                .is_err(),
            "a cross-object receipt must reject"
        );
        std::fs::write(&completion_receipt_path, &completion_receipt_bytes).expect("restore");

        // Untrusted roots: a verifier with no keys accepts nothing.
        let empty_roots = SupervisorTrustRoots {
            schema_version: SUPERVISOR_AUTH_SCHEMA_VERSION,
            keys: BTreeMap::new(),
        };
        assert!(
            published
                .store
                .load_verified_chain(&empty_roots, &published.chain_identity)
                .is_err(),
            "a chain under unknown keys must reject"
        );

        // Missing receipt file: simulate a partial publication by renaming
        // the receipt aside (the pending-file crash state).
        let aside = receipts.join(format!(
            ".{}.pending",
            published.index_completion_identity()
        ));
        std::fs::rename(&completion_receipt_path, &aside).expect("rename receipt aside");
        assert!(
            published
                .store
                .load_verified_chain(&published.trust_roots, &published.chain_identity)
                .is_err(),
            "a partially written chain (missing receipt) must reject"
        );
    }

    /// A content-address collision (different bytes at the same identity) is
    /// a typed quarantine, and a leftover pending file fails publication
    /// closed instead of being silently overwritten.
    #[test]
    fn f3_collision_and_stale_pending_fail_closed() {
        let published = publish_fixture_chain(33);
        let name = published.index_execution_identity();

        let collision = published.store.publish_atomic(
            "objects",
            &name,
            b"different bytes at the same address",
        );
        let reason = format!("{:?}", collision.expect_err("collision must be typed"));
        assert!(reason.contains("collision"), "typed collision: {reason}");

        let fresh = test_identity("fresh-address");
        let pending = published
            .store
            .child_dir("objects")
            .join(format!(".{fresh}.pending"));
        std::fs::write(&pending, b"crash leftover").expect("plant stale pending");
        let refused = published.store.publish_atomic("objects", &fresh, b"bytes");
        let reason = format!("{:?}", refused.expect_err("stale pending must fail closed"));
        assert!(
            reason.contains("pending"),
            "typed pending refusal: {reason}"
        );
    }

    /// The chain index is the commit point: without it the chain does not
    /// load, and a tampered index fails its own content address.
    #[test]
    fn f3_index_is_the_commit_point() {
        let published = publish_fixture_chain(34);
        let chain_path = published
            .store
            .child_dir("chains")
            .join(&published.chain_identity);
        let index_bytes = std::fs::read(&chain_path).expect("read index");

        let mut tampered = index_bytes.clone();
        let flip = tampered.len() / 2;
        tampered[flip] = tampered[flip].wrapping_add(1);
        std::fs::write(&chain_path, &tampered).expect("tamper index");
        assert!(
            published
                .store
                .load_verified_chain(&published.trust_roots, &published.chain_identity)
                .is_err(),
            "a tampered index must fail its content address"
        );

        let aside = published
            .store
            .child_dir("chains")
            .join(format!(".{}.pending", published.chain_identity));
        std::fs::rename(&chain_path, &aside).expect("uncommit index");
        assert!(
            published
                .store
                .load_verified_chain(&published.trust_roots, &published.chain_identity)
                .is_err(),
            "objects without a committed index are not a loadable chain"
        );
    }

    /// A verified absence publishes and reloads; it never yields a chain.
    #[test]
    fn f3_absence_publishes_without_a_chain() {
        let (mut authority, trust_roots) = test_authority(35);
        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let execution_receipt = authority
            .sign_execution(&execution, now_ns())
            .expect("sign execution");
        let absence = authority
            .sign_absence(&execution, "supervisor_interrupted", now_ns())
            .expect("sign absence");
        let root = tempfile::tempdir().expect("store root");
        let store = ArtifactStoreV4ChainStore::open(root.path()).expect("open store");
        store
            .publish_verified_absence(&trust_roots, &execution, &execution_receipt, &absence)
            .expect("publish absence");
        let (_, execution_identity) = execution.identity().expect("identity");
        let stored = store
            .read_entry("absences", &execution_identity)
            .expect("read stored absence");
        let reloaded: CompletionAbsenceV4 = decode_canonical(&stored).expect("decode absence");
        reloaded.verify(&trust_roots).expect("absence verifies");
        assert!(
            std::fs::read_dir(store.child_dir("chains"))
                .expect("chains dir")
                .next()
                .is_none(),
            "an absence never commits a chain index"
        );
    }

    /// The pre-policy canary: every v4 chain artifact fed to a LEGACY
    /// classifier is rejected with a typed reason instead of classifying as
    /// `UnauthenticatedLegacy` — while genuine legacy fixtures still classify
    /// legacy (replayable, release-ineligible) exactly as before.
    #[test]
    fn f3_v4_canary_blocks_legacy_fallthrough_without_breaking_legacy() {
        let published = publish_fixture_chain(36);
        let (execution_bytes, _) = published.execution.identity().expect("bytes");
        let (completion_bytes, _) = published.completion.identity().expect("bytes");
        let index_bytes = std::fs::read(
            published
                .store
                .child_dir("chains")
                .join(&published.chain_identity),
        )
        .expect("read index");
        let receipt_bytes = std::fs::read(
            published
                .store
                .child_dir("receipts")
                .join(published.index_execution_identity()),
        )
        .expect("read receipt");

        for (label, bytes) in [
            ("execution object", execution_bytes.as_slice()),
            ("completion object", completion_bytes.as_slice()),
            ("chain index", index_bytes.as_slice()),
            ("signed receipt", receipt_bytes.as_slice()),
        ] {
            assert!(
                v4_chain_shape_rejection(bytes).is_some(),
                "{label} must trip the canary"
            );
            let campaign = crate::classify_campaign_report_schema(bytes);
            let reason = format!(
                "{:?}",
                campaign.expect_err("legacy campaign classifier must reject v4 bytes")
            );
            assert!(
                reason.contains("v4 chain-shaped"),
                "typed canary rejection for {label}: {reason}"
            );
            assert!(
                crate::classify_artifact_object_schema(bytes).is_err(),
                "legacy object classifier must reject v4 bytes ({label})"
            );
        }

        // Genuine legacy fixtures still classify as frozen
        // UnauthenticatedLegacy — replayable, never current, never
        // release-eligible.
        let legacy = br#"{"schema_version":1,"anything":"else"}"#;
        assert!(v4_chain_shape_rejection(legacy).is_none());
        match crate::classify_campaign_report_schema(legacy).expect("legacy classifies") {
            crate::SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version } => {
                assert_eq!(schema_version, 1);
            }
            other => panic!("legacy fixture must stay UnauthenticatedLegacy, got {other:?}"),
        }
    }

    impl PublishedChain {
        fn index_execution_identity(&self) -> String {
            let (_, identity) = self.execution.identity().expect("identity");
            identity
        }

        fn index_completion_identity(&self) -> String {
            let (_, identity) = self.completion.identity().expect("identity");
            identity
        }
    }
}

#[cfg(test)]
mod f4_consumer_policy_tests {
    use super::tests::{
        completion_from, fixture_execution, now_ns, sh_sha256, sh_spec, test_authority,
        test_identity,
    };
    use super::*;
    use std::time::Duration;

    struct VerifiedFixture {
        chain: VerifiedV4Chain,
        binding: AdmittedChainBinding,
        _root: tempfile::TempDir,
    }

    /// Mock-free E2E prelude: REAL supervised child -> signed chain ->
    /// atomic store publication -> verified reload. Everything downstream
    /// consumes the reloaded chain, exactly as a real consumer would.
    fn verified_fixture(seed: u8, script: &str) -> VerifiedFixture {
        let (mut authority, trust_roots) = test_authority(seed);
        let spec = sh_spec(script, Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        let completion = completion_from(&execution, &termination);
        let execution_receipt = authority
            .sign_execution(&execution, now_ns())
            .expect("sign execution");
        let completion_receipt = authority
            .sign_completion(&execution, &completion, now_ns())
            .expect("sign completion");
        let root = tempfile::tempdir().expect("store root");
        let store = ArtifactStoreV4ChainStore::open(root.path()).expect("open store");
        let chain_identity = store
            .publish_verified_chain(
                &trust_roots,
                &execution,
                &execution_receipt,
                &completion,
                &completion_receipt,
            )
            .expect("publish");
        let chain = store
            .load_verified_chain(&trust_roots, &chain_identity)
            .expect("verified reload");
        let binding = AdmittedChainBinding {
            source_identity_sha256: chain.execution.source_identity_sha256.clone(),
            build_identity_sha256: chain.execution.predecessor_build_identity_sha256.clone(),
            executable_sha256: chain.execution.executable_sha256.clone(),
        };
        VerifiedFixture {
            chain,
            binding,
            _root: root,
        }
    }

    fn complete_coverage(chain: &VerifiedV4Chain) -> CoverageWitnessSet {
        let mut required = std::collections::BTreeSet::new();
        required.insert("latency".to_owned());
        required.insert("quality".to_owned());
        let mut witnessed = BTreeMap::new();
        for label in &required {
            witnessed.insert(
                label.clone(),
                (
                    chain.execution.machine_profile.clone(),
                    test_identity(label),
                ),
            );
        }
        CoverageWitnessSet {
            required,
            witnessed,
        }
    }

    fn release() -> ReleaseAuthority {
        ReleaseAuthority {
            authority_label: "release-policy-v1".to_owned(),
        }
    }

    /// DiagnosticAttempt is always NoClaim — even for a fully authentic,
    /// admitted, passing, covered, released chain.
    #[test]
    fn f4_diagnostic_attempt_is_always_no_claim() {
        let fixture = verified_fixture(50, "exit 0");
        let coverage = complete_coverage(&fixture.chain);
        let output = evaluate_consumer_claim(
            ConsumerMode::FixturePublic,
            ConsumerClaim::DiagnosticAttempt,
            &fixture.chain,
            &fixture.binding,
            AdmissionState::Admitted,
            DecisionState::Pass,
            &coverage,
            Some(&release()),
            None,
        )
        .expect("evaluate");
        assert_eq!(
            output,
            PromotionOutput::NoClaim {
                reason_code: "diagnostic_attempt_is_never_a_claim".to_owned()
            }
        );
    }

    /// AdmittedCampaignEvidence records Pass AND Miss exactly as evaluated;
    /// a Miss on an authentic chain stays a Miss, never relabeled.
    #[test]
    fn f4_admitted_evidence_records_pass_and_miss_without_relabeling() {
        let fixture = verified_fixture(51, "exit 0");
        let coverage = complete_coverage(&fixture.chain);
        for decision in [DecisionState::Pass, DecisionState::Miss] {
            let output = evaluate_consumer_claim(
                ConsumerMode::FixturePublic,
                ConsumerClaim::AdmittedCampaignEvidence,
                &fixture.chain,
                &fixture.binding,
                AdmissionState::Admitted,
                decision,
                &coverage,
                None,
                None,
            )
            .expect("evaluate");
            assert_eq!(output, PromotionOutput::RecordedEvidence { decision });
        }
    }

    /// The full replacement ladder qualifies — and every single missing rung
    /// drops it to a typed NoClaim.
    #[test]
    fn f4_replacement_qualification_ladder_and_rejection_matrix() {
        let fixture = verified_fixture(52, "exit 0");
        let coverage = complete_coverage(&fixture.chain);
        let qualify = |admission: AdmissionState,
                       decision: DecisionState,
                       coverage: &CoverageWitnessSet,
                       authority: Option<&ReleaseAuthority>,
                       binding: &AdmittedChainBinding| {
            evaluate_consumer_claim(
                ConsumerMode::FixturePublic,
                ConsumerClaim::ReplacementQualified,
                &fixture.chain,
                binding,
                admission,
                decision,
                coverage,
                authority,
                None,
            )
            .expect("evaluate")
        };

        assert_eq!(
            qualify(
                AdmissionState::Admitted,
                DecisionState::Pass,
                &coverage,
                Some(&release()),
                &fixture.binding,
            ),
            PromotionOutput::Qualified
        );

        for admission in [AdmissionState::Unadmitted, AdmissionState::NoDecision] {
            assert_eq!(
                qualify(
                    admission,
                    DecisionState::Pass,
                    &coverage,
                    Some(&release()),
                    &fixture.binding,
                ),
                PromotionOutput::NoClaim {
                    reason_code: "qualification_requires_admitted_evidence".to_owned()
                }
            );
        }
        for decision in [
            DecisionState::Miss,
            DecisionState::NoDecision,
            DecisionState::Quarantine,
        ] {
            assert_eq!(
                qualify(
                    AdmissionState::Admitted,
                    decision,
                    &coverage,
                    Some(&release()),
                    &fixture.binding,
                ),
                PromotionOutput::NoClaim {
                    reason_code: "qualification_requires_a_pass_decision".to_owned()
                }
            );
        }

        // Missing coverage witness.
        let mut incomplete = coverage.clone();
        incomplete.witnessed.remove("latency");
        assert_eq!(
            qualify(
                AdmissionState::Admitted,
                DecisionState::Pass,
                &incomplete,
                Some(&release()),
                &fixture.binding,
            ),
            PromotionOutput::NoClaim {
                reason_code: "qualification_coverage_incomplete".to_owned()
            }
        );

        // Cross-worker mix: one witness from a different machine profile.
        let mut mixed = coverage.clone();
        mixed.witnessed.insert(
            "latency".to_owned(),
            ("some-other-host".to_owned(), test_identity("latency")),
        );
        assert_eq!(
            qualify(
                AdmissionState::Admitted,
                DecisionState::Pass,
                &mixed,
                Some(&release()),
                &fixture.binding,
            ),
            PromotionOutput::NoClaim {
                reason_code: "qualification_coverage_cross_worker_mix".to_owned()
            }
        );

        // Missing release authority.
        assert_eq!(
            qualify(
                AdmissionState::Admitted,
                DecisionState::Pass,
                &coverage,
                None,
                &fixture.binding,
            ),
            PromotionOutput::NoClaim {
                reason_code: "qualification_requires_explicit_release_authority".to_owned()
            }
        );

        // Stale source/ELF: the chain does not bind the admitted identities.
        let stale = AdmittedChainBinding {
            executable_sha256: test_identity("some-other-executable"),
            ..fixture.binding.clone()
        };
        assert_eq!(
            qualify(
                AdmissionState::Admitted,
                DecisionState::Pass,
                &coverage,
                Some(&release()),
                &stale,
            ),
            PromotionOutput::NoClaim {
                reason_code: "chain_does_not_bind_admitted_identities".to_owned()
            }
        );

        // An empty required set is a malformed claim, not a passing one.
        let empty = CoverageWitnessSet {
            required: std::collections::BTreeSet::new(),
            witnessed: BTreeMap::new(),
        };
        assert!(
            evaluate_consumer_claim(
                ConsumerMode::FixturePublic,
                ConsumerClaim::ReplacementQualified,
                &fixture.chain,
                &fixture.binding,
                AdmissionState::Admitted,
                DecisionState::Pass,
                &empty,
                Some(&release()),
                None,
            )
            .is_err(),
            "an empty required-coverage set is structurally malformed"
        );
    }

    /// A failed terminal completion can never carry a Pass — but the
    /// authentic failure remains recordable as a Miss, and can never
    /// qualify.
    #[test]
    fn f4_failed_terminal_completion_rejects_pass_and_qualification() {
        let fixture = verified_fixture(53, "exit 7");
        let coverage = complete_coverage(&fixture.chain);
        assert_eq!(
            fixture.chain.completion.terminal_outcome,
            TerminalOutcome::Failed
        );
        let pass_attempt = evaluate_consumer_claim(
            ConsumerMode::FixturePublic,
            ConsumerClaim::AdmittedCampaignEvidence,
            &fixture.chain,
            &fixture.binding,
            AdmissionState::Admitted,
            DecisionState::Pass,
            &coverage,
            None,
            None,
        )
        .expect("evaluate");
        assert_eq!(
            pass_attempt,
            PromotionOutput::NoClaim {
                reason_code: "pass_requires_successful_terminal_completion".to_owned()
            }
        );
        let miss = evaluate_consumer_claim(
            ConsumerMode::FixturePublic,
            ConsumerClaim::AdmittedCampaignEvidence,
            &fixture.chain,
            &fixture.binding,
            AdmissionState::Admitted,
            DecisionState::Miss,
            &coverage,
            None,
            None,
        )
        .expect("evaluate");
        assert_eq!(
            miss,
            PromotionOutput::RecordedEvidence {
                decision: DecisionState::Miss
            }
        );
        let qualification = evaluate_consumer_claim(
            ConsumerMode::FixturePublic,
            ConsumerClaim::ReplacementQualified,
            &fixture.chain,
            &fixture.binding,
            AdmissionState::Admitted,
            DecisionState::Pass,
            &coverage,
            Some(&release()),
            None,
        )
        .expect("evaluate");
        assert_eq!(
            qualification,
            PromotionOutput::NoClaim {
                reason_code: "qualification_requires_successful_completion".to_owned()
            }
        );
    }

    /// Partial durability (typed null) rejects qualification while the
    /// evidence itself stays recordable.
    #[test]
    fn f4_partial_durability_rejects_qualification_only() {
        let (mut authority, trust_roots) = test_authority(54);
        let spec = sh_spec("exit 0", Duration::from_secs(10), 4096);
        let execution = fixture_execution(&mut authority, &spec, &sh_sha256());
        let termination =
            supervise_execution(&spec, &sh_sha256(), &SupervisionCancel::new()).expect("supervise");
        let mut completion = completion_from(&execution, &termination);
        completion.durability_sha256 = None;
        let execution_receipt = authority
            .sign_execution(&execution, now_ns())
            .expect("sign execution");
        let completion_receipt = authority
            .sign_completion(&execution, &completion, now_ns())
            .expect("sign completion");
        let root = tempfile::tempdir().expect("store root");
        let store = ArtifactStoreV4ChainStore::open(root.path()).expect("open store");
        let chain_identity = store
            .publish_verified_chain(
                &trust_roots,
                &execution,
                &execution_receipt,
                &completion,
                &completion_receipt,
            )
            .expect("publish");
        let chain = store
            .load_verified_chain(&trust_roots, &chain_identity)
            .expect("reload");
        let binding = AdmittedChainBinding {
            source_identity_sha256: chain.execution.source_identity_sha256.clone(),
            build_identity_sha256: chain.execution.predecessor_build_identity_sha256.clone(),
            executable_sha256: chain.execution.executable_sha256.clone(),
        };
        let coverage = complete_coverage(&chain);
        let qualification = evaluate_consumer_claim(
            ConsumerMode::FixturePublic,
            ConsumerClaim::ReplacementQualified,
            &chain,
            &binding,
            AdmissionState::Admitted,
            DecisionState::Pass,
            &coverage,
            Some(&release()),
            None,
        )
        .expect("evaluate");
        assert_eq!(
            qualification,
            PromotionOutput::NoClaim {
                reason_code: "qualification_requires_durable_evidence".to_owned()
            }
        );
        let recorded = evaluate_consumer_claim(
            ConsumerMode::FixturePublic,
            ConsumerClaim::AdmittedCampaignEvidence,
            &chain,
            &binding,
            AdmissionState::Admitted,
            DecisionState::Pass,
            &coverage,
            None,
            None,
        )
        .expect("evaluate");
        assert_eq!(
            recorded,
            PromotionOutput::RecordedEvidence {
                decision: DecisionState::Pass
            }
        );
    }

    /// PrivateLocal policy is independently governed: it gates PrivateLocal
    /// claims and can never block FixturePublic evidence.
    #[test]
    fn f4_private_local_policy_cannot_block_fixture_public() {
        let fixture = verified_fixture(55, "exit 0");
        let coverage = complete_coverage(&fixture.chain);
        // The private policy is UNSATISFIED — FixturePublic is unaffected.
        let public = evaluate_consumer_claim(
            ConsumerMode::FixturePublic,
            ConsumerClaim::AdmittedCampaignEvidence,
            &fixture.chain,
            &fixture.binding,
            AdmissionState::Admitted,
            DecisionState::Pass,
            &coverage,
            None,
            Some(false),
        )
        .expect("evaluate");
        assert_eq!(
            public,
            PromotionOutput::RecordedEvidence {
                decision: DecisionState::Pass
            }
        );
        // The same claim in PrivateLocal mode is gated by that policy.
        let private = evaluate_consumer_claim(
            ConsumerMode::PrivateLocal,
            ConsumerClaim::AdmittedCampaignEvidence,
            &fixture.chain,
            &fixture.binding,
            AdmissionState::Admitted,
            DecisionState::Pass,
            &coverage,
            None,
            Some(false),
        )
        .expect("evaluate");
        assert_eq!(
            private,
            PromotionOutput::NoClaim {
                reason_code: "private_local_policy_not_satisfied".to_owned()
            }
        );
    }
}
