//! Whole-bundle privacy policy and authenticated local artifact envelopes.
//!
//! Public fixtures remain deterministic and explicitly classified. Private
//! artifacts are encrypted as one opaque payload; no nested query, metadata,
//! explanation, snippet, error, manifest, shrink reproduction, log, or
//! filename field is trusted to redact itself.

use std::fmt;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chacha20poly1305::aead::{Aead, KeyInit as AeadKeyInit, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use hmac::{Hmac, KeyInit as HmacKeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// Current schema for the privacy policy bound into every protected payload.
pub const ARTIFACT_PRIVACY_POLICY_SCHEMA_VERSION: u32 = 1;
const ARTIFACT_ENVELOPE_SCHEMA_VERSION: u32 = 1;
const PRIVATE_ENCRYPTION_ALGORITHM: &str = "xchacha20poly1305";
const PUBLIC_INTEGRITY_ALGORITHM: &str = "sha256";
const PRIVATE_DIGEST_ALGORITHM: &str = "hmac-sha256";
const ENVELOPE_AAD_DOMAIN: &str = "frankensearch:quill-gauntlet:privacy-envelope:v1";
const PUBLIC_INTEGRITY_DOMAIN: &[u8] =
    b"frankensearch:quill-gauntlet:public-envelope-integrity:v1\0";
const CONTENT_IDENTITY_DOMAIN: &[u8] =
    b"frankensearch:quill-gauntlet:private-content-identity:v1\0";
const MAX_PROTECTED_PAYLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Shortest private retention window accepted by the durable policy.
pub const PRIVATE_ARTIFACT_MIN_RETENTION_SECONDS: u64 = 60;
/// Longest private retention window accepted by the durable policy.
pub const PRIVATE_ARTIFACT_MAX_RETENTION_SECONDS: u64 = 30 * 24 * 60 * 60;

/// Closed classification for every gauntlet artifact bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactClassification {
    /// Committed or generated corpus whose contents are intentionally public.
    FixturePublic,
    /// Sensitive user-derived material retained only on the local machine.
    PrivateLocal,
}

/// Closed content-kind vocabulary authenticated by private envelopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactContentKind {
    /// One immutable engine-comparison object.
    ArtifactObject,
    /// Mutable reference to an immutable object.
    RunManifest,
    /// Campaign reservation written before engine execution.
    CampaignReservation,
    /// Completed campaign report.
    CampaignReport,
    /// Minimized differential reproduction.
    ShrunkReproduction,
    /// Query-suite or corpus manifest.
    Manifest,
    /// Bounded operator diagnostic.
    Diagnostic,
    /// Structured log payload.
    Log,
    /// Sensitive source filename or path component.
    Filename,
}

impl ArtifactContentKind {
    const fn file_tag(self) -> &'static str {
        match self {
            Self::ArtifactObject => "object",
            Self::RunManifest => "run",
            Self::CampaignReservation => "reservation",
            Self::CampaignReport => "report",
            Self::ShrunkReproduction => "shrink",
            Self::Manifest => "manifest",
            Self::Diagnostic => "diagnostic",
            Self::Log => "log",
            Self::Filename => "name",
        }
    }
}

/// Export destination checked before any artifact leaves its local store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactExportDestination {
    /// Replay by an authorized process on the producing machine.
    LocalReplay,
    /// CI artifact collection.
    CiArtifact,
    /// Release-evidence publication.
    ReleaseArtifact,
    /// Any external upload or transport.
    ExternalUpload,
}

/// Versioned, serialized policy bound into content identity and AEAD AAD.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactPrivacyPolicy {
    /// Policy schema version.
    pub schema_version: u32,
    /// Public-fixture or private-local classification.
    pub classification: ArtifactClassification,
    /// Private retention bound. Public fixtures must leave it absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retention_seconds: Option<u64>,
    /// Whether any non-local export is admitted.
    pub local_only: bool,
}

impl Default for ArtifactPrivacyPolicy {
    fn default() -> Self {
        Self::fixture_public()
    }
}

impl ArtifactPrivacyPolicy {
    /// Deterministic policy for committed/generated public fixtures.
    #[must_use]
    pub const fn fixture_public() -> Self {
        Self {
            schema_version: ARTIFACT_PRIVACY_POLICY_SCHEMA_VERSION,
            classification: ArtifactClassification::FixturePublic,
            retention_seconds: None,
            local_only: false,
        }
    }

    /// Fail-closed policy for encrypted, bounded, local-only artifacts.
    ///
    /// # Errors
    ///
    /// Returns an error when the retention window is shorter than one minute
    /// or longer than thirty days.
    pub fn private_local(retention: Duration) -> Result<Self, ArtifactPrivacyError> {
        let retention_seconds = retention.as_secs();
        let policy = Self {
            schema_version: ARTIFACT_PRIVACY_POLICY_SCHEMA_VERSION,
            classification: ArtifactClassification::PrivateLocal,
            retention_seconds: Some(retention_seconds),
            local_only: true,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), ArtifactPrivacyError> {
        if self.schema_version != ARTIFACT_PRIVACY_POLICY_SCHEMA_VERSION {
            return Err(ArtifactPrivacyError::InvalidPolicy {
                reason: "unknown artifact privacy policy version",
            });
        }
        match self.classification {
            ArtifactClassification::FixturePublic => {
                if self.retention_seconds.is_some() || self.local_only {
                    return Err(ArtifactPrivacyError::InvalidPolicy {
                        reason: "public fixture policy carries private-only fields",
                    });
                }
            }
            ArtifactClassification::PrivateLocal => {
                let Some(retention_seconds) = self.retention_seconds else {
                    return Err(ArtifactPrivacyError::InvalidPolicy {
                        reason: "private-local policy is missing bounded retention",
                    });
                };
                if !(PRIVATE_ARTIFACT_MIN_RETENTION_SECONDS
                    ..=PRIVATE_ARTIFACT_MAX_RETENTION_SECONDS)
                    .contains(&retention_seconds)
                {
                    return Err(ArtifactPrivacyError::InvalidPolicy {
                        reason: "private-local retention is outside the admitted bounds",
                    });
                }
                if !self.local_only {
                    return Err(ArtifactPrivacyError::InvalidPolicy {
                        reason: "private-local policy permits non-local export",
                    });
                }
            }
        }
        Ok(())
    }
}

/// Secret key for private-local artifact authentication and encryption.
///
/// The key has no serialization implementation, its debug representation is
/// always redacted, and its backing bytes are zeroized on drop.
#[derive(Clone)]
pub struct PrivateArtifactKey(Zeroizing<[u8; 32]>);

impl PrivateArtifactKey {
    /// Import 256 bits from an already-authorized secret source.
    #[must_use]
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(Zeroizing::new(bytes))
    }

    /// Generate a new key from the operating system random source.
    ///
    /// # Errors
    ///
    /// Returns an error when the operating system random source is unavailable.
    pub fn generate() -> Result<Self, ArtifactPrivacyError> {
        let mut bytes = [0_u8; 32];
        getrandom::getrandom(&mut bytes)
            .map_err(|_| ArtifactPrivacyError::RandomSourceUnavailable)?;
        Ok(Self::from_bytes(bytes))
    }

    fn expose(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for PrivateArtifactKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PrivateArtifactKey(<redacted>)")
    }
}

/// Runtime policy plus an optional in-memory private key.
#[derive(Clone)]
pub struct ArtifactPrivacyContext {
    policy: ArtifactPrivacyPolicy,
    private_key: Option<PrivateArtifactKey>,
}

impl fmt::Debug for ArtifactPrivacyContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactPrivacyContext")
            .field("policy", &self.policy)
            .field(
                "private_key",
                &self.private_key.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

impl Default for ArtifactPrivacyContext {
    fn default() -> Self {
        Self::fixture_public()
    }
}

impl ArtifactPrivacyContext {
    /// Context for deterministic, explicitly public fixture artifacts.
    #[must_use]
    pub const fn fixture_public() -> Self {
        Self {
            policy: ArtifactPrivacyPolicy::fixture_public(),
            private_key: None,
        }
    }

    /// Context for encrypted local-only artifacts.
    ///
    /// # Errors
    ///
    /// Returns an error when the retention policy is out of bounds.
    pub fn private_local(
        key: PrivateArtifactKey,
        retention: Duration,
    ) -> Result<Self, ArtifactPrivacyError> {
        Ok(Self {
            policy: ArtifactPrivacyPolicy::private_local(retention)?,
            private_key: Some(key),
        })
    }

    /// Durable policy carried by this context.
    #[must_use]
    pub const fn policy(&self) -> &ArtifactPrivacyPolicy {
        &self.policy
    }

    /// Admit or deny one export destination.
    ///
    /// # Errors
    ///
    /// Private-local artifacts reject CI, release, and external upload.
    pub fn authorize_export(
        &self,
        destination: ArtifactExportDestination,
    ) -> Result<(), ArtifactPrivacyError> {
        self.policy.validate()?;
        if self.policy.classification == ArtifactClassification::PrivateLocal
            && destination != ArtifactExportDestination::LocalReplay
        {
            return Err(ArtifactPrivacyError::ExportDenied { destination });
        }
        Ok(())
    }

    /// Encrypt or integrity-wrap a complete serialized payload.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid policy, missing key, oversized payload,
    /// clock failure, random-source failure, or serialization failure.
    pub fn seal(
        &self,
        kind: ArtifactContentKind,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ArtifactPrivacyError> {
        self.seal_at(kind, plaintext, unix_now()?)
    }

    fn seal_at(
        &self,
        kind: ArtifactContentKind,
        plaintext: &[u8],
        now_unix_seconds: u64,
    ) -> Result<Vec<u8>, ArtifactPrivacyError> {
        self.policy.validate()?;
        if plaintext.len() > MAX_PROTECTED_PAYLOAD_BYTES {
            return Err(ArtifactPrivacyError::PayloadTooLarge);
        }
        let plaintext_len =
            u64::try_from(plaintext.len()).map_err(|_| ArtifactPrivacyError::PayloadTooLarge)?;
        let (created_unix_seconds, expires_unix_seconds) =
            self.times_for_envelope(now_unix_seconds)?;
        let aad = EnvelopeAad {
            domain: ENVELOPE_AAD_DOMAIN,
            envelope_schema_version: ARTIFACT_ENVELOPE_SCHEMA_VERSION,
            policy: &self.policy,
            content_kind: kind,
            created_unix_seconds,
            expires_unix_seconds,
            plaintext_len,
        };
        let aad_bytes = serde_json::to_vec(&aad)?;
        let protection = match self.policy.classification {
            ArtifactClassification::FixturePublic => EnvelopeProtection::PublicSha256 {
                algorithm: PUBLIC_INTEGRITY_ALGORITHM.to_owned(),
                payload_sha256: public_integrity_hex(&aad_bytes, plaintext),
                payload_hex: lower_hex(plaintext),
            },
            ArtifactClassification::PrivateLocal => {
                let key = self
                    .private_key
                    .as_ref()
                    .ok_or(ArtifactPrivacyError::MissingPrivateKey)?;
                let cipher = XChaCha20Poly1305::new(Key::from_slice(key.expose()));
                let mut nonce_bytes = [0_u8; 24];
                getrandom::getrandom(&mut nonce_bytes)
                    .map_err(|_| ArtifactPrivacyError::RandomSourceUnavailable)?;
                let nonce = XNonce::from_slice(&nonce_bytes);
                let ciphertext = cipher
                    .encrypt(
                        nonce,
                        Payload {
                            msg: plaintext,
                            aad: &aad_bytes,
                        },
                    )
                    .map_err(|_| ArtifactPrivacyError::EncryptionFailed)?;
                EnvelopeProtection::PrivateXChaCha20Poly1305 {
                    algorithm: PRIVATE_ENCRYPTION_ALGORITHM.to_owned(),
                    nonce_hex: lower_hex(&nonce_bytes),
                    ciphertext_hex: lower_hex(&ciphertext),
                }
            }
        };
        let envelope = ArtifactEnvelope {
            envelope_schema_version: ARTIFACT_ENVELOPE_SCHEMA_VERSION,
            policy: self.policy.clone(),
            content_kind: kind,
            created_unix_seconds,
            expires_unix_seconds,
            plaintext_len,
            protection,
        };
        Ok(serde_json::to_vec(&envelope)?)
    }

    /// Authenticate and open a payload under the exact active policy.
    ///
    /// # Errors
    ///
    /// Plaintext legacy input, classification downgrade, policy drift,
    /// expiration, wrong keys, and tampering all fail closed.
    pub fn open(
        &self,
        expected_kind: ArtifactContentKind,
        envelope_bytes: &[u8],
    ) -> Result<OpenedArtifactBytes, ArtifactPrivacyError> {
        self.open_at(expected_kind, envelope_bytes, unix_now()?)
    }

    fn open_at(
        &self,
        expected_kind: ArtifactContentKind,
        envelope_bytes: &[u8],
        now_unix_seconds: u64,
    ) -> Result<OpenedArtifactBytes, ArtifactPrivacyError> {
        self.policy.validate()?;
        if envelope_bytes.len() > MAX_PROTECTED_PAYLOAD_BYTES.saturating_mul(2) {
            return Err(ArtifactPrivacyError::PayloadTooLarge);
        }
        let envelope: ArtifactEnvelope = serde_json::from_slice(envelope_bytes)
            .map_err(|_| ArtifactPrivacyError::InvalidEnvelope)?;
        envelope.validate_shape()?;
        if envelope.policy != self.policy {
            return Err(ArtifactPrivacyError::PolicyMismatch);
        }
        if envelope.content_kind != expected_kind {
            return Err(ArtifactPrivacyError::ContentKindMismatch);
        }
        if matches!(
            envelope.retention_status_at(now_unix_seconds),
            ArtifactRetentionStatus::Expired { .. }
        ) {
            return Err(ArtifactPrivacyError::RetentionExpired);
        }
        let aad = envelope.aad();
        let aad_bytes = serde_json::to_vec(&aad)?;
        let plaintext = match &envelope.protection {
            EnvelopeProtection::PublicSha256 {
                algorithm,
                payload_sha256,
                payload_hex,
            } => {
                if self.policy.classification != ArtifactClassification::FixturePublic
                    || algorithm != PUBLIC_INTEGRITY_ALGORITHM
                {
                    return Err(ArtifactPrivacyError::ProtectionMismatch);
                }
                let payload = decode_lower_hex(payload_hex)?;
                if public_integrity_hex(&aad_bytes, &payload) != *payload_sha256 {
                    return Err(ArtifactPrivacyError::AuthenticationFailed);
                }
                payload
            }
            EnvelopeProtection::PrivateXChaCha20Poly1305 {
                algorithm,
                nonce_hex,
                ciphertext_hex,
            } => {
                if self.policy.classification != ArtifactClassification::PrivateLocal
                    || algorithm != PRIVATE_ENCRYPTION_ALGORITHM
                {
                    return Err(ArtifactPrivacyError::ProtectionMismatch);
                }
                let key = self
                    .private_key
                    .as_ref()
                    .ok_or(ArtifactPrivacyError::MissingPrivateKey)?;
                let nonce_bytes = decode_lower_hex(nonce_hex)?;
                let nonce: [u8; 24] = nonce_bytes
                    .try_into()
                    .map_err(|_| ArtifactPrivacyError::InvalidEnvelope)?;
                let ciphertext = decode_lower_hex(ciphertext_hex)?;
                let cipher = XChaCha20Poly1305::new(Key::from_slice(key.expose()));
                cipher
                    .decrypt(
                        XNonce::from_slice(&nonce),
                        Payload {
                            msg: &ciphertext,
                            aad: &aad_bytes,
                        },
                    )
                    .map_err(|_| ArtifactPrivacyError::AuthenticationFailed)?
            }
        };
        if u64::try_from(plaintext.len()).ok() != Some(envelope.plaintext_len) {
            return Err(ArtifactPrivacyError::AuthenticationFailed);
        }
        Ok(OpenedArtifactBytes(Zeroizing::new(plaintext)))
    }

    /// Read classification and retention metadata without opening ciphertext.
    ///
    /// # Errors
    ///
    /// Returns an error for legacy plaintext or a malformed/unknown envelope.
    pub fn inspect(
        envelope_bytes: &[u8],
    ) -> Result<ArtifactEnvelopeMetadata, ArtifactPrivacyError> {
        let envelope: ArtifactEnvelope = serde_json::from_slice(envelope_bytes)
            .map_err(|_| ArtifactPrivacyError::InvalidEnvelope)?;
        envelope.validate_shape()?;
        Ok(ArtifactEnvelopeMetadata {
            envelope_schema_version: envelope.envelope_schema_version,
            policy: envelope.policy,
            content_kind: envelope.content_kind,
            created_unix_seconds: envelope.created_unix_seconds,
            expires_unix_seconds: envelope.expires_unix_seconds,
            plaintext_len: envelope.plaintext_len,
        })
    }

    /// Policy-bound stable content identity.
    ///
    /// Public content uses SHA-256. Private content uses HMAC-SHA-256 and
    /// therefore does not expose a reusable unkeyed digest of the query.
    ///
    /// # Errors
    ///
    /// Returns an error when the policy is invalid or a private key is absent.
    pub fn content_identity(
        &self,
        kind: ArtifactContentKind,
        content: &[u8],
    ) -> Result<String, ArtifactPrivacyError> {
        self.policy.validate()?;
        let policy_bytes = serde_json::to_vec(&self.policy)?;
        match self.policy.classification {
            ArtifactClassification::FixturePublic => {
                let mut hasher = Sha256::new();
                hasher.update(CONTENT_IDENTITY_DOMAIN);
                hasher.update(policy_bytes);
                hasher.update(kind.file_tag().as_bytes());
                hasher.update([0]);
                hasher.update(content);
                Ok(sha256_digest_hex(hasher.finalize()))
            }
            ArtifactClassification::PrivateLocal => {
                let key = self
                    .private_key
                    .as_ref()
                    .ok_or(ArtifactPrivacyError::MissingPrivateKey)?;
                let mut mac = <Hmac<Sha256> as HmacKeyInit>::new_from_slice(key.expose())
                    .map_err(|_| ArtifactPrivacyError::MissingPrivateKey)?;
                mac.update(CONTENT_IDENTITY_DOMAIN);
                mac.update(&policy_bytes);
                mac.update(kind.file_tag().as_bytes());
                mac.update(&[0]);
                mac.update(content);
                Ok(lower_hex(&mac.finalize().into_bytes()))
            }
        }
    }

    /// Opaque, path-safe filename derived without exposing source text.
    ///
    /// # Errors
    ///
    /// Returns an error when private identity cannot be computed.
    pub fn opaque_filename(
        &self,
        kind: ArtifactContentKind,
        sensitive_name: &[u8],
    ) -> Result<String, ArtifactPrivacyError> {
        let identity = self.content_identity(kind, sensitive_name)?;
        let classification = match self.policy.classification {
            ArtifactClassification::FixturePublic => "public",
            ArtifactClassification::PrivateLocal => "private",
        };
        Ok(format!(
            "{classification}-{}-{}.artifact",
            kind.file_tag(),
            &identity[..32]
        ))
    }

    /// Produce a typed redacted value suitable for diagnostics and logs.
    ///
    /// # Errors
    ///
    /// Returns an error when private identity cannot be computed.
    pub fn redact(
        &self,
        kind: ArtifactContentKind,
        sensitive: &[u8],
    ) -> Result<RedactedArtifactValue, ArtifactPrivacyError> {
        let digest = self.content_identity(kind, sensitive)?;
        let (algorithm, byte_len) = match self.policy.classification {
            ArtifactClassification::FixturePublic => (
                PUBLIC_INTEGRITY_ALGORITHM.to_owned(),
                Some(u64::try_from(sensitive.len()).unwrap_or(u64::MAX)),
            ),
            ArtifactClassification::PrivateLocal => (PRIVATE_DIGEST_ALGORITHM.to_owned(), None),
        };
        Ok(RedactedArtifactValue {
            classification: self.policy.classification,
            algorithm,
            digest,
            byte_len,
        })
    }

    /// Validate restrictive permissions for a private artifact file.
    ///
    /// Public fixtures have no private permission requirement. On Unix,
    /// private files must have no group/other permission bits.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata cannot be read, permissions are broad,
    /// or the platform cannot prove the required permission model.
    pub fn validate_persisted_permissions(&self, path: &Path) -> Result<(), ArtifactPrivacyError> {
        if self.policy.classification == ArtifactClassification::FixturePublic {
            return Ok(());
        }
        validate_private_permissions(path)
    }

    fn times_for_envelope(
        &self,
        now_unix_seconds: u64,
    ) -> Result<(Option<u64>, Option<u64>), ArtifactPrivacyError> {
        match self.policy.classification {
            ArtifactClassification::FixturePublic => Ok((None, None)),
            ArtifactClassification::PrivateLocal => {
                let retention =
                    self.policy
                        .retention_seconds
                        .ok_or(ArtifactPrivacyError::InvalidPolicy {
                            reason: "private-local retention is missing",
                        })?;
                let expires = now_unix_seconds
                    .checked_add(retention)
                    .ok_or(ArtifactPrivacyError::ClockOutOfRange)?;
                Ok((Some(now_unix_seconds), Some(expires)))
            }
        }
    }
}

/// Opened artifact bytes which are zeroized when dropped.
pub struct OpenedArtifactBytes(Zeroizing<Vec<u8>>);

impl OpenedArtifactBytes {
    /// Borrow the authenticated plaintext.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for OpenedArtifactBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenedArtifactBytes")
            .field("len", &self.0.len())
            .finish_non_exhaustive()
    }
}

/// Non-secret envelope facts available before decryption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactEnvelopeMetadata {
    /// Envelope schema version.
    pub envelope_schema_version: u32,
    /// Structurally valid declared privacy policy; opening authenticates it.
    pub policy: ArtifactPrivacyPolicy,
    /// Structurally valid declared content kind; opening authenticates it.
    pub content_kind: ArtifactContentKind,
    /// Creation time for private-local data.
    pub created_unix_seconds: Option<u64>,
    /// Mandatory expiration for private-local data.
    pub expires_unix_seconds: Option<u64>,
    /// Plaintext byte length authenticated inside the envelope.
    pub plaintext_len: u64,
}

impl ArtifactEnvelopeMetadata {
    /// Evaluate the retention decision at a supplied Unix timestamp.
    #[must_use]
    pub fn retention_status_at(&self, now_unix_seconds: u64) -> ArtifactRetentionStatus {
        retention_status(self.expires_unix_seconds, now_unix_seconds)
    }
}

/// Bounded retention state for a protected payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactRetentionStatus {
    /// Public fixture has no private retention deadline.
    PublicFixture,
    /// Private artifact remains locally replayable until the deadline.
    RetainUntil { expires_unix_seconds: u64 },
    /// Private artifact must no longer be opened or exported.
    Expired { expired_unix_seconds: u64 },
}

/// Content-minimized diagnostic representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedactedArtifactValue {
    /// Policy classification used for the digest.
    pub classification: ArtifactClassification,
    /// `sha256` for public fixtures or `hmac-sha256` for private input.
    pub algorithm: String,
    /// Lowercase policy-bound digest.
    pub digest: String,
    /// Public byte length; private values omit length to reduce leakage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_len: Option<u64>,
}

/// Fail-closed privacy-policy errors. Variants deliberately omit payloads,
/// filenames, ciphertext, nonces, and keys.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactPrivacyError {
    /// Policy fields are inconsistent or unsupported.
    #[error("invalid artifact privacy policy: {reason}")]
    InvalidPolicy { reason: &'static str },
    /// Private operation was attempted without key material.
    #[error("private artifact key is required")]
    MissingPrivateKey,
    /// Input is not a current recognized envelope.
    #[error("artifact is not a recognized current privacy envelope")]
    InvalidEnvelope,
    /// Active and persisted policy differ.
    #[error("artifact privacy policy does not match the active context")]
    PolicyMismatch,
    /// Caller expected a different protected content type.
    #[error("artifact content kind does not match the requested operation")]
    ContentKindMismatch,
    /// Envelope protection does not match its classification.
    #[error("artifact protection mode does not match its classification")]
    ProtectionMismatch,
    /// Ciphertext or public integrity data failed authentication.
    #[error("artifact authentication failed")]
    AuthenticationFailed,
    /// Private encryption failed before any bytes were published.
    #[error("private artifact encryption failed")]
    EncryptionFailed,
    /// Retention deadline has passed.
    #[error("private artifact retention has expired")]
    RetentionExpired,
    /// Destination is forbidden for private artifacts.
    #[error("private artifact export denied for {destination:?}")]
    ExportDenied {
        /// Rejected destination.
        destination: ArtifactExportDestination,
    },
    /// Payload exceeds the bounded in-memory envelope contract.
    #[error("artifact payload exceeds the privacy envelope size bound")]
    PayloadTooLarge,
    /// System time cannot be represented safely.
    #[error("system clock is outside the artifact privacy range")]
    ClockOutOfRange,
    /// Operating-system randomness was unavailable.
    #[error("operating-system randomness is unavailable")]
    RandomSourceUnavailable,
    /// Private file permissions are broader than owner-only.
    #[error("private artifact permissions are not owner-only")]
    InsecurePermissions,
    /// Platform cannot prove owner-only permission semantics.
    #[error("private artifact permission verification is unsupported")]
    PermissionVerificationUnsupported,
    /// JSON envelope encoding failed.
    #[error("artifact privacy envelope encoding failed")]
    Json(#[from] serde_json::Error),
    /// Permission metadata could not be inspected.
    #[error("artifact permission inspection failed")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Serialize)]
struct EnvelopeAad<'a> {
    domain: &'static str,
    envelope_schema_version: u32,
    policy: &'a ArtifactPrivacyPolicy,
    content_kind: ArtifactContentKind,
    created_unix_seconds: Option<u64>,
    expires_unix_seconds: Option<u64>,
    plaintext_len: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactEnvelope {
    envelope_schema_version: u32,
    policy: ArtifactPrivacyPolicy,
    content_kind: ArtifactContentKind,
    created_unix_seconds: Option<u64>,
    expires_unix_seconds: Option<u64>,
    plaintext_len: u64,
    protection: EnvelopeProtection,
}

impl ArtifactEnvelope {
    fn validate_shape(&self) -> Result<(), ArtifactPrivacyError> {
        if self.envelope_schema_version != ARTIFACT_ENVELOPE_SCHEMA_VERSION {
            return Err(ArtifactPrivacyError::InvalidEnvelope);
        }
        self.policy.validate()?;
        let times_match = match self.policy.classification {
            ArtifactClassification::FixturePublic => {
                self.created_unix_seconds.is_none() && self.expires_unix_seconds.is_none()
            }
            ArtifactClassification::PrivateLocal => {
                let Some(created) = self.created_unix_seconds else {
                    return Err(ArtifactPrivacyError::InvalidEnvelope);
                };
                let Some(expires) = self.expires_unix_seconds else {
                    return Err(ArtifactPrivacyError::InvalidEnvelope);
                };
                self.policy
                    .retention_seconds
                    .and_then(|retention| created.checked_add(retention))
                    == Some(expires)
            }
        };
        if !times_match
            || usize::try_from(self.plaintext_len)
                .map_or(true, |length| length > MAX_PROTECTED_PAYLOAD_BYTES)
        {
            return Err(ArtifactPrivacyError::InvalidEnvelope);
        }
        match (&self.policy.classification, &self.protection) {
            (
                ArtifactClassification::FixturePublic,
                EnvelopeProtection::PublicSha256 {
                    algorithm,
                    payload_sha256,
                    payload_hex,
                },
            ) if algorithm == PUBLIC_INTEGRITY_ALGORITHM
                && is_lower_hex(payload_sha256, 64)
                && payload_hex.len()
                    == usize::try_from(self.plaintext_len)
                        .unwrap_or(usize::MAX)
                        .saturating_mul(2) =>
            {
                Ok(())
            }
            (
                ArtifactClassification::PrivateLocal,
                EnvelopeProtection::PrivateXChaCha20Poly1305 {
                    algorithm,
                    nonce_hex,
                    ciphertext_hex,
                },
            ) if algorithm == PRIVATE_ENCRYPTION_ALGORITHM
                && is_lower_hex(nonce_hex, 48)
                && ciphertext_hex.len()
                    == usize::try_from(self.plaintext_len)
                        .unwrap_or(usize::MAX)
                        .saturating_add(16)
                        .saturating_mul(2) =>
            {
                Ok(())
            }
            _ => Err(ArtifactPrivacyError::ProtectionMismatch),
        }
    }

    fn aad(&self) -> EnvelopeAad<'_> {
        EnvelopeAad {
            domain: ENVELOPE_AAD_DOMAIN,
            envelope_schema_version: self.envelope_schema_version,
            policy: &self.policy,
            content_kind: self.content_kind,
            created_unix_seconds: self.created_unix_seconds,
            expires_unix_seconds: self.expires_unix_seconds,
            plaintext_len: self.plaintext_len,
        }
    }

    fn retention_status_at(&self, now_unix_seconds: u64) -> ArtifactRetentionStatus {
        retention_status(self.expires_unix_seconds, now_unix_seconds)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum EnvelopeProtection {
    PublicSha256 {
        algorithm: String,
        payload_sha256: String,
        payload_hex: String,
    },
    PrivateXChaCha20Poly1305 {
        algorithm: String,
        nonce_hex: String,
        ciphertext_hex: String,
    },
}

fn retention_status(
    expires_unix_seconds: Option<u64>,
    now_unix_seconds: u64,
) -> ArtifactRetentionStatus {
    match expires_unix_seconds {
        None => ArtifactRetentionStatus::PublicFixture,
        Some(expires) if now_unix_seconds < expires => ArtifactRetentionStatus::RetainUntil {
            expires_unix_seconds: expires,
        },
        Some(expires) => ArtifactRetentionStatus::Expired {
            expired_unix_seconds: expires,
        },
    }
}

fn unix_now() -> Result<u64, ArtifactPrivacyError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ArtifactPrivacyError::ClockOutOfRange)
        .map(|duration| duration.as_secs())
}

fn public_integrity_hex(aad: &[u8], payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PUBLIC_INTEGRITY_DOMAIN);
    hasher.update(Sha256::digest(aad));
    hasher.update(payload);
    sha256_digest_hex(hasher.finalize())
}

fn sha256_digest_hex(bytes: impl AsRef<[u8]>) -> String {
    lower_hex(bytes.as_ref())
}

fn lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn decode_lower_hex(value: &str) -> Result<Vec<u8>, ArtifactPrivacyError> {
    if !value.len().is_multiple_of(2)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ArtifactPrivacyError::InvalidEnvelope);
    }
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let [high_byte, low_byte] = *pair;
            let high = hex_nibble(high_byte).ok_or(ArtifactPrivacyError::InvalidEnvelope)?;
            let low = hex_nibble(low_byte).ok_or(ArtifactPrivacyError::InvalidEnvelope)?;
            Ok((high << 4) | low)
        })
        .collect()
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(unix)]
fn validate_private_permissions(path: &Path) -> Result<(), ArtifactPrivacyError> {
    use std::os::unix::fs::PermissionsExt as _;

    let mode = std::fs::metadata(path)?.permissions().mode();
    if mode.trailing_zeros() >= 6 {
        Ok(())
    } else {
        Err(ArtifactPrivacyError::InsecurePermissions)
    }
}

#[cfg(not(unix))]
fn validate_private_permissions(_path: &Path) -> Result<(), ArtifactPrivacyError> {
    Err(ArtifactPrivacyError::PermissionVerificationUnsupported)
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY_BYTES: [u8; 32] = [0x5a; 32];

    fn private_context() -> ArtifactPrivacyContext {
        ArtifactPrivacyContext::private_local(
            PrivateArtifactKey::from_bytes(KEY_BYTES),
            Duration::from_secs(300),
        )
        .expect("private policy")
    }

    #[test]
    fn fixture_public_is_deterministic_labeled_and_exportable() {
        let context = ArtifactPrivacyContext::fixture_public();
        let payload = br#"{"query":"committed synthetic fixture"}"#;
        let first = context
            .seal_at(ArtifactContentKind::CampaignReport, payload, 10)
            .expect("first public envelope");
        let second = context
            .seal_at(ArtifactContentKind::CampaignReport, payload, 99)
            .expect("second public envelope");
        assert_eq!(first, second);
        let metadata = ArtifactPrivacyContext::inspect(&first).expect("public metadata");
        assert_eq!(
            metadata.policy.classification,
            ArtifactClassification::FixturePublic
        );
        assert_eq!(metadata.created_unix_seconds, None);
        assert_eq!(metadata.expires_unix_seconds, None);
        assert_eq!(
            context
                .open_at(ArtifactContentKind::CampaignReport, &first, 500)
                .expect("open public")
                .as_bytes(),
            payload
        );
        for destination in [
            ArtifactExportDestination::LocalReplay,
            ArtifactExportDestination::CiArtifact,
            ArtifactExportDestination::ReleaseArtifact,
            ArtifactExportDestination::ExternalUpload,
        ] {
            context
                .authorize_export(destination)
                .expect("public export");
        }
    }

    #[test]
    fn private_whole_bundle_hides_every_nested_canary_and_replays_with_key() {
        let context = private_context();
        let canaries = [
            "QUERY_CANARY_μ",
            "METADATA_CANARY",
            "EXPLANATION_CANARY",
            "SNIPPET_CANARY",
            "ERROR_CANARY",
            "MANIFEST_CANARY",
            "SHRINK_CANARY",
            "LOG_CANARY",
            "FILENAME_CANARY ../private path",
        ];
        let payload = serde_json::to_vec(&serde_json::json!({
            "query": canaries[0],
            "metadata": canaries[1],
            "explanation": canaries[2],
            "snippet": canaries[3],
            "error": canaries[4],
            "manifest": canaries[5],
            "shrunk_reproduction": canaries[6],
            "log": canaries[7],
            "filename": canaries[8],
        }))
        .expect("canary bundle");
        let envelope = context
            .seal_at(ArtifactContentKind::CampaignReport, &payload, 1_000)
            .expect("private envelope");
        let serialized = std::str::from_utf8(&envelope).expect("JSON envelope");
        for canary in canaries {
            assert!(
                !serialized.contains(canary),
                "private envelope leaked nested canary"
            );
        }
        let opened = context
            .open_at(ArtifactContentKind::CampaignReport, &envelope, 1_001)
            .expect("authorized replay");
        assert_eq!(opened.as_bytes(), payload);
        assert!(!format!("{opened:?}").contains(canaries[0]));
    }

    #[test]
    fn private_export_is_local_only_and_expiry_is_fail_closed() {
        let context = private_context();
        context
            .authorize_export(ArtifactExportDestination::LocalReplay)
            .expect("local replay");
        for destination in [
            ArtifactExportDestination::CiArtifact,
            ArtifactExportDestination::ReleaseArtifact,
            ArtifactExportDestination::ExternalUpload,
        ] {
            assert!(matches!(
                context.authorize_export(destination),
                Err(ArtifactPrivacyError::ExportDenied { .. })
            ));
        }
        let envelope = context
            .seal_at(ArtifactContentKind::Manifest, b"private", 100)
            .expect("private manifest");
        assert!(
            context
                .open_at(ArtifactContentKind::Manifest, &envelope, 399)
                .is_ok()
        );
        assert!(matches!(
            context.open_at(ArtifactContentKind::Manifest, &envelope, 400),
            Err(ArtifactPrivacyError::RetentionExpired)
        ));
    }

    #[test]
    fn private_rejects_plaintext_public_downgrade_wrong_key_and_tamper() {
        let private = private_context();
        let public = ArtifactPrivacyContext::fixture_public();
        let public_envelope = public
            .seal_at(ArtifactContentKind::ArtifactObject, b"public", 10)
            .expect("public envelope");
        assert!(matches!(
            private.open_at(
                ArtifactContentKind::ArtifactObject,
                b"{\"query\":\"raw\"}",
                10
            ),
            Err(ArtifactPrivacyError::InvalidEnvelope)
        ));
        assert!(matches!(
            private.open_at(ArtifactContentKind::ArtifactObject, &public_envelope, 10),
            Err(ArtifactPrivacyError::PolicyMismatch)
        ));

        let envelope = private
            .seal_at(ArtifactContentKind::ArtifactObject, b"private", 10)
            .expect("private envelope");
        assert!(matches!(
            public.open_at(ArtifactContentKind::ArtifactObject, &envelope, 10),
            Err(ArtifactPrivacyError::PolicyMismatch)
        ));
        let wrong_key = ArtifactPrivacyContext::private_local(
            PrivateArtifactKey::from_bytes([0x33; 32]),
            Duration::from_secs(300),
        )
        .expect("wrong-key context");
        assert!(matches!(
            wrong_key.open_at(ArtifactContentKind::ArtifactObject, &envelope, 10),
            Err(ArtifactPrivacyError::AuthenticationFailed)
        ));

        let mut value: serde_json::Value =
            serde_json::from_slice(&envelope).expect("private envelope value");
        let ciphertext = value["protection"]["ciphertext_hex"]
            .as_str()
            .expect("ciphertext");
        let mut replacement = ciphertext.as_bytes().to_vec();
        replacement[0] = if replacement[0] == b'0' { b'1' } else { b'0' };
        let replacement = String::from_utf8(replacement).expect("lowercase hex remains UTF-8");
        value["protection"]["ciphertext_hex"] = serde_json::Value::String(replacement);
        let tampered = serde_json::to_vec(&value).expect("tampered envelope");
        assert!(matches!(
            private.open_at(ArtifactContentKind::ArtifactObject, &tampered, 10),
            Err(ArtifactPrivacyError::AuthenticationFailed)
        ));
    }

    #[test]
    fn policy_header_kind_unknown_fields_and_public_digest_are_authenticated() {
        let private = private_context();
        let envelope = private
            .seal_at(ArtifactContentKind::Diagnostic, b"secret", 10)
            .expect("diagnostic");
        assert!(matches!(
            private.open_at(ArtifactContentKind::Log, &envelope, 10),
            Err(ArtifactPrivacyError::ContentKindMismatch)
        ));

        let mut relabeled_private: serde_json::Value =
            serde_json::from_slice(&envelope).expect("envelope");
        relabeled_private["content_kind"] = serde_json::json!("log");
        assert!(matches!(
            private.open_at(
                ArtifactContentKind::Log,
                &serde_json::to_vec(&relabeled_private).expect("relabeled private envelope"),
                10
            ),
            Err(ArtifactPrivacyError::AuthenticationFailed)
        ));

        let mut unknown: serde_json::Value = serde_json::from_slice(&envelope).expect("envelope");
        unknown["unknown_mode"] = serde_json::json!("future");
        assert!(matches!(
            private.open_at(
                ArtifactContentKind::Diagnostic,
                &serde_json::to_vec(&unknown).expect("unknown"),
                10
            ),
            Err(ArtifactPrivacyError::InvalidEnvelope)
        ));

        let public = ArtifactPrivacyContext::fixture_public();
        let mut public_value: serde_json::Value = serde_json::from_slice(
            &public
                .seal_at(ArtifactContentKind::Manifest, b"public", 10)
                .expect("public"),
        )
        .expect("public envelope");
        public_value["protection"]["payload_sha256"] = serde_json::Value::String("0".repeat(64));
        assert!(matches!(
            public.open_at(
                ArtifactContentKind::Manifest,
                &serde_json::to_vec(&public_value).expect("public tamper"),
                10
            ),
            Err(ArtifactPrivacyError::AuthenticationFailed)
        ));

        let mut relabeled: serde_json::Value = serde_json::from_slice(
            &public
                .seal_at(ArtifactContentKind::Manifest, b"public", 10)
                .expect("public"),
        )
        .expect("public envelope");
        relabeled["content_kind"] = serde_json::json!("log");
        assert!(matches!(
            public.open_at(
                ArtifactContentKind::Log,
                &serde_json::to_vec(&relabeled).expect("relabeled public envelope"),
                10
            ),
            Err(ArtifactPrivacyError::AuthenticationFailed)
        ));
    }

    #[test]
    fn private_digests_filenames_and_debug_output_do_not_leak_source() {
        let context = private_context();
        let canary = b"query with spaces; $(shell) \xce\xbc /private/path";
        let first = context
            .content_identity(ArtifactContentKind::Filename, canary)
            .expect("first identity");
        let second = context
            .content_identity(ArtifactContentKind::Filename, canary)
            .expect("second identity");
        assert_eq!(first, second);
        let filename = context
            .opaque_filename(ArtifactContentKind::Filename, canary)
            .expect("opaque filename");
        assert!(!filename.contains("query"));
        assert!(!filename.contains("path"));
        assert!(filename.starts_with("private-name-"));
        assert!(filename.bytes().all(|byte| byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'-' | b'.')));
        let redacted = context
            .redact(ArtifactContentKind::Log, canary)
            .expect("redacted log");
        assert_eq!(redacted.algorithm, PRIVATE_DIGEST_ALGORITHM);
        assert_eq!(redacted.byte_len, None);
        assert!(!format!("{redacted:?}").contains("query"));
        assert!(!format!("{context:?}").contains("5a5a"));
        assert!(!format!("{:?}", PrivateArtifactKey::from_bytes(KEY_BYTES)).contains("5a5a"));
    }

    #[cfg(unix)]
    #[test]
    fn private_permissions_require_owner_only_mode() {
        use std::io::Write as _;
        use std::os::unix::fs::OpenOptionsExt as _;
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("private.artifact");
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&path)
            .expect("private file");
        file.write_all(b"ciphertext").expect("private bytes");
        private_context()
            .validate_persisted_permissions(&path)
            .expect("owner-only permissions");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
            .expect("broaden test permissions");
        assert!(matches!(
            private_context().validate_persisted_permissions(&path),
            Err(ArtifactPrivacyError::InsecurePermissions)
        ));
    }

    #[test]
    fn policy_rejects_missing_out_of_range_and_downgrade_fields() {
        for seconds in [
            0,
            PRIVATE_ARTIFACT_MIN_RETENTION_SECONDS - 1,
            PRIVATE_ARTIFACT_MAX_RETENTION_SECONDS + 1,
        ] {
            assert!(ArtifactPrivacyPolicy::private_local(Duration::from_secs(seconds)).is_err());
        }
        let malformed = serde_json::json!({
            "schema_version": ARTIFACT_PRIVACY_POLICY_SCHEMA_VERSION,
            "classification": "private_local",
            "local_only": false
        });
        let policy: ArtifactPrivacyPolicy =
            serde_json::from_value(malformed).expect("structural policy");
        assert!(policy.validate().is_err());
    }
}
