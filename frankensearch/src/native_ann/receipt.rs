//! Shared selection receipts for the native fixed-authority generation protocol.
//!
//! A receipt is an explicit trust input supplied by the caller, not authority
//! discovered beside an index. Reading it validates its complete serialized
//! contract. Opening it admits the exact expected head through a qualified
//! physical root and retains every component needed by the selected reader.
//! Neither operation publishes, repairs, follows a newer head, or loads a model.
//!
//! Receipt and root I/O are synchronous and belong on the caller's blocking lane.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, AuthorityRefV1, AuthoritySlotV1, EmbedderTierTag,
    FrozenEmbeddingIdentityBundleV1, GenerationRootSecurityProfileV1,
};
use frankensearch_index::FsviV2IdentityBinding;
use frankensearch_index::generation_root::authority_publisher::{
    AUTHORITY_PUBLISHER_LOCK_BYTES_V1, AttemptPermitV1, ExpectedAuthorityPairV1,
};
use frankensearch_index::generation_root::generation_reader::{
    OpenedGenerationSnapshotV1, SnapshotOpenOutcomeV1, SnapshotRefusalV1,
};
use frankensearch_index::generation_root::{GenerationRootAnchorLayout, QualifiedGenerationRoot};
use frankensearch_quill::{DEFAULT_SCHEMA, QuillConfig};
use serde::{Deserialize, Serialize};

use super::{NativeGenerationSnapshot, NativeGenerationTopology};
use crate::{Cx, SearchError, SearchResult};

/// Publication inputs saved before changing authority, never a claim of success.
///
/// The enclosing receipt supplies the root and candidate. Both physical old
/// slots are preserved so recovery can prove an aborted attempt without
/// substituting a newly observed head.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeGenerationSavedAttempt {
    /// Identity of the writer that prepared the attempt.
    pub writer_id: [u8; 16],
    /// Unique identity of the prepared publication attempt.
    pub attempt_id: [u8; 16],
    /// Canonical authority reference in physical slot zero before publication.
    pub expected_first: Option<Vec<u8>>,
    /// Canonical authority reference in physical slot one before publication.
    pub expected_second: Option<Vec<u8>>,
    /// Idempotency key retained across reconciliation.
    pub idempotency_key: [u8; 16],
    /// Whether publication required an external anti-rollback floor.
    pub requires_floor: bool,
}

impl NativeGenerationSavedAttempt {
    /// Preserve the complete expected pair from a prepared publication permit.
    #[must_use]
    pub fn from_permit(permit: &AttemptPermitV1, requires_floor: bool) -> Self {
        Self {
            writer_id: permit.writer_id,
            attempt_id: permit.attempt_id,
            expected_first: permit
                .expected
                .first
                .map(|slot| slot.authority.canonical_bytes().to_vec()),
            expected_second: permit
                .expected
                .second
                .map(|slot| slot.authority.canonical_bytes().to_vec()),
            idempotency_key: permit.idempotency_key,
            requires_floor,
        }
    }

    /// Reconstitute and validate the original publication permit without I/O.
    ///
    /// This restores structure only. The caller must still enforce the external
    /// floor requirement when choosing a publication or recovery policy.
    ///
    /// # Errors
    ///
    /// Rejects malformed canonical references, invalid physical slots, reserved
    /// identities, and a candidate that is not the expected pair's successor.
    pub fn restore(
        &self,
        root_id: [u8; 16],
        candidate: AuthorityRefV1,
    ) -> SearchResult<AttemptPermitV1> {
        let decode = |bytes: &Option<Vec<u8>>, index| -> SearchResult<Option<AuthoritySlotV1>> {
            bytes
                .as_ref()
                .map(|bytes| {
                    let authority =
                        AuthorityRefV1::from_canonical_bytes(bytes).map_err(authority_error)?;
                    AuthoritySlotV1::new(index, root_id, authority).map_err(authority_error)
                })
                .transpose()
        };
        AttemptPermitV1::new(
            root_id,
            self.writer_id,
            self.attempt_id,
            candidate,
            ExpectedAuthorityPairV1 {
                first: decode(&self.expected_first, 0)?,
                second: decode(&self.expected_second, 1)?,
            },
            self.idempotency_key,
        )
        .map_err(authority_error)
    }
}

/// Explicit, bounded selection of one identity-complete native generation.
///
/// The serialized fields and schemas are shared with the native executable.
/// Fields are public for publication and external receipt persistence; a value
/// constructed or deserialized directly is not admitted until validation
/// succeeds. Every read and snapshot-opening operation validates it again.
///
/// Only the explicit cooperative-local profile is supported. A receipt that
/// requires an external floor is refused rather than downgraded on restart.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeGenerationReceipt {
    /// Exact versioned native authority-selection receipt schema.
    pub schema: String,
    /// Absolute path used to qualify the selected physical generation root.
    pub root: PathBuf,
    /// Expected immutable identity of the generation root.
    pub root_id: [u8; 16],
    /// Explicit generation-root security profile.
    pub profile: String,
    /// Canonical bytes of the exact expected activation authority.
    pub authority: Vec<u8>,
    /// Immutable artifact generation shared by all retained components.
    pub generation: ArtifactGenerationIdentityV1,
    /// Independently frozen identity of the primary vector tier.
    pub identity: FrozenEmbeddingIdentityBundleV1,
    /// Primary vector tier; progressive selections use the fast tier here.
    pub tier: EmbedderTierTag,
    /// Independently frozen quality-tier identity for progressive selections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_identity: Option<FrozenEmbeddingIdentityBundleV1>,
    /// Actual covered quality documents for a nonempty proper partial subset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_documents: Option<u64>,
    /// Total live documents in the selected complete source cohort.
    pub documents: u64,
    /// Upper bound on the selected manifest and component image bytes.
    pub image_bytes: u64,
    /// Original publication inputs when this receipt can authorize recovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt: Option<NativeGenerationSavedAttempt>,
}

impl NativeGenerationReceipt {
    /// Legacy single-tier selection schema without a publication attempt.
    pub const SELECTION_ONLY_SCHEMA: &'static str = "frankensearch.native.authority-selection.v1";
    /// Single-tier selection schema with a persisted publication attempt.
    pub const EXACT_SCHEMA: &'static str = "frankensearch.native.authority-selection.v2";
    /// Complete independently bound fast and quality tier schema.
    pub const PROGRESSIVE_SCHEMA: &'static str = "frankensearch.native.authority-selection.v3";
    /// Independently bound fast tier and authenticated partial quality schema.
    pub const PARTIAL_QUALITY_SCHEMA: &'static str = "frankensearch.native.authority-selection.v4";
    /// Maximum encoded receipt size accepted from a file.
    pub const MAX_RECEIPT_BYTES: usize = 64 * 1024;
    /// Maximum live document count in the native receipt contract.
    pub const MAX_DOCUMENTS: u64 = 100_000;
    /// Maximum authenticated manifest links traversed for a historical read.
    pub const MAX_HISTORY_HOPS: u64 = 1_024;
    /// Security profile supported by this receipt protocol.
    pub const PROFILE: GenerationRootSecurityProfileV1 =
        GenerationRootSecurityProfileV1::CooperativeLocal;

    /// Read and validate a bounded regular non-symlink receipt file.
    ///
    /// # Errors
    ///
    /// Propagates I/O errors and refuses an oversized, malformed, unsupported,
    /// inconsistent, or floor-dependent receipt before opening its root.
    pub fn read(path: &Path) -> SearchResult<Self> {
        if !fs::symlink_metadata(path)?.is_file() {
            return Err(invalid(
                "file",
                "the authority receipt must be a regular non-symlink file",
            ));
        }
        let mut bytes = Vec::new();
        File::open(path)?
            .take(Self::MAX_RECEIPT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > Self::MAX_RECEIPT_BYTES {
            return Err(invalid("size", "authority receipt exceeds 64 KiB"));
        }
        let receipt: Self = serde_json::from_slice(&bytes)
            .map_err(|_| invalid("json", "invalid authority selection receipt"))?;
        receipt.validate()?;
        Ok(receipt)
    }

    /// Validate the complete receipt contract without filesystem I/O.
    ///
    /// # Errors
    ///
    /// Refuses unsupported schemas or profiles, missing or inconsistent saved
    /// attempts, unavailable external floors, invalid independent bindings,
    /// impossible coverage, and inconsistent image or document bounds.
    pub fn validate(&self) -> SearchResult<()> {
        if !self.root.is_absolute()
            || self.root_id == [0; 16]
            || self.profile != Self::PROFILE.as_str()
            || self.documents > Self::MAX_DOCUMENTS
            || self.image_bytes == 0
        {
            return Err(invalid("contract", "invalid authority receipt contract"));
        }
        let authority = self.authority()?;
        match (
            self.schema.as_str(),
            self.attempt.as_ref(),
            self.quality_identity.as_ref(),
        ) {
            (Self::EXACT_SCHEMA, Some(attempt), None)
            | (Self::PROGRESSIVE_SCHEMA | Self::PARTIAL_QUALITY_SCHEMA, Some(attempt), Some(_)) => {
                if attempt.requires_floor {
                    return Err(invalid(
                        "external-floor",
                        "the recorded attempt requires an unavailable anti-rollback floor",
                    ));
                }
                attempt.restore(self.root_id, authority)?;
            }
            (Self::SELECTION_ONLY_SCHEMA, None, None) => {}
            _ => {
                return Err(invalid(
                    "schema",
                    "authority receipt schema and prepared attempt disagree",
                ));
            }
        }
        match (self.schema.as_str(), self.quality_documents) {
            (Self::PARTIAL_QUALITY_SCHEMA, Some(count)) if count > 0 && count < self.documents => {}
            (Self::PARTIAL_QUALITY_SCHEMA, _) | (_, Some(_)) => {
                return Err(invalid(
                    "coverage",
                    "partial quality receipt requires a nonempty proper quality subset",
                ));
            }
            _ => {}
        }
        if self.quality_identity.is_some() && self.tier != EmbedderTierTag::Fast {
            return Err(invalid(
                "tier",
                "a full authority receipt must bind fast and quality independently",
            ));
        }
        if self.image_bytes < authority.manifest_len {
            return Err(invalid(
                "image-bytes",
                "authority receipt image length is inconsistent",
            ));
        }
        self.binding()?;
        self.quality_binding()?;
        Ok(())
    }

    /// Restore the exact saved attempt after validating its security policy.
    ///
    /// # Errors
    ///
    /// Propagates receipt validation failures and refuses selection-only
    /// receipts, which cannot authorize publication reconciliation.
    pub fn recovery_attempt(&self) -> SearchResult<AttemptPermitV1> {
        self.validate()?;
        self.attempt
            .as_ref()
            .ok_or_else(|| {
                invalid(
                    "attempt",
                    "this selection-only receipt has no durable publication attempt",
                )
            })?
            .restore(self.root_id, self.authority()?)
    }

    /// Decode and validate the exact canonical authority reference.
    ///
    /// # Errors
    ///
    /// Refuses malformed or noncanonical authority bytes.
    pub fn authority(&self) -> SearchResult<AuthorityRefV1> {
        AuthorityRefV1::from_canonical_bytes(&self.authority).map_err(authority_error)
    }

    /// Validate the primary vector generation and frozen storage binding.
    ///
    /// # Errors
    ///
    /// Propagates generation, frozen-identity, and FSVI v2 binding failures.
    pub fn binding(&self) -> SearchResult<FsviV2IdentityBinding> {
        FsviV2IdentityBinding::new(self.generation, self.identity.clone())
    }

    /// Validate the independently declared quality binding, when present.
    ///
    /// # Errors
    ///
    /// Propagates generation, frozen-identity, and FSVI v2 binding failures.
    pub fn quality_binding(&self) -> SearchResult<Option<FsviV2IdentityBinding>> {
        self.quality_identity
            .as_ref()
            .map(|identity| FsviV2IdentityBinding::new(self.generation, identity.clone()))
            .transpose()
    }

    /// Return the declared topology of an already validated receipt.
    ///
    /// Snapshot opening independently verifies this declaration against the
    /// retained component owners; this accessor alone does not admit images.
    #[must_use]
    pub fn topology(&self) -> NativeGenerationTopology {
        self.quality_documents.map_or_else(
            || {
                if self.quality_identity.is_some() {
                    NativeGenerationTopology::FullProgressive
                } else if self.tier == EmbedderTierTag::Quality {
                    NativeGenerationTopology::QualityOnly
                } else {
                    NativeGenerationTopology::FastOnly
                }
            },
            |quality_documents| NativeGenerationTopology::PartialQuality {
                quality_documents,
                total_documents: self.documents,
                coverage_ppm: u32::try_from(
                    quality_documents.saturating_mul(1_000_000) / self.documents.max(1),
                )
                .unwrap_or(u32::MAX),
            },
        )
    }

    /// Stable diagnostic name of the declared topology.
    #[must_use]
    pub fn topology_name(&self) -> &'static str {
        match self.topology() {
            NativeGenerationTopology::FastOnly => "fast_only",
            NativeGenerationTopology::QualityOnly => "quality_only",
            NativeGenerationTopology::FullProgressive => "full_progressive",
            NativeGenerationTopology::PartialQuality { .. } => "partial_quality",
        }
    }

    /// Compare tier shape without freezing a generation's coverage counts.
    ///
    /// Both receipts must be validated separately. This comparison authorizes
    /// neither a new root nor a new authority, provider, or retained snapshot.
    #[must_use]
    pub fn same_topology(&self, other: &Self) -> bool {
        self.tier == other.tier
            && self.quality_identity.is_some() == other.quality_identity.is_some()
            && self.quality_documents.is_some() == other.quality_documents.is_some()
    }

    /// Qualify the selected root and admit its exact expected current snapshot.
    ///
    /// # Errors
    ///
    /// Propagates receipt, cancellation, root qualification, expected-head,
    /// component, identity, topology, document-count, and image-budget failures.
    /// No different head or fallback snapshot is selected on failure.
    pub fn open(&self, cx: &Cx, max_image_bytes: usize) -> SearchResult<NativeGenerationSnapshot> {
        self.validate()?;
        checkpoint(cx)?;
        let root = admit_root(&self.root)?;
        self.open_on_root(cx, &root, max_image_bytes)
    }

    /// Open on an already qualified physical root without reopening its path.
    ///
    /// Mutation and long-lived serving callers use this to retain one root
    /// capability across admission, model loading, and expected-old publication.
    ///
    /// # Errors
    ///
    /// Applies the same receipt and complete-snapshot checks as opening a root.
    pub fn open_on_root(
        &self,
        cx: &Cx,
        root: &QualifiedGenerationRoot,
        max_image_bytes: usize,
    ) -> SearchResult<NativeGenerationSnapshot> {
        self.validate()?;
        checkpoint(cx)?;
        let budget = image_budget(max_image_bytes)?.min(self.image_bytes);
        let opened = root
            .open_expected_generation_snapshot_with_budget(
                self.root_id,
                Self::PROFILE,
                None,
                self.authority()?,
                budget,
            )
            .map_err(root_error)?;
        checkpoint(cx)?;
        let SnapshotOpenOutcomeV1::Opened(opened) = opened else {
            if matches!(
                opened,
                SnapshotOpenOutcomeV1::Refused(SnapshotRefusalV1::ExpectedCurrentChanged)
            ) {
                return Err(invalid(
                    "expected-current",
                    "the fixed authority head differs from the trusted receipt",
                ));
            }
            return Err(invalid(
                "admission",
                &format!("authority snapshot admission refused: {opened:?}"),
            ));
        };
        self.admit_opened(cx, opened, image_budget_usize(budget)?)
    }

    /// Bind an already retained complete closure to this validated receipt.
    ///
    /// The closure must select this exact root identity and authority. Every
    /// independent vector owner, lexical component, source, and topology passes
    /// the native generation admission appropriate for the receipt schema.
    ///
    /// # Errors
    ///
    /// Propagates receipt and native component admission failures, cancellation,
    /// or mismatched root, authority, topology, and live document count.
    pub fn admit_opened(
        &self,
        cx: &Cx,
        opened: Arc<OpenedGenerationSnapshotV1>,
        max_image_bytes: usize,
    ) -> SearchResult<NativeGenerationSnapshot> {
        self.validate()?;
        checkpoint(cx)?;
        if opened.root_id() != self.root_id || opened.head().authority != self.authority()? {
            return Err(invalid(
                "expected-current",
                "the fixed authority head differs from the trusted receipt",
            ));
        }
        if opened.profile() != Self::PROFILE {
            return Err(invalid(
                "profile",
                "the retained generation differs from the receipt security profile",
            ));
        }
        let max_image_bytes =
            image_budget_usize(image_budget(max_image_bytes)?.min(self.image_bytes))?;
        let snapshot = if let Some(quality) = self.quality_binding()? {
            let admit = if self.quality_documents.is_some() {
                NativeGenerationSnapshot::admit_partial_quality
            } else {
                NativeGenerationSnapshot::admit_progressive
            };
            admit(
                cx,
                opened,
                &self.binding()?,
                &quality,
                DEFAULT_SCHEMA,
                QuillConfig::default(),
                max_image_bytes,
            )?
        } else {
            NativeGenerationSnapshot::admit_exact(
                cx,
                opened,
                &self.binding()?,
                DEFAULT_SCHEMA,
                QuillConfig::default(),
                max_image_bytes,
            )?
        };
        if snapshot.topology() != self.topology()
            || snapshot.metadata().total_documents != self.documents
        {
            return Err(invalid(
                "topology",
                "the retained generation differs from the selected tier or document count",
            ));
        }
        Ok(snapshot)
    }

    /// Admit this explicit current or historical read selection on one root.
    ///
    /// With a current receipt, this receipt selects an authenticated ancestor of
    /// the independently admitted current closure. An old receipt alone never
    /// authorizes history. Both closures share one total image budget, and
    /// traversal is bounded by the receipt protocol's maximum history hops.
    ///
    /// # Errors
    ///
    /// Propagates current receipt and snapshot failures. Refuses a different
    /// root/profile, the current head as a historical target, unavailable
    /// authenticated history, and an insufficient combined image budget.
    pub fn open_read_selection(
        &self,
        cx: &Cx,
        current_receipt: Option<&Path>,
        max_image_bytes: usize,
    ) -> SearchResult<(QualifiedGenerationRoot, Arc<NativeGenerationSnapshot>)> {
        checkpoint(cx)?;
        self.validate()?;
        let current = current_receipt.map(Self::read).transpose()?;
        if let Some(current) = &current {
            if current.root != self.root
                || current.root_id != self.root_id
                || current.profile != self.profile
            {
                return Err(selection_error("historical_root_mismatch"));
            }
            if current.authority()? == self.authority()? {
                return Err(selection_error("historical_target_is_current"));
            }
        }
        let root = admit_root(&self.root)?;
        let Some(current) = current else {
            let snapshot = self.open_on_root(cx, &root, max_image_bytes)?;
            return Ok((root, Arc::new(snapshot)));
        };
        // Charge authenticated image lengths rather than the receipt's upper
        // bound, including each independently admitted copy of a shared image.
        let current = current.open_on_root(cx, &root, max_image_bytes)?;
        let retained = current.retained_snapshot();
        let remaining = retained
            .retained_image_bytes()
            .and_then(|used| u64::try_from(max_image_bytes).ok()?.checked_sub(used))
            .filter(|remaining| *remaining > 0)
            .ok_or_else(|| selection_error("historical_image_budget_exceeded"))?;
        let target_budget = remaining.min(self.image_bytes);
        checkpoint(cx)?;
        let opened = root
            .open_historical_generation_snapshot_with_budget(
                retained,
                self.authority()?,
                Self::MAX_HISTORY_HOPS,
                remaining,
            )
            .map_err(root_error)?;
        checkpoint(cx)?;
        let SnapshotOpenOutcomeV1::Opened(opened) = opened else {
            return Err(selection_error("historical_target_unavailable"));
        };
        let snapshot = self.admit_opened(cx, opened, image_budget_usize(target_budget)?)?;
        checkpoint(cx)?;
        Ok((root, Arc::new(snapshot)))
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: "native_ann.receipt".to_owned(),
            reason: "cancellation requested".to_owned(),
        });
    }
    Ok(())
}

fn admit_root(path: &Path) -> SearchResult<QualifiedGenerationRoot> {
    let layout =
        GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1).map_err(root_error)?;
    QualifiedGenerationRoot::admit(path, layout).map_err(root_error)
}

fn image_budget(bytes: usize) -> SearchResult<u64> {
    u64::try_from(bytes).map_err(|_| invalid("image-bytes", "image budget does not fit u64"))
}

fn image_budget_usize(bytes: u64) -> SearchResult<usize> {
    usize::try_from(bytes).map_err(|_| invalid("image-bytes", "image budget does not fit usize"))
}

fn authority_error(
    source: frankensearch_core::generation::GenerationAuthorityErrorV1,
) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "native_generation_receipt.authority",
        source: Box::new(source),
    }
}

fn root_error(source: frankensearch_index::generation_root::GenerationRootError) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "native_generation_receipt.root",
        source: Box::new(source),
    }
}

fn invalid(value: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "native_ann.receipt".to_owned(),
        value: value.to_owned(),
        reason: reason.to_owned(),
    }
}

fn selection_error(code: &'static str) -> SearchError {
    SearchError::InvalidConfig {
        field: "native_ann.receipt.selection".to_owned(),
        value: code.to_owned(),
        reason: "the requested retained history selection was refused".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};

    use super::*;

    fn receipt() -> NativeGenerationReceipt {
        let authority = AuthorityRefV1::new(1, [0x52; 16], 256, [0x53; 32], None).unwrap();
        let permit = AttemptPermitV1::new(
            [0x51; 16],
            [0x61; 16],
            [0x62; 16],
            authority,
            ExpectedAuthorityPairV1::default(),
            [0x63; 16],
        )
        .unwrap();
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("receipt-contract", 2);
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F32;
        identity.storage.endianness = "little-endian".to_owned();
        NativeGenerationReceipt {
            schema: NativeGenerationReceipt::EXACT_SCHEMA.to_owned(),
            root: std::env::current_dir()
                .unwrap()
                .join("unused-authority-root"),
            root_id: [0x51; 16],
            profile: NativeGenerationReceipt::PROFILE.as_str().to_owned(),
            authority: authority.canonical_bytes().to_vec(),
            generation: ArtifactGenerationIdentityV1::new(1, [0x54; 16]).unwrap(),
            identity: identity.freeze().unwrap(),
            tier: EmbedderTierTag::Fast,
            quality_identity: None,
            quality_documents: None,
            documents: 2,
            image_bytes: 4096,
            attempt: Some(NativeGenerationSavedAttempt::from_permit(&permit, false)),
        }
    }

    #[test]
    fn public_receipt_read_preserves_all_native_schemas_and_independent_bindings() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("selection.json");
        for schema in [
            NativeGenerationReceipt::SELECTION_ONLY_SCHEMA,
            NativeGenerationReceipt::EXACT_SCHEMA,
            NativeGenerationReceipt::PROGRESSIVE_SCHEMA,
            NativeGenerationReceipt::PARTIAL_QUALITY_SCHEMA,
        ] {
            let mut selected = receipt();
            selected.schema = schema.to_owned();
            if schema == NativeGenerationReceipt::SELECTION_ONLY_SCHEMA {
                selected.attempt = None;
            }
            if matches!(
                schema,
                NativeGenerationReceipt::PROGRESSIVE_SCHEMA
                    | NativeGenerationReceipt::PARTIAL_QUALITY_SCHEMA
            ) {
                let mut quality = selected.identity.identity.clone();
                quality.producer.implementation_revision = "independent-quality".to_owned();
                selected.quality_identity = Some(quality.freeze().unwrap());
            }
            if schema == NativeGenerationReceipt::PARTIAL_QUALITY_SCHEMA {
                selected.quality_documents = Some(1);
            }
            let encoded = serde_json::to_vec(&selected).unwrap();
            fs::write(&path, &encoded).unwrap();
            let reopened = NativeGenerationReceipt::read(&path).unwrap();
            assert_eq!(serde_json::to_vec(&reopened).unwrap(), encoded);
            assert_eq!(reopened.authority().unwrap(), selected.authority().unwrap());
            assert_eq!(reopened.topology(), selected.topology());
            let quality_binding = reopened.quality_binding().unwrap();
            assert_eq!(
                quality_binding
                    .as_ref()
                    .map(FsviV2IdentityBinding::frozen_identity),
                selected.quality_identity.as_ref(),
            );
            assert_eq!(
                reopened.recovery_attempt().is_ok(),
                schema != NativeGenerationReceipt::SELECTION_ONLY_SCHEMA,
            );
        }
    }

    #[test]
    fn public_receipt_read_refuses_floor_downgrade_and_unknown_fields() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("selection.json");
        let mut selected = receipt();
        selected.attempt.as_mut().unwrap().requires_floor = true;
        fs::write(&path, serde_json::to_vec(&selected).unwrap()).unwrap();
        assert!(matches!(
            NativeGenerationReceipt::read(&path),
            Err(SearchError::InvalidConfig { field, value, .. })
                if field == "native_ann.receipt" && value == "external-floor"
        ));
        let mut encoded = serde_json::to_value(receipt()).unwrap();
        encoded["unknown_selection_policy"] = serde_json::json!(true);
        fs::write(&path, serde_json::to_vec(&encoded).unwrap()).unwrap();
        assert!(matches!(
            NativeGenerationReceipt::read(&path),
            Err(SearchError::InvalidConfig { field, value, .. })
                if field == "native_ann.receipt" && value == "json"
        ));
    }

    #[test]
    fn public_receipt_read_is_bounded_and_requires_a_regular_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oversized.json");
        fs::write(
            &path,
            vec![b' '; NativeGenerationReceipt::MAX_RECEIPT_BYTES + 1],
        )
        .unwrap();
        assert!(matches!(
            NativeGenerationReceipt::read(&path),
            Err(SearchError::InvalidConfig { value, .. }) if value == "size"
        ));
        assert!(matches!(
            NativeGenerationReceipt::read(directory.path()),
            Err(SearchError::InvalidConfig { value, .. }) if value == "file"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn public_receipt_read_refuses_a_symlink_to_a_valid_selection() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("selection.json");
        let alias = directory.path().join("selection-link.json");
        fs::write(&target, serde_json::to_vec(&receipt()).unwrap()).unwrap();
        std::os::unix::fs::symlink(&target, &alias).unwrap();
        assert!(matches!(
            NativeGenerationReceipt::read(&alias),
            Err(SearchError::InvalidConfig { value, .. }) if value == "file"
        ));
        NativeGenerationReceipt::read(&target).unwrap();
    }
}
