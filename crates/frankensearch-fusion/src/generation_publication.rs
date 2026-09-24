//! Complete, pre-staged generation publication over the fixed authority root.
//!
//! Component producers first join their independently derived receipts with
//! [`ExactGenerationComponentsV1::admit`]. [`GenerationCandidateV1`] derives the
//! activation manifest from that join, rather than accepting another set of
//! uncorroborated component hashes. The caller stages the content-addressed
//! component objects and the candidate's manifest in the qualified root.
//! [`GenerationPublisherV1::publish`] then admits and durably verifies every declared
//! objects before asking the existing authority publisher to switch the head.
//!
//! These operations are synchronous and cancellation-aware BETWEEN bounded
//! filesystem operations. Async callers must use their bounded blocking pool;
//! this module neither creates a runtime nor spawns detached work. Immutable
//! objects must remain immutable under the generation-root ownership contract.
//! A retained descriptor and a checksum are not permission to rewrite an object.
//!
//! This is the publication boundary, not an engine parser or an artifact writer.
//! Producers still own engine validation, generation/embedding identity, and
//! source-checkpoint provenance. In particular, it does not change the fsfs CLI's
//! current artifact layout or turn a two-file rename into an atomic generation.
//! New candidates use activation-manifest schema 2, which explicitly binds ANN
//! presence or absence. Exact search generations publish their three mandatory
//! components without inventing an accelerator. Schema-1 predecessors remain
//! readable under the same authority, fencing and anti-rollback protocol.

#![forbid(unsafe_code)]

use std::fmt;
use std::sync::Arc;

use asupersync::Cx;
use frankensearch_core::generation::{
    ActivationManifest, ActivationManifestV2, ArtifactGenerationIdentityV1, AuthorityRefV1,
    AuthoritySlotV1, ExactGenerationComponentsV1, GENERATION_AUTHORITY_SLOT_BYTES_V1,
    GENERATION_LOCK_FRAME_BYTES_V1, GenerationAuthorityActionV1, GenerationAuthorityErrorV1,
    GenerationComponentReceiptV1, GenerationComponentReceiptsV2, GenerationComponentRole,
    GenerationLockFrameKindV1, GenerationLockFrameV1, GenerationRootSecurityProfileV1,
    resolve_authority_slots_v1, verify_authority_manifest_reference,
};
use frankensearch_index::generation_root::authority_publisher::{
    AUTHORITY_PUBLISHER_LOCK_BYTES_V1, AntiRollbackFloorProviderV1, AttemptPermitV1,
    AuthorityPublisherV1, ExpectedAuthorityPairV1, PublicationOutcomeV1,
};
use frankensearch_index::generation_root::generation_reader::{
    GenerationSnapshotCellV1, OpenedGenerationSnapshotV1, SnapshotOpenOutcomeV1, SnapshotRefusalV1,
    activation_manifest_name_v1, activation_manifest_path_v1, component_object_path_v1,
    is_retained_object_name_v1,
};
use frankensearch_index::generation_root::{
    GENERATION_ROOT_AUTHORITY_FILE_NAME, GENERATION_ROOT_IMMUTABLE_FILE_MODE,
    GENERATION_ROOT_LOCK_FILE_NAME, GENERATION_ROOT_MAX_FILE_BYTES, GenerationFileExpectation,
    GenerationRootEntryKind, GenerationRootError, GenerationRootInventory, QualifiedGenerationFile,
    QualifiedGenerationRoot,
};

/// A refusal before entering the authority publisher. No authority bytes were
/// issued by this call. Publication uncertainty is an OUTCOME, not this error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationPublicationErrorV1 {
    /// A manifest, authority, frame, or predecessor was invalid.
    Authority(GenerationAuthorityErrorV1),
    /// The sum of the declared owned images exceeds the caller's explicit budget.
    AdmissionBudget {
        /// Required bytes, or `u64::MAX` when the sum overflowed.
        required: u64,
        /// Caller-selected maximum.
        limit: u64,
    },
    /// A single component exceeds the underlying root's file ceiling.
    ComponentTooLarge {
        /// Component that could not be admitted.
        role: GenerationComponentRole,
    },
    /// The caller's expected pair does not describe the candidate's predecessor.
    ExpectedPredecessorMismatch,
    /// Current LOCK or AUTHORITY bytes no longer describe the expected head.
    StaleExpectedAuthority,
    /// A prior attempt must be reconciled, never automatically retried.
    PendingReconciliation,
    /// A stop request was observed before invoking the authority publisher.
    Cancelled {
        /// Stable operation boundary.
        phase: &'static str,
    },
    /// A candidate object or retained root failed descriptor-owned admission.
    Root(GenerationRootError),
    /// A sealed root contained an undeclared object, alias, WAL, or foreign file.
    InvalidRootClosure,
}

impl fmt::Display for GenerationPublicationErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "generation publication refused: {self:?}")
    }
}

impl std::error::Error for GenerationPublicationErrorV1 {}

impl From<GenerationAuthorityErrorV1> for GenerationPublicationErrorV1 {
    fn from(error: GenerationAuthorityErrorV1) -> Self {
        Self::Authority(error)
    }
}

impl From<GenerationRootError> for GenerationPublicationErrorV1 {
    fn from(error: GenerationRootError) -> Self {
        Self::Root(error)
    }
}

/// An immutable activation manifest built from one successfully joined set of
/// engine receipts. No setters and no deserialization can bypass that join.
#[derive(Debug, Clone)]
pub struct GenerationCandidateV1 {
    authority: AuthorityRefV1,
    manifest: ActivationManifest,
    manifest_bytes: Vec<u8>,
    owned_image_bytes: u64,
}

impl GenerationCandidateV1 {
    /// Derive a genesis or exact successor from joined component receipts.
    ///
    /// The object id identifies the immutable manifest to stage; the writer
    /// fence and artifact generation must come from the caller's trusted build
    /// session. The source checkpoint and document set are derived from the
    /// component join and cannot be supplied independently.
    ///
    /// # Errors
    ///
    /// Refuses invalid identities, sequence exhaustion, oversized
    /// objects, or an overflowing aggregate image size.
    pub fn new(
        object_id: [u8; 16],
        predecessor: Option<AuthorityRefV1>,
        action: GenerationAuthorityActionV1,
        generation: ArtifactGenerationIdentityV1,
        writer_fence_sha256: [u8; 32],
        components: &ExactGenerationComponentsV1,
    ) -> Result<Self, GenerationPublicationErrorV1> {
        let authority_sequence = predecessor.map_or(Ok(1), |head| head.next_sequence())?;
        let manifest = ActivationManifest::V2(ActivationManifestV2::new(
            authority_sequence,
            predecessor,
            action,
            generation,
            writer_fence_sha256,
            components.source_checkpoint(),
            components.docset_digest(),
            GenerationComponentReceiptsV2 {
                vector: components.vector().bytes,
                lexical: components.lexical().bytes,
                ann: components.ann().map(|ann| ann.bytes),
                metadata: components.metadata().bytes,
            },
        )?);
        let (manifest_len, manifest_sha256) = manifest.object_receipt();
        let authority = AuthorityRefV1::new(
            authority_sequence,
            object_id,
            manifest_len,
            manifest_sha256,
            predecessor.map(|head| head.fingerprint()),
        )?;
        let mut owned_image_bytes = manifest_len;
        for (role, receipt) in component_receipts(&manifest) {
            if receipt.byte_len > GENERATION_ROOT_MAX_FILE_BYTES {
                return Err(GenerationPublicationErrorV1::ComponentTooLarge { role });
            }
            owned_image_bytes = owned_image_bytes.checked_add(receipt.byte_len).ok_or(
                GenerationPublicationErrorV1::AdmissionBudget {
                    required: u64::MAX,
                    limit: u64::MAX,
                },
            )?;
        }
        let manifest_bytes = manifest.canonical_bytes();
        Ok(Self {
            authority,
            manifest,
            manifest_bytes,
            owned_image_bytes,
        })
    }

    /// Exact authority that publication will propose.
    #[must_use]
    pub const fn authority(&self) -> AuthorityRefV1 {
        self.authority
    }

    /// Manifest to stage; its component locators are content-addressed.
    #[must_use]
    pub const fn manifest(&self) -> &ActivationManifest {
        &self.manifest
    }

    /// Exact canonical bytes to write to [`Self::manifest_name`].
    #[must_use]
    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest_bytes
    }

    /// Exact top-level manifest object name, with no caller-controlled path.
    #[must_use]
    pub fn manifest_name(&self) -> String {
        activation_manifest_name_v1(self.authority.object_id)
    }

    /// Aggregate byte length of the declared images admitted during preflight.
    /// This is not a process RSS cap: existing readers and filesystem work have
    /// their own memory requirements. Images are dropped before reopening.
    #[must_use]
    pub const fn owned_image_bytes(&self) -> u64 {
        self.owned_image_bytes
    }

    fn check_budget(&self, limit: u64) -> Result<(), GenerationPublicationErrorV1> {
        if self.owned_image_bytes > limit {
            return Err(GenerationPublicationErrorV1::AdmissionBudget {
                required: self.owned_image_bytes,
                limit,
            });
        }
        Ok(())
    }

    fn check_predecessor(
        &self,
        pair: ExpectedAuthorityPairV1,
        root_id: [u8; 16],
    ) -> Result<(), GenerationPublicationErrorV1> {
        for slot in [pair.first, pair.second].into_iter().flatten() {
            if slot.root_id != root_id {
                return Err(GenerationAuthorityErrorV1::RootMismatch.into());
            }
        }
        let head = resolve_authority_slots_v1(pair.first, pair.second)?;
        if head.map(|slot| slot.authority) != self.manifest.predecessor() {
            return Err(GenerationPublicationErrorV1::ExpectedPredecessorMismatch);
        }
        Ok(())
    }
}

fn component_receipts(
    manifest: &ActivationManifest,
) -> impl Iterator<Item = (GenerationComponentRole, GenerationComponentReceiptV1)> {
    manifest.components().iter()
}

fn checkpoint(cx: &Cx, phase: &'static str) -> Result<(), GenerationPublicationErrorV1> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        Err(GenerationPublicationErrorV1::Cancelled { phase })
    } else {
        Ok(())
    }
}

fn authenticated_pair(
    bytes: &[u8],
    root_id: [u8; 16],
) -> Result<ExpectedAuthorityPairV1, GenerationPublicationErrorV1> {
    if bytes.len() != 2 * GENERATION_AUTHORITY_SLOT_BYTES_V1 {
        return Err(GenerationAuthorityErrorV1::InvalidSlotLength.into());
    }
    let decode = |index: u8| {
        let start = usize::from(index) * GENERATION_AUTHORITY_SLOT_BYTES_V1;
        let frame = &bytes[start..start + GENERATION_AUTHORITY_SLOT_BYTES_V1];
        if frame.iter().all(|byte| *byte == 0) {
            Ok(None)
        } else {
            AuthoritySlotV1::from_authenticated_bytes(frame, index, root_id).map(Some)
        }
    };
    let pair = ExpectedAuthorityPairV1 {
        first: decode(0)?,
        second: decode(1)?,
    };
    resolve_authority_slots_v1(pair.first, pair.second)?;
    Ok(pair)
}

fn check_completed_attempt(
    bytes: &[u8],
    root_id: [u8; 16],
    pair: ExpectedAuthorityPairV1,
) -> Result<(), GenerationPublicationErrorV1> {
    if u64::try_from(bytes.len()).ok() != Some(AUTHORITY_PUBLISHER_LOCK_BYTES_V1) {
        return Err(GenerationAuthorityErrorV1::InvalidSlotLength.into());
    }
    let decode = |index: usize, kind: GenerationLockFrameKindV1| {
        let start = index * GENERATION_LOCK_FRAME_BYTES_V1;
        let bytes = &bytes[start..start + GENERATION_LOCK_FRAME_BYTES_V1];
        if bytes.iter().all(|byte| *byte == 0) {
            return Ok(None);
        }
        let frame = GenerationLockFrameV1::from_authenticated_bytes(bytes, root_id)?;
        if frame.kind != kind {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "generation_publication.lock_frame.kind",
            });
        }
        Ok(Some(frame))
    };
    let owner = decode(0, GenerationLockFrameKindV1::Owner)?;
    let attempt = decode(1, GenerationLockFrameKindV1::Attempt)?;
    if let Some(attempt) = attempt
        && !owner.is_some_and(|owner| {
            owner.writer_id == attempt.writer_id
                && owner.attempt_id == attempt.attempt_id
                && owner.fence == attempt.fence
                && owner.authority_fingerprint == attempt.authority_fingerprint
        })
    {
        return Err(GenerationPublicationErrorV1::PendingReconciliation);
    }
    // Completion must describe the selected head, not merely a matching pair
    // of diagnostic frames left behind by a different publication.
    frankensearch_core::generation::resolve_authority_slots_with_locks_v1(
        pair.first,
        pair.second,
        owner,
        None,
    )?;
    Ok(())
}

fn check_root_closure(
    inventory: &GenerationRootInventory,
    device: u64,
) -> Result<(), GenerationPublicationErrorV1> {
    for entry in inventory.entries() {
        let name = entry.name().as_encoded_bytes();
        if name == GENERATION_ROOT_LOCK_FILE_NAME.as_bytes()
            || name == GENERATION_ROOT_AUTHORITY_FILE_NAME.as_bytes()
        {
            // The qualified root and held guard independently admit anchors.
            continue;
        }
        if !is_retained_object_name_v1(name)
            || entry.kind() != GenerationRootEntryKind::RegularFile
            || entry.hard_links() != 1
            || entry.device() != device
            || entry.mode() & 0o7777 != GENERATION_ROOT_IMMUTABLE_FILE_MODE
        {
            return Err(GenerationPublicationErrorV1::InvalidRootClosure);
        }
    }
    if !inventory.aliases().is_empty() {
        return Err(GenerationPublicationErrorV1::InvalidRootClosure);
    }
    Ok(())
}

/// Read-only preparation. The shared anchor guard excludes cooperative
/// publishers during the complete admission/barrier pass. The authority
/// publisher subsequently performs its OWN exclusive expected-pair fence.
// The platform fallback's QualifiedGenerationFile is uninhabited. Admission
// fails before reaching any of the following owned-file operations there.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos")),
    allow(unreachable_code)
)]
fn prepare(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    root_id: [u8; 16],
    candidate: &GenerationCandidateV1,
    expected: ExpectedAuthorityPairV1,
    max_owned_image_bytes: u64,
) -> Result<Vec<QualifiedGenerationFile>, GenerationPublicationErrorV1> {
    candidate.check_budget(max_owned_image_bytes)?;
    candidate.check_predecessor(expected, root_id)?;
    checkpoint(cx, "generation.prepare")?;
    let guard = root.read_guard()?;
    let observed = authenticated_pair(guard.authority_bytes(), root_id)?;
    if observed != expected {
        return Err(GenerationPublicationErrorV1::StaleExpectedAuthority);
    }
    check_completed_attempt(guard.lock_bytes(), root_id, observed)?;

    let path = activation_manifest_path_v1(candidate.authority.object_id)?;
    let manifest_file = root.admit_file(
        &path,
        GenerationFileExpectation::immutable(
            candidate.authority.manifest_len,
            candidate.authority.manifest_sha256,
        )?,
    )?;
    let manifest = ActivationManifest::from_canonical_bytes(manifest_file.as_bytes())?;
    verify_authority_manifest_reference(&candidate.authority, &manifest)?;

    let mut files: Vec<QualifiedGenerationFile> = Vec::with_capacity(5);
    files.push(manifest_file);
    for (role, receipt) in component_receipts(&manifest) {
        checkpoint(cx, "generation.component_admission")?;
        files.push(root.admit_file(
            &component_object_path_v1(role, receipt.sha256)?,
            GenerationFileExpectation::immutable(receipt.byte_len, receipt.sha256)?,
        )?);
    }
    for file in &files {
        checkpoint(cx, "generation.component_durability")?;
        file.sync_durable()?;
    }
    checkpoint(cx, "generation.directory_durability")?;
    root.sync_directory_durable()?;
    check_root_closure(&guard.inventory()?, root.witness().device())?;
    checkpoint(cx, "generation.prepared")?;
    guard.release()?;
    // Keep the admitted owners alive through the authority operation. They
    // never replace the separate fresh published-authority read afterwards.
    Ok(files)
}

/// Reader activation after a known successful authority switch.
///
/// An inability to open the committed generation is not a failed commit and must not be retried
/// as one. In every non-installed case the previous snapshot remains installed.
#[derive(Debug)]
pub enum GenerationActivationV1 {
    /// A fresh authority-resolved snapshot is now installed. Another publisher
    /// may already have superseded this call's committed authority; inspect head.
    Installed(Arc<OpenedGenerationSnapshotV1>),
    /// Cancellation arrived after the commit; serving still retains its old head.
    DeferredByCancellation,
    /// Fresh-reader admission refused the disk state without replacing readers.
    Refused(SnapshotRefusalV1),
    /// Reader infrastructure failed without replacing readers.
    Failed(GenerationRootError),
}

/// Terminal publication result, with commit knowledge separate from activation.
#[derive(Debug)]
pub struct GenerationPublicationResultV1 {
    /// Unmodified outcome of the underlying authority publisher.
    pub publication: PublicationOutcomeV1,
    /// Present only for a known committed outcome.
    pub activation: Option<GenerationActivationV1>,
}

/// Serial publisher plus a retained, immutable serving snapshot.
///
/// The `&mut self` write API prevents same-instance concurrent publication.
/// Across instances/processes the root's existing kernel flock and expected-pair
/// protocol remain authoritative. Snapshot loads return owned `Arc`s; the hot
/// path never reads an independently selected set of component paths.
#[derive(Debug)]
pub struct GenerationPublisherV1<'root> {
    root: &'root QualifiedGenerationRoot,
    publisher: AuthorityPublisherV1<'root>,
    root_id: [u8; 16],
    profile: GenerationRootSecurityProfileV1,
    snapshots: GenerationSnapshotCellV1,
    pending: Option<AttemptPermitV1>,
}

impl<'root> GenerationPublisherV1<'root> {
    /// Bind one serving/publication session to a qualified root and fixed profile.
    ///
    /// # Errors
    ///
    /// Rejects reserved root/writer identities. Nothing is published or opened.
    pub fn new(
        root: &'root QualifiedGenerationRoot,
        root_id: [u8; 16],
        writer_id: [u8; 16],
        profile: GenerationRootSecurityProfileV1,
    ) -> Result<Self, GenerationPublicationErrorV1> {
        Ok(Self {
            root,
            publisher: root.authority_publisher(root_id, writer_id, profile)?,
            root_id,
            profile,
            snapshots: GenerationSnapshotCellV1::new(),
            pending: None,
        })
    }

    /// Pin all component bytes of the last admitted serving generation at once.
    #[must_use]
    pub fn snapshot(&self) -> Option<Arc<OpenedGenerationSnapshotV1>> {
        self.snapshots.load()
    }

    /// Exact unresolved attempt; callers may retain it for fresh-process recovery
    /// through the authority publisher. This session refuses another publication.
    #[must_use]
    pub const fn pending_attempt(&self) -> Option<AttemptPermitV1> {
        self.pending
    }

    /// Refresh serving state without mutating authority or discarding old readers.
    ///
    /// # Errors
    ///
    /// Reports cancellation, unresolved publication, or root infrastructure errors.
    pub fn refresh(
        &mut self,
        cx: &Cx,
        floor: Option<&dyn AntiRollbackFloorProviderV1>,
    ) -> Result<SnapshotOpenOutcomeV1, GenerationPublicationErrorV1> {
        if self.pending.is_some() {
            return Err(GenerationPublicationErrorV1::PendingReconciliation);
        }
        checkpoint(cx, "generation.refresh")?;
        self.snapshots
            .refresh(self.root, self.root_id, self.profile, floor)
            .map_err(Into::into)
    }

    /// Admit and durably verify a pre-staged complete generation, switch its
    /// authority once, then open the actual published head for serving.
    ///
    /// `max_owned_image_bytes` bounds the sum of the declared preflight images, not
    /// RSS, page cache, or snapshots retained by existing queries. The caller
    /// must stage objects through its trusted writer before calling this method.
    /// No artifact is created, overwritten, deleted, or renamed here.
    ///
    /// Cancellation is checked before entering the authority publisher. Once it
    /// starts, its exact commit/unknown outcome is preserved even if cancellation
    /// arrives: no ordinary cancellation error can conceal an issued slot write.
    ///
    /// # Errors
    ///
    /// Every returned error precedes authority mutation by this invocation. A
    /// possibly issued authority write is always `CommitOutcomeUnknown` instead.
    pub fn publish(
        &mut self,
        cx: &Cx,
        candidate: &GenerationCandidateV1,
        expected: ExpectedAuthorityPairV1,
        floor: Option<&dyn AntiRollbackFloorProviderV1>,
        idempotency_key: [u8; 16],
        max_owned_image_bytes: u64,
    ) -> Result<GenerationPublicationResultV1, GenerationPublicationErrorV1> {
        if self.pending.is_some() {
            return Err(GenerationPublicationErrorV1::PendingReconciliation);
        }
        self.profile.require_mutation_authorized()?;
        if idempotency_key == [0; 16] {
            return Err(GenerationAuthorityErrorV1::InvalidField {
                field: "anti_rollback_floor.idempotency_key",
            }
            .into());
        }
        let prepared = prepare(
            cx,
            self.root,
            self.root_id,
            candidate,
            expected,
            max_owned_image_bytes,
        )?;
        checkpoint(cx, "generation.publish")?;
        let publication =
            self.publisher
                .publish(candidate.authority, expected, floor, idempotency_key)?;
        // Do not retain a second complete copy beside the serving snapshot.
        drop(prepared);
        let activation = match &publication {
            PublicationOutcomeV1::Committed { .. } => Some(self.activate(cx, floor)),
            PublicationOutcomeV1::NotCommitted(_) => None,
            PublicationOutcomeV1::CommitOutcomeUnknown(permit) => {
                self.pending = Some(*permit);
                None
            }
        };
        Ok(GenerationPublicationResultV1 {
            publication,
            activation,
        })
    }

    fn activate(
        &self,
        cx: &Cx,
        floor: Option<&dyn AntiRollbackFloorProviderV1>,
    ) -> GenerationActivationV1 {
        if checkpoint(cx, "generation.activate").is_err() {
            return GenerationActivationV1::DeferredByCancellation;
        }
        match self
            .snapshots
            .refresh(self.root, self.root_id, self.profile, floor)
        {
            Ok(SnapshotOpenOutcomeV1::Opened(snapshot)) => {
                GenerationActivationV1::Installed(snapshot)
            }
            Ok(SnapshotOpenOutcomeV1::Refused(refusal)) => GenerationActivationV1::Refused(refusal),
            Err(error) => GenerationActivationV1::Failed(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::generation::{
        CanonicalDocsetV1, CommitRange, ExactComponentReceiptV1, SourceCheckpointV1,
    };
    use sha2::{Digest as _, Sha256};

    // Opaque component fixtures exercise publication, not engine conformance.
    // Engine producers are responsible for the independently derived receipts
    // that this module joins and binds to the actual stored byte images.
    fn bytes(role: GenerationComponentRole, generation: u8) -> Vec<u8> {
        format!("component:{}:generation:{generation}", role.as_str()).into_bytes()
    }

    fn joined(generation: u8, with_ann: bool) -> ExactGenerationComponentsV1 {
        let documents =
            CanonicalDocsetV1::from_ordered_live_documents(["doc-a", "doc-b"]).expect("docset");
        let checkpoint = SourceCheckpointV1::derive(&CommitRange {
            low: 1,
            high: u64::from(generation),
        });
        let receipt = |role| {
            let bytes = bytes(role, generation);
            ExactComponentReceiptV1 {
                role,
                bytes: GenerationComponentReceiptV1 {
                    byte_len: u64::try_from(bytes.len()).expect("fixture length"),
                    sha256: Sha256::digest(&bytes).into(),
                },
                docset_digest: documents.digest(),
                live_document_count: 2,
                source_checkpoint: checkpoint.to_bytes(),
            }
        };
        ExactGenerationComponentsV1::admit(
            receipt(GenerationComponentRole::Vector),
            receipt(GenerationComponentRole::Lexical),
            with_ann.then(|| receipt(GenerationComponentRole::Ann)),
            receipt(GenerationComponentRole::Metadata),
        )
        .expect("joined receipts")
    }

    fn candidate(generation: u8, predecessor: Option<AuthorityRefV1>) -> GenerationCandidateV1 {
        candidate_with_ann(generation, predecessor, true)
    }

    fn candidate_with_ann(
        generation: u8,
        predecessor: Option<AuthorityRefV1>,
        with_ann: bool,
    ) -> GenerationCandidateV1 {
        GenerationCandidateV1::new(
            [generation; 16],
            predecessor,
            GenerationAuthorityActionV1::Activate,
            ArtifactGenerationIdentityV1::new(u64::from(generation), [generation; 16])
                .expect("artifact generation"),
            [0x33; 32],
            &joined(generation, with_ann),
        )
        .expect("candidate")
    }

    #[test]
    fn candidate_derives_manifest_from_the_joined_component_receipts() {
        let joined = joined(1, true);
        let candidate = candidate(1, None);
        let manifest = ActivationManifest::from_canonical_bytes(candidate.manifest_bytes())
            .expect("canonical manifest");
        verify_authority_manifest_reference(&candidate.authority(), &manifest)
            .expect("exact reference");
        assert_eq!(
            manifest.source_checkpoint_sha256(),
            joined.source_checkpoint()
        );
        assert_eq!(manifest.document_set_sha256(), joined.docset_digest());
        assert_eq!(manifest.components().vector, joined.vector().bytes);
        assert_eq!(manifest.components().lexical, joined.lexical().bytes);
        assert_eq!(
            manifest.components().ann,
            Some(joined.ann().expect("ANN").bytes)
        );
        assert_eq!(manifest.components().metadata, joined.metadata().bytes);
        assert_eq!(candidate.authority().sequence, 1);
        assert!(candidate.authority().predecessor.is_none());
    }

    #[test]
    fn candidate_budget_covers_every_component_and_the_manifest() {
        let candidate = candidate(1, None);
        let expected = candidate.authority().manifest_len
            + component_receipts(candidate.manifest())
                .map(|(_, receipt)| receipt.byte_len)
                .sum::<u64>();
        assert_eq!(candidate.owned_image_bytes(), expected);
        assert!(candidate.check_budget(expected).is_ok());
        assert_eq!(
            candidate.check_budget(expected - 1),
            Err(GenerationPublicationErrorV1::AdmissionBudget {
                required: expected,
                limit: expected - 1,
            })
        );
        assert!(candidate.check_budget(0).is_err());
    }

    #[test]
    fn absent_ann_is_not_relabelled_as_a_vector_component() {
        let candidate = candidate_with_ann(1, None, false);
        let manifest = ActivationManifest::from_canonical_bytes(candidate.manifest_bytes())
            .expect("exact generation is canonical");
        assert!(matches!(manifest, ActivationManifest::V2(_)));
        assert!(manifest.components().ann.is_none());
        assert_eq!(component_receipts(&manifest).count(), 3);
        let expected = candidate.authority().manifest_len
            + component_receipts(&manifest)
                .map(|(_, receipt)| receipt.byte_len)
                .sum::<u64>();
        assert_eq!(candidate.owned_image_bytes(), expected);
        assert!(candidate.check_budget(expected).is_ok());
        assert!(candidate.check_budget(expected - 1).is_err());
        verify_authority_manifest_reference(&candidate.authority(), &manifest)
            .expect("absence is bound by the authority");
    }

    #[test]
    fn successor_binds_the_full_predecessor_and_root() {
        let first = candidate(1, None);
        let second = candidate(2, Some(first.authority()));
        let pair = ExpectedAuthorityPairV1 {
            first: None,
            second: Some(AuthoritySlotV1::new(1, [7; 16], first.authority()).expect("slot")),
        };
        assert!(second.check_predecessor(pair, [7; 16]).is_ok());
        assert_eq!(second.authority().sequence, 2);
        assert_eq!(
            second.authority().predecessor,
            Some(first.authority().fingerprint())
        );
        assert_eq!(
            second.check_predecessor(ExpectedAuthorityPairV1::default(), [7; 16]),
            Err(GenerationPublicationErrorV1::ExpectedPredecessorMismatch)
        );
        assert_eq!(
            second.check_predecessor(pair, [8; 16]),
            Err(GenerationPublicationErrorV1::Authority(
                GenerationAuthorityErrorV1::RootMismatch
            ))
        );
        let fork = AuthorityRefV1::new(1, [9; 16], 42, [9; 32], None).expect("fork reference");
        let fork_pair = ExpectedAuthorityPairV1 {
            first: None,
            second: Some(AuthoritySlotV1::new(1, [7; 16], fork).expect("fork slot")),
        };
        assert_eq!(
            second.check_predecessor(fork_pair, [7; 16]),
            Err(GenerationPublicationErrorV1::ExpectedPredecessorMismatch)
        );
    }

    #[test]
    fn a_torn_slot_is_not_treated_as_an_absent_predecessor() {
        let root = [7; 16];
        let candidate = candidate(1, None);
        let slot = AuthoritySlotV1::new(1, root, candidate.authority()).expect("slot");
        let mut image = vec![0; 2 * GENERATION_AUTHORITY_SLOT_BYTES_V1];
        image[GENERATION_AUTHORITY_SLOT_BYTES_V1..].copy_from_slice(&slot.encode().expect("frame"));
        let observed = authenticated_pair(&image, root).expect("pair");
        assert_eq!(observed.second, Some(slot));
        image[GENERATION_AUTHORITY_SLOT_BYTES_V1 + 100] ^= 1;
        assert!(authenticated_pair(&image, root).is_err());
        assert!(authenticated_pair(&image[..image.len() - 1], root).is_err());
        assert!(authenticated_pair(&image, [8; 16]).is_err());
    }

    #[test]
    fn unmatched_attempt_is_refused_even_when_the_authority_did_not_change() {
        let root = [7; 16];
        let first = candidate(1, None);
        let pair = ExpectedAuthorityPairV1 {
            first: None,
            second: Some(AuthoritySlotV1::new(1, root, first.authority()).expect("slot")),
        };
        let mut bytes = vec![0; 2 * GENERATION_LOCK_FRAME_BYTES_V1];
        let attempt = GenerationLockFrameV1::new(
            GenerationLockFrameKindV1::Attempt,
            root,
            [8; 16],
            [9; 16],
            1,
            first.authority().fingerprint(),
        )
        .expect("attempt");
        bytes[GENERATION_LOCK_FRAME_BYTES_V1..].copy_from_slice(&attempt.encode().expect("encode"));
        assert_eq!(
            check_completed_attempt(&bytes, root, pair),
            Err(GenerationPublicationErrorV1::PendingReconciliation)
        );
        let owner = GenerationLockFrameV1::new(
            GenerationLockFrameKindV1::Owner,
            root,
            [8; 16],
            [9; 16],
            1,
            first.authority().fingerprint(),
        )
        .expect("owner");
        bytes[..GENERATION_LOCK_FRAME_BYTES_V1].copy_from_slice(&owner.encode().expect("encode"));
        assert!(check_completed_attempt(&bytes, root, pair).is_ok());
        let foreign_owner = GenerationLockFrameV1::new(
            GenerationLockFrameKindV1::Owner,
            root,
            [6; 16],
            [9; 16],
            1,
            first.authority().fingerprint(),
        )
        .expect("different writer");
        bytes[..GENERATION_LOCK_FRAME_BYTES_V1]
            .copy_from_slice(&foreign_owner.encode().expect("encode"));
        assert_eq!(
            check_completed_attempt(&bytes, root, pair),
            Err(GenerationPublicationErrorV1::PendingReconciliation)
        );
    }

    #[cfg(target_os = "linux")]
    mod disk {
        use super::*;
        use asupersync::test_utils::run_test_with_cx;
        use frankensearch_index::generation_root::generation_reader::component_object_name_v1;
        use frankensearch_index::generation_root::{
            GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationRootAnchorLayout,
        };
        use std::fs::{self, DirBuilder, OpenOptions};
        use std::io::Write as _;
        use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
        use std::path::{Path, PathBuf};
        use std::sync::OnceLock;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        const ROOT_ID: [u8; 16] = [0x71; 16];
        const WRITER_ID: [u8; 16] = [0x72; 16];
        static BASE: OnceLock<PathBuf> = OnceLock::new();
        static NEXT: AtomicU64 = AtomicU64::new(0);

        // Match the root substrate's tests: qualified private local storage,
        // no /tmp symlink shortcut, no deletion, no unsupported-filesystem skip.
        fn root_path() -> PathBuf {
            let base = BASE.get_or_init(|| {
                let home = PathBuf::from(std::env::var_os("HOME").expect("private HOME"));
                let time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos();
                let base = home.join(format!(
                    ".frankensearch-publication-tests-{}-{time}",
                    std::process::id()
                ));
                DirBuilder::new()
                    .mode(0o700)
                    .create(&base)
                    .expect("private parent");
                base
            });
            let root = base.join(format!("case-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
            DirBuilder::new()
                .mode(0o700)
                .create(&root)
                .expect("private root");
            for (name, len) in [
                ("LOCK", AUTHORITY_PUBLISHER_LOCK_BYTES_V1),
                ("AUTHORITY", GENERATION_ROOT_AUTHORITY_FILE_BYTES),
            ] {
                write_new(
                    &root.join(name),
                    &vec![0; usize::try_from(len).expect("length")],
                    0o600,
                );
            }
            root
        }

        fn write_new(path: &Path, bytes: &[u8], mode: u32) {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(mode)
                .open(path)
                .expect("new object");
            file.write_all(bytes).expect("object bytes");
            file.sync_all().expect("durable object");
        }

        fn stage(
            root: &Path,
            candidate: &GenerationCandidateV1,
            generation: u8,
            omit: Option<GenerationComponentRole>,
        ) {
            for (role, receipt) in component_receipts(candidate.manifest()) {
                if omit == Some(role) {
                    continue;
                }
                write_new(
                    &root.join(component_object_name_v1(role, receipt.sha256)),
                    &bytes(role, generation),
                    0o400,
                );
            }
            write_new(
                &root.join(candidate.manifest_name()),
                candidate.manifest_bytes(),
                0o400,
            );
        }

        fn admit(path: &Path) -> QualifiedGenerationRoot {
            QualifiedGenerationRoot::admit(
                path,
                GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1).expect("layout"),
            )
            .expect("qualified root")
        }

        fn session(root: &QualifiedGenerationRoot) -> GenerationPublisherV1<'_> {
            GenerationPublisherV1::new(
                root,
                ROOT_ID,
                WRITER_ID,
                GenerationRootSecurityProfileV1::CooperativeLocal,
            )
            .expect("session")
        }

        fn publish(
            session: &mut GenerationPublisherV1<'_>,
            cx: &Cx,
            candidate: &GenerationCandidateV1,
            expected: ExpectedAuthorityPairV1,
        ) -> GenerationPublicationResultV1 {
            session
                .publish(
                    cx,
                    candidate,
                    expected,
                    None,
                    [0x73; 16],
                    candidate.owned_image_bytes(),
                )
                .expect("publication")
        }

        #[test]
        fn complete_publication_and_fresh_reopen_retain_old_reader_bytes() {
            run_test_with_cx(|cx| async move {
                let path = root_path();
                let first = candidate(1, None);
                stage(&path, &first, 1, None);
                let root = admit(&path);
                let mut session = session(&root);
                let result = publish(
                    &mut session,
                    &cx,
                    &first,
                    ExpectedAuthorityPairV1::default(),
                );
                assert!(matches!(
                    result.publication,
                    PublicationOutcomeV1::Committed { .. }
                ));
                assert!(matches!(
                    result.activation,
                    Some(GenerationActivationV1::Installed(_))
                ));
                let old = session.snapshot().expect("old reader");
                let second = candidate(2, Some(first.authority()));
                stage(&path, &second, 2, None);
                let result = publish(&mut session, &cx, &second, old.authority_pair());
                assert!(matches!(
                    result.publication,
                    PublicationOutcomeV1::Committed { .. }
                ));
                let new = session.snapshot().expect("new reader");
                assert_eq!(new.head().authority, second.authority());
                assert_eq!(old.head().authority, first.authority());
                for (role, _) in component_receipts(first.manifest()) {
                    assert_eq!(
                        old.closure()
                            .bytes(role)
                            .expect("declared old role")
                            .as_ref(),
                        bytes(role, 1)
                    );
                    assert_eq!(
                        new.closure()
                            .bytes(role)
                            .expect("declared new role")
                            .as_ref(),
                        bytes(role, 2)
                    );
                }
                let fresh_root = admit(&path);
                let reopened = fresh_root
                    .open_generation_snapshot(
                        ROOT_ID,
                        GenerationRootSecurityProfileV1::CooperativeLocal,
                        None,
                    )
                    .expect("fresh open");
                let SnapshotOpenOutcomeV1::Opened(reopened) = reopened else {
                    panic!("fresh admission refused"); // ubs:ignore — cfg(test) assertion: a refused fresh open must fail the publication test.
                };
                assert_eq!(reopened.head().authority, second.authority());
            });
        }

        #[test]
        fn exact_publication_reopens_and_queries_the_retained_fsvi_owner() {
            use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
            use frankensearch_index::exact_component_adapters::vector_component_receipt;
            use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

            run_test_with_cx(|cx| async move {
                let path = root_path();
                let generation = ArtifactGenerationIdentityV1::new(1, [1; 16]).expect("generation");
                let mut identity =
                    EmbeddingIdentityBundleV1::explicit_test_model("exact-publication", 2);
                "fsvi-v2".clone_into(&mut identity.storage.format);
                identity.storage.quantization = QuantizationFormat::F16;
                "little-endian".clone_into(&mut identity.storage.endianness);
                let binding =
                    FsviV2IdentityBinding::new(generation, identity.freeze().expect("identity"))
                        .expect("binding");
                // Producer staging lives outside the sealed generation root.
                let source = path.with_extension("source-fsvi");
                let mut writer = VectorIndex::create_v2(&source, binding.clone()).expect("writer");
                writer
                    .write_record("doc-a", &[1.0, 0.0])
                    .expect("first vector");
                writer
                    .write_record("doc-b", &[0.0, 1.0])
                    .expect("second vector");
                writer.finish().expect("complete vector image");
                let vector_bytes: Arc<[u8]> = fs::read(&source).expect("producer image").into();
                let owner = ValidatedFsviBytes::from_arc(Arc::clone(&vector_bytes), &binding)
                    .expect("owner");
                let checkpoint = SourceCheckpointV1::derive(&CommitRange { low: 1, high: 1 });
                let vector =
                    vector_component_receipt(owner.witness(), ["doc-a", "doc-b"], checkpoint)
                        .expect("engine-derived vector receipt");
                // Lexical and metadata stay opaque publication fixtures here;
                // this test exercises retained FSVI serving, not their parsers.
                let peers = joined(1, false);
                let components = ExactGenerationComponentsV1::admit(
                    vector,
                    peers.lexical().clone(),
                    None,
                    peers.metadata().clone(),
                )
                .expect("same source and ordered documents");
                let candidate = GenerationCandidateV1::new(
                    [1; 16],
                    None,
                    GenerationAuthorityActionV1::Activate,
                    generation,
                    [0x33; 32],
                    &components,
                )
                .expect("exact candidate");
                stage(&path, &candidate, 1, Some(GenerationComponentRole::Vector));
                write_new(
                    &path.join(component_object_name_v1(
                        GenerationComponentRole::Vector,
                        candidate.manifest().components().vector.sha256,
                    )),
                    &vector_bytes,
                    0o400,
                );
                let root = admit(&path);
                let mut publisher = session(&root);
                let result = publish(
                    &mut publisher,
                    &cx,
                    &candidate,
                    ExpectedAuthorityPairV1::default(),
                );
                assert!(matches!(
                    result.publication,
                    PublicationOutcomeV1::Committed { .. }
                ));
                assert!(matches!(
                    result.activation,
                    Some(GenerationActivationV1::Installed(_))
                ));
                drop(result);
                drop(publisher);
                drop(root);
                drop(owner);
                drop(vector_bytes);

                let root = admit(&path);
                let SnapshotOpenOutcomeV1::Opened(snapshot) = root
                    .open_generation_snapshot(
                        ROOT_ID,
                        GenerationRootSecurityProfileV1::CooperativeLocal,
                        None,
                    )
                    .expect("fresh authority resolution")
                else {
                    panic!("exact generation must reopen"); // ubs:ignore — cfg(test) refusal must fail.
                };
                assert!(snapshot.manifest().components().ann.is_none());
                assert!(
                    snapshot
                        .closure()
                        .bytes(GenerationComponentRole::Ann)
                        .is_none()
                );
                let retained = ValidatedFsviBytes::from_arc(
                    snapshot
                        .closure()
                        .bytes(GenerationComponentRole::Vector)
                        .expect("retained vector"),
                    &binding,
                )
                .expect("admit exact retained bytes without reopening vector path");
                assert_eq!(
                    retained.witness().generation,
                    snapshot.manifest().generation()
                );
                assert_eq!(
                    retained
                        .search_top_k(&[1.0, 0.0], 1, None)
                        .expect("first query")[0]
                        .doc_id,
                    "doc-a"
                );
                assert_eq!(
                    retained
                        .search_top_k(&[0.0, 1.0], 1, None)
                        .expect("second query")[0]
                        .doc_id,
                    "doc-b"
                );
                assert!(!fs::read_dir(&path).expect("root inventory").any(|entry| {
                    entry
                        .expect("entry")
                        .path()
                        .extension()
                        .is_some_and(|extension| extension == "ann")
                }));
            });
        }

        #[test]
        fn publication_switches_ann_exact_and_ann_without_borrowing_historical_graphs() {
            run_test_with_cx(|cx| async move {
                let path = root_path();
                let root = admit(&path);
                let mut publisher = session(&root);
                let mut retained: Vec<Arc<OpenedGenerationSnapshotV1>> = Vec::new();
                for (generation, with_ann) in [(1, true), (2, false), (3, true)] {
                    let predecessor = retained.last().map(|old| old.head().authority);
                    let expected = retained
                        .last()
                        .map_or(ExpectedAuthorityPairV1::default(), |old| {
                            old.authority_pair()
                        });
                    let candidate = candidate_with_ann(generation, predecessor, with_ann);
                    stage(&path, &candidate, generation, None);
                    let result = publish(&mut publisher, &cx, &candidate, expected);
                    assert!(matches!(
                        result.publication,
                        PublicationOutcomeV1::Committed { .. }
                    ));
                    let installed = publisher.snapshot().expect("installed complete generation");
                    assert_eq!(installed.manifest().components().ann.is_some(), with_ann);
                    assert_eq!(
                        installed
                            .closure()
                            .bytes(GenerationComponentRole::Ann)
                            .is_some(),
                        with_ann
                    );
                    let SnapshotOpenOutcomeV1::Opened(fresh) = root
                        .open_generation_snapshot(
                            ROOT_ID,
                            GenerationRootSecurityProfileV1::CooperativeLocal,
                            None,
                        )
                        .expect("fresh open")
                    else {
                        panic!("published generation must reopen"); // ubs:ignore — cfg(test) refusal must fail.
                    };
                    assert_eq!(fresh.head().authority, candidate.authority());
                    assert_eq!(
                        fresh
                            .closure()
                            .bytes(GenerationComponentRole::Ann)
                            .is_some(),
                        with_ann
                    );
                    retained.push(installed);
                }
                assert!(
                    retained[1]
                        .closure()
                        .bytes(GenerationComponentRole::Ann)
                        .is_none()
                );
                for (snapshot, generation) in [(&retained[0], 1), (&retained[2], 3)] {
                    assert_eq!(
                        snapshot
                            .closure()
                            .bytes(GenerationComponentRole::Ann)
                            .expect("selected graph")
                            .as_ref(),
                        bytes(GenerationComponentRole::Ann, generation)
                    );
                }
                assert_ne!(retained[0].head().authority, retained[2].head().authority);
            });
        }

        #[test]
        fn corrupt_declared_ann_never_becomes_an_exact_publication() {
            run_test_with_cx(|cx| async move {
                let path = root_path();
                let first = candidate_with_ann(1, None, false);
                stage(&path, &first, 1, None);
                let root = admit(&path);
                let mut publisher = session(&root);
                let _ = publish(
                    &mut publisher,
                    &cx,
                    &first,
                    ExpectedAuthorityPairV1::default(),
                );
                let old = publisher.snapshot().expect("exact reader");
                let second = candidate(2, Some(first.authority()));
                stage(&path, &second, 2, Some(GenerationComponentRole::Ann));
                let receipt = second.manifest().components().ann.expect("declared ANN");
                let mut wrong_graph = bytes(GenerationComponentRole::Ann, 2);
                wrong_graph[0] ^= 1;
                write_new(
                    &path.join(component_object_name_v1(
                        GenerationComponentRole::Ann,
                        receipt.sha256,
                    )),
                    &wrong_graph,
                    0o400,
                );
                let authority = fs::read(path.join("AUTHORITY")).expect("authority");
                let lock = fs::read(path.join("LOCK")).expect("lock");
                let result = publisher.publish(
                    &cx,
                    &second,
                    old.authority_pair(),
                    None,
                    [0x74; 16],
                    second.owned_image_bytes(),
                );
                assert!(matches!(result, Err(GenerationPublicationErrorV1::Root(_))));
                assert_eq!(
                    fs::read(path.join("AUTHORITY")).expect("authority"),
                    authority
                );
                assert_eq!(fs::read(path.join("LOCK")).expect("lock"), lock);
                assert!(Arc::ptr_eq(
                    &old,
                    &publisher.snapshot().expect("previous reader")
                ));
                assert!(old.closure().file(GenerationComponentRole::Ann).is_none());
            });
        }

        #[test]
        fn missing_component_never_switches_authority_or_replaces_the_reader() {
            run_test_with_cx(|cx| async move {
                for missing in [
                    GenerationComponentRole::Vector,
                    GenerationComponentRole::Lexical,
                    GenerationComponentRole::Ann,
                    GenerationComponentRole::Metadata,
                ] {
                    let path = root_path();
                    let first = candidate(1, None);
                    stage(&path, &first, 1, None);
                    let root = admit(&path);
                    let mut session = session(&root);
                    let _ = publish(
                        &mut session,
                        &cx,
                        &first,
                        ExpectedAuthorityPairV1::default(),
                    );
                    let old = session.snapshot().expect("old");
                    let second = candidate(2, Some(first.authority()));
                    stage(&path, &second, 2, Some(missing));
                    let before_authority = fs::read(path.join("AUTHORITY")).expect("authority");
                    let before_lock = fs::read(path.join("LOCK")).expect("lock");
                    let result = session.publish(
                        &cx,
                        &second,
                        old.authority_pair(),
                        None,
                        [0x74; 16],
                        second.owned_image_bytes(),
                    );
                    assert!(
                        matches!(result, Err(GenerationPublicationErrorV1::Root(_))),
                        "missing {missing:?}: {result:?}"
                    );
                    assert_eq!(
                        fs::read(path.join("AUTHORITY")).expect("authority"),
                        before_authority
                    );
                    assert_eq!(fs::read(path.join("LOCK")).expect("lock"), before_lock);
                    assert!(Arc::ptr_eq(&old, &session.snapshot().expect("retained")));
                }
            });
        }

        #[test]
        fn cancelled_or_over_budget_candidate_leaves_both_anchors_unchanged() {
            run_test_with_cx(|cx| async move {
                let path = root_path();
                let root = admit(&path);
                let mut session = session(&root);
                let candidate = candidate(1, None);
                // Do not stage anything: both early refusals must precede file admission.
                let authority = fs::read(path.join("AUTHORITY")).expect("authority");
                let lock = fs::read(path.join("LOCK")).expect("lock");
                let budget = session.publish(
                    &cx,
                    &candidate,
                    ExpectedAuthorityPairV1::default(),
                    None,
                    [0x75; 16],
                    0,
                );
                assert!(matches!(
                    budget,
                    Err(GenerationPublicationErrorV1::AdmissionBudget { .. })
                ));
                cx.set_cancel_requested(true);
                let cancelled = session.publish(
                    &cx,
                    &candidate,
                    ExpectedAuthorityPairV1::default(),
                    None,
                    [0x75; 16],
                    candidate.owned_image_bytes(),
                );
                assert!(matches!(
                    cancelled,
                    Err(GenerationPublicationErrorV1::Cancelled { .. })
                ));
                cx.set_cancel_requested(false);
                assert_eq!(
                    fs::read(path.join("AUTHORITY")).expect("authority"),
                    authority
                );
                assert_eq!(fs::read(path.join("LOCK")).expect("lock"), lock);
                assert!(session.snapshot().is_none());
            });
        }
    }
}
