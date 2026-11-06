//! Exact selected source admission for rebuilding damaged derived components.
//!
//! These capabilities deliberately cannot become search snapshots: source-only
//! recovery omits vectors and ANN, while vector recovery omits ANN. A repair
//! must build a fresh complete generation and publish it against the captured
//! physical authority pair.

use super::{
    ActivationManifest, ArtifactGenerationIdentityV1, AuthorityRefV1, AuthoritySlotV1,
    ExpectedAuthorityPairV1, GenerationAuthorityErrorV1, GenerationComponentRole,
    GenerationFileExpectation, GenerationRootAnchorWitness, GenerationRootErrorKind,
    GenerationRootObjectWitness, GenerationRootResult, GenerationRootSecurityProfileV1,
    QualifiedGenerationFile, QualifiedGenerationRoot, Round, SNAPSHOT_OPEN_MAX_ATTEMPTS_V1,
    SnapshotRefusalV1, activation_manifest_name_v1, activation_manifest_path_v1, admit_component,
    check_closure, check_image_budget, component_object_name_v1, same_snapshot_anchors,
    verify_authority_manifest_reference,
};
use super::{GENERATION_ROOT_AUTHORITY_FILE_NAME, GENERATION_ROOT_LOCK_FILE_NAME};
use std::fmt;
use std::sync::Arc;

/// Exact selected control state and authenticated source images for recovery.
///
/// Only the activation manifest, lexical image and metadata image are owned.
/// Engine-specific source admission must still prove their document set,
/// source checkpoint and identities before exporting documents. This value
/// has no query interface and no conversion to a complete generation snapshot.
/// It is neither serializable nor reconstructible from an unauthenticated path.
pub struct OpenedGenerationSourceV1 {
    root: GenerationRootObjectWitness,
    anchors: GenerationRootAnchorWitness,
    root_id: [u8; 16],
    profile: GenerationRootSecurityProfileV1,
    pair: ExpectedAuthorityPairV1,
    head: AuthoritySlotV1,
    manifest_file: QualifiedGenerationFile,
    manifest: ActivationManifest,
    lexical: QualifiedGenerationFile,
    metadata: QualifiedGenerationFile,
}

impl OpenedGenerationSourceV1 {
    /// Physical root admitted for this recovery selection.
    #[must_use]
    pub const fn root_witness(&self) -> GenerationRootObjectWitness {
        self.root
    }

    /// Control anchor identities retained across both resolution rounds.
    #[must_use]
    pub const fn anchor_witnesses(&self) -> GenerationRootAnchorWitness {
        self.anchors
    }

    /// Root identity authenticated by the selected control state.
    #[must_use]
    pub const fn root_id(&self) -> [u8; 16] {
        self.root_id
    }

    /// Explicitly supported recovery profile; no external floor is bypassed.
    #[must_use]
    pub const fn profile(&self) -> GenerationRootSecurityProfileV1 {
        self.profile
    }

    /// Physical pair a repair must preserve as its publication expectation.
    #[must_use]
    pub const fn authority_pair(&self) -> ExpectedAuthorityPairV1 {
        self.pair
    }

    /// Exact current authority and real slot selected for recovery.
    #[must_use]
    pub const fn head(&self) -> AuthoritySlotV1 {
        self.head
    }

    /// Authenticated current activation manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ActivationManifest {
        &self.manifest
    }

    /// Retained immutable activation image.
    #[must_use]
    pub fn manifest_bytes(&self) -> Arc<[u8]> {
        self.manifest_file.bytes()
    }

    /// Retained lexical image at the activation manifest's exact receipt.
    #[must_use]
    pub fn lexical_bytes(&self) -> Arc<[u8]> {
        self.lexical.bytes()
    }

    /// Retained metadata image at the activation manifest's exact receipt.
    #[must_use]
    pub fn metadata_bytes(&self) -> Arc<[u8]> {
        self.metadata.bytes()
    }

    /// Owned encoded image bytes, excluding derived components and parsed memory.
    #[must_use]
    pub fn retained_image_bytes(&self) -> u64 {
        self.manifest_file
            .witness()
            .byte_len()
            .saturating_add(self.lexical.witness().byte_len())
            .saturating_add(self.metadata.witness().byte_len())
    }
}

impl fmt::Debug for OpenedGenerationSourceV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenedGenerationSourceV1")
            .field("root", &self.root)
            .field("head_sequence", &self.head.authority.sequence)
            .field("head_slot", &self.head.slot_index)
            .field("retained_image_bytes", &self.retained_image_bytes())
            .finish_non_exhaustive()
    }
}

/// Result of an explicit source recovery open; never a partial search snapshot.
#[derive(Debug)]
pub enum SourceOpenOutcomeV1 {
    /// Authenticated source images and their frozen publication expectation.
    Opened(Arc<OpenedGenerationSourceV1>),
    /// Typed refusal without usable source state.
    Refused(SnapshotRefusalV1),
}

/// Selected authenticated source and vector images for model-free recovery.
///
/// No ANN bytes are admitted and this value cannot become a search snapshot.
/// An engine must validate the vector image's full identity and join it to the
/// retained source checkpoint and docset before rebuilding derived indexes.
pub struct OpenedGenerationVectorSourceV1 {
    source: Arc<OpenedGenerationSourceV1>,
    vector: QualifiedGenerationFile,
}

impl OpenedGenerationVectorSourceV1 {
    /// Source images and the frozen physical publication expectation.
    #[must_use]
    pub const fn source(&self) -> &Arc<OpenedGenerationSourceV1> {
        &self.source
    }

    /// Exact vector image at the selected manifest's receipt.
    #[must_use]
    pub fn vector_bytes(&self) -> Arc<[u8]> {
        self.vector.bytes()
    }

    /// Total encoded source/vector bytes, excluding ANN and parsed memory.
    #[must_use]
    pub fn retained_image_bytes(&self) -> u64 {
        self.source
            .retained_image_bytes()
            .saturating_add(self.vector.witness().byte_len())
    }
}

impl fmt::Debug for OpenedGenerationVectorSourceV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenedGenerationVectorSourceV1")
            .field("source", &self.source)
            .field("retained_image_bytes", &self.retained_image_bytes())
            .finish_non_exhaustive()
    }
}

/// Result of source/vector admission with no query or ANN capability.
#[derive(Debug)]
pub enum VectorSourceOpenOutcomeV1 {
    /// Authenticated immutable source/vector images and frozen authority pair.
    Opened(Arc<OpenedGenerationVectorSourceV1>),
    /// Typed refusal without usable recovery state.
    Refused(SnapshotRefusalV1),
}

impl QualifiedGenerationRoot {
    /// Admit healthy source and vectors even when the selected ANN is damaged.
    ///
    /// Source admission freezes the exact current head, pair and physical
    /// anchors. Vector admission then checks the manifest receipt and a final
    /// shared-lock resolution of that same selection. A newer winner is never
    /// adopted. The budget includes manifest, lexical, metadata and vector
    /// encoded bytes and is checked before vector I/O; ANN is never read.
    /// Unread ANN objects still undergo regular-file and alias inventory checks.
    ///
    /// This read-only capability supports `CooperativeLocal` without an
    /// external floor, exactly like source-only recovery. Engine-level vector
    /// and source identity validation remains mandatory before a repair build.
    ///
    /// # Errors
    /// Returns infrastructure errors or typed source, vector, budget, closure
    /// and concurrent-authority refusals.
    pub fn open_expected_generation_vector_source_for_recovery_with_budget(
        &self,
        root_id: [u8; 16],
        expected: AuthorityRefV1,
        expected_generation: ArtifactGenerationIdentityV1,
        max_owned_image_bytes: u64,
    ) -> GenerationRootResult<VectorSourceOpenOutcomeV1> {
        let source = match self.open_expected_generation_source_for_recovery_with_budget(
            root_id,
            expected,
            expected_generation,
            max_owned_image_bytes,
        )? {
            SourceOpenOutcomeV1::Opened(source) => source,
            SourceOpenOutcomeV1::Refused(refusal) => {
                return Ok(VectorSourceOpenOutcomeV1::Refused(refusal));
            }
        };
        let receipts = source.manifest().components();
        if let Err(refusal) = check_image_budget(
            expected.manifest_len,
            [
                receipts.lexical.byte_len,
                receipts.metadata.byte_len,
                receipts.vector.byte_len,
            ],
            max_owned_image_bytes,
        ) {
            return Ok(VectorSourceOpenOutcomeV1::Refused(refusal));
        }
        let vector = match admit_component(self, GenerationComponentRole::Vector, receipts.vector)?
        {
            Ok(file) => file,
            Err(refusal) => return Ok(VectorSourceOpenOutcomeV1::Refused(refusal)),
        };
        for _ in 0..SNAPSHOT_OPEN_MAX_ATTEMPTS_V1 {
            let (resolved, anchors, inventory) =
                match self.resolution_round(root_id, source.profile(), None)? {
                    Round::Resolved(resolved) => *resolved,
                    Round::Retry => continue,
                    Round::Refused(refusal) => {
                        return Ok(VectorSourceOpenOutcomeV1::Refused(refusal));
                    }
                };
            if resolved.pair != source.authority_pair() || resolved.head != source.head() {
                return Ok(VectorSourceOpenOutcomeV1::Refused(
                    SnapshotRefusalV1::ExpectedCurrentChanged,
                ));
            }
            if !same_snapshot_anchors(&source.anchor_witnesses(), &anchors) {
                return Ok(VectorSourceOpenOutcomeV1::Refused(
                    SnapshotRefusalV1::Closure(GenerationRootErrorKind::ObjectChanged),
                ));
            }
            let manifest_name = activation_manifest_name_v1(expected.object_id);
            let lexical_name =
                component_object_name_v1(GenerationComponentRole::Lexical, receipts.lexical.sha256);
            let metadata_name = component_object_name_v1(
                GenerationComponentRole::Metadata,
                receipts.metadata.sha256,
            );
            let vector_name =
                component_object_name_v1(GenerationComponentRole::Vector, receipts.vector.sha256);
            let declared = [
                GENERATION_ROOT_LOCK_FILE_NAME,
                GENERATION_ROOT_AUTHORITY_FILE_NAME,
                manifest_name.as_str(),
                lexical_name.as_str(),
                metadata_name.as_str(),
                vector_name.as_str(),
            ];
            if let Err(violation) = check_closure(&inventory, self.witness().device(), &declared) {
                return Ok(VectorSourceOpenOutcomeV1::Refused(
                    SnapshotRefusalV1::ClosureViolation(violation),
                ));
            }
            return Ok(VectorSourceOpenOutcomeV1::Opened(Arc::new(
                OpenedGenerationVectorSourceV1 { source, vector },
            )));
        }
        Ok(VectorSourceOpenOutcomeV1::Refused(
            SnapshotRefusalV1::ConcurrentSwitch {
                attempts: SNAPSHOT_OPEN_MAX_ATTEMPTS_V1,
            },
        ))
    }

    /// Admit selected source while vector or ANN components may be damaged.
    ///
    /// The caller explicitly selects the current authority and artifact
    /// generation. The activation manifest and its lexical/metadata objects
    /// receive exact length/digest, descriptor-relative admission. Derived
    /// images are never read, so this also supports repairing generation one
    /// without a healthy ancestor. Missing derived objects are permitted;
    /// any surviving objects still undergo inventory checks for regular files,
    /// aliases, devices, retained object names and write-ahead logs.
    ///
    /// Two shared-lock resolutions must agree on the entire physical pair and
    /// anchors. The first resolved state remains frozen across transient
    /// retries: a newer winner is refused instead of becoming a repair base.
    /// Pending publication refuses recovery. Only `CooperativeLocal` without
    /// an external floor is supported, matching explicit local recovery.
    ///
    /// The budget covers the activation, lexical and metadata encoded images,
    /// and is checked before each class of reads. It excludes parsed source
    /// memory, which the engine adapter must bound separately. This operation
    /// does not mutate control state or admit a queryable generation.
    ///
    /// # Errors
    /// Returns infrastructure errors or typed stale-head, manifest, closure,
    /// image-budget and control-state refusals.
    pub fn open_expected_generation_source_for_recovery_with_budget(
        &self,
        root_id: [u8; 16],
        expected: AuthorityRefV1,
        expected_generation: ArtifactGenerationIdentityV1,
        max_owned_image_bytes: u64,
    ) -> GenerationRootResult<SourceOpenOutcomeV1> {
        if root_id == [0; 16] {
            return Ok(SourceOpenOutcomeV1::Refused(SnapshotRefusalV1::Authority(
                GenerationAuthorityErrorV1::InvalidField {
                    field: "generation_reader.source_recovery.root_id",
                },
            )));
        }
        if let Err(error) = expected.validate() {
            return Ok(SourceOpenOutcomeV1::Refused(SnapshotRefusalV1::Authority(
                error,
            )));
        }
        let profile = GenerationRootSecurityProfileV1::CooperativeLocal;
        let mut fence = None;
        for _ in 0..SNAPSHOT_OPEN_MAX_ATTEMPTS_V1 {
            let (first, first_anchors, _) = match self.resolution_round(root_id, profile, None)? {
                Round::Resolved(resolved) => *resolved,
                Round::Retry => continue,
                Round::Refused(refusal) => return Ok(SourceOpenOutcomeV1::Refused(refusal)),
            };
            if first.head.authority != expected {
                return Ok(SourceOpenOutcomeV1::Refused(
                    SnapshotRefusalV1::ExpectedCurrentChanged,
                ));
            }
            if let Some((captured, captured_anchors)) = fence {
                if first != captured {
                    return Ok(SourceOpenOutcomeV1::Refused(
                        SnapshotRefusalV1::ExpectedCurrentChanged,
                    ));
                }
                if !same_snapshot_anchors(&first_anchors, &captured_anchors) {
                    return Ok(SourceOpenOutcomeV1::Refused(SnapshotRefusalV1::Closure(
                        GenerationRootErrorKind::ObjectChanged,
                    )));
                }
            } else {
                fence = Some((first, first_anchors));
            }
            if let Err(refusal) =
                check_image_budget(expected.manifest_len, [], max_owned_image_bytes)
            {
                return Ok(SourceOpenOutcomeV1::Refused(refusal));
            }
            let path = activation_manifest_path_v1(expected.object_id)?;
            let expectation = GenerationFileExpectation::immutable(
                expected.manifest_len,
                expected.manifest_sha256,
            )?;
            let manifest_file = match self.admit_file(&path, expectation) {
                Ok(file) => file,
                Err(error) => {
                    return Ok(SourceOpenOutcomeV1::Refused(SnapshotRefusalV1::Closure(
                        error.kind(),
                    )));
                }
            };
            let manifest = match ActivationManifest::from_canonical_bytes(manifest_file.as_bytes())
                .and_then(|manifest| {
                    verify_authority_manifest_reference(&expected, &manifest).map(|()| manifest)
                }) {
                Ok(manifest) => manifest,
                Err(error) => {
                    return Ok(SourceOpenOutcomeV1::Refused(SnapshotRefusalV1::Manifest(
                        error,
                    )));
                }
            };
            if manifest.generation() != expected_generation {
                return Ok(SourceOpenOutcomeV1::Refused(SnapshotRefusalV1::Manifest(
                    GenerationAuthorityErrorV1::InvalidField {
                        field: "generation_reader.source_recovery.generation",
                    },
                )));
            }
            let receipts = manifest.components();
            if let Err(refusal) = check_image_budget(
                expected.manifest_len,
                [receipts.lexical.byte_len, receipts.metadata.byte_len],
                max_owned_image_bytes,
            ) {
                return Ok(SourceOpenOutcomeV1::Refused(refusal));
            }
            let lexical =
                match admit_component(self, GenerationComponentRole::Lexical, receipts.lexical)? {
                    Ok(file) => file,
                    Err(refusal) => return Ok(SourceOpenOutcomeV1::Refused(refusal)),
                };
            let metadata = match admit_component(
                self,
                GenerationComponentRole::Metadata,
                receipts.metadata,
            )? {
                Ok(file) => file,
                Err(refusal) => return Ok(SourceOpenOutcomeV1::Refused(refusal)),
            };
            let (second, anchors, inventory) =
                match self.resolution_round(root_id, profile, None)? {
                    Round::Resolved(resolved) => *resolved,
                    Round::Retry => continue,
                    Round::Refused(refusal) => return Ok(SourceOpenOutcomeV1::Refused(refusal)),
                };
            if second != first {
                return Ok(SourceOpenOutcomeV1::Refused(
                    SnapshotRefusalV1::ExpectedCurrentChanged,
                ));
            }
            if !same_snapshot_anchors(&first_anchors, &anchors) {
                return Ok(SourceOpenOutcomeV1::Refused(SnapshotRefusalV1::Closure(
                    GenerationRootErrorKind::ObjectChanged,
                )));
            }
            // Only objects actually admitted above can use the declared-name
            // exemption. Unread vector/ANN entries must independently pass the
            // inventory's regular-file and single-link checks.
            let manifest_name = activation_manifest_name_v1(expected.object_id);
            let lexical_name =
                component_object_name_v1(GenerationComponentRole::Lexical, receipts.lexical.sha256);
            let metadata_name = component_object_name_v1(
                GenerationComponentRole::Metadata,
                receipts.metadata.sha256,
            );
            let declared = [
                GENERATION_ROOT_LOCK_FILE_NAME,
                GENERATION_ROOT_AUTHORITY_FILE_NAME,
                manifest_name.as_str(),
                lexical_name.as_str(),
                metadata_name.as_str(),
            ];
            if let Err(violation) = check_closure(&inventory, self.witness().device(), &declared) {
                return Ok(SourceOpenOutcomeV1::Refused(
                    SnapshotRefusalV1::ClosureViolation(violation),
                ));
            }
            #[cfg_attr(
                not(any(target_os = "linux", target_os = "macos")),
                allow(unreachable_code)
            )]
            let source = OpenedGenerationSourceV1 {
                root: self.witness(),
                anchors,
                root_id,
                profile,
                pair: second.pair,
                head: second.head,
                manifest_file,
                manifest,
                lexical,
                metadata,
            };
            return Ok(SourceOpenOutcomeV1::Opened(Arc::new(source)));
        }
        Ok(SourceOpenOutcomeV1::Refused(
            SnapshotRefusalV1::ConcurrentSwitch {
                attempts: SNAPSHOT_OPEN_MAX_ATTEMPTS_V1,
            },
        ))
    }
}
