//! Exact offline recovery scope for immutable-authority identity migrations.
//!
//! This module performs no I/O and does not attest a caller's inputs. The
//! executor must derive them from one retained qualified root, admitted old
//! images, independently verified local providers, the frozen complete
//! candidate, and the original request. It must compare that scope again at
//! the publication boundary and keep the original authority CAS expectation.

use serde::{Serialize, Serializer};

use super::{
    NetworkPolicy, RecoveryContractError, RecoveryPlan, RecoveryRequest, RetrievalTopology,
    Retryability, SemanticReadiness, TrustedRecoveryContext, VerifiedSemanticProvenance, plan,
    simple_action,
};
use crate::generation::{
    ArtifactGenerationIdentityV1, AuthorityRefV1, CommitRange, EmbeddingSpaceKindV1,
    FrozenEmbeddingIdentityBundleV1,
};

/// Origin of the complete, already prepared source cohort for this rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImmutableReindexSource {
    /// Reuse exactly the retained corpus and its input-preparation contract.
    Retained,
    /// The caller supplied a complete, freshly prepared source cohort.
    Prepared,
}

/// Complete old and replacement identity for one retained semantic tier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImmutableReindexTierBinding {
    /// Identity independently admitted from the retained generation.
    pub previous: FrozenEmbeddingIdentityBundleV1,
    /// Identity independently supplied by the loaded replacement provider and
    /// matched to the candidate's persisted encoding.
    pub replacement: FrozenEmbeddingIdentityBundleV1,
    /// Verified provenance of the replacement provider. This executor admits
    /// only local artifacts; a remote or daemon assertion cannot substitute.
    pub provenance: VerifiedSemanticProvenance,
}

/// Path-free facts identifying one complete immutable reindex attempt.
///
/// These are trusted runtime inputs, not an attestation format. In particular,
/// the executor must check that the actual candidate manifest authenticates
/// the named generation, source, census, identities and all component bytes.
/// The reference check here proves the exact predecessor relationship only.
/// No query text, filter text, model directory or corpus path belongs here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImmutableReindexBinding {
    /// Persistent logical identity admitted on the retained qualified root.
    pub root_id: [u8; 16],
    /// Exact current selection captured before candidate construction.
    #[serde(serialize_with = "serialize_authority")]
    pub expected_authority: AuthorityRefV1,
    /// Exact prepared activation whose manifest authenticates the candidate.
    #[serde(serialize_with = "serialize_authority")]
    pub candidate_authority: AuthorityRefV1,
    /// Artifact generation admitted from the old selection.
    pub expected_generation: ArtifactGenerationIdentityV1,
    /// Fresh complete artifact generation named by the candidate manifest.
    pub candidate_generation: ArtifactGenerationIdentityV1,
    /// Source checkpoint admitted from the old selection.
    pub previous_source: CommitRange,
    /// Complete candidate source checkpoint; neither boundary may move back.
    pub source: CommitRange,
    /// Old live-document census, independently admitted from its components.
    pub previous_documents: u64,
    /// Nonzero complete candidate live-document census across every tier.
    pub documents: u64,
    /// Whether input preparation may change for the complete new corpus.
    pub source_kind: ImmutableReindexSource,
    /// Digest of the exact original request, frozen before candidate work.
    pub original_request_sha256: [u8; 32],
    /// Digest of its full query policy, including filter, retrieval selection,
    /// result limit and response policy. Replaying a different policy is not
    /// authorized by this binding.
    pub query_policy_sha256: [u8; 32],
    /// Fast-only uses `fast`; quality-only uses `quality`; full uses both.
    pub fast: Option<ImmutableReindexTierBinding>,
    /// Independent quality identity whenever that tier exists in the topology.
    pub quality: Option<ImmutableReindexTierBinding>,
}

fn serialize_authority<S>(authority: &AuthorityRefV1, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    authority.canonical_bytes().as_slice().serialize(serializer)
}

fn invalid(field: &'static str) -> RecoveryContractError {
    RecoveryContractError::InvalidImmutableReindexBinding { field }
}

impl ImmutableReindexBinding {
    /// Complete topology named by the two independent tier slots.
    ///
    /// # Errors
    ///
    /// Refuses an absent semantic topology; there is no partial-quality,
    /// lexical-only, hash-control or ANN-only recovery scope.
    pub fn topology(&self) -> Result<RetrievalTopology, RecoveryContractError> {
        match (&self.fast, &self.quality) {
            (Some(_), None) => Ok(RetrievalTopology::FastOnly),
            (None, Some(_)) => Ok(RetrievalTopology::QualityOnly),
            (Some(_), Some(_)) => Ok(RetrievalTopology::FullProgressive),
            (None, None) => Err(invalid("topology")),
        }
    }

    /// Validate this one complete, forward, nonempty identity migration.
    ///
    /// # Errors
    ///
    /// Rejects malformed authority or generation identities, a candidate
    /// outside the exact next authority slot, source rollback, empty target
    /// corpus, unverified/non-semantic producers, unchanged query identities,
    /// and relabeled retained input. Errors expose only bounded field names.
    pub fn validate(&self) -> Result<(), RecoveryContractError> {
        self.topology()?;
        if self.root_id == [0; 16] {
            return Err(invalid("root_id"));
        }
        self.expected_authority
            .validate()
            .map_err(|_| invalid("expected_authority"))?;
        self.candidate_authority
            .validate()
            .map_err(|_| invalid("candidate_authority"))?;
        let next = self
            .expected_authority
            .next_sequence()
            .map_err(|_| invalid("expected_authority"))?;
        if self.candidate_authority.sequence != next
            || self.candidate_authority.predecessor != Some(self.expected_authority.fingerprint())
            || self.candidate_authority.object_id == self.expected_authority.object_id
            || self.candidate_authority.manifest_sha256 == self.expected_authority.manifest_sha256
        {
            return Err(invalid("candidate_authority"));
        }
        self.expected_generation
            .validate()
            .map_err(|_| invalid("expected_generation"))?;
        self.candidate_generation
            .validate()
            .map_err(|_| invalid("candidate_generation"))?;
        if self.candidate_generation.sequence <= self.expected_generation.sequence {
            return Err(invalid("candidate_generation"));
        }
        validate_source(&self.previous_source, "previous_source")?;
        validate_source(&self.source, "source")?;
        if self.source.low < self.previous_source.low
            || self.source.high < self.previous_source.high
        {
            return Err(invalid("source"));
        }
        if self.documents == 0 {
            return Err(invalid("documents"));
        }
        if self.original_request_sha256 == [0; 32] {
            return Err(invalid("original_request_sha256"));
        }
        if self.query_policy_sha256 == [0; 32] {
            return Err(invalid("query_policy_sha256"));
        }
        if self.source_kind == ImmutableReindexSource::Retained
            && (self.source != self.previous_source || self.documents != self.previous_documents)
        {
            return Err(invalid("retained_source"));
        }
        let mut identity_changed = false;
        for tier in [&self.fast, &self.quality].into_iter().flatten() {
            tier.previous
                .validate()
                .map_err(|_| invalid("previous_identity"))?;
            tier.replacement
                .validate()
                .map_err(|_| invalid("replacement_identity"))?;
            if tier.previous.identity.space.kind != EmbeddingSpaceKindV1::Semantic
                || tier.replacement.identity.space.kind != EmbeddingSpaceKindV1::Semantic
            {
                return Err(invalid("semantic_identity"));
            }
            if tier.provenance != VerifiedSemanticProvenance::Local {
                return Err(invalid("producer_provenance"));
            }
            if self.source_kind == ImmutableReindexSource::Retained
                && tier.previous.identity.input != tier.replacement.identity.input
            {
                return Err(invalid("retained_input_contract"));
            }
            identity_changed |= tier.previous.identity.space != tier.replacement.identity.space
                || tier.previous.identity.producer != tier.replacement.identity.producer
                || tier.previous.identity.input != tier.replacement.identity.input;
        }
        if !identity_changed {
            return Err(invalid("identity_mismatch"));
        }
        Ok(())
    }
}

fn validate_source(source: &CommitRange, field: &'static str) -> Result<(), RecoveryContractError> {
    source
        .high
        .checked_sub(source.low)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| invalid(field))?;
    Ok(())
}

/// Explicit process-local consent for one exact immutable reindex attempt.
///
/// The caller creates this only at an operator-authorized execution boundary,
/// after independently collecting the entire binding. It is not inferred from
/// readiness, a general "yes" policy, or a serialized recovery plan. Its exact
/// authority expectation remains single-selection CAS scope, not permission to
/// discover or rebase onto another current head.
///
/// ```compile_fail
/// use frankensearch_core::recovery_plan::ImmutableReindexAuthorization;
/// let _: ImmutableReindexAuthorization = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct ImmutableReindexAuthorization {
    binding: ImmutableReindexBinding,
}

impl ImmutableReindexAuthorization {
    /// Record explicit authorization for this already resolved exact scope.
    ///
    /// # Errors
    ///
    /// Returns a bounded binding error for an invalid or unsupported attempt.
    pub fn authorize(binding: &ImmutableReindexBinding) -> Result<Self, RecoveryContractError> {
        binding.validate()?;
        Ok(Self {
            binding: binding.clone(),
        })
    }
}

/// A canonical recovery plan whose immutable reindex scope is explicitly bound.
///
/// The executor accepts this type, validates its scope against independently
/// frozen execution facts, and performs the retained-authority CAS. Serialization
/// is reporting only: it cannot restore this capability or unblock the generic
/// planner. The current readiness and response contract remain unchanged until
/// the runtime proves the new generation and replays the original request.
///
/// ```compile_fail
/// use frankensearch_core::recovery_plan::BoundImmutableReindexPlan;
/// let _: BoundImmutableReindexPlan = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct BoundImmutableReindexPlan {
    plan: RecoveryPlan,
    binding: ImmutableReindexBinding,
}

impl BoundImmutableReindexPlan {
    /// Truthful plan for reporting; no generic executable argv is invented.
    #[must_use]
    pub const fn plan(&self) -> &RecoveryPlan {
        &self.plan
    }

    /// Exact authorized scope, exposed without mutable access.
    #[must_use]
    pub const fn binding(&self) -> &ImmutableReindexBinding {
        &self.binding
    }

    /// Recheck every bound fact at the executor's publication boundary.
    ///
    /// `actual` must be independently reconstructed from the retained root,
    /// selected old images, complete frozen candidate, local providers and
    /// original request, never copied from a wire plan or from this accessor.
    ///
    /// # Errors
    ///
    /// Rejects malformed execution facts or any change to the authorized scope.
    pub fn validate_binding(
        &self,
        actual: &ImmutableReindexBinding,
    ) -> Result<(), RecoveryContractError> {
        actual.validate()?;
        if self.binding != *actual {
            return Err(RecoveryContractError::ImmutableReindexBindingChanged);
        }
        Ok(())
    }
}

impl RecoveryPlan {
    /// Bind the narrow offline immutable-authority identity-migration executor.
    ///
    /// Generic planning stays capability-blocked. This consumes an actual
    /// canonical `IdentityMismatch` plan, independently established immutable
    /// execution facts, and explicit exact consent. It does not grant model
    /// acquisition, daemon repair, ANN-only repair, partial-tier mutation or
    /// recovery from an untrusted current generation.
    ///
    /// # Errors
    ///
    /// Refuses unsupported state/policy/topology, a noncanonical plan, invalid
    /// binding facts, or an authorization issued for any different scope.
    pub fn bind_immutable_reindex(
        mut self,
        binding: ImmutableReindexBinding,
        authorization: ImmutableReindexAuthorization,
    ) -> Result<BoundImmutableReindexPlan, RecoveryContractError> {
        let unsupported = |field| RecoveryContractError::UnsupportedImmutableReindexPlan { field };
        if self.state != SemanticReadiness::IdentityMismatch {
            return Err(unsupported("state"));
        }
        if self.policy.network != NetworkPolicy::Offline
            || self.policy.acquisition_authorization.is_some()
        {
            return Err(unsupported("policy"));
        }
        binding.validate()?;
        // ubs:ignore[rust.security.constant-time-compare] -- These are public retrieval-topology enum values, not secret credentials.
        if binding.topology()? != self.requested_topology {
            return Err(unsupported("requested_topology"));
        }
        let ImmutableReindexAuthorization {
            binding: authorized_scope,
        } = authorization;
        // ubs:ignore[rust.security.constant-time-compare] -- Authorization clones this caller-visible generation/source/model scope; it contains no secret or bearer token.
        if authorized_scope != binding {
            return Err(RecoveryContractError::MismatchedImmutableReindexAuthorization);
        }
        let canonical = plan(TrustedRecoveryContext::new(
            &self.state,
            RecoveryRequest {
                mode: self.mode,
                requested_topology: self.requested_topology,
            },
            &self.policy,
            None,
            // This path has no timed acquisition authorization and never
            // reads or synthesizes ambient clock authority.
            0,
        ))?;
        if self != canonical {
            return Err(unsupported("canonical_plan"));
        }
        self.action = Some(simple_action(
            "recovery.action.reindex_full",
            "Build and verify one complete immutable candidate with the explicitly authorized \
             local producers. Publish only against the retained authority, replay the original \
             request once, and restore the predecessor through a higher authority sequence if \
             replay fails. Existing source and artifact images remain intact; cancellation or \
             uncertain publication must retain their actual runtime outcome.",
            &[],
            false,
            true,
            true,
            false,
            "recovery.post.index_rebuilt",
        ));
        self.retryability = Retryability::AfterAction;
        Ok(BoundImmutableReindexPlan {
            plan: self,
            binding,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ZeroSignalReason;
    use crate::generation::{
        EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    use crate::recovery_plan::{
        InteractionPolicy, RecoveryPolicy, RequestMode, SemanticProvenance, UntrustedRecoveryPlan,
    };
    use crate::traits::ModelTier;

    fn tier(name: &str) -> ImmutableReindexTierBinding {
        // Pure contract fixture, not a claim that a real local model was loaded.
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, 2);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 42,
        }];
        identity.producer.provenance_manifest_fingerprint = "c".repeat(64);
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.endianness = "little-endian".to_owned();
        let previous = identity.freeze().unwrap();
        identity.producer.implementation_revision = "replacement-producer".to_owned();
        ImmutableReindexTierBinding {
            previous,
            replacement: identity.freeze().unwrap(),
            provenance: VerifiedSemanticProvenance::Local,
        }
    }

    fn binding(topology: RetrievalTopology) -> ImmutableReindexBinding {
        let expected_authority = AuthorityRefV1::new(1, [2; 16], 128, [3; 32], None).unwrap();
        let candidate_authority = AuthorityRefV1::new(
            2,
            [4; 16],
            128,
            [5; 32],
            Some(expected_authority.fingerprint()),
        )
        .unwrap();
        ImmutableReindexBinding {
            root_id: [1; 16],
            expected_authority,
            candidate_authority,
            expected_generation: ArtifactGenerationIdentityV1::new(1, [6; 16]).unwrap(),
            candidate_generation: ArtifactGenerationIdentityV1::new(2, [7; 16]).unwrap(),
            previous_source: CommitRange { low: 1, high: 5 },
            source: CommitRange { low: 1, high: 6 },
            previous_documents: 2,
            documents: 3,
            source_kind: ImmutableReindexSource::Prepared,
            original_request_sha256: [8; 32],
            query_policy_sha256: [9; 32],
            fast: matches!(
                topology,
                RetrievalTopology::FastOnly | RetrievalTopology::FullProgressive
            )
            .then(|| tier("fast-contract-fixture")),
            quality: matches!(
                topology,
                RetrievalTopology::QualityOnly | RetrievalTopology::FullProgressive
            )
            .then(|| tier("quality-contract-fixture")),
        }
    }

    fn policy() -> RecoveryPolicy {
        RecoveryPolicy {
            interaction: InteractionPolicy::NonInteractive,
            network: NetworkPolicy::Offline,
            acquisition_authorization: None,
        }
    }

    fn blocked(topology: RetrievalTopology) -> RecoveryPlan {
        plan(TrustedRecoveryContext::new(
            &SemanticReadiness::IdentityMismatch,
            RecoveryRequest {
                mode: RequestMode::ExplicitSemantic,
                requested_topology: topology,
            },
            &policy(),
            None,
            1,
        ))
        .unwrap()
    }

    fn bind(scope: ImmutableReindexBinding) -> BoundImmutableReindexPlan {
        let authorization = ImmutableReindexAuthorization::authorize(&scope).unwrap();
        blocked(scope.topology().unwrap())
            .bind_immutable_reindex(scope, authorization)
            .unwrap()
    }

    #[test]
    fn all_complete_topologies_bind_without_claiming_current_readiness_or_generic_support() {
        for topology in [
            RetrievalTopology::FastOnly,
            RetrievalTopology::QualityOnly,
            RetrievalTopology::FullProgressive,
        ] {
            for mode in [RequestMode::ExplicitSemantic, RequestMode::Hybrid] {
                let scope = binding(topology);
                let ordinary = plan(TrustedRecoveryContext::new(
                    &SemanticReadiness::IdentityMismatch,
                    RecoveryRequest {
                        mode,
                        requested_topology: topology,
                    },
                    &policy(),
                    None,
                    1,
                ))
                .unwrap();
                assert_eq!(ordinary.retryability(), Retryability::BlockedByCapability);
                let old_response = ordinary.response_contract().cloned();
                let authorization = ImmutableReindexAuthorization::authorize(&scope).unwrap();
                let bound = ordinary
                    .clone()
                    .bind_immutable_reindex(scope.clone(), authorization)
                    .unwrap();
                bound.validate_binding(&scope).unwrap();
                assert_eq!(bound.plan().state(), &SemanticReadiness::IdentityMismatch);
                assert_eq!(bound.plan().provenance(), SemanticProvenance::Unavailable);
                assert!(!bound.plan().semantic_available());
                assert_eq!(bound.plan().response_contract(), old_response.as_ref());
                assert_eq!(bound.plan().retryability(), Retryability::AfterAction);
                let action = bound.plan().action().unwrap();
                assert_eq!(action.code(), "recovery.action.reindex_full");
                assert_eq!(
                    action.expected_postcondition(),
                    "recovery.post.index_rebuilt"
                );
                assert!(action.preserves_old_data());
                assert!(!action.potentially_destructive());
                assert!(action.consent_required());
                assert!(!action.network_required());
                assert!(action.prerequisites().is_empty());
                assert!(action.argv().is_empty());
                assert!(action.required_authorization().is_none());
                // The binding is a separate capability, not a global planner switch.
                assert_eq!(ordinary.retryability(), Retryability::BlockedByCapability);
                assert_eq!(
                    blocked(topology).retryability(),
                    Retryability::BlockedByCapability
                );
            }
        }
    }

    #[test]
    fn exact_authorization_and_execution_binding_reject_every_scope_substitution() {
        type Mutation = fn(&mut ImmutableReindexBinding);
        let mutations: &[(&str, Mutation)] = &[
            ("root", |scope| scope.root_id = [11; 16]),
            ("old authority", |scope| {
                scope.expected_authority.manifest_sha256 = [12; 32];
                scope.candidate_authority.predecessor =
                    Some(scope.expected_authority.fingerprint());
            }),
            ("candidate authority", |scope| {
                scope.candidate_authority.manifest_sha256 = [13; 32];
            }),
            ("old generation", |scope| {
                scope.expected_generation.nonce = [14; 16];
            }),
            ("candidate generation", |scope| {
                scope.candidate_generation.nonce = [15; 16];
            }),
            ("old source", |scope| scope.previous_source.high = 4),
            ("new source", |scope| scope.source.high = 7),
            ("old census", |scope| scope.previous_documents = 1),
            ("new census", |scope| scope.documents = 4),
            ("source origin", |scope| {
                scope.source_kind = ImmutableReindexSource::Retained;
                scope.source = scope.previous_source.clone();
                scope.documents = scope.previous_documents;
            }),
            ("original request", |scope| {
                scope.original_request_sha256 = [16; 32];
            }),
            ("query policy", |scope| scope.query_policy_sha256 = [17; 32]),
            ("old fast identity", |scope| {
                let tier = scope.fast.as_mut().unwrap();
                let mut identity = tier.previous.identity.clone();
                identity.storage.quantization = QuantizationFormat::F16;
                tier.previous = identity.freeze().unwrap();
            }),
            ("new fast identity", |scope| {
                let tier = scope.fast.as_mut().unwrap();
                let mut identity = tier.replacement.identity.clone();
                identity.producer.protocol_revision.push_str("-different");
                tier.replacement = identity.freeze().unwrap();
            }),
            ("old quality identity", |scope| {
                let tier = scope.quality.as_mut().unwrap();
                let mut identity = tier.previous.identity.clone();
                identity.storage.quantization = QuantizationFormat::F16;
                tier.previous = identity.freeze().unwrap();
            }),
            ("new quality identity", |scope| {
                let tier = scope.quality.as_mut().unwrap();
                let mut identity = tier.replacement.identity.clone();
                identity.producer.protocol_revision.push_str("-different");
                tier.replacement = identity.freeze().unwrap();
            }),
        ];
        let original = binding(RetrievalTopology::FullProgressive);
        let bound = bind(original.clone());
        for (name, mutate) in mutations {
            let mut different = original.clone();
            mutate(&mut different);
            different.validate().unwrap();
            let authorization = ImmutableReindexAuthorization::authorize(&original).unwrap();
            assert_eq!(
                blocked(RetrievalTopology::FullProgressive)
                    .bind_immutable_reindex(different.clone(), authorization)
                    .unwrap_err(),
                RecoveryContractError::MismatchedImmutableReindexAuthorization,
                "{name}"
            );
            assert_eq!(
                bound.validate_binding(&different).unwrap_err(),
                RecoveryContractError::ImmutableReindexBindingChanged,
                "{name}"
            );
        }
    }

    #[test]
    fn malformed_authority_lineage_census_or_request_cannot_receive_consent() {
        type Mutation = fn(&mut ImmutableReindexBinding);
        let mutations: &[(&str, Mutation)] = &[
            ("root", |scope| scope.root_id = [0; 16]),
            ("old authority schema", |scope| {
                scope.expected_authority.schema_version += 1;
            }),
            ("candidate authority schema", |scope| {
                scope.candidate_authority.schema_version += 1;
            }),
            ("authority sequence skip", |scope| {
                scope.candidate_authority.sequence += 1;
            }),
            ("authority predecessor", |scope| {
                scope.candidate_authority.predecessor = Some([11; 32]);
            }),
            ("reused manifest object", |scope| {
                scope.candidate_authority.object_id = scope.expected_authority.object_id;
            }),
            ("reused manifest bytes", |scope| {
                scope.candidate_authority.manifest_sha256 =
                    scope.expected_authority.manifest_sha256;
            }),
            ("old generation", |scope| {
                scope.expected_generation.nonce = [0; 16];
            }),
            ("candidate generation", |scope| {
                scope.candidate_generation.nonce = [0; 16];
            }),
            ("artifact sequence rollback", |scope| {
                scope.candidate_generation.sequence = scope.expected_generation.sequence;
            }),
            ("old source", |scope| scope.previous_source.low = 6),
            ("backward low", |scope| scope.source.low = 0),
            ("backward high", |scope| scope.source.high = 4),
            ("source span overflow", |scope| {
                scope.source = CommitRange {
                    low: 0,
                    high: u64::MAX,
                }
            }),
            ("empty target", |scope| scope.documents = 0),
            ("missing request", |scope| {
                scope.original_request_sha256 = [0; 32];
            }),
            ("missing policy", |scope| {
                scope.query_policy_sha256 = [0; 32];
            }),
            ("no topology", |scope| {
                scope.fast = None;
                scope.quality = None;
            }),
        ];
        for (name, mutate) in mutations {
            let mut scope = binding(RetrievalTopology::FullProgressive);
            mutate(&mut scope);
            assert!(
                ImmutableReindexAuthorization::authorize(&scope).is_err(),
                "{name}"
            );
        }
        let mut exhausted = binding(RetrievalTopology::FastOnly);
        exhausted.expected_authority.sequence = u64::MAX;
        exhausted.expected_authority.predecessor = Some([12; 32]);
        assert!(ImmutableReindexAuthorization::authorize(&exhausted).is_err());
        let mut fresh = binding(RetrievalTopology::FastOnly);
        fresh.previous_documents = 0;
        assert!(ImmutableReindexAuthorization::authorize(&fresh).is_ok());
    }

    #[test]
    fn both_tiers_require_complete_semantic_identity_and_verified_local_provenance() {
        for quality in [false, true] {
            for provenance in [
                VerifiedSemanticProvenance::Remote,
                VerifiedSemanticProvenance::Daemon,
            ] {
                let mut scope = binding(RetrievalTopology::FullProgressive);
                let selected = if quality {
                    &mut scope.quality
                } else {
                    &mut scope.fast
                };
                selected.as_mut().unwrap().provenance = provenance;
                assert_eq!(
                    scope.validate().unwrap_err(),
                    invalid("producer_provenance")
                );
            }
            for previous in [false, true] {
                let mut scope = binding(RetrievalTopology::FullProgressive);
                let selected = if quality {
                    &mut scope.quality
                } else {
                    &mut scope.fast
                };
                let tier = selected.as_mut().unwrap();
                let identity = if previous {
                    &mut tier.previous
                } else {
                    &mut tier.replacement
                };
                *identity = EmbeddingIdentityBundleV1::explicit_test_model("hash-control", 2)
                    .freeze()
                    .unwrap();
                assert_eq!(scope.validate().unwrap_err(), invalid("semantic_identity"));
                let mut scope = binding(RetrievalTopology::FullProgressive);
                let selected = if quality {
                    &mut scope.quality
                } else {
                    &mut scope.fast
                };
                let tier = selected.as_mut().unwrap();
                let identity = if previous {
                    &mut tier.previous
                } else {
                    &mut tier.replacement
                };
                identity.canonical_bytes[0] ^= 1;
                assert!(scope.validate().is_err());
            }
        }
        let mut unchanged = binding(RetrievalTopology::FastOnly);
        let tier = unchanged.fast.as_mut().unwrap();
        tier.replacement = tier.previous.clone();
        assert_eq!(
            unchanged.validate().unwrap_err(),
            invalid("identity_mismatch")
        );
        let tier = unchanged.fast.as_mut().unwrap();
        let mut storage_only = tier.previous.identity.clone();
        storage_only.storage.quantization = QuantizationFormat::F16;
        tier.replacement = storage_only.freeze().unwrap();
        assert_eq!(
            unchanged.validate().unwrap_err(),
            invalid("identity_mismatch")
        );
        let mut one_changed = binding(RetrievalTopology::FullProgressive);
        let quality = one_changed.quality.as_mut().unwrap();
        quality.replacement = quality.previous.clone();
        assert!(one_changed.validate().is_ok());
    }

    #[test]
    fn retained_source_cannot_be_recounted_recheckpointed_or_relabeled() {
        for quality in [false, true] {
            let mut scope = binding(RetrievalTopology::FullProgressive);
            scope.source_kind = ImmutableReindexSource::Retained;
            scope.source = scope.previous_source.clone();
            scope.documents = scope.previous_documents;
            scope.validate().unwrap();
            let mut changed = scope.clone();
            changed.source.high += 1;
            assert_eq!(changed.validate().unwrap_err(), invalid("retained_source"));
            changed = scope.clone();
            changed.documents += 1;
            assert_eq!(changed.validate().unwrap_err(), invalid("retained_source"));
            let selected = if quality {
                &mut scope.quality
            } else {
                &mut scope.fast
            };
            let tier = selected.as_mut().unwrap();
            let mut identity = tier.replacement.identity.clone();
            identity.input.canonicalization.push_str("-new-preparation");
            identity.space.input_contract_fingerprint = identity.input.fingerprint();
            identity.producer.space_fingerprint = identity.space.fingerprint();
            tier.replacement = identity.freeze().unwrap();
            assert_eq!(
                scope.validate().unwrap_err(),
                invalid("retained_input_contract")
            );
            scope.source_kind = ImmutableReindexSource::Prepared;
            assert!(scope.validate().is_ok());
        }
    }

    #[test]
    fn canonical_state_policy_and_complete_topology_are_required() {
        let scope = binding(RetrievalTopology::FastOnly);
        for state in [
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Local,
            },
            SemanticReadiness::ModelMissing {
                tier: ModelTier::Fast,
            },
            SemanticReadiness::ModelUnloadable {
                tier: ModelTier::Fast,
            },
            SemanticReadiness::IndexAbsent,
            SemanticReadiness::DaemonMismatch,
            SemanticReadiness::ManifestUnsafe,
            SemanticReadiness::AnnStale,
            SemanticReadiness::GenerationIncomplete,
            SemanticReadiness::IndexEmpty(ZeroSignalReason::NewlyCreatedEmpty),
            SemanticReadiness::PartialQualityCoverage {
                provenance: VerifiedSemanticProvenance::Local,
                coverage_ppm: 500_000,
            },
            SemanticReadiness::RemoteUnverified,
        ] {
            let ordinary = plan(TrustedRecoveryContext::new(
                &state,
                RecoveryRequest {
                    mode: RequestMode::ExplicitSemantic,
                    requested_topology: RetrievalTopology::FastOnly,
                },
                &policy(),
                None,
                1,
            ))
            .unwrap();
            let authorization = ImmutableReindexAuthorization::authorize(&scope).unwrap();
            assert!(matches!(
                ordinary.bind_immutable_reindex(scope.clone(), authorization),
                Err(RecoveryContractError::UnsupportedImmutableReindexPlan { field: "state" })
            ));
        }
        let mut online = policy();
        online.network = NetworkPolicy::Allowed;
        let ordinary = plan(TrustedRecoveryContext::new(
            &SemanticReadiness::IdentityMismatch,
            RecoveryRequest {
                mode: RequestMode::ExplicitSemantic,
                requested_topology: RetrievalTopology::FastOnly,
            },
            &online,
            None,
            1,
        ))
        .unwrap();
        assert!(
            ordinary
                .bind_immutable_reindex(
                    scope.clone(),
                    ImmutableReindexAuthorization::authorize(&scope).unwrap()
                )
                .is_err()
        );
        let full = binding(RetrievalTopology::FullProgressive);
        assert!(
            blocked(RetrievalTopology::FastOnly)
                .bind_immutable_reindex(
                    full.clone(),
                    ImmutableReindexAuthorization::authorize(&full).unwrap()
                )
                .is_err()
        );
        for mutate in [
            (|plan: &mut RecoveryPlan| plan.semantic_available = true) as fn(&mut RecoveryPlan),
            |plan| plan.retryability = Retryability::AfterAction,
            |plan| {
                plan.action
                    .as_mut()
                    .unwrap()
                    .argv
                    .push("unbound-command".to_owned());
            },
            |plan| plan.action.as_mut().unwrap().preserves_old_data = true,
            |plan| plan.action.as_mut().unwrap().prerequisites.clear(),
        ] {
            let mut forged = blocked(RetrievalTopology::FastOnly);
            mutate(&mut forged);
            let authorization = ImmutableReindexAuthorization::authorize(&scope).unwrap();
            assert!(matches!(
                forged.bind_immutable_reindex(scope.clone(), authorization),
                Err(RecoveryContractError::UnsupportedImmutableReindexPlan {
                    field: "canonical_plan"
                })
            ));
        }
    }

    #[test]
    fn reported_bound_plan_cannot_be_promoted_through_untrusted_generic_validation() {
        let bound = bind(binding(RetrievalTopology::FullProgressive));
        let reported = serde_json::to_value(bound.plan()).unwrap();
        let untrusted: UntrustedRecoveryPlan = serde_json::from_value(reported).unwrap();
        assert!(matches!(
            untrusted.validate_against(TrustedRecoveryContext::new(
                &SemanticReadiness::IdentityMismatch,
                RecoveryRequest {
                    mode: RequestMode::ExplicitSemantic,
                    requested_topology: RetrievalTopology::FullProgressive
                },
                &policy(),
                None,
                1,
            )),
            Err(RecoveryContractError::InconsistentRecoveryPlan { .. })
        ));
        let report = serde_json::to_value(&bound).unwrap();
        assert_eq!(
            report["binding"]["candidate_authority"],
            serde_json::json!(
                bound
                    .binding()
                    .candidate_authority
                    .canonical_bytes()
                    .as_slice()
            )
        );
        assert!(report.get("authorization").is_none());
        assert!(
            report["plan"]["action"]["argv"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}
