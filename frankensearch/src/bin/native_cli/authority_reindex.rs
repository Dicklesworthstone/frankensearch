//! One explicitly authorized offline identity migration, preview, and replay.
//!
//! The generic recovery planner cannot publish generations. This executor binds
//! its narrower capability to actual admitted old images, local providers and
//! one complete candidate, then uses the original authority pair exactly once.

use frankensearch_core::RetrievalTopology;
use frankensearch_core::generation::CanonicalDocsetV1;
use frankensearch_core::recovery_plan::{
    BoundImmutableReindexPlan, ImmutableReindexAuthorization, ImmutableReindexBinding,
    ImmutableReindexPartialQuality, ImmutableReindexSource, ImmutableReindexTierBinding,
    InteractionPolicy, NetworkPolicy, RecoveryPolicy, RecoveryRequest, RequestMode,
    SemanticReadiness, TrustedRecoveryContext, VerifiedSemanticProvenance, plan,
};
use sha2::Digest as _;

use super::{
    Arc, Command, Cx, EmbedderTierTag, Error, GenerationPublisherV1, MAX_RECEIPT_BYTES, Models,
    NativeBuildRetrieval, NativeBuiltExactGeneration, NativeGenerationSnapshot,
    NativeGenerationTopology, NativeSearchPhase, Options, PARTIAL_QUALITY_RECEIPT_SCHEMA, PROFILE,
    PROGRESSIVE_RECEIPT_SCHEMA, PreparedPublication, PublicationFailure, PublicationOptions,
    PublishedGeneration, QualifiedGenerationRoot, QueryMode, QueryPlan, RECEIPT_SCHEMA, Receipt,
    ReceiptQueryMode, Result, SCHEMA, SavedAttempt, ScoredResult, SearchError, Sha256, Write,
    WriterIdentityV1, activation_manifest_path_v1, annotate_execution, annotate_scope,
    annotate_selection, annotate_snapshot_retrieval, bad, checkpoint, emit, encode, new_generation,
    new_path, prepare_partial_quality_rebuild, prepare_publication, publication_error,
    publish_and_admit_with_reconciliation, query, query_admitted, query_controls, rebuild_identity,
    same_security_object, save_receipt, separate_destinations, stage_prepared_and_publish,
    validate_query,
};

type Phases = Vec<(&'static str, Vec<ScoredResult>)>;

struct Inputs<'a> {
    options: &'a Options,
    root: &'a QualifiedGenerationRoot,
    selected: &'a Receipt,
    previous: &'a NativeGenerationSnapshot,
    built: &'a NativeBuiltExactGeneration,
    loaded: &'a Models,
    text: &'a str,
    mode: QueryMode,
    query_policy: &'a query::Policy,
}

/// Resolve the exact replay mode before loading any replacement provider.
pub(super) fn query_mode(selected: &Receipt, options: &Options) -> Result<QueryMode> {
    if selected.quality_documents.is_some() {
        if !options.retained_source {
            return Err(super::serve_refused(
                "partial_quality_rebuild_query_requires_retained_source",
            ));
        }
        if options.mode != Some(QueryMode::Partial) {
            return Err(super::serve_refused(
                "partial_quality_rebuild_query_requires_partial_mode",
            ));
        }
        selected.query_mode(options.mode)
    } else if options.mode.is_some() {
        Err(super::serve_refused("reindex_query_mode_conflict"))
    } else {
        selected.query_mode(None)
    }
}

/// Called only after the normal rebuild path has prepared the complete corpus.
/// Direct callers must still present the same explicit command authorization.
#[allow(clippy::too_many_arguments)]
pub(super) async fn execute(
    cx: &Cx,
    options: &Options,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    previous: &NativeGenerationSnapshot,
    built: &NativeBuiltExactGeneration,
    loaded: &Models,
    output: &mut impl Write,
) -> Result<()> {
    let policy = query_controls::load_policy(cx, options)?;
    Box::pin(execute_with_policy(
        cx, options, root, selected, previous, built, loaded, &policy, output,
    ))
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_with_policy(
    cx: &Cx,
    options: &Options,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    previous: &NativeGenerationSnapshot,
    built: &NativeBuiltExactGeneration,
    loaded: &Models,
    policy: &query::Policy,
    output: &mut impl Write,
) -> Result<()> {
    checkpoint(cx)?;
    if options.command != Command::Rebuild
        || !options.migrate_model
        || options.lexical_only
        || options.from_receipt.is_none()
        || options.rollback_receipt.is_none()
        || options.limit == 0
        || options.limit > 1_000
    {
        return Err(bad(
            "reindex query requires explicit migration authorization",
        ));
    }
    let text = options
        .query
        .as_deref()
        .ok_or_else(|| bad("reindex query is required"))?;
    validate_query(text)?;
    if let Some(filter) = &options.filter {
        filter.validate()?;
    }
    let mode = query_mode(selected, options)?;
    query_controls::validate_startup(options, mode)?;
    if policy.maximum_ms() != options.timeout_ms
        || policy.full_query != options.full_query
        || policy.rerank.is_some() != options.reranker_dir.is_some()
        || policy
            .rerank
            .as_ref()
            .is_some_and(|rerank| rerank.window != options.rerank_window)
    {
        return Err(bad(
            "reindex query policy differs from its authorized options",
        ));
    }
    let receipt_path = new_path(&options.receipt)?;
    let rollback_path = new_path(
        options
            .rollback_receipt
            .as_deref()
            .ok_or_else(|| bad("rollback receipt is required"))?,
    )?;
    separate_destinations(&[&selected.root, &receipt_path, &rollback_path])?;
    let inputs = Inputs {
        options,
        root,
        selected,
        previous,
        built,
        loaded,
        text,
        mode,
        query_policy: policy,
    };
    let prepared = if mode == QueryMode::Partial {
        prepare_partial_quality_rebuild(cx, built, previous, options.max_image_bytes)?
    } else {
        prepare_publication(built, Some(previous), options.max_image_bytes)?
    };
    let binding = inputs
        .binding(cx, &prepared)
        .map_err(|_| bad("reindex binding admission refused"))?;
    let recovery_policy = RecoveryPolicy {
        interaction: InteractionPolicy::NonInteractive,
        network: NetworkPolicy::Offline,
        acquisition_authorization: None,
    };
    let recovery = plan(TrustedRecoveryContext::new(
        &SemanticReadiness::IdentityMismatch,
        RecoveryRequest {
            mode: RequestMode::Hybrid,
            requested_topology: binding.topology()?,
        },
        &recovery_policy,
        None,
        0,
    ))?
    .bind_immutable_reindex(
        binding.clone(),
        ImmutableReindexAuthorization::authorize(&binding)?,
    )?;

    // This proves a real query over exactly the frozen candidate, including
    // hydration and both producers for full topology, before staging AUTHORITY.
    Box::pin(inputs.preview(cx, &prepared))
        .await
        .map_err(|error| {
            inputs.failure(
                &recovery,
                "not_attempted",
                "previous_retained",
                if cancelled(cx, error.as_ref()) {
                    "preview_cancelled"
                } else if timed_out(error.as_ref()) {
                    "preview_timed_out"
                } else {
                    "preview_refused"
                },
            )
        })?;
    checkpoint(cx)?;
    inputs
        .binding(cx, &prepared)
        .and_then(|actual| recovery.validate_binding(&actual).map_err(Into::into))
        .map_err(|_| {
            inputs.failure(
                &recovery,
                "not_attempted",
                "previous_retained",
                "binding_refused",
            )
        })?;
    root.revalidate()?;
    let published = stage_prepared_and_publish(
        cx,
        root,
        &selected.root,
        &receipt_path,
        built,
        prepared,
        PublicationOptions {
            max_image_bytes: options.max_image_bytes,
            scope: options
                .filter
                .as_ref()
                .map(|filter| (filter, options.max_source_bytes)),
            reconcile_unknown: true,
        },
    )
    .map_err(|mut error| {
        if let Some(failure) = error.downcast_mut::<PublicationFailure>() {
            failure.0["recovery"] = serde_json::json!(recovery);
            failure.0["replay"] = serde_json::json!("not_attempted");
            let status = if failure.0["serving"] == "superseded" {
                "not_attempted_superseded"
            } else if failure.0["publication"] == "committed" {
                "deferred"
            } else if failure.0["publication"] == "unknown" {
                "blocked_on_reconciliation"
            } else {
                "not_required"
            };
            failure.0["rollback"] = serde_json::json!({
                "status": status, "target_receipt": options.from_receipt,
                "requested_receipt": rollback_path,
            });
        }
        error
    })?;

    // Buffer the requested answer until every required replay phase succeeds.
    // Both two-tier modes consume refinement, even for a nonstreaming response.
    let mut phases = Phases::new();
    let replay = async {
        // Publication work does not spend the original query's time budget.
        // Replay receives one fresh budget of exactly the authorized duration;
        // every phase shares it and an expiry follows the normal rollback path.
        let deadline = policy.start(cx, None)?;
        Box::pin(query_admitted(
            cx,
            &published.snapshot,
            published.scope.as_ref(),
            loaded,
            QueryPlan {
                text,
                limit: options.limit,
                mode: inputs.mode,
                stream: matches!(inputs.mode, QueryMode::Full | QueryMode::Partial)
                    || policy.rerank.is_some(),
                deadline: deadline.as_ref(),
                policy: Some(policy),
                max_source_bytes: options.max_source_bytes,
            },
            |phase, results| {
                validate_results(phase, results)?;
                phases.push((phase, results.to_vec()));
                Ok(())
            },
        ))
        .await?;
        validate_phases(inputs.mode, policy.rerank.is_some(), &phases)?;
        checkpoint(cx)
    }
    .await;
    if replay.is_err() {
        let mut failure = inputs.failure(
            &recovery,
            "committed",
            "replay_failed",
            if replay
                .as_ref()
                .err()
                .is_some_and(|error| cancelled(cx, error.as_ref()))
            {
                "replay_cancelled"
            } else if replay
                .as_ref()
                .err()
                .is_some_and(|error| timed_out(error.as_ref()))
            {
                "replay_timed_out"
            } else {
                "replay_refused"
            },
        );
        if let Some(report) = failure.downcast_mut::<PublicationFailure>() {
            report.0["authority"] = serde_json::json!(published.receipt.authority);
            report.0["receipt"] = serde_json::json!(receipt_path);
            report.0["replay"] = serde_json::json!("failed");
            // A masked Cx still exposes cancellation to the native admission
            // checkpoints. Do not reset it or invent an unrelated cleanup Cx.
            report.0["rollback"] = if checkpoint(cx).is_err() {
                serde_json::json!({
                    "status": "deferred", "reason": "caller_cancelled",
                    "target_receipt": options.from_receipt,
                    "requested_receipt": rollback_path,
                })
            } else {
                match inputs.rollback(cx, &published, &rollback_path) {
                    Ok(receipt) => serde_json::json!({
                        "status": "committed", "serving": "activated",
                        "receipt": rollback_path, "selection": receipt,
                        "restores": selected.authority,
                    }),
                    Err(error) => error.downcast_ref::<PublicationFailure>().map_or_else(
                        || {
                            serde_json::json!({
                                "status": "deferred", "reason": "rollback_preparation_refused",
                                "target_receipt": options.from_receipt,
                                "requested_receipt": rollback_path,
                            })
                        },
                        |failed| {
                            let mut rollback = failed.0.clone();
                            rollback["status"] = rollback["publication"].clone();
                            rollback
                        },
                    ),
                }
            };
        }
        return Err(failure);
    }

    // The bound input plan still truthfully describes IdentityMismatch. Only
    // admitted publication plus replay establishes the actual coverage state.
    let readiness = match published.snapshot.topology() {
        NativeGenerationTopology::PartialQuality { coverage_ppm, .. } => {
            SemanticReadiness::PartialQualityCoverage {
                provenance: VerifiedSemanticProvenance::Local,
                coverage_ppm,
            }
        }
        _ => SemanticReadiness::Ready {
            provenance: VerifiedSemanticProvenance::Local,
        },
    };
    let terminal = plan(TrustedRecoveryContext::new(
        &readiness,
        RecoveryRequest {
            mode: RequestMode::Hybrid,
            requested_topology: match published.snapshot.topology() {
                NativeGenerationTopology::FastOnly => RetrievalTopology::FastOnly,
                NativeGenerationTopology::QualityOnly => RetrievalTopology::QualityOnly,
                NativeGenerationTopology::FullProgressive
                | NativeGenerationTopology::PartialQuality { .. } => {
                    RetrievalTopology::FullProgressive
                }
            },
        },
        &recovery_policy,
        None,
        0,
    ))
    .map_err(|_| {
        let mut error = publication_error(
            &published.receipt,
            "committed",
            "activated",
            "terminal_plan_refused",
            None,
        );
        if let Some(failure) = error.downcast_mut::<PublicationFailure>() {
            failure.0["receipt"] = serde_json::json!(receipt_path);
            failure.0["replay"] = serde_json::json!("complete");
        }
        error
    })?;
    for (position, (phase, results)) in phases.iter().enumerate() {
        if !options.stream && position + 1 != phases.len() {
            continue;
        }
        let mut response = serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "results", "operation": "reindex",
            "publication": "committed", "serving": "activated", "replay": "complete",
            "tier": inputs.mode.as_str(), "mode": inputs.mode.as_str(),
            "topology": published.receipt.topology_name(), "phase": phase,
            "generation": published.receipt.generation, "authority": published.receipt.authority,
            "profile": published.receipt.profile, "results": results,
            "receipt": receipt_path, "replaces": selected.authority,
            "recovery": recovery, "recovery_result": terminal,
            "rollback": {"status": "not_required"},
        });
        annotate_snapshot_retrieval(&mut response, &published.snapshot, Some(inputs.mode));
        annotate_selection(&mut response, &published.snapshot);
        annotate_scope(
            &mut response,
            published.scope.as_ref().map(|scope| scope.len()),
        );
        annotate_execution(&mut response, inputs.mode);
        query_controls::annotate(policy, inputs.mode, &mut response);
        response["timeout_ms"] = serde_json::json!(policy.maximum_ms());
        emit(output, &response).map_err(|_| {
            let mut error = publication_error(
                &published.receipt,
                "committed",
                "activated",
                "result_delivery_failed",
                None,
            );
            if let Some(failure) = error.downcast_mut::<PublicationFailure>() {
                failure.0["receipt"] = serde_json::json!(receipt_path);
                failure.0["replay"] = serde_json::json!("complete");
                failure.0["recovery"] = serde_json::json!(recovery);
            }
            error
        })?;
    }
    Ok(())
}

impl Inputs<'_> {
    fn binding(&self, cx: &Cx, prepared: &PreparedPublication) -> Result<ImmutableReindexBinding> {
        self.root.revalidate()?;
        self.selected.validate()?;
        let retained = self.previous.retained_snapshot();
        let anchors = self.root.anchor_witnesses();
        let old_anchors = retained.anchor_witnesses();
        let primary_tag = if self.selected.tier == EmbedderTierTag::Fast {
            "fast"
        } else {
            "quality"
        };
        if !same_security_object(retained.root_witness(), self.root.witness())
            || !same_security_object(anchors.lock(), old_anchors.lock())
            || !same_security_object(anchors.authority(), old_anchors.authority())
            || retained.profile() != PROFILE
            || retained.root_id() != self.selected.root_id
            || self.previous.authority() != self.selected.authority()?
            || retained.manifest().generation() != self.selected.generation
            || self.previous.topology() != self.selected.topology()
            || self.previous.metadata().total_documents != self.selected.documents
            || self.previous.metadata().embedders.get(primary_tag) != Some(&self.selected.identity)
            || (self.selected.quality_identity.is_some()
                && self.previous.metadata().embedders.get("quality")
                    != self.selected.quality_identity.as_ref())
            || prepared.root_id != retained.root_id()
            || prepared.expected != retained.authority_pair()
        {
            return Err(bad(
                "reindex retained selection differs from the admitted execution scope",
            ));
        }
        let primary = ImmutableReindexTierBinding {
            previous: self.selected.identity.clone(),
            replacement: rebuild_identity(
                &self.selected.identity,
                self.loaded.fast.as_ref(),
                true,
                self.options.retained_source,
            )?,
            // This entry point receives only the CLI's verified local loaders;
            // it neither accepts provenance from a request nor loads remotely.
            provenance: VerifiedSemanticProvenance::Local,
        };
        if &primary.replacement != self.built.binding().frozen_identity() {
            return Err(bad("reindex provider differs from the complete candidate"));
        }
        let quality = match (
            &self.selected.quality_identity,
            &self.loaded.quality,
            self.built.quality_binding(),
        ) {
            (Some(previous), Some(model), Some(built)) => {
                let replacement =
                    rebuild_identity(previous, model.as_ref(), true, self.options.retained_source)?;
                if &replacement != built.frozen_identity() {
                    return Err(bad(
                        "reindex quality provider differs from the complete candidate",
                    ));
                }
                Some(ImmutableReindexTierBinding {
                    previous: previous.clone(),
                    replacement,
                    provenance: VerifiedSemanticProvenance::Local,
                })
            }
            (None, None, None) => None,
            _ => {
                return Err(bad(
                    "reindex providers must preserve the complete stored topology",
                ));
            }
        };
        let policy = serde_json::to_vec(&serde_json::json!({
            "mode": self.mode.as_str(), "limit": self.options.limit,
            "filter": self.options.filter, "stream": self.options.stream,
            "timeout_ms": self.query_policy.maximum_ms(),
            "full_query": self.query_policy.full_query,
            "reranker": self.query_policy.rerank.as_ref().map(|rerank| {
                serde_json::json!({
                    "id": rerank.model.id(), "window": rerank.window,
                    "source": self.options.reranker_dir,
                })
            }),
            "max_source_bytes": self.options.max_source_bytes,
            "max_image_bytes": self.options.max_image_bytes,
            "primary_retrieval": retrieval_policy(self.built.retrieval_configuration()),
            "quality_retrieval": self.built.quality_retrieval_configuration().map(retrieval_policy),
        }))?;
        let mut request = Sha256::new();
        request.update(b"frankensearch.native.immutable-reindex.request.v2\0");
        request.update(u64::try_from(self.text.len())?.to_le_bytes());
        request.update(self.text.as_bytes());
        request.update(&policy);
        let mut query_policy = Sha256::new();
        query_policy.update(b"frankensearch.native.immutable-reindex.policy.v2\0");
        query_policy.update(&policy);
        let (fast, quality) = if self.previous.topology() == NativeGenerationTopology::QualityOnly {
            (None, Some(primary))
        } else {
            (Some(primary), quality)
        };
        let partial_quality = if self.mode == QueryMode::Partial {
            let covered = self.previous.partial_quality_document_ids(cx)?;
            let candidate = self.built.preview_candidate(cx, &prepared.candidate)?;
            if candidate.partial_quality_document_ids(cx)? != covered {
                return Err(super::serve_refused(
                    "partial_quality_rebuild_coverage_mismatch",
                ));
            }
            let documents = CanonicalDocsetV1::from_ordered_live_documents(covered)?;
            Some(ImmutableReindexPartialQuality {
                documents: u64::try_from(documents.len())?,
                document_set_sha256: documents.digest(),
            })
        } else {
            None
        };
        let binding = ImmutableReindexBinding {
            root_id: retained.root_id(),
            expected_authority: self.previous.authority(),
            candidate_authority: prepared.candidate.authority(),
            expected_generation: retained.manifest().generation(),
            candidate_generation: self.built.binding().generation(),
            previous_source: self.previous.metadata().commit_range.clone(),
            source: self.built.metadata().commit_range.clone(),
            previous_documents: self.previous.metadata().total_documents,
            documents: self.built.metadata().total_documents,
            source_kind: if self.options.retained_source {
                ImmutableReindexSource::Retained
            } else {
                ImmutableReindexSource::Prepared
            },
            original_request_sha256: request.finalize().into(),
            query_policy_sha256: query_policy.finalize().into(),
            fast,
            quality,
            partial_quality,
        };
        binding.validate()?;
        Ok(binding)
    }

    async fn preview(&self, cx: &Cx, prepared: &PreparedPublication) -> Result<()> {
        let deadline = self.query_policy.start(cx, None)?;
        Box::pin(query::within(
            cx,
            deadline.as_ref(),
            self.preview_candidate(cx, prepared),
        ))
        .await
    }

    async fn preview_candidate(&self, cx: &Cx, prepared: &PreparedPublication) -> Result<()> {
        let preview = Arc::new(self.built.preview_candidate(cx, &prepared.candidate)?);
        let scope = self
            .options
            .filter
            .as_ref()
            .map(|filter| {
                preview.scope(cx, self.options.max_source_bytes, |document| {
                    Ok(filter.matches(document))
                })
            })
            .transpose()?;
        let mut phases = Phases::new();
        if matches!(self.mode, QueryMode::Full | QueryMode::Partial) {
            let quality = self
                .loaded
                .quality
                .as_deref()
                .ok_or_else(|| bad("quality provider missing"))?;
            let query = match (
                self.mode == QueryMode::Partial,
                &scope,
                self.query_policy.query_pool(),
            ) {
                (true, Some(scope), Some(pool)) => scope
                    .search_progressive_with_partial_quality_and_pool(
                        cx,
                        self.loaded.fast.as_ref(),
                        quality,
                        self.text,
                        self.options.limit,
                        pool,
                    )?,
                (true, None, Some(pool)) => preview
                    .search_progressive_with_partial_quality_and_pool(
                        cx,
                        self.loaded.fast.as_ref(),
                        quality,
                        self.text,
                        self.options.limit,
                        pool,
                    )?,
                (true, Some(scope), None) => scope.search_progressive_with_partial_quality(
                    cx,
                    self.loaded.fast.as_ref(),
                    quality,
                    self.text,
                    self.options.limit,
                )?,
                (true, None, None) => preview.search_progressive_with_partial_quality(
                    cx,
                    self.loaded.fast.as_ref(),
                    quality,
                    self.text,
                    self.options.limit,
                )?,
                (false, Some(scope), Some(pool)) => scope
                    .search_progressive_with_quality_and_pool(
                        cx,
                        self.loaded.fast.as_ref(),
                        quality,
                        self.text,
                        self.options.limit,
                        pool,
                    )?,
                (false, None, Some(pool)) => preview.search_progressive_with_quality_and_pool(
                    cx,
                    self.loaded.fast.as_ref(),
                    quality,
                    self.text,
                    self.options.limit,
                    pool,
                )?,
                (false, Some(scope), None) => scope.search_progressive_with_quality(
                    cx,
                    self.loaded.fast.as_ref(),
                    quality,
                    self.text,
                    self.options.limit,
                )?,
                (false, None, None) => preview.search_progressive_with_quality(
                    cx,
                    self.loaded.fast.as_ref(),
                    quality,
                    self.text,
                    self.options.limit,
                )?,
            };
            let mut query =
                query_controls::configure(query, self.query_policy, self.options.max_source_bytes)?;
            while let Some(phase) = query.next_phase().await? {
                let (name, results) = match phase {
                    NativeSearchPhase::Initial { results, .. } => ("initial", results),
                    NativeSearchPhase::Refined { results, .. } => ("refined", results),
                    NativeSearchPhase::RefinementFailed { error, .. }
                    | NativeSearchPhase::RerankFailed { error, .. } => return Err(error.into()),
                    NativeSearchPhase::Reranked { results, .. } => ("reranked", results),
                };
                validate_results(name, &results)?;
                phases.push((name, results));
            }
        } else if self.query_policy.rerank.is_some() {
            let tier = if self.mode == QueryMode::Quality {
                EmbedderTierTag::Quality
            } else {
                EmbedderTierTag::Fast
            };
            let query = match &scope {
                Some(scope) => scope.search_tier_progressive(
                    cx,
                    tier,
                    self.loaded.fast.as_ref(),
                    self.text,
                    self.options.limit,
                    self.query_policy.query_pool(),
                )?,
                None => preview.search_tier_progressive(
                    cx,
                    tier,
                    self.loaded.fast.as_ref(),
                    self.text,
                    self.options.limit,
                    self.query_policy.query_pool(),
                )?,
            };
            let mut query = query_controls::configure_reranker(
                query,
                self.query_policy,
                self.options.max_source_bytes,
            )?;
            while let Some(phase) = query.next_phase().await? {
                let (name, results) = match phase {
                    NativeSearchPhase::Initial { results, .. } => (
                        if self.mode == QueryMode::Quality {
                            "quality"
                        } else {
                            "initial"
                        },
                        results,
                    ),
                    NativeSearchPhase::Reranked { results, .. } => ("reranked", results),
                    NativeSearchPhase::RerankFailed { error, .. }
                    | NativeSearchPhase::RefinementFailed { error, .. } => return Err(error.into()),
                    NativeSearchPhase::Refined { .. } => {
                        return Err(bad("single-tier preview cannot produce refinement"));
                    }
                };
                validate_results(name, &results)?;
                phases.push((name, results));
            }
        } else {
            let (phase, results) = if self.mode == QueryMode::Quality {
                (
                    "quality",
                    match (&scope, self.query_policy.query_pool()) {
                        (Some(scope), Some(pool)) => {
                            scope
                                .search_quality_text_with_pool(
                                    cx,
                                    self.loaded.fast.as_ref(),
                                    self.text,
                                    self.options.limit,
                                    pool,
                                )
                                .await?
                        }
                        (None, Some(pool)) => {
                            preview
                                .search_quality_text_with_pool(
                                    cx,
                                    self.loaded.fast.as_ref(),
                                    self.text,
                                    self.options.limit,
                                    pool,
                                )
                                .await?
                        }
                        (Some(scope), None) => {
                            scope
                                .search_quality_text(
                                    cx,
                                    self.loaded.fast.as_ref(),
                                    self.text,
                                    self.options.limit,
                                )
                                .await?
                        }
                        (None, None) => {
                            preview
                                .search_quality_text(
                                    cx,
                                    self.loaded.fast.as_ref(),
                                    self.text,
                                    self.options.limit,
                                )
                                .await?
                        }
                    },
                )
            } else {
                (
                    "initial",
                    match (&scope, self.query_policy.query_pool()) {
                        (Some(scope), Some(pool)) => {
                            scope
                                .search_text_with_pool(
                                    cx,
                                    self.loaded.fast.as_ref(),
                                    self.text,
                                    self.options.limit,
                                    pool,
                                )
                                .await?
                        }
                        (None, Some(pool)) => {
                            preview
                                .search_text_with_pool(
                                    cx,
                                    self.loaded.fast.as_ref(),
                                    self.text,
                                    self.options.limit,
                                    pool,
                                )
                                .await?
                        }
                        (Some(scope), None) => {
                            scope
                                .search_text(
                                    cx,
                                    self.loaded.fast.as_ref(),
                                    self.text,
                                    self.options.limit,
                                )
                                .await?
                        }
                        (None, None) => {
                            preview
                                .search_text(
                                    cx,
                                    self.loaded.fast.as_ref(),
                                    self.text,
                                    self.options.limit,
                                )
                                .await?
                        }
                    },
                )
            };
            validate_results(phase, &results)?;
            phases.push((phase, results));
        }
        validate_phases(self.mode, self.query_policy.rerank.is_some(), &phases)
    }

    fn failure(
        &self,
        recovery: &BoundImmutableReindexPlan,
        publication: &str,
        serving: &str,
        reason: &str,
    ) -> Box<dyn Error + Send + Sync> {
        let mut error = publication_error(self.selected, publication, serving, reason, None);
        if let Some(failure) = error.downcast_mut::<PublicationFailure>() {
            failure.0["operation"] = serde_json::json!("reindex");
            failure.0["recovery"] = serde_json::json!(recovery);
        }
        error
    }

    fn rollback(
        &self,
        cx: &Cx,
        published: &PublishedGeneration,
        path: &std::path::Path,
    ) -> Result<Receipt> {
        checkpoint(cx)?;
        let writer = WriterIdentityV1::for_current_process()?;
        let object_id = new_generation(1)?.nonce;
        let mut fence = Sha256::new();
        fence.update(b"frankensearch.native.immutable-reindex.rollback.v1\0");
        fence.update(writer.writer_id);
        fence.update(writer.nonce);
        fence.update(object_id);
        let candidate = self.previous.rollback_candidate(
            cx,
            &published.snapshot,
            object_id,
            fence.finalize().into(),
        )?;
        if candidate.owned_image_bytes() > u64::try_from(self.options.max_image_bytes)? {
            return Err(bad("reindex rollback exceeds the encoded-image budget"));
        }
        let mut restoration = GenerationPublisherV1::new(
            self.root,
            self.selected.root_id,
            writer.writer_id,
            PROFILE,
        )?;
        let attempt = restoration.prepare_attempt(
            &candidate,
            // The old retained target predates the rebuild. Only the actual
            // new selection supplies the pair this compensating CAS may change.
            published.snapshot.retained_snapshot().authority_pair(),
            new_generation(1)?.nonce,
            new_generation(1)?.nonce,
        )?;
        let receipt = Receipt {
            schema: if self.selected.quality_documents.is_some() {
                PARTIAL_QUALITY_RECEIPT_SCHEMA
            } else if self.selected.quality_identity.is_some() {
                PROGRESSIVE_RECEIPT_SCHEMA
            } else {
                RECEIPT_SCHEMA
            }
            .to_owned(),
            root: self.selected.root.clone(),
            root_id: self.selected.root_id,
            profile: self.selected.profile.clone(),
            authority: candidate.authority().canonical_bytes().to_vec(),
            generation: self.selected.generation,
            identity: self.selected.identity.clone(),
            tier: self.selected.tier,
            quality_identity: self.selected.quality_identity.clone(),
            quality_documents: self.selected.quality_documents,
            documents: self.selected.documents,
            image_bytes: candidate.owned_image_bytes(),
            attempt: Some(SavedAttempt::from_permit(&attempt, false)),
        };
        receipt.validate()?;
        encode(&receipt, MAX_RECEIPT_BYTES)?;
        checkpoint(cx)?;
        self.root.stage_immutable_file(
            &activation_manifest_path_v1(candidate.authority().object_id)?,
            candidate.manifest_bytes(),
        )?;
        self.root.revalidate()?;
        save_receipt(&receipt, path)?;
        publish_and_admit_with_reconciliation(
            cx,
            &mut restoration,
            &candidate,
            &attempt,
            &receipt,
            self.options.max_image_bytes,
            true,
        )
        .map_err(|mut error| {
            if let Some(failure) = error.downcast_mut::<PublicationFailure>() {
                failure.0["receipt"] = serde_json::json!(path);
            }
            error
        })?;
        Ok(receipt)
    }
}

fn cancelled(cx: &Cx, error: &(dyn Error + 'static)) -> bool {
    cx.is_cancel_requested()
        || error
            .downcast_ref::<SearchError>()
            .is_some_and(|error| matches!(error, SearchError::Cancelled { .. }))
}

fn timed_out(error: &(dyn Error + 'static)) -> bool {
    error
        .downcast_ref::<SearchError>()
        .is_some_and(|error| matches!(error, SearchError::SearchTimeout { .. }))
}

fn retrieval_policy(retrieval: NativeBuildRetrieval) -> serde_json::Value {
    match retrieval {
        NativeBuildRetrieval::Exact => serde_json::json!({"kind": "exact"}),
        NativeBuildRetrieval::Hnsw { params, seed } => serde_json::json!({
            "kind": "hnsw", "seed": seed, "m": params.m, "m0": params.m0,
            "ef_construction": params.ef_construction, "ef_search": params.ef_search,
        }),
    }
}

fn validate_results(phase: &str, results: &[ScoredResult]) -> Result<()> {
    let semantic = results.iter().any(|hit| {
        match phase {
            "initial" => hit.fast_score,
            "reranked" => hit.rerank_score,
            _ => hit.quality_score,
        }
        .is_some_and(f32::is_finite)
    });
    if results.is_empty() || !semantic || results.iter().any(|hit| !hit.score.is_finite()) {
        return Err(bad(
            "reindex query requires a nonempty semantic result in every required phase",
        ));
    }
    Ok(())
}

fn validate_phases(mode: QueryMode, rerank: bool, phases: &Phases) -> Result<()> {
    let expected: &[&str] = match mode {
        QueryMode::Fast if rerank => &["initial", "reranked"],
        QueryMode::Fast => &["initial"],
        QueryMode::Quality if rerank => &["quality", "reranked"],
        QueryMode::Quality => &["quality"],
        QueryMode::Full | QueryMode::Partial if rerank => &["initial", "refined", "reranked"],
        QueryMode::Full | QueryMode::Partial => &["initial", "refined"],
        QueryMode::Lexical => return Err(bad("lexical queries cannot prove semantic recovery")),
    };
    if phases
        .iter()
        .map(|(name, _)| *name)
        .eq(expected.iter().copied())
    {
        Ok(())
    } else {
        Err(bad(
            "reindex query did not complete the requested semantic topology",
        ))
    }
}
