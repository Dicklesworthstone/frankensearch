//! Complete or incrementally expand quality coverage using one retained authority.

use frankensearch_index::vector_cohort::VectorCohortCoverage;

use super::{
    Arc, AuthorityModels, AuthorityRefV1, AuthoritySession, BlockingPoolHandle, Cx, Deserialize,
    Embedder, EmbedderTierTag, Filter, HnswParams, IndexableDocument, MAX_DOCUMENTS,
    MAX_RECORD_BYTES, Models, NativeBuildPrecision, NativeBuildRetrieval,
    NativeBuiltExactGeneration, NativeGenerationSnapshot, NativeGenerationTopology, Options, Path,
    PathBuf, PublicationIntent, PublicationOptions, PublishedGeneration, QualifiedGenerationRoot,
    QueryMode, Receipt, Result, Write, admit_root, bad, checkpoint, emit_indexed,
    execution_context, load_authority_models, model_activation, models, new_generation, new_path,
    prepare_partial_quality_completion, prepare_publication_with_intent, rebuild_precision,
    retained_rebuild_documents, separate_destinations, serve_refused, stage_prepared_and_publish,
};

pub(super) async fn run(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let context = execution_context(cx, pool)?;
    Box::pin(with_loader(&context, options, output, || {
        load_authority_models(
            &context,
            options,
            QueryMode::Full,
            context.blocking_pool_handle(),
        )
    }))
    .await
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Retrieval {
    Exact,
    Hnsw,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    op: String,
    pub(super) id: Option<String>,
    expected_authority: Vec<u8>,
    work_dir: PathBuf,
    new_receipt: PathBuf,
    #[serde(default, deserialize_with = "super::super::serve::supplied_array")]
    quality_ids: Option<Vec<String>>,
    #[serde(default, deserialize_with = "supplied_quality_document_limit")]
    max_new_quality_documents: Option<usize>,
    retrieval: Option<Retrieval>,
    model_dir: Option<PathBuf>,
    quality_backend: Option<String>,
    quality_model_dir: Option<PathBuf>,
}

// Explicit null must not turn a bounded request into full completion.
fn supplied_quality_document_limit<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    usize::deserialize(deserializer).map(Some)
}

struct CompletionSpec {
    work: PathBuf,
    receipt: PathBuf,
    quality_ids: Option<Vec<String>>,
    max_new_quality_documents: Option<usize>,
    retrieval: Option<NativeBuildRetrieval>,
    batch_size: usize,
    max_source_bytes: usize,
    max_image_bytes: usize,
}

struct PreparedCompletion {
    spec: CompletionSpec,
    sequence: u64,
    quality_precision: NativeBuildPrecision,
    quality_retrieval: NativeBuildRetrieval,
}

async fn with_loader<F>(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    load_models: F,
) -> Result<()>
where
    F: FnOnce() -> Result<Models>,
{
    checkpoint(cx)?;
    validate_completion_selection(
        options.quality_ids.is_some(),
        options.max_new_quality_documents,
    )?;
    let selected = Receipt::read(
        options
            .from_receipt
            .as_deref()
            .ok_or_else(|| bad("missing predecessor receipt"))?,
    )?;
    let root = admit_root(&selected.root)?;
    let snapshot = selected.open_on_root(cx, &root, options.max_image_bytes)?;
    let prepared = prepare_completion(
        cx,
        &selected,
        &snapshot,
        CompletionSpec {
            work: options
                .work
                .clone()
                .ok_or_else(|| bad("missing work directory"))?,
            receipt: options.receipt.clone(),
            quality_ids: options.quality_ids.clone(),
            max_new_quality_documents: options.max_new_quality_documents,
            retrieval: options.retrieval,
            batch_size: options.batch_size,
            max_source_bytes: options.max_source_bytes,
            max_image_bytes: options.max_image_bytes,
        },
    )?;
    let loaded = load_models();
    checkpoint(cx)?;
    let loaded = loaded?;
    let built = Box::pin(build_completion(cx, &snapshot, &prepared, &loaded)).await?;
    let published = publish_completion(cx, &root, &selected, &snapshot, &prepared, &built, None)?;
    emit_indexed(
        output,
        &published.receipt,
        &prepared.spec.receipt,
        built.retrieval_configuration(),
        built.quality_retrieval_configuration(),
    )
}

fn validate_completion_selection(
    has_ids: bool,
    max_new_quality_documents: Option<usize>,
) -> Result<()> {
    if has_ids && max_new_quality_documents.is_some() {
        return Err(bad(
            "--quality-ids and --max-new-quality-documents cannot be combined",
        ));
    }
    if max_new_quality_documents.is_some_and(|limit| limit == 0 || limit > MAX_DOCUMENTS) {
        return Err(bad("the new quality document limit is out of range"));
    }
    Ok(())
}

fn prepare_completion(
    cx: &Cx,
    selected: &Receipt,
    snapshot: &NativeGenerationSnapshot,
    mut spec: CompletionSpec,
) -> Result<PreparedCompletion> {
    checkpoint(cx)?;
    validate_completion_selection(spec.quality_ids.is_some(), spec.max_new_quality_documents)?;
    if !matches!(
        selected.topology(),
        NativeGenerationTopology::FastOnly | NativeGenerationTopology::PartialQuality { .. }
    ) {
        return Err(bad(
            "quality completion requires a fast-only or partial-quality authority receipt",
        ));
    }
    if snapshot.retained_snapshot().selected_history_of().is_some()
        || snapshot.authority() != selected.authority()?
    {
        return Err(serve_refused("quality_completion_base_not_current"));
    }
    if spec.batch_size == 0
        || spec.batch_size > MAX_DOCUMENTS
        || spec.max_source_bytes == 0
        || spec.max_image_bytes == 0
    {
        return Err(serve_refused("quality_completion_invalid_limits"));
    }
    spec.work = new_path(&spec.work)?;
    spec.receipt = new_path(&spec.receipt)?;
    separate_destinations(&[&selected.root, &spec.work, &spec.receipt])?;
    snapshot.authority().next_sequence()?;
    let sequence = selected
        .generation
        .sequence
        .checked_add(1)
        .ok_or_else(|| bad("the artifact generation sequence is exhausted"))?;
    // Enforce the ordinary CLI source/record limits before model loading. The
    // completion plan then retains that same admitted source, never a file path.
    let documents = retained_rebuild_documents(cx, snapshot, spec.max_source_bytes)?;
    if let Some(ids) = &spec.quality_ids {
        validate_backfill_selection(cx, snapshot, &documents, ids)?;
    }
    if spec.max_new_quality_documents.is_some() {
        validate_next_batch_source(cx, snapshot, &documents)?;
    }
    drop(documents);
    // A partial predecessor already owns independently admitted quality rows.
    // Preserve their encoding so completion can reuse them exactly; only a new
    // quality tier receives the CLI's default F32 storage policy.
    let quality_precision = selected
        .quality_identity
        .as_ref()
        .map(rebuild_precision)
        .transpose()?
        .unwrap_or(NativeBuildPrecision::F32);
    let quality_retrieval = spec
        .retrieval
        .or_else(|| snapshot.quality_retrieval_configuration())
        .unwrap_or_default();
    checkpoint(cx)?;
    Ok(PreparedCompletion {
        spec,
        sequence,
        quality_precision,
        quality_retrieval,
    })
}

async fn build_completion(
    cx: &Cx,
    snapshot: &NativeGenerationSnapshot,
    prepared: &PreparedCompletion,
    loaded: &Models,
) -> Result<NativeBuiltExactGeneration> {
    let quality = loaded.quality.as_ref().ok_or_else(|| {
        bad("quality completion requires an independently supplied quality model")
    })?;
    let completion = snapshot
        .begin_quality_completion(
            cx,
            &prepared.spec.work,
            new_generation(prepared.sequence)?,
            Arc::clone(&loaded.fast),
            Arc::clone(quality),
            prepared.spec.max_source_bytes,
        )?
        .with_batch_size(prepared.spec.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
        .with_quality_storage(prepared.quality_precision, prepared.quality_retrieval)?;
    if let Some(ids) = &prepared.spec.quality_ids {
        completion
            .build_partial(cx, ids.iter().cloned(), prepared.spec.max_image_bytes)
            .await
            .map_err(Into::into)
    } else if let Some(limit) = prepared.spec.max_new_quality_documents {
        completion
            .build_next_batch(cx, limit, prepared.spec.max_image_bytes)
            .await
            .map_err(Into::into)
    } else {
        completion
            .build(cx, prepared.spec.max_image_bytes)
            .await
            .map_err(Into::into)
    }
}

fn publish_completion(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    snapshot: &NativeGenerationSnapshot,
    completion: &PreparedCompletion,
    built: &NativeBuiltExactGeneration,
    scope: Option<(&Filter, usize)>,
) -> Result<PublishedGeneration> {
    let budget = completion.spec.max_image_bytes;
    let prepared = if built.coverage() == VectorCohortCoverage::PartialQuality {
        prepare_partial_quality_completion(cx, built, snapshot, budget)?
    } else {
        prepare_publication_with_intent(
            built,
            Some(snapshot),
            budget,
            PublicationIntent::CompleteQuality,
        )?
    };
    stage_prepared_and_publish(
        cx,
        root,
        &selected.root,
        &completion.spec.receipt,
        built,
        prepared,
        PublicationOptions {
            max_image_bytes: budget,
            scope,
            reconcile_unknown: false,
        },
    )
}

fn validate_next_batch_source(
    cx: &Cx,
    snapshot: &NativeGenerationSnapshot,
    documents: &[IndexableDocument],
) -> Result<()> {
    checkpoint(cx)?;
    let covered: std::collections::BTreeSet<_> = match snapshot.topology() {
        NativeGenerationTopology::FastOnly => std::collections::BTreeSet::new(),
        NativeGenerationTopology::PartialQuality { .. } => snapshot
            .partial_quality_document_ids(cx)?
            .into_iter()
            .collect(),
        NativeGenerationTopology::QualityOnly | NativeGenerationTopology::FullProgressive => {
            return Err(bad("a quality batch requires incomplete quality coverage"));
        }
    };
    if !documents
        .iter()
        .any(|document| !covered.contains(&document.id))
    {
        return Err(bad(
            "a quality batch requires at least one uncovered source document",
        ));
    }
    checkpoint(cx)?;
    Ok(())
}

fn validate_backfill_selection(
    cx: &Cx,
    snapshot: &NativeGenerationSnapshot,
    documents: &[IndexableDocument],
    ids: &[String],
) -> Result<()> {
    checkpoint(cx)?;
    let selected: std::collections::BTreeSet<_> = ids.iter().map(String::as_str).collect();
    if selected.is_empty() || selected.len() != ids.len() || selected.len() >= documents.len() {
        return Err(bad(
            "quality backfill requires a unique nonempty proper subset; omit --quality-ids for full completion",
        ));
    }
    let source_ids: std::collections::BTreeSet<_> = documents
        .iter()
        .map(|document| document.id.as_str())
        .collect();
    if !selected.is_subset(&source_ids) {
        return Err(bad(
            "quality backfill IDs must belong to the selected source",
        ));
    }
    let previous = match snapshot.topology() {
        NativeGenerationTopology::FastOnly => Vec::new(),
        NativeGenerationTopology::PartialQuality { .. } => {
            snapshot.partial_quality_document_ids(cx)?
        }
        NativeGenerationTopology::QualityOnly | NativeGenerationTopology::FullProgressive => {
            return Err(bad("quality backfill requires incomplete quality coverage"));
        }
    };
    if selected.len() <= previous.len() || previous.iter().any(|id| !selected.contains(id.as_str()))
    {
        return Err(bad(
            "quality backfill must retain every covered document and add new IDs",
        ));
    }
    checkpoint(cx)?;
    Ok(())
}

impl AuthoritySession {
    pub(super) async fn complete_quality(
        &mut self,
        cx: &Cx,
        request: Request,
    ) -> Result<serde_json::Value> {
        Box::pin(
            self.complete_quality_with_loader(cx, request, |cx, directory, options| {
                models::load_single(
                    cx,
                    directory,
                    EmbedderTierTag::Quality,
                    options,
                    cx.blocking_pool_handle(),
                )
            }),
        )
        .await
    }

    async fn complete_quality_with_loader<F>(
        &mut self,
        cx: &Cx,
        request: Request,
        load_quality: F,
    ) -> Result<serde_json::Value>
    where
        F: FnOnce(&Cx, Option<&Path>, &models::QualityOptions) -> Result<Arc<dyn Embedder>>,
    {
        checkpoint(cx)?;
        self.require_current_selection()?;
        let limits = self
            .update_limits
            .ok_or_else(|| serve_refused("updates_disabled"))?;
        if !self.allow_model_activation {
            return Err(serve_refused("model_activation_disabled"));
        }
        self.models.admit(&self.receipt)?;
        let mut loaded = self.models.models()?.clone();
        super::super::serve::validate_id(request.id.as_deref())
            .map_err(|_| serve_refused("invalid_request_id"))?;
        if request.op != "complete_quality" {
            return Err(serve_refused("invalid_quality_completion_request"));
        }
        for path in [request.work_dir.as_path(), request.new_receipt.as_path()]
            .into_iter()
            .chain(request.model_dir.as_deref())
            .chain(request.quality_model_dir.as_deref())
        {
            model_activation::validate_path(path)?;
        }
        let expected = AuthorityRefV1::from_canonical_bytes(&request.expected_authority)
            .map_err(|_| serve_refused("invalid_expected_authority"))?;
        if expected != self.snapshot.authority() {
            return Err(serve_refused("stale_expected_authority"));
        }
        let quality_retained = loaded.quality.is_some();
        if quality_retained
            && (request.model_dir.is_some()
                || request.quality_backend.is_some()
                || request.quality_model_dir.is_some())
        {
            return Err(serve_refused("quality_completion_retains_quality_model"));
        }
        let options = models::QualityOptions {
            backend: request
                .quality_backend
                .as_deref()
                .map(models::QualityBackend::parse)
                .transpose()?,
            directory: request.quality_model_dir,
        };
        if !quality_retained {
            options.validate_usage(true)?;
            if options.backend.unwrap_or_default() == models::QualityBackend::Onnx
                && request.model_dir.is_none()
            {
                return Err(serve_refused("quality_completion_model_path_required"));
            }
        }
        // Reuse the session's qualified root and exact authority. A retained
        // reader remains queryable after a competing publication, but cannot
        // become the writable base of a new quality completion.
        let current = self
            .receipt
            .open_on_root(cx, &self.root, self.max_image_bytes)
            .map_err(|error| {
                model_activation::bounded_error(error.into(), "quality_completion_base_not_current")
            })?;
        if current.retained_snapshot().authority_pair()
            != self.snapshot.retained_snapshot().authority_pair()
        {
            return Err(serve_refused("quality_completion_base_changed"));
        }
        drop(current);
        let prepared = prepare_completion(
            cx,
            &self.receipt,
            &self.snapshot,
            CompletionSpec {
                work: request.work_dir,
                receipt: request.new_receipt,
                quality_ids: request.quality_ids,
                max_new_quality_documents: request.max_new_quality_documents,
                retrieval: request.retrieval.map(|retrieval| match retrieval {
                    Retrieval::Exact => NativeBuildRetrieval::Exact,
                    Retrieval::Hnsw => NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 42,
                    },
                }),
                batch_size: limits.batch_size,
                max_source_bytes: limits.max_source_bytes,
                max_image_bytes: self.max_image_bytes,
            },
        )
        .map_err(|error| {
            model_activation::bounded_error(error, "quality_completion_preflight_failed")
        })?;
        if !quality_retained {
            let quality = load_quality(cx, request.model_dir.as_deref(), &options);
            checkpoint(cx)?;
            loaded.quality = Some(quality.map_err(|error| {
                model_activation::bounded_error(error, "quality_completion_load_failed")
            })?);
        }
        // Query caches stay separate from document inference. An existing
        // partial tier keeps both caches; a newly added quality tier receives
        // its own cache while retaining the exact fast cache owner.
        let cached = if quality_retained {
            self.cached_query_models.clone()
        } else {
            self.cache_replacement_models(cx, &loaded, true)?
        };
        let built = Box::pin(build_completion(cx, &self.snapshot, &prepared, &loaded)).await?;
        super::admit_serving_model(built.binding().frozen_identity(), loaded.fast.as_ref())?;
        super::admit_serving_model(
            built
                .quality_binding()
                .ok_or_else(|| serve_refused("quality_completion_missing_quality"))?
                .frozen_identity(),
            loaded
                .quality
                .as_ref()
                .ok_or_else(|| serve_refused("quality_completion_missing_quality"))?
                .as_ref(),
        )?;
        checkpoint(cx)?;
        let previously_covered = self.receipt.quality_documents.unwrap_or(0);
        let published = publish_completion(
            cx,
            &self.root,
            &self.receipt,
            &self.snapshot,
            &prepared,
            &built,
            self.fixed_filter
                .as_ref()
                .map(|filter| (filter, self.max_source_bytes)),
        )?;
        // Publication returns the exact admitted snapshot and scope. Install
        // it without reopening paths or checkpointing after the durable commit.
        self.receipt = published.receipt;
        self.snapshot = published.snapshot;
        self.scope = published.scope;
        self.models = AuthorityModels::Semantic(loaded);
        self.cached_query_models = cached;
        let mut response = self.response("quality_completed", true, None, request.id.as_deref());
        response["operation"] = serde_json::json!("complete_quality");
        response["publication"] = serde_json::json!("committed");
        response["serving"] = serde_json::json!("activated");
        response["previous_authority"] = serde_json::json!(expected.canonical_bytes().as_slice());
        response["receipt"] = serde_json::json!(prepared.spec.receipt);
        response["selection"] = serde_json::json!(self.receipt);
        response["documents"] = serde_json::json!(self.receipt.documents);
        response["quality_documents_added"] = serde_json::json!(
            self.receipt
                .quality_documents
                .unwrap_or(self.receipt.documents)
                - previously_covered
        );
        response["models_loaded"] = serde_json::json!(usize::from(!quality_retained));
        response["fast_model_retained"] = serde_json::json!(true);
        response["quality_model_retained"] = serde_json::json!(quality_retained);
        self.annotate_query_capabilities(&mut response);
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        Command, MAX_DOCUMENTS, NativeBuildRetrieval, ReceiptQueryMode, is_command,
    };
    use super::*;

    fn arguments() -> Vec<String> {
        [
            "authority-complete-quality",
            "--from-receipt",
            "fast.json",
            "--work-dir",
            "new-work",
            "--receipt",
            "full.json",
        ]
        .map(str::to_owned)
        .into()
    }

    #[test]
    fn warm_quality_completion_requires_explicit_arrays_and_rejects_writer_extras() {
        let value = serde_json::json!({
            "op": "complete_quality",
            "expected_authority": [],
            "work_dir": "/new-quality-work",
            "new_receipt": "/new-quality.json",
            "model_dir": "/local-models",
        });
        assert!(matches!(
            serde_json::from_value::<super::super::AuthorityRequest>(value.clone()).unwrap(),
            super::super::AuthorityRequest::CompleteQuality(_)
        ));
        let parsed: Request = serde_json::from_value(value.clone()).unwrap();
        assert!(parsed.quality_ids.is_none());
        for invalid in [
            serde_json::Value::Null,
            serde_json::json!("all"),
            serde_json::json!({}),
        ] {
            let mut candidate = value.clone();
            candidate["quality_ids"] = invalid;
            assert!(serde_json::from_value::<super::super::AuthorityRequest>(candidate).is_err());
        }
        for field in [
            "documents",
            "changes",
            "filter",
            "source_low",
            "tier",
            "batch_size",
        ] {
            let mut candidate = value.clone();
            candidate[field] = serde_json::json!([]);
            assert!(
                serde_json::from_value::<Request>(candidate).is_err(),
                "{field}"
            );
        }
        for invalid in [
            serde_json::Value::Null,
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("2"),
        ] {
            let mut candidate = value.clone();
            candidate["max_new_quality_documents"] = invalid;
            assert!(serde_json::from_value::<super::super::AuthorityRequest>(candidate).is_err());
        }
        let mut automatic = value.clone();
        automatic["max_new_quality_documents"] = serde_json::json!(2);
        let parsed: Request = serde_json::from_value(automatic).unwrap();
        assert_eq!(parsed.max_new_quality_documents, Some(2));
        assert!(parsed.quality_ids.is_none());
        let mut partial = value;
        partial["quality_ids"] = serde_json::json!(["garden", "retry"]);
        partial["retrieval"] = serde_json::json!("hnsw");
        let parsed: Request = serde_json::from_value(partial.clone()).unwrap();
        assert_eq!(parsed.quality_ids.unwrap(), ["garden", "retry"]);
        partial["retrieval"] = serde_json::json!("automatic");
        assert!(serde_json::from_value::<Request>(partial).is_err());

        // An explicit null mutation list must not become permission to clear
        // the current corpus through an accompanying empty full inventory.
        for field in ["changes", "documents"] {
            let mut update = serde_json::json!({
                "op": "update", "expected_authority": [],
                "work_dir": "/new-update-work", "new_receipt": "/new-update.json",
                "source_low": 1, "source_high": 2, "documents": [],
            });
            update[field] = serde_json::Value::Null;
            assert!(serde_json::from_value::<super::super::AuthorityRequest>(update).is_err());
        }
    }

    #[test]
    fn completion_has_explicit_topology_intent_and_never_accepts_replacement_sources() {
        let options = Options::parse(arguments()).unwrap().unwrap();
        assert_eq!(options.command, Command::CompleteQuality);
        assert!(is_command("authority-complete-quality"));
        for extra in [
            vec!["--input", "replacement.jsonl"],
            vec!["--source-low", "1"],
            vec!["--source-high", "2"],
            vec!["--migrate-model"],
            vec!["--retained-source"],
            vec!["--current-receipt", "other.json"],
            vec!["--tier", "full"],
            vec!["--query", "retry"],
        ] {
            let mut args = arguments();
            args.extend(extra.into_iter().map(str::to_owned));
            assert!(Options::parse(args).is_err());
        }
        let mut args = arguments();
        args.extend(
            [
                "--retrieval",
                "hnsw",
                "--batch-size",
                "2",
                "--max-source-bytes",
                "8192",
            ]
            .map(str::to_owned),
        );
        let options = Options::parse(args).unwrap().unwrap();
        assert_eq!(options.batch_size, 2);
        assert_eq!(options.max_source_bytes, 8192);
        assert!(matches!(
            options.retrieval,
            Some(NativeBuildRetrieval::Hnsw { .. })
        ));
    }

    #[test]
    fn bounded_completion_accepts_only_explicit_unique_quality_membership() {
        let mut args = arguments();
        args.extend(["--quality-ids", r#"["garden","retry"]"#].map(str::to_owned));
        let options = Options::parse(args).unwrap().unwrap();
        assert_eq!(
            options.quality_ids,
            Some(vec!["garden".to_owned(), "retry".to_owned()])
        );
        for selection in [
            "[]",
            r#"["retry","retry"]"#,
            r#"[""]"#,
            r#"["\u0000"]"#,
            r#"{"id":"retry"}"#,
        ] {
            let mut args = arguments();
            args.extend(["--quality-ids", selection].map(str::to_owned));
            assert!(Options::parse(args).is_err(), "{selection}");
        }
    }

    #[test]
    fn automatic_completion_accepts_a_positive_bounded_count_without_explicit_ids() {
        for limit in [1, MAX_DOCUMENTS] {
            let mut args = arguments();
            args.extend(["--max-new-quality-documents".to_owned(), limit.to_string()]);
            let options = Options::parse(args.clone()).unwrap().unwrap();
            assert_eq!(options.max_new_quality_documents, Some(limit));
            assert!(options.quality_ids.is_none());
            args.extend(["--quality-ids", r#"["garden"]"#].map(str::to_owned));
            assert!(Options::parse(args).is_err());
        }
        for value in [
            "0".to_owned(),
            (MAX_DOCUMENTS + 1).to_string(),
            usize::MAX.to_string(),
            "-1".to_owned(),
            "not-a-count".to_owned(),
        ] {
            let mut args = arguments();
            args.extend(["--max-new-quality-documents".to_owned(), value]);
            assert!(Options::parse(args).is_err());
        }
        let mut args = arguments();
        args.push("--max-new-quality-documents".to_owned());
        assert!(Options::parse(args).is_err());
    }

    #[test]
    fn registered_fast_and_quality_models_share_the_retained_prepared_input_contract() {
        use frankensearch::embed::model_manifest::ModelArtifactManifestV1;

        // This checks registered contract compatibility, not model availability
        // or real inference. Model-specific tokenizers/normalization remain in
        // the independently authenticated space identities.
        let fast = [
            ModelArtifactManifestV1::potion_128m_native().unwrap(),
            ModelArtifactManifestV1::potion_base_8m_native().unwrap(),
            ModelArtifactManifestV1::potion_base_32m_native().unwrap(),
        ];
        let quality = [
            ModelArtifactManifestV1::minilm_fastembed().unwrap(),
            ModelArtifactManifestV1::minilm_native_frankentorch().unwrap(),
            ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap(),
            ModelArtifactManifestV1::multilingual_minilm_native_frankentorch().unwrap(),
        ];
        for fast in &fast {
            for quality in &quality {
                assert_eq!(
                    fast.execution.input_contract,
                    quality.execution.input_contract
                );
            }
        }
    }

    #[cfg(target_os = "linux")]
    mod lifecycle {
        use super::super::super::{
            CommitRange, DEFAULT_IMAGE_BYTES, Embedder, EmbedderTierTag, HnswParams,
            IndexableDocument, MAX_INPUT_BYTES, NativeGenerationSnapshot, NativeIndexBuilder,
            PublicationFailure, SearchError, create_root, prepare_publication, stage_and_publish,
        };
        use super::*;
        use frankensearch_core::generation::{
            EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
            GenerationComponentRole,
        };
        use frankensearch_core::{ModelCategory, SearchFuture, SearchResult};
        use std::fs;
        use std::os::unix::fs::DirBuilderExt as _;
        use std::path::PathBuf;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        struct Provider {
            identity: EmbeddingIdentityBundleV1,
            foreign: EmbeddingIdentityBundleV1,
            calls: AtomicUsize,
            fault: AtomicUsize,
            drifted: AtomicBool,
        }

        impl Provider {
            fn new(quality: bool) -> Arc<Self> {
                let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(
                    if quality {
                        "completion-quality"
                    } else {
                        "completion-fast"
                    },
                    if quality { 3 } else { 2 },
                );
                // Synthetic providers exercise production binding and authority
                // publication, without claiming real-model relevance evidence.
                identity.space.kind = EmbeddingSpaceKindV1::Semantic;
                identity.space.hash_control = None;
                identity.space.artifact_manifest_fingerprint = "a".repeat(64);
                identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
                    role: "weights".to_owned(),
                    sha256: if quality { "c" } else { "b" }.repeat(64),
                    size: 1,
                }];
                identity.producer.space_fingerprint = identity.space.fingerprint();
                identity.validate().unwrap();
                let mut foreign = identity.clone();
                foreign
                    .producer
                    .implementation_revision
                    .push_str("-foreign");
                Arc::new(Self {
                    identity,
                    foreign,
                    calls: AtomicUsize::new(0),
                    fault: AtomicUsize::new(0),
                    drifted: AtomicBool::new(false),
                })
            }
        }

        impl Embedder for Provider {
            fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
                Ok(if self.drifted.load(Ordering::SeqCst) {
                    &self.foreign
                } else {
                    &self.identity
                })
            }

            fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                Box::pin(async move {
                    self.calls.fetch_add(1, Ordering::SeqCst);
                    match self.fault.load(Ordering::SeqCst) {
                        1 => {
                            return Err(SearchError::InvalidConfig {
                                field: "completion.test".to_owned(),
                                value: "inference-forbidden".to_owned(),
                                reason: "injected provider failure".to_owned(),
                            });
                        }
                        2 => {
                            cx.set_cancel_requested(true);
                        }
                        3 => self.drifted.store(true, Ordering::SeqCst),
                        _ => {}
                    }
                    let mut values = vec![0.0; self.dimension()];
                    values[usize::from(!text.contains("retry"))] = 1.0;
                    Ok(values)
                })
            }

            fn dimension(&self) -> usize {
                usize::try_from(self.identity.space.dimension).unwrap()
            }
            fn id(&self) -> &str {
                &self.identity.space.logical_model_id
            }
            fn model_name(&self) -> &str {
                self.id()
            }
            fn is_semantic(&self) -> bool {
                true
            }
            fn category(&self) -> ModelCategory {
                ModelCategory::TransformerEmbedder
            }
        }

        struct Fixture {
            parent: PathBuf,
            fast: Arc<Provider>,
            selected: Receipt,
            snapshot: Arc<NativeGenerationSnapshot>,
        }

        async fn fixture(cx: &Cx, ann: bool) -> Fixture {
            fixture_with_quality(cx, ann, None).await
        }

        async fn fixture_with_quality(
            cx: &Cx,
            ann: bool,
            quality: Option<Arc<Provider>>,
        ) -> Fixture {
            fixture_with_documents(
                cx,
                ann,
                quality,
                vec![
                    IndexableDocument::new("retry", "retry network requests")
                        .with_metadata("revision", "old"),
                    IndexableDocument::new("garden", "garden flowers"),
                ],
            )
            .await
        }

        async fn fixture_with_documents(
            cx: &Cx,
            ann: bool,
            quality: Option<Arc<Provider>>,
            documents: Vec<IndexableDocument>,
        ) -> Fixture {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let parent = PathBuf::from(
                std::env::var_os("FRANKENSEARCH_TEST_TMPDIR")
                    .or_else(|| std::env::var_os("HOME"))
                    .unwrap(),
            )
            .join(format!(
                ".frankensearch-quality-completion-{}-{stamp}",
                std::process::id(),
            ));
            fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
            let root = create_root(&parent.join("root")).expect("qualified Linux root required");
            let fast = Provider::new(false);
            let builder = NativeIndexBuilder::new(
                parent.join("original-work"),
                new_generation(1).unwrap(),
                fast.clone(),
            )
            .unwrap()
            .with_fast_storage(
                NativeBuildPrecision::F16,
                if ann {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 17,
                    }
                } else {
                    NativeBuildRetrieval::Exact
                },
            )
            .add_documents(documents);
            let built = if let Some(quality) = quality {
                builder
                    .with_quality_embedder(quality)
                    .unwrap()
                    .with_quality_storage(
                        NativeBuildPrecision::F16,
                        if ann {
                            NativeBuildRetrieval::Hnsw {
                                params: HnswParams::default(),
                                seed: 29,
                            }
                        } else {
                            NativeBuildRetrieval::Exact
                        },
                    )
                    .unwrap()
                    .build_partial_quality_generation(
                        cx,
                        CommitRange { low: 4, high: 9 },
                        ["garden".to_owned()],
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap()
            } else {
                builder
                    .build_exact_generation(
                        cx,
                        EmbedderTierTag::Fast,
                        CommitRange { low: 4, high: 9 },
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap()
            };
            let published = stage_and_publish(
                cx,
                &root,
                &parent.join("root"),
                &parent.join("fast.json"),
                &built,
                None,
                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
            )
            .unwrap();
            // Completion must use admitted source bytes, not the original build
            // directory or any original corpus path.
            fs::rename(
                parent.join("original-work"),
                parent.join("unavailable-work"),
            )
            .unwrap();
            fast.fault.store(1, Ordering::SeqCst);
            Fixture {
                parent,
                fast,
                selected: published.receipt,
                snapshot: published.snapshot,
            }
        }

        fn options(fixture: &Fixture, label: &str) -> Options {
            let mut options = Options::parse(arguments()).unwrap().unwrap();
            options.from_receipt = Some(fixture.parent.join("fast.json"));
            options.work = Some(fixture.parent.join(format!("{label}-work")));
            options.receipt = fixture.parent.join(format!("{label}.json"));
            options
        }

        fn backfill_documents() -> Vec<IndexableDocument> {
            vec![
                IndexableDocument::new("retry", "retry network requests")
                    .with_metadata("revision", "old"),
                IndexableDocument::new("garden", "garden flowers"),
                IndexableDocument::new("notes", "field notes"),
                IndexableDocument::new("uncovered", "a final document"),
            ]
        }

        #[test]
        fn automatic_completion_resumes_from_actual_membership_and_finishes_full() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    for starts_partial in [false, true] {
                        let quality = Provider::new(true);
                        let mut documents = backfill_documents();
                        // The existing covered ID is not the first source ID.
                        // Resumption must inspect membership, not use its count as a cursor.
                        documents.push(IndexableDocument::new("a-first", "alphabetically first"));
                        let fixture = fixture_with_documents(
                            &cx,
                            ann,
                            starts_partial.then(|| quality.clone()),
                            documents,
                        )
                        .await;
                        let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                        let batches = if starts_partial {
                            vec![
                                (1, vec!["a-first", "garden"]),
                                (2, vec!["a-first", "garden", "notes", "retry"]),
                                (8, vec!["a-first", "garden", "notes", "retry", "uncovered"]),
                            ]
                        } else {
                            vec![
                                (1, vec!["a-first"]),
                                (2, vec!["a-first", "garden", "notes"]),
                                (8, vec!["a-first", "garden", "notes", "retry", "uncovered"]),
                            ]
                        };
                        let mut current_path = fixture.parent.join("fast.json");
                        let mut current_receipt = fixture.selected.clone();
                        let mut current = Arc::clone(&fixture.snapshot);
                        let mut covered = usize::from(starts_partial);
                        for (step, (limit, expected_ids)) in batches.into_iter().enumerate() {
                            let mut batch = options(&fixture, &format!("automatic-{step}"));
                            batch.from_receipt = Some(current_path.clone());
                            batch.max_new_quality_documents = Some(limit);
                            batch.batch_size = 1;
                            let quality_calls = quality.calls.load(Ordering::SeqCst);
                            let mut output = Vec::new();
                            with_loader(&cx, &batch, &mut output, || {
                                Ok(Models {
                                    fast: fixture.fast.clone(),
                                    quality: Some(quality.clone()),
                                })
                            })
                            .await
                            .unwrap();
                            let added = expected_ids.len() - covered;
                            assert!(added > 0 && added <= limit);
                            assert_eq!(quality.calls.load(Ordering::SeqCst), quality_calls + added);
                            assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                            let receipt = Receipt::read(&batch.receipt).unwrap();
                            let reopened = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                            assert_eq!(receipt.identity, fixture.selected.identity);
                            if current_receipt.quality_identity.is_some() {
                                assert_eq!(
                                    receipt.quality_identity,
                                    current_receipt.quality_identity
                                );
                            }
                            assert_eq!(
                                receipt.generation.sequence,
                                current_receipt.generation.sequence + 1
                            );
                            assert_eq!(
                                reopened.retained_snapshot().manifest().predecessor(),
                                Some(current.authority())
                            );
                            assert_eq!(
                                reopened.retained_snapshot().manifest().components().lexical,
                                fixture
                                    .snapshot
                                    .retained_snapshot()
                                    .manifest()
                                    .components()
                                    .lexical,
                            );
                            assert_eq!(
                                reopened.metadata().commit_range,
                                fixture.snapshot.metadata().commit_range
                            );
                            assert_eq!(
                                reopened.retrieval_configuration(),
                                fixture.snapshot.retrieval_configuration()
                            );
                            assert_eq!(
                                serde_json::to_value(
                                    reopened.source_documents(&cx, MAX_INPUT_BYTES).unwrap()
                                )
                                .unwrap(),
                                serde_json::to_value(
                                    fixture
                                        .snapshot
                                        .source_documents(&cx, MAX_INPUT_BYTES)
                                        .unwrap()
                                )
                                .unwrap(),
                            );
                            if expected_ids.len() == 5 {
                                assert_eq!(receipt.quality_documents, None);
                                assert_eq!(
                                    reopened.topology(),
                                    NativeGenerationTopology::FullProgressive
                                );
                                assert_eq!(
                                    receipt.query_mode(Some(QueryMode::Full)).unwrap(),
                                    QueryMode::Full
                                );
                                assert_eq!(
                                    receipt.query_mode(Some(QueryMode::Quality)).unwrap(),
                                    QueryMode::Quality
                                );
                            } else {
                                assert_eq!(
                                    receipt.quality_documents,
                                    Some(u64::try_from(expected_ids.len()).unwrap())
                                );
                                assert_eq!(
                                    reopened.partial_quality_document_ids(&cx).unwrap(),
                                    expected_ids
                                );
                                assert!(receipt.query_mode(Some(QueryMode::Full)).is_err());
                                assert_eq!(
                                    receipt.query_mode(Some(QueryMode::Partial)).unwrap(),
                                    QueryMode::Partial
                                );
                            }
                            let report: serde_json::Value =
                                serde_json::from_slice(&output).unwrap();
                            assert_eq!(report["publication"], "committed");
                            assert_eq!(
                                report["selection"]["quality_documents"],
                                serde_json::to_value(receipt.quality_documents).unwrap()
                            );
                            covered = expected_ids.len();
                            current_path = batch.receipt;
                            current_receipt = receipt;
                            current = Arc::new(reopened);
                        }
                        // Full and stale receipts cannot load a model for another batch.
                        for (label, path) in [
                            ("already-full", current_path),
                            ("stale", fixture.parent.join("fast.json")),
                        ] {
                            let mut batch = options(&fixture, label);
                            batch.from_receipt = Some(path);
                            batch.max_new_quality_documents = Some(1);
                            let mut loads = 0;
                            assert!(
                                with_loader(&cx, &batch, &mut Vec::new(), || {
                                    loads += 1;
                                    Ok(Models {
                                        fast: fixture.fast.clone(),
                                        quality: Some(quality.clone()),
                                    })
                                })
                                .await
                                .is_err()
                            );
                            assert_eq!(loads, 0);
                            assert!(!batch.work.as_ref().unwrap().exists());
                            assert!(!batch.receipt.exists());
                        }
                        fixture.fast.fault.store(0, Ordering::SeqCst);
                        let hits = current
                            .search_progressive_with_quality(
                                &cx,
                                fixture.fast.as_ref(),
                                quality.as_ref(),
                                "final document",
                                5,
                            )
                            .unwrap()
                            .collect()
                            .await
                            .unwrap();
                        assert!(
                            hits.iter()
                                .find(|hit| hit.doc_id == "uncovered")
                                .unwrap()
                                .quality_score
                                .is_some()
                        );
                        assert_eq!(fixture.snapshot.topology(), fixture.selected.topology());
                    }
                }
            });
        }

        #[test]
        fn automatic_completion_refuses_empty_or_invalid_work_before_loading_models() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for empty in [false, true] {
                    let fixture = fixture_with_documents(
                        &cx,
                        false,
                        None,
                        if empty {
                            Vec::new()
                        } else {
                            backfill_documents()
                        },
                    )
                    .await;
                    let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                    let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                    for case in if empty {
                        vec!["empty"]
                    } else {
                        vec![
                            "zero",
                            "too-many",
                            "explicit-ids",
                            "source-budget",
                            "cancelled",
                        ]
                    } {
                        let mut batch = options(&fixture, case);
                        batch.max_new_quality_documents = Some(1);
                        match case {
                            "zero" => batch.max_new_quality_documents = Some(0),
                            "too-many" => batch.max_new_quality_documents = Some(MAX_DOCUMENTS + 1),
                            "explicit-ids" => batch.quality_ids = Some(vec!["garden".to_owned()]),
                            "source-budget" => batch.max_source_bytes = 1,
                            "cancelled" => cx.set_cancel_requested(true),
                            _ => {}
                        }
                        let mut loads = 0;
                        let mut output = Vec::new();
                        let result = with_loader(&cx, &batch, &mut output, || {
                            loads += 1;
                            Err(bad("invalid automatic work must not load models"))
                        })
                        .await;
                        cx.set_cancel_requested(false);
                        assert!(result.is_err(), "{case}");
                        assert_eq!(loads, 0, "{case}");
                        assert!(output.is_empty());
                        assert!(!batch.work.as_ref().unwrap().exists());
                        assert!(!batch.receipt.exists());
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(
                            fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                            authority
                        );
                    }
                    fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                }
            });
        }

        #[test]
        fn automatic_completion_retries_failed_batches_and_fences_a_concurrent_final_batch() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let quality = Provider::new(true);
                let fixture =
                    fixture_with_documents(&cx, true, Some(quality.clone()), backfill_documents())
                        .await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                for fault in [1, 2, 3] {
                    let failed_quality = Provider::new(true);
                    failed_quality.fault.store(fault, Ordering::SeqCst);
                    let mut batch = options(&fixture, &format!("automatic-fault-{fault}"));
                    batch.max_new_quality_documents = Some(1);
                    batch.batch_size = 1;
                    let mut output = Vec::new();
                    let result = with_loader(&cx, &batch, &mut output, || {
                        Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(failed_quality.clone()),
                        })
                    })
                    .await;
                    cx.set_cancel_requested(false);
                    assert!(result.is_err());
                    if fault == 2 {
                        assert!(matches!(
                            result.unwrap_err().downcast_ref::<SearchError>(),
                            Some(SearchError::Cancelled { .. })
                        ));
                    }
                    assert_eq!(failed_quality.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert!(output.is_empty());
                    assert!(!batch.receipt.exists());
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                    assert_eq!(
                        fixture
                            .selected
                            .open(&cx, DEFAULT_IMAGE_BYTES)
                            .unwrap()
                            .partial_quality_document_ids(&cx)
                            .unwrap(),
                        ["garden"]
                    );
                }
                // Retry with the same receipt: the failed work consumed no IDs.
                let mut retry = options(&fixture, "automatic-retry");
                retry.max_new_quality_documents = Some(1);
                let quality_calls = quality.calls.load(Ordering::SeqCst);
                with_loader(&cx, &retry, &mut Vec::new(), || {
                    Ok(Models {
                        fast: fixture.fast.clone(),
                        quality: Some(quality.clone()),
                    })
                })
                .await
                .unwrap();
                assert_eq!(quality.calls.load(Ordering::SeqCst), quality_calls + 1);
                let current_receipt = Receipt::read(&retry.receipt).unwrap();
                let current = current_receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    current.partial_quality_document_ids(&cx).unwrap(),
                    ["garden", "notes"]
                );

                let winner = current
                    .begin_quality_completion(
                        &cx,
                        fixture.parent.join("automatic-winner-work"),
                        new_generation(current_receipt.generation.sequence + 1).unwrap(),
                        fixture.fast.clone(),
                        quality.clone(),
                        MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .build_partial(
                        &cx,
                        ["garden", "notes", "retry"].map(str::to_owned),
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap();
                let prepared =
                    prepare_partial_quality_completion(&cx, &winner, &current, DEFAULT_IMAGE_BYTES)
                        .unwrap();
                let root = admit_root(&current_receipt.root).unwrap();
                let expected = current.retained_snapshot().authority_pair();
                let mut losing = options(&fixture, "automatic-stale-final");
                losing.from_receipt = Some(retry.receipt.clone());
                losing.max_new_quality_documents = Some(8);
                let mut winning_receipt = None;
                let mut output = Vec::new();
                let error = with_loader(&cx, &losing, &mut output, || {
                    let published = stage_prepared_and_publish(
                        &cx,
                        &root,
                        &current_receipt.root,
                        &fixture.parent.join("automatic-winner.json"),
                        &winner,
                        prepared,
                        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                    )?;
                    winning_receipt = Some(published.receipt);
                    Ok(Models {
                        fast: fixture.fast.clone(),
                        quality: Some(quality.clone()),
                    })
                })
                .await
                .unwrap_err();
                assert_eq!(
                    error.downcast_ref::<PublicationFailure>().unwrap().0["publication"],
                    "not_committed"
                );
                assert!(output.is_empty());
                assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                let winner = winning_receipt.unwrap();
                let reopened = winner.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    reopened.partial_quality_document_ids(&cx).unwrap(),
                    ["garden", "notes", "retry"]
                );
                let attempted = Receipt::read(&losing.receipt).unwrap();
                assert_eq!(
                    attempted.quality_documents, None,
                    "the losing final batch was complete"
                );
                assert_eq!(attempted.recovery_attempt().unwrap().expected, expected);
                assert_ne!(attempted.authority().unwrap(), winner.authority().unwrap());
                assert!(attempted.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
                assert_eq!(
                    current.partial_quality_document_ids(&cx).unwrap(),
                    ["garden", "notes"]
                );
            });
        }

        fn warm_session(
            cx: &Cx,
            fixture: &Fixture,
            quality: Option<Arc<Provider>>,
        ) -> AuthoritySession {
            let mut session = AuthoritySession::new(
                cx,
                admit_root(&fixture.selected.root).unwrap(),
                fixture.selected.clone(),
                Arc::clone(&fixture.snapshot),
                Models {
                    fast: fixture.fast.clone(),
                    quality: quality.map(|model| model as Arc<dyn Embedder>),
                },
                (8, DEFAULT_IMAGE_BYTES),
                false,
            )
            .unwrap()
            .with_updates(2, MAX_INPUT_BYTES)
            .unwrap()
            .with_query_cache(
                cx,
                super::super::super::query_cache::Config::parse("4").unwrap(),
            )
            .unwrap()
            .with_filter(
                cx,
                Some(serde_json::from_value(serde_json::json!({"id_prefix": "retry"})).unwrap()),
                MAX_INPUT_BYTES,
            )
            .unwrap();
            session.allow_model_activation = true;
            session
        }

        fn warm_request_value(
            session: &AuthoritySession,
            fixture: &Fixture,
            label: &str,
            ids: Option<&[&str]>,
        ) -> serde_json::Value {
            let mut request = serde_json::json!({
                "op": "complete_quality", "id": label,
                "expected_authority": session.snapshot.authority().canonical_bytes().as_slice(),
                "work_dir": fixture.parent.join(format!("{label}-work")),
                "new_receipt": fixture.parent.join(format!("{label}.json")),
            });
            if let Some(ids) = ids {
                request["quality_ids"] = serde_json::json!(ids);
            }
            if session.models.models().unwrap().quality.is_none() {
                request["model_dir"] = serde_json::json!(fixture.parent.join("local-models"));
            }
            request
        }

        #[test]
        fn warm_quality_completion_reuses_models_caches_scope_and_retained_rows() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    for starts_partial in [false, true] {
                        let quality = Provider::new(true);
                        let fixture = fixture_with_documents(
                            &cx,
                            ann,
                            starts_partial.then(|| quality.clone()),
                            backfill_documents(),
                        )
                        .await;
                        let mut session =
                            warm_session(&cx, &fixture, starts_partial.then(|| quality.clone()));
                        let raw_fast = Arc::clone(&session.models.models().unwrap().fast);
                        let cached_fast =
                            Arc::clone(&session.query_models().models().unwrap().fast);
                        let cached_quality =
                            session.query_models().models().unwrap().quality.clone();
                        fixture.fast.fault.store(0, Ordering::SeqCst);
                        // Production query caching binds the producer identity;
                        // raw embed intentionally bypasses the serving cache.
                        cached_fast.embed_bound(&cx, "cached retry").await.unwrap();
                        fixture.fast.fault.store(1, Ordering::SeqCst);
                        let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                        let quality_calls = quality.calls.load(Ordering::SeqCst);
                        let loads = AtomicUsize::new(0);
                        let mut request = warm_request_value(
                            &session,
                            &fixture,
                            "warm-partial",
                            Some(&["garden", "notes", "retry"]),
                        );
                        if !starts_partial {
                            request["retrieval"] =
                                serde_json::json!(if ann { "hnsw" } else { "exact" });
                        }
                        let response = session
                            .complete_quality_with_loader(
                                &cx,
                                serde_json::from_value(request).unwrap(),
                                |_, _, _| -> Result<Arc<dyn Embedder>> {
                                    loads.fetch_add(1, Ordering::SeqCst);
                                    Ok(quality.clone())
                                },
                            )
                            .await
                            .unwrap();
                        assert_eq!(loads.load(Ordering::SeqCst), usize::from(!starts_partial));
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(
                            quality.calls.load(Ordering::SeqCst) - quality_calls,
                            3 - usize::from(starts_partial),
                        );
                        assert_eq!(response["publication"], "committed");
                        assert_eq!(response["serving"], "activated");
                        assert_eq!(
                            response["query_modes"],
                            serde_json::json!(["fast", "partial", "lexical"])
                        );
                        assert_eq!(response["default_mode"], "fast");
                        assert_eq!(response["quality_completion_enabled"], true);
                        assert_eq!(response["scope"]["eligible_documents"], 1);
                        assert_eq!(session.scope.as_ref().unwrap().len(), 1);
                        assert!(Arc::ptr_eq(
                            &raw_fast,
                            &session.models.models().unwrap().fast
                        ));
                        assert!(Arc::ptr_eq(
                            &cached_fast,
                            &session.query_models().models().unwrap().fast
                        ));
                        if let Some(previous) = cached_quality {
                            assert!(Arc::ptr_eq(
                                &previous,
                                session
                                    .query_models()
                                    .models()
                                    .unwrap()
                                    .quality
                                    .as_ref()
                                    .unwrap(),
                            ));
                        }
                        cached_fast.embed_bound(&cx, "cached retry").await.unwrap();
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(
                            session.snapshot.retrieval_configuration(),
                            fixture.snapshot.retrieval_configuration(),
                        );
                        assert_eq!(
                            session
                                .snapshot
                                .retained_snapshot()
                                .closure()
                                .bytes(GenerationComponentRole::Lexical),
                            fixture
                                .snapshot
                                .retained_snapshot()
                                .closure()
                                .bytes(GenerationComponentRole::Lexical),
                        );
                        assert_eq!(
                            session.snapshot.metadata().commit_range,
                            fixture.snapshot.metadata().commit_range
                        );
                        let partial = Arc::clone(&session.snapshot);
                        let quality_retrieval = partial.quality_retrieval_configuration();
                        let quality_identity = session.receipt.quality_identity.clone();
                        let cached_quality = Arc::clone(
                            session
                                .query_models()
                                .models()
                                .unwrap()
                                .quality
                                .as_ref()
                                .unwrap(),
                        );
                        cached_quality
                            .embed_bound(&cx, "cached quality retry")
                            .await
                            .unwrap();
                        let quality_calls = quality.calls.load(Ordering::SeqCst);
                        let request = warm_request_value(&session, &fixture, "warm-full", None);
                        let response = session
                            .complete_quality_with_loader(
                                &cx,
                                serde_json::from_value(request).unwrap(),
                                |_, _, _| {
                                    panic!("partial completion must retain its quality model")
                                },
                            )
                            .await
                            .unwrap();
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(quality.calls.load(Ordering::SeqCst), quality_calls + 1);
                        assert_eq!(response["quality_documents_added"], 1);
                        assert_eq!(response["models_loaded"], 0);
                        assert_eq!(response["quality_model_retained"], true);
                        assert_eq!(
                            response["query_modes"],
                            serde_json::json!(["fast", "quality", "full", "lexical"])
                        );
                        assert_eq!(response["default_mode"], "full");
                        assert_eq!(response["quality_completion_enabled"], false);
                        assert_eq!(session.receipt.quality_identity, quality_identity);
                        assert_eq!(
                            session.snapshot.quality_retrieval_configuration(),
                            quality_retrieval
                        );
                        assert!(Arc::ptr_eq(
                            &cached_quality,
                            session
                                .query_models()
                                .models()
                                .unwrap()
                                .quality
                                .as_ref()
                                .unwrap(),
                        ));
                        quality.fault.store(1, Ordering::SeqCst);
                        cached_quality
                            .embed_bound(&cx, "cached quality retry")
                            .await
                            .unwrap();
                        assert_eq!(quality.calls.load(Ordering::SeqCst), quality_calls + 1);
                        assert_eq!(
                            session.snapshot.topology(),
                            NativeGenerationTopology::FullProgressive
                        );
                        assert_eq!(partial.partial_quality_document_ids(&cx).unwrap().len(), 3);
                        assert_eq!(fixture.snapshot.topology(), fixture.selected.topology());
                        assert_eq!(
                            fixture
                                .snapshot
                                .source_documents(&cx, MAX_INPUT_BYTES)
                                .unwrap()
                                .len(),
                            4
                        );
                        Receipt::read(&fixture.parent.join("warm-full.json"))
                            .unwrap()
                            .open(&cx, DEFAULT_IMAGE_BYTES)
                            .unwrap();
                    }
                }
            });
        }

        #[test]
        fn warm_quality_completion_preflights_grants_membership_and_source_budget() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for starts_partial in [false, true] {
                    let quality = Provider::new(true);
                    let fixture = fixture_with_documents(
                        &cx,
                        false,
                        starts_partial.then(|| quality.clone()),
                        backfill_documents(),
                    )
                    .await;
                    let mut session =
                        warm_session(&cx, &fixture, starts_partial.then(|| quality.clone()));
                    let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                    let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                    let quality_calls = quality.calls.load(Ordering::SeqCst);
                    let mut selections = vec![
                        vec![],
                        vec!["garden", "garden"],
                        vec!["garden", "missing"],
                        vec!["garden", "notes", "retry", "uncovered"],
                    ];
                    if starts_partial {
                        selections.extend([vec!["garden"], vec!["notes", "retry"]]);
                    }
                    for (index, ids) in selections.into_iter().enumerate() {
                        let label = format!("invalid-warm-subset-{index}");
                        let request = warm_request_value(&session, &fixture, &label, Some(&ids));
                        let loaded = AtomicUsize::new(0);
                        let error = session
                            .complete_quality_with_loader(
                                &cx,
                                serde_json::from_value(request).unwrap(),
                                |_, _, _| -> Result<Arc<dyn Embedder>> {
                                    loaded.fetch_add(1, Ordering::SeqCst);
                                    Ok(quality.clone())
                                },
                            )
                            .await
                            .unwrap_err();
                        assert!(
                            error
                                .to_string()
                                .contains("quality_completion_preflight_failed")
                        );
                        assert_eq!(loaded.load(Ordering::SeqCst), 0);
                        assert!(!fixture.parent.join(format!("{label}-work")).exists());
                        assert!(!fixture.parent.join(format!("{label}.json")).exists());
                    }
                    let granted_limits = session.update_limits;
                    for refusal in [
                        "updates",
                        "model-grant",
                        "source-budget",
                        "model-options",
                        "zero-batch",
                        "excessive-batch",
                        "mixed-selection",
                    ] {
                        session.update_limits = granted_limits;
                        session.allow_model_activation = true;
                        let mut value = warm_request_value(
                            &session,
                            &fixture,
                            refusal,
                            Some(&["garden", "retry"]),
                        );
                        match refusal {
                            "updates" => session.update_limits = None,
                            "model-grant" => session.allow_model_activation = false,
                            "source-budget" => {
                                session.update_limits.as_mut().unwrap().max_source_bytes = 1;
                            }
                            "model-options" if starts_partial => {
                                value["model_dir"] =
                                    serde_json::json!(fixture.parent.join("different-models"));
                            }
                            "model-options" => {
                                value.as_object_mut().unwrap().remove("model_dir");
                            }
                            "zero-batch" | "excessive-batch" => {
                                value.as_object_mut().unwrap().remove("quality_ids");
                                value["max_new_quality_documents"] =
                                    serde_json::json!(if refusal == "zero-batch" {
                                        0
                                    } else {
                                        MAX_DOCUMENTS + 1
                                    });
                            }
                            "mixed-selection" => {
                                value["max_new_quality_documents"] = serde_json::json!(2);
                            }
                            _ => unreachable!(),
                        }
                        let error = session
                            .complete_quality_with_loader(
                                &cx,
                                serde_json::from_value(value).unwrap(),
                                |_, _, _| panic!("preflight must precede model loading"),
                            )
                            .await
                            .unwrap_err();
                        let response = session.update_failure(1, None, error.as_ref());
                        assert_eq!(response["publication"], "not_committed");
                        assert_eq!(response["serving"], "not_activated");
                        assert!(!fixture.parent.join(format!("{refusal}-work")).exists());
                        assert!(!fixture.parent.join(format!("{refusal}.json")).exists());
                    }
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                    assert!(Arc::ptr_eq(&session.snapshot, &fixture.snapshot));
                    assert_eq!(session.scope.as_ref().unwrap().len(), 1);
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert_eq!(quality.calls.load(Ordering::SeqCst), quality_calls);
                }
            });
        }

        #[test]
        fn warm_quality_completion_faults_and_stale_heads_keep_the_retained_reader() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let fixture = fixture_with_documents(&cx, false, None, backfill_documents()).await;
                let mut session = warm_session(&cx, &fixture, None);
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                let cached_fast = Arc::clone(&session.query_models().models().unwrap().fast);
                for fault in [1, 2, 3] {
                    let quality = Provider::new(true);
                    quality.fault.store(fault, Ordering::SeqCst);
                    let label = format!("warm-fault-{fault}");
                    let request = warm_request_value(&session, &fixture, &label, None);
                    let error = session
                        .complete_quality_with_loader(
                            &cx,
                            serde_json::from_value(request).unwrap(),
                            |_, _, _| -> Result<Arc<dyn Embedder>> { Ok(quality.clone()) },
                        )
                        .await
                        .unwrap_err();
                    if fault == 2 {
                        assert!(matches!(
                            error.downcast_ref::<SearchError>(),
                            Some(SearchError::Cancelled { .. })
                        ));
                    }
                    cx.set_cancel_requested(false);
                    assert!(!fixture.parent.join(format!("{label}.json")).exists());
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                    assert!(Arc::ptr_eq(&session.snapshot, &fixture.snapshot));
                    assert!(Arc::ptr_eq(
                        &cached_fast,
                        &session.query_models().models().unwrap().fast
                    ));
                    assert!(session.models.models().unwrap().quality.is_none());
                    assert_eq!(session.scope.as_ref().unwrap().len(), 1);
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                }
                // An independently published successor leaves this session's
                // retained reader valid but cannot become an implicit write base.
                let quality = Provider::new(true);
                let mut winner = options(&fixture, "warm-competing-winner");
                winner.quality_ids = Some(vec!["garden".to_owned()]);
                with_loader(&cx, &winner, &mut Vec::new(), || {
                    Ok(Models {
                        fast: fixture.fast.clone(),
                        quality: Some(quality),
                    })
                })
                .await
                .unwrap();
                let winner_authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                assert_ne!(winner_authority, authority);
                let request = warm_request_value(&session, &fixture, "stale-warm-base", None);
                let error = session
                    .complete_quality_with_loader(
                        &cx,
                        serde_json::from_value(request).unwrap(),
                        |_, _, _| panic!("stale current authority must refuse before loading"),
                    )
                    .await
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("quality_completion_base_not_current")
                );
                assert!(!fixture.parent.join("stale-warm-base-work").exists());
                assert!(!fixture.parent.join("stale-warm-base.json").exists());
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    winner_authority
                );
                assert!(Arc::ptr_eq(&session.snapshot, &fixture.snapshot));
                assert_eq!(
                    session
                        .snapshot
                        .source_documents(&cx, MAX_INPUT_BYTES)
                        .unwrap()
                        .len(),
                    4
                );
            });
        }

        #[test]
        fn warm_quality_completion_automatic_batches_resume_and_finish_with_full_admission() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for starts_partial in [false, true] {
                    let quality = Provider::new(true);
                    let fixture = fixture_with_documents(
                        &cx,
                        true,
                        starts_partial.then(|| quality.clone()),
                        backfill_documents(),
                    )
                    .await;
                    let mut session =
                        warm_session(&cx, &fixture, starts_partial.then(|| quality.clone()));
                    let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                    let cached_fast = Arc::clone(&session.query_models().models().unwrap().fast);
                    let mut cached_quality =
                        session.query_models().models().unwrap().quality.clone();
                    let loads = AtomicUsize::new(0);
                    let mut covered = usize::from(starts_partial);
                    let mut partial_reader = None;
                    for step in 0..2 {
                        let label = format!("automatic-warm-{step}");
                        let mut request = warm_request_value(&session, &fixture, &label, None);
                        request["max_new_quality_documents"] = serde_json::json!(2);
                        let response = session
                            .complete_quality_with_loader(
                                &cx,
                                serde_json::from_value(request).unwrap(),
                                |_, _, _| -> Result<Arc<dyn Embedder>> {
                                    loads.fetch_add(1, Ordering::SeqCst);
                                    Ok(quality.clone())
                                },
                            )
                            .await
                            .unwrap();
                        let next_covered = (covered + 2).min(4);
                        assert_eq!(response["quality_documents_added"], next_covered - covered);
                        assert_eq!(response["publication"], "committed");
                        assert_eq!(response["serving"], "activated");
                        assert_eq!(quality.calls.load(Ordering::SeqCst), next_covered);
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(loads.load(Ordering::SeqCst), usize::from(!starts_partial));
                        assert!(Arc::ptr_eq(
                            &cached_fast,
                            &session.query_models().models().unwrap().fast,
                        ));
                        if let Some(previous) = cached_quality.as_ref() {
                            assert!(Arc::ptr_eq(
                                previous,
                                session
                                    .query_models()
                                    .models()
                                    .unwrap()
                                    .quality
                                    .as_ref()
                                    .unwrap(),
                            ));
                        }
                        cached_quality = session.query_models().models().unwrap().quality.clone();
                        assert_eq!(response["scope"]["eligible_documents"], 1);
                        assert_eq!(
                            session.snapshot.metadata().commit_range,
                            fixture.snapshot.metadata().commit_range
                        );
                        if next_covered < 4 {
                            let expected = if starts_partial {
                                vec!["garden".to_owned(), "notes".to_owned(), "retry".to_owned()]
                            } else {
                                vec!["garden".to_owned(), "notes".to_owned()]
                            };
                            assert_eq!(
                                session.snapshot.partial_quality_document_ids(&cx).unwrap(),
                                expected
                            );
                            assert_eq!(response["default_mode"], "fast");
                            assert_eq!(response["quality_completion_enabled"], true);
                            partial_reader = Some(Arc::clone(&session.snapshot));
                        } else {
                            assert_eq!(
                                session.snapshot.topology(),
                                NativeGenerationTopology::FullProgressive
                            );
                            assert_eq!(response["default_mode"], "full");
                            assert_eq!(response["quality_completion_enabled"], false);
                            assert_eq!(session.receipt.quality_documents, None);
                        }
                        covered = next_covered;
                    }
                    assert_eq!(
                        partial_reader
                            .unwrap()
                            .partial_quality_document_ids(&cx)
                            .unwrap()
                            .len(),
                        2 + usize::from(starts_partial),
                    );
                    assert_eq!(fixture.snapshot.topology(), fixture.selected.topology());
                    Receipt::read(&fixture.parent.join("automatic-warm-1.json"))
                        .unwrap()
                        .open(&cx, DEFAULT_IMAGE_BYTES)
                        .unwrap();
                }
            });
        }

        #[test]
        fn warm_quality_completion_wire_reports_actual_coverage_and_serves_the_new_selection() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let quality = Provider::new(true);
                let fixture =
                    fixture_with_documents(&cx, true, Some(quality.clone()), backfill_documents())
                        .await;
                let mut session = warm_session(&cx, &fixture, Some(quality));
                fixture.fast.fault.store(0, Ordering::SeqCst);
                let mut input = Vec::new();
                for request in [
                    warm_request_value(
                        &session,
                        &fixture,
                        "wire-partial",
                        Some(&["garden", "notes", "retry"]),
                    ),
                    serde_json::json!({"id": "partial-query", "query": "retry", "mode": "partial"}),
                    serde_json::json!({"id": "too-soon", "query": "retry", "mode": "full"}),
                ] {
                    serde_json::to_writer(&mut input, &request).unwrap();
                    input.push(b'\n');
                }
                let mut output = Vec::new();
                super::super::super::serve_session(
                    &cx,
                    &mut session,
                    &mut std::io::Cursor::new(input),
                    &mut output,
                )
                .await
                .unwrap();
                let frames: Vec<serde_json::Value> = std::str::from_utf8(&output)
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                assert_eq!(frames[0]["quality_completion_enabled"], true);
                let partial = frames
                    .iter()
                    .find(|frame| frame["id"] == "wire-partial")
                    .unwrap();
                assert_eq!(partial["event"], "quality_completed");
                assert_eq!(partial["request"], 1);
                assert_eq!(partial["publication"], "committed");
                assert_eq!(partial["models_loaded"], 0);
                assert_eq!(
                    partial["query_modes"],
                    serde_json::json!(["fast", "partial", "lexical"])
                );
                assert_eq!(partial["default_mode"], "fast");
                let results = frames
                    .iter()
                    .find(|frame| frame["id"] == "partial-query")
                    .unwrap();
                assert_eq!(results["ok"], true);
                assert_eq!(results["mode"], "partial");
                assert_eq!(results["scope"]["eligible_documents"], 1);
                assert!(
                    results["results"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|hit| hit["doc_id"] == "retry")
                );
                assert_eq!(
                    frames
                        .iter()
                        .find(|frame| frame["id"] == "too-soon")
                        .unwrap()["ok"],
                    false
                );
                let partial_reader = Arc::clone(&session.snapshot);

                let mut stale = warm_request_value(&session, &fixture, "wire-stale", None);
                stale["expected_authority"] =
                    serde_json::json!(fixture.snapshot.authority().canonical_bytes().as_slice());
                let mut input = Vec::new();
                for request in [
                    stale,
                    warm_request_value(&session, &fixture, "wire-full", None),
                    serde_json::json!({"id": "quality-query", "query": "retry", "mode": "quality"}),
                ] {
                    serde_json::to_writer(&mut input, &request).unwrap();
                    input.push(b'\n');
                }
                let mut output = Vec::new();
                super::super::super::serve_session(
                    &cx,
                    &mut session,
                    &mut std::io::Cursor::new(input),
                    &mut output,
                )
                .await
                .unwrap();
                let frames: Vec<serde_json::Value> = std::str::from_utf8(&output)
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                let stale = frames
                    .iter()
                    .find(|frame| frame["id"] == "wire-stale")
                    .unwrap();
                assert_eq!(stale["code"], "stale_expected_authority");
                assert_eq!(stale["publication"], "not_committed");
                assert_eq!(stale["serving"], "not_activated");
                assert!(!fixture.parent.join("wire-stale-work").exists());
                let full = frames
                    .iter()
                    .find(|frame| frame["id"] == "wire-full")
                    .unwrap();
                assert_eq!(full["event"], "quality_completed");
                assert_eq!(full["publication"], "committed");
                assert_eq!(full["models_loaded"], 0);
                assert_eq!(full["default_mode"], "full");
                assert_eq!(
                    full["query_modes"],
                    serde_json::json!(["fast", "quality", "full", "lexical"])
                );
                assert_eq!(full["quality_completion_enabled"], false);
                let results = frames
                    .iter()
                    .find(|frame| frame["id"] == "quality-query")
                    .unwrap();
                assert_eq!(results["ok"], true);
                assert_eq!(results["mode"], "quality");
                assert_eq!(results["scope"]["eligible_documents"], 1);
                assert_eq!(results["results"][0]["doc_id"], "retry");
                assert_eq!(
                    partial_reader
                        .partial_quality_document_ids(&cx)
                        .unwrap()
                        .len(),
                    3
                );
                assert_eq!(
                    session.snapshot.topology(),
                    NativeGenerationTopology::FullProgressive
                );
            });
        }

        #[test]
        fn bounded_completion_publishes_truthful_partial_receipts_and_resumes_to_full() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    for starts_partial in [false, true] {
                        let quality = Provider::new(true);
                        let fixture = fixture_with_documents(
                            &cx,
                            ann,
                            starts_partial.then(|| quality.clone()),
                            backfill_documents(),
                        )
                        .await;
                        let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                        let mut bounded = options(&fixture, "bounded");
                        bounded.quality_ids =
                            Some(["garden", "notes", "retry"].map(str::to_owned).into());
                        let mut output = Vec::new();
                        with_loader(&cx, &bounded, &mut output, || {
                            Ok(Models {
                                fast: fixture.fast.clone(),
                                quality: Some(quality.clone()),
                            })
                        })
                        .await
                        .unwrap();
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(quality.calls.load(Ordering::SeqCst), 3);
                        let receipt = Receipt::read(&bounded.receipt).unwrap();
                        assert_eq!(receipt.quality_documents, Some(3));
                        assert_eq!(receipt.identity, fixture.selected.identity);
                        assert_eq!(
                            receipt.query_mode(Some(QueryMode::Partial)).unwrap(),
                            QueryMode::Partial
                        );
                        assert!(receipt.query_mode(Some(QueryMode::Full)).is_err());
                        let partial = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                        assert_eq!(
                            partial.topology(),
                            NativeGenerationTopology::PartialQuality {
                                quality_documents: 3,
                                total_documents: 4,
                                coverage_ppm: 750_000,
                            }
                        );
                        assert_eq!(
                            partial.partial_quality_document_ids(&cx).unwrap(),
                            vec!["garden".to_owned(), "notes".to_owned(), "retry".to_owned()]
                        );
                        assert_eq!(
                            partial.retained_snapshot().manifest().components().lexical,
                            fixture
                                .snapshot
                                .retained_snapshot()
                                .manifest()
                                .components()
                                .lexical,
                        );
                        assert_eq!(
                            partial.metadata().commit_range,
                            fixture.snapshot.metadata().commit_range
                        );
                        assert_eq!(
                            partial.retrieval_configuration(),
                            fixture.snapshot.retrieval_configuration()
                        );
                        let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                        assert_eq!(report["publication"], "committed");
                        assert_eq!(report["selection"]["quality_documents"], 3);

                        let mut complete = options(&fixture, "resumed-full");
                        complete.from_receipt = Some(bounded.receipt.clone());
                        with_loader(&cx, &complete, &mut Vec::new(), || {
                            Ok(Models {
                                fast: fixture.fast.clone(),
                                quality: Some(quality.clone()),
                            })
                        })
                        .await
                        .unwrap();
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(quality.calls.load(Ordering::SeqCst), 4);
                        let complete = Receipt::read(&complete.receipt).unwrap();
                        let full = complete.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                        assert_eq!(full.topology(), NativeGenerationTopology::FullProgressive);
                        assert_eq!(complete.quality_identity, receipt.quality_identity);
                        assert_eq!(complete.quality_documents, None);
                        // Retained readers remain pinned to their original coverage.
                        assert_eq!(partial.partial_quality_document_ids(&cx).unwrap().len(), 3);
                        assert_eq!(fixture.snapshot.topology(), fixture.selected.topology());
                    }
                }
            });
        }

        #[test]
        fn bounded_completion_refuses_invalid_or_regressing_membership_before_loading() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let quality = Provider::new(true);
                let fixture =
                    fixture_with_documents(&cx, false, Some(quality.clone()), backfill_documents())
                        .await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                let selections = [
                    vec![],
                    vec!["garden", "garden"],
                    vec!["garden", "unknown"],
                    vec!["garden", "notes", "retry", "uncovered"],
                    vec!["garden"],
                    vec!["notes", "retry"],
                ];
                for (index, selection) in selections.into_iter().enumerate() {
                    let mut options = options(&fixture, &format!("invalid-backfill-{index}"));
                    options.quality_ids = Some(selection.into_iter().map(str::to_owned).collect());
                    let loaded = AtomicUsize::new(0);
                    let mut output = Vec::new();
                    assert!(
                        with_loader(&cx, &options, &mut output, || {
                            loaded.fetch_add(1, Ordering::SeqCst);
                            Ok(Models {
                                fast: fixture.fast.clone(),
                                quality: Some(quality.clone()),
                            })
                        })
                        .await
                        .is_err()
                    );
                    assert_eq!(loaded.load(Ordering::SeqCst), 0);
                    assert!(output.is_empty());
                    assert!(!options.work.as_ref().unwrap().exists());
                    assert!(!options.receipt.exists());
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                }
                assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            });
        }

        #[test]
        fn bounded_completion_publication_authenticates_membership_instead_of_only_counts() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let seed = fixture_with_documents(&cx, false, None, backfill_documents()).await;
                let quality = Provider::new(true);
                // Derive both sides from one admitted fast source. Rebuilding
                // Quill independently would introduce wall-clock manifest bytes
                // and could fail source equality before the membership guard.
                let mut candidates = Vec::new();
                for (label, ids) in [
                    ("same-count", vec!["retry"]),
                    ("dropped-old", vec!["notes", "retry"]),
                ] {
                    let built = seed
                        .snapshot
                        .begin_quality_completion(
                            &cx,
                            seed.parent.join(format!("substituted-{label}")),
                            new_generation(3).unwrap(),
                            seed.fast.clone(),
                            quality.clone(),
                            MAX_INPUT_BYTES,
                        )
                        .unwrap()
                        .build_partial(&cx, ids.into_iter().map(str::to_owned), DEFAULT_IMAGE_BYTES)
                        .await
                        .unwrap();
                    candidates.push((label, built));
                }
                let first = seed
                    .snapshot
                    .begin_quality_completion(
                        &cx,
                        seed.parent.join("selected-partial-work"),
                        new_generation(2).unwrap(),
                        seed.fast.clone(),
                        quality,
                        MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .build_partial(&cx, ["garden".to_owned()], DEFAULT_IMAGE_BYTES)
                    .await
                    .unwrap();
                let prepared = prepare_partial_quality_completion(
                    &cx,
                    &first,
                    &seed.snapshot,
                    DEFAULT_IMAGE_BYTES,
                )
                .unwrap();
                let root = admit_root(&seed.selected.root).unwrap();
                let selected = stage_prepared_and_publish(
                    &cx,
                    &root,
                    &seed.selected.root,
                    &seed.parent.join("selected-partial.json"),
                    &first,
                    prepared,
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )
                .unwrap();
                let authority = fs::read(seed.selected.root.join("AUTHORITY")).unwrap();
                for (label, built) in candidates {
                    // A fully valid candidate with matching source and producers
                    // must still prove it retained the actual old quality member.
                    assert_eq!(
                        built.component_bytes(GenerationComponentRole::Lexical),
                        selected
                            .snapshot
                            .retained_snapshot()
                            .closure()
                            .bytes(GenerationComponentRole::Lexical),
                    );
                    prepare_publication_with_intent(
                        &built,
                        Some(&selected.snapshot),
                        DEFAULT_IMAGE_BYTES,
                        PublicationIntent::BackfillQuality,
                    )
                    .unwrap();
                    let error = prepare_partial_quality_completion(
                        &cx,
                        &built,
                        &selected.snapshot,
                        DEFAULT_IMAGE_BYTES,
                    )
                    .err()
                    .expect("substituted quality coverage must refuse publication");
                    assert!(
                        error
                            .to_string()
                            .contains("quality_backfill_coverage_not_expanded"),
                        "{label}: {error}",
                    );
                    assert_eq!(
                        fs::read(seed.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                    selected.receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                }
            });
        }

        #[test]
        fn bounded_completion_faults_leave_the_exact_predecessor_selected() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let old_quality = Provider::new(true);
                let fixture =
                    fixture_with_documents(&cx, true, Some(old_quality), backfill_documents())
                        .await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                for fault in [1, 2, 3] {
                    let quality = Provider::new(true);
                    quality.fault.store(fault, Ordering::SeqCst);
                    let mut options = options(&fixture, &format!("bounded-fault-{fault}"));
                    options.quality_ids = Some(["garden", "retry"].map(str::to_owned).into());
                    let mut output = Vec::new();
                    let result = with_loader(&cx, &options, &mut output, || {
                        Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(quality.clone()),
                        })
                    })
                    .await;
                    cx.set_cancel_requested(false);
                    assert!(result.is_err());
                    assert!(quality.calls.load(Ordering::SeqCst) > 0);
                    assert!(output.is_empty());
                    assert!(!options.receipt.exists());
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                    let retained = fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(
                        retained.partial_quality_document_ids(&cx).unwrap(),
                        vec!["garden".to_owned()]
                    );
                }
            });
        }

        #[test]
        fn completion_reuses_fast_inference_and_quill_then_reopens_full_progressive_authority() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    let fixture = fixture(&cx, ann).await;
                    let quality = Provider::new(true);
                    let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                    let mut options = options(&fixture, "complete");
                    options.retrieval = Some(if ann {
                        NativeBuildRetrieval::Exact
                    } else {
                        NativeBuildRetrieval::Hnsw {
                            params: HnswParams::default(),
                            seed: 29,
                        }
                    });
                    let mut output = Vec::new();
                    with_loader(&cx, &options, &mut output, || {
                        Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(quality.clone()),
                        })
                    })
                    .await
                    .unwrap();
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert_eq!(quality.calls.load(Ordering::SeqCst), 2);
                    let receipt = Receipt::read(&options.receipt).unwrap();
                    let reopened = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(
                        reopened.topology(),
                        NativeGenerationTopology::FullProgressive
                    );
                    assert_eq!(receipt.identity, fixture.selected.identity);
                    assert_eq!(receipt.quality_binding().unwrap().unwrap().dimension(), 3);
                    assert_eq!(
                        reopened.metadata().commit_range,
                        fixture.snapshot.metadata().commit_range
                    );
                    assert_eq!(
                        reopened.retrieval_configuration(),
                        fixture.snapshot.retrieval_configuration()
                    );
                    assert_eq!(
                        reopened.quality_retrieval_configuration(),
                        options.retrieval
                    );
                    assert_eq!(
                        reopened
                            .retained_snapshot()
                            .closure()
                            .bytes(GenerationComponentRole::Lexical),
                        fixture
                            .snapshot
                            .retained_snapshot()
                            .closure()
                            .bytes(GenerationComponentRole::Lexical),
                    );
                    assert_eq!(
                        serde_json::to_value(
                            reopened.source_documents(&cx, MAX_INPUT_BYTES).unwrap()
                        )
                        .unwrap(),
                        serde_json::to_value(
                            fixture
                                .snapshot
                                .source_documents(&cx, MAX_INPUT_BYTES)
                                .unwrap()
                        )
                        .unwrap(),
                    );
                    fixture.fast.fault.store(0, Ordering::SeqCst);
                    let results = reopened
                        .search_refined_text(
                            &cx,
                            fixture.fast.as_ref(),
                            quality.as_ref(),
                            "retry",
                            2,
                        )
                        .await
                        .unwrap();
                    assert!(results.iter().any(|result| result.quality_score.is_some()));
                    assert_eq!(
                        fixture.snapshot.topology(),
                        NativeGenerationTopology::FastOnly
                    );
                    assert!(
                        !fixture
                            .snapshot
                            .search_text(&cx, fixture.fast.as_ref(), "retry", 1)
                            .await
                            .unwrap()
                            .is_empty()
                    );
                    assert!(fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
                    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                    assert_eq!(report["publication"], "committed");
                }
            });
        }

        #[test]
        fn partial_completion_reuses_covered_rows_and_inherits_quality_storage_and_retrieval() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    let quality = Provider::new(true);
                    let fixture = fixture_with_quality(&cx, ann, Some(quality.clone())).await;
                    let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                    assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                    let options = options(&fixture, "complete-partial");
                    let mut output = Vec::new();
                    with_loader(&cx, &options, &mut output, || {
                        Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(quality.clone()),
                        })
                    })
                    .await
                    .unwrap();
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert_eq!(quality.calls.load(Ordering::SeqCst), 2);
                    let receipt = Receipt::read(&options.receipt).unwrap();
                    assert_eq!(
                        receipt.topology(),
                        NativeGenerationTopology::FullProgressive
                    );
                    assert_eq!(receipt.identity, fixture.selected.identity);
                    assert_eq!(receipt.quality_identity, fixture.selected.quality_identity);
                    assert_eq!(receipt.quality_documents, None);
                    let reopened = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(
                        reopened.quality_retrieval_configuration(),
                        fixture.snapshot.quality_retrieval_configuration(),
                    );
                    assert_eq!(
                        reopened.retained_snapshot().manifest().components().lexical,
                        fixture
                            .snapshot
                            .retained_snapshot()
                            .manifest()
                            .components()
                            .lexical,
                    );
                    assert_eq!(
                        reopened.metadata().commit_range,
                        fixture.snapshot.metadata().commit_range,
                    );
                    // Completion makes a previously uncovered document eligible
                    // for direct quality retrieval after an independent reopen.
                    let hits = reopened
                        .search_quality_text(&cx, quality.as_ref(), "retry", 1)
                        .await
                        .unwrap();
                    assert_eq!(hits[0].doc_id, "retry");
                    assert!(matches!(
                        fixture.snapshot.topology(),
                        NativeGenerationTopology::PartialQuality {
                            quality_documents: 1,
                            total_documents: 2,
                            ..
                        }
                    ));
                    fixture.fast.fault.store(0, Ordering::SeqCst);
                    assert!(
                        !fixture
                            .snapshot
                            .search_text(&cx, fixture.fast.as_ref(), "retry", 1)
                            .await
                            .unwrap()
                            .is_empty()
                    );
                    assert!(fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
                    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                    assert_eq!(report["publication"], "committed");
                    assert!(report["selection"]["quality_identity"].is_object());
                    assert!(report["selection"]["quality_documents"].is_null());
                }
            });
        }

        #[test]
        fn partial_completion_refuses_foreign_quality_before_inference_and_preserves_selection() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let quality = Provider::new(true);
                let fixture = fixture_with_quality(&cx, false, Some(quality.clone())).await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                let foreign = Provider::new(true);
                foreign.drifted.store(true, Ordering::SeqCst);
                let options = options(&fixture, "foreign-quality");
                assert!(
                    with_loader(&cx, &options, &mut Vec::new(), || Ok(Models {
                        fast: fixture.fast.clone(),
                        quality: Some(foreign.clone()),
                    }))
                    .await
                    .is_err()
                );
                assert_eq!(foreign.calls.load(Ordering::SeqCst), 0);
                assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                assert!(!options.work.unwrap().exists());
                assert!(!options.receipt.exists());
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    authority,
                );
                fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            });
        }

        #[test]
        fn partial_completion_inference_failure_cancellation_and_drift_preserve_coverage() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let original_quality = Provider::new(true);
                let fixture =
                    fixture_with_quality(&cx, false, Some(original_quality.clone())).await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                for fault in [1, 2, 3] {
                    let quality = Provider::new(true);
                    quality.fault.store(fault, Ordering::SeqCst);
                    let options = options(&fixture, &format!("partial-fault-{fault}"));
                    let mut output = Vec::new();
                    let result = with_loader(&cx, &options, &mut output, || {
                        Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(quality.clone()),
                        })
                    })
                    .await;
                    cx.set_cancel_requested(false);
                    assert!(result.is_err());
                    assert!(output.is_empty());
                    assert!(!options.receipt.exists());
                    assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert_eq!(original_quality.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority,
                    );
                    let reopened = fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(reopened.topology(), fixture.snapshot.topology());
                    assert!(matches!(
                        reopened.topology(),
                        NativeGenerationTopology::PartialQuality {
                            quality_documents: 1,
                            total_documents: 2,
                            ..
                        }
                    ));
                }
            });
        }

        #[test]
        fn partial_completion_competing_winner_fences_stale_candidate_and_requires_explicit_intent()
        {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let quality = Provider::new(true);
                let fixture = fixture_with_quality(&cx, true, Some(quality.clone())).await;
                let completed = fixture
                    .snapshot
                    .begin_quality_completion(
                        &cx,
                        fixture.parent.join("stale-work"),
                        new_generation(2).unwrap(),
                        fixture.fast.clone(),
                        quality.clone(),
                        MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .build(&cx, DEFAULT_IMAGE_BYTES)
                    .await
                    .unwrap();
                assert!(
                    prepare_publication(&completed, Some(&fixture.snapshot), DEFAULT_IMAGE_BYTES)
                        .is_err()
                );
                let prepared = prepare_publication_with_intent(
                    &completed,
                    Some(&fixture.snapshot),
                    DEFAULT_IMAGE_BYTES,
                    PublicationIntent::CompleteQuality,
                )
                .unwrap();
                let winner_options = options(&fixture, "winner-completion");
                with_loader(&cx, &winner_options, &mut Vec::new(), || {
                    Ok(Models {
                        fast: fixture.fast.clone(),
                        quality: Some(quality.clone()),
                    })
                })
                .await
                .unwrap();
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let root = admit_root(&fixture.selected.root).unwrap();
                assert!(
                    stage_prepared_and_publish(
                        &cx,
                        &root,
                        &fixture.selected.root,
                        &fixture.parent.join("stale-completion.json"),
                        &completed,
                        prepared,
                        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                    )
                    .is_err()
                );
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    authority,
                );
                let receipt = Receipt::read(&winner_options.receipt).unwrap();
                let reopened = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    reopened.topology(),
                    NativeGenerationTopology::FullProgressive
                );
            });
        }

        #[test]
        fn completion_failure_cancellation_and_quality_identity_drift_preserve_incumbent() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let fixture = fixture(&cx, false).await;
                let before = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                for fault in [1, 2, 3] {
                    let quality = Provider::new(true);
                    quality.fault.store(fault, Ordering::SeqCst);
                    let options = options(&fixture, &format!("refused-{fault}"));
                    assert!(
                        with_loader(&cx, &options, &mut Vec::new(), || Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(quality.clone()),
                        }))
                        .await
                        .is_err()
                    );
                    cx.set_cancel_requested(false);
                    assert!(!options.receipt.exists());
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        before
                    );
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                }
                let wrong_fast = Provider::new(false);
                wrong_fast.drifted.store(true, Ordering::SeqCst);
                let options = options(&fixture, "wrong-fast");
                let quality = Provider::new(true);
                assert!(
                    with_loader(&cx, &options, &mut Vec::new(), || Ok(Models {
                        fast: wrong_fast.clone(),
                        quality: Some(quality.clone()),
                    }))
                    .await
                    .is_err()
                );
                assert_eq!(wrong_fast.calls.load(Ordering::SeqCst), 0);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
                assert!(!options.work.unwrap().exists());
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    before
                );
            });
        }

        #[test]
        fn competing_publication_refuses_a_completed_quality_candidate_without_rebasing() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let fixture = fixture(&cx, false).await;
                let quality = Provider::new(true);
                let completed = fixture
                    .snapshot
                    .begin_quality_completion(
                        &cx,
                        fixture.parent.join("completion-work"),
                        new_generation(2).unwrap(),
                        fixture.fast.clone(),
                        quality,
                        MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .build(&cx, DEFAULT_IMAGE_BYTES)
                    .await
                    .unwrap();
                assert!(
                    prepare_publication(&completed, Some(&fixture.snapshot), DEFAULT_IMAGE_BYTES)
                        .is_err()
                );
                let prepared = prepare_publication_with_intent(
                    &completed,
                    Some(&fixture.snapshot),
                    DEFAULT_IMAGE_BYTES,
                    PublicationIntent::CompleteQuality,
                )
                .unwrap();
                let winner = fixture
                    .snapshot
                    .begin_update(
                        &cx,
                        fixture.parent.join("winner-work"),
                        new_generation(2).unwrap(),
                        fixture.fast.clone(),
                        MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .build_exact_generation(
                        &cx,
                        EmbedderTierTag::Fast,
                        fixture.snapshot.metadata().commit_range.clone(),
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap();
                let root = admit_root(&fixture.selected.root).unwrap();
                let published = stage_and_publish(
                    &cx,
                    &root,
                    &fixture.selected.root,
                    &fixture.parent.join("winner.json"),
                    &winner,
                    Some(&fixture.snapshot),
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )
                .unwrap();
                let winner_authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                assert!(
                    stage_prepared_and_publish(
                        &cx,
                        &root,
                        &fixture.selected.root,
                        &fixture.parent.join("stale.json"),
                        &completed,
                        prepared,
                        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                    )
                    .is_err()
                );
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    winner_authority
                );
                published.receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    published.snapshot.topology(),
                    NativeGenerationTopology::FastOnly
                );
                assert_eq!(
                    fixture.snapshot.authority(),
                    fixture.selected.authority().unwrap()
                );
            });
        }
    }
}
