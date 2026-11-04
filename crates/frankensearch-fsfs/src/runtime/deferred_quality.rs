//! Complete deferred quality work from the exact input retained with a selected
//! generation.
//!
//! Queue claims and document revisions use the storage queue; the
//! ordinary complete-generation writer owns the publication boundary.
//!
//! A failed or cancelled attempt leaves the selected queue pending. Its private
//! candidate can contain partial work, but no partial quality tier is selected.
//! A later process starts again from the immutable pending generation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use asupersync::Cx;
use frankensearch_core::generation::QuantizationFormat;
use frankensearch_core::{EmbeddingIdentityBundleV1, SearchError, SearchResult};
use frankensearch_embed::RegisteredQualityModel;
use frankensearch_storage::{
    BuildTrigger, ClaimOutcome, ContentHasher, DocumentRecord, JobQueueConfig, PersistentJobQueue,
    RecordBuildParams, Storage, StorageConfig,
};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};

use super::{
    AdmittedEmbedder, CheckpointFileEntry, FSFS_VECTOR_QUALITY_INDEX_FILE, FsfsIndexPayload,
    FsfsRuntime, IndexManifestEntry, IndexRunControl, IndexSentinel, IngestionClass,
    PendingIndexDocument, REASON_VECTOR_QUALITY_TIER_BUILT,
    REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE, SearchExecutionMode, VectorIndex,
    content_sha256_hex, ingestion_class_label, pressure_timestamp_ms, protect_vector_generations,
    retained_reuse, retained_search_checkpoint, validate_retained_catalog_path,
};
use crate::generation_store::{
    CompleteGenerationStore, GenerationPublication, PublishedGeneration,
};
use crate::lifecycle::PublicationLease;

const BACKLOG_NAME: &str = "fsfs.deferred_quality.v1";
const SOURCE_METADATA: &str = "fsfs_deferred_quality_v1";
const MAX_INPUT_BYTES: usize = 8 * 1024;
const JOB_BATCH_SIZE: usize = 32;

/// One atomic quality completion. The generation publication retains its
/// distinct visible-but-durability-uncertain outcome.
#[derive(Debug)]
#[must_use]
pub struct DeferredQualityCompletion {
    /// Selection outcome, including an already-visible durability uncertainty.
    pub publication: GenerationPublication,
    /// Every requested semantic source in this completed quality tier.
    pub completed_documents: usize,
    /// Unchanged membership checkpoint of the deferred source generation.
    pub source_hash: String,
    /// Complete producer identity admitted for every returned embedding.
    pub producer_fingerprint: String,
}

/// A watch restart can rebuild unavailable work; explicit completion must
/// retain the specific reason that prevented resuming the selected backlog.
#[derive(Debug)]
pub(super) enum DeferredQualityAttempt {
    Completed(DeferredQualityCompletion),
    MissingBacklog,
    ProducerChanged,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Backlog {
    // 1 binds original source checkpoints; 2 binds retained embedding text.
    // A version-2 fence never grants filesystem discovery or reuse authority.
    schema_version: u16,
    generation_root: String,
    source_hash: String,
    producer: EmbeddingIdentityBundleV1,
    requested_documents: usize,
    queued_documents: usize,
    unavailable_documents: usize,
}

impl Backlog {
    fn queue_key(&self) -> String {
        format!(
            "fsfs.quality.{}.{}.{}",
            content_sha256_hex(self.generation_root.as_bytes()),
            self.source_hash,
            self.producer.fingerprint(),
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceInput {
    revision: i64,
    // Source bytes for backlog v1; canonical embedding bytes for backlog v2.
    source_sha256: String,
}

fn invalid(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "index.complete_quality".to_owned(),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

fn storage_error<E: std::error::Error + Send + Sync + 'static>(source: E) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fsfs.deferred_quality",
        source: Box::new(source),
    }
}

fn declared_producer(runtime: &FsfsRuntime) -> SearchResult<Option<EmbeddingIdentityBundleV1>> {
    let Some(selection) = RegisteredQualityModel::from_name(&runtime.config.indexing.quality_model)
    else {
        // Deferral has historically required no quality installation or even
        // a registered selection. Keep fast/lexical indexing available; an
        // unregistered request simply cannot acquire a resumable producer fence.
        return Ok(None);
    };
    #[cfg(test)]
    if let Some(embedder) = super::test_quality_embedder_override() {
        return embedder.identity().cloned().map(Some);
    }
    selection
        .artifact_manifest()?
        .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
        .map(Some)
}

fn captures_quality(runtime: &FsfsRuntime) -> bool {
    // Only the retained builder owns this exclusion root. Legacy watch uses
    // an unrestricted queue runner in its mutable catalog; do not put jobs
    // requiring generation-scoped completion into that runner's work stream.
    runtime.discovery_excluded_root.is_some()
        && runtime.quality_work_deferred()
        && !runtime.lexical_only_indexing
        && !runtime.config.indexing.quality_model.trim().is_empty()
        && RegisteredQualityModel::from_name(&runtime.config.indexing.quality_model).is_some()
}

fn open_catalog(runtime: &FsfsRuntime, create: bool) -> SearchResult<Storage> {
    let path = runtime.resolve_storage_db_path()?;
    if create {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    } else if !path.try_exists()? {
        return Err(invalid(
            "the selected generation has no exact deferred input catalog; run a full source reindex before completing quality",
        ));
    }
    Storage::open(StorageConfig {
        db_path: path,
        ..StorageConfig::default()
    })
}

/// Called while the ordinary indexer owns its publication lease and the exact
/// bounded embedding input. Never reconstruct this from lexical text: lexical
/// canonicalization intentionally retains code that embedding input collapses.
pub(super) fn capture_index_batch(
    cx: &Cx,
    runtime: &FsfsRuntime,
    documents: &[PendingIndexDocument],
) -> SearchResult<()> {
    if !captures_quality(runtime)
        || !documents
            .iter()
            .any(|document| document.ingestion_class == IngestionClass::FullSemanticLexical)
    {
        return Ok(());
    }
    retained_search_checkpoint(cx)?;
    let storage = open_catalog(runtime, true)?;
    for pending in documents {
        retained_search_checkpoint(cx)?;
        if pending.ingestion_class != IngestionClass::FullSemanticLexical {
            continue;
        }
        let text = &pending.document.content;
        if text.is_empty() || text.len() > MAX_INPUT_BYTES {
            return Err(invalid(
                "deferred quality input exceeds its canonical text bound",
            ));
        }
        let created_at = storage
            .get_document(&pending.file_key)?
            .map_or(pending.revision, |document| document.created_at);
        let mut document = DocumentRecord::new(
            &pending.file_key,
            text.chars().take(400).collect::<String>(),
            ContentHasher::hash(text),
            text.chars().count(),
            created_at,
            pending.revision.max(created_at),
        );
        let mut metadata =
            serde_json::to_value(&pending.document.metadata).map_err(storage_error)?;
        metadata[SOURCE_METADATA] = serde_json::to_value(SourceInput {
            revision: pending.revision,
            source_sha256: pending.content_hash_hex.clone(),
        })
        .map_err(storage_error)?;
        document.metadata = Some(metadata);
        // The retained text is authoritative. No mutable source-path fallback
        // may be used by a worker after the generation is selected.
        document.source_path = None;
        storage.upsert_document(&document)?;
        storage.set_document_content(&pending.file_key, text)?;
    }
    retained_search_checkpoint(cx)
}

fn source_input(document: &DocumentRecord) -> SearchResult<Option<SourceInput>> {
    document
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get(SOURCE_METADATA))
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(storage_error)
}

fn record_backlog(storage: &Storage, backlog: &Backlog, completed: usize) -> SearchResult<()> {
    storage.record_index_build(&RecordBuildParams {
        index_name: BACKLOG_NAME.to_owned(),
        index_type: "deferred_quality".to_owned(),
        embedder_id: backlog.queue_key(),
        embedder_revision: Some(backlog.producer.fingerprint()),
        dimension: i64::from(backlog.producer.space.dimension),
        record_count: i64::try_from(completed).map_err(|_| invalid("backlog count overflow"))?,
        source_doc_count: i64::try_from(backlog.requested_documents)
            .map_err(|_| invalid("backlog count overflow"))?,
        build_duration_ms: 0,
        trigger: BuildTrigger::Manual,
        file_path: Some(FSFS_VECTOR_QUALITY_INDEX_FILE.to_owned()),
        file_size_bytes: None,
        file_hash: Some(backlog.source_hash.clone()),
        schema_version: Some(i64::from(backlog.schema_version)),
        config_json: Some(serde_json::to_string(backlog).map_err(storage_error)?),
        fec_path: None,
        fec_size_bytes: None,
        notes: None,
        mean_norm: None,
        variance: None,
    })
}

/// Seal pending work into the same catalog as its complete canonical input.
/// Older carried generations can lack exact input. Keep their existing index
/// behavior and explicitly record that completion needs one full source reindex.
#[allow(clippy::arc_with_non_send_sync)]
pub(super) fn finish_index(
    cx: &Cx,
    runtime: &FsfsRuntime,
    payload: &FsfsIndexPayload,
) -> SearchResult<()> {
    if !captures_quality(runtime) {
        return Ok(());
    }
    // An initially empty corpus still declares its deferred producer. Without
    // this zero-count fence, a later append cannot retain completable quality
    // inputs, even though the selected generation advertises deferral.
    retained_search_checkpoint(cx)?;
    let Some(producer) = declared_producer(runtime)? else {
        return Ok(());
    };
    producer.validate()?;
    let storage = Arc::new(open_catalog(runtime, true)?);
    let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
    let mut backlog = Backlog {
        schema_version: 1,
        generation_root: payload.generation.index_root.clone(),
        source_hash: payload.generation.source_hash_hex.clone(),
        producer,
        requested_documents: payload.quality_deferred_files,
        queued_documents: 0,
        unavailable_documents: 0,
    };
    let key = backlog.queue_key();
    for (id, entry) in &payload.input_checkpoint.files {
        retained_search_checkpoint(cx)?;
        if entry.ingestion_class != ingestion_class_label(IngestionClass::FullSemanticLexical) {
            continue;
        }
        let retained = match storage.get_document(id)? {
            Some(document) => match source_input(&document)? {
                Some(source)
                    if (source.revision, source.source_sha256.as_str())
                        == (entry.revision, entry.content_hash_hex.as_str()) =>
                {
                    storage.get_document_content(id)?.filter(|text| {
                        text.len() <= MAX_INPUT_BYTES
                            && u64::try_from(text.len()).ok() == Some(entry.canonical_bytes)
                            && !text.trim().is_empty()
                    })
                }
                _ => None,
            },
            None => None,
        };
        let Some(text) = retained else {
            backlog.unavailable_documents += 1;
            continue;
        };
        queue.enqueue(id, &key, &ContentHasher::hash(&text), 0)?;
        backlog.queued_documents += 1;
    }
    if backlog.queued_documents + backlog.unavailable_documents != backlog.requested_documents {
        return Err(invalid(
            "deferred quality coverage disagrees with source membership",
        ));
    }
    record_backlog(&storage, &backlog, 0)?;
    tracing::info!(
        queued = backlog.queued_documents,
        unavailable = backlog.unavailable_documents,
        source_hash = backlog.source_hash,
        producer = backlog.producer.fingerprint(),
        "retained exact deferred quality work in the generation catalog"
    );
    retained_search_checkpoint(cx)
}

fn read_backlog(storage: &Storage) -> SearchResult<Backlog> {
    read_optional_backlog(storage)?.ok_or_else(|| {
        invalid("this generation has no producer-bound quality backlog; select a registered quality producer and run a full source reindex")
    })
}

fn read_optional_backlog(storage: &Storage) -> SearchResult<Option<Backlog>> {
    let Some(metadata) = storage.get_index_metadata(BACKLOG_NAME)? else {
        return Ok(None);
    };
    let backlog: Backlog =
        serde_json::from_str(metadata.config_json.as_deref().ok_or_else(|| {
            invalid("the deferred quality catalog has no source and producer descriptor")
        })?)
        .map_err(storage_error)?;
    backlog.producer.validate()?;
    if !matches!(backlog.schema_version, 1 | 2)
        || backlog
            .queued_documents
            .checked_add(backlog.unavailable_documents)
            != Some(backlog.requested_documents)
        || metadata.embedder_id != backlog.queue_key()
        || metadata.embedder_revision.as_deref() != Some(backlog.producer.fingerprint().as_str())
        || metadata.file_hash.as_deref() != Some(backlog.source_hash.as_str())
        || metadata.index_type != "deferred_quality"
        || metadata.schema_version != Some(i64::from(backlog.schema_version))
        || metadata.dimension != i64::from(backlog.producer.space.dimension)
        || usize::try_from(metadata.source_doc_count).ok() != Some(backlog.requested_documents)
        || metadata.record_count != 0
    {
        return Err(invalid(
            "deferred quality catalog descriptor is inconsistent",
        ));
    }
    Ok(Some(backlog))
}

fn validate_pending_jobs(
    storage: &Storage,
    key: &str,
    manifests: &BTreeMap<String, IndexManifestEntry>,
    checkpoints: Option<&BTreeMap<String, CheckpointFileEntry>>,
) -> SearchResult<()> {
    let expected = manifests
        .iter()
        .filter(|(_, entry)| {
            entry.ingestion_class == ingestion_class_label(IngestionClass::FullSemanticLexical)
        })
        .map(|(id, _)| id.as_str())
        .collect::<BTreeSet<_>>();
    let rows = storage
        .connection()
        .query_with_params_sync(
            "SELECT doc_id, content_hash, status FROM embedding_jobs WHERE embedder_id = ?1;",
            &[SqliteValue::Text(key.to_owned().into())],
        )
        .map_err(storage_error)?;
    if rows.len() != expected.len() {
        return Err(invalid(
            "deferred queue does not cover exactly the selected semantic membership",
        ));
    }
    let mut seen = BTreeSet::new();
    for row in &rows {
        let (
            Some(SqliteValue::Text(id)),
            Some(SqliteValue::Blob(hash)),
            Some(SqliteValue::Text(status)),
        ) = (row.get(0), row.get(1), row.get(2))
        else {
            return Err(invalid("invalid deferred queue row"));
        };
        let id = id.as_str();
        if !expected.contains(id) || !seen.insert(id) || status.as_str() != "pending" {
            return Err(invalid(
                "deferred queue has missing, duplicate, or non-pending work",
            ));
        }
        let document = storage
            .get_document(id)?
            .ok_or_else(|| invalid("deferred document is absent"))?;
        let source = source_input(&document)?
            .ok_or_else(|| invalid("deferred input source fence is absent"))?;
        let text = storage.get_document_content(id)?.ok_or_else(|| {
            invalid("exact deferred input is absent; a preview cannot replace it")
        })?;
        if hash.as_ref() != document.content_hash.as_slice()
            || document.content_hash != ContentHasher::hash(&text)
            || source.revision != manifests[id].revision
            || checkpoints.map_or_else(
                || source.source_sha256 != content_sha256_hex(text.as_bytes()),
                |checkpoints| {
                    checkpoints.get(id).is_none_or(|entry| {
                        (source.revision, source.source_sha256.as_str())
                            != (entry.revision, entry.content_hash_hex.as_str())
                    })
                },
            )
            || source.source_sha256.len() != 64
            || !source
                .source_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || text.len() > MAX_INPUT_BYTES
            || u64::try_from(text.len()).ok() != Some(manifests[id].canonical_bytes)
            || text.trim().is_empty()
        {
            return Err(invalid(
                "deferred job input differs from its retained source revision",
            ));
        }
    }
    Ok(())
}

/// An empty queue owes no inference, but its exact generation-bound descriptor
/// still records the requested producer. Carry that intent without inventing
/// inputs or filesystem reuse authority. Completed and intentional fast-only
/// generations must not acquire a quality tier from stale catalog metadata.
fn open_carried_backlog(
    runtime: &FsfsRuntime,
    predecessor: &PublishedGeneration,
    previous: &IndexSentinel,
) -> SearchResult<Option<(Storage, Backlog)>> {
    let deferred = previous
        .reason_codes
        .iter()
        .any(|reason| reason == REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE);
    if (!deferred
        && predecessor
            .path()
            .join(FSFS_VECTOR_QUALITY_INDEX_FILE)
            .try_exists()?)
        || !runtime.resolve_storage_db_path()?.try_exists()?
    {
        return Ok(None);
    }
    let storage = open_catalog(runtime, false)?;
    let Some(metadata) = storage.get_index_metadata(BACKLOG_NAME)? else {
        return Ok(None);
    };
    if !deferred
        && (metadata.record_count != 0
            || metadata.source_doc_count != 0
            || metadata.file_hash.as_deref() != Some(previous.source_hash_hex.as_str()))
    {
        return Ok(None);
    }
    let backlog = read_backlog(&storage)?;
    if !deferred
        && (backlog.requested_documents != 0
            || backlog.generation_root != predecessor.path().display().to_string()
            || backlog.source_hash != previous.source_hash_hex)
    {
        return Ok(None);
    }
    Ok(Some((storage, backlog)))
}

/// Move an admitted source backlog to an owned-text fence in the independently
/// copied candidate. Version 2 certifies embedding input, never filesystem bytes
/// or upgrade reuse. Validate every old job before replacing its source fence.
pub(super) fn begin_content_upserts(
    cx: &Cx,
    candidate: &FsfsRuntime,
    predecessor: &PublishedGeneration,
) -> SearchResult<bool> {
    retained_search_checkpoint(cx)?;
    let previous = FsfsRuntime::read_index_sentinel(predecessor.path())?
        .ok_or_else(|| invalid("batch predecessor has no completion sentinel"))?;
    let Some((storage, mut backlog)) = open_carried_backlog(candidate, predecessor, &previous)?
    else {
        return Ok(false);
    };
    if backlog.generation_root != predecessor.path().display().to_string()
        || backlog.source_hash != previous.source_hash_hex
        || backlog.unavailable_documents != 0
    {
        return Err(invalid(
            "content upserts require exact deferred inputs; run `fsfs index --complete-quality` or rebuild from sources first",
        ));
    }
    let manifests = FsfsRuntime::read_matching_manifest_generation(predecessor.path())?
        .ok_or_else(|| invalid("batch predecessor membership manifests disagree"))?;
    if manifests
        .values()
        .filter(|entry| {
            entry.ingestion_class == ingestion_class_label(IngestionClass::FullSemanticLexical)
        })
        .count()
        != backlog.requested_documents
    {
        return Err(invalid(
            "deferred quality coverage disagrees with selected membership",
        ));
    }
    let checkpoint = if backlog.schema_version == 1 && backlog.requested_documents != 0 {
        Some(retained_reuse::completed_input_checkpoint(cx, predecessor)?)
    } else {
        None
    };
    validate_pending_jobs(
        &storage,
        &backlog.queue_key(),
        &manifests,
        checkpoint.as_ref().map(|checkpoint| &checkpoint.files),
    )?;
    for (id, entry) in &manifests {
        retained_search_checkpoint(cx)?;
        if entry.ingestion_class != ingestion_class_label(IngestionClass::FullSemanticLexical) {
            continue;
        }
        let text = storage
            .get_document_content(id)?
            .ok_or_else(|| invalid("admitted deferred input disappeared"))?;
        capture_content_upsert(&storage, id, &text, entry.revision)?;
    }
    backlog.schema_version = 2;
    record_backlog(&storage, &backlog, 0)?;
    retained_search_checkpoint(cx)?;
    Ok(true)
}

/// The caller supplies its exact canonical embedding input, not a lexical
/// preview or an inferred source path. Only an unpublished candidate is opened.
pub(super) fn capture_content_upsert(
    storage: &Storage,
    id: &str,
    text: &str,
    revision: i64,
) -> SearchResult<()> {
    if text.trim().is_empty() || text.len() > MAX_INPUT_BYTES {
        return Err(invalid(
            "retained quality input exceeds its canonical text bound",
        ));
    }
    let mut document = storage
        .get_document(id)?
        .ok_or_else(|| invalid("retained quality document is absent"))?;
    let metadata = document
        .metadata
        .get_or_insert_with(|| serde_json::json!({}));
    metadata[SOURCE_METADATA] = serde_json::to_value(SourceInput {
        revision,
        source_sha256: content_sha256_hex(text.as_bytes()),
    })
    .map_err(storage_error)?;
    storage.upsert_document(&document)?;
    storage.set_document_content(id, text)?;
    Ok(())
}

/// Reconcile the new selected membership and revisions under a fresh queue key.
/// Old work is retired only in the candidate, never in the selected generation.
#[allow(clippy::arc_with_non_send_sync)]
pub(super) fn finish_content_upserts(
    cx: &Cx,
    candidate: &FsfsRuntime,
    destination: &Path,
) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    let storage = Arc::new(open_catalog(candidate, false)?);
    let mut backlog = read_backlog(&storage)?;
    if backlog.schema_version != 2 {
        return Err(invalid(
            "content mutation has no admitted owned-input backlog",
        ));
    }
    let old_key = backlog.queue_key();
    let manifests = FsfsRuntime::read_matching_manifest_generation(destination)?
        .ok_or_else(|| invalid("batch candidate membership manifests disagree"))?;
    let mut sentinel = FsfsRuntime::read_index_sentinel(destination)?
        .ok_or_else(|| invalid("batch candidate has no completion sentinel"))?;
    backlog.generation_root = destination.display().to_string();
    backlog.source_hash.clone_from(&sentinel.source_hash_hex);
    backlog.requested_documents = manifests
        .values()
        .filter(|entry| {
            entry.ingestion_class == ingestion_class_label(IngestionClass::FullSemanticLexical)
        })
        .count();
    backlog.queued_documents = backlog.requested_documents;
    storage
        .connection()
        .execute_with_params_sync(
            "DELETE FROM embedding_jobs WHERE embedder_id = ?1;",
            &[SqliteValue::Text(old_key.into())],
        )
        .map_err(storage_error)?;
    let key = backlog.queue_key();
    let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
    for (id, entry) in &manifests {
        retained_search_checkpoint(cx)?;
        if entry.ingestion_class != ingestion_class_label(IngestionClass::FullSemanticLexical) {
            continue;
        }
        let text = storage
            .get_document_content(id)?
            .ok_or_else(|| invalid("batch candidate lost its exact deferred input"))?;
        queue.enqueue(id, &key, &ContentHasher::hash(&text), 0)?;
    }
    validate_pending_jobs(&storage, &key, &manifests, None)?;
    record_backlog(&storage, &backlog, 0)?;
    if backlog.requested_documents == 0 {
        sentinel
            .reason_codes
            .retain(|reason| reason != REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE);
        candidate.write_index_sentinel(destination, &sentinel)?;
    } else if !sentinel
        .reason_codes
        .iter()
        .any(|reason| reason == REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE)
    {
        sentinel
            .reason_codes
            .push(REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE.to_owned());
        candidate.write_index_sentinel(destination, &sentinel)?;
    }
    retained_search_checkpoint(cx)
}

/// Carry only unchanged pending inputs through deletion or compaction. The
/// caller owns an independently copied candidate and the store's build lease.
/// Never reconstruct missing input from a preview or a mutable source path.
/// A mixed batch supplies the checkpoint its ordinary inherited-input path
/// already admitted, including invalidation of every touched skip witness.
pub(super) fn carry_after_maintenance<F>(
    cx: &Cx,
    runtime: &FsfsRuntime,
    predecessor: &PublishedGeneration,
    destination: &Path,
    project_inputs: F,
) -> SearchResult<()>
where
    F: FnOnce() -> SearchResult<Option<super::IndexingCheckpoint>>,
{
    retained_search_checkpoint(cx)?;
    let previous = FsfsRuntime::read_index_sentinel(predecessor.path())?
        .ok_or_else(|| invalid("maintenance predecessor has no completion sentinel"))?;
    let Some((storage, mut backlog)) = open_carried_backlog(runtime, predecessor, &previous)?
    else {
        return Ok(());
    };
    if backlog.generation_root != predecessor.path().display().to_string()
        || backlog.source_hash != previous.source_hash_hex
        || backlog.unavailable_documents != 0
    {
        return Err(invalid(
            "maintenance requires a consistent exact deferred backlog; run a full source reindex to establish new work",
        ));
    }
    let manifests = FsfsRuntime::read_matching_manifest_generation(destination)?
        .ok_or_else(|| invalid("maintenance candidate membership manifests disagree"))?;
    let requested = manifests
        .values()
        .filter(|entry| {
            entry.ingestion_class == ingestion_class_label(IngestionClass::FullSemanticLexical)
        })
        .count();
    if requested > backlog.requested_documents {
        return Err(invalid(
            "maintenance cannot grow the declared deferred membership",
        ));
    }
    let checkpoint =
        if backlog.schema_version == 2 || backlog.requested_documents == 0 {
            None
        } else if requested == 0 {
            // No remaining semantic member owes inference, including deletion of
            // the last document. The original checkpoint still fences the empty
            // queue validation; no empty source-reuse receipt need be fabricated.
            Some(retained_reuse::completed_input_checkpoint(cx, predecessor)?)
        } else {
            Some(project_inputs()?.ok_or_else(|| {
                invalid("maintenance cannot preserve the deferred source checkpoint")
            })?)
        };
    let old_key = backlog.queue_key();
    validate_pending_jobs(
        &storage,
        &old_key,
        &manifests,
        checkpoint.as_ref().map(|checkpoint| &checkpoint.files),
    )?;
    let mut sentinel = FsfsRuntime::read_index_sentinel(destination)?
        .ok_or_else(|| invalid("maintenance candidate has no completion sentinel"))?;
    backlog.generation_root = destination.display().to_string();
    backlog.source_hash.clone_from(&sentinel.source_hash_hex);
    backlog.requested_documents = requested;
    backlog.queued_documents = requested;
    storage
        .connection()
        .execute_with_params_sync(
            "UPDATE embedding_jobs SET embedder_id = ?1 WHERE embedder_id = ?2;",
            &[
                SqliteValue::Text(backlog.queue_key().into()),
                SqliteValue::Text(old_key.into()),
            ],
        )
        .map_err(storage_error)?;
    record_backlog(&storage, &backlog, 0)?;
    if requested == 0 {
        sentinel
            .reason_codes
            .retain(|reason| reason != REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE);
        runtime.write_index_sentinel(destination, &sentinel)?;
    }
    tracing::info!(
        queued = requested,
        source_hash = backlog.source_hash,
        producer = backlog.producer.fingerprint(),
        "preserved exact deferred quality work across retained maintenance"
    );
    retained_search_checkpoint(cx)
}

impl FsfsRuntime {
    /// Finish the exact quality backlog of an explicitly selected generation.
    ///
    /// No source path is scanned, opened, or canonicalized. All vector writes
    /// and queue transitions occur in an independently copied successor. An
    /// error or cancellation keeps the incumbent and all its pending jobs;
    /// retrying can repeat inference from that immutable input. This operation
    /// does not resume partial unselected candidates or run a background worker.
    ///
    /// # Errors
    /// Rejects active deferral, missing exact input, a changed generation or
    /// producer, corrupt queues, unavailable models, and publication failures.
    #[allow(clippy::future_not_send)]
    pub async fn complete_retained_quality(
        &self,
        cx: &Cx,
        store_root: &Path,
        expected: &PublishedGeneration,
    ) -> SearchResult<DeferredQualityCompletion> {
        match self
            .complete_retained_quality_with_precommit(cx, store_root, expected, |_| Ok(()))
            .await?
        {
            DeferredQualityAttempt::Completed(completed) => Ok(completed),
            DeferredQualityAttempt::MissingBacklog => Err(invalid(
                "selected generation has no retained quality backlog; run a full source reindex",
            )),
            DeferredQualityAttempt::ProducerChanged => Err(invalid(
                "deferred quality producer changed; run a full source reindex to establish new work",
            )),
        }
    }

    /// The watch owner rechecks its source and notification authority after
    /// sealing, at the same publication boundary as ordinary source updates.
    /// Missing backlog metadata and a newly selected producer have distinct
    /// outcomes before inference, so watch can use its ordinary source rebuild
    /// while explicit completion preserves its refusal diagnostic.
    #[allow(
        clippy::future_not_send,
        clippy::too_many_lines,
        clippy::arc_with_non_send_sync
    )]
    pub(super) async fn complete_retained_quality_with_precommit<F>(
        &self,
        cx: &Cx,
        store_root: &Path,
        expected: &PublishedGeneration,
        precommit: F,
    ) -> SearchResult<DeferredQualityAttempt>
    where
        F: FnOnce(&Cx) -> SearchResult<()>,
    {
        let request = self.search_blocking_pool.as_ref().map_or_else(
            || cx.clone(),
            |pool| cx.clone().with_blocking_pool_handle(Some(pool.clone())),
        );
        let cx = &request;
        retained_search_checkpoint(cx)?;
        if self.config.pressure.degradation_override
            == crate::config::DegradationOverrideMode::ForcePaused
        {
            return Err(invalid(
                "quality completion is disabled while indexing is paused",
            ));
        }
        if self.quality_disabled_by_policy() || self.lexical_only_indexing {
            return Err(invalid(
                "clear quality deferral and fast-only policy before completing quality",
            ));
        }
        if cx.blocking_pool_handle().is_none() {
            return Err(invalid(
                "quality completion requires a caller-owned model blocking pool",
            ));
        }
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let producer = declared_producer(self)?
            .ok_or_else(|| invalid("quality completion requires a registered quality model"))?;
        let mut control = IndexRunControl::new(&self.config)?;
        control.admit_work(cx, "complete_quality.start").await?;
        let store = CompleteGenerationStore::open(cx, store_root)?;
        let build = store.begin(cx)?;
        let predecessor = store
            .active(cx)?
            .ok_or_else(|| invalid("no complete generation is selected"))?;
        if &predecessor != expected {
            return Err(invalid(
                "selected generation differs from the expected quality backlog",
            ));
        }
        let sentinel = Self::read_index_sentinel(predecessor.path())?
            .ok_or_else(|| invalid("selected generation has no completion sentinel"))?;
        if !sentinel
            .reason_codes
            .iter()
            .any(|reason| reason == REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE)
            || predecessor
                .path()
                .join(FSFS_VECTOR_QUALITY_INDEX_FILE)
                .try_exists()?
        {
            return Err(invalid(
                "selected generation does not have an unbuilt deferred quality tier",
            ));
        }
        let mut input = self.cli_input.clone();
        input.index_dir = Some(build.path().to_path_buf());
        input.complete_quality = false;
        input.daemon = false;
        input.daemon_socket = None;
        let candidate = self.clone().with_cli_input(input);
        if retained_reuse::copy_selected_generation(cx, &candidate, &store, build.path())?
            != predecessor
        {
            return Err(invalid(
                "selected quality backlog changed while copying its generation",
            ));
        }
        let lease = PublicationLease::acquire(build.path())?;
        let storage = Arc::new(open_catalog(&candidate, false)?);
        let Some(backlog) = read_optional_backlog(&storage)? else {
            return Ok(DeferredQualityAttempt::MissingBacklog);
        };
        if backlog.generation_root != predecessor.path().display().to_string()
            || backlog.source_hash != sentinel.source_hash_hex
        {
            return Err(invalid(
                "deferred quality source generation or producer changed; run a full source reindex to establish new work",
            ));
        }
        if backlog.unavailable_documents != 0 {
            return Err(invalid(
                "some deferred documents lack exact retained input; run a full source reindex before completing quality",
            ));
        }
        let checkpoint = if backlog.schema_version == 1 {
            Some(retained_reuse::completed_input_checkpoint(
                cx,
                &predecessor,
            )?)
        } else {
            None
        };
        let manifests = if let Some(checkpoint) = &checkpoint {
            Self::read_checkpoint_manifest_generation(predecessor.path(), checkpoint)?
        } else {
            Self::read_matching_manifest_generation(predecessor.path())?
        }
        .ok_or_else(|| invalid("selected membership manifests and input fence disagree"))?;
        let key = backlog.queue_key();
        validate_pending_jobs(
            &storage,
            &key,
            &manifests,
            checkpoint.as_ref().map(|checkpoint| &checkpoint.files),
        )?;
        if backlog.producer != producer {
            // A configuration change may request a source rebuild, but cannot
            // conceal damaged source fences or queued input in its predecessor.
            return Ok(DeferredQualityAttempt::ProducerChanged);
        }
        control
            .admit_work(cx, "complete_quality.model_load")
            .await?;
        let permit =
            asupersync::sync::OwnedMutexGuard::lock(Arc::clone(&candidate.quality_load_gate), cx)
                .await
                .map_err(|error| SearchError::Cancelled {
                    phase: "complete_quality.model_load".to_owned(),
                    reason: error.to_string(),
                })?;
        let embedder = if let Some(embedder) = permit.0.as_ref().map(Arc::clone) {
            drop(permit);
            Some(embedder)
        } else {
            let initialization = candidate.start_quality_initialization(cx, permit)?;
            Self::join_search_worker(cx, initialization).await??
        }
        .ok_or_else(|| invalid("the deferred quality producer is unavailable"))?;
        Self::ensure_semantic_embedder_admissible(embedder.as_ref(), false)?;
        let admitted = AdmittedEmbedder::admit(cx, embedder)?;
        if admitted.identity != backlog.producer {
            return Err(invalid(
                "loaded quality producer differs from the queued producer identity",
            ));
        }
        lease.fence("deferred quality candidate vector creation")?;
        let mut vector = VectorIndex::replace_with_empty(
            &build.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE),
            admitted.embedder().id(),
            &admitted.identity.fingerprint(),
            admitted.embedder().dimension(),
        )?;
        Self::admit_quality_generation_for_embedder(&vector, admitted.embedder())?;
        let native_batch = admitted.embedder().bound_batch_is_native();
        let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
        let mut completed = 0_usize;
        loop {
            control.admit_work(cx, "complete_quality.batch").await?;
            let jobs =
                queue.claim_batch_for_embedder("fsfs.complete_quality", JOB_BATCH_SIZE, &key)?;
            if jobs.is_empty() {
                break;
            }
            let mut texts = Vec::with_capacity(jobs.len());
            for job in &jobs {
                retained_search_checkpoint(cx)?;
                let ClaimOutcome::Applied(document) = queue.check_claim(job, None)? else {
                    return Err(invalid(
                        "deferred quality job lost its source revision before inference",
                    ));
                };
                let text = storage
                    .get_document_content(&job.doc_id)?
                    .ok_or_else(|| invalid("exact deferred input disappeared"))?;
                if job.content_hash != Some(document.content_hash) || text.len() > MAX_INPUT_BYTES {
                    return Err(invalid("deferred quality job has a different input digest"));
                }
                texts.push(text);
            }
            let borrowed = texts.iter().map(String::as_str).collect::<Vec<_>>();
            if admitted.embedder().bound_batch_is_native() != native_batch {
                return Err(invalid(
                    "quality producer changed its bound batching contract",
                ));
            }
            let embeddings = if native_batch {
                admitted.embed_batch_admitted(cx, &borrowed).await?
            } else {
                let mut values = Vec::with_capacity(borrowed.len());
                for text in borrowed {
                    control.admit_work(cx, "complete_quality.embedding").await?;
                    values.push(admitted.embed_admitted(cx, text).await?);
                }
                values
            };
            if embeddings.len() != jobs.len() {
                return Err(invalid(
                    "quality inference did not return one bound vector per queued input",
                ));
            }
            let mut entries = Vec::with_capacity(jobs.len());
            for (job, values) in jobs.iter().zip(embeddings) {
                retained_search_checkpoint(cx)?;
                let hash = job
                    .content_hash
                    .as_ref()
                    .ok_or_else(|| invalid("deferred job has no input digest"))?;
                if !matches!(
                    queue.check_claim(job, Some(hash))?,
                    ClaimOutcome::Applied(_)
                ) {
                    return Err(invalid(
                        "deferred quality job lost ownership after inference",
                    ));
                }
                let norm = values.iter().map(|value| value * value).sum::<f32>();
                if norm <= 0.0 || !norm.is_finite() {
                    return Err(invalid("quality producer returned an unusable vector"));
                }
                entries.push((job.doc_id.clone(), values));
            }
            control.admit_work(cx, "complete_quality.persist").await?;
            lease.fence("deferred quality candidate batch persistence")?;
            vector.append_batch(&entries)?;
            for job in &jobs {
                retained_search_checkpoint(cx)?;
                let hash = job
                    .content_hash
                    .as_ref()
                    .ok_or_else(|| invalid("deferred job has no input digest"))?;
                if queue.complete(job, hash)? != ClaimOutcome::Applied(()) {
                    return Err(invalid(
                        "deferred quality job lost ownership before completion",
                    ));
                }
                storage.mark_embedded(&job.doc_id, admitted.embedder().id())?;
                completed += 1;
            }
        }
        if completed != backlog.requested_documents || vector.live_doc_ids()?.len() != completed {
            return Err(invalid(
                "deferred quality queue did not complete its entire source membership",
            ));
        }
        control.admit_work(cx, "complete_quality.finish").await?;
        vector.compact()?;
        vector.vacuum()?;
        Self::admit_quality_generation_for_embedder(&vector, admitted.embedder())?;
        drop(vector);
        record_backlog(&storage, &backlog, completed)?;
        drop(queue);
        drop(storage);
        let mut sentinel = Self::read_index_sentinel(build.path())?
            .ok_or_else(|| invalid("quality candidate has no completion sentinel"))?;
        "complete-quality".clone_into(&mut sentinel.command);
        sentinel.generated_at_ms = pressure_timestamp_ms();
        sentinel
            .reason_codes
            .retain(|reason| !reason.starts_with("vector.quality_tier."));
        sentinel
            .reason_codes
            .push(REASON_VECTOR_QUALITY_TIER_BUILT.to_owned());
        for reason in protect_vector_generations(build.path(), "complete deferred quality") {
            if !sentinel.reason_codes.contains(&reason) {
                sentinel.reason_codes.push(reason);
            }
        }
        candidate.write_index_sentinel(build.path(), &sentinel)?;
        lease.fence("deferred quality candidate complete")?;
        drop(lease);
        let resources = Box::pin(
            candidate.prepare_search_execution_resources_at_root_with_modes(
                cx,
                build.path(),
                SearchExecutionMode::Full,
                SearchExecutionMode::Full,
            ),
        )
        .await?;
        if resources.quality_vector_index.is_none() {
            return Err(invalid(
                "completed quality generation did not admit its serving resources",
            ));
        }
        drop(resources);
        control.admit_work(cx, "complete_quality.publish").await?;
        admitted.check_current_identity()?;
        let publication = build.publish_with_precommit(
            cx,
            |_, path| Self::validate_search_generation_at_root(path, SearchExecutionMode::Full),
            precommit,
        )?;
        Ok(DeferredQualityAttempt::Completed(
            DeferredQualityCompletion {
                publication,
                completed_documents: completed,
                source_hash: backlog.source_hash,
                producer_fingerprint: backlog.producer.fingerprint(),
            },
        ))
    }
}

#[cfg(all(test, feature = "semantic-loaders"))]
mod tests {
    use std::fmt::Write as _;
    use std::future::Future;
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU8, Ordering};

    use asupersync::runtime::RuntimeBuilder;
    use frankensearch_core::{
        Canonicalizer, DefaultCanonicalizer, Embedder, IdentityBoundEmbedding, ModelCategory,
        SearchFuture,
    };

    use super::*;
    use crate::adapters::cli::{CliCommand, CliInput};
    use crate::config::DegradationOverrideMode;
    use crate::output_schema::SearchOutputPhase;
    use crate::runtime::{
        FSFS_VECTOR_INDEX_FILE, set_test_fast_embedder, set_test_quality_embedder,
        test_fast_embedder_override, test_quality_embedder_override,
    };

    struct Producer {
        id: &'static str,
        identity: EmbeddingIdentityBundleV1,
        inputs: Mutex<Vec<String>>,
        outcome: AtomicU8,
    }

    impl Producer {
        fn new(id: &'static str) -> Self {
            Self {
                id,
                identity: EmbeddingIdentityBundleV1::explicit_test_model(id, 3),
                inputs: Mutex::new(Vec::new()),
                outcome: AtomicU8::new(0),
            }
        }
    }

    impl Embedder for Producer {
        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &'static str {
            "Explicit deferred-quality wiring control"
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn dimension(&self) -> usize {
            3
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::TransformerEmbedder
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn embed<'a>(&'a self, _: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            // The provider overrides bound inference and does not advertise a
            // native bound batch. A caller bypassing it would miss the negative
            // controls and this counted input entirely.
            Box::pin(async move {
                if text == "probe" {
                    Ok(vec![1.0, 0.0, 0.0])
                } else {
                    Err(invalid("raw inference must not bypass the bound producer"))
                }
            })
        }

        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.inputs.lock().unwrap().push(text.to_owned());
                let outcome = self.outcome.load(Ordering::SeqCst);
                if outcome == 1 {
                    cx.set_cancel_requested(true);
                }
                if outcome == 3 {
                    return Err(invalid("controlled quality inference failure"));
                }
                Ok(IdentityBoundEmbedding {
                    values: vec![1.0, 0.0, 0.0],
                    identity: if outcome == 2 {
                        EmbeddingIdentityBundleV1::explicit_test_model("foreign-response", 3)
                    } else {
                        self.identity.clone()
                    },
                })
            })
        }

        fn embed_batch_bound<'a>(
            &'a self,
            cx: &'a Cx,
            texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
            Box::pin(async move {
                let mut responses = Vec::with_capacity(texts.len());
                for text in texts {
                    responses.push(self.embed_bound(cx, text).await?);
                }
                Ok(responses)
            })
        }
    }

    struct Providers {
        fast: Option<Arc<dyn Embedder>>,
        quality: Option<Arc<dyn Embedder>>,
    }

    impl Providers {
        fn install(fast: Arc<Producer>, quality: Arc<Producer>) -> Self {
            let previous = Self {
                fast: test_fast_embedder_override(),
                quality: test_quality_embedder_override(),
            };
            set_test_fast_embedder(Some(fast));
            set_test_quality_embedder(Some(quality));
            previous
        }
    }

    impl Drop for Providers {
        fn drop(&mut self) {
            set_test_fast_embedder(self.fast.take());
            set_test_quality_embedder(self.quality.take());
        }
    }

    fn run<F: Future<Output = ()>>(test: impl FnOnce(Cx) -> F) {
        let scheduler = RuntimeBuilder::current_thread()
            .blocking_threads(0, 2)
            .build()
            .unwrap();
        scheduler.block_on(async move {
            let cx = Cx::current().expect("registered current-thread root");
            test(cx).await;
        });
    }

    fn runtime(parent: &Path, deferred: bool) -> FsfsRuntime {
        let mut config = crate::FsfsConfig::default();
        config.indexing.offline = true;
        config.indexing.fast_window_max_per_file = 1;
        "all-minilm-l6-v2".clone_into(&mut config.indexing.quality_model);
        config.search.rerank = false;
        config.search.fast_only = false;
        config.pressure.profile = crate::config::PressureProfile::Strict;
        config.pressure.degradation_override = if deferred {
            DegradationOverrideMode::ForceEmbedDeferred
        } else {
            DegradationOverrideMode::Auto
        };
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Index,
            target_path: deferred.then(|| parent.join("source")),
            index_dir: Some(parent.join("store")),
            quiet: true,
            ..CliInput::default()
        })
    }

    fn source_fixture(parent: &Path) -> BTreeMap<String, String> {
        let source = parent.join("source");
        fs::create_dir(&source).unwrap();
        let mut code = String::new();
        for number in 0..40 {
            writeln!(&mut code, "let variable_{number} = {number};").unwrap();
        }
        let documents = BTreeMap::from([
            (
                "alpha.md".to_owned(),
                "sharedtoken quality retained source alpha".to_owned(),
            ),
            (
                "beta.md".to_owned(),
                format!("sharedtoken retained code sample\n```rust\n{code}\n```\nquality tail"),
            ),
        ]);
        for (id, text) in &documents {
            fs::write(source.join(id), text).unwrap();
        }
        documents
            .into_iter()
            .map(|(id, text)| (id, DefaultCanonicalizer::default().canonicalize(&text)))
            .collect()
    }

    fn durable(publication: GenerationPublication) -> PublishedGeneration {
        match publication {
            GenerationPublication::Durable(generation) => generation,
            GenerationPublication::VisibleButDurabilityUncertain { .. } => {
                panic!("test publication must be durable");
            }
        }
    }

    fn files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut result = BTreeMap::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for entry in fs::read_dir(directory).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    result.insert(
                        path.strip_prefix(root).unwrap().to_path_buf(),
                        fs::read(path).unwrap(),
                    );
                }
            }
        }
        result
    }

    #[test]
    fn restart_finishes_exact_quality_input_without_sources_or_fast_reembedding() {
        run(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let expected_inputs = source_fixture(directory.path());
            let fast = Arc::new(Producer::new("deferred-fast"));
            let quality = Arc::new(Producer::new("deferred-quality"));
            let _providers = Providers::install(Arc::clone(&fast), Arc::clone(&quality));
            let indexed = runtime(directory.path(), true)
                .rebuild_retained_generation(&cx, &directory.path().join("store"))
                .await
                .unwrap();
            let first = durable(indexed);
            assert!(quality.inputs.lock().unwrap().is_empty());
            assert!(!first.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE).exists());
            let before = files(first.path());
            fast.inputs.lock().unwrap().clear();
            fs::rename(
                directory.path().join("source"),
                directory.path().join("source-gone"),
            )
            .unwrap();

            // New runtime means no retained source-preparation state or model
            // cache from the indexing invocation can supply missing input.
            let restarted = runtime(directory.path(), false);
            let result = restarted
                .complete_retained_quality(&cx, &directory.path().join("store"), &first)
                .await
                .unwrap();
            assert_eq!(result.completed_documents, expected_inputs.len());
            let next = durable(result.publication);
            assert!(fast.inputs.lock().unwrap().is_empty());
            let mut observed = quality.inputs.lock().unwrap().clone();
            observed.sort();
            let mut expected = expected_inputs.into_values().collect::<Vec<_>>();
            expected.sort();
            assert_eq!(observed, expected);
            assert_eq!(files(first.path()), before);
            assert_eq!(
                fs::read(next.path().join(FSFS_VECTOR_INDEX_FILE)).unwrap(),
                before[Path::new(FSFS_VECTOR_INDEX_FILE)]
            );
            let successor_before = files(next.path());
            let mut reader = restarted
                .open_retained_search(&cx, &directory.path().join("store"))
                .await
                .unwrap();
            let phases = reader
                .search(
                    &cx,
                    "how do sharedtoken documents retain quality input across restarts",
                    5,
                )
                .await
                .unwrap();
            // Strict pressure deliberately serves only the fast phase even
            // after quality completion. A performance reader must still admit
            // and execute the newly published quality tier.
            assert_eq!(phases.len(), 1);
            assert_eq!(phases[0].phase, SearchOutputPhase::Initial);
            let mut performance_config = restarted.config().clone();
            performance_config.pressure.profile = crate::config::PressureProfile::Performance;
            let performance =
                FsfsRuntime::new(performance_config).with_cli_input(restarted.cli_input.clone());
            let mut reader = performance
                .open_retained_search(&cx, &directory.path().join("store"))
                .await
                .unwrap();
            let phases = reader
                .search(
                    &cx,
                    "how do sharedtoken documents retain quality input across restarts",
                    5,
                )
                .await
                .unwrap();
            assert_eq!(phases.last().unwrap().phase, SearchOutputPhase::Refined);
            assert!(!phases.last().unwrap().hits.is_empty());
            assert_eq!(files(next.path()), successor_before);
        });
    }

    #[test]
    fn deferred_upserts_retain_latest_inputs_and_resume_after_source_loss() {
        use super::super::retained_batch::RetainedMutation;

        run(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let expected = source_fixture(directory.path());
            let fast = Arc::new(Producer::new("deferred-fast"));
            let quality = Arc::new(Producer::new("deferred-quality"));
            let _providers = Providers::install(Arc::clone(&fast), Arc::clone(&quality));
            let deferred = runtime(directory.path(), true);
            let root = directory.path().join("store");
            let first = durable(
                deferred
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            );
            let before = files(first.path());
            fast.inputs.lock().unwrap().clear();
            fs::rename(
                directory.path().join("source"),
                directory.path().join("source-gone"),
            )
            .unwrap();
            let updated = deferred
                .apply_retained_batch(
                    &cx,
                    &root,
                    &first,
                    &[
                        RetainedMutation::Upsert {
                            id: "alpha.md".to_owned(),
                            text: "replacement retained alpha".to_owned(),
                        },
                        RetainedMutation::Upsert {
                            id: "gamma.md".to_owned(),
                            text: "superseded retained gamma".to_owned(),
                        },
                    ],
                )
                .await
                .unwrap();
            assert_eq!((updated.upserted, updated.deleted), (2, 0));
            let updated = durable(updated.publication.unwrap());
            let updated_bytes = files(updated.path());
            assert!(!updated.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE).exists());
            assert_eq!(files(first.path()), before);
            assert!(quality.inputs.lock().unwrap().is_empty());

            let restarted = runtime(directory.path(), false);
            let latest = restarted
                .apply_retained_batch(
                    &cx,
                    &root,
                    &updated,
                    &[
                        RetainedMutation::Upsert {
                            id: "gamma.md".to_owned(),
                            text: "latest retained gamma".to_owned(),
                        },
                        RetainedMutation::Delete {
                            id: "alpha.md".to_owned(),
                        },
                    ],
                )
                .await
                .unwrap();
            assert_eq!((latest.upserted, latest.deleted), (1, 1));
            let latest = durable(latest.publication.unwrap());
            let latest_bytes = files(latest.path());
            assert_eq!(files(updated.path()), updated_bytes);
            assert_eq!(fast.inputs.lock().unwrap().len(), 3);
            assert!(quality.inputs.lock().unwrap().is_empty());
            fast.inputs.lock().unwrap().clear();
            let (compacted, _) = restarted
                .compact_retained_generation(&cx, &root)
                .await
                .unwrap();
            let compacted = durable(compacted.unwrap());
            let compacted_bytes = files(compacted.path());
            assert!(fast.inputs.lock().unwrap().is_empty());

            // The CLI has its own retained append kernel. Exercise it, rather
            // than assuming the public mixed-batch path also wires that route.
            let forced_file = directory.path().join("forced-cli.jsonl");
            fs::write(
                &forced_file,
                serde_json::json!({"id": "gamma.md", "text": "intermediate CLI gamma"}).to_string(),
            )
            .unwrap();
            let mut forced_cli = runtime(directory.path(), true);
            forced_cli.cli_input.command = CliCommand::AppendBatch;
            forced_cli.cli_input.input_file = Some(forced_file);
            let (publication, count) = forced_cli
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 1);
            let cli_first = durable(publication.unwrap());
            let cli_first_bytes = files(cli_first.path());
            assert_eq!(files(compacted.path()), compacted_bytes);

            let cleared_file = directory.path().join("cleared-cli.jsonl");
            fs::write(
                &cleared_file,
                format!(
                    "{}\n{}\n",
                    serde_json::json!({"id": "gamma.md", "text": "final CLI gamma"}),
                    serde_json::json!({"id": "delta.md", "text": "new CLI delta"}),
                ),
            )
            .unwrap();
            let mut cleared_cli = runtime(directory.path(), false);
            cleared_cli.cli_input.command = CliCommand::AppendBatch;
            cleared_cli.cli_input.input_file = Some(cleared_file);
            let (publication, count) = cleared_cli
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 2);
            let cli_latest = durable(publication.unwrap());
            let cli_latest_bytes = files(cli_latest.path());
            assert_eq!(files(cli_first.path()), cli_first_bytes);
            assert_eq!(fast.inputs.lock().unwrap().len(), 3);
            assert!(quality.inputs.lock().unwrap().is_empty());
            fast.inputs.lock().unwrap().clear();

            // A new caller must still admit the original queued producer.
            set_test_quality_embedder(Some(Arc::new(Producer::new("foreign-quality"))));
            let error = runtime(directory.path(), false)
                .complete_retained_quality(&cx, &root, &cli_latest)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "index.complete_quality"));
            assert_eq!(files(cli_latest.path()), cli_latest_bytes);
            assert!(quality.inputs.lock().unwrap().is_empty());
            set_test_quality_embedder(Some(quality.clone()));

            cx.set_cancel_requested(true);
            assert!(matches!(
                restarted
                    .complete_retained_quality(&cx, &root, &cli_latest)
                    .await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(files(cli_latest.path()), cli_latest_bytes);
            let completed = restarted
                .complete_retained_quality(&cx, &root, &cli_latest)
                .await
                .unwrap();
            assert_eq!(completed.completed_documents, 3);
            let completed = durable(completed.publication);
            assert_eq!(files(first.path()), before);
            assert_eq!(files(updated.path()), updated_bytes);
            assert_eq!(files(latest.path()), latest_bytes);
            assert_eq!(files(compacted.path()), compacted_bytes);
            assert_eq!(files(cli_first.path()), cli_first_bytes);
            assert_eq!(files(cli_latest.path()), cli_latest_bytes);
            assert_eq!(
                fs::read(completed.path().join(FSFS_VECTOR_INDEX_FILE)).unwrap(),
                cli_latest_bytes[Path::new(FSFS_VECTOR_INDEX_FILE)]
            );
            assert_eq!(
                quality
                    .inputs
                    .lock()
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from([
                    expected["beta.md"].clone(),
                    "final CLI gamma".to_owned(),
                    "new CLI delta".to_owned(),
                ])
            );
            assert_eq!(
                VectorIndex::open_read_only(&completed.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE))
                    .unwrap()
                    .live_doc_ids()
                    .unwrap(),
                std::collections::HashSet::from([
                    "beta.md".to_owned(),
                    "gamma.md".to_owned(),
                    "delta.md".to_owned(),
                ])
            );
            assert!(fast.inputs.lock().unwrap().is_empty());

            fast.inputs.lock().unwrap().clear();
            quality.inputs.lock().unwrap().clear();
            let completed_bytes = files(completed.path());
            let updated = restarted
                .apply_retained_batch(
                    &cx,
                    &root,
                    &completed,
                    &[RetainedMutation::Upsert {
                        id: "new.md".to_owned(),
                        text: "new retained content".to_owned(),
                    }],
                )
                .await
                .unwrap();
            assert_eq!((updated.upserted, updated.deleted), (1, 0));
            let updated = durable(updated.publication.unwrap());
            assert!(
                updated
                    .path()
                    .join(FSFS_VECTOR_QUALITY_INDEX_FILE)
                    .is_file()
            );
            assert_eq!(
                fast.inputs.lock().unwrap().as_slice(),
                &["new retained content"]
            );
            assert_eq!(
                quality.inputs.lock().unwrap().as_slice(),
                &["new retained content"]
            );
            assert_eq!(files(completed.path()), completed_bytes);
        });
    }

    #[test]
    fn owned_quality_fence_rejects_same_length_text_and_digest_substitution() {
        for replace_queue_digest in [false, true] {
            let storage = Arc::new(Storage::open_in_memory().unwrap());
            let original = "safe retained body";
            let corrupt = "evil retained body";
            assert_eq!(original.len(), corrupt.len());
            let mut document = DocumentRecord::new(
                "owned.md",
                original,
                ContentHasher::hash(original),
                original.chars().count(),
                7,
                7,
            );
            document.metadata = Some(serde_json::json!({
                SOURCE_METADATA: {
                    "revision": 7,
                    "source_sha256": content_sha256_hex(original.as_bytes()),
                },
            }));
            storage.upsert_document(&document).unwrap();
            storage.set_document_content("owned.md", original).unwrap();
            let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
            queue
                .enqueue("owned.md", "owned-quality", &document.content_hash, 0)
                .unwrap();
            let manifests = BTreeMap::from([(
                "owned.md".to_owned(),
                IndexManifestEntry {
                    file_key: "owned.md".to_owned(),
                    revision: 7,
                    ingestion_class: ingestion_class_label(IngestionClass::FullSemanticLexical)
                        .to_owned(),
                    canonical_bytes: original.len() as u64,
                    reason_code: "retained_batch".to_owned(),
                    fast_windows: None,
                },
            )]);
            validate_pending_jobs(&storage, "owned-quality", &manifests, None).unwrap();
            // The public setter refuses this corruption. Inject damaged
            // stored bytes directly, as a corrupt catalog would present them.
            storage
                .connection()
                .execute_with_params_sync(
                    "UPDATE document_contents SET canonical_text = ?1;",
                    &[SqliteValue::Text(corrupt.to_owned().into())],
                )
                .unwrap();
            if replace_queue_digest {
                document.content_hash = ContentHasher::hash(corrupt);
                storage.upsert_document(&document).unwrap();
                storage.set_document_content("owned.md", corrupt).unwrap();
                queue
                    .enqueue("owned.md", "owned-quality", &document.content_hash, 0)
                    .unwrap();
            }
            let error =
                validate_pending_jobs(&storage, "owned-quality", &manifests, None).unwrap_err();
            if replace_queue_digest {
                assert!(matches!(error, SearchError::InvalidConfig { reason, .. }
                    if reason.contains("input differs")));
            } else {
                assert!(matches!(
                    error,
                    SearchError::SubsystemError {
                        subsystem: "storage",
                        ..
                    }
                ));
            }
        }
    }

    #[test]
    fn maintenance_preserves_pending_quality_without_sources_or_reembedding() {
        run(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let expected = source_fixture(directory.path());
            let fast = Arc::new(Producer::new("deferred-fast"));
            let quality = Arc::new(Producer::new("deferred-quality"));
            let _providers = Providers::install(Arc::clone(&fast), Arc::clone(&quality));
            let deferred = runtime(directory.path(), true);
            let root = directory.path().join("store");
            let first = durable(
                deferred
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            );
            let first_bytes = files(first.path());
            fast.inputs.lock().unwrap().clear();
            let (compacted, _) = deferred
                .compact_retained_generation(&cx, &root)
                .await
                .unwrap();
            let compacted = durable(compacted.unwrap());
            let compacted_bytes = files(compacted.path());
            let mut deleting = deferred.clone();
            deleting.cli_input.delete_ids = vec!["beta.md".to_owned()];
            let (deleted, count) = deleting
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 1);
            let deleted = durable(deleted.unwrap());
            let deleted_bytes = files(deleted.path());
            assert!(fast.inputs.lock().unwrap().is_empty());
            assert!(quality.inputs.lock().unwrap().is_empty());
            fs::rename(
                directory.path().join("source"),
                directory.path().join("source-gone"),
            )
            .unwrap();
            let restarted = runtime(directory.path(), false);
            let completed = restarted
                .complete_retained_quality(&cx, &root, &deleted)
                .await
                .unwrap();
            assert_eq!(completed.completed_documents, 1);
            let completed = durable(completed.publication);
            assert_eq!(
                quality.inputs.lock().unwrap().as_slice(),
                &[expected["alpha.md"].clone()]
            );
            assert!(fast.inputs.lock().unwrap().is_empty());
            assert_eq!(files(first.path()), first_bytes);
            assert_eq!(files(compacted.path()), compacted_bytes);
            assert_eq!(files(deleted.path()), deleted_bytes);
            assert_eq!(
                fs::read(completed.path().join(FSFS_VECTOR_INDEX_FILE)).unwrap(),
                deleted_bytes[Path::new(FSFS_VECTOR_INDEX_FILE)]
            );
            assert_eq!(
                VectorIndex::open_read_only(&completed.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE))
                    .unwrap()
                    .live_doc_ids()
                    .unwrap(),
                std::collections::HashSet::from(["alpha.md".to_owned()])
            );
        });
    }

    #[test]
    fn deleting_every_deferred_document_retires_owed_quality_without_inference() {
        run(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            source_fixture(directory.path());
            let fast = Arc::new(Producer::new("deferred-fast"));
            let quality = Arc::new(Producer::new("deferred-quality"));
            let _providers = Providers::install(Arc::clone(&fast), Arc::clone(&quality));
            let mut deferred = runtime(directory.path(), true);
            let root = directory.path().join("store");
            let first = durable(
                deferred
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            );
            let before = files(first.path());
            fast.inputs.lock().unwrap().clear();
            deferred.cli_input.delete_ids = vec!["alpha.md".to_owned(), "beta.md".to_owned()];
            let (next, count) = deferred
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 2);
            let next = durable(next.unwrap());
            let sentinel = FsfsRuntime::read_index_sentinel(next.path())
                .unwrap()
                .unwrap();
            assert_eq!(sentinel.indexed_files, 0);
            assert!(
                !sentinel
                    .reason_codes
                    .iter()
                    .any(|r| r == REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE)
            );
            let mut input = deferred.cli_input.clone();
            input.index_dir = Some(next.path().to_path_buf());
            let candidate = deferred.clone().with_cli_input(input);
            let backlog = read_backlog(&open_catalog(&candidate, false).unwrap()).unwrap();
            assert_eq!(
                (
                    backlog.requested_documents,
                    backlog.queued_documents,
                    backlog.unavailable_documents
                ),
                (0, 0, 0)
            );
            assert_eq!(backlog.generation_root, next.path().display().to_string());
            assert!(fast.inputs.lock().unwrap().is_empty());
            assert!(quality.inputs.lock().unwrap().is_empty());
            assert_eq!(files(first.path()), before);
        });
    }

    #[test]
    fn mixed_delete_keeps_pending_quality_completable_without_sources() {
        run(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let expected = source_fixture(directory.path());
            fs::write(directory.path().join("source/skipped.md"), b"\0binary\0").unwrap();
            let fast = Arc::new(Producer::new("deferred-fast"));
            let quality = Arc::new(Producer::new("deferred-quality"));
            let _providers = Providers::install(Arc::clone(&fast), Arc::clone(&quality));
            let deferred = runtime(directory.path(), true);
            let root = directory.path().join("store");
            let first = durable(
                deferred
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            );
            let first_bytes = files(first.path());
            let first_receipt: serde_json::Value =
                serde_json::from_slice(&fs::read(first.path().join("FSFS-REUSE.json")).unwrap())
                    .unwrap();
            assert!(
                first_receipt["checkpoint"]["content_skipped"]
                    .get("skipped.md")
                    .is_some()
            );
            fast.inputs.lock().unwrap().clear();
            let result = deferred
                .apply_retained_batch(
                    &cx,
                    &root,
                    &first,
                    &[
                        super::super::retained_batch::RetainedMutation::Delete {
                            id: "beta.md".to_owned(),
                        },
                        super::super::retained_batch::RetainedMutation::Delete {
                            id: "skipped.md".to_owned(),
                        },
                    ],
                )
                .await
                .unwrap();
            assert_eq!((result.upserted, result.deleted), (0, 1));
            let next = durable(result.publication.unwrap());
            let next_bytes = files(next.path());
            let next_receipt: serde_json::Value =
                serde_json::from_slice(&fs::read(next.path().join("FSFS-REUSE.json")).unwrap())
                    .unwrap();
            assert!(
                next_receipt["checkpoint"]["content_skipped"]
                    .get("skipped.md")
                    .is_none()
            );
            assert!(quality.inputs.lock().unwrap().is_empty());
            let absent = [super::super::retained_batch::RetainedMutation::Delete {
                id: "absent.md".to_owned(),
            }];
            let noop = deferred
                .apply_retained_batch(&cx, &root, &next, &absent)
                .await
                .unwrap();
            assert!(noop.publication.is_none());
            assert_eq!((noop.upserted, noop.deleted), (0, 0));
            assert!(
                deferred
                    .apply_retained_batch(&cx, &root, &first, &absent)
                    .await
                    .is_err()
            );
            assert_eq!(files(next.path()), next_bytes);
            fs::rename(
                directory.path().join("source"),
                directory.path().join("source-gone"),
            )
            .unwrap();
            let completed = runtime(directory.path(), false)
                .complete_retained_quality(&cx, &root, &next)
                .await
                .unwrap();
            assert_eq!(completed.completed_documents, 1);
            let completed = durable(completed.publication);
            assert_eq!(
                quality.inputs.lock().unwrap().as_slice(),
                std::slice::from_ref(&expected["alpha.md"])
            );
            assert!(fast.inputs.lock().unwrap().is_empty());
            assert_eq!(files(first.path()), first_bytes);
            assert_eq!(files(next.path()), next_bytes);
            assert_eq!(
                fs::read(completed.path().join(FSFS_VECTOR_INDEX_FILE)).unwrap(),
                next_bytes[Path::new(FSFS_VECTOR_INDEX_FILE)]
            );
            assert_eq!(
                VectorIndex::open_read_only(&completed.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE))
                    .unwrap()
                    .live_doc_ids()
                    .unwrap(),
                std::collections::HashSet::from(["alpha.md".to_owned()])
            );
        });
    }

    #[test]
    fn mixed_delete_of_last_pending_documents_clears_quality_debt() {
        run(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            source_fixture(directory.path());
            let fast = Arc::new(Producer::new("deferred-fast"));
            let quality = Arc::new(Producer::new("deferred-quality"));
            let _providers = Providers::install(Arc::clone(&fast), Arc::clone(&quality));
            let deferred = runtime(directory.path(), true);
            let root = directory.path().join("store");
            let first = durable(
                deferred
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            );
            let before = files(first.path());
            fast.inputs.lock().unwrap().clear();
            let operations = ["alpha.md", "beta.md"].map(|id| {
                super::super::retained_batch::RetainedMutation::Delete { id: id.to_owned() }
            });
            let result = deferred
                .apply_retained_batch(&cx, &root, &first, &operations)
                .await
                .unwrap();
            assert_eq!((result.upserted, result.deleted), (0, 2));
            let next = durable(result.publication.unwrap());
            let sentinel = FsfsRuntime::read_index_sentinel(next.path())
                .unwrap()
                .unwrap();
            assert_eq!(sentinel.indexed_files, 0);
            assert!(
                !sentinel
                    .reason_codes
                    .iter()
                    .any(|reason| reason == REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE)
            );
            let mut input = deferred.cli_input.clone();
            input.index_dir = Some(next.path().to_path_buf());
            let candidate = deferred.clone().with_cli_input(input);
            let backlog = read_backlog(&open_catalog(&candidate, false).unwrap()).unwrap();
            assert_eq!(
                (
                    backlog.requested_documents,
                    backlog.queued_documents,
                    backlog.unavailable_documents
                ),
                (0, 0, 0)
            );
            assert_eq!(backlog.generation_root, next.path().display().to_string());
            assert!(fast.inputs.lock().unwrap().is_empty());
            assert!(quality.inputs.lock().unwrap().is_empty());
            assert_eq!(files(first.path()), before);
        });
    }

    #[test]
    fn empty_backlog_preserves_producer_intent_through_compaction_and_cli_upserts() {
        use super::super::retained_batch::RetainedMutation;

        for owned_inputs in [false, true] {
            run(|cx| async move {
                let directory = tempfile::tempdir().unwrap();
                source_fixture(directory.path());
                let fast = Arc::new(Producer::new("deferred-fast"));
                let quality = Arc::new(Producer::new("deferred-quality"));
                let _providers = Providers::install(Arc::clone(&fast), Arc::clone(&quality));
                let deferred = runtime(directory.path(), true);
                let cleared = runtime(directory.path(), false);
                let root = directory.path().join("store");
                let first = durable(
                    deferred
                        .rebuild_retained_generation(&cx, &root)
                        .await
                        .unwrap(),
                );
                let before = files(first.path());
                let mut history = vec![(first, before)];
                if owned_inputs {
                    let result = deferred
                        .apply_retained_batch(
                            &cx,
                            &root,
                            &history.last().unwrap().0,
                            &[RetainedMutation::Upsert {
                                id: "alpha.md".to_owned(),
                                text: "replacement initial alpha".to_owned(),
                            }],
                        )
                        .await
                        .unwrap();
                    let next = durable(result.publication.unwrap());
                    let before = files(next.path());
                    history.push((next, before));
                }
                fs::rename(
                    directory.path().join("source"),
                    directory.path().join("source-gone"),
                )
                .unwrap();
                fast.inputs.lock().unwrap().clear();
                let operations = ["alpha.md", "beta.md"]
                    .map(|id| RetainedMutation::Delete { id: id.to_owned() });
                let result = deferred
                    .apply_retained_batch(&cx, &root, &history.last().unwrap().0, &operations)
                    .await
                    .unwrap();
                let empty = durable(result.publication.unwrap());
                let before = files(empty.path());
                history.push((empty, before));
                let (compacted, _) = cleared
                    .compact_retained_generation(&cx, &root)
                    .await
                    .unwrap();
                if let Some(publication) = compacted {
                    let empty = durable(publication);
                    let before = files(empty.path());
                    history.push((empty, before));
                }
                let sentinel = FsfsRuntime::read_index_sentinel(history.last().unwrap().0.path())
                    .unwrap()
                    .unwrap();
                assert_eq!(sentinel.indexed_files, 0);
                assert!(
                    !sentinel
                        .reason_codes
                        .iter()
                        .any(|reason| reason == REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE)
                );
                assert!(fast.inputs.lock().unwrap().is_empty());
                assert!(quality.inputs.lock().unwrap().is_empty());

                // Empty membership must retain the original producer rather
                // than borrowing the caller's newly available model identity.
                let foreign = Arc::new(Producer::new("foreign-empty-quality"));
                set_test_quality_embedder(Some(foreign.clone()));
                for (forced, id, text) in [
                    (true, "gamma.md", "intermediate owned gamma"),
                    (false, "delta.md", "latest owned delta"),
                ] {
                    let input = directory.path().join(format!("{id}.jsonl"));
                    fs::write(
                        &input,
                        serde_json::json!({"id": id, "text": text}).to_string(),
                    )
                    .unwrap();
                    let mut caller = runtime(directory.path(), forced);
                    caller.cli_input.command = CliCommand::AppendBatch;
                    caller.cli_input.input_file = Some(input);
                    let (publication, count) =
                        caller.append_retained_generation(&cx, &root).await.unwrap();
                    assert_eq!(count, 1);
                    let next = durable(publication.unwrap());
                    let sentinel = FsfsRuntime::read_index_sentinel(next.path())
                        .unwrap()
                        .unwrap();
                    assert!(
                        sentinel
                            .reason_codes
                            .iter()
                            .any(|reason| reason == REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE)
                    );
                    assert!(!next.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE).exists());
                    let before = files(next.path());
                    history.push((next, before));
                    if forced {
                        let result = cleared
                            .apply_retained_batch(
                                &cx,
                                &root,
                                &history.last().unwrap().0,
                                &[RetainedMutation::Delete { id: id.to_owned() }],
                            )
                            .await
                            .unwrap();
                        let empty = durable(result.publication.unwrap());
                        let before = files(empty.path());
                        history.push((empty, before));
                        let (compacted, _) = cleared
                            .compact_retained_generation(&cx, &root)
                            .await
                            .unwrap();
                        if let Some(publication) = compacted {
                            let empty = durable(publication);
                            let before = files(empty.path());
                            history.push((empty, before));
                        }
                    }
                }
                assert_eq!(fast.inputs.lock().unwrap().len(), 2);
                assert!(quality.inputs.lock().unwrap().is_empty());
                let selected = &history.last().unwrap().0;
                let error = runtime(directory.path(), false)
                    .complete_retained_quality(&cx, &root, selected)
                    .await
                    .unwrap_err();
                assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                    if field == "index.complete_quality"));
                assert!(foreign.inputs.lock().unwrap().is_empty());
                set_test_quality_embedder(Some(quality.clone()));
                let fast_before = fs::read(selected.path().join(FSFS_VECTOR_INDEX_FILE)).unwrap();
                let completed = runtime(directory.path(), false)
                    .complete_retained_quality(&cx, &root, selected)
                    .await
                    .unwrap();
                assert_eq!(completed.completed_documents, 1);
                let completed = durable(completed.publication);
                assert_eq!(
                    fs::read(completed.path().join(FSFS_VECTOR_INDEX_FILE)).unwrap(),
                    fast_before
                );
                assert_eq!(
                    quality.inputs.lock().unwrap().as_slice(),
                    &["latest owned delta"]
                );
                assert_eq!(
                    VectorIndex::open_read_only(
                        &completed.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE)
                    )
                    .unwrap()
                    .live_doc_ids()
                    .unwrap(),
                    std::collections::HashSet::from(["delta.md".to_owned()])
                );
                for (generation, before) in &history {
                    assert_eq!(files(generation.path()), *before);
                }
            });
        }
    }

    #[test]
    fn initially_empty_deferred_index_retains_quality_intent_for_cli_append() {
        for (compact, forced_append) in [(false, false), (false, true), (true, false), (true, true)]
        {
            run(|cx| async move {
                let directory = tempfile::tempdir().unwrap();
                fs::create_dir(directory.path().join("source")).unwrap();
                let fast = Arc::new(Producer::new("initial-empty-fast"));
                let quality = Arc::new(Producer::new("initial-empty-quality"));
                let _providers = Providers::install(Arc::clone(&fast), Arc::clone(&quality));
                let deferred = runtime(directory.path(), true);
                let cleared = runtime(directory.path(), false);
                let root = directory.path().join("store");
                let first = durable(
                    deferred
                        .rebuild_retained_generation(&cx, &root)
                        .await
                        .unwrap(),
                );
                let sentinel = FsfsRuntime::read_index_sentinel(first.path())
                    .unwrap()
                    .unwrap();
                assert_eq!(sentinel.indexed_files, 0);
                assert!(!first.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE).exists());
                let before = files(first.path());
                let mut history = vec![(first, before)];
                fs::rename(
                    directory.path().join("source"),
                    directory.path().join("source-gone"),
                )
                .unwrap();
                if compact {
                    let (publication, _) = cleared
                        .compact_retained_generation(&cx, &root)
                        .await
                        .unwrap();
                    if let Some(publication) = publication {
                        let next = durable(publication);
                        let before = files(next.path());
                        history.push((next, before));
                    }
                }
                fast.inputs.lock().unwrap().clear();
                assert!(quality.inputs.lock().unwrap().is_empty());
                let input = directory.path().join("append.jsonl");
                fs::write(
                    &input,
                    serde_json::json!({"id": "gamma.md", "text": "latest initial-empty gamma"})
                        .to_string(),
                )
                .unwrap();
                let mut caller = runtime(directory.path(), forced_append);
                caller.cli_input.command = CliCommand::AppendBatch;
                caller.cli_input.input_file = Some(input);
                let (publication, count) =
                    caller.append_retained_generation(&cx, &root).await.unwrap();
                assert_eq!(count, 1);
                let appended = durable(publication.unwrap());
                let sentinel = FsfsRuntime::read_index_sentinel(appended.path())
                    .unwrap()
                    .unwrap();
                assert_eq!(sentinel.indexed_files, 1);
                assert!(
                    sentinel
                        .reason_codes
                        .iter()
                        .any(|reason| reason == REASON_VECTOR_QUALITY_TIER_DEFERRED_PRESSURE)
                );
                assert!(
                    !appended
                        .path()
                        .join(FSFS_VECTOR_QUALITY_INDEX_FILE)
                        .exists()
                );
                assert_eq!(fast.inputs.lock().unwrap().len(), 1);
                assert!(quality.inputs.lock().unwrap().is_empty());
                let fast_before = fs::read(appended.path().join(FSFS_VECTOR_INDEX_FILE)).unwrap();
                let before = files(appended.path());
                history.push((appended, before));
                let selected = &history.last().unwrap().0;
                let foreign = Arc::new(Producer::new("foreign-initial-empty-quality"));
                set_test_quality_embedder(Some(foreign.clone()));
                let error = runtime(directory.path(), false)
                    .complete_retained_quality(&cx, &root, selected)
                    .await
                    .unwrap_err();
                assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                    if field == "index.complete_quality"));
                assert!(foreign.inputs.lock().unwrap().is_empty());
                set_test_quality_embedder(Some(quality.clone()));
                let completed = runtime(directory.path(), false)
                    .complete_retained_quality(&cx, &root, selected)
                    .await
                    .unwrap();
                assert_eq!(completed.completed_documents, 1);
                let completed = durable(completed.publication);
                assert_eq!(
                    fs::read(completed.path().join(FSFS_VECTOR_INDEX_FILE)).unwrap(),
                    fast_before
                );
                assert_eq!(
                    quality.inputs.lock().unwrap().as_slice(),
                    &["latest initial-empty gamma"]
                );
                assert_eq!(
                    VectorIndex::open_read_only(
                        &completed.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE)
                    )
                    .unwrap()
                    .live_doc_ids()
                    .unwrap(),
                    std::collections::HashSet::from(["gamma.md".to_owned()])
                );
                for (generation, before) in &history {
                    assert_eq!(files(generation.path()), *before);
                }
            });
        }
    }

    #[test]
    fn maintenance_refuses_corrupt_pending_fences_without_changing_selection() {
        for corruption in [
            "generation",
            "source",
            "status",
            "source_digest",
            "projection",
        ] {
            run(|cx| async move {
                let directory = tempfile::tempdir().unwrap();
                source_fixture(directory.path());
                let fast = Arc::new(Producer::new("deferred-fast"));
                let quality = Arc::new(Producer::new("deferred-quality"));
                let _providers = Providers::install(fast, Arc::clone(&quality));
                let deferred = runtime(directory.path(), true);
                let root = directory.path().join("store");
                let first = durable(
                    deferred
                        .rebuild_retained_generation(&cx, &root)
                        .await
                        .unwrap(),
                );
                let before = files(first.path());
                let store = CompleteGenerationStore::open(&cx, &root).unwrap();
                let build = store.begin(&cx).unwrap();
                let mut input = deferred.cli_input.clone();
                input.index_dir = Some(build.path().to_path_buf());
                let candidate = deferred.clone().with_cli_input(input);
                retained_reuse::copy_selected_generation(&cx, &candidate, &store, build.path())
                    .unwrap();
                let storage = open_catalog(&candidate, false).unwrap();
                let mut backlog = read_backlog(&storage).unwrap();
                match corruption {
                    "generation" => {
                        backlog.generation_root.push_str("-foreign");
                        record_backlog(&storage, &backlog, 0).unwrap();
                    }
                    "source" => {
                        "0123456789abcdef".clone_into(&mut backlog.source_hash);
                        record_backlog(&storage, &backlog, 0).unwrap();
                    }
                    "status" => {
                        storage
                            .connection()
                            .execute_sync("UPDATE embedding_jobs SET status = 'processing';")
                            .unwrap();
                    }
                    "source_digest" => {
                        let mut document = storage.get_document("alpha.md").unwrap().unwrap();
                        document.metadata.as_mut().unwrap()[SOURCE_METADATA]["source_sha256"] =
                            serde_json::json!("cd".repeat(32));
                        storage.upsert_document(&document).unwrap();
                    }
                    "projection" => {}
                    _ => unreachable!(),
                }
                drop(storage);
                assert!(
                    carry_after_maintenance(&cx, &candidate, &first, build.path(), || {
                        if corruption == "projection" {
                            Ok(None)
                        } else {
                            super::super::retained_batch::retain_unchanged_inputs(
                                &cx,
                                &candidate,
                                &first,
                                build.path(),
                            )
                        }
                    })
                    .is_err(),
                    "{corruption}"
                );
                assert_eq!(store.active(&cx).unwrap(), Some(first.clone()));
                assert_eq!(files(first.path()), before);
                assert!(quality.inputs.lock().unwrap().is_empty());
            });
        }
    }

    #[test]
    fn cancelled_or_foreign_inference_preserves_pending_generation_for_retry() {
        for outcome in [1, 2, 3] {
            run(|cx| async move {
                let directory = tempfile::tempdir().unwrap();
                source_fixture(directory.path());
                let fast = Arc::new(Producer::new("deferred-fast"));
                let quality = Arc::new(Producer::new("deferred-quality"));
                let _providers = Providers::install(fast, Arc::clone(&quality));
                let root = directory.path().join("store");
                let first = durable(
                    runtime(directory.path(), true)
                        .rebuild_retained_generation(&cx, &root)
                        .await
                        .unwrap(),
                );
                let before = files(first.path());
                quality.outcome.store(outcome, Ordering::SeqCst);
                let failed = runtime(directory.path(), false)
                    .complete_retained_quality(&cx, &root, &first)
                    .await;
                cx.set_cancel_requested(false);
                assert!(failed.is_err());
                assert_eq!(
                    CompleteGenerationStore::open(&cx, &root)
                        .unwrap()
                        .active(&cx)
                        .unwrap()
                        .unwrap(),
                    first
                );
                assert_eq!(files(first.path()), before);
                quality.outcome.store(0, Ordering::SeqCst);
                let retried = runtime(directory.path(), false)
                    .complete_retained_quality(&cx, &root, &first)
                    .await
                    .unwrap();
                assert_eq!(retried.completed_documents, 2);
                assert_ne!(durable(retried.publication), first);
            });
        }
    }

    #[test]
    fn wrong_selected_producer_and_replaced_generation_refuse_before_inference() {
        run(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            source_fixture(directory.path());
            let fast = Arc::new(Producer::new("deferred-fast"));
            let quality = Arc::new(Producer::new("deferred-quality"));
            let _providers = Providers::install(fast, Arc::clone(&quality));
            let root = directory.path().join("store");
            let first = durable(
                runtime(directory.path(), true)
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            );
            let foreign = Arc::new(Producer::new("foreign-producer"));
            set_test_quality_embedder(Some(foreign.clone()));
            let error = runtime(directory.path(), false)
                .complete_retained_quality(&cx, &root, &first)
                .await
                .expect_err("foreign producer cannot complete the selected backlog");
            assert!(error.to_string().contains("producer changed"), "{error}");
            assert!(foreign.inputs.lock().unwrap().is_empty());
            set_test_quality_embedder(Some(quality.clone()));
            fs::write(
                directory.path().join("source/alpha.md"),
                "latest sharedtoken document revision",
            )
            .unwrap();
            let second = durable(
                runtime(directory.path(), true)
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
            );
            let before = files(second.path());
            assert!(
                runtime(directory.path(), false)
                    .complete_retained_quality(&cx, &root, &first)
                    .await
                    .is_err()
            );
            assert!(quality.inputs.lock().unwrap().is_empty());
            assert_eq!(files(second.path()), before);
            let completed = runtime(directory.path(), false)
                .complete_retained_quality(&cx, &root, &second)
                .await
                .unwrap();
            assert_eq!(completed.completed_documents, 2);
            assert!(
                quality
                    .inputs
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|text| text == "latest sharedtoken document revision")
            );
        });
    }

    const CHILD_ROOT: &str = "FSFS_DEFERRED_QUALITY_CHILD_ROOT";
    const CHILD_PHASE: &str = "FSFS_DEFERRED_QUALITY_CHILD_PHASE";

    #[test]
    fn fresh_process_child() {
        let Some(parent) = std::env::var_os(CHILD_ROOT).map(PathBuf::from) else {
            return;
        };
        let phase = std::env::var(CHILD_PHASE).unwrap();
        run(|cx| async move {
            let fast = Arc::new(Producer::new("deferred-fast"));
            let quality = Arc::new(Producer::new("deferred-quality"));
            let _providers = Providers::install(Arc::clone(&fast), Arc::clone(&quality));
            let root = parent.join("store");
            let selected = if phase == "defer" {
                durable(
                    runtime(&parent, true)
                        .rebuild_retained_generation(&cx, &root)
                        .await
                        .unwrap(),
                )
            } else {
                let first = CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap()
                    .unwrap();
                let result = runtime(&parent, false)
                    .complete_retained_quality(&cx, &root, &first)
                    .await
                    .unwrap();
                assert_eq!(result.completed_documents, 2);
                assert!(fast.inputs.lock().unwrap().is_empty());
                durable(result.publication)
            };
            fs::write(
                parent.join(format!("{phase}.json")),
                serde_json::to_vec(&serde_json::json!({
                    "selected": selected.id(),
                    "inputs": *quality.inputs.lock().unwrap(),
                }))
                .unwrap(),
            )
            .unwrap();
        });
    }

    #[test]
    fn pending_quality_survives_a_real_process_exit_and_missing_source_tree() {
        let directory = tempfile::tempdir().unwrap();
        let mut expected = source_fixture(directory.path())
            .into_values()
            .collect::<Vec<_>>();
        expected.sort();
        let executable = std::env::current_exe().unwrap();
        for phase in ["defer", "complete"] {
            if phase == "complete" {
                fs::rename(
                    directory.path().join("source"),
                    directory.path().join("source-gone"),
                )
                .unwrap();
            }
            let output = Command::new(&executable)
                .args([
                    "--exact",
                    "runtime::deferred_quality::tests::fresh_process_child",
                    "--nocapture",
                ])
                .env(CHILD_ROOT, directory.path())
                .env(CHILD_PHASE, phase)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{phase} child failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let deferred: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.path().join("defer.json")).unwrap())
                .unwrap();
        let completed: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.path().join("complete.json")).unwrap())
                .unwrap();
        assert_eq!(deferred["inputs"], serde_json::json!([]));
        let mut actual: Vec<String> = serde_json::from_value(completed["inputs"].clone()).unwrap();
        actual.sort();
        assert_eq!(actual, expected);
        assert_ne!(deferred["selected"], completed["selected"]);
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn corrupt_job_input_and_source_fences_are_refused_before_inference() {
        for corruption in [
            "job_digest",
            "source_revision",
            "retained_text",
            "status",
            "source_digest",
            "checkpoint_digest",
            "checkpoint_revision",
            "missing_checkpoint",
        ] {
            let storage = Arc::new(Storage::open_in_memory().unwrap());
            let text = "sharedtoken exact quality input";
            let hash = ContentHasher::hash(text);
            let length = text.chars().count();
            let mut document = DocumentRecord::new("alpha.md", text, hash, length, 7, 7);
            document.metadata = Some(serde_json::json!({
                SOURCE_METADATA: { "revision": 7, "source_sha256": "ab".repeat(32) },
            }));
            storage.upsert_document(&document).unwrap();
            storage.set_document_content("alpha.md", text).unwrap();
            let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
            queue
                .enqueue("alpha.md", "generation-quality", &document.content_hash, 0)
                .unwrap();
            let mut manifests = BTreeMap::from([(
                "alpha.md".to_owned(),
                IndexManifestEntry {
                    file_key: "alpha.md".to_owned(),
                    revision: 7,
                    ingestion_class: ingestion_class_label(IngestionClass::FullSemanticLexical)
                        .to_owned(),
                    canonical_bytes: text.len() as u64,
                    reason_code: "fixture".to_owned(),
                    fast_windows: None,
                },
            )]);
            let mut checkpoints = BTreeMap::from([(
                "alpha.md".to_owned(),
                CheckpointFileEntry {
                    revision: 7,
                    ingestion_class: ingestion_class_label(IngestionClass::FullSemanticLexical)
                        .to_owned(),
                    canonical_bytes: text.len() as u64,
                    reason_code: "fixture".to_owned(),
                    lexical_indexed: true,
                    semantic_indexed: true,
                    content_hash_hex: "ab".repeat(32),
                    fast_windows: None,
                    canonical_lines: None,
                    stamp: None,
                },
            )]);
            validate_pending_jobs(
                &storage,
                "generation-quality",
                &manifests,
                Some(&checkpoints),
            )
            .unwrap();
            match corruption {
                "job_digest" => {
                    storage
                        .connection()
                        .execute_with_params_sync(
                            "UPDATE embedding_jobs SET content_hash = ?1;",
                            &[SqliteValue::Blob(vec![9; 32].into())],
                        )
                        .unwrap();
                }
                "source_revision" => manifests.get_mut("alpha.md").unwrap().revision = 8,
                "source_digest" => {
                    document.metadata.as_mut().unwrap()[SOURCE_METADATA]["source_sha256"] =
                        serde_json::json!("cd".repeat(32));
                    storage.upsert_document(&document).unwrap();
                }
                "checkpoint_digest" => {
                    checkpoints.get_mut("alpha.md").unwrap().content_hash_hex = "cd".repeat(32);
                }
                "checkpoint_revision" => checkpoints.get_mut("alpha.md").unwrap().revision = 8,
                "missing_checkpoint" => checkpoints.clear(),
                "retained_text" => {
                    storage
                        .connection()
                        .execute_with_params_sync(
                            "UPDATE document_contents SET canonical_text = ?1;",
                            &[SqliteValue::Text("corrupt body".to_owned().into())],
                        )
                        .unwrap();
                }
                "status" => {
                    storage
                        .connection()
                        .execute_sync("UPDATE embedding_jobs SET status = 'processing';")
                        .unwrap();
                }
                _ => unreachable!(),
            }
            assert!(
                validate_pending_jobs(
                    &storage,
                    "generation-quality",
                    &manifests,
                    Some(&checkpoints)
                )
                .is_err(),
                "{corruption}"
            );
        }
    }

    #[test]
    fn capture_skips_legacy_and_metadata_only_work_without_creating_a_catalog() {
        let directory = tempfile::tempdir().unwrap();
        let cx = Cx::for_testing();
        let text = "sharedtoken exact retained input";
        let pending = PendingIndexDocument {
            ingestion_class: IngestionClass::FullSemanticLexical,
            file_key: "alpha.md".to_owned(),
            ingestion_class_label: ingestion_class_label(IngestionClass::FullSemanticLexical)
                .to_owned(),
            canonical_bytes: text.len() as u64,
            canonical_lines: 1,
            revision: 7,
            reason_code: "fixture".to_owned(),
            content_hash_hex: content_sha256_hex(text.as_bytes()),
            lexical_required: true,
            semantic_reused: false,
            fast_windows: None,
            document: frankensearch_core::IndexableDocument::new("alpha.md", text),
            lexical_text: text.to_owned(),
            stamp: None,
        };
        let legacy = runtime(directory.path(), true);
        capture_index_batch(&cx, &legacy, std::slice::from_ref(&pending)).unwrap();
        assert!(!directory.path().join("store").exists());

        let mut retained = legacy.clone();
        retained.discovery_excluded_root = Some(directory.path().join("store"));
        let mut path_only = pending.clone();
        path_only.ingestion_class = IngestionClass::MetadataOnly;
        path_only.document.content = "alpha.md".to_owned();
        capture_index_batch(&cx, &retained, &[path_only]).unwrap();
        assert!(!directory.path().join("store").exists());

        let mut metadata_config = retained.config.clone();
        metadata_config.pressure.degradation_override = DegradationOverrideMode::ForceMetadataOnly;
        let mut metadata =
            FsfsRuntime::new(metadata_config).with_cli_input(retained.cli_input.clone());
        metadata
            .discovery_excluded_root
            .clone_from(&retained.discovery_excluded_root);
        capture_index_batch(&cx, &metadata, std::slice::from_ref(&pending)).unwrap();
        assert!(!directory.path().join("store").exists());

        // The same full input in an owned retained build must persist: an
        // omitted capture consumer cannot pass only the negative controls.
        capture_index_batch(&cx, &retained, &[pending]).unwrap();
        let storage = open_catalog(&retained, false).unwrap();
        assert_eq!(
            storage.get_document_content("alpha.md").unwrap().as_deref(),
            Some(text)
        );
    }
}
