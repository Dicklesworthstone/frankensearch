//! Reuse proven indexing work in a fresh complete generation.
//!
//! The ordinary indexer already revalidates source hashes and both producers
//! when resuming a checkpoint. Keep its observed inputs, copy the predecessor
//! to independent files, and feed that existing path rather than introduce a
//! second embedding cache. No serving artifact is opened for writing.
//!
//! Receipts bind every completed input, including the final batch, to the
//! running executable bytes and configuration that produced it on Linux. A
//! fresh Linux process can reuse that work only after the immutable selected
//! bundle is verified; other platforms retain process-scoped reuse. This
//! Mutable legacy roots use the same receipt compatibility rule, with a digest
//! binding the evidence to their actual serving artifacts under the publication
//! lease. They still pass through ordinary indexing and publication admission.

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    CheckpointFileEntry, CliCommand, FSFS_CHECKPOINT_FILE, FsfsIndexPayload, FsfsRuntime,
    INDEXING_CHECKPOINT_SCHEMA_VERSION, IndexingCheckpoint, SearchExecutionMode,
    content_sha256_hex, retained_search_checkpoint, write_indexing_checkpoint,
};
use crate::generation_store::{
    COMPLETE_GENERATION_MANIFEST, CompleteGenerationStore, PublishedGeneration,
};

// This file is itself loaded through `#[path]`, so an unannotated child
// module would resolve beside it in `runtime/`, not in `runtime/retained_reuse/`.
#[path = "retained_reuse/append_input.rs"]
mod append_input;
pub(super) use append_input::read_append_documents;
#[path = "retained_reuse/execution.rs"]
mod execution;

const RECEIPT_FILE: &str = "FSFS-REUSE.json";
const RECEIPT_VERSION: u16 = 2;
const LEGACY_RECEIPT_FILE: &str = "FSFS-LEGACY-REUSE.json";
const LEGACY_RECEIPT_VERSION: u16 = 1;
const MAX_RECEIPT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_COPY_DEPTH: usize = 64;
const MAX_COPY_ENTRIES: usize = 200_000;
const MAX_LEGACY_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024 * 1024;
static SESSION: OnceLock<String> = OnceLock::new();

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReuseReceipt {
    version: u16,
    session: String,
    // Optional only so existing version-1 receipts remain readable and become
    // cold misses. Absence never grants cross-process reuse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    executable_sha256: Option<String>,
    configuration_sha256: String,
    checkpoint: IndexingCheckpoint,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyReuseReceipt {
    version: u16,
    receipt: ReuseReceipt,
    // Bind the receipt itself as well as the artifacts: changing a content
    // hash in otherwise valid JSON must not authorize stale vector reuse.
    state_sha256: String,
}

fn reuse_error(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "complete_generation.reuse".to_owned(),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

fn session_id() -> SearchResult<&'static str> {
    if let Some(session) = SESSION.get() {
        return Ok(session);
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| reuse_error("cannot establish indexing session identity"))?;
    // A cooperative-process identity, not an authentication credential.
    let session = format!("{}-{:032x}", std::process::id(), now.as_nanos());
    Ok(SESSION.get_or_init(|| session))
}

fn configuration_digest(runtime: &FsfsRuntime) -> SearchResult<String> {
    // Persist a digest, not the configuration (which can contain private paths
    // or provider settings). A conservative mismatch simply starts cold.
    let bytes =
        serde_json::to_vec(runtime.config()).map_err(|source| SearchError::SubsystemError {
            subsystem: "fsfs.complete_generation.reuse_config",
            source: Box::new(source),
        })?;
    Ok(content_sha256_hex(&bytes))
}

fn open_regular(path: &Path) -> SearchResult<File> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(reuse_error(
            "reuse input is not a regular, non-symlink file",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(reuse_error("reuse input changed its filesystem type"));
    }
    Ok(file)
}

fn read_json<T: serde::de::DeserializeOwned>(cx: &Cx, path: &Path) -> SearchResult<T> {
    retained_search_checkpoint(cx)?;
    let file = open_regular(path)?;
    if file.metadata()?.len() > MAX_RECEIPT_BYTES {
        return Err(reuse_error("reuse input exceeds the 64 MiB limit"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_RECEIPT_BYTES + 1).read_to_end(&mut bytes)?;
    retained_search_checkpoint(cx)?;
    if bytes.len() as u64 > MAX_RECEIPT_BYTES {
        return Err(reuse_error("reuse input grew beyond the 64 MiB limit"));
    }
    serde_json::from_slice(&bytes).map_err(|source| SearchError::SubsystemError {
        subsystem: "fsfs.complete_generation.reuse_receipt",
        source: Box::new(source),
    })
}

fn proven(entry: &CheckpointFileEntry) -> bool {
    entry.lexical_indexed
        && entry.semantic_indexed
        && entry.content_hash_hex.len() == 64
        && entry
            .content_hash_hex
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
}

fn compatible_execution(receipt: &ReuseReceipt, executable: Option<&str>, session: &str) -> bool {
    match (receipt.executable_sha256.as_deref(), executable) {
        (Some(expected), Some(actual)) => {
            expected.len() == 64
                && expected.bytes().all(|byte| byte.is_ascii_hexdigit())
                && expected == actual
        }
        (None, None) => receipt.session == session,
        // A failed or newly available proof is not permission to replace the
        // receipt's scope with a weaker one, even inside the same process.
        _ => false,
    }
}

fn compatible_receipt(runtime: &FsfsRuntime, receipt: &ReuseReceipt) -> SearchResult<bool> {
    Ok(receipt.version == RECEIPT_VERSION
        && compatible_execution(receipt, execution::fingerprint(), session_id()?)
        && receipt.configuration_sha256 == configuration_digest(runtime)?
        && receipt.checkpoint.target_root == runtime.resolve_target_root()?.display().to_string())
}

impl FsfsRuntime {
    /// Execute the original indexer with an optional independently copied seed.
    /// The outer retained builder still owns model admission, source precommit,
    /// sealing, publication and cancellation. A failed candidate stays inert.
    pub(super) async fn run_retained_index_with_reuse(
        &self,
        cx: &Cx,
        store: &CompleteGenerationStore,
        candidate_root: &Path,
    ) -> SearchResult<()> {
        execution::prepare(cx).await?;
        seed_candidate(cx, self, store, candidate_root)?;
        let payload = Box::pin(self.run_one_shot_index_scaffold_internal(
            cx,
            CliCommand::Index,
            |_| Ok(()),
            false,
            false,
        ))
        .await?;
        Self::validate_search_generation_at_root(candidate_root, SearchExecutionMode::Full)?;
        let receipt = completed_receipt(self, candidate_root, &payload)?;
        write_receipt(cx, candidate_root, &receipt)?;
        Ok(())
    }
}

fn completed_receipt(
    runtime: &FsfsRuntime,
    root: &Path,
    payload: &FsfsIndexPayload,
) -> SearchResult<ReuseReceipt> {
    let checkpoint = &payload.input_checkpoint;
    let final_state = &payload.generation;
    if !final_state.generation_complete
        || !checkpoint.artifacts_durable
        || checkpoint.schema_version != INDEXING_CHECKPOINT_SCHEMA_VERSION
        || checkpoint.target_root != final_state.target_root
        || checkpoint.index_root != final_state.index_root
        || checkpoint.index_root != root.display().to_string()
        || checkpoint.embedder_id != payload.vector_generation.id
        || checkpoint.embedder_dimension != payload.vector_generation.dimension
    {
        return Err(reuse_error(
            "observed checkpoint does not belong to the completed candidate",
        ));
    }
    let manifests = FsfsRuntime::read_matching_manifest_generation(root)?
        .ok_or_else(|| reuse_error("completed candidate has no matching manifests"))?;
    if manifests.len() != checkpoint.files.len()
        || manifests.iter().any(|(key, manifest)| {
            !checkpoint.files.get(key).is_some_and(|entry| {
                entry.revision == manifest.revision
                    && entry.ingestion_class == manifest.ingestion_class
                    && entry.canonical_bytes == manifest.canonical_bytes
                    && entry.reason_code == manifest.reason_code
                    && entry.content_hash_hex.len() == 64
                    && entry
                        .content_hash_hex
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit())
            })
        })
    {
        return Err(reuse_error(
            "completed input evidence does not cover its exact manifest",
        ));
    }
    if FsfsRuntime::read_checkpoint_manifest_generation(root, checkpoint)?.is_none() {
        return Err(reuse_error(
            "retained input evidence disagrees with final generation metadata",
        ));
    }
    Ok(ReuseReceipt {
        version: RECEIPT_VERSION,
        session: session_id()?.to_owned(),
        executable_sha256: execution::fingerprint().map(str::to_owned),
        configuration_sha256: configuration_digest(runtime)?,
        checkpoint: checkpoint.clone(),
    })
}

pub(super) async fn prepare_legacy_reuse(cx: &Cx) -> SearchResult<()> {
    execution::prepare(cx).await
}

/// Read only while holding the legacy publication lease, before any artifact
/// is opened for writing. An interruption checkpoint always wins, including
/// an unreadable one; old completion evidence must never resurrect its rows.
pub(super) fn legacy_checkpoint(
    cx: &Cx,
    runtime: &FsfsRuntime,
    root: &Path,
) -> SearchResult<Option<IndexingCheckpoint>> {
    let read = || -> SearchResult<Option<IndexingCheckpoint>> {
        retained_search_checkpoint(cx)?;
        if runtime.cli_input.full_reindex {
            return Ok(None);
        }
        for excluded in [FSFS_CHECKPOINT_FILE, COMPLETE_GENERATION_MANIFEST] {
            match fs::symlink_metadata(root.join(excluded)) {
                Ok(_) => return Ok(None),
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        let path = root.join(LEGACY_RECEIPT_FILE);
        match fs::symlink_metadata(&path) {
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
            Ok(_) => {}
        }
        let evidence: LegacyReuseReceipt = read_json(cx, &path)?;
        let receipt = &evidence.receipt;
        if evidence.version != LEGACY_RECEIPT_VERSION
            || !compatible_receipt(runtime, receipt)?
            || receipt.checkpoint.index_root != root.display().to_string()
            || receipt.checkpoint.schema_version != INDEXING_CHECKPOINT_SCHEMA_VERSION
            || !receipt.checkpoint.artifacts_durable
            || super::checkpoint_has_deferred_semantic_rows(&receipt.checkpoint)
            || legacy_state_digest(cx, root, receipt)? != evidence.state_sha256
        {
            return Ok(None);
        }
        let Some(sentinel) = FsfsRuntime::read_index_sentinel(root)? else {
            return Ok(None);
        };
        if !sentinel.generation_complete
            || FsfsRuntime::read_checkpoint_manifest_generation(root, &receipt.checkpoint)?
                .is_none()
        {
            return Ok(None);
        }
        FsfsRuntime::validate_search_generation_at_root(root, SearchExecutionMode::Full)?;
        retained_search_checkpoint(cx)?;
        tracing::info!(
            covered_files = receipt.checkpoint.files.len(),
            "admitted completed legacy inputs; source hashes and active producers decide reuse"
        );
        Ok(Some(evidence.receipt.checkpoint))
    };
    match read() {
        Err(error @ SearchError::Cancelled { .. }) => Err(error),
        Err(error) => {
            tracing::warn!(%error, "completed legacy reuse evidence is unavailable; indexing starts cold");
            Ok(None)
        }
        result => result,
    }
}

/// Optimization evidence is published after successful generation admission,
/// with no live checkpoint left behind. Failure to save it is a cold next run,
/// not a reason to retract an already completed generation.
pub(super) fn retain_legacy_checkpoint(
    cx: &Cx,
    runtime: &FsfsRuntime,
    root: &Path,
    payload: &FsfsIndexPayload,
) -> SearchResult<()> {
    let write = || -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        crate::generation_store::reject_published_write(root)?;
        FsfsRuntime::validate_search_generation_at_root(root, SearchExecutionMode::Full)?;
        let receipt = completed_receipt(runtime, root, payload)?;
        let state_sha256 = legacy_state_digest(cx, root, &receipt)?;
        let evidence = LegacyReuseReceipt {
            version: LEGACY_RECEIPT_VERSION,
            receipt,
            state_sha256,
        };
        let bytes =
            serde_json::to_vec(&evidence).map_err(|source| SearchError::SubsystemError {
                subsystem: "fsfs.index.completed_reuse",
                source: Box::new(source),
            })?;
        if bytes.len() as u64 > MAX_RECEIPT_BYTES {
            return Err(reuse_error(
                "completed legacy input evidence exceeds its byte limit",
            ));
        }
        retained_search_checkpoint(cx)?;
        super::write_durable(root.join(LEGACY_RECEIPT_FILE), bytes)?;
        Ok(())
    };
    match write() {
        Err(error @ SearchError::Cancelled { .. }) => Err(error),
        Err(error) => {
            tracing::warn!(%error, "could not retain completed legacy inputs; next indexing run starts cold");
            Ok(())
        }
        result => result,
    }
}

/// An optimization witness for a cooperative mutable root, not an immutable
/// generation seal. Include all selected lexical and vector bytes (including
/// WALs), membership manifests, CURRENT pointers, and the completion sentinel.
/// Cache/catalog/explain files cannot affect checkpoint reuse and are omitted.
fn legacy_state_digest(cx: &Cx, root: &Path, receipt: &ReuseReceipt) -> SearchResult<String> {
    retained_search_checkpoint(cx)?;
    if !fs::symlink_metadata(root)?.file_type().is_dir() {
        return Err(reuse_error(
            "legacy index root is not a non-symlink directory",
        ));
    }
    // This inspection path never adopts an orphan directory or repairs CURRENT.
    let layout = FsfsRuntime::resolve_sealed_lexical_engine(root)?;
    if layout.engine() != Some(super::BlueGreenEngine::Quill) {
        return Err(reuse_error(
            "completed legacy evidence requires a Quill generation",
        ));
    }
    let lexical = layout
        .engine_dir()
        .ok_or_else(|| reuse_error("completed legacy generation has no lexical directory"))?;
    if lexical == root || !lexical.starts_with(root) {
        return Err(reuse_error(
            "legacy lexical artifacts escape the index root",
        ));
    }
    if !fs::symlink_metadata(layout.lexical_root())?
        .file_type()
        .is_dir()
    {
        return Err(reuse_error(
            "legacy lexical root is not a non-symlink directory",
        ));
    }
    let bytes = serde_json::to_vec(receipt).map_err(|source| SearchError::SubsystemError {
        subsystem: "fsfs.index.completed_reuse",
        source: Box::new(source),
    })?;
    if bytes.len() as u64 > MAX_RECEIPT_BYTES {
        return Err(reuse_error(
            "completed legacy input evidence exceeds its byte limit",
        ));
    }
    let mut digest = Sha256::new();
    digest.update(b"fsfs.legacy_completed_reuse.v1\0");
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    let mut stats = CopyStats::default();
    for path in [
        root.join(super::CURRENT_FILE_NAME),
        root.join("lexical").join(super::CURRENT_FILE_NAME),
        root.join(super::FSFS_SENTINEL_FILE),
        root.join(super::FSFS_LEXICAL_MANIFEST_FILE),
        root.join("vector"),
        lexical,
    ] {
        legacy_hash_artifact(cx, root, &path, 0, &mut stats, &mut digest)?;
    }
    retained_search_checkpoint(cx)?;
    Ok(super::sha256_digest_hex(digest.finalize()))
}

fn legacy_hash_artifact(
    cx: &Cx,
    root: &Path,
    path: &Path,
    depth: usize,
    stats: &mut CopyStats,
    digest: &mut Sha256,
) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    if depth > MAX_COPY_DEPTH {
        return Err(reuse_error(
            "legacy artifact inventory exceeds its depth limit",
        ));
    }
    let label = path
        .strip_prefix(root)
        .ok()
        .and_then(Path::to_str)
        .ok_or_else(|| reuse_error("legacy artifact has an invalid relative path"))?;
    digest.update((label.len() as u64).to_le_bytes());
    digest.update(label.as_bytes());
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            digest.update([0]);
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    if metadata.is_dir() {
        digest.update([1]);
        let mut entries = collect_copy_entries(cx, fs::read_dir(path)?, stats)?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            // Quill's writer admission record is cleared at release and is
            // not search data. Every other engine artifact participates.
            if entry.file_name() == "LOCK" && entry.file_type()?.is_file() {
                continue;
            }
            legacy_hash_artifact(cx, root, &entry.path(), depth + 1, stats, digest)?;
        }
        digest.update([2]);
        return Ok(());
    }
    if !metadata.is_file() {
        return Err(reuse_error(
            "legacy artifacts contain a symlink or special file",
        ));
    }
    digest.update([3]);
    let mut file = open_regular(path)?;
    let before = file.metadata()?;
    let expected = before.len();
    stats.bytes = stats
        .bytes
        .checked_add(expected)
        .filter(|&bytes| bytes <= MAX_LEGACY_ARTIFACT_BYTES)
        .ok_or_else(|| reuse_error("legacy artifacts exceed the reuse byte budget"))?;
    digest.update(expected.to_le_bytes());
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    let mut observed = 0_u64;
    loop {
        retained_search_checkpoint(cx)?;
        let remaining = expected.saturating_sub(observed).saturating_add(1);
        let width = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| reuse_error("legacy artifact read length overflow"))?;
        let count = file.read(&mut buffer[..width])?;
        if count == 0 {
            break;
        }
        observed = observed.saturating_add(count as u64);
        if observed > expected {
            return Err(reuse_error("legacy artifact grew while proving reuse"));
        }
        digest.update(&buffer[..count]);
    }
    let after = file.metadata()?;
    if observed != expected || after.len() != expected || before.modified()? != after.modified()? {
        return Err(reuse_error("legacy artifact changed while proving reuse"));
    }
    retained_search_checkpoint(cx)?;
    Ok(())
}

fn write_receipt(cx: &Cx, root: &Path, receipt: &ReuseReceipt) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    let bytes = serde_json::to_vec(receipt).map_err(|source| SearchError::SubsystemError {
        subsystem: "fsfs.complete_generation.reuse_receipt",
        source: Box::new(source),
    })?;
    if bytes.len() as u64 > MAX_RECEIPT_BYTES {
        return Err(reuse_error(
            "retained input evidence exceeds the 64 MiB limit",
        ));
    }
    // The seed excludes its old receipt. Never overwrite a file whose origin
    // is unknown, and never put a live checkpoint into a sealed generation.
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(RECEIPT_FILE))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    File::open(root)?.sync_all()?;
    tracing::info!(
        cross_process_reuse = receipt.executable_sha256.is_some(),
        eligible_semantic_files = receipt
            .checkpoint
            .files
            .values()
            .filter(|entry| proven(entry))
            .count(),
        covered_files = receipt
            .checkpoint
            .files
            .values()
            .filter(|entry| !entry.content_hash_hex.is_empty())
            .count(),
        total_files = receipt.checkpoint.files.len(),
        "retained completed indexing input evidence; uncovered files require recomputation"
    );
    Ok(())
}

#[derive(Default)]
struct CopyStats {
    entries: usize,
    files: usize,
    bytes: u64,
}

/// Copy a complete admitted predecessor for an explicit mutation, independently
/// of indexing-reuse evidence. The caller must hold `store.begin()` throughout
/// this operation and publish only through that build's predecessor check.
///
/// Unlike checkpoint reuse, deletion and compaction do not infer any new
/// embedding from source text. They preserve the copied producers and discard
/// the old indexing receipt, which cannot describe the changed membership.
pub(super) fn copy_selected_generation(
    cx: &Cx,
    runtime: &FsfsRuntime,
    store: &CompleteGenerationStore,
    destination: &Path,
) -> SearchResult<PublishedGeneration> {
    retained_search_checkpoint(cx)?;
    let predecessor = store.active(cx)?.ok_or_else(|| {
        reuse_error("no complete generation has been published; index the store first")
    })?;
    FsfsRuntime::validate_search_generation_at_root(predecessor.path(), SearchExecutionMode::Full)?;
    if FsfsRuntime::resolve_lexical_engine(predecessor.path())?.engine()
        == Some(frankensearch_quill::BlueGreenEngine::Tantivy)
    {
        // Ordinary search admission can migrate a legacy Tantivy root by
        // rereading source files. An explicit membership mutation must never
        // trigger that rebuild and silently reintroduce a deleted document.
        return Err(reuse_error(
            "complete-generation mutations require a Quill or vector-only bundle; rebuild the legacy lexical generation first",
        ));
    }
    let mut sentinel = FsfsRuntime::read_index_sentinel(predecessor.path())?
        .ok_or_else(|| reuse_error("selected generation has no sentinel"))?;
    if FsfsRuntime::read_matching_manifest_generation(predecessor.path())?.is_none() {
        return Err(reuse_error(
            "selected generation has no matching membership manifests",
        ));
    }
    if !fs::symlink_metadata(destination)?.file_type().is_dir()
        || fs::read_dir(destination)?.next().transpose()?.is_some()
    {
        return Err(reuse_error(
            "mutation requires an empty, non-symlink candidate directory",
        ));
    }
    let mut stats = CopyStats::default();
    copy_tree(cx, predecessor.path(), destination, 0, &mut stats)?;
    if store.active(cx)?.as_ref() != Some(&predecessor) {
        return Err(reuse_error(
            "selected predecessor changed while copying the mutation candidate",
        ));
    }
    sentinel.index_root = destination.display().to_string();
    runtime.write_index_sentinel(destination, &sentinel)?;
    retained_search_checkpoint(cx)?;
    tracing::info!(
        predecessor = predecessor.id(),
        copied_files = stats.files,
        copied_bytes = stats.bytes,
        "copied complete generation for isolated mutation"
    );
    Ok(predecessor)
}

/// Must be called while the caller owns `store.begin()` and before opening any
/// candidate resources. Missing/incompatible evidence starts cold; malformed
/// evidence or damaged selection is an error, not permission to reuse data.
fn seed_candidate(
    cx: &Cx,
    runtime: &FsfsRuntime,
    store: &CompleteGenerationStore,
    destination: &Path,
) -> SearchResult<usize> {
    retained_search_checkpoint(cx)?;
    if runtime.cli_input.full_reindex {
        return Ok(0);
    }
    let Some(predecessor) = store.active(cx)? else {
        return Ok(0);
    };
    let path = predecessor.path().join(RECEIPT_FILE);
    match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error.into()),
        Ok(_) => {}
    }
    let encoded: serde_json::Value = read_json(cx, &path)?;
    let version = encoded
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| reuse_error("reuse receipt has no valid version"))?;
    if version != u64::from(RECEIPT_VERSION) {
        return Ok(0);
    }
    let mut receipt: ReuseReceipt =
        serde_json::from_value(encoded).map_err(|source| SearchError::SubsystemError {
            subsystem: "fsfs.complete_generation.reuse_receipt",
            source: Box::new(source),
        })?;
    if !compatible_receipt(runtime, &receipt)? {
        return Ok(0);
    }
    let eligible = receipt
        .checkpoint
        .files
        .values()
        .filter(|entry| proven(entry))
        .count();
    if eligible == 0 {
        return Ok(0);
    }
    if !receipt.checkpoint.artifacts_durable
        || receipt.checkpoint.schema_version != INDEXING_CHECKPOINT_SCHEMA_VERSION
        || FsfsRuntime::read_checkpoint_manifest_generation(
            predecessor.path(),
            &receipt.checkpoint,
        )?
        .is_none()
    {
        return Err(reuse_error(
            "retained checkpoint is inconsistent with its selected generation",
        ));
    }
    FsfsRuntime::validate_search_generation_at_root(predecessor.path(), SearchExecutionMode::Full)?;
    let mut sentinel = FsfsRuntime::read_index_sentinel(predecessor.path())?
        .ok_or_else(|| reuse_error("selected generation has no sentinel"))?;
    if !fs::symlink_metadata(destination)?.file_type().is_dir()
        || fs::read_dir(destination)?.next().transpose()?.is_some()
    {
        return Err(reuse_error(
            "reuse requires an empty, non-symlink candidate directory",
        ));
    }
    let mut stats = CopyStats::default();
    copy_tree(cx, predecessor.path(), destination, 0, &mut stats)?;
    // The store lease excludes cooperating publication. Rechecking also refuses
    // an out-of-protocol pointer change or source mutation during the copy.
    if store.active(cx)?.as_ref() != Some(&predecessor) {
        return Err(reuse_error(
            "selected predecessor changed while copying the seed",
        ));
    }
    let label = destination.display().to_string();
    sentinel.index_root.clone_from(&label);
    receipt.checkpoint.index_root = label;
    runtime.write_index_sentinel(destination, &sentinel)?;
    write_indexing_checkpoint(destination, &receipt.checkpoint)?;
    File::open(destination)?.sync_all()?;
    tracing::info!(
        predecessor = predecessor.id(),
        seeded_across_process = receipt.session != session_id()?,
        eligible_semantic_files = eligible,
        copied_files = stats.files,
        copied_bytes = stats.bytes,
        "seeded isolated generation; ordinary source-hash and producer checks decide actual reuse"
    );
    Ok(eligible)
}

/// Charge entries while enumerating, including siblings awaiting recursion.
/// Collecting an unbounded directory before checking this budget would defeat
/// both the memory bound and cooperative cancellation. The iterator may read
/// one over-budget entry to distinguish an exact fit from overflow, but that
/// entry is never retained and no subsequent entry is requested.
fn collect_copy_entries<T, I>(cx: &Cx, mut source: I, stats: &mut CopyStats) -> SearchResult<Vec<T>>
where
    I: Iterator<Item = std::io::Result<T>>,
{
    let mut entries = Vec::new();
    loop {
        retained_search_checkpoint(cx)?;
        let next = source.next();
        // EOF and an iterator that returns data after cancellation are not
        // permission to return a successfully admitted inventory.
        retained_search_checkpoint(cx)?;
        let Some(entry) = next else { break };
        let entry = entry?;
        let charged = stats
            .entries
            .checked_add(1)
            .filter(|&count| count <= MAX_COPY_ENTRIES)
            .ok_or_else(|| reuse_error("predecessor copy exceeds its entry limit"))?;
        if entries.len() == entries.capacity() {
            let remaining = MAX_COPY_ENTRIES - stats.entries;
            let growth = entries.capacity().max(16).min(remaining);
            entries
                .try_reserve_exact(growth)
                .map_err(|_| reuse_error("cannot reserve bounded predecessor inventory"))?;
        }
        stats.entries = charged;
        entries.push(entry);
    }
    Ok(entries)
}

fn copy_tree(
    cx: &Cx,
    source: &Path,
    destination: &Path,
    depth: usize,
    stats: &mut CopyStats,
) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    if depth > MAX_COPY_DEPTH || !fs::symlink_metadata(source)?.file_type().is_dir() {
        return Err(reuse_error(
            "invalid or excessively deep predecessor directory",
        ));
    }
    let mut entries = collect_copy_entries(cx, fs::read_dir(source)?, stats)?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        retained_search_checkpoint(cx)?;
        let name = entry.file_name();
        let file_type = entry.file_type()?;
        if !file_type.is_dir() && !file_type.is_file() {
            return Err(reuse_error(
                "predecessor contains a symlink or special filesystem object",
            ));
        }
        if depth == 0
            && [
                COMPLETE_GENERATION_MANIFEST,
                RECEIPT_FILE,
                LEGACY_RECEIPT_FILE,
                FSFS_CHECKPOINT_FILE,
            ]
            .iter()
            .any(|excluded| name == *excluded)
        {
            if !file_type.is_file() {
                return Err(reuse_error(
                    "generation control artifact is not a regular file",
                ));
            }
            continue;
        }
        let target = destination.join(&name);
        if file_type.is_dir() {
            fs::create_dir(&target)?;
            copy_tree(cx, &entry.path(), &target, depth + 1, stats)?;
        } else {
            let mut input = open_regular(&entry.path())?;
            let expected = input.metadata()?.len();
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&target)?;
            // No hard links: tombstones, mmap writes, catalogs and lexical
            // commits in the candidate must never reach a retained reader.
            let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
            let mut copied = 0_u64;
            let mut digest = Sha256::new();
            loop {
                retained_search_checkpoint(cx)?;
                let count = input.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                copied = copied
                    .checked_add(count as u64)
                    .ok_or_else(|| reuse_error("copied artifact length overflow"))?;
                if copied > expected {
                    return Err(reuse_error("predecessor artifact grew during copying"));
                }
                output.write_all(&buffer[..count])?;
                digest.update(&buffer[..count]);
            }
            if copied != expected || input.metadata()?.len() != expected {
                return Err(reuse_error(
                    "predecessor artifact changed length during copying",
                ));
            }
            output.sync_all()?;
            let mut verification = open_regular(&target)?;
            let mut copied_digest = Sha256::new();
            loop {
                retained_search_checkpoint(cx)?;
                let count = verification.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                copied_digest.update(&buffer[..count]);
            }
            if digest.finalize() != copied_digest.finalize() {
                return Err(reuse_error(
                    "copied artifact digest differs from its source stream",
                ));
            }
            stats.files += 1;
            stats.bytes = stats
                .bytes
                .checked_add(copied)
                .ok_or_else(|| reuse_error("total copied artifact length overflow"))?;
        }
    }
    File::open(destination)?.sync_all()?;
    Ok(())
}

#[cfg(all(test, unix))]
mod copy_tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use std::os::unix::fs::MetadataExt;

    fn inventory_fixture(
        parent: &Path,
        names: &[&str],
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let source = parent.join("source");
        let destination = parent.join("destination");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&destination).unwrap();
        for name in names {
            let path = source.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, name.as_bytes()).unwrap();
        }
        (source, destination)
    }

    fn assert_entry_limit(result: SearchResult<()>) {
        assert!(
            matches!(result, Err(SearchError::InvalidConfig { field, reason, .. })
                if field == "complete_generation.reuse" && reason.contains("entry limit"))
        );
    }

    #[test]
    fn copy_inventory_accepts_exact_budget_without_double_charging() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (source, destination) = inventory_fixture(parent.path(), &["b", "a"]);
            let mut stats = CopyStats {
                entries: MAX_COPY_ENTRIES - 2,
                ..CopyStats::default()
            };
            copy_tree(&cx, &source, &destination, 0, &mut stats).unwrap();
            assert_eq!(stats.entries, MAX_COPY_ENTRIES);
            assert_eq!(stats.files, 2);
            assert_eq!(stats.bytes, 2);
            for name in ["a", "b"] {
                assert_eq!(fs::read(destination.join(name)).unwrap(), name.as_bytes());
                assert_eq!(fs::read(source.join(name)).unwrap(), name.as_bytes());
            }
            // An empty directory remains legal when the global budget is full.
            let empty = parent.path().join("empty");
            fs::create_dir(&empty).unwrap();
            assert!(
                collect_copy_entries(&cx, fs::read_dir(empty).unwrap(), &mut stats)
                    .unwrap()
                    .is_empty()
            );
        });
    }

    #[test]
    fn copy_inventory_rejects_oversized_directory_before_creating_files() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (source, destination) = inventory_fixture(parent.path(), &["a", "b", "c"]);
            let mut stats = CopyStats {
                entries: MAX_COPY_ENTRIES - 2,
                ..CopyStats::default()
            };
            assert_entry_limit(copy_tree(&cx, &source, &destination, 0, &mut stats));
            assert_eq!(stats.entries, MAX_COPY_ENTRIES);
            assert_eq!(stats.files, 0);
            assert_eq!(stats.bytes, 0);
            assert_eq!(fs::read_dir(destination).unwrap().count(), 0);
            for name in ["a", "b", "c"] {
                assert_eq!(fs::read(source.join(name)).unwrap(), name.as_bytes());
            }
        });
    }

    #[test]
    fn copy_inventory_charges_pending_siblings_before_recursing() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (source, destination) = inventory_fixture(parent.path(), &["a/data"]);
            fs::create_dir(source.join("b")).unwrap();
            let mut stats = CopyStats {
                entries: MAX_COPY_ENTRIES - 2,
                ..CopyStats::default()
            };
            assert_entry_limit(copy_tree(&cx, &source, &destination, 0, &mut stats));
            // Both root directories have been charged before descending into
            // a. The child cannot steal the pending sibling's inventory slot.
            assert_eq!(stats.entries, MAX_COPY_ENTRIES);
            assert_eq!(stats.files, 0);
            assert_eq!(stats.bytes, 0);
            assert!(destination.join("a").is_dir());
            assert!(!destination.join("a/data").exists());
            assert!(!destination.join("b").exists());
            assert_eq!(fs::read(source.join("a/data")).unwrap(), b"a/data");
        });
    }

    #[test]
    fn copy_inventory_counts_excluded_generation_control_artifacts() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (source, destination) =
                inventory_fixture(parent.path(), &[COMPLETE_GENERATION_MANIFEST, "data"]);
            let mut stats = CopyStats {
                entries: MAX_COPY_ENTRIES - 1,
                ..CopyStats::default()
            };
            assert_entry_limit(copy_tree(&cx, &source, &destination, 0, &mut stats));
            assert_eq!(fs::read_dir(destination).unwrap().count(), 0);
            assert_eq!(stats.entries, MAX_COPY_ENTRIES);
            assert_eq!(stats.files, 0);
            assert!(source.join(COMPLETE_GENERATION_MANIFEST).is_file());
        });
    }

    #[test]
    fn copy_inventory_stops_unbounded_iterators_at_first_over_budget_entry() {
        run_test_with_cx(|cx| async move {
            let calls = std::cell::Cell::new(0_usize);
            let source = std::iter::from_fn(|| {
                calls.set(calls.get() + 1);
                Some(Ok(calls.get()))
            });
            let mut stats = CopyStats {
                entries: MAX_COPY_ENTRIES - 3,
                ..CopyStats::default()
            };
            assert_entry_limit(collect_copy_entries(&cx, source, &mut stats).map(|_| ()));
            assert_eq!(calls.get(), 4);
            assert_eq!(stats.entries, MAX_COPY_ENTRIES);
        });
    }

    #[test]
    fn copy_inventory_observes_cancellation_before_data_after_data_and_at_eof() {
        run_test_with_cx(|cx| async move {
            let calls = std::cell::Cell::new(0_usize);
            let source = std::iter::once_with(|| {
                calls.set(calls.get() + 1);
                Ok(7_u8)
            });
            cx.set_cancel_requested(true);
            let result = collect_copy_entries(&cx, source, &mut CopyStats::default());
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(calls.get(), 0);

            let mut stats = CopyStats::default();
            let source = std::iter::once_with(|| {
                cx.set_cancel_requested(true);
                Ok(7_u8)
            });
            let result = collect_copy_entries(&cx, source, &mut stats);
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(stats.entries, 0);

            let mut calls = 0;
            let source = std::iter::from_fn(|| {
                calls += 1;
                if calls == 1 {
                    Some(Ok(7_u8))
                } else {
                    cx.set_cancel_requested(true);
                    None
                }
            });
            let result = collect_copy_entries(&cx, source, &mut stats);
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(stats.entries, 1);
            assert_eq!(calls, 2);
        });
    }

    #[test]
    fn copy_inventory_propagates_listing_errors_without_reading_further() {
        run_test_with_cx(|cx| async move {
            let mut source = [
                Ok(1_u8),
                Err(std::io::Error::from(ErrorKind::PermissionDenied)),
                Ok(2),
            ]
            .into_iter();
            let mut stats = CopyStats::default();
            assert!(matches!(collect_copy_entries(&cx, &mut source, &mut stats),
                    Err(SearchError::Io(error)) if error.kind() == ErrorKind::PermissionDenied));
            assert_eq!(stats.entries, 1);
            assert_eq!(source.next().unwrap().unwrap(), 2);
        });
    }

    #[test]
    fn copy_inventory_counter_cannot_wrap_back_under_the_limit() {
        run_test_with_cx(|cx| async move {
            let mut stats = CopyStats {
                entries: usize::MAX,
                ..CopyStats::default()
            };
            assert_entry_limit(
                collect_copy_entries(&cx, std::iter::once(Ok(1_u8)), &mut stats).map(|_| ()),
            );
            assert_eq!(stats.entries, usize::MAX);
        });
    }

    #[test]
    fn mutation_copy_refuses_legacy_lexical_migration_before_copying() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, parent.path()).unwrap();
            let first = store.begin(&cx).unwrap();
            fs::create_dir(first.path().join("lexical")).unwrap();
            // The layout detector needs only this legacy marker. The refusal
            // must happen before any loader, source rebuild or candidate copy.
            fs::write(first.path().join("lexical/meta.json"), b"{}").unwrap();
            let publication = first.publish(&cx, |_, _| Ok(())).unwrap();
            let crate::generation_store::GenerationPublication::Durable(predecessor) = publication
            else {
                panic!("fixture publication must be durable"); // ubs:ignore — cfg(test) assertion.
            };
            let build = store.begin(&cx).unwrap();
            let runtime = FsfsRuntime::new(crate::config::FsfsConfig::default());
            let error = copy_selected_generation(&cx, &runtime, &store, build.path()).unwrap_err();
            assert!(
                matches!(error, SearchError::InvalidConfig { field, reason, .. }
                if field == "complete_generation.reuse" && reason.contains("legacy lexical generation"))
            );
            assert_eq!(fs::read_dir(build.path()).unwrap().count(), 0);
            assert_eq!(store.active(&cx).unwrap(), Some(predecessor));
        });
    }

    #[test]
    fn seed_copy_uses_independent_inodes_and_does_not_copy_control_artifacts() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let source = parent.path().join("source");
            let destination = parent.path().join("destination");
            fs::create_dir_all(source.join("vector")).unwrap();
            fs::create_dir(&destination).unwrap();
            fs::write(source.join("vector/fast.idx"), b"original vector bytes").unwrap();
            for name in [
                RECEIPT_FILE,
                COMPLETE_GENERATION_MANIFEST,
                FSFS_CHECKPOINT_FILE,
            ] {
                fs::write(source.join(name), b"source-only control record").unwrap();
            }
            let mut stats = CopyStats::default();
            copy_tree(&cx, &source, &destination, 0, &mut stats).unwrap();
            let original = source.join("vector/fast.idx");
            let copied = destination.join("vector/fast.idx");
            assert_eq!(fs::read(&original).unwrap(), fs::read(&copied).unwrap());
            assert_ne!(
                fs::metadata(&original).unwrap().ino(),
                fs::metadata(&copied).unwrap().ino()
            );
            fs::write(copied, b"candidate mutation").unwrap();
            assert_eq!(fs::read(original).unwrap(), b"original vector bytes");
            assert_eq!(stats.files, 1);
            assert_eq!(stats.bytes, 21);
            for name in [
                RECEIPT_FILE,
                COMPLETE_GENERATION_MANIFEST,
                FSFS_CHECKPOINT_FILE,
            ] {
                assert!(!destination.join(name).exists());
                assert!(source.join(name).is_file());
            }
        });
    }

    #[test]
    fn seed_copy_refuses_existing_files_and_symlinks_without_overwriting_them() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let source = parent.path().join("source");
            let destination = parent.path().join("destination");
            fs::create_dir(&source).unwrap();
            fs::create_dir(&destination).unwrap();
            fs::write(source.join("data"), b"source").unwrap();
            fs::write(destination.join("data"), b"independent evidence").unwrap();
            assert!(copy_tree(&cx, &source, &destination, 0, &mut CopyStats::default()).is_err());
            assert_eq!(
                fs::read(destination.join("data")).unwrap(),
                b"independent evidence"
            );
            let links = parent.path().join("links");
            fs::create_dir(&links).unwrap();
            std::os::unix::fs::symlink(source.join("data"), links.join("linked")).unwrap();
            let empty = parent.path().join("empty");
            fs::create_dir(&empty).unwrap();
            assert!(copy_tree(&cx, &links, &empty, 0, &mut CopyStats::default()).is_err());
            assert_eq!(fs::read_dir(empty).unwrap().count(), 0);
            assert_eq!(fs::read(source.join("data")).unwrap(), b"source");
        });
    }

    #[test]
    fn cancelled_seed_does_not_allocate_files() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let source = parent.path().join("source");
            let destination = parent.path().join("destination");
            fs::create_dir(&source).unwrap();
            fs::create_dir(&destination).unwrap();
            fs::write(source.join("data"), b"retained").unwrap();
            cx.set_cancel_requested(true);
            let result = copy_tree(&cx, &source, &destination, 0, &mut CopyStats::default());
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(fs::read_dir(destination).unwrap().count(), 0);
            assert_eq!(fs::read(source.join("data")).unwrap(), b"retained");
        });
    }

    #[test]
    fn receipt_reader_bounds_bytes_and_rejects_symlink_inputs() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let large = parent.path().join("large");
            File::create(&large)
                .unwrap()
                .set_len(MAX_RECEIPT_BYTES + 1)
                .unwrap();
            assert!(matches!(
                read_json::<serde_json::Value>(&cx, &large),
                Err(SearchError::InvalidConfig { .. })
            ));
            let linked = parent.path().join("linked");
            std::os::unix::fs::symlink(&large, &linked).unwrap();
            assert!(matches!(
                read_json::<serde_json::Value>(&cx, &linked),
                Err(SearchError::InvalidConfig { .. })
            ));
        });
    }
}

#[cfg(all(test, unix, not(feature = "embedded-models")))]
mod generation_tests {
    use super::super::{CheckpointReuse, IndexCandidate, checkpoint_entry_reuse};
    use super::*;
    use crate::config::IngestionClass;
    use crate::generation_store::{GenerationPublication, PublishedGeneration};
    use crate::{CliInput, FsfsConfig};
    use asupersync::test_utils::run_test_with_cx;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn fixture(parent: &Path, count: usize) -> (FsfsRuntime, PathBuf, PathBuf) {
        let source = parent.join("source");
        let root = parent.join("store");
        fs::create_dir_all(&source).unwrap();
        for number in 0..count {
            fs::write(
                source.join(format!("doc-{number}.md")),
                format!("sharedtoken document {number}"),
            )
            .unwrap();
        }
        let mut config = FsfsConfig::default();
        config.indexing.offline = true;
        config.indexing.quality_model.clear();
        config.indexing.embedding_batch_size = 1;
        config.search.fast_only = true;
        config.search.rerank = false;
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Index,
            target_path: Some(source.clone()),
            index_dir: Some(root.clone()),
            quiet: true,
            ..CliInput::default()
        });
        (runtime, source, root)
    }

    mod completed_batch_tests {
        use super::*;
        use frankensearch_core::{
            Canonicalizer, DefaultCanonicalizer, Embedder, EmbeddingIdentityBundleV1,
            ModelCategory, SearchFuture,
        };
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Default)]
        struct InferenceCounts {
            probes: AtomicUsize,
            documents: AtomicUsize,
            batches: AtomicUsize,
        }

        struct CountingEmbedder {
            id: &'static str,
            identity: EmbeddingIdentityBundleV1,
            counts: Arc<InferenceCounts>,
            dimension: usize,
            fail_after_documents: Option<usize>,
            cancel_on_failure: bool,
        }

        impl CountingEmbedder {
            fn new(id: &'static str, dimension: usize, drift: bool) -> Self {
                let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(
                    id,
                    u32::try_from(dimension).unwrap(),
                );
                if drift {
                    identity
                        .producer
                        .implementation_revision
                        .push_str("-changed");
                }
                identity.validate().unwrap();
                Self {
                    id,
                    identity,
                    counts: Arc::default(),
                    dimension,
                    fail_after_documents: None,
                    cancel_on_failure: false,
                }
            }

            fn vector(&self, text: &str) -> Vec<f32> {
                let mut vector = vec![0.0; self.dimension];
                let slot = text.bytes().fold(0_usize, |sum, byte| {
                    (sum + usize::from(byte)) % self.dimension
                });
                vector[slot] = 1.0;
                vector
            }
        }

        impl Embedder for CountingEmbedder {
            fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
                Ok(&self.identity)
            }

            fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                Box::pin(async move {
                    if text == "probe" {
                        self.counts.probes.fetch_add(1, Ordering::SeqCst);
                    } else {
                        let previous = self.counts.documents.fetch_add(1, Ordering::SeqCst);
                        if self
                            .fail_after_documents
                            .is_some_and(|limit| previous >= limit)
                        {
                            if self.cancel_on_failure {
                                return Err(SearchError::Cancelled {
                                    phase: "test.fast_window_batch".to_owned(),
                                    reason: "cancel after one successful passage".to_owned(),
                                });
                            }
                            return Err(std::io::Error::from(ErrorKind::TimedOut).into());
                        }
                    }
                    Ok(self.vector(text))
                })
            }

            fn embed_batch<'a>(
                &'a self,
                cx: &'a Cx,
                texts: &'a [&'a str],
            ) -> SearchFuture<'a, Vec<Vec<f32>>> {
                Box::pin(async move {
                    self.counts.batches.fetch_add(1, Ordering::SeqCst);
                    let mut vectors = Vec::with_capacity(texts.len());
                    for text in texts {
                        vectors.push(self.embed(cx, text).await?);
                    }
                    Ok(vectors)
                })
            }

            fn dimension(&self) -> usize {
                self.dimension
            }
            fn id(&self) -> &str {
                self.id
            }
            fn model_name(&self) -> &str {
                self.id
            }
            fn is_semantic(&self) -> bool {
                true
            }
            fn category(&self) -> ModelCategory {
                ModelCategory::StaticEmbedder
            }
        }

        struct RestoreEmbedders {
            fast: Option<Arc<dyn Embedder>>,
            quality: Option<Arc<dyn Embedder>>,
        }

        impl RestoreEmbedders {
            fn install(fast: Arc<dyn Embedder>, quality: Arc<dyn Embedder>) -> Self {
                let previous = Self {
                    fast: super::super::super::test_fast_embedder_override(),
                    quality: super::super::super::test_quality_embedder_override(),
                };
                super::super::super::set_test_fast_embedder(Some(fast));
                super::super::super::set_test_quality_embedder(Some(quality));
                previous
            }
        }

        impl Drop for RestoreEmbedders {
            fn drop(&mut self) {
                super::super::super::set_test_fast_embedder(self.fast.take());
                super::super::super::set_test_quality_embedder(self.quality.take());
            }
        }

        fn assert_index_calls(report: &serde_json::Value, fast: usize, quality: usize) {
            assert_eq!(report["fast_documents"], fast);
            assert_eq!(report["quality_documents"], quality);
            assert_eq!(report["fast_probes"], 1, "readiness probes still run");
            assert_eq!(
                report["quality_probes"], 1,
                "quality readiness remains independent"
            );
            assert_eq!(report["checkpoint_present"], false);
            assert_eq!(report["receipt_present"], true);
        }

        async fn run_counted_legacy(cx: &Cx, parent: &Path, operation: &str) -> serde_json::Value {
            run_counted_legacy_windows(cx, parent, operation, 1).await
        }

        async fn run_counted_legacy_windows(
            cx: &Cx,
            parent: &Path,
            operation: &str,
            max_windows: usize,
        ) -> serde_json::Value {
            let (mut runtime, source, root) = fixture(parent, 0);
            runtime.config.indexing.fast_window_max_per_file = max_windows;
            runtime.config.search.fast_only = false;
            runtime.config.search.quality_timeout_ms = 5_000;
            "reuse-quality".clone_into(&mut runtime.config.indexing.quality_model);
            if operation == "config_drift" {
                runtime.config.indexing.embedding_batch_size = 2;
            }
            runtime.cli_input.full_reindex = operation == "force";
            if operation == "daemon" {
                runtime.cli_input.daemon_socket = Some(parent.join("daemon.sock"));
            }
            let fast = Arc::new(CountingEmbedder::new(
                "reuse-fast",
                4,
                operation == "fast_drift",
            ));
            let quality = Arc::new(CountingEmbedder::new(
                "reuse-quality",
                6,
                operation == "quality_drift",
            ));
            let _restore = RestoreEmbedders::install(fast.clone(), quality.clone());
            let payload = runtime
                .run_one_shot_index_scaffold_internal(
                    cx,
                    CliCommand::Index,
                    |_| Ok(()),
                    false,
                    true,
                )
                .await
                .unwrap();
            let serialized_payload = serde_json::to_value(&payload).unwrap();
            if max_windows == 1 {
                assert!(serialized_payload.get("fast_window_coverage").is_none());
            }
            let report = serde_json::json!({
                "fast_documents": fast.counts.documents.load(Ordering::SeqCst),
                "quality_documents": quality.counts.documents.load(Ordering::SeqCst),
                "fast_probes": fast.counts.probes.load(Ordering::SeqCst),
                "quality_probes": quality.counts.probes.load(Ordering::SeqCst),
                "checkpoint_present": root.join(FSFS_CHECKPOINT_FILE).exists(),
                "receipt_present": root.join(LEGACY_RECEIPT_FILE).exists(),
                "indexed_files": payload.generation.indexed_files,
                "fast_window_coverage": serialized_payload.get("fast_window_coverage"),
                "fast_rows": frankensearch_index::VectorIndex::open_read_only(
                    &root.join(super::super::super::FSFS_VECTOR_INDEX_FILE)
                ).unwrap().live_doc_ids().unwrap().len(),
            });
            assert!(payload.generation.generation_complete);
            assert_eq!(payload.generation.fast_window_max_per_file, max_windows);
            let manifests = FsfsRuntime::read_index_manifest(&root).unwrap().unwrap();
            for (relative, producer) in [
                (super::super::super::FSFS_VECTOR_INDEX_FILE, fast.as_ref()),
                (
                    super::super::super::FSFS_VECTOR_QUALITY_INDEX_FILE,
                    quality.as_ref(),
                ),
            ] {
                let index =
                    frankensearch_index::VectorIndex::open_read_only(&root.join(relative)).unwrap();
                assert_eq!(index.embedder_revision(), producer.identity.fingerprint());
                let mut expected_rows = std::collections::BTreeMap::new();
                for manifest in &manifests {
                    let text = fs::read_to_string(source.join(&manifest.file_key)).unwrap();
                    if relative == super::super::super::FSFS_VECTOR_INDEX_FILE {
                        if let Some(plan) = manifest.fast_windows.as_ref() {
                            let lexical =
                                super::super::super::LEXICAL_CANONICALIZER.canonicalize(&text);
                            for (id, passage) in plan
                                .row_ids(&manifest.file_key)
                                .into_iter()
                                .zip(plan.texts(&lexical).unwrap())
                            {
                                expected_rows.insert(id, producer.vector(passage));
                            }
                            continue;
                        }
                    }
                    expected_rows.insert(
                        manifest.file_key.clone(),
                        producer.vector(&DefaultCanonicalizer::default().canonicalize(&text)),
                    );
                }
                assert_eq!(
                    index.live_doc_ids().unwrap(),
                    expected_rows.keys().cloned().collect()
                );
                for row in 0..index.record_count() {
                    if !index.is_deleted(row) {
                        assert_eq!(
                            index.vector_at_f32(row).unwrap(),
                            expected_rows[index.doc_id_at(row).unwrap()],
                        );
                    }
                }
            }
            let lexical = FsfsRuntime::resolve_lexical_engine(&root)
                .unwrap()
                .engine_dir()
                .unwrap();
            let index = frankensearch_quill::QuillSearchIndex::open(
                cx,
                lexical,
                frankensearch_quill::QuillConfig::default(),
            )
            .await
            .unwrap();
            let hits = index.search_results(cx, "sharedtoken", 20).unwrap();
            assert_eq!(hits.len(), payload.generation.indexed_files);
            assert!(
                hits.iter()
                    .all(|hit| source.join(hit.doc_id.as_str()).is_file())
            );
            report
        }

        #[test]
        fn fast_window_index_reopens_reuses_and_reconciles_shrunk_and_removed_sources() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (_, source, root) = fixture(parent.path(), 2);
                let long = format!(
                    "sharedtoken {} subterranean irrigation at the far end",
                    "ordinary introduction ".repeat(600),
                );
                fs::write(source.join("doc-0.md"), &long).unwrap();
                let first = run_counted_legacy_windows(&cx, parent.path(), "index", 4).await;
                assert_index_calls(&first, 5, 2);
                assert_eq!(first["fast_rows"], 5);
                let coverage = &first["fast_window_coverage"];
                assert_eq!(coverage["source_files"], 2);
                assert_eq!(coverage["window_rows"], 5);
                assert_eq!(coverage["capped_files"], 1);
                assert!(
                    coverage["covered_characters"].as_u64().unwrap()
                        < coverage["total_characters"].as_u64().unwrap()
                );
                let receipt: LegacyReuseReceipt =
                    read_json(&cx, &root.join(LEGACY_RECEIPT_FILE)).unwrap();
                let plan = receipt.receipt.checkpoint.files["doc-0.md"]
                    .fast_windows
                    .as_ref()
                    .unwrap();
                assert_eq!(plan.windows.len(), 4);
                let entry = &receipt.receipt.checkpoint.files["doc-0.md"];
                let manifest = FsfsRuntime::read_matching_manifest_generation(&root)
                    .unwrap()
                    .unwrap()
                    .remove("doc-0.md")
                    .unwrap();
                let probe = IndexCandidate {
                    file_path: source.join("doc-0.md"),
                    file_key: "doc-0.md".to_owned(),
                    modified_ms: u64::try_from(entry.revision).unwrap(),
                    ingestion_class: IngestionClass::FullSemanticLexical,
                };
                let mut live = plan.row_ids("doc-0.md").into_iter().collect::<HashSet<_>>();
                let reuse = |ids: &HashSet<String>| {
                    checkpoint_entry_reuse(
                        &receipt.receipt.checkpoint,
                        entry,
                        &manifest,
                        &probe,
                        &entry.content_hash_hex,
                        &receipt.receipt.checkpoint.embedder_id,
                        receipt.receipt.checkpoint.embedder_dimension,
                        false,
                        ids,
                    )
                };
                assert_eq!(reuse(&live), CheckpointReuse::Complete);
                live.remove(&plan.row_ids("doc-0.md")[1]);
                assert_eq!(reuse(&live), CheckpointReuse::LexicalOnly);
                let lexical = super::super::super::LEXICAL_CANONICALIZER.canonicalize(&long);
                assert!(
                    plan.texts(&lexical)
                        .unwrap()
                        .last()
                        .unwrap()
                        .contains("subterranean")
                );
                assert_index_calls(
                    &run_counted_legacy_windows(&cx, parent.path(), "index", 4).await,
                    0,
                    0,
                );
                fs::write(source.join("doc-0.md"), "sharedtoken short replacement").unwrap();
                let shrunk = run_counted_legacy_windows(&cx, parent.path(), "index", 4).await;
                assert_index_calls(&shrunk, 1, 1);
                assert_eq!(shrunk["fast_rows"], 2);
                fs::rename(source.join("doc-0.md"), parent.path().join("removed.md")).unwrap();
                let removed = run_counted_legacy_windows(&cx, parent.path(), "index", 4).await;
                assert_index_calls(&removed, 0, 0);
                assert_eq!(removed["fast_rows"], 1);
                assert_eq!(removed["indexed_files"], 1);
            });
        }

        #[test]
        fn fast_window_policy_changes_rebuild_and_default_restores_exact_prefix_rows() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (_, source, root) = fixture(parent.path(), 2);
                fs::write(
                    source.join("doc-0.md"),
                    format!(
                        "sharedtoken ```rust\n{}\n``` deep tail",
                        "let x = 1;\n".repeat(900)
                    ),
                )
                .unwrap();
                assert_index_calls(
                    &run_counted_legacy_windows(&cx, parent.path(), "index", 4).await,
                    5,
                    2,
                );
                assert_index_calls(
                    &run_counted_legacy_windows(&cx, parent.path(), "index", 2).await,
                    3,
                    2,
                );
                let prefix = run_counted_legacy(&cx, parent.path(), "index").await;
                assert_index_calls(&prefix, 2, 2);
                assert_eq!(prefix["fast_rows"], 2);
                assert!(prefix["fast_window_coverage"].is_null());
                assert!(
                    FsfsRuntime::read_index_manifest(&root)
                        .unwrap()
                        .unwrap()
                        .iter()
                        .all(|entry| entry.fast_windows.is_none())
                );
                assert_index_calls(&run_counted_legacy(&cx, parent.path(), "index").await, 0, 0);
            });
        }

        #[test]
        fn fast_window_later_batch_failure_never_publishes_a_partial_source() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (mut runtime, source, root) = fixture(parent.path(), 1);
                runtime.config.indexing.fast_window_max_per_file = 4;
                fs::write(
                    source.join("doc-0.md"),
                    format!("sharedtoken {} deep tail", "long introduction ".repeat(500)),
                )
                .unwrap();
                let mut fast = CountingEmbedder::new("reuse-fast", 4, false);
                fast.fail_after_documents = Some(1);
                let _restore = RestoreEmbedders::install(
                    Arc::new(fast),
                    Arc::new(CountingEmbedder::new("reuse-quality", 6, false)),
                );
                let payload = runtime
                    .run_one_shot_index_scaffold_internal(
                        &cx,
                        CliCommand::Index,
                        |_| Ok(()),
                        false,
                        true,
                    )
                    .await
                    .unwrap();
                assert!(!payload.generation.generation_complete);
                assert_eq!(payload.semantic_deferred_files, 1);
                assert_eq!(payload.semantic_indexed_files, 0);
                assert!(!payload.input_checkpoint.files["doc-0.md"].semantic_indexed);
                assert!(!root.join(LEGACY_RECEIPT_FILE).exists());
                let index = frankensearch_index::VectorIndex::open_read_only(
                    &root.join(super::super::super::FSFS_VECTOR_INDEX_FILE),
                )
                .unwrap();
                assert!(index.live_doc_ids().unwrap().is_empty());
            });
        }

        #[test]
        fn fast_windows_of_short_files_share_one_batch() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (mut runtime, _, root) = fixture(parent.path(), 5);
                runtime.config.indexing.fast_window_max_per_file = 4;
                runtime.config.indexing.embedding_batch_size = 8;
                let fast = Arc::new(CountingEmbedder::new("reuse-fast", 4, false));
                let _restore = RestoreEmbedders::install(
                    fast.clone(),
                    Arc::new(CountingEmbedder::new("reuse-quality", 6, false)),
                );
                let payload = runtime
                    .run_one_shot_index_scaffold_internal(
                        &cx,
                        CliCommand::Index,
                        |_| Ok(()),
                        false,
                        true,
                    )
                    .await
                    .unwrap();
                assert_eq!(payload.semantic_indexed_files, 5);
                assert_eq!(fast.counts.documents.load(Ordering::SeqCst), 5);
                assert_eq!(
                    fast.counts.batches.load(Ordering::SeqCst),
                    1,
                    "five one-window files fit one batch"
                );
                let index = frankensearch_index::VectorIndex::open_read_only(
                    &root.join(super::super::super::FSFS_VECTOR_INDEX_FILE),
                )
                .unwrap();
                assert_eq!(index.live_doc_ids().unwrap().len(), 5);
            });
        }

        #[test]
        fn fast_window_batch_failure_defers_every_file_it_touched() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (mut runtime, source, root) = fixture(parent.path(), 3);
                runtime.config.indexing.fast_window_max_per_file = 4;
                runtime.config.indexing.embedding_batch_size = 3;
                // Windows [doc-0: 1][doc-1: 4][doc-2: 1] pack as
                // [0, 1, 1] [1, 1, 2]: doc-1 spans both batches.
                fs::write(
                    source.join("doc-1.md"),
                    format!("sharedtoken {} deep tail", "long introduction ".repeat(500)),
                )
                .unwrap();
                let mut fast = CountingEmbedder::new("reuse-fast", 4, false);
                fast.fail_after_documents = Some(3);
                let _restore = RestoreEmbedders::install(
                    Arc::new(fast),
                    Arc::new(CountingEmbedder::new("reuse-quality", 6, false)),
                );
                let payload = runtime
                    .run_one_shot_index_scaffold_internal(
                        &cx,
                        CliCommand::Index,
                        |_| Ok(()),
                        false,
                        true,
                    )
                    .await
                    .unwrap();
                assert!(!payload.generation.generation_complete);
                assert_eq!(payload.semantic_indexed_files, 1);
                assert_eq!(payload.semantic_deferred_files, 2);
                let files = &payload.input_checkpoint.files;
                assert!(files["doc-0.md"].semantic_indexed);
                assert!(!files["doc-1.md"].semantic_indexed);
                assert!(!files["doc-2.md"].semantic_indexed);
                let index = frankensearch_index::VectorIndex::open_read_only(
                    &root.join(super::super::super::FSFS_VECTOR_INDEX_FILE),
                )
                .unwrap();
                assert_eq!(
                    index.live_doc_ids().unwrap(),
                    HashSet::from(["doc-0.md".to_owned()])
                );
            });
        }

        #[test]
        fn fast_window_cancelled_batch_keeps_checkpoint_incomplete_and_wal_empty() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (mut runtime, source, root) = fixture(parent.path(), 1);
                runtime.config.indexing.fast_window_max_per_file = 4;
                fs::write(
                    source.join("doc-0.md"),
                    format!("sharedtoken {} deep tail", "long introduction ".repeat(500)),
                )
                .unwrap();
                let mut fast = CountingEmbedder::new("reuse-fast", 4, false);
                fast.fail_after_documents = Some(1);
                fast.cancel_on_failure = true;
                let _restore = RestoreEmbedders::install(
                    Arc::new(fast),
                    Arc::new(CountingEmbedder::new("reuse-quality", 6, false)),
                );
                let result = runtime
                    .run_one_shot_index_scaffold_internal(
                        &cx,
                        CliCommand::Index,
                        |_| Ok(()),
                        false,
                        true,
                    )
                    .await;
                assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                let checkpoint: IndexingCheckpoint =
                    read_json(&cx, &root.join(FSFS_CHECKPOINT_FILE)).unwrap();
                assert!(!checkpoint.artifacts_durable);
                assert!(
                    checkpoint
                        .files
                        .values()
                        .all(|entry| !entry.semantic_indexed)
                );
                assert!(!root.join(LEGACY_RECEIPT_FILE).exists());
                let index = frankensearch_index::VectorIndex::open_read_only(
                    &root.join(super::super::super::FSFS_VECTOR_INDEX_FILE),
                )
                .unwrap();
                assert!(index.live_doc_ids().unwrap().is_empty());
            });
        }

        /// A warm query daemon maps both generations under the reader side of
        /// the map lock until asked to exit. An unchanged re-index must ask,
        /// then reuse both tiers, rather than be refused and rebuild them.
        #[cfg(unix)]
        #[test]
        fn completed_legacy_reuse_quiesces_a_query_daemon_holding_the_generations() {
            use std::io::BufRead;
            use std::os::unix::net::{UnixListener, UnixStream};

            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (_, _, root) = fixture(parent.path(), 2);
                assert_index_calls(&run_counted_legacy(&cx, parent.path(), "index").await, 2, 2);

                let socket = parent.path().join("daemon.sock");
                let listener = UnixListener::bind(&socket).unwrap();
                let readers = [
                    super::super::super::FSFS_VECTOR_INDEX_FILE,
                    super::super::super::FSFS_VECTOR_QUALITY_INDEX_FILE,
                ]
                .map(|relative| {
                    frankensearch_index::VectorIndex::open_read_only(&root.join(relative)).unwrap()
                });
                let daemon = std::thread::spawn(move || {
                    let (stream, _) = listener.accept().unwrap();
                    let mut request = String::new();
                    std::io::BufReader::new(stream)
                        .read_line(&mut request)
                        .unwrap();
                    drop(readers);
                    request
                });

                let report = run_counted_legacy(&cx, parent.path(), "daemon").await;
                if !daemon.is_finished() {
                    // Unblock the stand-in so a refused run fails, not hangs.
                    let _ = UnixStream::connect(&socket);
                }
                assert_eq!(daemon.join().unwrap(), ":shutdown\n");
                assert_index_calls(&report, 0, 0);
            });
        }

        #[test]
        fn completed_legacy_reuse_skips_both_tiers_and_reconciles_changed_membership() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (_, source, _) = fixture(parent.path(), 2);
                assert_index_calls(&run_counted_legacy(&cx, parent.path(), "index").await, 2, 2);
                assert_index_calls(&run_counted_legacy(&cx, parent.path(), "index").await, 0, 0);
                assert_index_calls(&run_counted_legacy(&cx, parent.path(), "force").await, 2, 2);

                let path = source.join("doc-0.md");
                let before = fs::metadata(&path).unwrap();
                fs::write(&path, "sharedtoken revision 0").unwrap();
                File::options()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_times(std::fs::FileTimes::new().set_modified(before.modified().unwrap()))
                    .unwrap();
                assert_eq!(fs::metadata(&path).unwrap().len(), before.len());
                assert_eq!(
                    fs::metadata(&path).unwrap().modified().unwrap(),
                    before.modified().unwrap()
                );
                assert_index_calls(&run_counted_legacy(&cx, parent.path(), "index").await, 1, 1);

                // Move outside discovery instead of deleting the fixture.
                fs::rename(&path, parent.path().join("retired.md")).unwrap();
                let removed = run_counted_legacy(&cx, parent.path(), "index").await;
                assert_index_calls(&removed, 0, 0);
                assert_eq!(removed["indexed_files"], 1);
                fs::write(source.join("added.md"), "sharedtoken newly added content").unwrap();
                assert_index_calls(&run_counted_legacy(&cx, parent.path(), "index").await, 1, 1);
            });
        }

        #[test]
        fn completed_legacy_reuse_revalidates_configuration_and_independent_producers() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                fixture(parent.path(), 2);
                for (operation, fast, quality) in [
                    ("index", 2, 2),
                    ("config_drift", 2, 2),
                    ("config_drift", 0, 0),
                    ("index", 2, 2),
                    ("quality_drift", 0, 2),
                    ("index", 0, 2),
                    ("fast_drift", 2, 2),
                    ("fast_drift", 0, 0),
                ] {
                    let report = run_counted_legacy(&cx, parent.path(), operation).await;
                    assert_index_calls(&report, fast, quality);
                }
            });
        }

        #[test]
        fn completed_legacy_reuse_rejects_mutated_artifacts_and_input_evidence() {
            run_test_with_cx(|cx| async move {
                for mutation in [
                    "vector",
                    "lexical",
                    "lexical_ghost",
                    "receipt",
                    "malformed",
                    "version",
                    "executable",
                ] {
                    let parent = tempfile::tempdir().unwrap();
                    let (_, source, root) = fixture(parent.path(), 2);
                    assert_index_calls(
                        &run_counted_legacy(&cx, parent.path(), "index").await,
                        2,
                        2,
                    );
                    let evidence_path = root.join(LEGACY_RECEIPT_FILE);
                    match mutation {
                        "vector" => {
                            let mut index = frankensearch_index::VectorIndex::open(
                                &root.join(super::super::super::FSFS_VECTOR_INDEX_FILE),
                            )
                            .unwrap();
                            index.soft_delete_batch(&["doc-0.md"]).unwrap();
                        }
                        "lexical" | "lexical_ghost" => {
                            let lexical = FsfsRuntime::resolve_lexical_engine(&root)
                                .unwrap()
                                .engine_dir()
                                .unwrap();
                            let index = frankensearch_quill::QuillIndex::create(
                                &cx,
                                lexical,
                                frankensearch_quill::QuillConfig::default(),
                            )
                            .await
                            .unwrap();
                            if mutation == "lexical" {
                                // Rewrite a live row: remove it before adding
                                // different stored content under the same id.
                                assert!(index.delete_document(&cx, "doc-0.md").await.unwrap());
                                index.commit(&cx).await.unwrap();
                            }
                            index
                                .index_document(
                                    &cx,
                                    &frankensearch_core::IndexableDocument::new(
                                        if mutation == "lexical_ghost" {
                                            "ghost.md"
                                        } else {
                                            "doc-0.md"
                                        },
                                        "sharedtoken different stored lexical content",
                                    ),
                                )
                                .await
                                .unwrap();
                            index.commit(&cx).await.unwrap();
                        }
                        "malformed" => fs::write(&evidence_path, b"{truncated").unwrap(),
                        _ => {
                            let mut evidence: LegacyReuseReceipt =
                                read_json(&cx, &evidence_path).unwrap();
                            match mutation {
                                "receipt" => {
                                    let path = source.join("doc-0.md");
                                    let modified = fs::metadata(&path).unwrap().modified().unwrap();
                                    let replacement = b"sharedtoken revision 0";
                                    fs::write(&path, replacement).unwrap();
                                    File::options()
                                        .write(true)
                                        .open(&path)
                                        .unwrap()
                                        .set_times(std::fs::FileTimes::new().set_modified(modified))
                                        .unwrap();
                                    evidence
                                        .receipt
                                        .checkpoint
                                        .files
                                        .get_mut("doc-0.md")
                                        .unwrap()
                                        .content_hash_hex = content_sha256_hex(replacement);
                                }
                                "version" => evidence.version += 1,
                                "executable" => {
                                    evidence.receipt.executable_sha256 = Some("0".repeat(64));
                                    // Even a self-consistent artifact witness cannot
                                    // grant reuse under another compiled producer.
                                    evidence.state_sha256 =
                                        legacy_state_digest(&cx, &root, &evidence.receipt).unwrap();
                                }
                                _ => unreachable!(),
                            }
                            fs::write(&evidence_path, serde_json::to_vec(&evidence).unwrap())
                                .unwrap();
                        }
                    }
                    let rebuilt = run_counted_legacy(&cx, parent.path(), "index").await;
                    assert_index_calls(&rebuilt, 2, 2);
                    let warm = run_counted_legacy(&cx, parent.path(), "index").await;
                    assert_index_calls(&warm, 0, 0);
                }
            });
        }

        #[test]
        fn completed_legacy_force_removes_uncheckpointed_lexical_rows() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (_, _, root) = fixture(parent.path(), 2);
                run_counted_legacy(&cx, parent.path(), "index").await;
                let lexical = FsfsRuntime::resolve_lexical_engine(&root)
                    .unwrap()
                    .engine_dir()
                    .unwrap();
                let index = frankensearch_quill::QuillIndex::create(
                    &cx,
                    lexical,
                    frankensearch_quill::QuillConfig::default(),
                )
                .await
                .unwrap();
                index
                    .index_document(
                        &cx,
                        &frankensearch_core::IndexableDocument::new(
                            "ghost.md",
                            "sharedtoken ghost",
                        ),
                    )
                    .await
                    .unwrap();
                index.commit(&cx).await.unwrap();
                assert_eq!(
                    index.search_results(&cx, "sharedtoken", 20).unwrap().len(),
                    3
                );
                drop(index);
                let evidence: LegacyReuseReceipt =
                    read_json(&cx, &root.join(LEGACY_RECEIPT_FILE)).unwrap();
                let mut interrupted = evidence.receipt.checkpoint;
                interrupted.artifacts_durable = false;
                write_indexing_checkpoint(&root, &interrupted).unwrap();
                assert_index_calls(&run_counted_legacy(&cx, parent.path(), "force").await, 2, 2);
            });
        }

        /// Use fresh producer instances for each run. The execution helper's
        /// restart suite independently proves the cross-process boundary; these
        /// controls exercise both tiers and the final, previously omitted batch.
        async fn run_counted(cx: &Cx, parent: &Path, operation: &str) -> serde_json::Value {
            let (mut runtime, source, root) = fixture(parent, 0);
            runtime.config.search.fast_only = false;
            runtime.config.search.quality_timeout_ms = 5_000;
            "reuse-quality".clone_into(&mut runtime.config.indexing.quality_model);
            runtime.cli_input.full_reindex = operation == "force";
            let fast = Arc::new(CountingEmbedder::new(
                "reuse-fast",
                4,
                operation == "fast_drift",
            ));
            let quality = Arc::new(CountingEmbedder::new(
                "reuse-quality",
                6,
                operation == "quality_drift",
            ));
            let _restore = RestoreEmbedders::install(fast.clone(), quality.clone());
            let generation = if operation == "append" {
                let input = parent.join("append.jsonl");
                fs::write(
                    &input,
                    "{\"id\":\"doc-0.md\",\"text\":\"appended replacement body\"}\n",
                )
                .unwrap();
                runtime.cli_input.command = CliCommand::AppendBatch;
                runtime.cli_input.input_file = Some(input);
                let (publication, count) =
                    runtime.append_retained_generation(cx, &root).await.unwrap();
                assert_eq!(count, 1);
                let Some(GenerationPublication::Durable(generation)) = publication else {
                    panic!("append must publish durably"); // ubs:ignore — cfg(test) assertion.
                };
                generation
            } else {
                publish(&runtime, cx, &root).await
            };
            let report = serde_json::json!({
                "generation": generation.id(),
                "fast_documents": fast.counts.documents.load(Ordering::SeqCst),
                "quality_documents": quality.counts.documents.load(Ordering::SeqCst),
                "fast_probes": fast.counts.probes.load(Ordering::SeqCst),
                "quality_probes": quality.counts.probes.load(Ordering::SeqCst),
                "fast_identity": fast.identity.fingerprint(),
                "quality_identity": quality.identity.fingerprint(),
                "checkpoint_present": generation.path().join(FSFS_CHECKPOINT_FILE).exists(),
                "receipt_present": generation.path().join(RECEIPT_FILE).exists(),
            });
            if operation != "append" {
                for (relative, producer) in [
                    (super::super::super::FSFS_VECTOR_INDEX_FILE, fast.as_ref()),
                    (
                        super::super::super::FSFS_VECTOR_QUALITY_INDEX_FILE,
                        quality.as_ref(),
                    ),
                ] {
                    let index = frankensearch_index::VectorIndex::open_read_only(
                        &generation.path().join(relative),
                    )
                    .unwrap();
                    assert_eq!(index.embedder_revision(), producer.identity.fingerprint());
                    assert_eq!(index.live_doc_ids().unwrap().len(), 2);
                    for row in 0..index.record_count() {
                        let text =
                            fs::read_to_string(source.join(index.doc_id_at(row).unwrap())).unwrap();
                        let expected =
                            producer.vector(&DefaultCanonicalizer::default().canonicalize(&text));
                        assert_eq!(index.vector_at_f32(row).unwrap(), expected);
                    }
                }
                let mut reader = runtime.open_retained_search(cx, &root).await.unwrap();
                let phases = reader.search(cx, "sharedtoken", 10).await.unwrap();
                assert_eq!(phases.last().unwrap().hits.len(), 2);
            }
            report
        }

        #[test]
        fn completed_two_tier_final_batch_reuses_and_force_recomputes_all_inputs() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                let (_, source, _) = fixture(parent.path(), 2);
                let first = run_counted(&cx, parent.path(), "index").await;
                assert_index_calls(&first, 2, 2);
                let unchanged = run_counted(&cx, parent.path(), "index").await;
                assert_index_calls(&unchanged, 0, 0);
                assert_ne!(
                    first["generation"], unchanged["generation"],
                    "reuse still publishes an isolated successor"
                );
                let forced = run_counted(&cx, parent.path(), "force").await;
                assert_index_calls(&forced, 2, 2);

                let path = source.join("doc-0.md");
                let before = fs::metadata(&path).unwrap();
                fs::write(&path, "sharedtoken revision 0").unwrap();
                File::options()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_times(std::fs::FileTimes::new().set_modified(before.modified().unwrap()))
                    .unwrap();
                assert_eq!(fs::metadata(&path).unwrap().len(), before.len());
                assert_eq!(
                    fs::metadata(&path).unwrap().modified().unwrap(),
                    before.modified().unwrap()
                );
                let changed = run_counted(&cx, parent.path(), "index").await;
                assert_index_calls(&changed, 1, 1);
                let warm = run_counted(&cx, parent.path(), "index").await;
                assert_index_calls(&warm, 0, 0);
            });
        }

        #[test]
        fn independent_producer_drift_and_append_invalidate_completed_reuse() {
            run_test_with_cx(|cx| async move {
                let parent = tempfile::tempdir().unwrap();
                fixture(parent.path(), 2);
                let first = run_counted(&cx, parent.path(), "index").await;
                let quality_changed = run_counted(&cx, parent.path(), "quality_drift").await;
                assert_index_calls(&quality_changed, 0, 2);
                assert_eq!(first["fast_identity"], quality_changed["fast_identity"]);
                assert_ne!(
                    first["quality_identity"],
                    quality_changed["quality_identity"]
                );
                // Return to the original quality producer, then change only fast.
                let restored = run_counted(&cx, parent.path(), "index").await;
                assert_index_calls(&restored, 0, 2);
                let fast_changed = run_counted(&cx, parent.path(), "fast_drift").await;
                assert_index_calls(&fast_changed, 2, 2);
                assert_ne!(restored["fast_identity"], fast_changed["fast_identity"]);
                assert_eq!(
                    restored["quality_identity"],
                    fast_changed["quality_identity"]
                );
                let restored = run_counted(&cx, parent.path(), "index").await;
                assert_index_calls(&restored, 2, 2);
                let appended = run_counted(&cx, parent.path(), "append").await;
                assert_eq!(appended["fast_documents"], 1);
                assert_eq!(appended["quality_documents"], 1);
                assert_eq!(
                    appended["receipt_present"], false,
                    "explicit mutation must not copy old source evidence"
                );
                let rebuilt = run_counted(&cx, parent.path(), "index").await;
                assert_index_calls(&rebuilt, 2, 2);
                let reused = run_counted(&cx, parent.path(), "index").await;
                assert_index_calls(&reused, 0, 0);
            });
        }
    }

    async fn publish(runtime: &FsfsRuntime, cx: &Cx, root: &Path) -> PublishedGeneration {
        match runtime.rebuild_retained_generation(cx, root).await.unwrap() {
            GenerationPublication::Durable(generation) => generation,
            other @ GenerationPublication::VisibleButDurabilityUncertain { .. } => {
                panic!("expected durable publication: {other:?}") // ubs:ignore — cfg(test) assertion.
            }
        }
    }

    fn candidate(runtime: &FsfsRuntime, path: &Path) -> FsfsRuntime {
        let mut input = runtime.cli_input.clone();
        input.index_dir = Some(path.to_path_buf());
        runtime.clone().with_cli_input(input)
    }

    fn publish_receipt_variant(
        cx: &Cx,
        runtime: &FsfsRuntime,
        root: &Path,
        receipt_bytes: Option<&[u8]>,
    ) -> PublishedGeneration {
        let store = CompleteGenerationStore::open(cx, root).unwrap();
        let build = store.begin(cx).unwrap();
        let next = candidate(runtime, build.path());
        copy_selected_generation(cx, &next, &store, build.path()).unwrap();
        if let Some(bytes) = receipt_bytes {
            fs::write(build.path().join(RECEIPT_FILE), bytes).unwrap();
        }
        // The fixture deliberately seals an unusable optimization receipt in
        // an otherwise valid generation. It never tampers with a sealed bundle.
        let publication = build
            .publish(cx, |_, path| {
                FsfsRuntime::validate_search_generation_at_root(path, SearchExecutionMode::Full)
            })
            .unwrap();
        let GenerationPublication::Durable(generation) = publication else {
            panic!("fixture publication must be durable"); // ubs:ignore — cfg(test) assertion.
        };
        generation
    }

    #[test]
    fn missing_or_unsupported_receipts_start_cold_and_malformed_receipts_refuse() {
        run_test_with_cx(|cx| async move {
            for (encoded, malformed) in [
                (None, false),
                (
                    Some(&b"{\"version\":1,\"session\":\"historical\"}"[..]),
                    false,
                ),
                (Some(&b"{\"version\":3}"[..]), false),
                (Some(&b"not json"[..]), true),
                (Some(&b"{\"version\":2}"[..]), true),
            ] {
                let parent = tempfile::tempdir().unwrap();
                let (runtime, _, root) = fixture(parent.path(), 2);
                publish(&runtime, &cx, &root).await;
                let selected = publish_receipt_variant(&cx, &runtime, &root, encoded);
                let store = CompleteGenerationStore::open(&cx, &root).unwrap();
                let build = store.begin(&cx).unwrap();
                let next = candidate(&runtime, build.path());
                let result = seed_candidate(&cx, &next, &store, build.path());
                if malformed {
                    assert!(
                        result.is_err(),
                        "invalid current evidence must not be accepted"
                    );
                } else {
                    assert_eq!(result.unwrap(), 0, "unsupported evidence grants no reuse");
                }
                assert_eq!(fs::read_dir(build.path()).unwrap().count(), 0);
                assert_eq!(store.active(&cx).unwrap(), Some(selected));
            }
        });
    }

    #[test]
    fn cancelled_reuse_and_existing_interruption_checkpoint_never_get_overwritten() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(parent.path(), 2);
            let selected = publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let build = store.begin(&cx).unwrap();
            let next = candidate(&runtime, build.path());
            cx.set_cancel_requested(true);
            assert!(matches!(
                seed_candidate(&cx, &next, &store, build.path()),
                Err(SearchError::Cancelled { .. }),
            ));
            assert_eq!(fs::read_dir(build.path()).unwrap().count(), 0);
            cx.set_cancel_requested(false);
            let checkpoint = build.path().join(FSFS_CHECKPOINT_FILE);
            fs::write(&checkpoint, b"existing interrupted work").unwrap();
            assert!(seed_candidate(&cx, &next, &store, build.path()).is_err());
            assert_eq!(fs::read(&checkpoint).unwrap(), b"existing interrupted work");
            assert_eq!(fs::read_dir(build.path()).unwrap().count(), 1);
            assert_eq!(store.active(&cx).unwrap(), Some(selected));
        });
    }

    #[test]
    fn completed_input_evidence_covers_the_final_uncheckpointed_batch() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(parent.path(), 2);
            let generation = publish(&runtime, &cx, &root).await;
            let receipt: ReuseReceipt =
                read_json(&cx, &generation.path().join(RECEIPT_FILE)).unwrap();
            assert_eq!(receipt.checkpoint.files.len(), 2);
            assert_eq!(
                receipt
                    .checkpoint
                    .files
                    .values()
                    .filter(|entry| proven(entry))
                    .count(),
                2
            );
            for (key, entry) in &receipt.checkpoint.files {
                assert_eq!(
                    entry.content_hash_hex,
                    content_sha256_hex(
                        &fs::read(Path::new(&receipt.checkpoint.target_root).join(key),).unwrap()
                    )
                );
                assert!(entry.semantic_indexed);
                assert!(entry.lexical_indexed);
            }
            assert!(!generation.path().join(FSFS_CHECKPOINT_FILE).exists());
            assert!(
                FsfsRuntime::read_checkpoint_manifest_generation(
                    generation.path(),
                    &receipt.checkpoint
                )
                .unwrap()
                .is_some()
            );
        });
    }

    #[test]
    fn reuse_seed_rebases_checkpoint_without_mutating_retained_generation() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(parent.path(), 4);
            let predecessor = publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let build = store.begin(&cx).unwrap();
            let next = candidate(&runtime, build.path());
            assert_eq!(seed_candidate(&cx, &next, &store, build.path()).unwrap(), 4);
            let checkpoint: IndexingCheckpoint =
                read_json(&cx, &build.path().join(FSFS_CHECKPOINT_FILE)).unwrap();
            assert_eq!(checkpoint.index_root, build.path().display().to_string());
            assert!(
                FsfsRuntime::read_checkpoint_manifest_generation(build.path(), &checkpoint)
                    .unwrap()
                    .is_some()
            );
            assert!(!build.path().join(COMPLETE_GENERATION_MANIFEST).exists());
            assert!(!build.path().join(RECEIPT_FILE).exists());
            assert_eq!(store.active(&cx).unwrap(), Some(predecessor.clone()));
            assert_eq!(
                FsfsRuntime::read_index_sentinel(predecessor.path())
                    .unwrap()
                    .unwrap()
                    .index_root,
                predecessor.path().display().to_string()
            );
        });
    }

    #[test]
    fn receipt_requires_matching_execution_configuration_and_source_and_full_bypasses_it() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(parent.path(), 4);
            let predecessor = publish(&runtime, &cx, &root).await;
            let mut receipt: ReuseReceipt =
                read_json(&cx, &predecessor.path().join(RECEIPT_FILE)).unwrap();
            assert!(compatible_receipt(&runtime, &receipt).unwrap());
            let executable = receipt.executable_sha256.clone();
            if let Some(digest) = receipt.executable_sha256.as_mut() {
                digest.push('x');
            } else {
                receipt.session.push('x');
            }
            assert!(!compatible_receipt(&runtime, &receipt).unwrap());
            receipt.executable_sha256 = executable;
            session_id().unwrap().clone_into(&mut receipt.session);
            receipt.configuration_sha256.push('x');
            assert!(!compatible_receipt(&runtime, &receipt).unwrap());
            receipt.configuration_sha256 = configuration_digest(&runtime).unwrap();
            receipt.checkpoint.target_root.push('x');
            assert!(!compatible_receipt(&runtime, &receipt).unwrap());
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let build = store.begin(&cx).unwrap();
            let mut next = candidate(&runtime, build.path());
            next.cli_input.full_reindex = true;
            assert_eq!(seed_candidate(&cx, &next, &store, build.path()).unwrap(), 0);
            assert_eq!(fs::read_dir(build.path()).unwrap().count(), 0);
            assert_eq!(store.active(&cx).unwrap(), Some(predecessor));
        });
    }

    #[test]
    fn execution_scope_never_downgrades_and_old_receipts_are_cold_not_corrupt() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(parent.path(), 4);
            let predecessor = publish(&runtime, &cx, &root).await;
            let mut receipt: ReuseReceipt =
                read_json(&cx, &predecessor.path().join(RECEIPT_FILE)).unwrap();
            let digest = "a".repeat(64);
            receipt.executable_sha256 = Some(digest.clone());
            assert!(compatible_execution(
                &receipt,
                Some(&digest),
                "another process"
            ));
            assert!(!compatible_execution(
                &receipt,
                Some(&"b".repeat(64)),
                &receipt.session
            ));
            assert!(!compatible_execution(&receipt, None, &receipt.session));
            receipt.executable_sha256 = None;
            assert!(compatible_execution(&receipt, None, &receipt.session));
            assert!(!compatible_execution(&receipt, None, "another process"));
            assert!(!compatible_execution(
                &receipt,
                Some(&digest),
                &receipt.session
            ));
            receipt.version = 1;
            let legacy = serde_json::to_vec(&receipt).unwrap();
            let decoded: ReuseReceipt = serde_json::from_slice(&legacy).unwrap();
            assert!(decoded.executable_sha256.is_none());
            assert!(!compatible_receipt(&runtime, &decoded).unwrap());
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(predecessor)
            );
        });
    }

    #[test]
    fn source_hash_not_metadata_decides_semantic_reuse() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(parent.path(), 4);
            let generation = publish(&runtime, &cx, &root).await;
            let receipt: ReuseReceipt =
                read_json(&cx, &generation.path().join(RECEIPT_FILE)).unwrap();
            let (key, entry) = receipt.checkpoint.files.first_key_value().unwrap();
            let manifests = FsfsRuntime::read_matching_manifest_generation(generation.path())
                .unwrap()
                .unwrap();
            let manifest = &manifests[key];
            let path = source.join("doc-0.md");
            let original_modified = fs::metadata(&path).unwrap().modified().unwrap();
            let probe = IndexCandidate {
                file_path: path.clone(),
                file_key: key.clone(),
                modified_ms: u64::try_from(entry.revision).unwrap(),
                ingestion_class: IngestionClass::FullSemanticLexical,
            };
            let live_ids = HashSet::from([key.clone()]);
            let classify = |hash: &str| {
                checkpoint_entry_reuse(
                    &receipt.checkpoint,
                    entry,
                    manifest,
                    &probe,
                    hash,
                    &receipt.checkpoint.embedder_id,
                    receipt.checkpoint.embedder_dimension,
                    false,
                    &live_ids,
                )
            };
            assert_eq!(
                classify(&content_sha256_hex(&fs::read(&path).unwrap())),
                CheckpointReuse::Complete
            );
            fs::write(&path, b"changed input document").unwrap();
            File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(original_modified))
                .unwrap();
            assert_ne!(
                classify(&content_sha256_hex(&fs::read(path).unwrap())),
                CheckpointReuse::Complete
            );
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(generation)
            );
        });
    }

    #[test]
    fn seeded_rebuild_reconciles_additions_removals_and_changes_while_old_reader_is_open() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(parent.path(), 4);
            let first = publish(&runtime, &cx, &root).await;
            let mut pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
            fs::write(
                source.join("doc-0.md"),
                "sharedtoken changed first document",
            )
            .unwrap();
            fs::rename(
                source.join("doc-1.md"),
                parent.path().join("outside-source.md"),
            )
            .unwrap();
            fs::write(source.join("doc-4.md"), "sharedtoken new fourth document").unwrap();
            let second = publish(&runtime, &cx, &root).await;
            assert_ne!(first.id(), second.id());
            assert_eq!(
                pinned
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                4
            );
            let mut current = runtime.open_retained_search(&cx, &root).await.unwrap();
            let phases = current.search(&cx, "sharedtoken", 10).await.unwrap();
            let hits = &phases.last().unwrap().hits;
            assert_eq!(hits.len(), 4);
            assert!(hits.iter().any(|hit| hit.path.ends_with("doc-4.md")));
            assert!(!hits.iter().any(|hit| hit.path.ends_with("doc-1.md")));
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(second)
            );
            assert!(first.path().is_dir());
        });
    }
}
