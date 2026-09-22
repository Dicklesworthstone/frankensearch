//! Reuse proven indexing work in a fresh complete generation.
//!
//! The ordinary indexer already revalidates source hashes and both producers
//! when resuming a checkpoint. Keep its observed inputs, copy the predecessor
//! to independent files, and feed that existing path rather than introduce a
//! second embedding cache. No serving artifact is opened for writing.
//!
//! Receipts are session-local: a different process starts cold. This binds
//! reuse to the same compiled extraction/canonicalization implementation, not
//! merely a package version that can stay unchanged across source edits. The
//! final uncheckpointed batch has no retained input evidence and is recomputed.

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
    INDEXING_CHECKPOINT_SCHEMA_VERSION, IndexingCheckpoint, IndexingProgressStage,
    SearchExecutionMode, content_sha256_hex, retained_search_checkpoint, write_indexing_checkpoint,
};
use crate::generation_store::{
    COMPLETE_GENERATION_MANIFEST, CompleteGenerationStore, PublishedGeneration,
};

mod append_input;
pub(super) use append_input::read_append_documents;

const RECEIPT_FILE: &str = "FSFS-REUSE.json";
const RECEIPT_VERSION: u16 = 1;
const MAX_RECEIPT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_COPY_DEPTH: usize = 64;
const MAX_COPY_ENTRIES: usize = 200_000;
static SESSION: OnceLock<String> = OnceLock::new();

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReuseReceipt {
    version: u16,
    session: String,
    configuration_sha256: String,
    checkpoint: IndexingCheckpoint,
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

fn compatible_receipt(runtime: &FsfsRuntime, receipt: &ReuseReceipt) -> SearchResult<bool> {
    Ok(receipt.version == RECEIPT_VERSION
        && receipt.session == session_id()?
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
        seed_candidate(cx, self, store, candidate_root)?;
        let mut observed = None;
        let payload = Box::pin(self.run_one_shot_index_scaffold_internal(
            cx,
            CliCommand::Index,
            |progress| {
                if matches!(progress.stage, IndexingProgressStage::Finalizing) {
                    // This file contains input hashes actually observed by the
                    // indexer, not hashes of files reread after indexing. The
                    // pipeline can overwrite/remove it after this callback.
                    observed = Some(read_json::<IndexingCheckpoint>(
                        cx,
                        &candidate_root.join(FSFS_CHECKPOINT_FILE),
                    )?);
                }
                Ok(())
            },
            false,
        ))
        .await?;
        Self::validate_search_generation_at_root(candidate_root, SearchExecutionMode::Full)?;
        if let Some(checkpoint) = observed {
            let receipt = completed_receipt(self, candidate_root, checkpoint, &payload)?;
            write_receipt(cx, candidate_root, &receipt)?;
        }
        Ok(())
    }
}

fn completed_receipt(
    runtime: &FsfsRuntime,
    root: &Path,
    mut checkpoint: IndexingCheckpoint,
    payload: &FsfsIndexPayload,
) -> SearchResult<ReuseReceipt> {
    let final_state = &payload.generation;
    if !final_state.generation_complete
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
    let mut observed = std::mem::take(&mut checkpoint.files);
    for (key, manifest) in manifests {
        let entry = observed.remove(&key).filter(|entry| {
            entry.revision == manifest.revision
                && entry.ingestion_class == manifest.ingestion_class
                && entry.canonical_bytes == manifest.canonical_bytes
                && entry.reason_code == manifest.reason_code
        });
        // Supply membership for the existing checkpoint validator, but never
        // invent a successful embedding or source hash for the uncovered tail.
        let entry = entry.unwrap_or(CheckpointFileEntry {
            revision: manifest.revision,
            ingestion_class: manifest.ingestion_class,
            canonical_bytes: manifest.canonical_bytes,
            reason_code: manifest.reason_code,
            lexical_indexed: false,
            semantic_indexed: false,
            content_hash_hex: String::new(),
        });
        checkpoint.files.insert(key, entry);
    }
    checkpoint.artifacts_durable = true;
    checkpoint
        .source_hash_hex
        .clone_from(&final_state.source_hash_hex);
    checkpoint
        .reason_codes
        .clone_from(&final_state.reason_codes);
    checkpoint.discovered_files = final_state.discovered_files;
    checkpoint.skipped_files = final_state.skipped_files;
    if FsfsRuntime::read_checkpoint_manifest_generation(root, &checkpoint)?.is_none() {
        return Err(reuse_error(
            "retained input evidence disagrees with final generation metadata",
        ));
    }
    Ok(ReuseReceipt {
        version: RECEIPT_VERSION,
        session: session_id()?.to_owned(),
        configuration_sha256: configuration_digest(runtime)?,
        checkpoint,
    })
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
    let mut receipt: ReuseReceipt = read_json(cx, &path)?;
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
        eligible_semantic_files = eligible,
        copied_files = stats.files,
        copied_bytes = stats.bytes,
        "seeded isolated generation; ordinary source-hash and producer checks decide actual reuse"
    );
    Ok(eligible)
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
    let mut entries = fs::read_dir(source)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        retained_search_checkpoint(cx)?;
        stats.entries += 1;
        if stats.entries > MAX_COPY_ENTRIES {
            return Err(reuse_error("predecessor copy exceeds its entry limit"));
        }
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
        fs::create_dir(&source).unwrap();
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

    #[test]
    fn completed_input_evidence_never_invents_hashes_for_uncheckpointed_tail() {
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
                1
            );
            let tail = receipt.checkpoint.files.values().last().unwrap();
            assert!(tail.content_hash_hex.is_empty());
            assert!(!tail.semantic_indexed);
            assert!(!tail.lexical_indexed);
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
    fn receipt_requires_same_process_configuration_and_source_and_full_bypasses_it() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(parent.path(), 4);
            let predecessor = publish(&runtime, &cx, &root).await;
            let mut receipt: ReuseReceipt =
                read_json(&cx, &predecessor.path().join(RECEIPT_FILE)).unwrap();
            assert!(compatible_receipt(&runtime, &receipt).unwrap());
            receipt.session.push('x');
            assert!(!compatible_receipt(&runtime, &receipt).unwrap());
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
