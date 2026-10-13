//! Source-authorized, bounded deltas for the complete-generation watcher.
//!
//! Notifications only invalidate reuse. A complete observation determines
//! membership, and the owner checks it again after sealing. Unsupported source
//! policies return to the normal indexer before a candidate is allocated.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Read};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};

use super::{
    DirtyWindow, SourceObservation, SourceRoot, SourceStamp, observation_io, source_changed,
};
use crate::config::{DegradationOverrideMode, IngestionClass};
use crate::file_classification::{DetectedType, FileClassificationDecision, IngestAction};
use crate::runtime::retained_batch::{
    RetainedMutation, SourceAttributes, SourceBatch, prepare_source_batch,
};
use crate::runtime::{
    FsfsRuntime, classify_file_for_ingest, is_pdf_file, normalize_file_key_for_index,
    retained_search_checkpoint, system_time_to_ms, watched_document,
};

const MAX_CHANGED_PATHS: usize = 128;
const MAX_SOURCE_BYTES: u64 = 8 * 1024 * 1024;
// Leave headroom for two canonical representations and their source metadata
// within the batch writer's 64 MiB retained-payload limit.
const MAX_RAW_BYTES: u64 = 16 * 1024 * 1024;

/// None is an unsupported/oversized plan, not an empty authoritative corpus.
fn changed_paths(
    cx: &Cx,
    baseline: &SourceObservation,
    current: &SourceObservation,
    hint: Option<&DirtyWindow>,
) -> SearchResult<Option<BTreeSet<PathBuf>>> {
    if hint.is_some_and(|hint| hint.force_rebuild) {
        return Ok(None);
    }
    // Every map is ordered by path, so one merge pass compares each path about
    // once; a lookup per map compared long absolute paths log n times each.
    let mut before = Cursor::new(&baseline.stamps);
    let mut before_classes = Cursor::new(&baseline.classes);
    let mut classes = Cursor::new(&current.classes);
    let mut changed = BTreeSet::new();
    for (path, stamp) in &current.stamps {
        retained_search_checkpoint(cx)?;
        let mut deleted = Vec::new();
        let unchanged = before.seek(path, |gone| deleted.push(gone)) == Some(stamp)
            && before_classes.seek(path, |_| {}) == classes.seek(path, |_| {})
            && !hint.is_some_and(|hint| hint.paths.iter().any(|hinted| path.starts_with(hinted)));
        for path in deleted.into_iter().chain((!unchanged).then_some(path)) {
            changed.insert(path.clone());
            if changed.len() > MAX_CHANGED_PATHS {
                return Ok(None);
            }
        }
    }
    for (path, _) in before.entries {
        retained_search_checkpoint(cx)?;
        changed.insert(path.clone());
        if changed.len() > MAX_CHANGED_PATHS {
            return Ok(None);
        }
    }
    Ok(Some(changed))
}

/// A forward position in one path-ordered map, sought by ascending paths.
struct Cursor<'a, V> {
    entries: std::iter::Peekable<std::collections::btree_map::Iter<'a, PathBuf, V>>,
}

impl<'a, V> Cursor<'a, V> {
    fn new(map: &'a BTreeMap<PathBuf, V>) -> Self {
        Self {
            entries: map.iter().peekable(),
        }
    }

    /// The value at `path`, handing every smaller key passed over to `skipped`.
    fn seek(&mut self, path: &Path, mut skipped: impl FnMut(&'a PathBuf)) -> Option<&'a V> {
        while let Some(&(key, value)) = self.entries.peek() {
            match key.as_path().cmp(path) {
                std::cmp::Ordering::Less => skipped(key),
                std::cmp::Ordering::Equal => {
                    self.entries.next();
                    return Some(value);
                }
                std::cmp::Ordering::Greater => return None,
            }
            self.entries.next();
        }
        None
    }
}

fn unambiguous_source(root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(root) else {
        return false;
    };
    let Some(text) = relative.to_str() else {
        return false;
    };
    // The original indexer has lossy filename normalization. Do not use a
    // potentially colliding spelling as authority for an exact-ID deletion.
    !(text.is_empty()
        || text.contains(['\\', '\0'])
        || text.len() > usize::from(u16::MAX)
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_))))
}

fn source_id(root: &Path, path: &Path) -> Option<String> {
    unambiguous_source(root, path).then(|| normalize_file_key_for_index(path, root))
}

/// Discovery exclusion is not physical absence. A directory, special file or
/// dangling symlink replacing an indexed source keeps the full indexer's policy
/// path; only `NotFound` admits an incremental removal. Never swallow I/O errors.
fn physically_absent(cx: &Cx, path: &Path) -> SearchResult<bool> {
    retained_search_checkpoint(cx)?;
    let metadata = fs::symlink_metadata(path);
    retained_search_checkpoint(cx)?;
    match metadata {
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(true),
        Err(error) => Err(error.into()),
        Ok(_) => Ok(false),
    }
}

fn recheck_removals(cx: &Cx, paths: &[PathBuf]) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    for path in paths {
        if !physically_absent(cx, path)? {
            return Err(source_changed());
        }
    }
    retained_search_checkpoint(cx)
}

/// Metadata-only discovery authorizes a path/title row, never a payload read.
/// This accepts large and non-text regular files, but not a raced symlink,
/// directory or special file. The owner repeats the check before publication.
fn metadata_modified_ms(cx: &Cx, path: &Path, expected: &SourceStamp) -> SearchResult<u64> {
    retained_search_checkpoint(cx)?;
    let metadata = fs::symlink_metadata(path);
    retained_search_checkpoint(cx)?;
    let metadata = metadata.map_err(observation_io)?;
    if !metadata.is_file() || &SourceStamp::from_metadata(&metadata) != expected {
        return Err(source_changed());
    }
    Ok(metadata
        .modified()
        .ok()
        .map(system_time_to_ms)
        .unwrap_or_default())
}

fn recheck_metadata_sources(cx: &Cx, sources: &[(PathBuf, SourceStamp)]) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    for (path, stamp) in sources {
        metadata_modified_ms(cx, path, stamp)?;
    }
    retained_search_checkpoint(cx)
}

/// Read exactly the observed regular file, with before/after descriptor and
/// pathname stamps. `O_NONBLOCK` prevents a raced FIFO from hanging the lane.
/// This runs on Asupersync's existing blocking pool, not a detached worker.
fn read_source(cx: &Cx, path: &Path, expected: &SourceStamp) -> SearchResult<(Vec<u8>, u64)> {
    retained_search_checkpoint(cx)?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(observation_io)?;
    let opened = file.metadata()?;
    if !opened.is_file() || &SourceStamp::from_metadata(&opened) != expected {
        return Err(source_changed());
    }
    let mut bytes = Vec::new();
    let capacity = usize::try_from(expected.bytes)
        .ok()
        .filter(|_| expected.bytes <= MAX_SOURCE_BYTES)
        .ok_or_else(source_changed)?;
    bytes.try_reserve_exact(capacity).map_err(|_| {
        SearchError::Io(std::io::Error::other("cannot reserve bounded source input"))
    })?;
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        retained_search_checkpoint(cx)?;
        let read = file.read(&mut buffer);
        retained_search_checkpoint(cx)?;
        let count = match read {
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            other => other?,
        };
        if count == 0 {
            break;
        }
        if count > capacity.saturating_sub(bytes.len()) {
            return Err(source_changed());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let current = fs::symlink_metadata(path).map_err(observation_io)?;
    if bytes.len() != capacity
        || !current.is_file()
        || &SourceStamp::from_metadata(&file.metadata()?) != expected
        || &SourceStamp::from_metadata(&current) != expected
    {
        return Err(source_changed());
    }
    retained_search_checkpoint(cx)?;
    let modified_ms = opened
        .modified()
        .ok()
        .map(system_time_to_ms)
        .unwrap_or_default();
    Ok((bytes, modified_ms))
}

struct ReadText {
    text: String,
    modified_ms: u64,
    classification: FileClassificationDecision,
}

type TextReads = BTreeMap<PathBuf, ReadText>;
type ReadJob = Box<dyn FnOnce() -> SearchResult<Option<TextReads>> + Send>;

/// Only this future owns the right to keep reading. The blocking job holds the
/// stop flag, not this guard, so a queued or running job cannot keep itself alive
/// after its caller abandons the await. Do not cancel the caller's shared Cx.
struct ReadGuard(Arc<AtomicBool>);

impl Drop for ReadGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn read_checkpoint(cx: &Cx, abandoned: &AtomicBool) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    if abandoned.load(Ordering::Acquire) {
        return Err(SearchError::Cancelled {
            phase: "complete_watch.read_batch".to_owned(),
            reason: "the source-read future was abandoned".to_owned(),
        });
    }
    Ok(())
}

/// One caller-owned blocking-pool job for the complete bounded text delta.
/// Per-file awaits can each pay a current-thread runtime wake delay (bd-0wxjd).
/// Metadata-only files and removals are deliberately absent from these requests.
/// Empty text inventories submit no work, and cancellation dominates no-op and
/// unsupported plans. A dropped await stops queued and between-file work. A
/// regular-file read already executing can finish its bounded scalar operation;
/// this does not claim to preempt an OS read. No job can publish independently.
async fn read_text_batch(
    cx: &Cx,
    requests: Vec<(PathBuf, SourceStamp)>,
) -> SearchResult<Option<TextReads>> {
    read_text_batch_with(cx, requests, |job| {
        asupersync::runtime::spawn_blocking(job)
    })
    .await
}

/// One submission boundary shared with deterministic queue/drop tests. The
/// production caller above always uses the existing Asupersync blocking pool.
async fn read_text_batch_with<F, Fut>(
    cx: &Cx,
    requests: Vec<(PathBuf, SourceStamp)>,
    submit: F,
) -> SearchResult<Option<TextReads>>
where
    F: FnOnce(ReadJob) -> Fut,
    Fut: std::future::Future<Output = SearchResult<Option<TextReads>>>,
{
    retained_search_checkpoint(cx)?;
    if !text_batch_supported(cx, &requests)? {
        return Ok(None);
    }
    if requests.is_empty() {
        return Ok(Some(BTreeMap::new()));
    }
    let task_cx = cx.clone();
    let abandoned = Arc::new(AtomicBool::new(false));
    let guard = ReadGuard(Arc::clone(&abandoned));
    let job = Box::new(move || read_text_sources(&task_cx, &requests, &abandoned, read_source));
    let result = submit(job).await;
    retained_search_checkpoint(cx)?;
    drop(guard);
    result
}

/// Check the WHOLE request before any file is opened, not only as it is read.
/// This repeats the planner's byte/count admission at the owned-work boundary.
fn text_batch_supported(cx: &Cx, requests: &[(PathBuf, SourceStamp)]) -> SearchResult<bool> {
    retained_search_checkpoint(cx)?;
    if requests.len() > MAX_CHANGED_PATHS {
        return Ok(false);
    }
    let mut remaining = MAX_RAW_BYTES;
    let mut seen = BTreeSet::new();
    for (path, stamp) in requests {
        retained_search_checkpoint(cx)?;
        if stamp.bytes > MAX_SOURCE_BYTES || stamp.bytes > remaining || !seen.insert(path) {
            return Ok(false);
        }
        remaining -= stamp.bytes;
    }
    retained_search_checkpoint(cx)?;
    Ok(true)
}

/// Preserve every scalar descriptor/path stamp and classification check. A
/// later I/O failure or unsupported body discards all earlier results, never
/// returning a partial batch or retrying a failed read with weaker admission.
/// UTF-8 decoding consumes each byte buffer, so the total owned source payload
/// is at most MAX_RAW_BYTES; it is moved, not copied, into the mutation batch.
fn read_text_sources<F>(
    cx: &Cx,
    requests: &[(PathBuf, SourceStamp)],
    abandoned: &AtomicBool,
    mut read: F,
) -> SearchResult<Option<TextReads>>
where
    F: FnMut(&Cx, &Path, &SourceStamp) -> SearchResult<(Vec<u8>, u64)>,
{
    read_checkpoint(cx, abandoned)?;
    let mut results = BTreeMap::new();
    for (path, stamp) in requests {
        read_checkpoint(cx, abandoned)?;
        let response = read(cx, path, stamp);
        read_checkpoint(cx, abandoned)?;
        let (bytes, modified_ms) = response?;
        let classification = classify_file_for_ingest(path, &bytes);
        read_checkpoint(cx, abandoned)?;
        if classification.detected_type != DetectedType::Text
            || classification.ingest_action != IngestAction::Index
        {
            return Ok(None);
        }
        let text = String::from_utf8(bytes);
        read_checkpoint(cx, abandoned)?;
        let Ok(text) = text else {
            return Ok(None);
        };
        results.insert(
            path.clone(),
            ReadText {
                text,
                modified_ms,
                classification,
            },
        );
    }
    read_checkpoint(cx, abandoned)?;
    Ok(Some(results))
}

pub(super) async fn prepare(
    cx: &Cx,
    runtime: &FsfsRuntime,
    source: &SourceRoot,
    baseline: &SourceObservation,
    current: &SourceObservation,
    hint: Option<&DirtyWindow>,
) -> SearchResult<Option<SourceBatch>> {
    retained_search_checkpoint(cx)?;
    // Forced indexing, explicit degradation and link aliases retain the normal
    // indexer's admission/policy path. A delta must not bypass those controls.
    if runtime.cli_input.full_reindex
        || runtime.config.pressure.degradation_override != DegradationOverrideMode::Auto
        || runtime.config.discovery.follow_symlinks
    {
        return Ok(None);
    }
    // An untouched lossy name can collide with a changed valid name. Reject
    // that entire source cohort, not just ambiguous names in the delta.
    for path in baseline.stamps.keys().chain(current.stamps.keys()) {
        retained_search_checkpoint(cx)?;
        if !unambiguous_source(&source.path, path) {
            return Ok(None);
        }
    }
    let Some(paths) = changed_paths(cx, baseline, current, hint)? else {
        return Ok(None);
    };
    if paths.is_empty() {
        return Ok(None);
    }
    let mut raw_bytes = 0_u64;
    let mut text_requests = Vec::new();
    for path in &paths {
        retained_search_checkpoint(cx)?;
        if source_id(&source.path, path).is_none() {
            return Ok(None);
        }
        if let Some(stamp) = current.stamps.get(path) {
            match current.classes.get(path) {
                // No file payload is opened, decoded, hashed or embedded. Its
                // size/format cannot consume the content-reading budget.
                Some(IngestionClass::MetadataOnly) => continue,
                Some(IngestionClass::FullSemanticLexical | IngestionClass::LexicalOnly) => {}
                _ => return Ok(None),
            }
            if stamp.bytes > MAX_SOURCE_BYTES || is_pdf_file(path) {
                return Ok(None);
            }
            raw_bytes = raw_bytes.saturating_add(stamp.bytes);
            if raw_bytes > MAX_RAW_BYTES {
                return Ok(None);
            }
            text_requests.push((path.clone(), stamp.clone()));
        } else if !physically_absent(cx, path)? {
            // An ignore-policy change or a replacement with an unindexed type
            // requires normal reconciliation, not a claimed physical deletion.
            return Ok(None);
        }
    }
    source.check()?;
    let Some(mut text_reads) = read_text_batch(cx, text_requests).await? else {
        return Ok(None);
    };
    source.check()?;
    let mut operations = Vec::with_capacity(paths.len());
    let mut attributes = BTreeMap::new();
    let mut removals = Vec::new();
    let mut metadata_sources = Vec::new();
    for path in paths {
        retained_search_checkpoint(cx)?;
        let Some(id) = source_id(&source.path, &path) else {
            return Ok(None);
        };
        let Some(stamp) = current.stamps.get(&path) else {
            // A deletion comes from two complete observations, never from the
            // kind or ordering of a native notification. Retain its exact path
            // for the final physical-absence check, not a normalized ID alias.
            operations.push(RetainedMutation::Delete { id });
            removals.push(path);
            continue;
        };
        let Some(mut class) = current.classes.get(&path).copied() else {
            return Ok(None);
        };
        let (text, modified_ms, classification) = if class == IngestionClass::MetadataOnly {
            let modified_ms = metadata_modified_ms(cx, &path, stamp)?;
            metadata_sources.push((path.clone(), stamp.clone()));
            (id.clone(), modified_ms, None)
        } else {
            let read = text_reads.remove(&path).ok_or_else(source_changed)?;
            (read.text, read.modified_ms, Some(read.classification))
        };
        if runtime.lexical_only_indexing && class == IngestionClass::FullSemanticLexical {
            class = IngestionClass::LexicalOnly;
        }
        // Reuse the same metadata construction as ordinary watched documents.
        // Content preparation remains solely in prepare_source_batch below.
        let document = watched_document(
            &path,
            &id,
            "",
            class,
            classification.as_ref(),
            Some(modified_ms),
        )
        .await;
        retained_search_checkpoint(cx)?;
        attributes.insert(
            id.clone(),
            SourceAttributes {
                modified_ms,
                ingestion_class: class,
                title: document.title,
                metadata: document.metadata,
            },
        );
        operations.push(RetainedMutation::Upsert { id, text });
    }
    if !text_reads.is_empty() {
        return Err(source_changed());
    }
    source.check()?;
    let Some(batch) = prepare_source_batch(cx, &operations, &attributes)? else {
        return Ok(None);
    };
    Ok(Some(batch.with_source_check(move |cx| {
        // Run AFTER the owner's post-seal observation. A recreated file hidden
        // by an ignore rule can leave that observation unchanged; it still
        // invalidates the removal evidence. At most 128 paths are retained.
        recheck_removals(cx, &removals)?;
        recheck_metadata_sources(cx, &metadata_sources)
    })))
}

#[cfg(test)]
#[path = "delta_tests.rs"]
mod tests;

#[cfg(test)]
mod metadata_tests {
    use std::future::Future;
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::AtomicUsize;
    use std::task::{Context, Poll, Wake, Waker};
    use std::time::Instant;

    use asupersync::test_utils::run_test_with_cx;
    use notify::Watcher;

    use super::super::{
        Changes, CompleteWatchSession, DEBOUNCE, is_source_changed, lock_changes,
        require_durable_publication,
    };
    use super::*;
    use crate::generation_store::{CompleteGenerationStore, PublishedGeneration};
    use crate::{CliCommand, CliInput, FsfsConfig, OutputFormat};

    fn fixture(parent: &Path) -> (FsfsRuntime, PathBuf, PathBuf) {
        let source = parent.join("source");
        let root = parent.join("store");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("alpha.md"), "privatebody original text").unwrap();
        fs::write(source.join("stable.md"), "stableword unchanged text").unwrap();
        let mut config = FsfsConfig::default();
        config.discovery.max_file_size_mb = 1;
        config.indexing.offline = true;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Watch,
            target_path: Some(source.clone()),
            index_dir: Some(root.clone()),
            quiet: true,
            format: OutputFormat::Jsonl,
            ..CliInput::default()
        });
        runtime.lexical_only_indexing = true;
        (runtime, source, root)
    }

    #[allow(clippy::used_underscore_binding)]
    fn session(runtime: &FsfsRuntime, cx: &Cx, root: &Path) -> CompleteWatchSession {
        let mut session = CompleteWatchSession::open(runtime, cx, root).unwrap();
        session._watcher.unwatch(&session.source.path).unwrap();
        session.changes = Arc::new(Mutex::new(Changes::default()));
        session
    }

    async fn change(session: &mut CompleteWatchSession, cx: &Cx, path: &Path) -> PublishedGeneration {
        let now = Instant::now();
        {
            let mut changes = lock_changes(&session.changes).unwrap();
            changes.record(now, false);
            changes.record_path(path);
        }
        require_durable_publication(session.advance(cx, now + DEBOUNCE).await.unwrap().unwrap())
            .unwrap()
    }

    fn grow_metadata_source(path: &Path) {
        OpenOptions::new().write(true).open(path).unwrap()
            .set_len(MAX_SOURCE_BYTES + 1).unwrap();
    }

    #[test]
    fn stat_only_preparation_accepts_large_non_utf8_and_pdf_metadata_without_content_reads() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, path, _) = fixture(directory.path());
            let source = SourceRoot::open(path.clone()).unwrap();
            let before = source.observe(&cx, &runtime.config.discovery).unwrap();
            for name in ["binary.md", "document.pdf"] {
                let file = path.join(name);
                fs::write(&file, [0xff, 0x00, 0xfe]).unwrap();
                grow_metadata_source(&file);
            }
            let after = source.observe(&cx, &runtime.config.discovery).unwrap();
            for name in ["binary.md", "document.pdf"] {
                assert_eq!(after.classes[&path.join(name)], IngestionClass::MetadataOnly);
            }
            assert!(prepare(&cx, &runtime, &source, &before, &after, None)
                .await.unwrap().is_some());
            // The same large file is still outside the content-reading lane's
            // limits. This change must not weaken lexical/semantic admission.
            let mut content = after;
            content.classes.insert(path.join("binary.md"), IngestionClass::LexicalOnly);
            assert!(prepare(&cx, &runtime, &source, &before, &content, None)
                .await.unwrap().is_none());
        });
    }

    #[test]
    fn metadata_stat_rejects_replacement_links_special_files_and_cancellation() {
        use std::os::unix::fs::symlink;
        use std::os::unix::net::UnixListener;

        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("source.md");
            fs::write(&path, b"\xff\0not decoded").unwrap();
            let expected = SourceStamp::from_metadata(&fs::symlink_metadata(&path).unwrap());
            metadata_modified_ms(&cx, &path, &expected).unwrap();
            let retained = directory.path().join("retained.md");
            fs::rename(&path, &retained).unwrap();
            fs::write(&path, b"\xff\0not decoded").unwrap();
            assert!(is_source_changed(&metadata_modified_ms(&cx, &path, &expected).unwrap_err()));
            let link = directory.path().join("alias.md");
            symlink(&retained, &link).unwrap();
            let link_stamp = SourceStamp::from_metadata(&fs::symlink_metadata(&link).unwrap());
            assert!(is_source_changed(&metadata_modified_ms(&cx, &link, &link_stamp).unwrap_err()));
            let socket = directory.path().join("socket");
            let _listener = UnixListener::bind(&socket).unwrap();
            let socket_stamp = SourceStamp::from_metadata(&fs::symlink_metadata(&socket).unwrap());
            assert!(is_source_changed(&metadata_modified_ms(&cx, &socket, &socket_stamp).unwrap_err()));
            cx.set_cancel_requested(true);
            let cancelled = metadata_modified_ms(&cx, &path, &expected);
            cx.set_cancel_requested(false);
            assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
        });
    }

    #[test]
    fn metadata_stat_evidence_is_checked_after_the_owners_precommit_callback() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source_path, root) = fixture(directory.path());
            let mut session = session(&runtime, &cx, &root);
            let old = require_durable_publication(
                session.advance(&cx, Instant::now()).await.unwrap().unwrap(),
            ).unwrap();
            let source = SourceRoot::open(source_path.clone()).unwrap();
            let before = source.observe(&cx, &runtime.config.discovery).unwrap();
            let file = source_path.join("alpha.md");
            grow_metadata_source(&file);
            let after = source.observe(&cx, &runtime.config.discovery).unwrap();
            assert_eq!(after.classes[&file], IngestionClass::MetadataOnly);
            let batch = prepare(&cx, &runtime, &source, &before, &after, None)
                .await.unwrap().unwrap();
            let retained = directory.path().join("preserved-large.md");
            let result = runtime.apply_retained_source_batch_with_precommit(
                &cx, &root, &old, batch, move |_| {
                    fs::rename(&file, retained)?;
                    fs::write(&file, "recreated during precommit")?;
                    Ok(())
                },
            ).await;
            assert!(is_source_changed(&result.err().expect("source replacement must refuse publication")));
            assert_eq!(CompleteGenerationStore::open(&cx, &root).unwrap().active(&cx).unwrap(), Some(old));
        });
    }

    #[test]
    fn complete_watcher_batches_metadata_demotion_rename_and_promotion_with_pinned_readers() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let mut session = session(&runtime, &cx, &root);
            let old = require_durable_publication(
                session.advance(&cx, Instant::now()).await.unwrap().unwrap(),
            ).unwrap();
            let mut query_runtime = runtime.clone();
            query_runtime.config.pressure.degradation_override = DegradationOverrideMode::ForceLexicalOnly;
            let mut pinned = query_runtime.open_retained_search(&cx, &root).await.unwrap();
            let before = FsfsRuntime::read_matching_manifest_generation(old.path()).unwrap().unwrap();
            grow_metadata_source(&source.join("alpha.md"));
            let second = change(&mut session, &cx, &source.join("alpha.md")).await;
            assert_eq!(FsfsRuntime::read_index_sentinel(second.path()).unwrap().unwrap().command, "retained-batch");
            let manifests = FsfsRuntime::read_matching_manifest_generation(second.path()).unwrap().unwrap();
            assert_eq!(manifests["alpha.md"].ingestion_class, "metadata_only");
            assert_eq!(manifests["alpha.md"].reason_code, "index.plan.metadata_only");
            assert!(manifests["alpha.md"].fast_windows.is_none());
            assert_eq!(serde_json::to_value(&manifests["stable.md"]).unwrap(),
                serde_json::to_value(&before["stable.md"]).unwrap());
            let mut metadata_reader = query_runtime.open_retained_search(&cx, &root).await.unwrap();
            assert!(metadata_reader.search(&cx, "privatebody", 10).await.unwrap().last().unwrap().hits.is_empty());
            assert_eq!(metadata_reader.search(&cx, "alpha", 10).await.unwrap().last().unwrap().hits.len(), 1);
            assert_eq!(pinned.search(&cx, "privatebody", 10).await.unwrap().last().unwrap().hits.len(), 1);

            fs::rename(source.join("alpha.md"), source.join("renamed.md")).unwrap();
            let third = change(&mut session, &cx, &source.join("alpha.md")).await;
            assert_eq!(FsfsRuntime::read_index_sentinel(third.path()).unwrap().unwrap().command, "retained-batch");
            let renamed = FsfsRuntime::read_matching_manifest_generation(third.path()).unwrap().unwrap();
            assert!(!renamed.contains_key("alpha.md"));
            assert_eq!(renamed["renamed.md"].ingestion_class, "metadata_only");
            let mut current = query_runtime.open_retained_search(&cx, &root).await.unwrap();
            assert_eq!(current.search(&cx, "renamed", 10).await.unwrap().last().unwrap().hits.len(), 1);
            assert!(current.search(&cx, "alpha", 10).await.unwrap().last().unwrap().hits.is_empty());

            fs::write(source.join("renamed.md"), "promotedword now index the body").unwrap();
            let fourth = change(&mut session, &cx, &source.join("renamed.md")).await;
            assert_eq!(FsfsRuntime::read_index_sentinel(fourth.path()).unwrap().unwrap().command, "retained-batch");
            let mut promoted = query_runtime.open_retained_search(&cx, &root).await.unwrap();
            assert_eq!(promoted.search(&cx, "promotedword", 10).await.unwrap().last().unwrap().hits.len(), 1);
            assert!(current.search(&cx, "promotedword", 10).await.unwrap().last().unwrap().hits.is_empty());
            assert_eq!(promoted.search(&cx, "stableword", 10).await.unwrap().last().unwrap().hits.len(), 1);
            assert_eq!(pinned.generation(), &old);
            assert_eq!(metadata_reader.generation(), &second);
            assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 4);
        });
    }

    fn request(path: &Path) -> (PathBuf, SourceStamp) {
        (
            path.to_path_buf(),
            SourceStamp::from_metadata(&fs::symlink_metadata(path).unwrap()),
        )
    }

    #[test]
    fn text_batch_preserves_full_bodies_timestamps_and_exact_path_identity() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let first = directory.path().join("alpha.md");
            let second = directory.path().join("βeta.md");
            fs::write(&first, "first body with repeated words\nfirst body").unwrap();
            fs::write(&second, "second body with unicode λ\n").unwrap();
            // Reverse order prevents an accidental positional/source-ID join
            // from looking correct just because the map sorts the same way.
            let requests = vec![request(&second), request(&first)];
            let results = read_text_batch(&cx, requests).await.unwrap().unwrap();
            assert_eq!(results.len(), 2);
            for path in [&first, &second] {
                let result = &results[path];
                assert_eq!(result.text, fs::read_to_string(path).unwrap());
                assert_eq!(
                    result.modified_ms,
                    system_time_to_ms(fs::metadata(path).unwrap().modified().unwrap())
                );
                assert_eq!(result.classification.detected_type, DetectedType::Text);
                assert_eq!(result.classification.ingest_action, IngestAction::Index);
            }
        });
    }

    #[test]
    fn text_batch_limits_refuse_before_io_and_accept_the_exact_boundaries() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let seed = directory.path().join("seed.md");
            fs::write(&seed, "seed text").unwrap();
            let (_, mut stamp) = request(&seed);
            stamp.bytes = MAX_SOURCE_BYTES;
            // The paths do not exist: a late bounds check would report an I/O
            // error instead of the unsupported-plan result asserted below.
            let first = (directory.path().join("missing-one.md"), stamp.clone());
            let second = (directory.path().join("missing-two.md"), stamp.clone());
            assert!(text_batch_supported(&cx, &[first.clone(), second.clone()]).unwrap());
            let mut one_byte = stamp.clone();
            one_byte.bytes = 1;
            let over_total = vec![first.clone(), second, (seed.clone(), one_byte.clone())];
            assert!(read_text_batch(&cx, over_total).await.unwrap().is_none());
            stamp.bytes = MAX_SOURCE_BYTES + 1;
            assert!(read_text_batch(&cx, vec![(first.0.clone(), stamp)]).await.unwrap().is_none());
            assert!(read_text_batch(&cx, vec![first.clone(), first]).await.unwrap().is_none());
            let mut maximum = (0..MAX_CHANGED_PATHS)
                .map(|number| (directory.path().join(format!("missing-{number}.md")), one_byte.clone()))
                .collect::<Vec<_>>();
            assert!(text_batch_supported(&cx, &maximum).unwrap());
            maximum.push((seed, one_byte));
            assert!(read_text_batch(&cx, maximum).await.unwrap().is_none());
            assert!(read_text_batch(&cx, Vec::new()).await.unwrap().unwrap().is_empty());
        });
    }

    #[test]
    fn text_batch_never_returns_partial_success_after_a_later_read_fails() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let first = directory.path().join("first.md");
            let second = directory.path().join("second.md");
            fs::write(&first, "first valid text").unwrap();
            fs::write(&second, "second valid text").unwrap();
            let stale = request(&second);
            fs::rename(&second, directory.path().join("preserved-second.md")).unwrap();
            fs::write(&second, "second valid text").unwrap();
            let result = read_text_batch(&cx, vec![request(&first), stale]).await;
            assert!(is_source_changed(&result.err().expect("replacement must refuse the whole batch")));

            // Unsupported content returns no partial text map and stops before
            // the missing third file. The normal indexer owns this case.
            for bytes in [b"\0binary\0payload".as_slice(), b"\xff\xfe\xff".as_slice()] {
                fs::write(&second, bytes).unwrap();
                let mut missing = request(&first);
                missing.0 = directory.path().join("not-read-after-refusal.md");
                assert!(read_text_batch(&cx, vec![request(&first), request(&second), missing])
                    .await.unwrap().is_none());
            }
        });
    }

    #[test]
    fn cancelled_text_batches_do_not_become_empty_or_unsupported_success() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("body.md");
            fs::write(&path, "valid source text").unwrap();
            for requests in [Vec::new(), vec![request(&path)]] {
                cx.set_cancel_requested(true);
                let result = read_text_batch(&cx, requests).await;
                cx.set_cancel_requested(false);
                assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            }
        });
    }

    #[test]
    fn complete_watcher_publishes_mixed_batched_text_and_stat_only_sources() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let mut watcher = session(&runtime, &cx, &root);
            let old = require_durable_publication(
                watcher.advance(&cx, Instant::now()).await.unwrap().unwrap(),
            ).unwrap();
            let mut query_runtime = runtime.clone();
            query_runtime.config.pressure.degradation_override = DegradationOverrideMode::ForceLexicalOnly;
            let mut pinned = query_runtime.open_retained_search(&cx, &root).await.unwrap();
            fs::write(source.join("alpha.md"), "batchedreplacement alpha text").unwrap();
            fs::write(source.join("second.md"), "batchedreplacement second text").unwrap();
            fs::write(source.join("inventory.pdf"), [0xff, 0x00, 0xfe]).unwrap();
            grow_metadata_source(&source.join("inventory.pdf"));
            let next = change(&mut watcher, &cx, &source.join("alpha.md")).await;
            assert_eq!(FsfsRuntime::read_index_sentinel(next.path()).unwrap().unwrap().command, "retained-batch");
            let before = FsfsRuntime::read_matching_manifest_generation(old.path()).unwrap().unwrap();
            let after = FsfsRuntime::read_matching_manifest_generation(next.path()).unwrap().unwrap();
            assert_eq!(serde_json::to_value(&before["stable.md"]).unwrap(),
                serde_json::to_value(&after["stable.md"]).unwrap());
            assert_eq!(after["inventory.pdf"].ingestion_class, "metadata_only");
            let mut reader = query_runtime.open_retained_search(&cx, &root).await.unwrap();
            assert_eq!(reader.search(&cx, "batchedreplacement", 10).await.unwrap().last().unwrap().hits.len(), 2);
            assert_eq!(reader.search(&cx, "inventory", 10).await.unwrap().last().unwrap().hits.len(), 1);
            assert!(reader.search(&cx, "privatebody", 10).await.unwrap().last().unwrap().hits.is_empty());
            assert_eq!(pinned.search(&cx, "privatebody", 10).await.unwrap().last().unwrap().hits.len(), 1);
            assert_eq!(pinned.generation(), &old);
        });
    }

    #[test]
    fn failed_batched_preparation_leaves_selection_intact_and_retry_keeps_every_change() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source_path, root) = fixture(directory.path());
            let mut watcher = session(&runtime, &cx, &root);
            let old = require_durable_publication(
                watcher.advance(&cx, Instant::now()).await.unwrap().unwrap(),
            ).unwrap();
            let source = SourceRoot::open(source_path.clone()).unwrap();
            let before = source.observe(&cx, &runtime.config.discovery).unwrap();
            fs::write(source_path.join("alpha.md"), "retrycohort alpha changed").unwrap();
            let second = source_path.join("second.md");
            fs::write(&second, "retrycohort second changed").unwrap();
            let stale = source.observe(&cx, &runtime.config.discovery).unwrap();
            fs::rename(&second, directory.path().join("preserved-second.md")).unwrap();
            fs::write(&second, "retrycohort second changed").unwrap();
            let result = prepare(&cx, &runtime, &source, &before, &stale, None).await;
            assert!(is_source_changed(&result.err().expect("later changed source must refuse preparation")));
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            assert_eq!(store.active(&cx).unwrap().as_ref(), Some(&old));
            assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 1);
            let current = source.observe(&cx, &runtime.config.discovery).unwrap();
            let batch = prepare(&cx, &runtime, &source, &before, &current, None).await.unwrap().unwrap();
            let outcome = runtime.apply_retained_source_batch_with_precommit(
                &cx, &root, &old, batch, |_| Ok(()),
            ).await.unwrap();
            assert_eq!((outcome.upserted, outcome.deleted), (2, 0));
            let next = require_durable_publication(outcome.publication.unwrap()).unwrap();
            assert_eq!(store.active(&cx).unwrap().as_ref(), Some(&next));
            let mut query_runtime = runtime.clone();
            query_runtime.config.pressure.degradation_override = DegradationOverrideMode::ForceLexicalOnly;
            let mut reader = query_runtime.open_retained_search(&cx, &root).await.unwrap();
            assert_eq!(reader.search(&cx, "retrycohort", 10).await.unwrap().last().unwrap().hits.len(), 2);
        });
    }

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    #[test]
    fn one_owned_submission_reads_all_text_and_noops_do_not_queue_work() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let mut requests = Vec::new();
            for name in ["first.md", "second.md", "third.md"] {
                let path = directory.path().join(name);
                fs::write(&path, format!("complete source for {name}")).unwrap();
                requests.push(request(&path));
            }
            let submitted = AtomicUsize::new(0);
            let results = read_text_batch_with(&cx, requests, |job| {
                submitted.fetch_add(1, Ordering::SeqCst);
                std::future::ready(job())
            }).await.unwrap().unwrap();
            assert_eq!(submitted.load(Ordering::SeqCst), 1);
            assert_eq!(results.len(), 3);
            assert!(read_text_batch_with(&cx, Vec::new(), |_| {
                submitted.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Err(source_changed()))
            }).await.unwrap().unwrap().is_empty());
            assert_eq!(submitted.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn unpolled_source_read_future_never_submits_work_or_cancels_its_caller() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("source.md");
            fs::write(&path, "original source text").unwrap();
            let submitted = AtomicUsize::new(0);
            let future = read_text_batch_with(&cx, vec![request(&path)], |job| {
                submitted.fetch_add(1, Ordering::SeqCst);
                std::future::ready(job())
            });
            drop(future);
            assert_eq!(submitted.load(Ordering::SeqCst), 0);
            assert!(!cx.is_cancel_requested());
            assert_eq!(read_text_batch(&cx, vec![request(&path)]).await.unwrap().unwrap().len(), 1);
        });
    }

    #[test]
    fn dropping_a_pending_read_future_stops_its_queued_job_before_file_io() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("source.md");
            fs::write(&path, "original source text").unwrap();
            let queued = Arc::new(Mutex::new(None::<ReadJob>));
            let captured = Arc::clone(&queued);
            let mut future = Box::pin(read_text_batch_with(&cx, vec![request(&path)], move |job| {
                assert!(captured.lock().unwrap().replace(job).is_none());
                std::future::pending()
            }));
            let waker = Waker::from(Arc::new(NoopWake));
            assert!(matches!(future.as_mut().poll(&mut Context::from_waker(&waker)), Poll::Pending));
            drop(future);
            // If dropping the future did not stop the queued job, this would
            // now be an I/O/source-changed error, not the exact local cancel.
            fs::rename(&path, directory.path().join("preserved-source.md")).unwrap();
            let job = queued.lock().unwrap().take().unwrap();
            assert!(matches!(job(), Err(SearchError::Cancelled { ref phase, .. })
                if phase == "complete_watch.read_batch"));
            assert!(!cx.is_cancel_requested());
            fs::write(&path, "fresh retry text").unwrap();
            let results = read_text_batch(&cx, vec![request(&path)]).await.unwrap().unwrap();
            assert_eq!(results[&path].text, "fresh retry text");
        });
    }

    #[test]
    fn abandonment_after_a_scalar_read_stops_the_batch_before_the_next_source() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let first = directory.path().join("first.md");
            let second = directory.path().join("second.md");
            fs::write(&first, "first source text").unwrap();
            fs::write(&second, "second source text").unwrap();
            let requests = vec![request(&first), request(&second)];
            for fail_read in [false, true] {
                let abandoned = Arc::new(AtomicBool::new(false));
                let mut guard = Some(ReadGuard(Arc::clone(&abandoned)));
                let mut reads = 0;
                let result = read_text_sources(&cx, &requests, &abandoned, |cx, path, stamp| {
                    reads += 1;
                    let result = read_source(cx, path, stamp);
                    // Deterministic boundary: the scalar read has completed,
                    // but its result has not yet entered the shared batch.
                    drop(guard.take());
                    if fail_read { Err(source_changed()) } else { result }
                });
                assert!(matches!(result, Err(SearchError::Cancelled { ref phase, .. })
                    if phase == "complete_watch.read_batch"));
                assert_eq!(reads, 1, "abandoned work must not open the second source");
                assert!(!cx.is_cancel_requested());
            }
        });
    }

    #[test]
    fn caller_cancellation_after_job_completion_discards_successful_read_results() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("source.md");
            fs::write(&path, "original source text").unwrap();
            let result = read_text_batch_with(&cx, vec![request(&path)], |job| {
                let result = job();
                assert!(result.as_ref().is_ok_and(|batch| batch.as_ref().is_some_and(|batch| batch.len() == 1)));
                cx.set_cancel_requested(true);
                std::future::ready(result)
            }).await;
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        });
    }
}
