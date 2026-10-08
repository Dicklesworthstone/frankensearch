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

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};

use super::{
    DirtyWindow, SourceObservation, SourceRoot, SourceStamp, observation_io, source_changed,
};
use crate::config::IngestionClass;
use crate::file_classification::{DetectedType, IngestAction};
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

pub(super) async fn prepare(
    cx: &Cx,
    runtime: &FsfsRuntime,
    source: &SourceRoot,
    baseline: &SourceObservation,
    current: &SourceObservation,
    hint: Option<&DirtyWindow>,
) -> SearchResult<Option<SourceBatch>> {
    retained_search_checkpoint(cx)?;
    // Link aliases and external subtrees require the full indexer's discovery
    // treatment. They must not create ambiguous deletion/update identities.
    if runtime.config.discovery.follow_symlinks {
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
    for path in &paths {
        retained_search_checkpoint(cx)?;
        if source_id(&source.path, path).is_none() {
            return Ok(None);
        }
        if let Some(stamp) = current.stamps.get(path) {
            if stamp.bytes > MAX_SOURCE_BYTES || is_pdf_file(path) {
                return Ok(None);
            }
            raw_bytes = raw_bytes.saturating_add(stamp.bytes);
            if raw_bytes > MAX_RAW_BYTES {
                return Ok(None);
            }
            if !matches!(
                current.classes.get(path),
                Some(IngestionClass::FullSemanticLexical | IngestionClass::LexicalOnly)
            ) {
                return Ok(None);
            }
        }
    }
    source.check()?;
    let mut operations = Vec::with_capacity(paths.len());
    let mut attributes = BTreeMap::new();
    for path in paths {
        retained_search_checkpoint(cx)?;
        let Some(id) = source_id(&source.path, &path) else {
            return Ok(None);
        };
        let Some(stamp) = current.stamps.get(&path) else {
            // A deletion comes from two complete observations, never from the
            // kind or ordering of a native notification.
            operations.push(RetainedMutation::Delete { id });
            continue;
        };
        let task_cx = cx.clone();
        let task_path = path.clone();
        let stamp = stamp.clone();
        let read =
            asupersync::runtime::spawn_blocking(move || read_source(&task_cx, &task_path, &stamp))
                .await;
        retained_search_checkpoint(cx)?;
        let (bytes, modified_ms) = read?;
        let classification = classify_file_for_ingest(&path, &bytes);
        if classification.detected_type != DetectedType::Text
            || classification.ingest_action != IngestAction::Index
        {
            return Ok(None);
        }
        let Ok(text) = String::from_utf8(bytes) else {
            return Ok(None);
        };
        let Some(mut class) = current.classes.get(&path).copied() else {
            return Ok(None);
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
            Some(&classification),
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
    source.check()?;
    prepare_source_batch(cx, &operations, &attributes)
}

#[cfg(test)]
#[path = "delta_tests.rs"]
mod tests;
