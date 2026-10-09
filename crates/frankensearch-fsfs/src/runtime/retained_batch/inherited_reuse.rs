//! Preserve, but never strengthen, source-reuse evidence across a mixed batch.
//!
//! A document mutation is not a new source indexing run. Carry the old receipt's
//! execution/configuration scope verbatim and invalidate every touched row. A
//! source-owned batch can replace invalidated evidence with its observed raw
//! input, but only under that same execution/configuration scope. The
//! ordinary retained indexer still decides compatibility, checks source hashes
//! and admits producers before actually reusing anything. In particular, this
//! does not hash canonical document text and call it a source-byte witness.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    Cx, FsfsRuntime, IndexManifestEntry, IngestionClass, PreparedBatch, PublishedGeneration,
    RetainedMutation, SearchResult, ingestion_class_label, invalid, retained_search_checkpoint,
};
use crate::runtime::{CheckpointFileEntry, INDEXING_CHECKPOINT_SCHEMA_VERSION, IndexingCheckpoint};

const RECEIPT_FILE: &str = "FSFS-REUSE.json";
const MAX_RECEIPT_BYTES: usize = 64 * 1024 * 1024;

/// Only source preparation constructs this witness. The public document API
/// supplies none. Fields bind the exact input and the prepared source contract;
/// vector/lexical completion is established later by the common batch writer.
pub(super) struct SourceInput {
    content_hash_hex: String,
    canonical_lines: u64,
    revision: i64,
    ingestion_class: String,
    canonical_bytes: u64,
    reason_code: String,
}

pub(super) fn record_source_inputs(
    cx: &Cx,
    operations: &[RetainedMutation],
    documents: &PreparedBatch,
) -> SearchResult<BTreeMap<String, SourceInput>> {
    retained_search_checkpoint(cx)?;
    let mut seen = HashSet::new();
    let mut inputs = BTreeMap::new();
    // Admission has validated every operation. Hash only the last body for each
    // ID, exactly as preparation chose it, including upsert/delete sequences.
    for operation in operations.iter().rev() {
        retained_search_checkpoint(cx)?;
        let (id, text) = match operation {
            RetainedMutation::Upsert { id, text } => (id, Some(text)),
            RetainedMutation::Delete { id } => (id, None),
        };
        if !seen.insert(id.as_str()) {
            continue;
        }
        let Some(text) = text else { continue };
        let body = documents.get(id).and_then(Option::as_ref)
            .ok_or_else(|| invalid("source input has no final prepared body"))?;
        let source = body.source.as_ref()
            .ok_or_else(|| invalid("opaque document input cannot certify a source"))?;
        let mut digest = Sha256::new();
        for chunk in text.as_bytes().chunks(64 * 1024) {
            retained_search_checkpoint(cx)?;
            digest.update(chunk);
        }
        let canonical_lines = crate::runtime::count_non_empty_lines(&body.embedding);
        retained_search_checkpoint(cx)?;
        inputs.insert(id.clone(), SourceInput {
            content_hash_hex: crate::runtime::sha256_digest_hex(digest.finalize()),
            canonical_lines,
            revision: body.revision(0),
            ingestion_class: ingestion_class_label(source.ingestion_class).to_owned(),
            canonical_bytes: u64::try_from(body.embedding.len())
                .map_err(|_| invalid("source canonical length overflow"))?,
            reason_code: body.reason(source.ingestion_class).to_owned(),
        });
    }
    retained_search_checkpoint(cx)?;
    Ok(inputs)
}

fn extend_source_inputs(
    cx: &Cx,
    files: &mut BTreeMap<String, CheckpointFileEntry>,
    changed: &HashSet<String>,
    inputs: &BTreeMap<String, SourceInput>,
) -> SearchResult<()> {
    for (id, input) in inputs {
        retained_search_checkpoint(cx)?;
        let entry = files.get_mut(id)
            .filter(|_| changed.contains(id))
            .ok_or_else(|| invalid("source evidence does not name a changed member"))?;
        if entry.revision != input.revision
            || entry.ingestion_class != input.ingestion_class
            || entry.canonical_bytes != input.canonical_bytes
            || entry.reason_code != input.reason_code
        {
            return Err(invalid("source evidence disagrees with the completed member"));
        }
        entry.lexical_indexed = true;
        entry.semantic_indexed = input.ingestion_class
            == ingestion_class_label(IngestionClass::FullSemanticLexical);
        entry.content_hash_hex.clone_from(&input.content_hash_hex);
        entry.canonical_lines = Some(input.canonical_lines);
    }
    retained_search_checkpoint(cx)
}

/// The version-2 wire envelope, not another compatibility policy. Never update
/// its scope to the mutator's executable or configuration. Future receipt
/// versions must be explicitly understood before their evidence is transformed.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    version: u16,
    session: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    executable_sha256: Option<String>,
    configuration_sha256: String,
    checkpoint: IndexingCheckpoint,
}

fn read_receipt(cx: &Cx, predecessor: &PublishedGeneration) -> SearchResult<Option<Receipt>> {
    retained_search_checkpoint(cx)?;
    // The copy verifies the serving artifacts against this same receipt. The
    // reuse file itself is intentionally NOT copied, so verify its actual bytes
    // separately instead of trusting a cached generation-digest observation.
    let artifacts = predecessor.sealed_artifacts()?;
    let Some((expected_len, expected_hash)) = artifacts.get(RECEIPT_FILE) else {
        return Ok(None);
    };
    if *expected_len > MAX_RECEIPT_BYTES as u64 {
        return Err(invalid("sealed reuse receipt exceeds 64 MiB"));
    }
    let path = predecessor.path().join(RECEIPT_FILE);
    if !fs::symlink_metadata(&path)?.file_type().is_file() {
        return Err(invalid("sealed reuse receipt is not a regular file"));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
    }
    let mut input = options.open(path)?;
    let metadata = input.metadata()?;
    if !metadata.is_file() || metadata.len() != *expected_len {
        return Err(invalid("reuse receipt differs from its sealed inventory"));
    }
    let mut bytes = Vec::new();
    let capacity = usize::try_from(*expected_len)
        .map_err(|_| invalid("reuse receipt length does not fit this platform"))?;
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("cannot reserve bounded reuse evidence"))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        retained_search_checkpoint(cx)?;
        let read = input.read(&mut buffer);
        retained_search_checkpoint(cx)?;
        let count = match read {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            other => other?,
        };
        if count == 0 {
            break;
        }
        if count > capacity.saturating_sub(bytes.len()) {
            return Err(invalid("reuse receipt grew beyond its sealed length"));
        }
        digest.update(&buffer[..count]);
        bytes.extend_from_slice(&buffer[..count]);
    }
    let actual: [u8; 32] = digest.finalize().into();
    if bytes.len() != capacity
        || frankensearch_storage::ContentHasher::to_hex(&actual) != *expected_hash
    {
        return Err(invalid("reuse receipt differs from its sealed inventory"));
    }
    retained_search_checkpoint(cx)?;
    let encoded: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| invalid("sealed reuse evidence is malformed"))?;
    drop(bytes);
    if encoded.get("version").and_then(serde_json::Value::as_u64) != Some(2) {
        return Ok(None);
    }
    let receipt: Receipt = serde_json::from_value(encoded)
        .map_err(|_| invalid("sealed version-2 reuse evidence is malformed"))?;
    if receipt.checkpoint.schema_version != INDEXING_CHECKPOINT_SCHEMA_VERSION
        || !receipt.checkpoint.artifacts_durable
        || receipt.checkpoint.index_root != predecessor.path().display().to_string()
        || FsfsRuntime::read_checkpoint_manifest_generation(predecessor.path(), &receipt.checkpoint)?
            .is_none()
    {
        return Err(invalid(
            "reuse evidence does not describe its sealed predecessor",
        ));
    }
    retained_search_checkpoint(cx)?;
    Ok(Some(receipt))
}

fn matches_manifest(entry: &CheckpointFileEntry, manifest: &IndexManifestEntry) -> bool {
    entry.revision == manifest.revision
        && entry.ingestion_class == manifest.ingestion_class
        && entry.canonical_bytes == manifest.canonical_bytes
        && entry.reason_code == manifest.reason_code
        && entry.fast_windows == manifest.fast_windows
}

/// Every output member gets an entry, but only an untouched exact match keeps
/// evidence. False flags and an empty hash explicitly require recomputation;
/// they must not inherit a previous body's hash, even for an identical upsert.
fn project_files(
    cx: &Cx,
    previous: &BTreeMap<String, CheckpointFileEntry>,
    manifests: &BTreeMap<String, IndexManifestEntry>,
    changed: &HashSet<String>,
) -> SearchResult<BTreeMap<String, CheckpointFileEntry>> {
    let mut files = BTreeMap::new();
    for (id, manifest) in manifests {
        retained_search_checkpoint(cx)?;
        let entry = if changed.contains(id) {
            CheckpointFileEntry {
                revision: manifest.revision,
                ingestion_class: manifest.ingestion_class.clone(),
                canonical_bytes: manifest.canonical_bytes,
                reason_code: manifest.reason_code.clone(),
                lexical_indexed: false,
                semantic_indexed: false,
                content_hash_hex: String::new(),
                fast_windows: manifest.fast_windows.clone(),
                canonical_lines: None,
            }
        } else {
            previous
                .get(id)
                .filter(|entry| matches_manifest(entry, manifest))
                .cloned()
                .ok_or_else(|| invalid("untouched membership changed outside the retained batch"))?
        };
        files.insert(id.clone(), entry);
    }
    retained_search_checkpoint(cx)?;
    Ok(files)
}

struct BoundedBytes(Vec<u8>);

impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0
            .len()
            .checked_add(bytes.len())
            .filter(|length| *length <= MAX_RECEIPT_BYTES)
            .ok_or_else(|| io::Error::other("inherited reuse evidence exceeds 64 MiB"))?;
        self.0
            .try_reserve(bytes.len())
            .map_err(|_| io::Error::other("cannot reserve inherited reuse evidence"))?;
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Called only after the independent candidate has passed ordinary search
/// admission, and before sealing. No filesystem source is opened here. Missing
/// or unsupported evidence stays cold. Errors leave the old selection intact.
pub(super) fn retain_inputs(
    cx: &Cx,
    runtime: &FsfsRuntime,
    predecessor: &PublishedGeneration,
    destination: &Path,
    changed: &HashSet<String>,
    source_inputs: &BTreeMap<String, SourceInput>,
) -> SearchResult<()> {
    let Some(mut receipt) = read_receipt(cx, predecessor)? else {
        return Ok(());
    };
    let manifests = FsfsRuntime::read_matching_manifest_generation(destination)?
        .ok_or_else(|| invalid("candidate reuse requires matching manifests"))?;
    let sentinel = FsfsRuntime::read_index_sentinel(destination)?
        .ok_or_else(|| invalid("candidate reuse requires a completion sentinel"))?;
    if !sentinel.generation_complete || sentinel.target_root != receipt.checkpoint.target_root {
        return Err(invalid(
            "candidate source scope differs from its inherited evidence",
        ));
    }
    receipt.checkpoint.files = project_files(cx, &receipt.checkpoint.files, &manifests, changed)?;
    // An older executable's receipt cannot certify this execution's newly
    // prepared source bytes. Never relabel the inherited scope to make it fit.
    if !source_inputs.is_empty() && runtime.source_reuse_scope_matches(
        cx, &receipt.session, receipt.executable_sha256.as_deref(),
        &receipt.configuration_sha256, &receipt.checkpoint.target_root,
    )? {
        extend_source_inputs(cx, &mut receipt.checkpoint.files, changed, source_inputs)?;
    }
    // An upsert can replace a source previously skipped for its content. Its
    // old skip record is no more transferable than its old embedding witness.
    receipt.checkpoint.content_skipped.retain(|id, _| !changed.contains(id));
    // Opaque replacements remain unproven. A source-owned batch may instead
    // have re-established all witnesses, including when no old member survives.
    if !receipt.checkpoint.files.values().any(|entry| {
        entry.lexical_indexed
            && entry.content_hash_hex.len() == 64
            && entry
                .content_hash_hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
    }) {
        return Ok(());
    }
    receipt.checkpoint.index_root = destination.display().to_string();
    receipt
        .checkpoint
        .source_hash_hex
        .clone_from(&sentinel.source_hash_hex);
    receipt.checkpoint.discovered_files = sentinel.discovered_files;
    receipt.checkpoint.skipped_files = sentinel.skipped_files;
    receipt.checkpoint.updated_at_ms = sentinel.generated_at_ms;
    if FsfsRuntime::read_checkpoint_manifest_generation(destination, &receipt.checkpoint)?.is_none() {
        // Reuse is optional. A checkpoint shape not admitted by the ordinary
        // resume path must stay cold, not weaken that path or fail an otherwise
        // complete document mutation. No receipt has been created at this point.
        tracing::debug!(
            "ordinary checkpoint admission refused inherited evidence; next indexing starts cold"
        );
        return Ok(());
    }
    retained_search_checkpoint(cx)?;
    let mut encoded = BoundedBytes(Vec::new());
    serde_json::to_writer(&mut encoded, &receipt)
        .map_err(|_| invalid("cannot encode bounded inherited reuse evidence"))?;
    retained_search_checkpoint(cx)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.join(RECEIPT_FILE))?;
    for chunk in encoded.0.chunks(64 * 1024) {
        retained_search_checkpoint(cx)?;
        output.write_all(chunk)?;
    }
    output.sync_all()?;
    File::open(destination)?.sync_all()?;
    retained_search_checkpoint(cx)
}

#[cfg(all(test, unix))]
mod tests;

#[cfg(all(test, unix))]
mod source_tests;
