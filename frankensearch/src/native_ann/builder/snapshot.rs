//! Explicitly selected native-vector/source snapshots, not generation authority.
//!
//! The receipt returned by `seal_for_reopen` must be retained in the caller's
//! trusted catalog. Reopen requires that receipt; neither directory existence
//! nor self-consistent JSON selects a generation. No CURRENT pointer, publication
//! lineage, rollback floor, or lexical-component authority is invented here.
//!
//! Artifact names are fixed and directories must remain trusted and immutable.
//! These path-based operations are NOT descriptor-relative admission against a
//! hostile concurrent filesystem writer. The generation-root API owns that law.
//! Reads do not acquire writer leases or start providers. Filesystem, hashing,
//! and graph decoding run synchronously on the caller's blocking/CPU lane.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, GenerationComponentReceiptV1,
};
use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes};
use frankensearch_index::native_hnsw::{
    NativeHnswGenerationReceiptV2, ValidatedNativeHnsw,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    NativeBuildPrecision, NativeBuiltIndex, NativeBuiltTier, TierPlan, validate_source_membership,
};
use super::super::{NativeAnnIndex, checkpoint, invalid};
use super::super::fallback::NativeBackend;
use crate::{Cx, Embedder, IndexableDocument, SearchError, SearchResult};

const SNAPSHOT_FILE: &str = "native.snapshot.json";
const SOURCE_FILE: &str = "native.source.jsonl";
const SOURCE_HEADER: &[u8] = b"{\"schema\":\"frankensearch.native-source.v1\"}\n";
const SNAPSHOT_SCHEMA: &str = "frankensearch.native-vector-source-snapshot.v1";
const SNAPSHOT_MAX_BYTES: u64 = 256 * 1024;

/// Explicit resource ceilings for reopening a selected native build.
///
/// Source JSON is read one bounded line at a time; no corpus-sized encoded JSON
/// buffer is materialized. Decoded sources and admitted vector/graph owners still
/// require their ordinary resident memory. These limits bound file input, not a
/// universal peak-RSS promise. The native graph decoder applies its own checks.
#[derive(Debug, Clone, Copy)]
pub struct NativeReopenLimits {
    /// Maximum total encoded source bytes, including the header.
    pub max_source_bytes: u64,
    /// Maximum encoded bytes in any single source-document line.
    pub max_document_bytes: u64,
    /// Maximum number of decoded source documents.
    pub max_documents: usize,
    /// Maximum byte length of each FSVI image.
    pub max_vector_bytes: u64,
    /// Maximum byte length of each selected native graph.
    pub max_graph_bytes: u64,
}

impl Default for NativeReopenLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 1024 * 1024 * 1024,
            max_document_bytes: 16 * 1024 * 1024,
            max_documents: 10_000_000,
            max_vector_bytes: 16 * 1024 * 1024 * 1024,
            max_graph_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

impl NativeReopenLimits {
    fn validate(self) -> SearchResult<()> {
        if self.max_source_bytes < SOURCE_HEADER.len() as u64
            || self.max_document_bytes == 0
            || self.max_document_bytes.checked_add(1).is_none()
            || usize::try_from(self.max_document_bytes).is_err()
            || self.max_vector_bytes == 0
            || self.max_graph_bytes == 0
        {
            return Err(rejected("limits", "invalid native reopen resource limits"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    byte_len: u64,
    sha256: [u8; 32],
}

impl Artifact {
    fn receipt(self) -> GenerationComponentReceiptV1 {
        GenerationComponentReceiptV1 { byte_len: self.byte_len, sha256: self.sha256 }
    }

    fn validate(self) -> SearchResult<()> {
        self.receipt().validate().map_err(|_| rejected("receipt", "invalid artifact receipt"))
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self { byte_len: bytes.len() as u64, sha256: Sha256::digest(bytes).into() }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedTier {
    producer: EmbeddingIdentityBundleV1,
    precision: NativeBuildPrecision,
    vector: Artifact,
    // The ENTIRE native receipt is retained, not just the docset/generation.
    // A different graph built over the same FSVI must not be silently selected.
    graph: Option<NativeHnswGenerationReceiptV2>,
}

impl SavedTier {
    fn capture(tier: &NativeBuiltTier) -> Self {
        let witness = tier.index.owner_witness();
        Self {
            producer: tier.producer_identity.clone(),
            precision: tier.precision,
            vector: Artifact { byte_len: witness.byte_len, sha256: witness.whole_image_sha256 },
            graph: tier.graph_receipt.clone(),
        }
    }

    fn plan(&self, embedder: Arc<dyn Embedder>) -> SearchResult<TierPlan> {
        let mut plan = TierPlan::new(embedder)?;
        plan.admit(&self.producer)?;
        plan.precision = self.precision;
        Ok(plan)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    schema: String,
    generation: ArtifactGenerationIdentityV1,
    documents: u64,
    source: Artifact,
    fast: SavedTier,
    quality: Option<SavedTier>,
}

impl NativeBuiltIndex {
    /// Persist the retained source cohort and seal a small vector/source descriptor.
    ///
    /// Returns the exact descriptor receipt to retain in the caller's trusted
    /// generation selection. Do not derive the expected receipt from a directory
    /// encountered during reopening: that would only prove self-consistency.
    /// Existing source/descriptor files are never overwritten, including partial
    /// artifacts left by a failed attempt. This operation does not publish Quill,
    /// change CURRENT, install a serving generation, or establish an antirollback
    /// floor. A hybrid build's vector descriptor is explicitly vector/source-only.
    ///
    /// Files and containing directories are synced before success. Sealing is
    /// supported on Linux/macOS; other platforms fail before creating files.
    /// Cancellation is checked before the final bounded descriptor write, not
    /// after committing it. An I/O failure can leave an unselected partial seal;
    /// directory/descriptor existence must never be interpreted as success.
    ///
    /// # Errors
    /// Rejects changed vector/graph files, changed producers, preexisting seal
    /// files, unsupported platforms, serialization, I/O, and cancellation.
    pub fn seal_for_reopen(&self, cx: &Cx) -> SearchResult<GenerationComponentReceiptV1> {
        checkpoint(cx, "native_ann.snapshot.seal_start")?;
        require_seal_platform()?;
        let directory = checked_directory(&self.directory)?;
        ensure_absent(&directory.join(SOURCE_FILE))?;
        ensure_absent(&directory.join(SNAPSHOT_FILE))?;
        let fast = SavedTier::capture(&self.fast);
        let quality = self.quality.as_ref().map(SavedTier::capture);
        verify_built_tier(cx, &self.fast, &fast)?;
        if let (Some(tier), Some(saved)) = (&self.quality, &quality) {
            verify_built_tier(cx, tier, saved)?;
        }
        let source = write_sources(cx, &directory.join(SOURCE_FILE), &self.documents)?;
        let snapshot = Snapshot {
            schema: SNAPSHOT_SCHEMA.to_owned(),
            generation: self.fast.index.owner_witness().generation,
            documents: self.documents.len() as u64,
            source, fast, quality,
        };
        let bytes = serde_json::to_vec(&snapshot)
            .map_err(|_| rejected("encoding", "could not encode native snapshot descriptor"))?;
        if bytes.len() as u64 > SNAPSHOT_MAX_BYTES {
            return Err(rejected("descriptor_size", "native descriptor exceeds the format bound"));
        }
        checkpoint(cx, "native_ann.snapshot.seal_commit")?;
        let mut file = create_private_new(&directory.join(SNAPSHOT_FILE))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        sync_directory(&directory)?;
        if let Some(parent) = directory.parent() {
            sync_directory(parent)?;
        }
        Ok(Artifact::from_bytes(&bytes).receipt())
    }

    /// Reopen a caller-selected complete source/fast/quality cohort without inference.
    ///
    /// The receipt must come from a successful seal or trusted external catalog,
    /// not from hashing the descriptor currently found in this directory. Every
    /// configured tier and its original graph are required. Missing/corrupt ANN
    /// never triggers a rebuild or implicit exact fallback; no quality tier is
    /// silently dropped. Callers supply already-loaded models with the original
    /// complete producer identities. No model is discovered or downloaded.
    ///
    /// Reopening uses no FSVI writer lease; multiple independently opened owners
    /// can coexist. Search and source lookup subsequently use owned memory, not
    /// these paths. This does not reopen a lexical index or select the newest
    /// generation, and it does not confer generation-root publication authority.
    ///
    /// # Errors
    /// Propagates selected-receipt, source, tier, graph, provider, resource-limit,
    /// I/O and cancellation failures. No partial cohort is returned.
    pub fn open_selected(
        cx: &Cx,
        directory: impl AsRef<Path>,
        expected: &GenerationComponentReceiptV1,
        fast: Arc<dyn Embedder>,
        quality: Option<Arc<dyn Embedder>>,
    ) -> SearchResult<Self> {
        Self::open_selected_with_limits(cx, directory, expected, fast, quality, NativeReopenLimits::default())
    }

    /// [`Self::open_selected`] with explicit per-artifact/source input ceilings.
    ///
    /// # Errors
    /// Has the same error surface, with limits validated before file reads.
    pub fn open_selected_with_limits(
        cx: &Cx,
        directory: impl AsRef<Path>,
        expected: &GenerationComponentReceiptV1,
        fast: Arc<dyn Embedder>,
        quality: Option<Arc<dyn Embedder>>,
        limits: NativeReopenLimits,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.snapshot.open_start")?;
        limits.validate()?;
        let expected = Artifact { byte_len: expected.byte_len, sha256: expected.sha256 };
        expected.validate()?;
        let directory = checked_directory(directory.as_ref())?;
        let bytes = read_selected(cx, &directory.join(SNAPSHOT_FILE), expected, SNAPSHOT_MAX_BYTES)?;
        let saved: Snapshot = serde_json::from_slice(&bytes)
            .map_err(|_| rejected("schema", "malformed native snapshot descriptor"))?;
        if saved.schema != SNAPSHOT_SCHEMA {
            return Err(rejected("schema", "unsupported native snapshot schema"));
        }
        saved.generation.validate()?;
        let count = usize::try_from(saved.documents)
            .map_err(|_| rejected("documents", "document count does not fit this platform"))?;
        if count > limits.max_documents || saved.quality.is_some() != quality.is_some() {
            return Err(rejected("topology", "document limit or required quality-provider topology disagrees"));
        }
        // Freeze/admit both providers BEFORE source/vector/graph allocation.
        let fast_plan = saved.fast.plan(fast)?;
        let quality_plan = match (&saved.quality, quality) {
            (Some(tier), Some(provider)) => Some(tier.plan(provider)?),
            (None, None) => None,
            _ => return Err(rejected("topology", "quality descriptor and provider must travel together")),
        };
        if let Some(quality) = &quality_plan {
            if fast_plan.identity.input.doc_id_semantics != quality.identity.input.doc_id_semantics
                || fast_plan.identity.space.kind != quality.identity.space.kind
            {
                return Err(rejected("topology", "tier source contracts or semantic/control kinds differ"));
            }
        }
        let documents = read_sources(cx, &directory.join(SOURCE_FILE), saved.source, count, limits)?;
        let fast_binding = fast_plan.binding(&saved.generation)?;
        let fast = open_tier(cx, directory.join("fast.fsvi"), fast_binding, &saved.fast, fast_plan, &documents, limits)?;
        let quality = match (&saved.quality, quality_plan) {
            (Some(tier), Some(plan)) => {
                let binding = plan.binding(&saved.generation)?;
                Some(open_tier(cx, directory.join("quality.fsvi"), binding, tier, plan, &documents, limits)?)
            }
            (None, None) => None,
            _ => return Err(rejected("topology", "quality plan disappeared during admission")),
        };
        checkpoint(cx, "native_ann.snapshot.open_complete")?;
        Ok(Self { directory, documents: documents.into(), fast, quality })
    }
}

fn verify_built_tier(cx: &Cx, tier: &NativeBuiltTier, saved: &SavedTier) -> SearchResult<()> {
    saved.plan(Arc::clone(&tier.embedder))?;
    // Stream verification: do not allocate a second vector image when sealing.
    verify_selected(cx, &tier.vector_path, saved.vector)?;
    open_regular(&tier.vector_path)?.sync_all()?;
    match (&tier.graph_path, &saved.graph) {
        (Some(path), Some(expected)) => {
            // Native admission validates the real graph AND its real receipt.
            // Comparing the complete returned receipt also rejects another graph
            // over the same FSVI owner, including changed search parameters.
            let (_, observed) = ValidatedNativeHnsw::load(Arc::clone(&tier.index.owner), path)?;
            same_graph_receipt(expected, &observed)?;
            checkpoint(cx, "native_ann.snapshot.seal_graph")?;
            open_regular(path)?.sync_all()?;
            open_regular(&graph_receipt_path(path))?.sync_all()?;
        }
        (None, None) => {}
        _ => return Err(rejected("graph", "native graph path and receipt disagree")),
    }
    Ok(())
}

fn open_tier(
    cx: &Cx,
    vector_path: PathBuf,
    binding: FsviV2IdentityBinding,
    saved: &SavedTier,
    plan: TierPlan,
    documents: &[IndexableDocument],
    limits: NativeReopenLimits,
) -> SearchResult<NativeBuiltTier> {
    // The caller constructs this path from fixed role names, never JSON paths.
    let bytes: Arc<[u8]> = read_selected(cx, &vector_path, saved.vector, limits.max_vector_bytes)?.into();
    let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding)
        .map_err(|_| rejected("vector", "selected vector image failed v2 admission"))?);
    validate_source_membership(cx, &owner, documents)?;
    let (index, graph_path) = match &saved.graph {
        None => (NativeAnnIndex::exact(cx, owner)?, None),
        Some(expected) => {
            let path = vector_path.with_extension("fshnsw");
            expected.validate().map_err(|_| rejected("graph", "invalid selected native graph receipt"))?;
            if expected.graph_byte_len > limits.max_graph_bytes
                || open_regular(&path)?.metadata()?.len() != expected.graph_byte_len
                || open_regular(&graph_receipt_path(&path))?.metadata()?.len() > SNAPSHOT_MAX_BYTES
            {
                return Err(rejected("graph_size", "selected native graph exceeds its receipt or input limit"));
            }
            checkpoint(cx, "native_ann.snapshot.graph_load")?;
            let (graph, observed) = ValidatedNativeHnsw::load(Arc::clone(&owner), &path)?;
            checkpoint(cx, "native_ann.snapshot.graph_loaded")?;
            same_graph_receipt(expected, &observed)?;
            let default_ef_search = usize::try_from(observed.params.ef_search)
                .map_err(|_| rejected("graph", "selected search width does not fit usize"))?;
            (NativeAnnIndex { owner, graph: NativeBackend::Ann(Box::new(graph)), default_ef_search }, Some(path))
        }
    };
    index.admit_identity(plan.embedder.identity()?)?;
    Ok(NativeBuiltTier {
        index, embedder: plan.embedder, producer_identity: plan.identity,
        precision: plan.precision, binding, vector_path, graph_path,
        graph_receipt: saved.graph.clone(),
    })
}

fn same_graph_receipt(expected: &NativeHnswGenerationReceiptV2, observed: &NativeHnswGenerationReceiptV2) -> SearchResult<()> {
    let expected = serde_json::to_vec(expected)
        .map_err(|_| rejected("graph", "could not encode selected graph receipt"))?;
    let observed = serde_json::to_vec(observed)
        .map_err(|_| rejected("graph", "could not encode observed graph receipt"))?;
    if expected != observed {
        return Err(rejected("graph", "loaded graph differs from the exact selected native receipt"));
    }
    Ok(())
}

fn graph_receipt_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".receipt");
    PathBuf::from(name)
}

fn write_sources(cx: &Cx, path: &Path, documents: &[IndexableDocument]) -> SearchResult<Artifact> {
    let mut output = BufWriter::new(create_private_new(path)?);
    let mut hash = Sha256::new();
    hash.update(SOURCE_HEADER);
    output.write_all(SOURCE_HEADER)?;
    let mut byte_len = SOURCE_HEADER.len() as u64;
    for document in documents {
        checkpoint(cx, "native_ann.snapshot.write_source")?;
        let encoded = serde_json::to_vec(document)
            .map_err(|_| rejected("source", "source document serialization failed"))?;
        byte_len = byte_len.checked_add(encoded.len() as u64).and_then(|n| n.checked_add(1))
            .ok_or_else(|| rejected("source_size", "source byte count overflowed"))?;
        output.write_all(&encoded)?;
        output.write_all(b"\n")?;
        hash.update(&encoded);
        hash.update(b"\n");
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(Artifact { byte_len, sha256: hash.finalize().into() })
}

fn read_sources(cx: &Cx, path: &Path, expected: Artifact, count: usize, limits: NativeReopenLimits) -> SearchResult<Vec<IndexableDocument>> {
    expected.validate()?;
    let file = open_regular(path)?;
    if expected.byte_len > limits.max_source_bytes || file.metadata()?.len() != expected.byte_len {
        return Err(rejected("source_size", "source bytes disagree with the selected receipt or limit"));
    }
    let mut input = BufReader::new(file);
    let mut header = [0_u8; SOURCE_HEADER.len()];
    input.read_exact(&mut header)?;
    if header.as_slice() != SOURCE_HEADER {
        return Err(rejected("source", "unsupported source stream header"));
    }
    let mut hash = Sha256::new();
    hash.update(header);
    let mut observed = header.len() as u64;
    let mut documents: Vec<IndexableDocument> = Vec::new();
    let mut encoded = Vec::new();
    loop {
        checkpoint(cx, "native_ann.snapshot.read_source")?;
        encoded.clear();
        (&mut input).take(limits.max_document_bytes + 1).read_until(b'\n', &mut encoded)?;
        if encoded.is_empty() { break; }
        observed = observed.checked_add(encoded.len() as u64)
            .ok_or_else(|| rejected("source_size", "source byte count overflowed"))?;
        if observed > expected.byte_len || encoded.len() as u64 > limits.max_document_bytes
            || encoded.last() != Some(&b'\n') || documents.len() >= count
        {
            return Err(rejected("source_size", "source line, byte count or document count exceeds the selected bound"));
        }
        hash.update(&encoded);
        let document: IndexableDocument = serde_json::from_slice(&encoded)
            .map_err(|_| rejected("source", "invalid encoded source document"))?;
        if document.id.is_empty() || documents.last().is_some_and(|previous| previous.id.as_str() >= document.id.as_str()) {
            return Err(rejected("source_order", "source identifiers must be strictly increasing and nonempty"));
        }
        documents.try_reserve(1)
            .map_err(|_| rejected("allocation", "cannot retain selected source documents"))?;
        documents.push(document);
    }
    let sha256: [u8; 32] = hash.finalize().into();
    if observed != expected.byte_len || sha256 != expected.sha256 || documents.len() != count {
        return Err(rejected("source_receipt", "source bytes or membership differ from the selected snapshot"));
    }
    Ok(documents)
}

fn read_selected(cx: &Cx, path: &Path, expected: Artifact, limit: u64) -> SearchResult<Vec<u8>> {
    expected.validate()?;
    if expected.byte_len > limit || usize::try_from(expected.byte_len).is_err() {
        return Err(rejected("artifact_size", "selected artifact exceeds its input or platform limit"));
    }
    let mut output = Vec::new();
    read_and_verify(cx, path, expected, |bytes| {
        output.try_reserve(bytes.len())
            .map_err(|_| rejected("allocation", "cannot retain selected artifact"))?;
        output.extend_from_slice(bytes);
        Ok(())
    })?;
    Ok(output)
}

fn verify_selected(cx: &Cx, path: &Path, expected: Artifact) -> SearchResult<()> {
    read_and_verify(cx, path, expected, |_| Ok(()))
}

fn read_and_verify<F>(cx: &Cx, path: &Path, expected: Artifact, mut consume: F) -> SearchResult<()>
where F: FnMut(&[u8]) -> SearchResult<()> {
    expected.validate()?;
    let mut input = open_regular(path)?;
    if input.metadata()?.len() != expected.byte_len {
        return Err(rejected("artifact_size", "artifact byte length differs from its selected receipt"));
    }
    let mut buffer = [0_u8; 64 * 1024];
    let mut hash = Sha256::new();
    let mut observed = 0_u64;
    loop {
        checkpoint(cx, "native_ann.snapshot.artifact_chunk")?;
        let read = input.read(&mut buffer)?;
        if read == 0 { break; }
        observed = observed.checked_add(read as u64)
            .ok_or_else(|| rejected("artifact_size", "artifact byte count overflowed"))?;
        if observed > expected.byte_len {
            return Err(rejected("artifact_size", "artifact grew beyond its selected receipt"));
        }
        hash.update(&buffer[..read]);
        consume(&buffer[..read])?;
    }
    let sha256: [u8; 32] = hash.finalize().into();
    if observed != expected.byte_len || sha256 != expected.sha256 {
        return Err(rejected("artifact_receipt", "artifact bytes differ from their selected receipt"));
    }
    Ok(())
}

fn checked_directory(path: &Path) -> SearchResult<PathBuf> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(rejected("directory", "native snapshot requires a real immutable directory"));
    }
    Ok(std::fs::canonicalize(path)?)
}

fn open_regular(path: &Path) -> SearchResult<File> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(rejected("file", "native snapshot artifacts must be regular non-symlink files"));
    }
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(rejected("file", "opened native snapshot artifact is not a regular file"));
    }
    Ok(file)
}

fn ensure_absent(path: &Path) -> SearchResult<()> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
        Ok(_) => Err(rejected("existing_seal", "an existing snapshot or partial source seal cannot be overwritten")),
    }
}

fn create_private_new(path: &Path) -> SearchResult<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn require_seal_platform() -> SearchResult<()> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    { Ok(()) }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { Err(rejected("platform", "native snapshot sealing requires Linux/macOS directory durability")) }
}

fn sync_directory(path: &Path) -> SearchResult<()> {
    require_seal_platform()?;
    File::open(path)?.sync_all()?;
    Ok(())
}

fn rejected(field: &str, reason: &str) -> SearchError {
    invalid(&format!("snapshot.{field}"), "rejected", reason)
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests;
