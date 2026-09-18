//! Selected restart of the complete native hybrid build.
//!
//! This extends the existing caller-selected source/vector seal. It does not
//! implement CURRENT activation, writer fencing, antirollback or hostile-path
//! admission. Directories must be trusted and immutable during seal/open, and
//! mapped lexical files must remain immutable while readers live. Checking the
//! selected inventory before and after opening detects ordinary publication
//! drift; it is not a defense against an adversarial filesystem ABA race.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use frankensearch_core::generation::GenerationComponentReceiptV1;
use frankensearch_quill::{QuillConfig, QuillSearchIndex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::NativeBuiltHybridIndex;
use super::super::{NativeBuiltIndex, NativeReopenLimits};
use super::super::snapshot::{
    Artifact, checked_directory, create_private_new, ensure_absent, open_regular,
    read_selected, require_seal_platform, sync_directory, verify_selected,
};
use crate::native_ann::{checkpoint, invalid};
use crate::{Cx, Embedder, SearchError, SearchResult};

const HYBRID_FILE: &str = "native.hybrid.json";
const HYBRID_SCHEMA: &str = "frankensearch.native-hybrid-snapshot.v1";
const MAX_DESCRIPTOR_BYTES: u64 = 32 * 1024 * 1024;
const MAX_FILES: usize = 131_073;

/// Input ceilings for selected hybrid reopening, not a peak-memory guarantee.
///
/// Quill's own decoder limits still apply. Inventory hashing uses a bounded
/// buffer; encoded descriptor bytes have an unconditional 32 MiB ceiling.
#[derive(Debug, Clone, Copy)]
pub struct NativeHybridReopenLimits {
    /// Existing source and vector/graph admission limits.
    pub vectors: NativeReopenLimits,
    /// Maximum number of lexical files, excluding the mutable writer LOCK.
    pub max_lexical_files: usize,
    /// Maximum bytes in any one lexical artifact.
    pub max_lexical_file_bytes: u64,
    /// Maximum total bytes in all selected lexical files.
    pub max_lexical_bytes: u64,
}

impl Default for NativeHybridReopenLimits {
    fn default() -> Self {
        Self {
            vectors: NativeReopenLimits::default(),
            max_lexical_files: MAX_FILES,
            max_lexical_file_bytes: 16 * 1024 * 1024 * 1024,
            max_lexical_bytes: 64 * 1024 * 1024 * 1024,
        }
    }
}

impl NativeHybridReopenLimits {
    fn validate(self) -> SearchResult<()> {
        self.vectors.validate()?;
        if self.max_lexical_files == 0 || self.max_lexical_files > MAX_FILES
            || self.max_lexical_file_bytes == 0 || self.max_lexical_bytes == 0
        {
            return Err(rejected("limits", "invalid lexical input limits"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LexicalFile {
    name: String,
    bytes: Artifact,
}

/// Captured while the build writer still owns the finalized publication.
/// Never recaptured from current paths when sealing an older retained reader.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LexicalSeal {
    generation: u64,
    files: Vec<LexicalFile>,
}

impl LexicalSeal {
    pub(super) const fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn capture(cx: &Cx, directory: &Path, generation: u64) -> SearchResult<Self> {
        let limits = NativeHybridReopenLimits::default();
        let names = lexical_names(cx, directory, limits.max_lexical_files)?;
        let mut files = Vec::new();
        let mut total = 0_u64;
        for name in names {
            let remaining = limits.max_lexical_bytes.checked_sub(total)
                .ok_or_else(|| rejected("size", "lexical byte count exceeds the build limit"))?;
            let bytes = fingerprint_file(cx, &directory.join(&name), remaining.min(limits.max_lexical_file_bytes))?;
            total = total.checked_add(bytes.byte_len)
                .ok_or_else(|| rejected("size", "lexical byte count overflowed"))?;
            files.push(LexicalFile { name, bytes });
        }
        let seal = Self { generation, files };
        seal.validate(limits)?;
        Ok(seal)
    }

    fn validate(&self, limits: NativeHybridReopenLimits) -> SearchResult<()> {
        limits.validate()?;
        if self.files.is_empty() || self.files.len() > limits.max_lexical_files {
            return Err(rejected("inventory", "lexical inventory exceeds its file-count bound"));
        }
        let mut previous: Option<&str> = None;
        let mut total = 0_u64;
        let mut manifest = false;
        for file in &self.files {
            validate_name(&file.name)?;
            if previous.is_some_and(|name| name >= file.name.as_str()) {
                return Err(rejected("inventory", "lexical names must be unique and ordered"));
            }
            previous = Some(&file.name);
            manifest |= file.name == "MANIFEST";
            file.bytes.validate()?;
            total = total.checked_add(file.bytes.byte_len)
                .ok_or_else(|| rejected("size", "lexical byte count overflowed"))?;
            if file.bytes.byte_len > limits.max_lexical_file_bytes || total > limits.max_lexical_bytes {
                return Err(rejected("size", "selected lexical input exceeds its limits"));
            }
        }
        if !manifest {
            return Err(rejected("manifest", "a finalized primary MANIFEST is required"));
        }
        Ok(())
    }

    fn verify(&self, cx: &Cx, directory: &Path, limits: NativeHybridReopenLimits, sync: bool) -> SearchResult<()> {
        self.validate(limits)?;
        let names = lexical_names(cx, directory, limits.max_lexical_files)?;
        if !names.iter().map(String::as_str).eq(self.files.iter().map(|file| file.name.as_str())) {
            return Err(rejected("inventory", "lexical artifacts differ from the selected inventory"));
        }
        for file in &self.files {
            let path = directory.join(&file.name);
            verify_selected(cx, &path, file.bytes)?;
            if sync {
                open_regular(&path)?.sync_all()?;
            }
        }
        checkpoint(cx, "native_ann.hybrid_snapshot.lexical_verified")?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HybridSnapshot {
    schema: String,
    vectors: Artifact,
    lexical: LexicalSeal,
}

impl NativeBuiltHybridIndex {
    /// Seal the complete source/vector/Quill cohort for selected restart.
    ///
    /// Call this instead of separately sealing `vectors()`. This first checks
    /// the lexical bytes captured at build, then seals the retained sources and
    /// required vector tiers, and finally writes the hybrid descriptor. Retain
    /// the returned receipt in the caller's trusted selection/catalog. Neither
    /// finding this file nor hashing an unknown directory authorizes opening it.
    ///
    /// Every lexical file is selected except Quill's mutable writer LOCK. Old
    /// manifests and tombstone/segment files cannot silently change or disappear.
    /// No repair, reindex, model substitution or lexical-only fallback occurs.
    /// Files and directories are synced before returning; cancellation is checked
    /// before the final descriptor write, not after committing it. An interrupted
    /// seal can leave unselected partial files, which are never overwritten.
    ///
    /// Supported on Linux/macOS under the trusted immutable-directory contract
    /// of the existing native source/vector seal. This is not CURRENT activation,
    /// a rollback floor, a writer fence or hardened generation-root publication.
    ///
    /// # Errors
    /// Refuses drift, existing complete/partial seals, unsupported platforms,
    /// source/vector/graph errors, serialization, I/O and cancellation.
    pub fn seal_for_reopen(&self, cx: &Cx) -> SearchResult<GenerationComponentReceiptV1> {
        checkpoint(cx, "native_ann.hybrid_snapshot.seal_start")?;
        require_seal_platform()?;
        let directory = checked_directory(&self.vectors.directory)?;
        ensure_absent(&directory.join(HYBRID_FILE))?;
        let lexical_path = checked_directory(&directory.join("lexical"))?;
        let limits = NativeHybridReopenLimits::default();
        self.lexical_seal.verify(cx, &lexical_path, limits, true)?;
        let vectors = self.vectors.seal_for_reopen(cx)?;
        self.lexical_seal.verify(cx, &lexical_path, limits, false)?;
        let saved = HybridSnapshot {
            schema: HYBRID_SCHEMA.to_owned(),
            vectors: Artifact { byte_len: vectors.byte_len, sha256: vectors.sha256 },
            lexical: self.lexical_seal.clone(),
        };
        let bytes = serde_json::to_vec(&saved)
            .map_err(|_| rejected("encoding", "could not encode hybrid descriptor"))?;
        if bytes.len() as u64 > MAX_DESCRIPTOR_BYTES {
            return Err(rejected("size", "hybrid descriptor exceeds its format bound"));
        }
        checkpoint(cx, "native_ann.hybrid_snapshot.seal_commit")?;
        let mut file = create_private_new(&directory.join(HYBRID_FILE))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        sync_directory(&lexical_path)?;
        sync_directory(&directory)?;
        if let Some(parent) = directory.parent() {
            sync_directory(parent)?;
        }
        Ok(Artifact::from_bytes(&bytes).receipt())
    }

    /// Reopen exactly the selected hybrid cohort, without embedding or writing.
    ///
    /// Uses already-loaded providers and the original source/vector seal. Quill
    /// opens read-only, without acquiring a writer lease or exposing refresh.
    /// All lexical bytes are checked both before and after Quill open, and the
    /// reader must report the captured generation and full source cardinality.
    /// Multiple independent readers may coexist. Progressive retrieval and
    /// reranking keep using restored build-time source text and pinned metadata.
    ///
    /// The directory must remain trusted and immutable during opening; pre/post
    /// checks are not hostile-filesystem/ABA admission. Keep mapped lexical
    /// artifacts immutable for the full reader lifetime. No newer generation is
    /// discovered, and a missing lexical arm is never accepted as vector-only.
    ///
    /// # Errors
    /// Returns exact-selection, lexical, source/vector, provider, limit, I/O or
    /// cancellation errors. No partially opened hybrid object is returned.
    pub async fn open_selected(
        cx: &Cx,
        directory: impl AsRef<Path> + Send,
        expected: &GenerationComponentReceiptV1,
        fast: Arc<dyn Embedder>,
        quality: Option<Arc<dyn Embedder>>,
    ) -> SearchResult<Self> {
        Self::open_selected_with_limits(cx, directory, expected, fast, quality, NativeHybridReopenLimits::default()).await
    }

    /// Selected hybrid reopening with explicit source/vector and lexical limits.
    ///
    /// Filesystem hashing and vector/graph admission are synchronous on the
    /// caller's execution lane; Quill's own open may suspend. No runtime or
    /// detached worker is created by this adapter.
    ///
    /// # Errors
    /// Has the errors of [`Self::open_selected`], with lexical limits checked
    /// before Quill is allowed to map or decode selected files.
    pub async fn open_selected_with_limits(
        cx: &Cx,
        directory: impl AsRef<Path> + Send,
        expected: &GenerationComponentReceiptV1,
        fast: Arc<dyn Embedder>,
        quality: Option<Arc<dyn Embedder>>,
        limits: NativeHybridReopenLimits,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.hybrid_snapshot.open_start")?;
        limits.validate()?;
        let directory = checked_directory(directory.as_ref())?;
        let receipt = Artifact { byte_len: expected.byte_len, sha256: expected.sha256 };
        let bytes = read_selected(cx, &directory.join(HYBRID_FILE), receipt, MAX_DESCRIPTOR_BYTES)?;
        let saved: HybridSnapshot = serde_json::from_slice(&bytes)
            .map_err(|_| rejected("schema", "malformed hybrid snapshot descriptor"))?;
        if saved.schema != HYBRID_SCHEMA {
            return Err(rejected("schema", "unsupported hybrid snapshot schema"));
        }
        saved.vectors.validate()?;
        saved.lexical.validate(limits)?;
        let lexical_path = checked_directory(&directory.join("lexical"))?;
        saved.lexical.verify(cx, &lexical_path, limits, false)?;
        let vectors = NativeBuiltIndex::open_selected_with_limits(
            cx, &directory, &saved.vectors.receipt(), fast, quality, limits.vectors,
        )?;
        let response = QuillSearchIndex::open(cx, &lexical_path, QuillConfig::default()).await;
        checkpoint(cx, "native_ann.hybrid_snapshot.lexical_opened")?;
        let lexical = response?;
        saved.lexical.verify(cx, &lexical_path, limits, false)?;
        Self::from_readers(cx, vectors, lexical, saved.lexical)
    }
}

fn validate_name(name: &str) -> SearchResult<()> {
    if name.is_empty() || name.len() > 255 || name == "." || name == ".." || name == "LOCK"
        || !name.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(rejected("name", "lexical artifacts must have safe single-component ASCII names"));
    }
    Ok(())
}

fn lexical_names(cx: &Cx, directory: &Path, limit: usize) -> SearchResult<Vec<String>> {
    let directory: PathBuf = checked_directory(directory)?;
    let mut names = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        checkpoint(cx, "native_ann.hybrid_snapshot.directory_entry")?;
        let entry = entry?;
        if entry.file_name() == "LOCK" {
            // QuillSearchIndex does not read writer-admission LOCK state.
            continue;
        }
        let name = entry.file_name().into_string()
            .map_err(|_| rejected("name", "lexical filename is not UTF-8"))?;
        validate_name(&name)?;
        if names.len() >= limit || !entry.file_type()?.is_file() {
            return Err(rejected("inventory", "lexical file count or file type is not admissible"));
        }
        names.push(name);
    }
    names.sort_unstable();
    Ok(names)
}

fn fingerprint_file(cx: &Cx, path: &Path, limit: u64) -> SearchResult<Artifact> {
    let mut file = open_regular(path)?;
    let byte_len = file.metadata()?.len();
    if byte_len == 0 || byte_len > limit {
        return Err(rejected("size", "lexical file exceeds its capture bound"));
    }
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut observed = 0_u64;
    loop {
        checkpoint(cx, "native_ann.hybrid_snapshot.capture_chunk")?;
        let read = file.read(&mut buffer)?;
        if read == 0 { break; }
        observed = observed.checked_add(read as u64)
            .ok_or_else(|| rejected("size", "lexical byte count overflowed"))?;
        if observed > byte_len {
            return Err(rejected("size", "lexical file grew during capture"));
        }
        hash.update(&buffer[..read]);
    }
    if observed != byte_len {
        return Err(rejected("size", "lexical file changed during capture"));
    }
    Ok(Artifact { byte_len, sha256: hash.finalize().into() })
}

fn rejected(field: &str, reason: &str) -> SearchError {
    invalid(&format!("hybrid_snapshot.{field}"), "rejected", reason)
}
