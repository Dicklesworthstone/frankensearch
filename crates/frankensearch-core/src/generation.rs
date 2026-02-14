//! Generation manifest schema and validator for Native Mode distributed search.
//!
//! A *generation* represents a complete, consistent snapshot of all search artifacts
//! (vector indices, lexical segments, embedder metadata) built from a contiguous window
//! of document commits. Replicas atomically activate a generation to serve queries,
//! ensuring no mixed-generation reads within a single request.
//!
//! The [`GenerationManifest`] captures everything needed to replicate, verify, and
//! activate a generation on any node. The [`ManifestValidator`] enforces structural
//! and semantic invariants before activation is permitted.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use crate::SearchError;

// ---------------------------------------------------------------------------
// Commit range
// ---------------------------------------------------------------------------

/// Contiguous range of commit sequence numbers that produced this generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitRange {
    /// First commit (inclusive) in the window.
    pub low: u64,
    /// Last commit (inclusive) in the window.
    pub high: u64,
}

impl CommitRange {
    /// Number of commits covered by this range.
    #[must_use]
    pub const fn len(&self) -> u64 {
        if self.high < self.low {
            return 0;
        }
        self.high - self.low + 1
    }

    /// Whether the range is empty (high < low after wrapping / invalid state).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.high < self.low
    }
}

// ---------------------------------------------------------------------------
// Embedder revision
// ---------------------------------------------------------------------------

/// Identity of the embedder used to build vector artifacts in this generation.
///
/// Activation is blocked when a node's runtime embedder revision diverges from the
/// manifest, preventing silent ranking inconsistency across replicas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderRevision {
    /// Human-readable model name (e.g. `"potion-128M"`, `"MiniLM-L6-v2"`).
    pub model_name: String,
    /// Hex-encoded hash of the model weights file, pinning exact weights.
    pub weights_hash: String,
    /// Output dimensionality of this embedder.
    pub dimension: u32,
    /// Quantization format used for stored vectors.
    pub quantization: QuantizationFormat,
}

/// Vector quantization format used in FSVI artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationFormat {
    /// IEEE 754 single-precision (32-bit).
    F32,
    /// IEEE 754 half-precision (16-bit). Default for frankensearch.
    F16,
    /// Signed 8-bit integer with per-vector scale factor.
    Int8,
    /// Signed 4-bit integer packed two per byte.
    Int4,
}

// ---------------------------------------------------------------------------
// Artifact descriptors
// ---------------------------------------------------------------------------

/// Descriptor for a single FSVI vector index shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorArtifact {
    /// Relative path within the generation directory (e.g. `"vectors/shard_0.fsvi"`).
    pub path: String,
    /// Byte size of the artifact file.
    pub size_bytes: u64,
    /// Hex-encoded checksum (SHA-256) of the file contents.
    pub checksum: String,
    /// Number of vectors stored in this shard.
    pub vector_count: u64,
    /// Vector dimensionality.
    pub dimension: u32,
    /// Which embedder tier produced these vectors.
    pub embedder_tier: EmbedderTierTag,
}

/// Tag identifying which tier of the two-tier system produced an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedderTierTag {
    /// Fast tier (e.g. potion-128M, ~0.57ms).
    Fast,
    /// Quality tier (e.g. MiniLM-L6-v2, ~128ms).
    Quality,
}

/// Descriptor for a Tantivy lexical index segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalArtifact {
    /// Relative path within the generation directory (e.g. `"lexical/segment_0"`).
    pub path: String,
    /// Byte size of all files in the segment directory.
    pub size_bytes: u64,
    /// Hex-encoded checksum (SHA-256) of the concatenated segment files.
    pub checksum: String,
    /// Number of documents indexed in this segment.
    pub document_count: u64,
}

/// Metadata for `RaptorQ` repair symbols protecting an artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairDescriptor {
    /// Path of the protected artifact (matches a `VectorArtifact.path` or `LexicalArtifact.path`).
    pub protected_artifact: String,
    /// Path to the `.fec` sidecar file containing repair symbols.
    pub sidecar_path: String,
    /// Number of source symbols the artifact was split into.
    pub source_symbols: u32,
    /// Number of repair symbols generated.
    pub repair_symbols: u32,
    /// Overhead ratio (`repair_symbols` / `source_symbols`).
    pub overhead_ratio: f64,
}

// ---------------------------------------------------------------------------
// Activation invariants
// ---------------------------------------------------------------------------

/// A predicate that must hold before a generation can be activated for serving.
///
/// Activation invariants enforce all-or-nothing readiness: every invariant must
/// pass, or the generation is rejected and the previous generation continues serving.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationInvariant {
    /// Machine-readable invariant identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// The kind of check this invariant represents.
    pub kind: InvariantKind,
}

/// Classification of activation invariant checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvariantKind {
    /// All listed artifacts must be present and pass checksum verification.
    AllArtifactsVerified,
    /// Embedder revision must match the node's runtime embedder.
    EmbedderRevisionMatch,
    /// Total vector count must match document count (no missing embeddings).
    VectorCountConsistency {
        /// Expected total vectors across all shards.
        expected_total: u64,
    },
    /// Generation must cover a commit range that is contiguous with the previous
    /// activated generation (no gaps in commit history).
    CommitContinuity {
        /// The `high` value of the previous generation's commit range.
        previous_high: u64,
    },
    /// Custom predicate supplied by the deployment.
    Custom {
        /// Name of the custom check.
        check_name: String,
    },
}

// ---------------------------------------------------------------------------
// Generation manifest
// ---------------------------------------------------------------------------

/// Complete manifest for a search generation.
///
/// This is the unit of replication and activation in Native Mode distributed search.
/// A node fetches the manifest, verifies all artifacts, checks invariants, and
/// atomically swaps the active generation pointer on success.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationManifest {
    /// Schema version for forward-compatible parsing.
    pub schema_version: u32,
    /// Unique identifier for this generation (content-derived or monotonic).
    pub generation_id: String,
    /// Hex-encoded hash (SHA-256) of the canonical serialized manifest body,
    /// computed with this field set to the empty string.
    pub manifest_hash: String,
    /// Commit range that produced this generation.
    pub commit_range: CommitRange,
    /// Timestamp (Unix millis) when generation build started.
    pub build_started_at: u64,
    /// Timestamp (Unix millis) when generation build completed.
    pub build_completed_at: u64,
    /// Embedder revisions used (keyed by tier tag stringified).
    pub embedders: BTreeMap<String, EmbedderRevision>,
    /// Vector index artifacts in this generation.
    pub vector_artifacts: Vec<VectorArtifact>,
    /// Lexical index artifacts in this generation.
    pub lexical_artifacts: Vec<LexicalArtifact>,
    /// Repair symbol descriptors for durability.
    pub repair_descriptors: Vec<RepairDescriptor>,
    /// Activation invariants that must all pass before serving.
    pub activation_invariants: Vec<ActivationInvariant>,
    /// Total document count across all artifacts.
    pub total_documents: u64,
    /// Optional free-form metadata (deployment tags, build host, etc.).
    pub metadata: BTreeMap<String, String>,
}

/// Current schema version for [`GenerationManifest`].
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Result of validating a [`GenerationManifest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationResult {
    /// Individual validation findings (empty means valid).
    pub findings: Vec<ValidationFinding>,
}

impl ValidationResult {
    /// Whether the manifest passes all validation checks.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.findings
            .iter()
            .all(|f| f.severity != FindingSeverity::Error)
    }

    /// Collect only error-severity findings.
    #[must_use]
    pub fn errors(&self) -> Vec<&ValidationFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Error)
            .collect()
    }

    /// Collect only warning-severity findings.
    #[must_use]
    pub fn warnings(&self) -> Vec<&ValidationFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Warning)
            .collect()
    }
}

/// A single validation finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationFinding {
    /// Which check produced this finding.
    pub check: &'static str,
    /// Severity of the finding.
    pub severity: FindingSeverity,
    /// Human-readable description.
    pub message: String,
}

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingSeverity {
    /// Informational, does not block activation.
    Info,
    /// Suspicious but not blocking.
    Warning,
    /// Blocks activation.
    Error,
}

/// Validates a [`GenerationManifest`] for structural and semantic correctness.
///
/// Returns a [`ValidationResult`] with all findings. Call [`ValidationResult::is_valid`]
/// to check whether the manifest is safe to activate.
#[must_use]
pub fn validate_manifest(manifest: &GenerationManifest) -> ValidationResult {
    let mut findings = Vec::new();

    check_schema_version(manifest, &mut findings);
    check_generation_id(manifest, &mut findings);
    check_manifest_hash(manifest, &mut findings);
    check_commit_range(manifest, &mut findings);
    check_timestamps(manifest, &mut findings);
    check_embedders(manifest, &mut findings);
    check_vector_artifacts(manifest, &mut findings);
    check_lexical_artifacts(manifest, &mut findings);
    check_repair_descriptors(manifest, &mut findings);
    check_activation_invariants(manifest, &mut findings);
    check_document_count_consistency(manifest, &mut findings);

    ValidationResult { findings }
}

/// Computes the canonical manifest hash for a generation manifest.
///
/// The canonical hash is SHA-256 over JSON serialization of the manifest with
/// `manifest_hash` cleared.
///
/// # Errors
///
/// Returns `SearchError::SubsystemError` if serialization fails.
pub fn compute_manifest_hash(manifest: &GenerationManifest) -> crate::SearchResult<String> {
    let mut canonical = manifest.clone();
    canonical.manifest_hash.clear();
    let serialized =
        serde_json::to_vec(&canonical).map_err(|source| SearchError::SubsystemError {
            subsystem: "generation_manifest",
            source: Box::new(source),
        })?;
    Ok(format!("{:x}", Sha256::digest(serialized)))
}

/// Convert a validation result into a `SearchResult`, producing an error
/// if any error-severity findings exist.
///
/// # Errors
///
/// Returns `SearchError::InvalidConfig` when validation fails.
pub fn require_valid(result: &ValidationResult) -> crate::SearchResult<()> {
    if result.is_valid() {
        return Ok(());
    }
    let messages: Vec<String> = result.errors().iter().map(|f| f.message.clone()).collect();
    Err(SearchError::InvalidConfig {
        field: "generation_manifest".into(),
        value: String::new(),
        reason: messages.join("; "),
    })
}

// ---------------------------------------------------------------------------
// Individual checks
// ---------------------------------------------------------------------------

fn check_schema_version(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.schema_version == 0 {
        f.push(ValidationFinding {
            check: "schema_version",
            severity: FindingSeverity::Error,
            message: "schema_version must be >= 1".into(),
        });
    } else if m.schema_version > MANIFEST_SCHEMA_VERSION {
        f.push(ValidationFinding {
            check: "schema_version",
            severity: FindingSeverity::Warning,
            message: format!(
                "schema_version {} is newer than supported {}; forward-compat may lose fields",
                m.schema_version, MANIFEST_SCHEMA_VERSION
            ),
        });
    }
}

fn check_generation_id(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.generation_id.is_empty() {
        f.push(ValidationFinding {
            check: "generation_id",
            severity: FindingSeverity::Error,
            message: "generation_id must not be empty".into(),
        });
    }
}

fn check_manifest_hash(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.manifest_hash.is_empty() {
        f.push(ValidationFinding {
            check: "manifest_hash",
            severity: FindingSeverity::Error,
            message: "manifest_hash must not be empty".into(),
        });
        return;
    }
    if !is_valid_sha256_hex(&m.manifest_hash) {
        f.push(ValidationFinding {
            check: "manifest_hash",
            severity: FindingSeverity::Error,
            message: "manifest_hash must be 64 lowercase/uppercase hex chars".into(),
        });
        return;
    }

    match compute_manifest_hash(m) {
        Ok(expected) => {
            if !m.manifest_hash.eq_ignore_ascii_case(&expected) {
                f.push(ValidationFinding {
                    check: "manifest_hash",
                    severity: FindingSeverity::Error,
                    message: format!(
                        "manifest_hash does not match canonical manifest body (expected {expected})"
                    ),
                });
            }
        }
        Err(err) => {
            f.push(ValidationFinding {
                check: "manifest_hash",
                severity: FindingSeverity::Error,
                message: format!("failed to recompute manifest_hash: {err}"),
            });
        }
    }
}

fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|c| c.is_ascii_hexdigit())
}

fn check_commit_range(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.commit_range.is_empty() {
        f.push(ValidationFinding {
            check: "commit_range",
            severity: FindingSeverity::Error,
            message: format!(
                "commit_range is invalid: high ({}) < low ({})",
                m.commit_range.high, m.commit_range.low
            ),
        });
    }
}

fn check_timestamps(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.build_started_at == 0 {
        f.push(ValidationFinding {
            check: "build_started_at",
            severity: FindingSeverity::Error,
            message: "build_started_at must be a positive Unix timestamp".into(),
        });
    }
    if m.build_completed_at == 0 {
        f.push(ValidationFinding {
            check: "build_completed_at",
            severity: FindingSeverity::Error,
            message: "build_completed_at must be a positive Unix timestamp".into(),
        });
    }
    if m.build_completed_at < m.build_started_at {
        f.push(ValidationFinding {
            check: "build_timestamps",
            severity: FindingSeverity::Error,
            message: format!(
                "build_completed_at ({}) is before build_started_at ({})",
                m.build_completed_at, m.build_started_at
            ),
        });
    }
}

fn check_embedders(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    if m.embedders.is_empty() {
        f.push(ValidationFinding {
            check: "embedders",
            severity: FindingSeverity::Error,
            message: "at least one embedder revision must be specified".into(),
        });
    }
    for (key, rev) in &m.embedders {
        if rev.model_name.is_empty() {
            f.push(ValidationFinding {
                check: "embedder_model_name",
                severity: FindingSeverity::Error,
                message: format!("embedder '{key}' has empty model_name"),
            });
        }
        if rev.weights_hash.is_empty() {
            f.push(ValidationFinding {
                check: "embedder_weights_hash",
                severity: FindingSeverity::Error,
                message: format!("embedder '{key}' has empty weights_hash"),
            });
        }
        if rev.dimension == 0 {
            f.push(ValidationFinding {
                check: "embedder_dimension",
                severity: FindingSeverity::Error,
                message: format!("embedder '{key}' has dimension 0"),
            });
        }
    }
}

fn check_vector_artifacts(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    for (i, art) in m.vector_artifacts.iter().enumerate() {
        if art.path.is_empty() {
            f.push(ValidationFinding {
                check: "vector_artifact_path",
                severity: FindingSeverity::Error,
                message: format!("vector_artifacts[{i}] has empty path"),
            });
        }
        if art.checksum.is_empty() {
            f.push(ValidationFinding {
                check: "vector_artifact_checksum",
                severity: FindingSeverity::Error,
                message: format!("vector_artifacts[{i}] '{}' has empty checksum", art.path),
            });
        }
        if art.dimension == 0 {
            f.push(ValidationFinding {
                check: "vector_artifact_dimension",
                severity: FindingSeverity::Error,
                message: format!("vector_artifacts[{i}] '{}' has dimension 0", art.path),
            });
        }
    }

    // Check for duplicate paths.
    let mut seen = std::collections::HashSet::new();
    for art in &m.vector_artifacts {
        if !seen.insert(&art.path) {
            f.push(ValidationFinding {
                check: "vector_artifact_duplicate",
                severity: FindingSeverity::Error,
                message: format!("duplicate vector artifact path: '{}'", art.path),
            });
        }
    }
}

fn check_lexical_artifacts(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    for (i, art) in m.lexical_artifacts.iter().enumerate() {
        if art.path.is_empty() {
            f.push(ValidationFinding {
                check: "lexical_artifact_path",
                severity: FindingSeverity::Error,
                message: format!("lexical_artifacts[{i}] has empty path"),
            });
        }
        if art.checksum.is_empty() {
            f.push(ValidationFinding {
                check: "lexical_artifact_checksum",
                severity: FindingSeverity::Error,
                message: format!("lexical_artifacts[{i}] '{}' has empty checksum", art.path),
            });
        }
    }

    let mut seen = std::collections::HashSet::new();
    for art in &m.lexical_artifacts {
        if !seen.insert(&art.path) {
            f.push(ValidationFinding {
                check: "lexical_artifact_duplicate",
                severity: FindingSeverity::Error,
                message: format!("duplicate lexical artifact path: '{}'", art.path),
            });
        }
    }
}

fn check_repair_descriptors(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    let all_artifact_paths: std::collections::HashSet<&str> = m
        .vector_artifacts
        .iter()
        .map(|a| a.path.as_str())
        .chain(m.lexical_artifacts.iter().map(|a| a.path.as_str()))
        .collect();

    for (i, rd) in m.repair_descriptors.iter().enumerate() {
        if !all_artifact_paths.contains(rd.protected_artifact.as_str()) {
            f.push(ValidationFinding {
                check: "repair_descriptor_target",
                severity: FindingSeverity::Error,
                message: format!(
                    "repair_descriptors[{i}] references unknown artifact '{}'",
                    rd.protected_artifact
                ),
            });
        }
        if rd.source_symbols == 0 {
            f.push(ValidationFinding {
                check: "repair_descriptor_symbols",
                severity: FindingSeverity::Error,
                message: format!(
                    "repair_descriptors[{i}] for '{}' has 0 source symbols",
                    rd.protected_artifact
                ),
            });
        }
        if rd.overhead_ratio.is_nan() || rd.overhead_ratio < 0.0 || rd.overhead_ratio > 10.0 {
            f.push(ValidationFinding {
                check: "repair_descriptor_overhead",
                severity: FindingSeverity::Warning,
                message: format!(
                    "repair_descriptors[{i}] overhead ratio {} is outside expected range [0, 10]",
                    rd.overhead_ratio
                ),
            });
        }
    }
}

fn check_activation_invariants(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    let mut seen_ids = std::collections::HashSet::new();
    for inv in &m.activation_invariants {
        if inv.id.is_empty() {
            f.push(ValidationFinding {
                check: "invariant_id",
                severity: FindingSeverity::Error,
                message: "activation invariant has empty id".into(),
            });
        }
        if !seen_ids.insert(&inv.id) {
            f.push(ValidationFinding {
                check: "invariant_duplicate",
                severity: FindingSeverity::Error,
                message: format!("duplicate activation invariant id: '{}'", inv.id),
            });
        }
    }
}

fn check_document_count_consistency(m: &GenerationManifest, f: &mut Vec<ValidationFinding>) {
    let vector_total: u64 = m.vector_artifacts.iter().map(|a| a.vector_count).sum();
    let lexical_total: u64 = m.lexical_artifacts.iter().map(|a| a.document_count).sum();

    // Vector count should match declared total (per tier, so may be 2x for two-tier).
    // We only warn if there's a gross mismatch.
    if m.total_documents == 0 && (!m.vector_artifacts.is_empty() || !m.lexical_artifacts.is_empty())
    {
        f.push(ValidationFinding {
            check: "total_documents",
            severity: FindingSeverity::Error,
            message: "total_documents is 0 but artifacts are present".into(),
        });
    }

    if !m.lexical_artifacts.is_empty() && lexical_total != m.total_documents {
        f.push(ValidationFinding {
            check: "lexical_document_count",
            severity: FindingSeverity::Warning,
            message: format!(
                "lexical document count ({lexical_total}) != total_documents ({})",
                m.total_documents
            ),
        });
    }

    // For two-tier, vector_total may be 2 * total_documents (fast + quality tier).
    // Flag only if it doesn't match any reasonable multiple.
    if !m.vector_artifacts.is_empty()
        && vector_total != m.total_documents
        && vector_total != m.total_documents * 2
    {
        f.push(ValidationFinding {
            check: "vector_count_consistency",
            severity: FindingSeverity::Warning,
            message: format!(
                "vector count ({vector_total}) doesn't match total_documents ({}) or 2x (two-tier)",
                m.total_documents
            ),
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_embedder() -> EmbedderRevision {
        EmbedderRevision {
            model_name: "potion-128M".into(),
            weights_hash: "abcdef1234567890".into(),
            dimension: 256,
            quantization: QuantizationFormat::F16,
        }
    }

    fn sample_vector_artifact(path: &str, count: u64) -> VectorArtifact {
        VectorArtifact {
            path: path.into(),
            size_bytes: 1024,
            checksum: "deadbeef".into(),
            vector_count: count,
            dimension: 256,
            embedder_tier: EmbedderTierTag::Fast,
        }
    }

    fn sample_lexical_artifact(path: &str, count: u64) -> LexicalArtifact {
        LexicalArtifact {
            path: path.into(),
            size_bytes: 2048,
            checksum: "cafebabe".into(),
            document_count: count,
        }
    }

    fn valid_manifest() -> GenerationManifest {
        let mut embedders = BTreeMap::new();
        embedders.insert("fast".into(), sample_embedder());

        let mut manifest = GenerationManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            generation_id: "gen-001".into(),
            manifest_hash: String::new(),
            commit_range: CommitRange { low: 1, high: 100 },
            build_started_at: 1_700_000_000_000,
            build_completed_at: 1_700_000_060_000,
            embedders,
            vector_artifacts: vec![sample_vector_artifact("vectors/shard_0.fsvi", 100)],
            lexical_artifacts: vec![sample_lexical_artifact("lexical/segment_0", 100)],
            repair_descriptors: vec![RepairDescriptor {
                protected_artifact: "vectors/shard_0.fsvi".into(),
                sidecar_path: "vectors/shard_0.fsvi.fec".into(),
                source_symbols: 64,
                repair_symbols: 13,
                overhead_ratio: 0.2,
            }],
            activation_invariants: vec![
                ActivationInvariant {
                    id: "all_artifacts".into(),
                    description: "All artifacts verified".into(),
                    kind: InvariantKind::AllArtifactsVerified,
                },
                ActivationInvariant {
                    id: "embedder_match".into(),
                    description: "Embedder revision matches runtime".into(),
                    kind: InvariantKind::EmbedderRevisionMatch,
                },
            ],
            total_documents: 100,
            metadata: BTreeMap::new(),
        };
        manifest.manifest_hash = compute_manifest_hash(&manifest).expect("hash");
        manifest
    }

    fn refresh_manifest_hash(manifest: &mut GenerationManifest) {
        manifest.manifest_hash = compute_manifest_hash(manifest).expect("hash");
    }

    #[test]
    fn valid_manifest_passes() {
        let m = valid_manifest();
        let r = validate_manifest(&m);
        assert!(r.is_valid(), "findings: {:#?}", r.findings);
        assert!(r.errors().is_empty());
    }

    #[test]
    fn schema_version_zero_is_error() {
        let mut m = valid_manifest();
        m.schema_version = 0;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "schema_version"));
    }

    #[test]
    fn future_schema_version_is_warning() {
        let mut m = valid_manifest();
        m.schema_version = MANIFEST_SCHEMA_VERSION + 1;
        refresh_manifest_hash(&mut m);
        let r = validate_manifest(&m);
        assert!(r.is_valid());
        assert!(!r.warnings().is_empty());
    }

    #[test]
    fn empty_generation_id_is_error() {
        let mut m = valid_manifest();
        m.generation_id = String::new();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "generation_id"));
    }

    #[test]
    fn empty_manifest_hash_is_error() {
        let mut m = valid_manifest();
        m.manifest_hash.clear();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "manifest_hash"));
    }

    #[test]
    fn malformed_manifest_hash_is_error() {
        let mut m = valid_manifest();
        m.manifest_hash = "not-a-sha256".into();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "manifest_hash"));
    }

    #[test]
    fn mismatched_manifest_hash_is_error() {
        let mut m = valid_manifest();
        m.manifest_hash = "0".repeat(64);
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "manifest_hash"
                    && f.message.contains("does not match canonical"))
        );
    }

    #[test]
    fn manifest_hash_match_is_case_insensitive() {
        let mut m = valid_manifest();
        m.manifest_hash = m.manifest_hash.to_uppercase();
        let r = validate_manifest(&m);
        assert!(r.is_valid());
    }

    #[test]
    fn invalid_commit_range_is_error() {
        let mut m = valid_manifest();
        m.commit_range = CommitRange { low: 50, high: 10 };
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "commit_range"));
    }

    #[test]
    fn zero_timestamps_are_errors() {
        let mut m = valid_manifest();
        m.build_started_at = 0;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "build_started_at"));
    }

    #[test]
    fn completed_before_started_is_error() {
        let mut m = valid_manifest();
        m.build_completed_at = m.build_started_at - 1;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "build_timestamps"));
    }

    #[test]
    fn no_embedders_is_error() {
        let mut m = valid_manifest();
        m.embedders.clear();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "embedders"));
    }

    #[test]
    fn embedder_empty_fields_are_errors() {
        let mut m = valid_manifest();
        m.embedders.insert(
            "bad".into(),
            EmbedderRevision {
                model_name: String::new(),
                weights_hash: String::new(),
                dimension: 0,
                quantization: QuantizationFormat::F16,
            },
        );
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        let errors = r.errors();
        assert!(errors.iter().any(|f| f.check == "embedder_model_name"));
        assert!(errors.iter().any(|f| f.check == "embedder_weights_hash"));
        assert!(errors.iter().any(|f| f.check == "embedder_dimension"));
    }

    #[test]
    fn duplicate_vector_artifact_paths_is_error() {
        let mut m = valid_manifest();
        m.vector_artifacts
            .push(sample_vector_artifact("vectors/shard_0.fsvi", 100));
        m.total_documents = 200;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "vector_artifact_duplicate")
        );
    }

    #[test]
    fn empty_artifact_path_is_error() {
        let mut m = valid_manifest();
        m.vector_artifacts.push(sample_vector_artifact("", 10));
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "vector_artifact_path"));
    }

    #[test]
    fn empty_artifact_checksum_is_error() {
        let mut m = valid_manifest();
        m.vector_artifacts[0].checksum = String::new();
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "vector_artifact_checksum")
        );
    }

    #[test]
    fn repair_descriptor_unknown_artifact_is_error() {
        let mut m = valid_manifest();
        m.repair_descriptors.push(RepairDescriptor {
            protected_artifact: "nonexistent.fsvi".into(),
            sidecar_path: "nonexistent.fsvi.fec".into(),
            source_symbols: 10,
            repair_symbols: 2,
            overhead_ratio: 0.2,
        });
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "repair_descriptor_target")
        );
    }

    #[test]
    fn repair_descriptor_zero_source_symbols_is_error() {
        let mut m = valid_manifest();
        m.repair_descriptors[0].source_symbols = 0;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(
            r.errors()
                .iter()
                .any(|f| f.check == "repair_descriptor_symbols")
        );
    }

    #[test]
    fn extreme_repair_overhead_is_warning() {
        let mut m = valid_manifest();
        m.repair_descriptors[0].overhead_ratio = 15.0;
        refresh_manifest_hash(&mut m);
        let r = validate_manifest(&m);
        assert!(r.is_valid());
        assert!(
            r.warnings()
                .iter()
                .any(|f| f.check == "repair_descriptor_overhead")
        );
    }

    #[test]
    fn duplicate_invariant_id_is_error() {
        let mut m = valid_manifest();
        m.activation_invariants
            .push(m.activation_invariants[0].clone());
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "invariant_duplicate"));
    }

    #[test]
    fn zero_total_documents_with_artifacts_is_error() {
        let mut m = valid_manifest();
        m.total_documents = 0;
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        assert!(r.errors().iter().any(|f| f.check == "total_documents"));
    }

    #[test]
    fn lexical_count_mismatch_is_warning() {
        let mut m = valid_manifest();
        m.lexical_artifacts[0].document_count = 50; // != total_documents (100)
        refresh_manifest_hash(&mut m);
        let r = validate_manifest(&m);
        assert!(r.is_valid());
        assert!(
            r.warnings()
                .iter()
                .any(|f| f.check == "lexical_document_count")
        );
    }

    #[test]
    fn two_tier_vector_count_accepted() {
        let mut m = valid_manifest();
        // Fast tier: 100 vectors + Quality tier: 100 vectors = 200 total, 100 docs
        m.vector_artifacts = vec![
            sample_vector_artifact("vectors/fast.fsvi", 100),
            sample_vector_artifact("vectors/quality.fsvi", 100),
        ];
        m.vector_artifacts[1].embedder_tier = EmbedderTierTag::Quality;
        // Update repair descriptor to reference the new fast tier artifact.
        m.repair_descriptors[0].protected_artifact = "vectors/fast.fsvi".into();
        refresh_manifest_hash(&mut m);
        let r = validate_manifest(&m);
        assert!(r.is_valid(), "findings: {:#?}", r.findings);
    }

    #[test]
    fn serde_roundtrip() {
        let m = valid_manifest();
        let json = serde_json::to_string_pretty(&m).expect("serialize");
        let deserialized: GenerationManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(m, deserialized);
    }

    #[test]
    fn commit_range_len_and_empty() {
        let range = CommitRange { low: 5, high: 10 };
        assert_eq!(range.len(), 6);
        assert!(!range.is_empty());

        let empty = CommitRange { low: 10, high: 5 };
        assert!(empty.is_empty());
    }

    #[test]
    fn single_commit_range() {
        let range = CommitRange { low: 42, high: 42 };
        assert_eq!(range.len(), 1);
        assert!(!range.is_empty());
    }

    #[test]
    fn require_valid_passes_for_valid_manifest() {
        let m = valid_manifest();
        let r = validate_manifest(&m);
        assert!(require_valid(&r).is_ok());
    }

    #[test]
    fn require_valid_fails_for_invalid_manifest() {
        let mut m = valid_manifest();
        m.generation_id = String::new();
        let r = validate_manifest(&m);
        let err = require_valid(&r).unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn empty_manifest_collects_multiple_errors() {
        let m = GenerationManifest {
            schema_version: 0,
            generation_id: String::new(),
            manifest_hash: String::new(),
            commit_range: CommitRange { low: 10, high: 5 },
            build_started_at: 0,
            build_completed_at: 0,
            embedders: BTreeMap::new(),
            vector_artifacts: vec![],
            lexical_artifacts: vec![],
            repair_descriptors: vec![],
            activation_invariants: vec![],
            total_documents: 0,
            metadata: BTreeMap::new(),
        };
        let r = validate_manifest(&m);
        assert!(!r.is_valid());
        // Should find at least: schema_version, generation_id, commit_range,
        // build_started_at, build_completed_at, embedders
        assert!(r.errors().len() >= 5, "found {} errors", r.errors().len());
    }

    #[test]
    fn metadata_is_preserved() {
        let mut m = valid_manifest();
        m.metadata.insert("build_host".into(), "node-7".into());
        m.metadata.insert("deployment".into(), "production".into());
        let json = serde_json::to_string(&m).expect("serialize");
        let deserialized: GenerationManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.metadata.get("build_host").unwrap(), "node-7");
    }

    #[test]
    fn invariant_kinds_serialize() {
        let kinds = vec![
            InvariantKind::AllArtifactsVerified,
            InvariantKind::EmbedderRevisionMatch,
            InvariantKind::VectorCountConsistency {
                expected_total: 500,
            },
            InvariantKind::CommitContinuity { previous_high: 99 },
            InvariantKind::Custom {
                check_name: "custom_check".into(),
            },
        ];
        for kind in &kinds {
            let json = serde_json::to_string(kind).expect("serialize");
            let back: InvariantKind = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(kind, &back);
        }
    }

    #[test]
    fn quantization_format_serialize() {
        for fmt in &[
            QuantizationFormat::F32,
            QuantizationFormat::F16,
            QuantizationFormat::Int8,
            QuantizationFormat::Int4,
        ] {
            let json = serde_json::to_string(fmt).expect("serialize");
            let back: QuantizationFormat = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(fmt, &back);
        }
    }
}
