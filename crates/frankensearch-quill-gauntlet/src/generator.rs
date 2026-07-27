//! Deterministic corpus and query generation for Quill differential campaigns.
//!
//! Every random-access document derives its own integer-only pseudo-random
//! stream. Generation is therefore invariant to iteration order, sharding, and
//! retries. Corpus identities hash the canonical generated documents, not only
//! their recipes.

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
#[cfg(not(windows))]
use std::fs::File;
#[cfg(not(windows))]
use std::io::Read as _;
use std::path::{Component, Path, PathBuf};

use frankensearch_core::IndexableDocument;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::DifferentialCaseMetadata;
use crate::{DifferentialCase, GauntletError};

/// Schema version for generator specifications and manifests.
pub const GENERATOR_SCHEMA_VERSION: u32 = 2;
/// Schema version for query manifests with explicit suite provenance and the
/// complete CASS Boolean/glob/range/structured-filter profile.
pub const QUERY_MANIFEST_SCHEMA_VERSION: u32 = 3;
/// Stable identity of this generator implementation.
pub const GENERATOR_ID: &str = "frankensearch-quill-gauntlet/generator-v2";
/// Maximum accepted document size, in UTF-8 bytes.
pub const MAX_DOCUMENT_BYTES: u32 = 2 * 1024 * 1024;
/// Maximum UTF-8 bytes in an external document ID carried into observations.
pub const MAX_DOCUMENT_ID_BYTES: usize = 64 * 1024;
/// Number of documents in the performance-gate xlarge corpus.
pub const XLARGE_DOCUMENT_COUNT: u64 = 1_000_000;
/// Maximum corpus size accepted by manifests and campaign generators.
pub const MAX_CORPUS_DOCUMENT_COUNT: u64 = XLARGE_DOCUMENT_COUNT;
/// Maximum cases accepted in one query manifest.
pub const MAX_QUERY_CASES: usize = 100_000;
/// Maximum UTF-8 bytes in one raw query.
pub const MAX_QUERY_TEXT_BYTES: usize = 1024 * 1024;
/// Maximum UTF-8 bytes across all variable text in one query suite.
pub const MAX_QUERY_SUITE_TEXT_BYTES: usize = 256 * 1024 * 1024;
/// Maximum UTF-8 bytes in one stable query or divergence ID.
pub const MAX_QUERY_ID_BYTES: usize = 1_024;
/// Number of documents in the original five-cluster relevance fixture.
pub const CORE_RELEVANCE_DOCUMENT_COUNT: usize = 100;
/// Number of documents in the complete shared fixture corpus.
pub const FULL_SHARED_DOCUMENT_COUNT: usize = 120;

const MAX_VOCABULARY_SIZE: u32 = 65_536;
const DEFAULT_VOCABULARY_SIZE: u32 = 4_096;
const MAX_FREQUENCY_REPETITIONS: usize = 65_536;
const MAX_GENERATED_QUERY_LIMIT: u64 = 100_000;
const MAX_QUERY_SOURCE_BYTES: usize = 64 * 1024;
const MAX_STRUCTURED_FILTER_VALUES: usize = 64;
const MAX_STRUCTURED_FILTER_VALUE_BYTES: usize = 4 * 1024;
const MAX_DOCUMENT_AUXILIARY_BYTES: u64 = 2 * 1024 * 1024;
const MAX_CANONICAL_DOCUMENT_BYTES: u64 = 32 * 1024 * 1024;
const MAX_REPOSITORY_ID_BYTES: usize = 1_024;
const MAX_REPOSITORY_PATH_BYTES: usize = MAX_DOCUMENT_ID_BYTES - "repo:".len();
const MAX_REPOSITORY_MANIFEST_TEXT_BYTES: usize = 256 * 1024 * 1024;
const CORPUS_HASH_DOMAIN: &[u8] = b"frankensearch/quill/corpus-content/v1\0";
const MANIFEST_HASH_DOMAIN: &[u8] = b"frankensearch/quill/corpus-manifest/v1\0";
const QUERY_HASH_DOMAIN: &[u8] = b"frankensearch/quill/query-content/v1\0";
const QUERY_MANIFEST_HASH_DOMAIN: &[u8] = b"frankensearch/quill/query-manifest/v1\0";
const QUERY_SOURCE_IDENTITY_DOMAIN: &[u8] = b"frankensearch/quill/query-source/v1\0";

const SHARED_CORPUS_JSON: &str = include_str!("../../../tests/fixtures/corpus.json");
const SHARED_QUERIES_JSON: &str = include_str!("../../../tests/fixtures/queries.json");
const SHARED_EDGE_CASES_JSON: &str = include_str!("../../../tests/fixtures/edge_cases.json");
const LANGUAGE_CONTRACT_JSON: &str =
    include_str!("../../../tests/fixtures/quill_language_contract.json");

/// Supported deterministic Zipf skews, represented without floating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZipfExponent {
    /// `s = 0.8`, the flattest required distribution.
    S08,
    /// `s = 1.1`, the middle required distribution.
    S11,
    /// `s = 1.4`, the most head-heavy required distribution.
    S14,
}

impl ZipfExponent {
    /// All skews required by the Quill performance contract.
    pub const ALL: [Self; 3] = [Self::S08, Self::S11, Self::S14];
}

/// Deterministic synthetic-corpus recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntheticCorpusSpec {
    /// Seed for all document-local streams.
    pub seed: u64,
    /// Number of random-access documents.
    pub document_count: u64,
    /// Number of terms in the sampled vocabulary.
    pub vocabulary_size: u32,
    /// Required Zipf skew.
    pub zipf_exponent: ZipfExponent,
    /// UTF-8 byte cap, never greater than [`MAX_DOCUMENT_BYTES`].
    pub max_document_bytes: u32,
}

impl SyntheticCorpusSpec {
    /// Construct the pinned one-million-document performance recipe.
    #[must_use]
    pub const fn xlarge(seed: u64, zipf_exponent: ZipfExponent) -> Self {
        Self {
            seed,
            document_count: XLARGE_DOCUMENT_COUNT,
            vocabulary_size: DEFAULT_VOCABULARY_SIZE,
            zipf_exponent,
            max_document_bytes: MAX_DOCUMENT_BYTES,
        }
    }

    /// Construct all three xlarge Zipf lanes from one seed.
    #[must_use]
    pub fn xlarge_matrix(seed: u64) -> [Self; 3] {
        ZipfExponent::ALL.map(|zipf_exponent| Self::xlarge(seed, zipf_exponent))
    }

    fn validate(&self) -> Result<(), GauntletError> {
        if self.document_count > MAX_CORPUS_DOCUMENT_COUNT {
            return Err(generator_error(format!(
                "document_count must be at most {MAX_CORPUS_DOCUMENT_COUNT}"
            )));
        }
        if self.vocabulary_size == 0 || self.vocabulary_size > MAX_VOCABULARY_SIZE {
            return Err(generator_error(format!(
                "vocabulary_size must be in 1..={MAX_VOCABULARY_SIZE}"
            )));
        }
        if self.max_document_bytes < 256 || self.max_document_bytes > MAX_DOCUMENT_BYTES {
            return Err(generator_error(format!(
                "max_document_bytes must be in 256..={MAX_DOCUMENT_BYTES}"
            )));
        }
        Ok(())
    }
}

/// Script lane used by a generated document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnicodeLane {
    /// Seven-bit ASCII.
    Ascii,
    /// Composed and decomposed Latin-1 text.
    Latin1,
    /// Chinese, Japanese, and Korean BMP characters.
    Cjk,
    /// CJK Unified Ideographs Extension B, encoded as four-byte UTF-8.
    CjkExtensionB,
    /// A deliberate mixture of ASCII, Latin-1, CJK, and Extension B.
    Mixed,
}

impl UnicodeLane {
    const ALL: [Self; 5] = [
        Self::Ascii,
        Self::Latin1,
        Self::Cjk,
        Self::CjkExtensionB,
        Self::Mixed,
    ];
}

/// Deliberate edge condition carried by a generated document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pathology {
    /// No content bytes.
    Empty,
    /// Whitespace-only content.
    Whitespace,
    /// One token.
    SingleToken,
    /// One term exactly 256 bytes long.
    Term256Bytes,
    /// Hyphen compounds that emit same-position alternatives.
    SamePositionHyphens,
    /// More than 65,535 occurrences of one term when the byte cap permits it.
    MaximumFrequency,
    /// A composed/decomposed Latin-1 mix.
    Latin1Normalization,
    /// CJK Extension-B boundary characters.
    CjkExtensionB,
    /// Mixed scripts in one document.
    MixedScripts,
    /// A code-like document filled close to its configured byte cap.
    NearLimitCode,
}

/// Typed CASS fields retained until an adapter-specific runner lowers them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CassDocumentFields {
    /// Agent name.
    pub agent: String,
    /// Workspace name.
    pub workspace: String,
    /// Stable source identifier.
    pub source_id: String,
    /// Repository-like source path.
    pub source_path: String,
    /// Origin-kind discriminator.
    pub origin_kind: String,
    /// Message position in the source.
    pub message_index: u64,
}

/// Engine-neutral generated document with canonical metadata ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedDocument {
    /// Stable document identifier.
    pub id: String,
    /// Optional boosted title.
    pub title: Option<String>,
    /// Searchable UTF-8 content.
    pub content: String,
    /// Deterministic numeric time used by CASS range fixtures.
    pub created_at_ms: i64,
    /// Typed CASS-only fields.
    pub cass: Option<CassDocumentFields>,
    /// Canonically ordered extensible metadata.
    pub metadata: BTreeMap<String, String>,
    /// Deliberate edge condition, when this is a pathology anchor.
    pub pathology: Option<Pathology>,
    /// Script lane used by the content.
    pub unicode_lane: UnicodeLane,
}

impl From<GeneratedDocument> for IndexableDocument {
    fn from(document: GeneratedDocument) -> Self {
        let mut metadata = document
            .metadata
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        metadata.insert(
            "created_at_ms".to_owned(),
            document.created_at_ms.to_string(),
        );
        if let Some(cass) = document.cass {
            metadata.insert("agent".to_owned(), cass.agent);
            metadata.insert("workspace".to_owned(), cass.workspace);
            metadata.insert("source_id".to_owned(), cass.source_id);
            metadata.insert("source_path".to_owned(), cass.source_path);
            metadata.insert("origin_kind".to_owned(), cass.origin_kind);
            metadata.insert("message_index".to_owned(), cass.message_index.to_string());
        }
        Self {
            id: document.id,
            content: document.content,
            title: document.title,
            metadata,
        }
    }
}

/// Synthetic corpus with deterministic random access and streaming iteration.
#[derive(Debug, Clone)]
pub struct SyntheticCorpus {
    spec: SyntheticCorpusSpec,
    zipf_cdf: Vec<u64>,
}

impl SyntheticCorpus {
    /// Validate a recipe and prepare its fixed-point Zipf lookup table.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid vocabulary or document-byte cap.
    pub fn new(spec: SyntheticCorpusSpec) -> Result<Self, GauntletError> {
        spec.validate()?;
        let zipf_cdf = build_zipf_cdf(spec.vocabulary_size, spec.zipf_exponent)?;
        Ok(Self { spec, zipf_cdf })
    }

    /// The exact replay recipe.
    #[must_use]
    pub const fn spec(&self) -> &SyntheticCorpusSpec {
        &self.spec
    }

    /// Number of generated documents.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.spec.document_count
    }

    /// Whether the corpus is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.spec.document_count == 0
    }

    /// Generate one document without materializing preceding documents.
    #[must_use]
    pub fn document_at(&self, index: u64) -> Option<GeneratedDocument> {
        (index < self.spec.document_count).then(|| self.generate_document(index))
    }

    /// Stream documents in ordinal order.
    #[must_use]
    pub const fn iter(&self) -> SyntheticCorpusIter<'_> {
        SyntheticCorpusIter {
            corpus: self,
            next_index: 0,
        }
    }

    /// Stream the complete corpus into a content-addressed manifest.
    ///
    /// This intentionally performs `O(document_count)` work while holding only
    /// one generated document at a time.
    ///
    /// # Errors
    ///
    /// Returns an error if canonical JSON serialization fails.
    pub fn manifest(&self) -> Result<CorpusManifest, GauntletError> {
        CorpusManifest::from_documents(
            CorpusSourceManifest::Synthetic {
                spec: self.spec.clone(),
            },
            self.iter(),
            Vec::new(),
        )
    }

    /// Replay and verify a previously recorded synthetic manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if the recipe, streamed content, counts, or hash differ.
    pub fn verify_manifest(&self, expected: &CorpusManifest) -> Result<(), GauntletError> {
        let actual = self.manifest()?;
        if &actual != expected {
            return Err(GauntletError::ManifestMismatch {
                reason: "synthetic corpus replay differs from its manifest".to_owned(),
            });
        }
        Ok(())
    }

    fn generate_document(&self, index: u64) -> GeneratedDocument {
        let mut rng = DeterministicRng::for_stream(self.spec.seed, 0x434f_5250_5553, index);
        let lane_count = u64::try_from(UnicodeLane::ALL.len()).unwrap_or(1);
        let lane_index = usize::try_from(index % lane_count).unwrap_or(0);
        let regular_unicode_lane = UnicodeLane::ALL[lane_index];
        let (content, pathology, unicode_lane) = match index {
            0 => (String::new(), Some(Pathology::Empty), UnicodeLane::Ascii),
            1 => (
                "  \n\t  ".to_owned(),
                Some(Pathology::Whitespace),
                UnicodeLane::Ascii,
            ),
            2 => (
                "singleton".to_owned(),
                Some(Pathology::SingleToken),
                UnicodeLane::Ascii,
            ),
            3 => (
                "t".repeat(256),
                Some(Pathology::Term256Bytes),
                UnicodeLane::Ascii,
            ),
            4 => (
                "bd-q3fy high-performance state-of-the-art tail".to_owned(),
                Some(Pathology::SamePositionHyphens),
                UnicodeLane::Ascii,
            ),
            5 => (
                maximum_frequency_content(usize_from_u32(self.spec.max_document_bytes)),
                Some(Pathology::MaximumFrequency),
                UnicodeLane::Ascii,
            ),
            6 => (
                "naïve cafe\u{301} résumé coo\u{308}perate".to_owned(),
                Some(Pathology::Latin1Normalization),
                UnicodeLane::Latin1,
            ),
            7 => (
                "extension b boundary 𠀀 𪛟 搜索".to_owned(),
                Some(Pathology::CjkExtensionB),
                UnicodeLane::CjkExtensionB,
            ),
            8 => (
                "mixed ASCII café 搜索 𠀀 code::symbol".to_owned(),
                Some(Pathology::MixedScripts),
                UnicodeLane::Mixed,
            ),
            9 => (
                near_limit_code(usize_from_u32(self.spec.max_document_bytes)),
                Some(Pathology::NearLimitCode),
                UnicodeLane::Ascii,
            ),
            _ => {
                let regular = self.generate_regular_content(&mut rng, regular_unicode_lane);
                let content = match cass_profile_anchor(index) {
                    Some(anchor) => format!("{anchor} {regular}"),
                    None => regular,
                };
                (
                    truncate_utf8(content, usize_from_u32(self.spec.max_document_bytes)),
                    None,
                    regular_unicode_lane,
                )
            }
        };
        let mut metadata = BTreeMap::new();
        metadata.insert("generator_id".to_owned(), GENERATOR_ID.to_owned());
        metadata.insert("ordinal".to_owned(), index.to_string());
        metadata.insert("seed".to_owned(), self.spec.seed.to_string());
        metadata.insert(
            "zipf_exponent".to_owned(),
            match self.spec.zipf_exponent {
                ZipfExponent::S08 => "0.8",
                ZipfExponent::S11 => "1.1",
                ZipfExponent::S14 => "1.4",
            }
            .to_owned(),
        );
        GeneratedDocument {
            id: format!("synthetic-{index:08}"),
            title: Some(format!("Synthetic document {index}")),
            content: truncate_utf8(content, usize_from_u32(self.spec.max_document_bytes)),
            created_at_ms: 1_700_000_000_000_i64
                .saturating_add(i64::try_from(index).unwrap_or(i64::MAX)),
            cass: Some(CassDocumentFields {
                agent: format!("agent-{}", index % 7),
                workspace: format!("workspace-{}", index % 5),
                source_id: format!("source-{}", index % 11),
                source_path: format!("src/generated/{index:08}.rs"),
                origin_kind: if index.is_multiple_of(2) {
                    "local".to_owned()
                } else {
                    "ssh".to_owned()
                },
                message_index: index,
            }),
            metadata,
            pathology,
            unicode_lane,
        }
    }

    fn generate_regular_content(
        &self,
        rng: &mut DeterministicRng,
        unicode_lane: UnicodeLane,
    ) -> String {
        let cap = usize_from_u32(self.spec.max_document_bytes);
        let target = sampled_document_length(rng, cap);
        let mut content = String::with_capacity(target);
        let prefix = match unicode_lane {
            UnicodeLane::Ascii => "fn indexed_record() { let search = ",
            UnicodeLane::Latin1 => "naïve café re\u{301}sume\u{301} ",
            UnicodeLane::Cjk => "搜索 引擎 文書 ",
            UnicodeLane::CjkExtensionB => "𠀀 𪛟 extension_b ",
            UnicodeLane::Mixed => "mixed café 搜索 𠀀 ",
        };
        push_bounded(&mut content, prefix, target);
        while content.len() < target {
            let rank = self.sample_zipf_rank(rng);
            if !push_ranked_term_bounded(&mut content, rank, target) {
                break;
            }
        }
        content
    }

    fn sample_zipf_rank(&self, rng: &mut DeterministicRng) -> u32 {
        let total = self.zipf_cdf.last().copied().unwrap_or(1);
        let draw = rng.bounded(total);
        let index = self.zipf_cdf.partition_point(|&value| value <= draw);
        u32::try_from(index + 1).unwrap_or(self.spec.vocabulary_size)
    }
}

impl<'a> IntoIterator for &'a SyntheticCorpus {
    type Item = GeneratedDocument;
    type IntoIter = SyntheticCorpusIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Streaming iterator over a synthetic corpus.
#[derive(Debug, Clone)]
pub struct SyntheticCorpusIter<'a> {
    corpus: &'a SyntheticCorpus,
    next_index: u64,
}

impl Iterator for SyntheticCorpusIter<'_> {
    type Item = GeneratedDocument;

    fn next(&mut self) -> Option<Self::Item> {
        let document = self.corpus.document_at(self.next_index)?;
        self.next_index += 1;
        Some(document)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.corpus.len().saturating_sub(self.next_index);
        usize::try_from(remaining).map_or((usize::MAX, None), |exact| (exact, Some(exact)))
    }
}

impl std::iter::FusedIterator for SyntheticCorpusIter<'_> {}

/// Reason a tracked repository entry was not indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepositorySkipReason {
    /// The tracked path was absent at snapshot time.
    Missing,
    /// The entry was not a regular file or was a symbolic link.
    NotRegularFile,
    /// The file exceeded the 2 MiB ingest boundary.
    TooLarge,
    /// The file was not valid UTF-8.
    BinaryOrNonUtf8,
}

/// A skipped tracked path recorded in a repository manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedRepositoryEntry {
    /// Normalized repository-relative path.
    pub path: String,
    /// Deterministic exclusion reason.
    pub reason: RepositorySkipReason,
}

/// Digest of one included repository file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepositoryFileDigest {
    /// Normalized repository-relative path.
    pub path: String,
    /// UTF-8 byte count.
    pub bytes: u64,
    /// Lowercase SHA-256 of the raw file bytes.
    pub sha256: String,
}

/// Caller-supplied tracked entry, useful for VCS adapters and deterministic tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositoryEntry {
    /// Repository-relative path.
    pub relative_path: PathBuf,
    /// Raw tracked bytes.
    pub bytes: Vec<u8>,
}

/// Prepared real-repository corpus and its content-addressed manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositorySnapshot {
    /// Included UTF-8 documents in normalized path order.
    pub documents: Vec<GeneratedDocument>,
    /// Canonical corpus manifest.
    pub manifest: CorpusManifest,
}

impl RepositorySnapshot {
    /// Build a snapshot from already-discovered tracked entries.
    ///
    /// # Errors
    ///
    /// Returns an error for unsafe paths, duplicate paths, or serialization
    /// failures. Oversize and non-UTF-8 entries are recorded as skips.
    pub fn from_entries(
        repository_id: impl Into<String>,
        entries: impl IntoIterator<Item = RepositoryEntry>,
    ) -> Result<Self, GauntletError> {
        let repository_id = repository_id.into();
        validate_repository_id(&repository_id)?;
        let mut normalized = Vec::new();
        for entry in entries {
            let path = normalize_relative_path(&entry.relative_path)?;
            normalized.push((path, entry.bytes));
            if normalized.len() > usize::try_from(MAX_CORPUS_DOCUMENT_COUNT).unwrap_or(usize::MAX) {
                return Err(generator_error(format!(
                    "repository snapshots may contain at most {MAX_CORPUS_DOCUMENT_COUNT} entries"
                )));
            }
        }
        normalized.sort_by(|left, right| left.0.cmp(&right.0));
        if normalized.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(generator_error(
                "repository entries contain a duplicate path",
            ));
        }
        Self::prepare_repository(repository_id, normalized, Vec::new())
    }

    /// Read only an explicit list of VCS-tracked paths below `root`.
    ///
    /// Discovery remains caller-owned (for example, `git ls-files`); this API
    /// never traverses untracked scratch directories. Paths are normalized and
    /// sorted before any content identity is computed.
    ///
    /// # Errors
    ///
    /// Returns an error for unsafe paths, duplicate paths, filesystem failures,
    /// or serialization failures.
    pub fn from_tracked_paths(
        root: &Path,
        repository_id: impl Into<String>,
        tracked_paths: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, GauntletError> {
        let repository_id = repository_id.into();
        validate_repository_id(&repository_id)?;
        let mut paths = Vec::new();
        for path in tracked_paths {
            paths.push(normalize_relative_path(&path)?);
            if paths.len() > usize::try_from(MAX_CORPUS_DOCUMENT_COUNT).unwrap_or(usize::MAX) {
                return Err(generator_error(format!(
                    "repository snapshots may contain at most {MAX_CORPUS_DOCUMENT_COUNT} tracked paths"
                )));
            }
        }
        paths.sort();
        if paths.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(generator_error(
                "tracked path list contains a duplicate path",
            ));
        }

        let mut entries = Vec::new();
        let mut skipped = Vec::new();
        for normalized in paths {
            match read_tracked_path_beneath(root, &normalized)? {
                TrackedPathRead::Bytes(bytes) => entries.push((normalized, bytes)),
                TrackedPathRead::Skipped(reason) => skipped.push(SkippedRepositoryEntry {
                    path: normalized,
                    reason,
                }),
            }
        }
        Self::prepare_repository(repository_id, entries, skipped)
    }

    fn prepare_repository(
        repository_id: String,
        entries: Vec<(String, Vec<u8>)>,
        mut skipped: Vec<SkippedRepositoryEntry>,
    ) -> Result<Self, GauntletError> {
        let tracked_path_count = entries.len().checked_add(skipped.len());
        if tracked_path_count.is_none_or(|count| {
            count > usize::try_from(MAX_CORPUS_DOCUMENT_COUNT).unwrap_or(usize::MAX)
        }) {
            return Err(generator_error(format!(
                "repository snapshots may contain at most {MAX_CORPUS_DOCUMENT_COUNT} tracked paths"
            )));
        }
        let mut documents = Vec::new();
        let mut files = Vec::new();
        for (path, bytes) in entries {
            if bytes.len() > usize_from_u32(MAX_DOCUMENT_BYTES) {
                skipped.push(SkippedRepositoryEntry {
                    path,
                    reason: RepositorySkipReason::TooLarge,
                });
                continue;
            }
            let Ok(content) = String::from_utf8(bytes.clone()) else {
                skipped.push(SkippedRepositoryEntry {
                    path,
                    reason: RepositorySkipReason::BinaryOrNonUtf8,
                });
                continue;
            };
            files.push(RepositoryFileDigest {
                path: path.clone(),
                bytes: u64_from_usize(bytes.len()),
                sha256: sha256_hex(&bytes),
            });
            let mut metadata = BTreeMap::new();
            metadata.insert("repository_id".to_owned(), repository_id.clone());
            metadata.insert("source_path".to_owned(), path.clone());
            let unicode_lane = classify_unicode_lane(&content);
            documents.push(GeneratedDocument {
                id: format!("repo:{path}"),
                title: Some(path.clone()),
                content,
                created_at_ms: 0,
                cass: Some(CassDocumentFields {
                    agent: "repository".to_owned(),
                    workspace: repository_id.clone(),
                    source_id: path.clone(),
                    source_path: path,
                    origin_kind: "repository_snapshot".to_owned(),
                    message_index: u64_from_usize(documents.len()),
                }),
                metadata,
                pathology: None,
                unicode_lane,
            });
        }
        skipped.sort_by(|left, right| left.path.cmp(&right.path));
        let manifest = CorpusManifest::from_documents(
            CorpusSourceManifest::Repository {
                repository_id,
                files,
            },
            &documents,
            skipped,
        )?;
        Ok(Self {
            documents,
            manifest,
        })
    }
}

enum TrackedPathRead {
    Bytes(Vec<u8>),
    Skipped(RepositorySkipReason),
}

#[cfg(not(windows))]
fn read_tracked_path_beneath(
    root: &Path,
    normalized_path: &str,
) -> Result<TrackedPathRead, GauntletError> {
    use rustix::fs::{CWD, Mode, OFlags, openat};
    use rustix::io::Errno;

    let directory_flags = OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::DIRECTORY;
    let root_descriptor =
        openat(CWD, root, directory_flags, Mode::empty()).map_err(std::io::Error::from)?;
    let mut directory = File::from(root_descriptor);
    let mut components = normalized_path.split('/').peekable();
    while let Some(component) = components.next() {
        let flags = if components.peek().is_some() {
            directory_flags
        } else {
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK
        };
        let descriptor = match openat(&directory, component, flags, Mode::empty()) {
            Ok(descriptor) => descriptor,
            Err(Errno::NOENT) => {
                return Ok(TrackedPathRead::Skipped(RepositorySkipReason::Missing));
            }
            Err(Errno::LOOP | Errno::NOTDIR) => {
                return Ok(TrackedPathRead::Skipped(
                    RepositorySkipReason::NotRegularFile,
                ));
            }
            Err(error) => return Err(std::io::Error::from(error).into()),
        };
        if components.peek().is_some() {
            directory = File::from(descriptor);
            continue;
        }
        let file = File::from(descriptor);
        if !file.metadata()?.is_file() {
            return Ok(TrackedPathRead::Skipped(
                RepositorySkipReason::NotRegularFile,
            ));
        }
        let mut bytes = Vec::new();
        file.take(u64::from(MAX_DOCUMENT_BYTES) + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > usize::try_from(MAX_DOCUMENT_BYTES).unwrap_or(usize::MAX) {
            return Ok(TrackedPathRead::Skipped(RepositorySkipReason::TooLarge));
        }
        return Ok(TrackedPathRead::Bytes(bytes));
    }
    Err(generator_error("tracked path contains no components"))
}

#[cfg(windows)]
fn read_tracked_path_beneath(
    _root: &Path,
    _normalized_path: &str,
) -> Result<TrackedPathRead, GauntletError> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "descriptor-relative no-symlink repository snapshots are unsupported on Windows; use RepositorySnapshot::from_entries",
    )
    .into())
}

/// Identifies one embedded shared source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFileDigest {
    /// Stable repository-relative fixture path.
    pub path: String,
    /// Lowercase SHA-256 after normalizing committed CRLF text to LF.
    pub sha256: String,
}

/// Corpus source and exact replay inputs recorded in a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CorpusSourceManifest {
    /// Integer-only synthetic generator.
    Synthetic {
        /// Complete replay recipe.
        spec: SyntheticCorpusSpec,
    },
    /// Embedded shared test fixtures.
    SharedFixtures {
        /// Explicit 100-document or 120-document view.
        view: SharedCorpusView,
        /// Digests of every source fixture participating in the view.
        sources: Vec<SourceFileDigest>,
    },
    /// Explicit VCS-tracked repository snapshot.
    Repository {
        /// Stable logical repository identity, never an absolute path.
        repository_id: String,
        /// Included files in normalized path order.
        files: Vec<RepositoryFileDigest>,
    },
}

/// Canonical, replay-verifiable corpus identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusManifest {
    /// Generator/manifest schema.
    pub schema_version: u32,
    /// Stable generator implementation identity.
    pub generator_id: String,
    /// Exact source recipe.
    pub source: CorpusSourceManifest,
    /// Number of included documents.
    pub document_count: u64,
    /// Sum of included document content bytes.
    pub total_content_bytes: u64,
    /// SHA-256 over length-framed canonical document JSON.
    pub content_sha256: String,
    /// Stable exclusions for repository snapshots.
    pub skipped_repository_entries: Vec<SkippedRepositoryEntry>,
}

/// Incremental verifier used by the campaign's actual indexing replay.
pub struct CorpusReplayVerifier<'a> {
    hasher: Sha256,
    document_count: u64,
    total_content_bytes: u64,
    expected_document_count: u64,
    expected_content_bytes: u64,
    source: &'a CorpusSourceManifest,
}

impl CorpusReplayVerifier<'_> {
    pub(super) fn observe(&mut self, document: &GeneratedDocument) -> Result<u64, GauntletError> {
        validate_generated_document_bounds(document)?;
        if let CorpusSourceManifest::Repository {
            repository_id,
            files,
        } = self.source
        {
            let ordinal = usize::try_from(self.document_count)
                .map_err(|_| manifest_error("repository document ordinal does not fit usize"))?;
            let file = files.get(ordinal).ok_or_else(|| {
                manifest_error("repository replay contains more documents than file digests")
            })?;
            validate_repository_document(repository_id, file, self.document_count, document)?;
        }
        let bytes = serde_json::to_vec(document)?;
        let canonical_bytes = u64_from_usize(bytes.len());
        if canonical_bytes > MAX_CANONICAL_DOCUMENT_BYTES {
            return Err(manifest_error(
                "canonical generated document exceeds the bounded replay size",
            ));
        }
        self.hasher.update(canonical_bytes.to_be_bytes());
        self.hasher.update(bytes);
        self.document_count = self
            .document_count
            .checked_add(1)
            .ok_or_else(|| generator_error("document count overflow while verifying replay"))?;
        self.total_content_bytes = self
            .total_content_bytes
            .checked_add(u64::try_from(document.content.len()).unwrap_or(u64::MAX))
            .ok_or_else(|| generator_error("content byte count overflow while verifying replay"))?;
        if self.document_count > self.expected_document_count
            || self.total_content_bytes > self.expected_content_bytes
        {
            return Err(manifest_error(
                "indexed document replay exceeded corpus manifest bounds",
            ));
        }
        Ok(canonical_bytes)
    }

    pub(super) fn finish(self, manifest: &CorpusManifest) -> Result<(), GauntletError> {
        let digest = lower_hex(&self.hasher.finalize());
        if self.document_count != manifest.document_count
            || self.total_content_bytes != manifest.total_content_bytes
            || digest != manifest.content_sha256
        {
            return Err(manifest_error(
                "indexed document replay does not match corpus content pins",
            ));
        }
        Ok(())
    }
}

fn validate_generated_document_bounds(document: &GeneratedDocument) -> Result<(), GauntletError> {
    if document.id.is_empty()
        || document.id.len() > MAX_DOCUMENT_ID_BYTES
        || document.content.len() > usize_from_u32(MAX_DOCUMENT_BYTES)
    {
        return Err(manifest_error(
            "generated document ID or content exceeds its bounded contract",
        ));
    }
    let mut auxiliary_bytes = u64_from_usize(document.id.len());
    if let Some(title) = &document.title {
        auxiliary_bytes = auxiliary_bytes
            .checked_add(u64_from_usize(title.len()))
            .ok_or_else(|| manifest_error("generated document metadata byte count overflow"))?;
    }
    for (key, value) in &document.metadata {
        auxiliary_bytes = auxiliary_bytes
            .checked_add(u64_from_usize(key.len()))
            .and_then(|bytes| bytes.checked_add(u64_from_usize(value.len())))
            .ok_or_else(|| manifest_error("generated document metadata byte count overflow"))?;
    }
    if let Some(cass) = &document.cass {
        for value in [
            cass.agent.as_str(),
            cass.workspace.as_str(),
            cass.source_id.as_str(),
            cass.source_path.as_str(),
            cass.origin_kind.as_str(),
        ] {
            auxiliary_bytes = auxiliary_bytes
                .checked_add(u64_from_usize(value.len()))
                .ok_or_else(|| manifest_error("generated document metadata byte count overflow"))?;
        }
    }
    if auxiliary_bytes > MAX_DOCUMENT_AUXILIARY_BYTES {
        return Err(manifest_error(
            "generated document auxiliary metadata exceeds its bounded contract",
        ));
    }
    Ok(())
}

fn validate_repository_document(
    repository_id: &str,
    file: &RepositoryFileDigest,
    ordinal: u64,
    document: &GeneratedDocument,
) -> Result<(), GauntletError> {
    let expected_metadata = BTreeMap::from([
        ("repository_id".to_owned(), repository_id.to_owned()),
        ("source_path".to_owned(), file.path.clone()),
    ]);
    let expected_cass = CassDocumentFields {
        agent: "repository".to_owned(),
        workspace: repository_id.to_owned(),
        source_id: file.path.clone(),
        source_path: file.path.clone(),
        origin_kind: "repository_snapshot".to_owned(),
        message_index: ordinal,
    };
    let content_bytes = document.content.as_bytes();
    if document.id != format!("repo:{}", file.path)
        || document.title.as_deref() != Some(file.path.as_str())
        || u64_from_usize(content_bytes.len()) != file.bytes
        || sha256_hex(content_bytes) != file.sha256
        || document.created_at_ms != 0
        || document.cass.as_ref() != Some(&expected_cass)
        || document.metadata != expected_metadata
        || document.pathology.is_some()
        || document.unicode_lane != classify_unicode_lane(&document.content)
    {
        return Err(manifest_error(
            "repository replay document does not match its ordered source digest",
        ));
    }
    Ok(())
}

impl CorpusManifest {
    pub(super) fn replay_verifier(&self) -> CorpusReplayVerifier<'_> {
        let mut hasher = Sha256::new();
        hasher.update(CORPUS_HASH_DOMAIN);
        CorpusReplayVerifier {
            hasher,
            document_count: 0,
            total_content_bytes: 0,
            expected_document_count: self.document_count,
            expected_content_bytes: self.total_content_bytes,
            source: &self.source,
        }
    }

    fn from_documents<I, D>(
        source: CorpusSourceManifest,
        documents: I,
        skipped_repository_entries: Vec<SkippedRepositoryEntry>,
    ) -> Result<Self, GauntletError>
    where
        I: IntoIterator<Item = D>,
        D: Borrow<GeneratedDocument>,
    {
        let (document_count, total_content_bytes, content_sha256) = hash_documents(documents)?;
        Ok(Self {
            schema_version: GENERATOR_SCHEMA_VERSION,
            generator_id: GENERATOR_ID.to_owned(),
            source,
            document_count,
            total_content_bytes,
            content_sha256,
            skipped_repository_entries,
        })
    }

    /// Canonical manifest JSON bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GauntletError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Domain-separated lowercase SHA-256 of the canonical manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn manifest_hash(&self) -> Result<String, GauntletError> {
        let mut hasher = Sha256::new();
        hasher.update(MANIFEST_HASH_DOMAIN);
        hasher.update(self.canonical_bytes()?);
        Ok(lower_hex(&hasher.finalize()))
    }

    pub(crate) fn validate_contract(&self) -> Result<(), GauntletError> {
        if self.schema_version != GENERATOR_SCHEMA_VERSION
            || self.generator_id != GENERATOR_ID
            || self.document_count > MAX_CORPUS_DOCUMENT_COUNT
            || validate_sha256(&self.content_sha256, "corpus content hash").is_err()
        {
            return Err(manifest_error(
                "corpus manifest schema, generator, or content hash is invalid",
            ));
        }
        match &self.source {
            CorpusSourceManifest::Synthetic { spec } => {
                let expected = SyntheticCorpus::new(spec.clone())?.manifest()?;
                if self != &expected {
                    return Err(manifest_error(
                        "synthetic manifest does not match its replay recipe",
                    ));
                }
            }
            CorpusSourceManifest::SharedFixtures { view, .. } => {
                let expected = SharedFixtureSuite::load()?.manifest(*view)?;
                if self != &expected {
                    return Err(manifest_error(
                        "shared-fixture manifest does not match its embedded sources",
                    ));
                }
            }
            CorpusSourceManifest::Repository {
                repository_id,
                files,
            } => {
                let tracked_path_count = files
                    .len()
                    .checked_add(self.skipped_repository_entries.len());
                if validate_repository_id(repository_id).is_err()
                    || files.len() != usize::try_from(self.document_count).unwrap_or(usize::MAX)
                    || tracked_path_count.is_none_or(|count| {
                        count > usize::try_from(MAX_CORPUS_DOCUMENT_COUNT).unwrap_or(usize::MAX)
                    })
                {
                    return Err(manifest_error(
                        "repository manifest identity or document count is invalid",
                    ));
                }
                let mut previous = None::<&str>;
                let mut included = BTreeSet::new();
                let mut byte_count = 0_u64;
                let mut manifest_text_bytes = repository_id.len();
                for file in files {
                    let normalized = normalize_relative_path(Path::new(&file.path))?;
                    manifest_text_bytes = manifest_text_bytes
                        .checked_add(file.path.len())
                        .ok_or_else(|| {
                            manifest_error("repository manifest text byte count overflow")
                        })?;
                    if normalized != file.path
                        || previous.is_some_and(|path| path >= file.path.as_str())
                        || file.bytes > u64::from(MAX_DOCUMENT_BYTES)
                        || validate_sha256(&file.sha256, "repository file hash").is_err()
                    {
                        return Err(manifest_error(
                            "repository file digests must be safe, sorted, unique, and hashed",
                        ));
                    }
                    byte_count = byte_count
                        .checked_add(file.bytes)
                        .ok_or_else(|| manifest_error("repository content byte count overflow"))?;
                    previous = Some(&file.path);
                    included.insert(file.path.as_str());
                }
                let mut previous_skip = None::<&str>;
                for skipped in &self.skipped_repository_entries {
                    let normalized = normalize_relative_path(Path::new(&skipped.path))?;
                    manifest_text_bytes = manifest_text_bytes
                        .checked_add(skipped.path.len())
                        .ok_or_else(|| {
                            manifest_error("repository manifest text byte count overflow")
                        })?;
                    if normalized != skipped.path
                        || previous_skip.is_some_and(|path| path >= skipped.path.as_str())
                        || included.contains(skipped.path.as_str())
                    {
                        return Err(manifest_error(
                            "repository exclusions must be safe, sorted, unique, and disjoint",
                        ));
                    }
                    previous_skip = Some(&skipped.path);
                }
                if byte_count != self.total_content_bytes
                    || manifest_text_bytes > MAX_REPOSITORY_MANIFEST_TEXT_BYTES
                {
                    return Err(manifest_error(
                        "repository file bytes or manifest text exceed their bounded contract",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Verify arbitrary replay documents against this manifest's content pins.
    ///
    /// # Errors
    ///
    /// Returns an error if counts, bytes, content digest, or serialization differ.
    pub fn verify_documents<I, D>(&self, documents: I) -> Result<(), GauntletError>
    where
        I: IntoIterator<Item = D>,
        D: Borrow<GeneratedDocument>,
    {
        self.validate_contract()?;
        let mut verifier = self.replay_verifier();
        for document in documents {
            let _ = verifier.observe(document.borrow())?;
        }
        verifier.finish(self)
    }
}

/// Selects the original relevance corpus or the complete shared corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedCorpusView {
    /// First 100 documents: five 20-document relevance clusters.
    Core100,
    /// Complete committed 120-document corpus including adversarial additions.
    Full120,
}

/// One committed relevance-query row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedRelevanceQuery {
    /// Raw query.
    pub query: String,
    /// Coarse semantic class from the shared fixture.
    pub query_class: String,
    /// Ordered ground-truth document IDs.
    pub relevant_ids: Vec<String>,
}

/// One committed edge-case row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedEdgeCase {
    /// Stable edge-case ID.
    pub id: String,
    /// Exact input, including nulls or whitespace.
    pub text: String,
    /// Normative expected behavior description.
    pub expected_behavior: String,
}

/// Parsed embedded shared fixtures with explicit 100/120 corpus views.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedFixtureSuite {
    documents: Vec<GeneratedDocument>,
    relevance_queries: Vec<SharedRelevanceQuery>,
    edge_cases: Vec<SharedEdgeCase>,
    harvested_contract_queries: Vec<HarvestedContractQuery>,
}

impl SharedFixtureSuite {
    /// Parse and validate the committed fixture suite embedded in this binary.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed fixtures or count/cluster drift.
    pub fn load() -> Result<Self, GauntletError> {
        let corpus: SharedCorpusFile = serde_json::from_str(SHARED_CORPUS_JSON)?;
        let relevance_queries: Vec<SharedRelevanceQuery> =
            serde_json::from_str(SHARED_QUERIES_JSON)?;
        let edge_file: SharedEdgeFile = serde_json::from_str(SHARED_EDGE_CASES_JSON)?;
        let language_contract: LanguageContractQueries =
            serde_json::from_str(LANGUAGE_CONTRACT_JSON)?;
        if corpus.documents.len() != FULL_SHARED_DOCUMENT_COUNT {
            return Err(generator_error(format!(
                "shared corpus must contain {FULL_SHARED_DOCUMENT_COUNT} documents, found {}",
                corpus.documents.len()
            )));
        }
        if edge_file.cases.len() != 21 {
            return Err(generator_error(format!(
                "shared edge fixture must contain 21 cases, found {}",
                edge_file.cases.len()
            )));
        }
        let expected_core_types = ["cooking", "mixed", "ml", "rust", "sysadmin"];
        let core_types = corpus.documents[..CORE_RELEVANCE_DOCUMENT_COUNT]
            .iter()
            .fold(BTreeMap::<&str, usize>::new(), |mut counts, document| {
                *counts.entry(document.doc_type.as_str()).or_default() += 1;
                counts
            });
        if expected_core_types
            .iter()
            .any(|kind| core_types.get(kind).copied() != Some(20))
        {
            return Err(generator_error(
                "shared core relevance view must retain five 20-document clusters",
            ));
        }
        let documents = corpus
            .documents
            .into_iter()
            .enumerate()
            .map(|(index, document)| document.into_generated(index))
            .collect();
        Ok(Self {
            documents,
            relevance_queries,
            edge_cases: edge_file.cases,
            harvested_contract_queries: language_contract.harvested_queries,
        })
    }

    /// Documents in the selected committed view.
    #[must_use]
    pub fn documents(&self, view: SharedCorpusView) -> &[GeneratedDocument] {
        match view {
            SharedCorpusView::Core100 => &self.documents[..CORE_RELEVANCE_DOCUMENT_COUNT],
            SharedCorpusView::Full120 => &self.documents,
        }
    }

    /// All 26 committed query fixtures, including the frozen known-miss control.
    #[must_use]
    pub fn relevance_queries(&self) -> &[SharedRelevanceQuery] {
        &self.relevance_queries
    }

    /// All 21 committed edge cases.
    #[must_use]
    pub fn edge_cases(&self) -> &[SharedEdgeCase] {
        &self.edge_cases
    }

    /// The seven broad query-surface anchors in the language contract.
    #[must_use]
    pub fn harvested_contract_queries(&self) -> &[HarvestedContractQuery] {
        &self.harvested_contract_queries
    }

    /// Content-address the selected committed corpus view and exact source files.
    ///
    /// # Errors
    ///
    /// Returns an error if canonical serialization fails.
    pub fn manifest(&self, view: SharedCorpusView) -> Result<CorpusManifest, GauntletError> {
        CorpusManifest::from_documents(
            CorpusSourceManifest::SharedFixtures {
                view,
                sources: shared_source_digests(),
            },
            self.documents(view),
            Vec::new(),
        )
    }
}

/// Default parser versus the CASS grammar adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuerySyntax {
    /// Native/default query syntax.
    Default,
    /// CASS OR-over-AND grammar and structured filters.
    Cass,
}

/// Required wildcard pattern families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobPatternClass {
    /// No wildcard.
    Exact,
    /// Literal prefix followed by `*`.
    Prefix,
    /// `*` followed by a literal suffix.
    Suffix,
    /// Literal surrounded by `*`.
    Substring,
    /// Multiple `*` or `?` operators.
    Complex,
}

/// Required range shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RangeClass {
    /// Inclusive lower and upper bounds.
    Inclusive,
    /// Lower bound only.
    From,
    /// Upper bound only.
    To,
}

/// Required structured-filter families in the CASS campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredFilterClass {
    /// Exact agent-name set.
    Agent,
    /// Exact workspace-name set.
    Workspace,
    /// Local-origin restriction.
    Local,
    /// Remote-origin restriction.
    Remote,
    /// Exact durable source identifier.
    SourceId,
    /// Multiple structured predicates combined as required clauses.
    Combined,
}

/// Semantic/syntactic purpose of a generated query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GeneratedQueryKind {
    /// One term.
    Term,
    /// Multiple unquoted terms.
    MultiTerm,
    /// Quoted phrase.
    Phrase,
    /// Boolean expression.
    Boolean,
    /// Wildcard query.
    Glob {
        /// Exact wildcard family.
        pattern_class: GlobPatternClass,
    },
    /// Structured numeric range.
    Range {
        /// Exact bound shape.
        range_class: RangeClass,
    },
    /// One or more structured CASS predicates.
    StructuredFilter {
        /// Exact structured-filter family.
        filter_class: StructuredFilterClass,
    },
    /// Explicit pagination probe.
    Paginated,
    /// Explicit count/no-count probe.
    Counted,
    /// One row harvested from a committed relevance fixture.
    Harvested {
        /// Coarse semantic class.
        semantic_class: String,
    },
}

/// Engine-neutral source restriction lowered by both CASS adapters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GeneratedSourceFilter {
    /// Do not restrict the durable source identity.
    #[default]
    All,
    /// Match documents indexed from the local machine.
    Local,
    /// Match documents indexed through SSH.
    Remote,
    /// Match one exact durable source identifier.
    SourceId {
        /// Stable source identifier.
        source_id: String,
    },
}

/// CASS structured filters kept separate from raw query syntax.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedQueryFilters {
    /// Exact agent names, combined with OR and required by the root query.
    #[serde(default)]
    pub agents: Vec<String>,
    /// Exact workspace names, combined with OR and required by the root query.
    #[serde(default)]
    pub workspaces: Vec<String>,
    /// Inclusive creation-time lower bound.
    pub created_from_ms: Option<i64>,
    /// Inclusive creation-time upper bound.
    pub created_to_ms: Option<i64>,
    /// Durable source restriction.
    #[serde(default)]
    pub source_filter: GeneratedSourceFilter,
}

impl GeneratedQueryFilters {
    pub(crate) fn is_empty(&self) -> bool {
        self.agents.is_empty()
            && self.workspaces.is_empty()
            && self.created_from_ms.is_none()
            && self.created_to_ms.is_none()
            && self.source_filter == GeneratedSourceFilter::All
    }

    fn text_bytes(&self) -> Option<usize> {
        self.agents
            .iter()
            .chain(&self.workspaces)
            .try_fold(0_usize, |bytes, value| bytes.checked_add(value.len()))
            .and_then(|bytes| match &self.source_filter {
                GeneratedSourceFilter::SourceId { source_id } => bytes.checked_add(source_id.len()),
                GeneratedSourceFilter::All
                | GeneratedSourceFilter::Local
                | GeneratedSourceFilter::Remote => Some(bytes),
            })
    }

    fn validate(&self) -> bool {
        let bounded_unique_values = |values: &[String]| {
            values.len() <= MAX_STRUCTURED_FILTER_VALUES
                && values
                    .iter()
                    .all(|value| is_canonical_text(value, MAX_STRUCTURED_FILTER_VALUE_BYTES))
                && values.iter().collect::<BTreeSet<_>>().len() == values.len()
        };
        let range_is_ordered = self
            .created_from_ms
            .zip(self.created_to_ms)
            .is_none_or(|(from, to)| from <= to);
        let source_is_bounded = match &self.source_filter {
            GeneratedSourceFilter::SourceId { source_id } => {
                is_canonical_text(source_id, MAX_STRUCTURED_FILTER_VALUE_BYTES)
            }
            GeneratedSourceFilter::All
            | GeneratedSourceFilter::Local
            | GeneratedSourceFilter::Remote => true,
        };
        bounded_unique_values(&self.agents)
            && bounded_unique_values(&self.workspaces)
            && range_is_ordered
            && source_is_bounded
    }
}

/// Rich engine-neutral query case; E6.2 adapters lower this without data loss.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedQueryCase {
    /// Stable case ID.
    pub id: String,
    /// Parser/schema flavor.
    pub syntax: QuerySyntax,
    /// Exact coverage class.
    pub query_kind: GeneratedQueryKind,
    /// Raw source string.
    pub query: String,
    /// Result limit.
    pub limit: u64,
    /// Pagination offset.
    pub offset: u64,
    /// Whether exact/thresholded count evidence is requested.
    pub count_requested: bool,
    /// Structured range filters.
    pub filters: GeneratedQueryFilters,
    /// Accepted divergence identifier, when applicable.
    pub expected_divergence: Option<String>,
    /// Fixture provenance.
    pub source: String,
}

impl GeneratedQueryCase {
    /// Lower the common comparator envelope for every generated query.
    ///
    /// Syntax and structured filters remain in this rich case and are passed
    /// separately to campaign adapters. The differential envelope carries only
    /// fields shared by both adapters and the comparator.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed corpus manifest hash.
    #[cfg(test)]
    fn to_differential_case(
        &self,
        generator_seed: u64,
        corpus_manifest_hash: &str,
    ) -> Result<DifferentialCase, GauntletError> {
        validate_sha256(corpus_manifest_hash, "corpus manifest hash")?;
        let mut case = DifferentialCase::new(&self.id, &self.query, self.limit);
        case.offset = self.offset;
        case.count_requested = self.count_requested;
        case.metadata = DifferentialCaseMetadata {
            generator_id: Some(GENERATOR_ID.to_owned()),
            generator_seed: Some(generator_seed),
            corpus_hash: Some(corpus_manifest_hash.to_owned()),
        };
        Ok(case)
    }
}

/// Versioned query-generation recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryGeneratorSpec {
    /// Independent query stream seed.
    pub seed: u64,
    /// Default result limit for non-pagination probes.
    pub default_limit: u64,
    /// Include all 26 committed query fixtures and seven E0.1 contract anchors.
    pub include_shared_relevance_queries: bool,
}

/// Provenance mode for a query suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuerySuiteSource {
    /// Cases must exactly replay from the embedded generator spec and fixtures.
    Generated,
    /// Cases were explicitly supplied and are content-addressed, not claimed as generator output.
    ExplicitCases,
}

/// Canonical query-suite identity bound to one corpus manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryManifest {
    /// Generator/manifest schema.
    pub schema_version: u32,
    /// Stable generator implementation identity.
    pub generator_id: String,
    /// Query recipe.
    pub spec: QueryGeneratorSpec,
    /// Whether the cases replay from `spec` or were explicitly supplied.
    pub source: QuerySuiteSource,
    /// Hash binding source mode, spec, corpus, count, and ordered content digest.
    pub source_identity_sha256: String,
    /// Manifest hash of the corpus queried.
    pub corpus_manifest_hash: String,
    /// Number of cases.
    pub query_count: u64,
    /// SHA-256 over length-framed canonical query JSON.
    pub content_sha256: String,
}

impl QueryManifest {
    fn validate_contract(&self, cases: &[GeneratedQueryCase]) -> Result<(), GauntletError> {
        let expected_source_identity = query_source_identity(
            self.source,
            &self.spec,
            &self.corpus_manifest_hash,
            self.query_count,
            &self.content_sha256,
        )?;
        if self.schema_version != QUERY_MANIFEST_SCHEMA_VERSION
            || self.generator_id != GENERATOR_ID
            || validate_sha256(&self.content_sha256, "query content hash").is_err()
            || self.source_identity_sha256 != expected_source_identity
            || validate_query_inputs(&self.spec, &self.corpus_manifest_hash, cases).is_err()
        {
            return Err(manifest_error(
                "query manifest schema, generator, recipe, or cases are invalid",
            ));
        }
        if self.source == QuerySuiteSource::Generated {
            let shared = SharedFixtureSuite::load()?;
            let expected = generated_query_cases(&self.spec, &shared)?;
            if expected != cases {
                return Err(manifest_error(
                    "generated query suite does not match its replay recipe",
                ));
            }
        }
        Ok(())
    }

    /// Domain-separated manifest hash.
    ///
    /// # Errors
    ///
    /// Returns an error if canonical serialization fails.
    pub fn manifest_hash(&self) -> Result<String, GauntletError> {
        let mut hasher = Sha256::new();
        hasher.update(QUERY_MANIFEST_HASH_DOMAIN);
        hasher.update(serde_json::to_vec(self)?);
        Ok(lower_hex(&hasher.finalize()))
    }

    /// Verify ordered query content and count.
    ///
    /// # Errors
    ///
    /// Returns an error when replayed queries differ.
    pub fn verify(&self, cases: &[GeneratedQueryCase]) -> Result<(), GauntletError> {
        self.validate_contract(cases)?;
        let (query_count, content_sha256) = hash_queries(cases)?;
        if query_count != self.query_count || content_sha256 != self.content_sha256 {
            return Err(GauntletError::ManifestMismatch {
                reason: "replayed query suite does not match query content pins".to_owned(),
            });
        }
        Ok(())
    }
}

/// Generated query suite plus replay manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedQuerySuite {
    /// Ordered generated and harvested cases.
    pub cases: Vec<GeneratedQueryCase>,
    /// Content-addressed replay manifest.
    pub manifest: QueryManifest,
}

impl GeneratedQuerySuite {
    /// Build a replay-verifiable suite from explicit rich query cases.
    ///
    /// This is the E6.2 seam for attaching reviewed divergence IDs or selecting
    /// an independently harvested query corpus without discarding syntax,
    /// filters, pagination, or provenance.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid corpus hash, an empty/duplicate case ID,
    /// an out-of-budget limit/offset, or canonical serialization failure.
    pub fn from_cases(
        spec: QueryGeneratorSpec,
        corpus_manifest_hash: impl Into<String>,
        cases: Vec<GeneratedQueryCase>,
    ) -> Result<Self, GauntletError> {
        let corpus_manifest_hash = corpus_manifest_hash.into();
        validate_query_inputs(&spec, &corpus_manifest_hash, &cases)?;
        let (query_count, content_sha256) = hash_queries(&cases)?;
        let source = QuerySuiteSource::ExplicitCases;
        let source_identity_sha256 = query_source_identity(
            source,
            &spec,
            &corpus_manifest_hash,
            query_count,
            &content_sha256,
        )?;
        let manifest = QueryManifest {
            schema_version: QUERY_MANIFEST_SCHEMA_VERSION,
            generator_id: GENERATOR_ID.to_owned(),
            spec,
            source,
            source_identity_sha256,
            corpus_manifest_hash,
            query_count,
            content_sha256,
        };
        Ok(Self { cases, manifest })
    }

    /// Generate the complete E6.1 query matrix.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid corpus hash, zero limit, malformed shared
    /// fixtures, or canonical serialization failure.
    pub fn generate(
        spec: QueryGeneratorSpec,
        corpus_manifest_hash: impl Into<String>,
        shared: &SharedFixtureSuite,
    ) -> Result<Self, GauntletError> {
        if spec.default_limit == 0 || spec.default_limit > MAX_GENERATED_QUERY_LIMIT {
            return Err(generator_error(format!(
                "default query limit must be in 1..={MAX_GENERATED_QUERY_LIMIT}"
            )));
        }
        let corpus_manifest_hash = corpus_manifest_hash.into();
        validate_sha256(&corpus_manifest_hash, "corpus manifest hash")?;
        let cases = generated_query_cases(&spec, shared)?;
        let (query_count, content_sha256) = hash_queries(&cases)?;
        let source = QuerySuiteSource::Generated;
        let source_identity_sha256 = query_source_identity(
            source,
            &spec,
            &corpus_manifest_hash,
            query_count,
            &content_sha256,
        )?;
        let manifest = QueryManifest {
            schema_version: QUERY_MANIFEST_SCHEMA_VERSION,
            generator_id: GENERATOR_ID.to_owned(),
            spec,
            source,
            source_identity_sha256,
            corpus_manifest_hash,
            query_count,
            content_sha256,
        };
        Ok(Self { cases, manifest })
    }
}

fn generated_query_cases(
    spec: &QueryGeneratorSpec,
    shared: &SharedFixtureSuite,
) -> Result<Vec<GeneratedQueryCase>, GauntletError> {
    let mut cases = constructed_query_matrix(spec.seed, spec.default_limit);
    if spec.include_shared_relevance_queries {
        cases.extend(
            shared
                .relevance_queries
                .iter()
                .enumerate()
                .map(|(index, query)| GeneratedQueryCase {
                    id: format!("harvested-{index:02}"),
                    syntax: QuerySyntax::Default,
                    query_kind: GeneratedQueryKind::Harvested {
                        semantic_class: query.query_class.clone(),
                    },
                    query: query.query.clone(),
                    limit: spec.default_limit,
                    offset: 0,
                    count_requested: true,
                    filters: GeneratedQueryFilters::default(),
                    expected_divergence: None,
                    source: "tests/fixtures/queries.json".to_owned(),
                }),
        );
        let contract_cases = shared
            .harvested_contract_queries
            .iter()
            .map(|query| contract_query_case(query, spec.default_limit))
            .collect::<Result<Vec<_>, _>>()?;
        cases.extend(contract_cases);
    }
    Ok(cases)
}

fn contract_query_case(
    query: &HarvestedContractQuery,
    limit: u64,
) -> Result<GeneratedQueryCase, GauntletError> {
    if query
        .filters
        .keys()
        .any(|key| !matches!(key.as_str(), "created_from" | "created_to"))
    {
        return Err(generator_error(format!(
            "contract query {} contains an unsupported filter key",
            query.id
        )));
    }
    let (syntax, query_kind) = match query.query_class.as_str() {
        "phrase" => (QuerySyntax::Default, GeneratedQueryKind::Phrase),
        "boolean" => (QuerySyntax::Cass, GeneratedQueryKind::Boolean),
        "glob" => (
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Prefix,
            },
        ),
        "range" => (
            QuerySyntax::Cass,
            GeneratedQueryKind::Range {
                range_class: RangeClass::Inclusive,
            },
        ),
        "identifier" | "short_keyword" | "natural_language" => (
            QuerySyntax::Default,
            GeneratedQueryKind::Harvested {
                semantic_class: query.query_class.clone(),
            },
        ),
        unsupported => {
            return Err(generator_error(format!(
                "contract query {} has unsupported class {unsupported:?}",
                query.id
            )));
        }
    };
    let source = format!(
        "tests/fixtures/quill_language_contract.json#kind={}#source={}#fact={}",
        query.source_kind,
        query.source.as_deref().unwrap_or(""),
        query.source_fact.as_deref().unwrap_or("")
    );
    Ok(GeneratedQueryCase {
        id: format!("contract-{}", query.id),
        syntax,
        query_kind,
        query: query.query.clone(),
        limit,
        offset: 0,
        count_requested: true,
        filters: GeneratedQueryFilters {
            created_from_ms: query.filters.get("created_from").copied(),
            created_to_ms: query.filters.get("created_to").copied(),
            ..GeneratedQueryFilters::default()
        },
        expected_divergence: None,
        source,
    })
}

/// Broad query anchor parsed from the committed language contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarvestedContractQuery {
    /// Stable fixture ID.
    pub id: String,
    /// Semantic/syntactic class label.
    pub query_class: String,
    /// Raw query.
    pub query: String,
    /// Source classification.
    pub source_kind: String,
    /// Source path, when harvested.
    #[serde(default)]
    pub source: Option<String>,
    /// Construction provenance, when synthesized.
    #[serde(default)]
    pub source_fact: Option<String>,
    /// Optional structured filters retained as generic JSON for fidelity.
    #[serde(default)]
    pub filters: BTreeMap<String, i64>,
}

#[derive(Debug, Deserialize)]
struct SharedCorpusFile {
    documents: Vec<SharedDocument>,
}

#[derive(Debug, Deserialize)]
struct SharedDocument {
    doc_id: String,
    title: String,
    content: String,
    created_at: String,
    doc_type: String,
    #[serde(default)]
    metadata: BTreeMap<String, serde_json::Value>,
}

impl SharedDocument {
    fn into_generated(self, index: usize) -> GeneratedDocument {
        let mut metadata = self
            .metadata
            .into_iter()
            .map(|(key, value)| {
                let value = value
                    .as_str()
                    .map_or_else(|| value.to_string(), ToOwned::to_owned);
                (key, value)
            })
            .collect::<BTreeMap<_, _>>();
        metadata.insert("created_at".to_owned(), self.created_at);
        metadata.insert("doc_type".to_owned(), self.doc_type);
        let unicode_lane = classify_unicode_lane(&self.content);
        GeneratedDocument {
            id: self.doc_id,
            title: Some(self.title),
            content: self.content,
            created_at_ms: i64::try_from(index).unwrap_or(i64::MAX),
            cass: None,
            metadata,
            pathology: None,
            unicode_lane,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SharedEdgeFile {
    cases: Vec<SharedEdgeCase>,
}

#[derive(Debug, Deserialize)]
struct LanguageContractQueries {
    harvested_queries: Vec<HarvestedContractQuery>,
}

fn constructed_query_matrix(seed: u64, default_limit: u64) -> Vec<GeneratedQueryCase> {
    let mut rng = DeterministicRng::for_stream(seed, 0x0051_5545_5249_4553, 0);
    let terms = ["ownership", "search", "cache", "token", "config"];
    let selected_index = usize::try_from(rng.bounded(u64_from_usize(terms.len()))).unwrap_or(0);
    let selected = terms[selected_index];
    let base = |id: &str, syntax: QuerySyntax, query_kind: GeneratedQueryKind, query: String| {
        GeneratedQueryCase {
            id: id.to_owned(),
            syntax,
            query_kind,
            query,
            limit: default_limit,
            offset: 0,
            count_requested: true,
            filters: GeneratedQueryFilters::default(),
            expected_divergence: None,
            source: "quill-e6.1-constructed".to_owned(),
        }
    };
    let mut cases = vec![
        base(
            "term",
            QuerySyntax::Default,
            GeneratedQueryKind::Term,
            selected.to_owned(),
        ),
        base(
            "multi-term",
            QuerySyntax::Default,
            GeneratedQueryKind::MultiTerm,
            "rust ownership borrowing".to_owned(),
        ),
        base(
            "phrase",
            QuerySyntax::Default,
            GeneratedQueryKind::Phrase,
            "\"error handling\"".to_owned(),
        ),
        base(
            "same-position-phrase",
            QuerySyntax::Default,
            GeneratedQueryKind::Phrase,
            "\"high performance tail\"".to_owned(),
        ),
        base(
            "boolean-default",
            QuerySyntax::Default,
            GeneratedQueryKind::Boolean,
            "auth AND token OR cache".to_owned(),
        ),
        base(
            "boolean-cass",
            QuerySyntax::Cass,
            GeneratedQueryKind::Boolean,
            "auth OR token cache".to_owned(),
        ),
        base(
            "boolean-cass-and",
            QuerySyntax::Cass,
            GeneratedQueryKind::Boolean,
            "auth AND token".to_owned(),
        ),
        base(
            "boolean-cass-or",
            QuerySyntax::Cass,
            GeneratedQueryKind::Boolean,
            "auth OR token".to_owned(),
        ),
        base(
            "boolean-cass-not",
            QuerySyntax::Cass,
            GeneratedQueryKind::Boolean,
            "auth NOT stale".to_owned(),
        ),
        base(
            "glob-exact",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Exact,
            },
            "config".to_owned(),
        ),
        base(
            "glob-prefix",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Prefix,
            },
            "config*".to_owned(),
        ),
        base(
            "glob-suffix",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Suffix,
            },
            "*config".to_owned(),
        ),
        base(
            "glob-substring",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Substring,
            },
            "*config*".to_owned(),
        ),
        base(
            "glob-complex",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Complex,
            },
            "con*fig*".to_owned(),
        ),
    ];
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            created_from_ms: Some(1_700_000_000_000),
            created_to_ms: Some(1_700_000_000_999),
            ..GeneratedQueryFilters::default()
        },
        ..base(
            "range-inclusive",
            QuerySyntax::Cass,
            GeneratedQueryKind::Range {
                range_class: RangeClass::Inclusive,
            },
            "cache".to_owned(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            created_from_ms: Some(1_700_000_000_000),
            created_to_ms: None,
            ..GeneratedQueryFilters::default()
        },
        ..base(
            "range-from",
            QuerySyntax::Cass,
            GeneratedQueryKind::Range {
                range_class: RangeClass::From,
            },
            "cache".to_owned(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            created_from_ms: None,
            created_to_ms: Some(1_700_000_000_999),
            ..GeneratedQueryFilters::default()
        },
        ..base(
            "range-to",
            QuerySyntax::Cass,
            GeneratedQueryKind::Range {
                range_class: RangeClass::To,
            },
            "cache".to_owned(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            agents: vec!["agent-0".to_owned(), "agent-3".to_owned()],
            ..GeneratedQueryFilters::default()
        },
        ..base(
            "filter-agent",
            QuerySyntax::Cass,
            GeneratedQueryKind::StructuredFilter {
                filter_class: StructuredFilterClass::Agent,
            },
            String::new(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            workspaces: vec!["workspace-1".to_owned(), "workspace-4".to_owned()],
            ..GeneratedQueryFilters::default()
        },
        ..base(
            "filter-workspace",
            QuerySyntax::Cass,
            GeneratedQueryKind::StructuredFilter {
                filter_class: StructuredFilterClass::Workspace,
            },
            String::new(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            source_filter: GeneratedSourceFilter::Local,
            ..GeneratedQueryFilters::default()
        },
        ..base(
            "filter-local",
            QuerySyntax::Cass,
            GeneratedQueryKind::StructuredFilter {
                filter_class: StructuredFilterClass::Local,
            },
            String::new(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            source_filter: GeneratedSourceFilter::Remote,
            ..GeneratedQueryFilters::default()
        },
        ..base(
            "filter-remote",
            QuerySyntax::Cass,
            GeneratedQueryKind::StructuredFilter {
                filter_class: StructuredFilterClass::Remote,
            },
            String::new(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            source_filter: GeneratedSourceFilter::SourceId {
                source_id: "source-2".to_owned(),
            },
            ..GeneratedQueryFilters::default()
        },
        ..base(
            "filter-source-id",
            QuerySyntax::Cass,
            GeneratedQueryKind::StructuredFilter {
                filter_class: StructuredFilterClass::SourceId,
            },
            String::new(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            agents: vec!["agent-0".to_owned(), "agent-1".to_owned()],
            workspaces: vec!["workspace-0".to_owned()],
            created_from_ms: Some(1_700_000_000_000),
            created_to_ms: Some(1_700_000_000_128),
            source_filter: GeneratedSourceFilter::SourceId {
                source_id: "source-0".to_owned(),
            },
        },
        ..base(
            "filter-combined",
            QuerySyntax::Cass,
            GeneratedQueryKind::StructuredFilter {
                filter_class: StructuredFilterClass::Combined,
            },
            String::new(),
        )
    });
    cases.push(GeneratedQueryCase {
        offset: 17,
        limit: 7,
        ..base(
            "paginated",
            QuerySyntax::Default,
            GeneratedQueryKind::Paginated,
            "search".to_owned(),
        )
    });
    cases.push(GeneratedQueryCase {
        count_requested: false,
        ..base(
            "uncounted",
            QuerySyntax::Default,
            GeneratedQueryKind::Counted,
            "search".to_owned(),
        )
    });
    cases.push(base(
        "counted",
        QuerySyntax::Default,
        GeneratedQueryKind::Counted,
        "search".to_owned(),
    ));
    cases
}

fn shared_source_digests() -> Vec<SourceFileDigest> {
    [
        ("tests/fixtures/corpus.json", SHARED_CORPUS_JSON),
        ("tests/fixtures/queries.json", SHARED_QUERIES_JSON),
        ("tests/fixtures/edge_cases.json", SHARED_EDGE_CASES_JSON),
        (
            "tests/fixtures/quill_language_contract.json",
            LANGUAGE_CONTRACT_JSON,
        ),
    ]
    .into_iter()
    .map(|(path, contents)| SourceFileDigest {
        path: path.to_owned(),
        sha256: sha256_lf_normalized(contents),
    })
    .collect()
}

fn hash_documents<I, D>(documents: I) -> Result<(u64, u64, String), GauntletError>
where
    I: IntoIterator<Item = D>,
    D: Borrow<GeneratedDocument>,
{
    let mut hasher = Sha256::new();
    hasher.update(CORPUS_HASH_DOMAIN);
    let mut count = 0_u64;
    let mut content_bytes = 0_u64;
    for document in documents {
        let document = document.borrow();
        let bytes = serde_json::to_vec(document)?;
        hasher.update(u64_from_usize(bytes.len()).to_be_bytes());
        hasher.update(bytes);
        count = count
            .checked_add(1)
            .ok_or_else(|| generator_error("document count overflow while hashing corpus"))?;
        content_bytes = content_bytes
            .checked_add(u64::try_from(document.content.len()).unwrap_or(u64::MAX))
            .ok_or_else(|| generator_error("content byte count overflow while hashing corpus"))?;
    }
    Ok((count, content_bytes, lower_hex(&hasher.finalize())))
}

fn hash_queries(cases: &[GeneratedQueryCase]) -> Result<(u64, String), GauntletError> {
    let mut hasher = Sha256::new();
    hasher.update(QUERY_HASH_DOMAIN);
    for case in cases {
        let bytes = serde_json::to_vec(case)?;
        hasher.update(u64_from_usize(bytes.len()).to_be_bytes());
        hasher.update(bytes);
    }
    Ok((u64_from_usize(cases.len()), lower_hex(&hasher.finalize())))
}

fn query_source_identity(
    source: QuerySuiteSource,
    spec: &QueryGeneratorSpec,
    corpus_manifest_hash: &str,
    query_count: u64,
    content_sha256: &str,
) -> Result<String, GauntletError> {
    let mut hasher = Sha256::new();
    hasher.update(QUERY_SOURCE_IDENTITY_DOMAIN);
    hasher.update(serde_json::to_vec(&(
        source,
        spec,
        corpus_manifest_hash,
        query_count,
        content_sha256,
    ))?);
    Ok(lower_hex(&hasher.finalize()))
}

fn validate_query_inputs(
    spec: &QueryGeneratorSpec,
    corpus_manifest_hash: &str,
    cases: &[GeneratedQueryCase],
) -> Result<(), GauntletError> {
    if spec.default_limit == 0 || spec.default_limit > MAX_GENERATED_QUERY_LIMIT {
        return Err(generator_error(format!(
            "default query limit must be in 1..={MAX_GENERATED_QUERY_LIMIT}"
        )));
    }
    validate_sha256(corpus_manifest_hash, "corpus manifest hash")?;
    if cases.is_empty() {
        return Err(generator_error(
            "query suite must contain at least one case",
        ));
    }
    if cases.len() > MAX_QUERY_CASES {
        return Err(generator_error(format!(
            "query suite may contain at most {MAX_QUERY_CASES} cases"
        )));
    }
    let mut ids = BTreeSet::new();
    let mut aggregate_text_bytes = 0_usize;
    for case in cases {
        let end = case.offset.checked_add(case.limit);
        let case_text_bytes = case
            .id
            .len()
            .checked_add(case.query.len())
            .and_then(|bytes| bytes.checked_add(case.source.len()))
            .and_then(|bytes| {
                bytes.checked_add(case.expected_divergence.as_ref().map_or(0, String::len))
            })
            .and_then(|bytes| {
                bytes.checked_add(match &case.query_kind {
                    GeneratedQueryKind::Harvested { semantic_class } => semantic_class.len(),
                    _ => 0,
                })
            })
            .and_then(|bytes| {
                case.filters
                    .text_bytes()
                    .and_then(|filter_bytes| bytes.checked_add(filter_bytes))
            })
            .ok_or_else(|| generator_error("query case text byte count overflow"))?;
        aggregate_text_bytes = aggregate_text_bytes
            .checked_add(case_text_bytes)
            .ok_or_else(|| generator_error("query suite text byte count overflow"))?;
        if !is_canonical_query_id(&case.id)
            || case.query.len() > MAX_QUERY_TEXT_BYTES
            || !is_canonical_text(&case.source, MAX_QUERY_SOURCE_BYTES)
            || case
                .expected_divergence
                .as_ref()
                .is_some_and(|id| !is_divergence_id(id))
            || matches!(
                &case.query_kind,
                GeneratedQueryKind::Harvested { semantic_class }
                    if !is_canonical_query_id(semantic_class)
            )
            || !case.filters.validate()
            || (case.syntax == QuerySyntax::Default && !case.filters.is_empty())
            || matches!(
                &case.query_kind,
                GeneratedQueryKind::StructuredFilter { .. }
            ) && (case.syntax != QuerySyntax::Cass || case.filters.is_empty())
            || !ids.insert(case.id.as_str())
            || case.limit > MAX_GENERATED_QUERY_LIMIT
            || case.offset > MAX_GENERATED_QUERY_LIMIT
            || end.is_none_or(|value| value > MAX_GENERATED_QUERY_LIMIT)
            || aggregate_text_bytes > MAX_QUERY_SUITE_TEXT_BYTES
        {
            return Err(generator_error(
                "query cases require unique non-empty IDs/sources and bounded limit/offset",
            ));
        }
    }
    Ok(())
}

pub fn is_canonical_query_id(value: &str) -> bool {
    is_canonical_text(value, MAX_QUERY_ID_BYTES)
}

fn is_canonical_text(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn is_divergence_id(value: &str) -> bool {
    value.len() == 7
        && value.starts_with("DIV-")
        && value.as_bytes()[4..].iter().all(u8::is_ascii_digit)
}

fn build_zipf_cdf(vocabulary_size: u32, exponent: ZipfExponent) -> Result<Vec<u64>, GauntletError> {
    let mut cumulative = 0_u64;
    let mut cdf = Vec::with_capacity(usize_from_u32(vocabulary_size));
    for rank in 1..=vocabulary_size {
        let denominator_q8 = zipf_denominator_q8(rank, exponent);
        let weight = (1_u64 << 55) / denominator_q8.max(1);
        cumulative = cumulative
            .checked_add(weight.max(1))
            .ok_or_else(|| generator_error("Zipf cumulative weight overflow"))?;
        cdf.push(cumulative);
    }
    Ok(cdf)
}

fn zipf_denominator_q8(rank: u32, exponent: ZipfExponent) -> u64 {
    let rank = u128::from(rank);
    let value = match exponent {
        ZipfExponent::S08 => integer_nth_root((rank.pow(4)) << 40, 5),
        ZipfExponent::S11 => rank * integer_nth_root(rank << 80, 10),
        ZipfExponent::S14 => rank * integer_nth_root((rank.pow(2)) << 40, 5),
    };
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn integer_nth_root(value: u128, exponent: u32) -> u128 {
    let mut low = 0_u128;
    let mut high = 1_u128;
    while power_leq(high, exponent, value) && high <= value / 2 {
        high *= 2;
    }
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if power_leq(middle, exponent, value) {
            low = middle;
        } else {
            high = middle;
        }
    }
    if power_leq(high, exponent, value) {
        high
    } else {
        low
    }
}

fn power_leq(base: u128, exponent: u32, limit: u128) -> bool {
    let mut product = 1_u128;
    for _ in 0..exponent {
        let Some(next) = product.checked_mul(base) else {
            return false;
        };
        if next > limit {
            return false;
        }
        product = next;
    }
    true
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn for_stream(seed: u64, domain: u64, ordinal: u64) -> Self {
        let mut mixer = Self {
            state: seed ^ domain.rotate_left(17) ^ ordinal.wrapping_mul(0xd6e8_feb8_6659_fd93),
        };
        Self {
            state: mixer.next_u64(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn bounded(&mut self, upper: u64) -> u64 {
        if upper <= 1 {
            return 0;
        }
        u64::try_from((u128::from(self.next_u64()) * u128::from(upper)) >> 64).unwrap_or(u64::MAX)
    }
}

fn sampled_document_length(rng: &mut DeterministicRng, cap: usize) -> usize {
    let bucket = rng.bounded(1_000);
    let (minimum, span) = if bucket < 750 {
        (64_usize, 448_usize)
    } else if bucket < 950 {
        (512, 7_680)
    } else if bucket < 999 {
        (8_192, 57_344)
    } else {
        (65_536, cap.saturating_sub(65_536).saturating_add(1))
    };
    let available_minimum = minimum.min(cap);
    let available_span = span.min(cap.saturating_sub(available_minimum).saturating_add(1));
    let span = u64_from_usize(available_span.max(1));
    available_minimum + usize::try_from(rng.bounded(span)).unwrap_or(usize::MAX)
}

fn cass_profile_anchor(index: u64) -> Option<&'static str> {
    match index {
        10 => Some("auth token cache config"),
        11 => Some("auth stale configured"),
        12 => Some("token cache configuration"),
        13 => Some("auth cache reconfig"),
        14 => Some("token configurable"),
        15 => Some("unrelated baseline"),
        _ => None,
    }
}

fn maximum_frequency_content(cap: usize) -> String {
    let repeat = "freq ";
    let tail = "tail";
    let repetitions = MAX_FREQUENCY_REPETITIONS.min(cap.saturating_sub(tail.len()) / repeat.len());
    let mut content = String::with_capacity(repetitions * repeat.len() + tail.len());
    for _ in 0..repetitions {
        content.push_str(repeat);
    }
    content.push_str(tail);
    content
}

fn near_limit_code(cap: usize) -> String {
    let line = "fn generated_record() { index.push(term00001); }\n";
    let tail = "// needle tail";
    let mut content = String::with_capacity(cap);
    while content.len() + line.len() + tail.len() <= cap {
        content.push_str(line);
    }
    push_bounded(&mut content, tail, cap);
    content
}

fn push_ranked_term_bounded(target: &mut String, rank: u32, limit: usize) -> bool {
    const MAX_TERM_BYTES: usize = "term65536".len();
    let separator = usize::from(!target.is_empty());
    if target.len() + separator + MAX_TERM_BYTES > limit {
        return false;
    }
    if separator == 1 {
        target.push(' ');
    }
    write!(target, "term{rank:05}").is_ok()
}

fn push_bounded(target: &mut String, source: &str, limit: usize) {
    let remaining = limit.saturating_sub(target.len());
    let mut boundary = source.len().min(remaining);
    while boundary > 0 && !source.is_char_boundary(boundary) {
        boundary -= 1;
    }
    target.push_str(&source[..boundary]);
}

fn truncate_utf8(mut value: String, limit: usize) -> String {
    if value.len() <= limit {
        return value;
    }
    let mut boundary = limit;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value.truncate(boundary);
    value
}

fn classify_unicode_lane(content: &str) -> UnicodeLane {
    if content.is_ascii() {
        return UnicodeLane::Ascii;
    }
    let mut latin = false;
    let mut cjk = false;
    let mut extension_b = false;
    let mut other = false;
    for character in content.chars().filter(|character| !character.is_ascii()) {
        match u32::from(character) {
            0x0080..=0x024f | 0x0300..=0x036f => latin = true,
            0x3040..=0x30ff | 0x3400..=0x9fff | 0xac00..=0xd7af => cjk = true,
            0x20000..=0x2a6df => extension_b = true,
            _ => other = true,
        }
    }
    match (latin, cjk, extension_b, other) {
        (true, false, false, false) => UnicodeLane::Latin1,
        (false, true, false, false) => UnicodeLane::Cjk,
        (false, false, true, false) => UnicodeLane::CjkExtensionB,
        _ => UnicodeLane::Mixed,
    }
}

fn normalize_relative_path(path: &Path) -> Result<String, GauntletError> {
    let Some(raw_path) = path.to_str() else {
        return Err(generator_error("repository path must be valid UTF-8"));
    };
    if raw_path.is_empty()
        || raw_path.len() > MAX_REPOSITORY_PATH_BYTES
        || raw_path.contains('\\')
        || path.is_absolute()
    {
        return Err(generator_error(format!(
            "repository path must be nonempty, relative, UTF-8, and slash-separated: {}",
            path.display()
        )));
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let Some(part) = part.to_str() else {
                    return Err(generator_error("repository path must be valid UTF-8"));
                };
                if part.is_empty() {
                    return Err(generator_error(
                        "repository path contains an empty component",
                    ));
                }
                parts.push(part);
            }
            _ => {
                return Err(generator_error(format!(
                    "repository path contains an unsafe component: {}",
                    path.display()
                )));
            }
        }
    }
    Ok(parts.join("/"))
}

fn validate_repository_id(repository_id: &str) -> Result<(), GauntletError> {
    if !is_canonical_text(repository_id, MAX_REPOSITORY_ID_BYTES)
        || repository_id.contains(['/', '\\'])
    {
        return Err(generator_error(
            "repository_id must be a bounded canonical logical name without path separators",
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<(), GauntletError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(generator_error(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    lower_hex(&Sha256::digest(bytes))
}

fn sha256_lf_normalized(text: &str) -> String {
    sha256_hex(text.replace("\r\n", "\n").as_bytes())
}

fn lower_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn usize_from_u32(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn u64_from_usize(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn generator_error(reason: impl Into<String>) -> GauntletError {
    GauntletError::InvalidGenerator {
        reason: reason.into(),
    }
}

fn manifest_error(reason: impl Into<String>) -> GauntletError {
    GauntletError::ManifestMismatch {
        reason: reason.into(),
    }
}

pub fn validate_generated_case_metadata(case: &DifferentialCase) -> Result<(), GauntletError> {
    if case.metadata.generator_id.as_deref() != Some(GENERATOR_ID) {
        return Ok(());
    }
    if case.metadata.generator_seed.is_none() {
        return Err(generator_error(
            "generated artifact case must retain its generator seed",
        ));
    }
    let hash = case
        .metadata
        .corpus_hash
        .as_deref()
        .ok_or_else(|| generator_error("generated artifact case must retain a corpus hash"))?;
    validate_sha256(hash, "generated artifact corpus manifest hash")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    const GENERATOR_GOLDEN_JSON: &str = include_str!("../fixtures/generator-v2.json");

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct GeneratorGolden {
        schema_version: u32,
        corpus_spec: SyntheticCorpusSpec,
        corpus_manifest: CorpusManifest,
        corpus_manifest_hash: String,
        document_ids: Vec<String>,
        document_pathologies: Vec<Option<Pathology>>,
        document_unicode_lanes: Vec<UnicodeLane>,
        query_manifest: QueryManifest,
        query_manifest_hash: String,
        query_ids: Vec<String>,
        query_syntaxes: Vec<QuerySyntax>,
        query_kinds: Vec<GeneratedQueryKind>,
    }

    fn small_spec(seed: u64, document_count: u64) -> SyntheticCorpusSpec {
        SyntheticCorpusSpec {
            seed,
            document_count,
            vocabulary_size: 128,
            zipf_exponent: ZipfExponent::S11,
            max_document_bytes: 1_024,
        }
    }

    fn generate_small_golden() -> (
        GeneratorGolden,
        Vec<GeneratedDocument>,
        Vec<GeneratedQueryCase>,
    ) {
        let corpus_spec = SyntheticCorpusSpec {
            max_document_bytes: 256,
            ..small_spec(0x5eed, 12)
        };
        let corpus = SyntheticCorpus::new(corpus_spec.clone()).expect("golden corpus");
        let documents = corpus.iter().collect::<Vec<_>>();
        let corpus_manifest = corpus.manifest().expect("corpus manifest");
        let corpus_manifest_hash = corpus_manifest.manifest_hash().expect("corpus hash");
        let shared = SharedFixtureSuite::load().expect("shared fixtures");
        let query_suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 0x5eee,
                default_limit: 20,
                include_shared_relevance_queries: false,
            },
            &corpus_manifest_hash,
            &shared,
        )
        .expect("golden query suite");
        let query_manifest_hash = query_suite.manifest.manifest_hash().expect("query hash");
        let golden = GeneratorGolden {
            schema_version: GENERATOR_SCHEMA_VERSION,
            corpus_spec,
            corpus_manifest,
            corpus_manifest_hash,
            document_ids: documents
                .iter()
                .map(|document| document.id.clone())
                .collect(),
            document_pathologies: documents
                .iter()
                .map(|document| document.pathology)
                .collect(),
            document_unicode_lanes: documents
                .iter()
                .map(|document| document.unicode_lane)
                .collect(),
            query_manifest: query_suite.manifest,
            query_manifest_hash,
            query_ids: query_suite
                .cases
                .iter()
                .map(|case| case.id.clone())
                .collect(),
            query_syntaxes: query_suite.cases.iter().map(|case| case.syntax).collect(),
            query_kinds: query_suite
                .cases
                .iter()
                .map(|case| case.query_kind.clone())
                .collect(),
        };
        (golden, documents, query_suite.cases)
    }

    #[test]
    fn committed_small_generator_output_is_an_exact_replay_golden() {
        let expected: GeneratorGolden =
            serde_json::from_str(GENERATOR_GOLDEN_JSON).expect("valid committed golden");
        let (actual, documents, queries) = generate_small_golden();
        assert_eq!(actual, expected);
        actual
            .corpus_manifest
            .verify_documents(&documents)
            .expect("corpus replay");
        actual
            .query_manifest
            .verify(&queries)
            .expect("query replay");
        assert_eq!(
            actual.corpus_manifest.manifest_hash().unwrap(),
            actual.corpus_manifest_hash
        );
        assert_eq!(
            actual.query_manifest.manifest_hash().unwrap(),
            actual.query_manifest_hash
        );
    }

    #[test]
    fn splitmix_stream_is_pinned_and_domain_separated() {
        let mut rng = DeterministicRng::for_stream(42, 7, 11);
        let actual = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        assert_eq!(
            actual,
            [
                18_253_566_837_895_693_334,
                12_315_204_370_234_691_178,
                10_379_684_996_734_907_096,
                4_158_975_417_640_783_922,
            ]
        );

        let mut different_domain = DeterministicRng::for_stream(42, 8, 11);
        assert_ne!(actual[0], different_domain.next_u64());
        let mut bounded = DeterministicRng::for_stream(42, 7, 11);
        assert_eq!(bounded.bounded(0), 0);
        assert_eq!(bounded.bounded(1), 0);
        assert!(bounded.bounded(17) < 17);
    }

    #[test]
    fn fixed_point_zipf_skews_are_deterministic_and_increasingly_head_heavy() {
        let head_hits = ZipfExponent::ALL.map(|exponent| {
            let cdf = build_zipf_cdf(1_024, exponent).expect("valid CDF");
            let mut rng = DeterministicRng::for_stream(9, 10, 11);
            (0..20_000)
                .map(|_| {
                    let draw = rng.bounded(*cdf.last().expect("nonempty CDF"));
                    cdf.partition_point(|&value| value <= draw) + 1
                })
                .filter(|rank| *rank <= 16)
                .count()
        });
        assert!(head_hits[0] < head_hits[1]);
        assert!(head_hits[1] < head_hits[2]);
        let pinned_cdf = ZipfExponent::ALL.map(|exponent| {
            let cdf = build_zipf_cdf(1_024, exponent).expect("valid CDF");
            [cdf[0], cdf[1], cdf[15], cdf[1_023]]
        });
        assert_eq!(
            pinned_cdf,
            [
                [
                    140_737_488_355_328,
                    221_701_077_161_988,
                    608_528_386_439_328,
                    2_190_841_941_626_733,
                ],
                [
                    140_737_488_355_328,
                    206_483_468_316_941,
                    426_789_846_743_358,
                    786_862_503_390_202,
                ],
                [
                    140_737_488_355_328,
                    194_192_676_810_764,
                    322_651_937_770_287,
                    415_333_824_121_154,
                ],
            ]
        );
        assert!(
            build_zipf_cdf(0, ZipfExponent::S08)
                .expect("empty lookup is representable")
                .is_empty()
        );
    }

    #[test]
    fn synthetic_pathologies_cover_boundaries_and_never_exceed_byte_cap() {
        let corpus = SyntheticCorpus::new(SyntheticCorpusSpec {
            seed: 17,
            document_count: 32,
            vocabulary_size: 256,
            zipf_exponent: ZipfExponent::S14,
            max_document_bytes: MAX_DOCUMENT_BYTES,
        })
        .expect("valid corpus");
        let documents = (0..10)
            .map(|index| corpus.document_at(index).expect("document exists"))
            .collect::<Vec<_>>();
        assert!(documents.iter().all(|document| {
            document.content.len() <= usize_from_u32(MAX_DOCUMENT_BYTES)
                && document.content.is_char_boundary(document.content.len())
        }));
        assert!(documents[0].content.is_empty());
        assert!(documents[1].content.trim().is_empty());
        assert_eq!(documents[2].content, "singleton");
        assert_eq!(documents[3].content.len(), 256);
        assert!(documents[3].content.bytes().all(|byte| byte == b't'));
        assert!(documents[4].content.contains("bd-q3fy"));
        assert_eq!(
            documents[5]
                .content
                .split_whitespace()
                .filter(|term| *term == "freq")
                .count(),
            MAX_FREQUENCY_REPETITIONS
        );
        assert!(documents[5].content.ends_with("tail"));
        assert!(documents[6].content.contains("cafe\u{301}"));
        assert!(documents[7].content.contains('𠀀'));
        assert!(documents[7].content.contains('𪛟'));
        assert!(documents[8].content.contains("café"));
        assert!(documents[9].content.ends_with("// needle tail"));
    }

    #[test]
    fn synthetic_cass_profile_anchors_and_sources_keep_campaign_non_vacuous() {
        let corpus = SyntheticCorpus::new(SyntheticCorpusSpec {
            seed: 23,
            document_count: 16,
            vocabulary_size: 128,
            zipf_exponent: ZipfExponent::S11,
            max_document_bytes: 512,
        })
        .expect("valid CASS corpus");
        let documents = corpus.iter().collect::<Vec<_>>();
        for (index, expected) in [
            (10, "auth token cache config"),
            (11, "auth stale configured"),
            (12, "token cache configuration"),
            (13, "auth cache reconfig"),
            (14, "token configurable"),
            (15, "unrelated baseline"),
        ] {
            assert!(
                documents[index].content.starts_with(expected),
                "document {index} dropped CASS anchor {expected}"
            );
        }
        assert!(cass_profile_anchor(9).is_none());
        assert!(cass_profile_anchor(16).is_none());
        assert!(cass_profile_anchor(u64::MAX).is_none());
        let origin_kinds = documents
            .iter()
            .map(|document| {
                document
                    .cass
                    .as_ref()
                    .expect("synthetic document has CASS fields")
                    .origin_kind
                    .as_str()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(origin_kinds, BTreeSet::from(["local", "ssh"]));
        assert!(documents.iter().any(|document| {
            document
                .cass
                .as_ref()
                .is_some_and(|cass| cass.source_id == "source-2")
        }));
        assert!(documents.iter().any(|document| {
            document.cass.as_ref().is_some_and(|cass| {
                (cass.agent == "agent-0" || cass.agent == "agent-1")
                    && cass.workspace == "workspace-0"
                    && cass.source_id == "source-0"
            })
        }));

        let corpus_manifest = corpus.manifest().expect("CASS corpus manifest");
        let corpus_hash = corpus_manifest.manifest_hash().expect("corpus hash");
        let query_suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 24,
                default_limit: 20,
                include_shared_relevance_queries: false,
            },
            &corpus_hash,
            &SharedFixtureSuite::load().expect("shared fixtures"),
        )
        .expect("CASS query suite");
        let case = |id: &str| {
            query_suite
                .cases
                .iter()
                .find(|case| case.id == id)
                .unwrap_or_else(|| panic!("missing generated CASS case {id}"))
        };
        assert_eq!(case("glob-complex").query, "con*fig*");
        assert_eq!(
            case("filter-local").filters.source_filter,
            GeneratedSourceFilter::Local
        );
        assert_eq!(
            case("filter-remote").filters.source_filter,
            GeneratedSourceFilter::Remote
        );
    }

    #[test]
    fn configured_byte_boundary_is_fail_closed_and_utf8_safe() {
        for cap in [MAX_DOCUMENT_BYTES - 1, MAX_DOCUMENT_BYTES] {
            let corpus = SyntheticCorpus::new(SyntheticCorpusSpec {
                max_document_bytes: cap,
                ..small_spec(4, 10)
            })
            .expect("accepted cap");
            let document = corpus.document_at(9).expect("near-limit anchor");
            assert!(document.content.len() <= usize_from_u32(cap));
            assert!(document.content.is_char_boundary(document.content.len()));
        }
        assert!(
            SyntheticCorpus::new(SyntheticCorpusSpec {
                max_document_bytes: MAX_DOCUMENT_BYTES + 1,
                ..small_spec(4, 10)
            })
            .is_err()
        );
        assert!(
            SyntheticCorpus::new(SyntheticCorpusSpec {
                vocabulary_size: 0,
                ..small_spec(4, 10)
            })
            .is_err()
        );
    }

    #[test]
    fn random_access_is_prefix_stable_and_seed_sensitive() {
        let short = SyntheticCorpus::new(small_spec(31, 12)).expect("valid corpus");
        let long = SyntheticCorpus::new(small_spec(31, 40)).expect("valid corpus");
        let changed = SyntheticCorpus::new(small_spec(32, 40)).expect("valid corpus");
        for index in 0..12 {
            assert_eq!(short.document_at(index), long.document_at(index));
        }
        assert_ne!(long.document_at(10), changed.document_at(10));
        assert!(short.document_at(12).is_none());
    }

    #[test]
    fn corpus_manifest_replays_and_detects_seed_content_and_config_mutations() {
        let corpus = SyntheticCorpus::new(small_spec(91, 12)).expect("valid corpus");
        let manifest = corpus.manifest().expect("manifest");
        corpus.verify_manifest(&manifest).expect("exact replay");
        assert_eq!(manifest.document_count, 12);
        assert_eq!(manifest.content_sha256.len(), 64);
        let bytes = manifest.canonical_bytes().expect("canonical bytes");
        let decoded: CorpusManifest = serde_json::from_slice(&bytes).expect("decode manifest");
        assert_eq!(decoded.canonical_bytes().expect("re-encode"), bytes);

        let changed_seed = SyntheticCorpus::new(small_spec(92, 12))
            .expect("valid corpus")
            .manifest()
            .expect("manifest");
        assert_ne!(
            manifest.manifest_hash().unwrap(),
            changed_seed.manifest_hash().unwrap()
        );

        let mut tampered = corpus.iter().collect::<Vec<_>>();
        tampered[10].content.push_str(" tamper");
        assert!(manifest.verify_documents(&tampered).is_err());

        let documents = corpus.iter().collect::<Vec<_>>();
        let mut forged_schema = manifest.clone();
        forged_schema.schema_version = 999;
        assert!(forged_schema.verify_documents(&documents).is_err());
        let mut forged_recipe = manifest.clone();
        let CorpusSourceManifest::Synthetic { spec } = &mut forged_recipe.source else {
            panic!("synthetic manifest source");
        };
        spec.seed = spec.seed.wrapping_add(1);
        assert!(forged_recipe.verify_documents(&documents).is_err());

        let mut oversized_recipe = manifest.clone();
        let CorpusSourceManifest::Synthetic { spec } = &mut oversized_recipe.source else {
            panic!("synthetic manifest source");
        };
        spec.document_count = MAX_CORPUS_DOCUMENT_COUNT + 1;
        oversized_recipe.document_count = MAX_CORPUS_DOCUMENT_COUNT + 1;
        assert!(
            oversized_recipe
                .verify_documents(std::iter::empty::<&GeneratedDocument>())
                .is_err()
        );
    }

    #[test]
    fn shared_fixtures_expose_explicit_core_and_full_views() {
        let shared = SharedFixtureSuite::load().expect("committed fixtures are valid");
        assert_eq!(
            shared.documents(SharedCorpusView::Core100).len(),
            CORE_RELEVANCE_DOCUMENT_COUNT
        );
        assert_eq!(
            shared.documents(SharedCorpusView::Full120).len(),
            FULL_SHARED_DOCUMENT_COUNT
        );
        assert_eq!(shared.relevance_queries().len(), 26);
        assert_eq!(shared.edge_cases().len(), 21);
        assert_eq!(shared.harvested_contract_queries().len(), 7);
        let core = shared
            .manifest(SharedCorpusView::Core100)
            .expect("manifest");
        let full = shared
            .manifest(SharedCorpusView::Full120)
            .expect("manifest");
        assert_ne!(core.content_sha256, full.content_sha256);
        core.verify_documents(shared.documents(SharedCorpusView::Core100))
            .expect("core replay");
        full.verify_documents(shared.documents(SharedCorpusView::Full120))
            .expect("full replay");
    }

    #[test]
    fn query_suite_covers_every_required_class_and_all_harvested_rows() {
        let shared = SharedFixtureSuite::load().expect("fixtures");
        let corpus_hash = shared
            .manifest(SharedCorpusView::Core100)
            .expect("manifest")
            .manifest_hash()
            .expect("hash");
        let suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 73,
                default_limit: 20,
                include_shared_relevance_queries: true,
            },
            corpus_hash,
            &shared,
        )
        .expect("suite");
        assert_eq!(
            suite
                .cases
                .iter()
                .filter(|case| case.source == "tests/fixtures/queries.json")
                .count(),
            26
        );
        assert_eq!(
            suite
                .cases
                .iter()
                .filter(|case| {
                    case.source
                        .starts_with("tests/fixtures/quill_language_contract.json#")
                })
                .count(),
            7
        );
        assert!(suite.cases.iter().any(|case| {
            case.id == "contract-harvest-range"
                && case.filters.created_from_ms == Some(1_700_000_000_000)
                && case.filters.created_to_ms == Some(1_700_000_000_999)
        }));
        let globs = suite
            .cases
            .iter()
            .filter_map(|case| match &case.query_kind {
                GeneratedQueryKind::Glob { pattern_class } => Some(*pattern_class),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            globs,
            BTreeSet::from([
                GlobPatternClass::Exact,
                GlobPatternClass::Prefix,
                GlobPatternClass::Suffix,
                GlobPatternClass::Substring,
                GlobPatternClass::Complex,
            ])
        );
        let ranges = suite
            .cases
            .iter()
            .filter_map(|case| match &case.query_kind {
                GeneratedQueryKind::Range { range_class } => Some(*range_class),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            ranges,
            BTreeSet::from([RangeClass::Inclusive, RangeClass::From, RangeClass::To])
        );
        let structured_filters = suite
            .cases
            .iter()
            .filter_map(|case| match &case.query_kind {
                GeneratedQueryKind::StructuredFilter { filter_class } => Some(*filter_class),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            structured_filters,
            BTreeSet::from([
                StructuredFilterClass::Agent,
                StructuredFilterClass::Workspace,
                StructuredFilterClass::Local,
                StructuredFilterClass::Remote,
                StructuredFilterClass::SourceId,
                StructuredFilterClass::Combined,
            ])
        );
        for boolean_id in ["boolean-cass-and", "boolean-cass-or", "boolean-cass-not"] {
            assert!(
                suite.cases.iter().any(|case| case.id == boolean_id),
                "CASS matrix dropped {boolean_id}"
            );
        }
        assert!(suite.cases.iter().any(|case| case.offset != 0));
        assert!(suite.cases.iter().any(|case| case.count_requested));
        assert!(suite.cases.iter().any(|case| !case.count_requested));
        assert!(
            suite
                .cases
                .iter()
                .any(|case| case.syntax == QuerySyntax::Cass)
        );
        assert!(
            suite
                .cases
                .iter()
                .all(|case| case.expected_divergence.is_none())
        );
        suite.manifest.verify(&suite.cases).expect("query replay");
    }

    #[test]
    fn differential_lowering_binds_manifest_and_preserves_common_fields() {
        let shared = SharedFixtureSuite::load().expect("fixtures");
        let corpus_hash = shared
            .manifest(SharedCorpusView::Core100)
            .expect("manifest")
            .manifest_hash()
            .expect("hash");
        let suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 4,
                default_limit: 10,
                include_shared_relevance_queries: false,
            },
            &corpus_hash,
            &shared,
        )
        .expect("suite");
        let term = suite.cases.iter().find(|case| case.id == "term").unwrap();
        let lowered = term
            .to_differential_case(4, &corpus_hash)
            .expect("default term is representable");
        assert_eq!(
            lowered.metadata.corpus_hash.as_deref(),
            Some(corpus_hash.as_str())
        );
        validate_generated_case_metadata(&lowered).expect("complete generated metadata");
        let paginated = suite
            .cases
            .iter()
            .find(|case| case.id == "paginated")
            .unwrap();
        let paginated_case = paginated
            .to_differential_case(4, &corpus_hash)
            .expect("pagination has a common envelope");
        assert_eq!(paginated_case.offset, paginated.offset);
        let cass = suite
            .cases
            .iter()
            .find(|case| case.id == "boolean-cass")
            .unwrap();
        assert_eq!(
            cass.to_differential_case(4, &corpus_hash)
                .expect("CASS has a common envelope")
                .query,
            cass.query
        );
        assert!(term.to_differential_case(4, "0123456789abcdef").is_err());
        let mut duplicate_cases = suite.cases.clone();
        duplicate_cases.push(suite.cases[0].clone());
        assert!(suite.manifest.verify(&duplicate_cases).is_err());
        let mut oversized_query = suite.cases[0].clone();
        oversized_query.query = "x".repeat(MAX_QUERY_TEXT_BYTES + 1);
        assert!(
            GeneratedQuerySuite::from_cases(
                suite.manifest.spec.clone(),
                &corpus_hash,
                vec![oversized_query],
            )
            .is_err()
        );
        let mut noncanonical_id = suite.cases[0].clone();
        noncanonical_id.id = " leading-space".to_owned();
        assert!(
            GeneratedQuerySuite::from_cases(
                suite.manifest.spec.clone(),
                &corpus_hash,
                vec![noncanonical_id],
            )
            .is_err()
        );
        let mut invalid_divergence_id = suite.cases[0].clone();
        invalid_divergence_id.expected_divergence = Some("unreviewed".to_owned());
        assert!(
            GeneratedQuerySuite::from_cases(
                suite.manifest.spec.clone(),
                &corpus_hash,
                vec![invalid_divergence_id],
            )
            .is_err()
        );
        let mut duplicate_filter = suite
            .cases
            .iter()
            .find(|case| case.id == "filter-agent")
            .expect("agent filter")
            .clone();
        duplicate_filter.filters.agents = vec!["agent-0".to_owned(), "agent-0".to_owned()];
        assert!(
            GeneratedQuerySuite::from_cases(
                suite.manifest.spec.clone(),
                &corpus_hash,
                vec![duplicate_filter],
            )
            .is_err()
        );
        let mut inverted_range = suite
            .cases
            .iter()
            .find(|case| case.id == "range-inclusive")
            .expect("range filter")
            .clone();
        inverted_range.filters.created_from_ms = Some(2);
        inverted_range.filters.created_to_ms = Some(1);
        assert!(
            GeneratedQuerySuite::from_cases(
                suite.manifest.spec.clone(),
                &corpus_hash,
                vec![inverted_range],
            )
            .is_err()
        );
        let mut default_with_filter = suite.cases[0].clone();
        default_with_filter.filters.agents = vec!["agent-0".to_owned()];
        assert!(
            GeneratedQuerySuite::from_cases(
                suite.manifest.spec.clone(),
                &corpus_hash,
                vec![default_with_filter],
            )
            .is_err()
        );
        let mut forged_manifest = suite.manifest.clone();
        forged_manifest.generator_id = "future-generator".to_owned();
        assert!(forged_manifest.verify(&suite.cases).is_err());
        let mut forged_spec = suite.manifest.clone();
        forged_spec.spec.seed = forged_spec.spec.seed.wrapping_add(1);
        assert!(forged_spec.verify(&suite.cases).is_err());
        assert_eq!(suite.manifest.source, QuerySuiteSource::Generated);
        assert!(
            GeneratedQuerySuite::generate(
                QueryGeneratorSpec {
                    seed: 4,
                    default_limit: MAX_GENERATED_QUERY_LIMIT + 1,
                    include_shared_relevance_queries: false,
                },
                &corpus_hash,
                &shared,
            )
            .is_err()
        );
    }

    #[test]
    fn contract_query_lowering_rejects_unknown_classes_and_filters() {
        let shared = SharedFixtureSuite::load().expect("fixtures");
        let mut query = shared.harvested_contract_queries()[0].clone();
        query.query_class = "future_class".to_owned();
        assert!(contract_query_case(&query, 10).is_err());
        query.query_class = "identifier".to_owned();
        query.filters.insert("future_filter".to_owned(), 1);
        assert!(contract_query_case(&query, 10).is_err());
    }

    #[test]
    fn repository_entries_are_sorted_content_addressed_and_fail_closed() {
        let snapshot = RepositorySnapshot::from_entries(
            "frankensearch",
            [
                RepositoryEntry {
                    relative_path: PathBuf::from("src/z.rs"),
                    bytes: b"fn z() {}\n".to_vec(),
                },
                RepositoryEntry {
                    relative_path: PathBuf::from("README.md"),
                    bytes: b"# repository\n".to_vec(),
                },
                RepositoryEntry {
                    relative_path: PathBuf::from("binary.bin"),
                    bytes: vec![0xff, 0xfe],
                },
                RepositoryEntry {
                    relative_path: PathBuf::from("large.txt"),
                    bytes: vec![b'x'; usize_from_u32(MAX_DOCUMENT_BYTES) + 1],
                },
            ],
        )
        .expect("snapshot");
        assert_eq!(snapshot.documents.len(), 2);
        assert_eq!(snapshot.documents[0].id, "repo:README.md");
        assert_eq!(snapshot.documents[1].id, "repo:src/z.rs");
        assert!(
            snapshot
                .documents
                .iter()
                .all(|document| document.unicode_lane == UnicodeLane::Ascii)
        );
        assert_eq!(snapshot.manifest.skipped_repository_entries.len(), 2);
        snapshot
            .manifest
            .verify_documents(&snapshot.documents)
            .expect("snapshot replay");
        let CorpusSourceManifest::Repository { files: source, .. } = &snapshot.manifest.source
        else {
            assert!(
                matches!(
                    snapshot.manifest.source,
                    CorpusSourceManifest::Repository { .. }
                ),
                "repository source expected"
            );
            return;
        };
        assert_eq!(source[0].path, "README.md");
        assert_eq!(source[1].path, "src/z.rs");

        let mut forged_digest = snapshot.manifest.clone();
        let CorpusSourceManifest::Repository { files, .. } = &mut forged_digest.source else {
            panic!("repository source expected");
        };
        files[0].sha256 = "0".repeat(64);
        assert!(forged_digest.verify_documents(&snapshot.documents).is_err());

        let mut swapped_sizes = snapshot.manifest.clone();
        let CorpusSourceManifest::Repository { files, .. } = &mut swapped_sizes.source else {
            panic!("repository source expected");
        };
        let first_bytes = files[0].bytes;
        files[0].bytes = files[1].bytes;
        files[1].bytes = first_bytes;
        assert!(swapped_sizes.verify_documents(&snapshot.documents).is_err());

        let mut oversized_document = snapshot.documents[0].clone();
        oversized_document.content = "x".repeat(usize_from_u32(MAX_DOCUMENT_BYTES) + 1);
        let oversized_bytes = oversized_document.content.as_bytes();
        let oversized_manifest = CorpusManifest::from_documents(
            CorpusSourceManifest::Repository {
                repository_id: "frankensearch".to_owned(),
                files: vec![RepositoryFileDigest {
                    path: "README.md".to_owned(),
                    bytes: u64_from_usize(oversized_bytes.len()),
                    sha256: sha256_hex(oversized_bytes),
                }],
            },
            [&oversized_document],
            Vec::new(),
        )
        .expect("forge oversized manifest DTO");
        assert!(
            oversized_manifest
                .verify_documents([&oversized_document])
                .is_err()
        );

        assert!(
            RepositorySnapshot::from_entries(
                "frankensearch",
                [RepositoryEntry {
                    relative_path: PathBuf::from("../escape"),
                    bytes: Vec::new(),
                }]
            )
            .is_err()
        );
        assert!(
            RepositorySnapshot::from_entries(
                "r".repeat(MAX_REPOSITORY_ID_BYTES + 1),
                std::iter::empty::<RepositoryEntry>(),
            )
            .is_err()
        );
        let mut forged_skip = snapshot.manifest.clone();
        forged_skip
            .skipped_repository_entries
            .push(SkippedRepositoryEntry {
                path: "x".repeat(MAX_REPOSITORY_PATH_BYTES + 1),
                reason: RepositorySkipReason::Missing,
            });
        assert!(forged_skip.verify_documents(&snapshot.documents).is_err());
        assert!(
            RepositorySnapshot::from_entries(
                "frankensearch",
                [RepositoryEntry {
                    relative_path: PathBuf::from("src\\platform-dependent.rs"),
                    bytes: Vec::new(),
                }]
            )
            .is_err()
        );
        assert!(
            RepositorySnapshot::from_entries(
                "frankensearch",
                [RepositoryEntry {
                    relative_path: PathBuf::from("x".repeat(MAX_DOCUMENT_ID_BYTES)),
                    bytes: b"bounded path".to_vec(),
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn external_content_script_classification_is_deterministic() {
        assert_eq!(classify_unicode_lane("plain ASCII"), UnicodeLane::Ascii);
        assert_eq!(
            classify_unicode_lane("naïve cafe\u{301}"),
            UnicodeLane::Latin1
        );
        assert_eq!(classify_unicode_lane("搜索 エンジン"), UnicodeLane::Cjk);
        assert_eq!(classify_unicode_lane("𠀀 𪛟"), UnicodeLane::CjkExtensionB);
        assert_eq!(classify_unicode_lane("café 搜索 𠀀"), UnicodeLane::Mixed);
    }

    #[cfg(not(windows))]
    #[test]
    fn tracked_repository_lane_reads_only_explicit_relative_paths() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let snapshot = RepositorySnapshot::from_tracked_paths(
            root,
            "frankensearch-quill-gauntlet",
            [
                PathBuf::from("src/lib.rs"),
                PathBuf::from("Cargo.toml"),
                PathBuf::from("missing-tracked-file"),
            ],
        )
        .expect("tracked snapshot");
        assert_eq!(snapshot.documents.len(), 2);
        assert_eq!(snapshot.documents[0].id, "repo:Cargo.toml");
        assert_eq!(snapshot.documents[1].id, "repo:src/lib.rs");
        assert_eq!(
            snapshot.manifest.skipped_repository_entries,
            [SkippedRepositoryEntry {
                path: "missing-tracked-file".to_owned(),
                reason: RepositorySkipReason::Missing,
            }]
        );
    }

    #[cfg(unix)]
    #[test]
    fn tracked_repository_lane_rejects_final_and_intermediate_symlinks() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("snapshot root");
        let outside = tempfile::tempdir().expect("outside root");
        let outside_file = outside.path().join("secret.txt");
        std::fs::write(&outside_file, b"outside target must never be read").expect("outside file");
        symlink(&outside_file, root.path().join("final-link")).expect("final symlink");
        symlink(outside.path(), root.path().join("intermediate")).expect("directory symlink");

        let snapshot = RepositorySnapshot::from_tracked_paths(
            root.path(),
            "symlink-regression",
            [
                PathBuf::from("final-link"),
                PathBuf::from("intermediate/secret.txt"),
            ],
        )
        .expect("symlinks are recorded as skipped entries");

        assert!(snapshot.documents.is_empty());
        assert_eq!(
            snapshot.manifest.skipped_repository_entries,
            [
                SkippedRepositoryEntry {
                    path: "final-link".to_owned(),
                    reason: RepositorySkipReason::NotRegularFile,
                },
                SkippedRepositoryEntry {
                    path: "intermediate/secret.txt".to_owned(),
                    reason: RepositorySkipReason::NotRegularFile,
                },
            ]
        );
    }

    #[cfg(windows)]
    #[test]
    fn tracked_repository_lane_reports_unsupported_on_windows() {
        let error = RepositorySnapshot::from_tracked_paths(
            Path::new(env!("CARGO_MANIFEST_DIR")),
            "frankensearch-quill-gauntlet",
            [PathBuf::from("Cargo.toml")],
        )
        .expect_err("descriptor-relative tracked reads are unsupported on Windows");
        assert!(matches!(
            error,
            GauntletError::Io(error) if error.kind() == std::io::ErrorKind::Unsupported
        ));
    }

    #[test]
    fn xlarge_recipe_is_exact_and_random_access_without_materialization() {
        let matrix = SyntheticCorpusSpec::xlarge_matrix(101);
        assert_eq!(
            matrix.each_ref().map(|spec| spec.zipf_exponent),
            ZipfExponent::ALL
        );
        let corpus = SyntheticCorpus::new(matrix[1].clone()).expect("xlarge recipe");
        assert_eq!(corpus.len(), XLARGE_DOCUMENT_COUNT);
        assert!(corpus.document_at(XLARGE_DOCUMENT_COUNT - 1).is_some());
        assert!(corpus.document_at(XLARGE_DOCUMENT_COUNT).is_none());
        assert_eq!(
            corpus.iter().size_hint(),
            (
                usize::try_from(XLARGE_DOCUMENT_COUNT).unwrap_or(usize::MAX),
                Some(usize::try_from(XLARGE_DOCUMENT_COUNT).unwrap_or(usize::MAX))
            )
        );
    }

    #[test]
    #[ignore = "nightly-only three-million-document xlarge matrix integrity lane"]
    fn nightly_xlarge_stream_hash_integrity() {
        let corpora = SyntheticCorpusSpec::xlarge_matrix(0x5eed)
            .map(|spec| SyntheticCorpus::new(spec).expect("xlarge recipe must remain valid"));
        let manifests = corpora
            .each_ref()
            .map(|corpus| corpus.manifest().expect("streamed xlarge manifest"));
        let actual = (
            manifests.each_ref().map(|manifest| manifest.document_count),
            manifests
                .each_ref()
                .map(|manifest| manifest.total_content_bytes),
            manifests
                .each_ref()
                .map(|manifest| manifest.content_sha256.clone()),
            manifests
                .each_ref()
                .map(|manifest| manifest.manifest_hash().expect("xlarge manifest hash")),
        );

        assert_eq!(
            actual,
            (
                [XLARGE_DOCUMENT_COUNT; 3],
                [3_978_050_328; 3],
                [
                    "80f670f33cca9cf086fb0c735889f6be03bf0ef31250bbcd8e5466708c2ad8cd".to_owned(),
                    "ec301008e69ec8407c5880da68795d82cbb0993029f313875dd1e54e1343ebd0".to_owned(),
                    "c6b1cb940bb25b23cf202ce43b248dcd9121a6278ead8d2d6c23bfa1cedfea7d".to_owned(),
                ],
                [
                    "bc8c37a9f914ad9f60ad6cf57512b6cabd54a124410a106e406fa20c5f2dbb7b".to_owned(),
                    "9fbd168d0e924c2a2f6d3e8aff3b714d37fb1a7cd8628c03e4478816ef62f1a0".to_owned(),
                    "67202aea2fb0b30b6e1350b2340218a9a741f391e3d35746e44cff1447a13ed8".to_owned(),
                ],
            )
        );
    }
}
