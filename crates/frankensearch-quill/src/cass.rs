//! Native CASS profile surface for Quill-backed indexes.
//!
//! The CASS lexical profile historically lived only in the Tantivy backend
//! (`frankensearch-lexical::cass_compat`). Quill already owns the parts that
//! actually needed an engine — [`crate::schema::CASS_SEMANTIC_SCHEMA`], the
//! [`crate::schema::Analyzer::CassHyphenNormalize`] and
//! [`crate::schema::Analyzer::CassPrefixNormalize`] pipelines, and
//! [`crate::query::CassQueryParser`] — so this module carries only the
//! remaining engine-independent pieces: the ingest document shape, the
//! derived-column rules, and the on-disk namespacing that lets an index
//! declare which schema generation produced it.
//!
//! # Why the schema sentinel changes
//!
//! A Quill index is FSLX; it cannot read Tantivy segment files at all. Any
//! consumer moving from the Tantivy CASS backend to this one must rebuild,
//! and [`CASS_SCHEMA_HASH`] is exactly the mechanism that makes that rebuild
//! automatic rather than a corrupt read: a consumer stores the sentinel beside
//! its index and calls [`cass_schema_hash_matches`] on open. A changed
//! sentinel means "discard and reindex", which is the correct and only safe
//! answer here.

use std::path::{Path, PathBuf};

use crate::index::SchemaDocument;
use crate::scribe::cass_generate_edge_ngrams;

/// Schema generation namespace for Quill-backed CASS indexes.
///
/// This is deliberately not `v8`: `v8` names the Tantivy-era on-disk
/// generation, and reusing it would invite a consumer to open FSLX segments
/// with a Tantivy reader or the reverse.
pub const CASS_SCHEMA_VERSION: &str = "v9-quill";

/// Rebuild sentinel recorded beside a CASS index.
///
/// It is a descriptive marker rather than a computed digest, matching the
/// incumbent contract: consumers compare it verbatim and rebuild on any
/// difference. It names the engine because the engine determines the on-disk
/// format, and the analyzer family because a tokenizer change silently
/// invalidates every posting.
pub const CASS_SCHEMA_HASH: &str = "quill-fslx-schema-v9-hyphen-cjk-bigrams-bounded-content-prefix-preview-stored-content-external";

/// Longest preview retained in the stored `preview` column, in characters.
pub const PREVIEW_MAX_CHARS: usize = 400;

/// Largest content prefix fed to the edge-ngram column, in bytes.
pub const CONTENT_PREFIX_MAX_BYTES: usize = 4 * 1024;

/// Whether a stored sentinel names this exact schema generation.
///
/// Any difference — older generation, Tantivy-era generation, or corruption —
/// answers `false`, which callers treat as "rebuild".
#[must_use]
pub fn cass_schema_hash_matches(stored: &str) -> bool {
    stored == CASS_SCHEMA_HASH
}

/// Resolve (and create) the schema-qualified index directory beneath `base`.
///
/// Namespacing by schema generation is what lets a rebuild land beside the old
/// index instead of on top of it, so a half-finished migration never leaves a
/// consumer pointing at mixed-generation segments.
///
/// # Errors
///
/// Returns the underlying I/O error when the directory cannot be created.
pub fn cass_index_dir(base: &Path) -> std::io::Result<PathBuf> {
    let dir = base.join("index").join(CASS_SCHEMA_VERSION);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Segment-count threshold at which a CASS index wants compaction.
pub const CASS_MERGE_SEGMENT_THRESHOLD: usize = 4;

/// Minimum interval between CASS compactions, in milliseconds.
pub const CASS_MERGE_COOLDOWN_MS: i64 = 300_000;

/// Whether a CASS index currently wants compaction, and why.
///
/// Mirrors the incumbent's shape and thresholds so a consumer's merge policy
/// does not silently change meaning across the engine swap. `last_merge_ts` and
/// `ms_since_last_merge` are supplied by the caller because Quill does not own
/// a global merge clock: a negative `ms_since_last_merge` means "never merged",
/// which the cooldown treats as elapsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassMergeStatus {
    /// Live segment count in the published snapshot.
    pub segment_count: usize,
    /// Epoch milliseconds of the last compaction, or 0 if never.
    pub last_merge_ts: i64,
    /// Milliseconds since the last compaction, negative if never.
    pub ms_since_last_merge: i64,
    /// Segment count at or above which compaction is wanted.
    pub merge_threshold: usize,
    /// Minimum interval between compactions.
    pub cooldown_ms: i64,
}

impl CassMergeStatus {
    /// Whether the segment count has reached the threshold and the cooldown has
    /// elapsed.
    #[must_use]
    pub const fn should_merge(&self) -> bool {
        self.segment_count >= self.merge_threshold
            && (self.ms_since_last_merge < 0 || self.ms_since_last_merge >= self.cooldown_ms)
    }

    /// Build a status from a live segment count and merge clock.
    #[must_use]
    pub const fn new(segment_count: usize, last_merge_ts: i64, now_ms: i64) -> Self {
        Self {
            segment_count,
            last_merge_ts,
            ms_since_last_merge: if last_merge_ts > 0 {
                now_ms - last_merge_ts
            } else {
                -1
            },
            merge_threshold: CASS_MERGE_SEGMENT_THRESHOLD,
            cooldown_ms: CASS_MERGE_COOLDOWN_MS,
        }
    }
}

/// One CASS message, owned.
///
/// The field set mirrors the incumbent ingest DTO exactly. The three derived
/// columns (`title_prefix`, `content_prefix`, `preview`) are deliberately
/// absent: they are computed at index time by [`CassDerivedColumns::derive`],
/// so a caller cannot desynchronize them from the source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassDocument {
    /// Agent that produced the session.
    pub agent: String,
    /// Normalized workspace identifier.
    pub workspace: Option<String>,
    /// Workspace string exactly as observed.
    pub workspace_original: Option<String>,
    /// Absolute path of the source transcript.
    pub source_path: String,
    /// Zero-based message ordinal within its conversation.
    pub msg_idx: u64,
    /// Message timestamp, seconds since the Unix epoch.
    pub created_at: Option<i64>,
    /// Conversation title, when the transcript carries one.
    pub title: Option<String>,
    /// Message body.
    pub content: String,
    /// Durable source identifier.
    pub source_id: String,
    /// Origin classification (local, imported, federated, ...).
    pub origin_kind: String,
    /// Origin host for federated sources.
    pub origin_host: Option<String>,
    /// Conversation grouping key.
    pub conversation_id: Option<i64>,
}

/// One CASS message, borrowed.
///
/// Ingest paths take this form so a caller streaming from its own store never
/// has to clone message bodies just to hand them to the indexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassDocumentRef<'a> {
    /// Agent that produced the session.
    pub agent: &'a str,
    /// Normalized workspace identifier.
    pub workspace: Option<&'a str>,
    /// Workspace string exactly as observed.
    pub workspace_original: Option<&'a str>,
    /// Absolute path of the source transcript.
    pub source_path: &'a str,
    /// Zero-based message ordinal within its conversation.
    pub msg_idx: u64,
    /// Message timestamp, seconds since the Unix epoch.
    pub created_at: Option<i64>,
    /// Conversation title, when the transcript carries one.
    pub title: Option<&'a str>,
    /// Message body.
    pub content: &'a str,
    /// Durable source identifier.
    pub source_id: &'a str,
    /// Origin classification (local, imported, federated, ...).
    pub origin_kind: &'a str,
    /// Origin host for federated sources.
    pub origin_host: Option<&'a str>,
    /// Conversation grouping key.
    pub conversation_id: Option<i64>,
}

impl CassDocument {
    /// Borrow this document without copying any field.
    #[must_use]
    pub fn as_ref(&self) -> CassDocumentRef<'_> {
        CassDocumentRef {
            agent: &self.agent,
            workspace: self.workspace.as_deref(),
            workspace_original: self.workspace_original.as_deref(),
            source_path: &self.source_path,
            msg_idx: self.msg_idx,
            created_at: self.created_at,
            title: self.title.as_deref(),
            content: &self.content,
            source_id: &self.source_id,
            origin_kind: &self.origin_kind,
            origin_host: self.origin_host.as_deref(),
            conversation_id: self.conversation_id,
        }
    }
}

/// Canonical CASS document identity.
///
/// # Why this is conversation-scoped
///
/// The Tantivy-era identity was `"{source_id}#{msg_idx}"`. That is NOT unique:
/// `source_id` is the *source*, not the conversation — every locally-discovered
/// conversation shares `LOCAL_SOURCE_ID` — so message 0 of every local
/// conversation collides on `"local#0"`. Tantivy tolerated it because a
/// document id there is an ordinary field with no uniqueness constraint. Quill
/// treats the document id as a primary key and refuses duplicates, which is how
/// the collision surfaced at all.
///
/// The conversation discriminator is `conversation_id` when the store has
/// assigned one, and the transcript path otherwise — a conversation always has
/// one or the other, and both are stable across reindexing.
#[must_use]
pub fn cass_document_identity(
    source_id: &str,
    conversation: CassConversationKey<'_>,
    msg_idx: u64,
) -> String {
    let path = conversation.source_path;
    conversation.id.map_or_else(
        || format!("{source_id}#{path}#{msg_idx}"),
        |id| format!("{source_id}#{path}#{id}#{msg_idx}"),
    )
}

/// What discriminates one conversation from another within a source.
///
/// Both axes are carried rather than one: the transcript path alone collides
/// when a single file holds several conversations, and the assigned id alone
/// collides when a caller reuses one id across conversations. Neither is
/// individually guaranteed, so the identity uses the path always and the id
/// additionally whenever the store has assigned one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassConversationKey<'a> {
    /// Absolute transcript path, always present.
    pub source_path: &'a str,
    /// The store's assigned conversation id, when it has one.
    pub id: Option<i64>,
}

impl<'a> CassConversationKey<'a> {
    /// The discriminator for `document`.
    #[must_use]
    pub const fn for_document(document: CassDocumentRef<'a>) -> Self {
        Self {
            source_path: document.source_path,
            id: document.conversation_id,
        }
    }

    /// A discriminator from explicit parts.
    #[must_use]
    pub const fn new(source_path: &'a str, id: Option<i64>) -> Self {
        Self { source_path, id }
    }
}

/// The three columns computed from a document rather than supplied with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassDerivedColumns {
    /// Edge n-grams over the title, empty when the document has no title.
    pub title_prefix: String,
    /// Edge n-grams over a bounded content prefix.
    pub content_prefix: String,
    /// Character-bounded, ellipsized content preview.
    pub preview: String,
    /// `conversation_id` in the canonical encoding for its stored column.
    ///
    /// The field is `I64 { indexed: false, fast: false }` with `stored: true`.
    /// Scribe validates a stored numeric column as exactly eight little-endian
    /// bytes, so this is the required encoding rather than a chosen one: a
    /// friendlier base-10 rendering is refused at accumulation with
    /// `InvalidNumericBytes`. Owned here so the borrowed value slices have
    /// something to point at.
    pub conversation_id_bytes: Option<[u8; 8]>,
}

impl CassDerivedColumns {
    /// Compute every derived column for one document.
    #[must_use]
    pub fn derive(document: CassDocumentRef<'_>) -> Self {
        let (content_prefix, preview) = cass_build_content_prefix_and_preview(document.content);
        Self {
            title_prefix: document
                .title
                .map_or_else(String::new, cass_generate_edge_ngrams),
            content_prefix,
            preview,
            conversation_id_bytes: document.conversation_id.map(i64::to_le_bytes),
        }
    }
}

/// Field ordinals of [`crate::schema::CASS_SEMANTIC_SCHEMA`], by name.
///
/// The ordinals are fixed by the compiled descriptor, so the CASS profile needs
/// no runtime field-handle lookup: this replaces the incumbent `CassFields`
/// struct outright. They are named rather than inlined so a schema reordering
/// is a compile-time edit in one place instead of a silent column swap.
pub mod field {
    /// `agent`, keyword.
    pub const AGENT: u16 = 0;
    /// `workspace`, keyword.
    pub const WORKSPACE: u16 = 1;
    /// `workspace_original`, stored only.
    pub const WORKSPACE_ORIGINAL: u16 = 2;
    /// `source_path`, stored only.
    pub const SOURCE_PATH: u16 = 3;
    /// `msg_idx`, indexed unsigned.
    pub const MSG_IDX: u16 = 4;
    /// `created_at`, indexed and fast signed.
    pub const CREATED_AT: u16 = 5;
    /// `title`, hyphen-normalized text.
    pub const TITLE: u16 = 6;
    /// `content`, hyphen-normalized text.
    pub const CONTENT: u16 = 7;
    /// `title_prefix`, prefix-normalized text.
    pub const TITLE_PREFIX: u16 = 8;
    /// `content_prefix`, prefix-normalized text.
    pub const CONTENT_PREFIX: u16 = 9;
    /// `preview`, stored only.
    pub const PREVIEW: u16 = 10;
    /// `source_id`, keyword.
    pub const SOURCE_ID: u16 = 11;
    /// `origin_kind`, keyword.
    pub const ORIGIN_KIND: u16 = 12;
    /// `origin_host`, keyword.
    pub const ORIGIN_HOST: u16 = 13;
    /// `conversation_id`, stored signed.
    pub const CONVERSATION_ID: u16 = 14;
}

/// The complete column set for one CASS document, ready for accumulation.
///
/// This is the value-construction half of CASS ingest, deliberately separated
/// from the ingest pipeline so it can be built and asserted on without a live
/// index. Text and byte columns borrow — from the document for source columns
/// and from [`CassDerivedColumns`] for derived ones — so building these copies
/// nothing.
///
/// Absent optional columns are omitted rather than written as empty strings: an
/// empty keyword is a real, matchable term, so writing one would make
/// "no workspace recorded" indistinguishable from "workspace is the empty
/// string" at query time.
#[derive(Debug, Clone)]
pub struct CassFieldValues<'a> {
    /// Analyzed and keyword text columns.
    pub indexed: Vec<crate::scribe::IndexedFieldValue<'a>>,
    /// Numeric columns.
    pub numeric: Vec<crate::scribe::IndexedNumericValue>,
    /// Stored-only byte columns.
    pub stored: Vec<crate::scribe::StoredFieldValue<'a>>,
}

impl<'a> CassFieldValues<'a> {
    /// Build every column for `document`, borrowing derived text from `derived`.
    #[must_use]
    pub fn build(document: CassDocumentRef<'a>, derived: &'a CassDerivedColumns) -> Self {
        use crate::scribe::{IndexedFieldValue, IndexedNumericValue, StoredFieldValue};

        let mut indexed = Vec::with_capacity(10);
        let mut numeric = Vec::with_capacity(2);
        let mut stored = Vec::with_capacity(4);

        indexed.push(IndexedFieldValue::new(field::AGENT, document.agent));
        indexed.push(IndexedFieldValue::new(field::SOURCE_ID, document.source_id));
        indexed.push(IndexedFieldValue::new(
            field::ORIGIN_KIND,
            document.origin_kind,
        ));
        indexed.push(IndexedFieldValue::new(field::CONTENT, document.content));
        indexed.push(IndexedFieldValue::new(
            field::CONTENT_PREFIX,
            &derived.content_prefix,
        ));
        if let Some(workspace) = document.workspace {
            indexed.push(IndexedFieldValue::new(field::WORKSPACE, workspace));
        }
        if let Some(origin_host) = document.origin_host {
            indexed.push(IndexedFieldValue::new(field::ORIGIN_HOST, origin_host));
        }
        if let Some(title) = document.title {
            indexed.push(IndexedFieldValue::new(field::TITLE, title));
            indexed.push(IndexedFieldValue::new(
                field::TITLE_PREFIX,
                &derived.title_prefix,
            ));
        }

        numeric.push(IndexedNumericValue::u64(field::MSG_IDX, document.msg_idx));
        if let Some(created_at) = document.created_at {
            numeric.push(IndexedNumericValue::i64(field::CREATED_AT, created_at));
        }

        stored.push(StoredFieldValue::new(
            field::SOURCE_PATH,
            document.source_path.as_bytes(),
        ));
        stored.push(StoredFieldValue::new(
            field::PREVIEW,
            derived.preview.as_bytes(),
        ));
        if let Some(workspace_original) = document.workspace_original {
            stored.push(StoredFieldValue::new(
                field::WORKSPACE_ORIGINAL,
                workspace_original.as_bytes(),
            ));
        }
        if let Some(conversation_id) = derived.conversation_id_bytes.as_ref() {
            stored.push(StoredFieldValue::new(
                field::CONVERSATION_ID,
                conversation_id,
            ));
        }

        Self {
            indexed,
            numeric,
            stored,
        }
    }
}

impl CassDocument {
    /// Project onto the owned form Quill ingests.
    ///
    /// This derives the computed columns and copies every value, because
    /// [`SchemaDocument`] owns its columns while [`CassFieldValues`] borrows
    /// them. A caller that already holds its documents in memory for the whole
    /// batch can skip the copy by building [`CassFieldValues`] directly.
    #[must_use]
    pub fn to_schema_document(&self) -> SchemaDocument {
        let derived = CassDerivedColumns::derive(self.as_ref());
        let values = CassFieldValues::build(self.as_ref(), &derived);
        SchemaDocument {
            id: cass_document_identity(
                &self.source_id,
                CassConversationKey::for_document(self.as_ref()),
                self.msg_idx,
            ),
            indexed: values
                .indexed
                .iter()
                .map(|value| (value.field_ord, value.text.to_owned()))
                .collect(),
            numeric: values
                .numeric
                .iter()
                .map(|value| (value.field_ord, value.value))
                .collect(),
            stored: values
                .stored
                .iter()
                .map(|value| (value.field_ord, value.bytes.to_vec()))
                .collect(),
        }
    }
}

/// Build the edge-ngram content prefix and the stored preview together.
///
/// They are returned as a pair because both read the same content and must
/// agree about it; computing them apart is how a prefix column drifts from the
/// preview a user is shown.
#[must_use]
pub fn cass_build_content_prefix_and_preview(content: &str) -> (String, String) {
    (
        cass_generate_edge_ngrams(cass_prefix_source(content, CONTENT_PREFIX_MAX_BYTES)),
        cass_build_preview(content, PREVIEW_MAX_CHARS),
    )
}

/// The largest char-boundary prefix of `content` that is at most `max_bytes`.
///
/// UTF-8 scalars are at most four bytes, so the boundary at or below
/// `max_bytes` is at most three bytes below it: the backward walk terminates
/// in at most four steps regardless of input length.
#[must_use]
pub fn cass_prefix_source(content: &str, max_bytes: usize) -> &str {
    if content.len() <= max_bytes {
        return content;
    }
    let mut end = max_bytes;
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    &content[..end]
}

/// Character-bounded preview, ellipsized only when it actually truncates.
#[must_use]
pub fn cass_build_preview(content: &str, max_chars: usize) -> String {
    let mut cut = content.len();
    for (count, (byte_idx, _)) in content.char_indices().enumerate() {
        if count == max_chars {
            cut = byte_idx;
            break;
        }
    }
    let truncated = cut < content.len();
    let mut out = String::with_capacity(cut + if truncated { '…'.len_utf8() } else { 0 });
    out.push_str(&content[..cut]);
    if truncated {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_document(
        source_id: &str,
        msg_idx: u64,
        agent: &str,
        title: &str,
        content: &str,
    ) -> CassDocument {
        CassDocument {
            agent: agent.to_owned(),
            workspace: Some("frankensearch".to_owned()),
            workspace_original: Some("FrankenSearch".to_owned()),
            source_path: format!("/transcripts/{source_id}.jsonl"),
            msg_idx,
            created_at: Some(1_700_000_000 + i64::try_from(msg_idx).expect("fixture ordinal")),
            title: Some(title.to_owned()),
            content: content.to_owned(),
            source_id: source_id.to_owned(),
            origin_kind: "local".to_owned(),
            origin_host: None,
            conversation_id: Some(42),
        }
    }

    /// A conversation at the incumbent's per-conversation indexing cap must be
    /// admitted, not refused.
    ///
    /// cass truncates a conversation's indexed body at 8 MiB and indexes the
    /// truncated text. Quill's admission refuses rather than truncates, so if
    /// its budget were tighter than that cap, migrating a real corpus would
    /// fail on the first large conversation — and fail the whole batch, not
    /// just that document. This pins that the incumbent's ceiling is inside
    /// Quill's admission.
    #[test]
    fn a_conversation_at_the_incumbent_content_cap_is_admitted() {
        const CAP_BYTES: usize = 8 * 1024 * 1024;

        // Realistic token mix rather than one repeated word: a single repeated
        // term would collapse to one dictionary entry and understate the term
        // bucket pressure a real 8 MiB conversation applies.
        let mut content = String::with_capacity(CAP_BYTES + 16);
        let mut ordinal = 0_u32;
        while content.len() < CAP_BYTES {
            use std::fmt::Write as _;
            let _ = write!(content, "token{:07} ", ordinal % 100_000);
            ordinal += 1;
        }

        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let directory = tempfile::tempdir().expect("cass index directory");
            let index = crate::index::QuillIndex::create_with_schema(
                &cx,
                directory.path(),
                crate::schema::CASS_SEMANTIC_SCHEMA,
                crate::QuillConfig::default(),
            )
            .await
            .expect("create a CASS-schema index");

            let document = sample_document("bulk", 0, "claude", "a very long session", &content)
                .to_schema_document();
            index
                .index_schema_documents(&cx, std::slice::from_ref(&document))
                .await
                .expect("a conversation at the incumbent cap must be admitted");
            index.commit(&cx).await.expect("publish the large document");

            let reader = crate::index::QuillSearchIndex::open_with_schema(
                &cx,
                directory.path(),
                crate::schema::CASS_SEMANTIC_SCHEMA,
                crate::QuillConfig::default(),
            )
            .await
            .expect("open the reader");
            assert_eq!(reader.doc_count().expect("doc count"), 1);

            let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
                .expect("build the CASS parser");
            let parsed = parser.parse("token0000042", &crate::query::CassQueryFilters::default());
            let result = reader
                .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
                .expect("search the large document");
            assert_eq!(
                result.total_count,
                Some(1),
                "a term from deep inside the capped body must still be findable"
            );
        });
    }

    /// Ingest and query throughput for the CASS schema path.
    ///
    /// Opt-in (`--ignored`) and release-only to be meaningful. This is a
    /// measurement probe, not a gate: it asserts nothing about timing, it only
    /// reports, so it can never fail the suite for being run on a loaded box.
    #[test]
    #[ignore = "measurement probe; run explicitly under --release --ignored"]
    fn cass_schema_ingest_and_query_throughput() {
        use std::time::Instant;

        const DOCUMENTS: usize = 20_000;
        const BATCH: usize = 500;

        // Deterministic pseudo-random text: a fixed LCG so the corpus is the
        // same on every run and two measurements are comparable.
        let vocabulary: Vec<String> = (0..4096).map(|index| format!("term{index:04}")).collect();
        let mut seed = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let agents = ["claude", "codex", "gemini", "local"];
        let corpus: Vec<CassDocument> = (0..DOCUMENTS)
            .map(|index| {
                let content = (0..120)
                    .map(|_| {
                        vocabulary[usize::try_from(next() % 4096).expect("vocabulary index")]
                            .as_str()
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                sample_document(
                    &format!("session{:05}", index / 50),
                    u64::try_from(index).expect("ordinal"),
                    agents[index % agents.len()],
                    "measurement session",
                    &content,
                )
            })
            .collect();
        let content_bytes: usize = corpus.iter().map(|document| document.content.len()).sum();

        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let directory = tempfile::tempdir().expect("cass index directory");
            let index = crate::index::QuillIndex::create_with_schema(
                &cx,
                directory.path(),
                crate::schema::CASS_SEMANTIC_SCHEMA,
                crate::QuillConfig::default(),
            )
            .await
            .expect("create a CASS-schema index");

            let projection_start = Instant::now();
            let projected: Vec<SchemaDocument> = corpus
                .iter()
                .map(CassDocument::to_schema_document)
                .collect();
            let projection = projection_start.elapsed();

            let ingest_start = Instant::now();
            for batch in projected.chunks(BATCH) {
                index
                    .index_schema_documents(&cx, batch)
                    .await
                    .expect("ingest a CASS batch");
            }
            let accumulate = ingest_start.elapsed();
            index.commit(&cx).await.expect("publish the CASS corpus");
            let ingest_total = ingest_start.elapsed();

            let reader = crate::index::QuillSearchIndex::open_with_schema(
                &cx,
                directory.path(),
                crate::schema::CASS_SEMANTIC_SCHEMA,
                crate::QuillConfig::default(),
            )
            .await
            .expect("open a CASS-schema reader");
            let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
                .expect("build the CASS query parser");
            let filters = crate::query::CassQueryFilters::default();

            let mut latencies = Vec::new();
            let mut hits = 0_u64;
            for round in 0..500_usize {
                let raw = format!("term{:04} term{:04}", round % 4096, (round * 7) % 4096);
                let parsed = parser.parse(&raw, &filters);
                let start = Instant::now();
                let result = reader
                    .search_preparsed_paginated(&cx, &parsed.query, 10, 0, false)
                    .expect("query the CASS corpus");
                latencies.push(start.elapsed());
                hits += result.hits.len() as u64;
            }
            latencies.sort_unstable();

            let mib = content_bytes as f64 / (1024.0 * 1024.0);
            println!("--- CASS schema path, {DOCUMENTS} docs, {mib:.1} MiB content ---");
            println!(
                "projection  {:>8.2?}  ({:.0} docs/s)",
                projection,
                DOCUMENTS as f64 / projection.as_secs_f64()
            );
            println!(
                "accumulate  {:>8.2?}  ({:.0} docs/s, {:.1} MiB/s)",
                accumulate,
                DOCUMENTS as f64 / accumulate.as_secs_f64(),
                mib / accumulate.as_secs_f64()
            );
            println!(
                "ingest+commit {:>6.2?}  ({:.0} docs/s, {:.1} MiB/s)",
                ingest_total,
                DOCUMENTS as f64 / ingest_total.as_secs_f64(),
                mib / ingest_total.as_secs_f64()
            );
            println!(
                "query p50 {:>8.2?}  p99 {:>8.2?}  max {:>8.2?}  ({hits} hits over 500 queries)",
                latencies[latencies.len() / 2],
                latencies[latencies.len() * 99 / 100],
                latencies[latencies.len() - 1],
            );
        });
    }

    /// A CASS-schema index must ingest and answer queries through the same
    /// writer the shipping five-field shape uses.
    ///
    /// This is the end-to-end claim the whole schema-document path exists to
    /// support: an index whose schema is wider than `IndexableDocument` can
    /// describe still routes through Quill's admission, accumulation, commit,
    /// and search. It asserts a field-scoped match *and* a field-scoped
    /// non-match, so a lowering that ignored field scope entirely would fail
    /// here rather than pass by accident.
    #[test]
    fn cass_schema_index_ingests_and_searches_end_to_end() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("cass index directory");
            let index = crate::index::QuillIndex::create_with_schema(
                &cx,
                directory.path(),
                crate::schema::CASS_SEMANTIC_SCHEMA,
                crate::QuillConfig::default(),
            )
            .await
            .expect("create a CASS-schema index");

            let documents = [
                sample_document(
                    "alpha",
                    0,
                    "claude",
                    "Borrow checker session",
                    "the borrow checker rejected this lifetime in rustc",
                ),
                sample_document(
                    "beta",
                    1,
                    "codex",
                    "Tokenizer throughput",
                    "tokenizer throughput regressed on long input in rustc",
                ),
            ]
            .iter()
            .map(CassDocument::to_schema_document)
            .collect::<Vec<_>>();

            index
                .index_schema_documents(&cx, &documents)
                .await
                .expect("ingest CASS documents");
            index.commit(&cx).await.expect("publish CASS documents");

            // Read through the shipping reader rather than the writer handle:
            // that is the path a consumer actually uses, and it additionally
            // proves the CASS schema survives a reopen from disk.
            let reader = crate::index::QuillSearchIndex::open_with_schema(
                &cx,
                directory.path(),
                crate::schema::CASS_SEMANTIC_SCHEMA,
                crate::QuillConfig::default(),
            )
            .await
            .expect("open a CASS-schema reader");

            let parser = crate::query::CassQueryParser::new(crate::schema::CASS_SEMANTIC_SCHEMA)
                .expect("build the CASS query parser");
            let filters = crate::query::CassQueryFilters::default();

            // `lifetime` appears in exactly one document, and in neither
            // title, so a match here is attributable to the content column.
            let parsed = parser.parse("lifetime", &filters);
            let hit = reader
                .search_preparsed_paginated(&cx, &parsed.query, 10, 0, true)
                .expect("search the CASS index");
            assert_eq!(
                hit.total_count,
                Some(1),
                "exactly one document mentions a lifetime"
            );
            assert_eq!(hit.doc_count, 2, "both documents are published and live");

            // `rustc` is in both documents, so the agent filter is the only
            // thing that can change this count. Asserting all three of
            // unfiltered / matching / non-matching is what distinguishes a
            // filter that restricts from one that is silently ignored (which
            // would answer 2 every time) or always refuses (0 every time).
            let shared = parser.parse("rustc", &filters);
            let shared_hit = reader
                .search_preparsed_paginated(&cx, &shared.query, 10, 0, true)
                .expect("search the CASS index for a shared term");
            assert_eq!(
                shared_hit.total_count,
                Some(2),
                "both documents mention rustc"
            );

            let by_agent = crate::query::CassQueryFilters {
                agents: vec!["codex".to_owned()],
                ..Default::default()
            };
            let scoped = parser.parse("rustc", &by_agent);
            let scoped_hit = reader
                .search_preparsed_paginated(&cx, &scoped.query, 10, 0, true)
                .expect("search the CASS index by agent");
            assert_eq!(
                scoped_hit.total_count,
                Some(1),
                "the agent filter must restrict a term both documents carry"
            );

            let by_absent_agent = crate::query::CassQueryFilters {
                agents: vec!["nobody".to_owned()],
                ..Default::default()
            };
            let absent = parser.parse("rustc", &by_absent_agent);
            let absent_hit = reader
                .search_preparsed_paginated(&cx, &absent.query, 10, 0, true)
                .expect("search the CASS index for an absent agent");
            assert_eq!(
                absent_hit.total_count,
                Some(0),
                "an agent that produced nothing must match nothing"
            );
        });
    }

    #[test]
    fn preview_is_character_bounded_and_only_ellipsizes_when_truncating() {
        assert_eq!(cass_build_preview("abc", 400), "abc");
        assert_eq!(cass_build_preview("abcdef", 3), "abc…");
        // Exactly at the bound must NOT gain an ellipsis: a preview that
        // claims truncation it did not perform is a lie to the reader.
        assert_eq!(cass_build_preview("abc", 3), "abc");
        // Multi-byte scalars count as one character, not one byte.
        let wide = "日本語のテキスト";
        assert_eq!(cass_build_preview(wide, 3), "日本語…");
        assert_eq!(cass_build_preview("", 5), "");
    }

    #[test]
    fn prefix_source_never_splits_a_scalar_and_is_bounded() {
        assert_eq!(cass_prefix_source("abcdef", 100), "abcdef");
        assert_eq!(cass_prefix_source("abcdef", 3), "abc");
        // "日" is three bytes, so a four-byte bound must fall back to one
        // scalar rather than slicing the second one in half.
        let wide = "日本";
        let cut = cass_prefix_source(wide, 4);
        assert_eq!(cut, "日");
        assert!(wide.is_char_boundary(cut.len()));
        // Every bound from 0..=len must land on a boundary.
        for bound in 0..=wide.len() {
            let slice = cass_prefix_source(wide, bound);
            assert!(wide.is_char_boundary(slice.len()), "bound {bound}");
            assert!(slice.len() <= bound);
        }
    }

    #[test]
    fn derived_columns_come_from_the_document_and_title_absence_is_empty() {
        let document = CassDocument {
            agent: "claude".to_owned(),
            workspace: Some("/repo".to_owned()),
            workspace_original: Some("/repo".to_owned()),
            source_path: "/tmp/session.jsonl".to_owned(),
            msg_idx: 7,
            created_at: Some(1_700_000_000),
            title: Some("hello world".to_owned()),
            content: "searchable content here".to_owned(),
            source_id: "local".to_owned(),
            origin_kind: "local".to_owned(),
            origin_host: None,
            conversation_id: Some(3),
        };
        let derived = CassDerivedColumns::derive(document.as_ref());
        assert!(derived.title_prefix.contains("he"));
        assert!(derived.content_prefix.contains("se"));
        assert_eq!(derived.preview, "searchable content here");

        let untitled = CassDocument {
            title: None,
            ..document
        };
        let derived = CassDerivedColumns::derive(untitled.as_ref());
        assert!(
            derived.title_prefix.is_empty(),
            "an absent title must contribute no prefix terms, not the string \"None\""
        );
    }

    #[test]
    fn schema_sentinel_rejects_the_tantivy_generation() {
        assert!(cass_schema_hash_matches(CASS_SCHEMA_HASH));
        assert!(
            !cass_schema_hash_matches(
                "tantivy-schema-v8-hyphen-cjk-bigrams-bounded-content-prefix-preview-stored-content-external"
            ),
            "the Tantivy-era sentinel must force a rebuild, never be accepted as current"
        );
        assert!(!cass_schema_hash_matches(""));
    }

    /// The ported derivation must be byte-identical to the incumbent.
    ///
    /// `preview` and `content_prefix` are persisted columns: any divergence
    /// silently changes what a rebuilt index matches and what a user is shown,
    /// and would do so without failing anything else. Parity is checked
    /// against the shipped Tantivy implementation rather than against
    /// hand-written expectations, so the assertion cannot drift with it.
    ///
    /// Only `cass_build_preview` is reachable — the incumbent keeps its prefix
    /// walk private — so the prefix boundary rule is pinned separately by
    /// `prefix_source_never_splits_a_scalar_and_is_bounded`, which checks every
    /// bound in `0..=len` rather than sampling.
    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn derivation_is_byte_identical_to_the_tantivy_incumbent() {
        use frankensearch_lexical::cass_compat::cass_build_preview as oracle_preview;

        let samples = [
            "",
            "a",
            "short content",
            "日本語のテキストとascii mixed",
            "hyphen-joined words and CamelCase Tokens",
            &"x".repeat(10_000),
            &"日".repeat(3_000),
        ];
        for sample in samples {
            assert_eq!(
                cass_build_preview(sample, PREVIEW_MAX_CHARS),
                oracle_preview(sample, PREVIEW_MAX_CHARS),
                "preview diverged for {} bytes",
                sample.len()
            );
        }
    }

    /// Every emitted ordinal must exist in the compiled schema with a matching
    /// storage shape, and absent optionals must emit nothing at all.
    ///
    /// The ordinals are hand-written constants, so nothing else stops
    /// `field::TITLE` from drifting onto the `content` column after a schema
    /// edit — this checks each one against the descriptor by name rather than
    /// trusting the constants.
    #[test]
    fn field_values_match_the_compiled_schema_and_omit_absent_columns() {
        use crate::schema::{CASS_SEMANTIC_SCHEMA, FieldKind};

        let named = |ordinal: u16| {
            CASS_SEMANTIC_SCHEMA
                .fields
                .iter()
                .find(|field| field.id == ordinal)
                .unwrap_or_else(|| panic!("ordinal {ordinal} is absent from the CASS schema"))
        };
        assert_eq!(named(field::AGENT).name, "agent");
        assert_eq!(named(field::CONTENT).name, "content");
        assert_eq!(named(field::TITLE).name, "title");
        assert_eq!(named(field::TITLE_PREFIX).name, "title_prefix");
        assert_eq!(named(field::CONTENT_PREFIX).name, "content_prefix");
        assert_eq!(named(field::PREVIEW).name, "preview");
        assert_eq!(named(field::MSG_IDX).name, "msg_idx");
        assert_eq!(named(field::CREATED_AT).name, "created_at");
        assert_eq!(named(field::CONVERSATION_ID).name, "conversation_id");
        assert_eq!(named(field::SOURCE_PATH).name, "source_path");
        assert_eq!(named(field::WORKSPACE).name, "workspace");
        assert_eq!(named(field::WORKSPACE_ORIGINAL).name, "workspace_original");
        assert_eq!(named(field::SOURCE_ID).name, "source_id");
        assert_eq!(named(field::ORIGIN_KIND).name, "origin_kind");
        assert_eq!(named(field::ORIGIN_HOST).name, "origin_host");

        let full = CassDocument {
            agent: "claude".to_owned(),
            workspace: Some("/repo".to_owned()),
            workspace_original: Some("/Repo".to_owned()),
            source_path: "/tmp/s.jsonl".to_owned(),
            msg_idx: 7,
            created_at: Some(1_700_000_000),
            title: Some("a title".to_owned()),
            content: "body text".to_owned(),
            source_id: "local".to_owned(),
            origin_kind: "local".to_owned(),
            origin_host: Some("host".to_owned()),
            conversation_id: Some(-3),
        };
        let derived = CassDerivedColumns::derive(full.as_ref());
        let values = CassFieldValues::build(full.as_ref(), &derived);

        // Every emitted column must be a real, correctly-shaped schema field.
        for value in &values.indexed {
            let field = named(value.field_ord);
            assert!(
                matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }),
                "{} is not an indexable column",
                field.name
            );
        }
        for value in &values.numeric {
            let field = named(value.field_ord);
            assert!(
                matches!(field.kind, FieldKind::I64 { .. } | FieldKind::U64 { .. }),
                "{} is not numeric",
                field.name
            );
            assert!(
                field.kind.has_numeric_column(),
                "{} owns no numeric column, so it must be written as stored bytes",
                field.name
            );
        }
        for value in &values.stored {
            assert!(named(value.field_ord).stored, "column is not stored");
        }
        // A signed conversation id must survive as readable text.
        // Canonical little-endian, which is what Scribe validates a stored
        // numeric column against; a negative value must round-trip.
        assert_eq!(derived.conversation_id_bytes, Some((-3_i64).to_le_bytes()));
        assert_eq!(
            derived.conversation_id_bytes.map(i64::from_le_bytes),
            Some(-3)
        );

        // Absent optionals emit nothing: an empty keyword is a matchable term,
        // so writing one would make "absent" indistinguishable from "empty".
        let sparse = CassDocument {
            workspace: None,
            workspace_original: None,
            title: None,
            created_at: None,
            origin_host: None,
            conversation_id: None,
            ..full
        };
        let sparse_derived = CassDerivedColumns::derive(sparse.as_ref());
        let sparse_values = CassFieldValues::build(sparse.as_ref(), &sparse_derived);
        for absent in [
            field::WORKSPACE,
            field::TITLE,
            field::TITLE_PREFIX,
            field::ORIGIN_HOST,
        ] {
            assert!(
                !sparse_values
                    .indexed
                    .iter()
                    .any(|value| value.field_ord == absent),
                "absent optional {absent} must not be written"
            );
        }
        assert!(
            !sparse_values
                .numeric
                .iter()
                .any(|value| value.field_ord == field::CREATED_AT)
        );
        for absent in [field::WORKSPACE_ORIGINAL, field::CONVERSATION_ID] {
            assert!(
                !sparse_values
                    .stored
                    .iter()
                    .any(|value| value.field_ord == absent)
            );
        }
        // Required columns are still present in the sparse case.
        assert!(
            sparse_values
                .indexed
                .iter()
                .any(|value| value.field_ord == field::CONTENT)
        );
        assert!(
            sparse_values
                .numeric
                .iter()
                .any(|value| value.field_ord == field::MSG_IDX)
        );
    }

    /// Identity must discriminate source, conversation, AND ordinal.
    ///
    /// The conversation axis is the one that matters most here: every locally
    /// discovered conversation shares one `source_id`, so an identity built
    /// only from source and ordinal collides on message 0 of every local
    /// conversation. Tantivy tolerated that; Quill refuses it as a duplicate
    /// primary key, which is exactly the bug this shape prevents.
    #[test]
    fn document_identity_discriminates_source_conversation_and_ordinal() {
        use CassConversationKey as Key;
        assert_eq!(
            cass_document_identity("local", Key::new("/a.jsonl", Some(7)), 0),
            "local#/a.jsonl#7#0"
        );

        assert_ne!(
            cass_document_identity("local", Key::new("/a.jsonl", Some(1)), 1),
            cass_document_identity("remote", Key::new("/a.jsonl", Some(1)), 1),
            "identity must not collide across sources"
        );
        assert_ne!(
            cass_document_identity("local", Key::new("/a.jsonl", Some(1)), 0),
            cass_document_identity("local", Key::new("/b.jsonl", Some(1)), 0),
            "message 0 of two conversations in ONE source, sharing a reused id, must not collide"
        );
        assert_ne!(
            cass_document_identity("local", Key::new("/a.jsonl", Some(1)), 0),
            cass_document_identity("local", Key::new("/a.jsonl", Some(1)), 1),
            "identity must not collide across ordinals"
        );
        // Without an assigned conversation id the transcript path carries the
        // same discrimination.
        assert_ne!(
            cass_document_identity("local", Key::new("/a.jsonl", None), 0),
            cass_document_identity("local", Key::new("/b.jsonl", None), 0),
            "message 0 of two unassigned conversations must not collide"
        );
    }

    /// A batch of conversations from one source must be ingestable.
    ///
    /// This is the regression test for the real defect: under the old
    /// source-scoped identity every conversation's message 0 was `local#0`, so
    /// a multi-conversation local batch was refused outright.
    #[test]
    fn a_multi_conversation_local_batch_has_unique_identities() {
        let documents: Vec<CassDocument> = (0..4)
            .flat_map(|conversation| {
                (0..3).map(move |msg_idx| {
                    let mut document = sample_document("local", msg_idx, "claude", "t", "body");
                    document.conversation_id = Some(conversation);
                    document
                })
            })
            .collect();
        let identities: std::collections::BTreeSet<String> = documents
            .iter()
            .map(|document| document.to_schema_document().id)
            .collect();
        assert_eq!(
            identities.len(),
            documents.len(),
            "every message in a multi-conversation local batch needs its own identity"
        );
    }
}
