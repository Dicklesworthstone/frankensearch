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

/// Canonical CASS document identity, `"{source_id}#{msg_idx}"`.
#[must_use]
pub fn cass_document_identity(source_id: &str, msg_idx: u64) -> String {
    format!("{source_id}#{msg_idx}")
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
    /// `conversation_id` rendered for the stored column.
    ///
    /// The field is `I64 { indexed: false, fast: false }` with `stored: true`,
    /// so it owns no numeric column and reaches disk as opaque bytes. This
    /// records the encoding as base-10 text rather than raw little-endian:
    /// the bytes are only ever read back by a consumer that must agree with
    /// the writer, and a self-describing form is one that a human debugging a
    /// stored payload can actually read. It is owned here so the borrowed
    /// value slices have something to point at.
    pub conversation_id_text: Option<String>,
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
            conversation_id_text: document.conversation_id.map(|id| id.to_string()),
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
        if let Some(conversation_id) = derived.conversation_id_text.as_deref() {
            stored.push(StoredFieldValue::new(
                field::CONVERSATION_ID,
                conversation_id.as_bytes(),
            ));
        }

        Self {
            indexed,
            numeric,
            stored,
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
        assert_eq!(derived.conversation_id_text.as_deref(), Some("-3"));

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

    #[test]
    fn document_identity_is_source_scoped() {
        assert_eq!(cass_document_identity("local", 0), "local#0");
        assert_ne!(
            cass_document_identity("local", 1),
            cass_document_identity("remote", 1),
            "identity must not collide across sources at the same ordinal"
        );
    }
}
