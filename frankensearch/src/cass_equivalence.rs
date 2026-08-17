//! Engine-equivalence comparator for the CASS lexical profile.
//!
//! A consumer migrating off the Tantivy CASS backend onto Quill needs to know
//! one thing before it ships: does the new engine return the same answers over
//! *its own* corpus? A synthetic test cannot answer that. This module is the
//! reusable half of the differential, so the same comparison can run against a
//! real corpus exported from a consumer's database.
//!
//! # Why this is a dev surface
//!
//! Answering the question requires both engines in one process. A published
//! consumer cannot contain Tantivy at all — `cargo publish` rejects a git
//! dependency even when it is optional — so this comparison is necessarily a
//! pre-release gate rather than a runtime fallback. It lives behind the
//! `cass-equivalence` feature and is never part of a shipping build.
//!
//! # What is compared, and what is deliberately not
//!
//! Compared: the exact match count, and the returned identifier *set* at a
//! limit far above any expected match count. Two capped hit lists only measure
//! where each engine cut its list, so a membership claim assembled from them is
//! not a membership claim at all — the count is the uncapped signal, and an
//! uncapped limit is what makes the set comparison meaningful.
//!
//! Not compared: rank order and score. The two engines do not share a BM25
//! accumulation order, and asserting a permutation here would encode the
//! incumbent's tie-breaking as a Quill requirement rather than testing what a
//! consumer depends on. A consumer that genuinely depends on exact ordering
//! needs a different, narrower proof than this one.

use std::collections::BTreeSet;

use frankensearch_lexical as tantivy_cass;
use frankensearch_quill as quill;

/// Hit limit used for every probe.
///
/// Far above any expected match count so the set comparison is about
/// membership rather than truncation.
pub const UNCAPPED_LIMIT: usize = 100_000;

/// Cutoff tie-group expansion budget for the incumbent observation.
pub const TIE_EXPANSION: usize = 256;

/// One probe's outcome across both engines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassProbeComparison {
    /// The raw query string, exactly as a consumer would type it.
    pub query: String,
    /// Exact match count reported by the Tantivy incumbent.
    pub incumbent_total: usize,
    /// Exact match count reported by Quill.
    pub quill_total: usize,
    /// Identifiers the incumbent returned and Quill did not.
    pub only_incumbent: Vec<String>,
    /// Identifiers Quill returned and the incumbent did not.
    pub only_quill: Vec<String>,
}

impl CassProbeComparison {
    /// Whether both engines agreed on count and membership.
    #[must_use]
    pub fn agrees(&self) -> bool {
        self.incumbent_total == self.quill_total
            && self.only_incumbent.is_empty()
            && self.only_quill.is_empty()
    }
}

/// The full comparison across a corpus and a probe set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassEquivalenceReport {
    /// Live document count in the incumbent index.
    pub incumbent_doc_count: usize,
    /// Live document count in the Quill index.
    pub quill_doc_count: usize,
    /// Every probe, in the order supplied.
    pub probes: Vec<CassProbeComparison>,
}

impl CassEquivalenceReport {
    /// Probes on which the engines disagreed.
    #[must_use]
    pub fn divergences(&self) -> Vec<&CassProbeComparison> {
        self.probes.iter().filter(|probe| !probe.agrees()).collect()
    }

    /// Probes for which the incumbent matched at least one document.
    ///
    /// A corpus on which every probe returns nothing would make the comparison
    /// vacuous — it would agree perfectly while proving nothing — so a caller
    /// gating on this report must check that this is meaningfully nonzero.
    #[must_use]
    pub fn discriminating_probes(&self) -> usize {
        self.probes
            .iter()
            .filter(|probe| probe.incumbent_total > 0)
            .count()
    }

    /// Whether the engines agreed on document count and every probe.
    #[must_use]
    pub fn equivalent(&self) -> bool {
        self.incumbent_doc_count == self.quill_doc_count && self.divergences().is_empty()
    }

    /// Human-readable divergence table, empty when equivalent.
    #[must_use]
    pub fn render_divergences(&self) -> String {
        self.divergences()
            .iter()
            .map(|probe| {
                format!(
                    "  {:>32}  incumbent={:<6} quill={:<6} only_incumbent={:?} only_quill={:?}",
                    probe.query,
                    probe.incumbent_total,
                    probe.quill_total,
                    probe.only_incumbent.iter().take(4).collect::<Vec<_>>(),
                    probe.only_quill.iter().take(4).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Probe set covering the query classes the CASS profile actually supports.
///
/// The incumbent's own rebuild evidence uses five probes; that is far too thin
/// to gate an engine swap. Every distinct lowering path deserves a probe,
/// because a class with no probe is a class where a divergence ships silently.
/// `terms` supplies corpus-specific vocabulary so the same probe shapes can run
/// against a synthetic corpus or a real one.
#[must_use]
pub fn default_probe_set(terms: &[String]) -> Vec<String> {
    let pick = |index: usize| -> &str {
        if terms.is_empty() {
            "term"
        } else {
            terms[index % terms.len()].as_str()
        }
    };
    let mut probes = Vec::new();

    // Single terms: the plainest posting-list read.
    for index in 0..12 {
        probes.push(pick(index * 7).to_owned());
    }
    // Multi-term, no explicit operator: cass's default combination semantics.
    for index in 0..6 {
        probes.push(format!("{} {}", pick(index), pick(index * 13 + 1)));
    }
    // Explicit boolean lowerings, each a distinct code path.
    for index in 0..6 {
        probes.push(format!("{} AND {}", pick(index), pick(index * 5 + 2)));
    }
    for index in 0..6 {
        probes.push(format!("{} OR {}", pick(index), pick(index * 11 + 3)));
    }
    for index in 0..4 {
        probes.push(format!("{} NOT {}", pick(index), pick(index * 3 + 4)));
    }
    // Quoted phrase: positions, not just membership.
    for index in 0..3 {
        probes.push(format!("\"{} {}\"", pick(index), pick(index + 1)));
    }
    // Prefix/wildcard: exercises the edge-ngram prefix columns.
    for index in 0..4 {
        let term = pick(index * 9);
        let cut = term.len().min(4);
        probes.push(format!("{}*", &term[..cut]));
    }
    // Hyphenated input: CassHyphenNormalize is an entire analyzer of its own,
    // so leaving it unprobed would leave a whole tokenizer untested.
    probes.push("well-known".to_owned());
    probes.push("multi-part-token".to_owned());
    // CJK: the schema advertises CJK bigrams.
    probes.push("日本語".to_owned());
    probes.push("検索".to_owned());
    // Case and punctuation normalization.
    probes.push(pick(1).to_uppercase());
    // A guaranteed miss, so the comparison includes the empty-result path.
    probes.push("zzzznonexistenttokenzzzz".to_owned());

    probes
}

/// Build both engines over `documents` and compare them across `queries`.
///
/// `documents` is the shared corpus; each is projected into both engines'
/// document shapes from the same source values, so any divergence is
/// attributable to the engines rather than to two different corpora.
///
/// # Errors
///
/// Returns a boxed error if either engine fails to build, ingest, commit, or
/// query. A failure here is an infrastructure failure, not a divergence —
/// divergences are reported in the returned value.
///
/// # Panics
///
/// Panics if a document ordinal does not fit the index's integer types, which
/// cannot happen for any corpus small enough to hold in memory.
pub async fn cass_engine_equivalence_report(
    cx: &asupersync::Cx,
    documents: &[quill::cass::CassDocument],
    queries: &[String],
) -> Result<CassEquivalenceReport, Box<dyn std::error::Error + Send + Sync>> {
    // ---- incumbent -------------------------------------------------------
    let mut oracle = tantivy_cass::CassTantivyIndex::in_memory_single_threaded_oracle()?;
    let incumbent_documents: Vec<tantivy_cass::CassDocument> = documents
        .iter()
        .map(|document| tantivy_cass::CassDocument {
            agent: document.agent.clone(),
            workspace: document.workspace.clone(),
            workspace_original: document.workspace_original.clone(),
            source_path: document.source_path.clone(),
            msg_idx: document.msg_idx,
            created_at: document.created_at,
            title: document.title.clone(),
            content: document.content.clone(),
            source_id: document.source_id.clone(),
            origin_kind: document.origin_kind.clone(),
            origin_host: document.origin_host.clone(),
            conversation_id: document.conversation_id,
        })
        .collect();
    oracle.add_cass_documents(&incumbent_documents)?;
    oracle.commit()?;

    // ---- Quill -----------------------------------------------------------
    let directory = tempfile::tempdir()?;
    let index = quill::QuillIndex::create_with_schema(
        cx,
        directory.path(),
        quill::schema::CASS_SEMANTIC_SCHEMA,
        quill::QuillConfig::default(),
    )
    .await?;
    let projected: Vec<quill::SchemaDocument> = documents
        .iter()
        .map(quill::cass::CassDocument::to_schema_document)
        .collect();
    index.index_schema_documents(cx, &projected).await?;
    index.commit(cx).await?;

    let reader = quill::QuillSearchIndex::open_with_schema(
        cx,
        directory.path(),
        quill::schema::CASS_SEMANTIC_SCHEMA,
        quill::QuillConfig::default(),
    )
    .await?;
    let parser = quill::query::CassQueryParser::new(quill::schema::CASS_SEMANTIC_SCHEMA)?;

    // ---- compare ---------------------------------------------------------
    let mut probes = Vec::with_capacity(queries.len());
    let mut incumbent_doc_count = 0_usize;

    for raw in queries {
        let observed = oracle.cass_oracle_observe_query(
            raw,
            &tantivy_cass::CassQueryFilters::default(),
            UNCAPPED_LIMIT,
            TIE_EXPANSION,
        )?;
        incumbent_doc_count = observed.doc_count;

        let parsed = parser.parse(raw, &quill::query::CassQueryFilters::default());
        let result =
            reader.search_preparsed_paginated(cx, &parsed.query, UNCAPPED_LIMIT, 0, true)?;

        let incumbent_ids: BTreeSet<&str> =
            observed.hits.iter().map(|hit| hit.doc_id.as_str()).collect();
        let quill_ids: BTreeSet<&str> = result
            .hits
            .iter()
            .map(|hit| hit.document_id.as_str())
            .collect();

        probes.push(CassProbeComparison {
            query: raw.clone(),
            incumbent_total: observed.total_count,
            quill_total: usize::try_from(result.total_count.unwrap_or_default())
                .expect("exact match count fits usize"),
            only_incumbent: incumbent_ids
                .difference(&quill_ids)
                .map(|id| (*id).to_owned())
                .collect(),
            only_quill: quill_ids
                .difference(&incumbent_ids)
                .map(|id| (*id).to_owned())
                .collect(),
        });
    }

    Ok(CassEquivalenceReport {
        incumbent_doc_count,
        quill_doc_count: usize::try_from(reader.doc_count()?).expect("doc count fits usize"),
        probes,
    })
}

/// Read a CASS corpus from newline-delimited JSON.
///
/// This is what lets the comparison run against a consumer's real corpus: it
/// exports its documents once, and the same gate that runs in CI on a synthetic
/// corpus runs unchanged on the real one. Each line is one document with the
/// field names of [`quill::cass::CassDocument`].
///
/// # Errors
///
/// Returns an error if the file cannot be read or a line is not a valid
/// document. The failing line number is included, because a corpus export with
/// one bad row should say which row.
pub fn load_cass_corpus_jsonl(
    path: &std::path::Path,
) -> Result<Vec<quill::cass::CassDocument>, Box<dyn std::error::Error + Send + Sync>> {
    use std::io::BufRead as _;

    let file = std::fs::File::open(path)?;
    let mut documents = Vec::new();
    for (offset, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: CassCorpusRow = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{}: {error}", path.display(), offset + 1))?;
        documents.push(row.into_document());
    }
    Ok(documents)
}

/// Serde shape for one exported corpus row.
///
/// Separate from [`quill::cass::CassDocument`] so the ingest type does not have
/// to carry a serde dependency for a dev-only exchange format.
#[derive(serde::Deserialize)]
struct CassCorpusRow {
    agent: String,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    workspace_original: Option<String>,
    source_path: String,
    msg_idx: u64,
    #[serde(default)]
    created_at: Option<i64>,
    #[serde(default)]
    title: Option<String>,
    content: String,
    source_id: String,
    origin_kind: String,
    #[serde(default)]
    origin_host: Option<String>,
    #[serde(default)]
    conversation_id: Option<i64>,
}

impl CassCorpusRow {
    fn into_document(self) -> quill::cass::CassDocument {
        quill::cass::CassDocument {
            agent: self.agent,
            workspace: self.workspace,
            workspace_original: self.workspace_original,
            source_path: self.source_path,
            msg_idx: self.msg_idx,
            created_at: self.created_at,
            title: self.title,
            content: self.content,
            source_id: self.source_id,
            origin_kind: self.origin_kind,
            origin_host: self.origin_host,
            conversation_id: self.conversation_id,
        }
    }
}
