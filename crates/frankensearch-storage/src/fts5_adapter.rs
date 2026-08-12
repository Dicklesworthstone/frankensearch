//! FTS5 alternative lexical search adapter.
//!
//! Uses `FrankenSQLite`'s built-in FTS5 implementation as an alternative to
//! Tantivy for BM25 full-text search. Both implement the split
//! [`frankensearch_core::LexicalRead`] / [`frankensearch_core::LexicalWrite`]
//! trait from `frankensearch-core`.
//!
//! # Advantages over Tantivy
//!
//! - Zero additional binary size (FTS5 is part of `FrankenSQLite`)
//! - MVCC concurrent reads and writes
//! - Single deployment artifact (one `.db` file)
//!
//! # Content mode
//!
//! The in-memory adapter below supports `Stored` and `Contentless` modes.
//! Persisted tables are inspected from their SQLite metadata so that `Stored`,
//! external-content, and contentless layouts are never inferred from caller
//! configuration.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use asupersync::Cx;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::SearchFuture;
use frankensearch_core::types::{IndexableDocument, ScoreSource, ScoredResult};
use fsqlite::{AsyncConnection, Row};
use fsqlite_ext_fts5::{Fts5Table, snippet as fts5_snippet};
use fsqlite_types::cx::Cx as FsqliteCx;
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::connection::{Storage, fsqlite_cx, map_storage_error_at};
use crate::schema::PORTER_FTS5_REBUILD_TABLE;

// ─── Constants ──────────────────────────────────────────────────────────────

/// BM25 boost applied to title field matches (mirrors Tantivy adapter).
const TITLE_BOOST: f64 = 2.0;

/// Maximum query length in characters before truncation.
const MAX_QUERY_LENGTH: usize = 10_000;

/// Default snippet window size in tokens.
const DEFAULT_SNIPPET_TOKENS: usize = 20;

/// The on-disk marker written after applying the 0.2.1 Porter rebuild.
///
/// FrankenSQLite 0.2.1 changes Porter token handling. A prior Porter index
/// must be rebuilt from its complete content source; accepting an unmarked
/// table would make terms silently unfindable.
pub const PORTER_FTS5_REBUILD_VERSION: i64 = 1;

/// Column index: content (primary search field).
const COL_CONTENT: usize = 2;
/// Column index: `metadata_json` (stored, not searched).
const COL_METADATA: usize = 3;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Content storage mode for the FTS5 index.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fts5ContentMode {
    /// FTS5 stores its own copy of the content (supports snippets).
    #[default]
    Stored,
    /// FTS5 indexes a separately governed content table.
    External,
    /// Index-only mode — no content retrieval or snippet support.
    Contentless,
}

/// Tokenizer selection for the FTS5 index.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fts5TokenizerChoice {
    /// Unicode-aware tokenizer with optional diacritic removal.
    #[default]
    Unicode61,
    /// English Porter stemming (wraps unicode61).
    Porter,
    /// Trigram tokenizer for substring matching (slower but more flexible).
    Trigram,
}

/// Configuration for the FTS5 lexical search adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fts5AdapterConfig {
    /// Content storage mode.
    #[serde(default)]
    pub content_mode: Fts5ContentMode,
    /// Tokenizer to use.
    #[serde(default)]
    pub tokenizer: Fts5TokenizerChoice,
    /// BM25 boost for title field matches.
    #[serde(default = "default_title_boost")]
    pub title_boost: f64,
}

fn default_title_boost() -> f64 {
    TITLE_BOOST
}

impl Default for Fts5AdapterConfig {
    fn default() -> Self {
        Self {
            content_mode: Fts5ContentMode::default(),
            tokenizer: Fts5TokenizerChoice::default(),
            title_boost: TITLE_BOOST,
        }
    }
}

// ─── Row ID mapping ─────────────────────────────────────────────────────────

/// Maps between string `doc_ids` and i64 rowids required by `Fts5Table`.
#[derive(Debug, Default)]
struct RowIdMap {
    doc_to_row: HashMap<String, i64>,
    row_to_doc: HashMap<i64, String>,
    next_rowid: i64,
}

impl RowIdMap {
    fn new() -> Self {
        Self {
            doc_to_row: HashMap::new(),
            row_to_doc: HashMap::new(),
            next_rowid: 1,
        }
    }

    fn get_or_assign(&mut self, doc_id: &str) -> i64 {
        if let Some(&rowid) = self.doc_to_row.get(doc_id) {
            return rowid;
        }
        let rowid = self.next_rowid;
        self.next_rowid += 1;
        self.doc_to_row.insert(doc_id.to_owned(), rowid);
        self.row_to_doc.insert(rowid, doc_id.to_owned());
        rowid
    }

    fn get_rowid(&self, doc_id: &str) -> Option<i64> {
        self.doc_to_row.get(doc_id).copied()
    }

    fn get_doc_id(&self, rowid: i64) -> Option<&str> {
        self.row_to_doc.get(&rowid).map(String::as_str)
    }

    fn remove(&mut self, doc_id: &str) -> Option<i64> {
        if let Some(rowid) = self.doc_to_row.remove(doc_id) {
            self.row_to_doc.remove(&rowid);
            Some(rowid)
        } else {
            None
        }
    }
}

// ─── FTS5 Lexical Search ────────────────────────────────────────────────────

/// FTS5-backed implementation of the split lexical capabilities.
///
/// Uses `FrankenSQLite`'s `Fts5Table` directly for full-text indexing
/// and BM25-ranked search. Thread-safe via internal `Mutex`.
pub struct Fts5LexicalSearch {
    table: Mutex<Fts5Table>,
    rowid_map: Mutex<RowIdMap>,
    config: Fts5AdapterConfig,
    doc_count: AtomicUsize,
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Fts5LexicalSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fts5LexicalSearch")
            .field("config", &self.config)
            .field("doc_count", &self.doc_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl Fts5LexicalSearch {
    /// Create a new FTS5 lexical search instance.
    #[must_use]
    pub fn new(config: Fts5AdapterConfig) -> Self {
        let columns = vec![
            "doc_id".to_owned(),
            "title".to_owned(),
            "content".to_owned(),
            "metadata_json".to_owned(),
        ];

        let table = Fts5Table::with_columns(columns);

        Self {
            table: Mutex::new(table),
            rowid_map: Mutex::new(RowIdMap::new()),
            config,
            doc_count: AtomicUsize::new(0),
        }
    }

    /// Create a new FTS5 lexical search instance with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(Fts5AdapterConfig::default())
    }

    /// Get the adapter configuration.
    #[must_use]
    pub fn config(&self) -> &Fts5AdapterConfig {
        &self.config
    }

    /// Truncate overly long queries to prevent pathological parsing.
    fn truncate_query(query: &str) -> &str {
        if query.len() <= MAX_QUERY_LENGTH {
            return query;
        }

        let Some((end, _)) = query.char_indices().nth(MAX_QUERY_LENGTH) else {
            return query;
        };
        warn!(
            original_len_bytes = query.len(),
            max_chars = MAX_QUERY_LENGTH,
            "fts5: query truncated"
        );
        &query[..end]
    }

    /// Build column values from an `IndexableDocument`.
    fn doc_to_columns(doc: &IndexableDocument) -> Vec<String> {
        let metadata_json = if doc.metadata.is_empty() {
            String::new()
        } else {
            serde_json::to_string(&doc.metadata).unwrap_or_default()
        };

        vec![
            doc.id.clone(),
            doc.title.clone().unwrap_or_default(),
            doc.content.clone(),
            metadata_json,
        ]
    }

    /// Search with snippet generation (richer result type).
    #[allow(clippy::significant_drop_tightening)]
    pub fn search_with_snippets(&self, query: &str, limit: usize) -> SearchResult<Vec<Fts5Hit>> {
        let query = Self::truncate_query(query);
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let table = self.table.lock().map_err(lock_error)?;
        let rowid_map = self.rowid_map.lock().map_err(lock_error)?;

        let search_results = table
            .search(query)
            .map_err(|e| SearchError::QueryParseError {
                query: query.to_owned(),
                detail: e.to_string(),
            })?;

        let query_terms: Vec<String> = query
            .split_whitespace()
            .map(|t| t.trim_matches('"').to_lowercase())
            .collect();

        let mut hits = Vec::with_capacity(search_results.len().min(limit));
        for (rank, (rowid, score)) in search_results.into_iter().take(limit).enumerate() {
            let doc_id = rowid_map.get_doc_id(rowid).unwrap_or("").to_owned();

            // FTS5 scores are negative (lower = better). Negate for positive.
            #[allow(clippy::cast_possible_truncation)]
            let bm25_score = (-score) as f32;

            // Generate snippet from content column if available.
            let snippet = table
                .get_document(rowid)
                .and_then(|cols| cols.get(COL_CONTENT))
                .map(|content| {
                    fts5_snippet(
                        content,
                        &query_terms,
                        "<b>",
                        "</b>",
                        "...",
                        DEFAULT_SNIPPET_TOKENS,
                    )
                });

            let metadata = table
                .get_document(rowid)
                .and_then(|cols| cols.get(COL_METADATA))
                .filter(|s| !s.is_empty())
                .and_then(|s| serde_json::from_str(s).ok());

            hits.push(Fts5Hit {
                doc_id,
                bm25_score,
                rank,
                snippet,
                metadata,
            });
        }

        debug!(hits = hits.len(), query, "fts5 search completed");
        Ok(hits)
    }

    /// Delete a single document by ID.
    ///
    /// Returns `true` if the document existed and was removed.
    pub fn delete_document(&self, doc_id: &str) -> SearchResult<bool> {
        let mut table = self.table.lock().map_err(lock_error)?;
        let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

        let Some(rowid) = rowid_map.remove(doc_id) else {
            return Ok(false);
        };
        table.delete_document(rowid);
        drop(rowid_map);
        drop(table);
        self.doc_count.fetch_sub(1, Ordering::Relaxed);
        Ok(true)
    }

    /// Delete all indexed documents.
    pub fn clear(&self) -> SearchResult<()> {
        let mut table = self.table.lock().map_err(lock_error)?;
        let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

        // Collect all rowids to delete.
        let rowids: Vec<i64> = rowid_map.row_to_doc.keys().copied().collect();
        for rowid in rowids {
            table.delete_document(rowid);
        }
        rowid_map.doc_to_row.clear();
        rowid_map.row_to_doc.clear();
        drop(rowid_map);
        drop(table);
        self.doc_count.store(0, Ordering::Relaxed);

        debug!("fts5: cleared all documents");
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PersistedFts5Metadata {
    content_mode: Fts5ContentMode,
    tokenizer: String,
}

/// A read-only, persisted FTS5 search path.
///
/// Unlike [`Fts5LexicalSearch`], this reader executes `MATCH` against the
/// virtual table in the supplied [`Storage`] database. It revalidates the
/// table DDL and the governed rebuild marker on every open and search, so an
/// unrebuilt Porter index cannot be queried through an in-memory side path.
/// The table must expose `doc_id` and `metadata_json` columns; those are the
/// persisted storage contract required to produce `ScoredResult` values.
pub struct PersistedFts5LexicalSearch {
    storage: Arc<Storage>,
    table_name: String,
    config: Fts5AdapterConfig,
    doc_count: usize,
}

impl std::fmt::Debug for PersistedFts5LexicalSearch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PersistedFts5LexicalSearch")
            .field("table_name", &self.table_name)
            .field("config", &self.config)
            .field("doc_count", &self.doc_count)
            .finish_non_exhaustive()
    }
}

impl PersistedFts5LexicalSearch {
    /// Open a governed persisted Porter FTS5 table.
    ///
    /// This checks the table's actual `sqlite_master` DDL rather than trusting
    /// an application-supplied content mode or tokenizer. It also requires the
    /// committed rebuild marker before returning a searchable reader.
    pub async fn open(cx: &Cx, storage: Arc<Storage>, table_name: &str) -> SearchResult<Self> {
        let table_name = validated_fts5_identifier(table_name)?;
        let fsqlite_cx = fsqlite_cx(cx);
        let metadata =
            ensure_porter_fts5_ready(&fsqlite_cx, storage.connection(), &table_name).await?;
        let doc_count =
            persisted_fts5_doc_count(&fsqlite_cx, storage.connection(), &table_name).await?;

        Ok(Self {
            storage,
            table_name,
            config: Fts5AdapterConfig {
                content_mode: metadata.content_mode,
                tokenizer: Fts5TokenizerChoice::Porter,
                title_boost: TITLE_BOOST,
            },
            doc_count,
        })
    }

    /// Return the verified table configuration read from persisted metadata.
    #[must_use]
    pub fn config(&self) -> &Fts5AdapterConfig {
        &self.config
    }
}

impl frankensearch_core::LexicalRead for PersistedFts5LexicalSearch {
    #[instrument(skip_all, fields(table = %self.table_name, query = %query, limit = limit))]
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let query = Fts5LexicalSearch::truncate_query(query);
            if query.trim().is_empty() {
                return Ok(Vec::new());
            }

            let fsqlite_cx = fsqlite_cx(cx);
            // This is intentionally in the live data path, not only in the
            // rebuild helper: external DDL or marker changes fail the search
            // closed before MATCH can return a stale Porter result.
            ensure_porter_fts5_ready(&fsqlite_cx, self.storage.connection(), &self.table_name)
                .await?;

            let limit = i64::try_from(limit).map_err(|_| SearchError::InvalidConfig {
                field: "fts5.limit".to_owned(),
                value: limit.to_string(),
                reason: "does not fit SQLite's signed integer limit".to_owned(),
            })?;
            let params = [
                SqliteValue::Text(query.to_owned().into()),
                SqliteValue::Integer(limit),
            ];
            let rows = self
                .storage
                .connection()
                .query_with_params(
                    &fsqlite_cx,
                    &format!(
                        "SELECT doc_id, metadata_json, bm25({0}) FROM {0} \
                         WHERE {0} MATCH ?1 ORDER BY bm25({0}), rowid LIMIT ?2;",
                        self.table_name
                    ),
                    &params,
                )
                .await
                .map_err(|error| map_storage_error_at("persisted Porter FTS5 search", error))?;

            rows.iter()
                .map(decode_persisted_fts5_row)
                .collect::<SearchResult<Vec<_>>>()
        })
    }

    fn doc_count(&self) -> usize {
        self.doc_count
    }
}

/// Rebuild a persisted Porter FTS5 table for FrankenSQLite 0.3.
///
/// The table's own DDL decides its content mode. Ordinary stored and external
/// tables use FTS5's `rebuild` command; contentless tables are rejected because
/// authoritative text and original rowids must be re-ingested instead.
///
/// Rebuild and marker promotion share one synchronous worker transaction. The
/// 0.3 synchronous cleanup path is independent of request cancellation, so an
/// ordinary error, failed commit, or panic always attempts exactly one rollback
/// before the original error or panic is returned to the caller.
pub async fn rebuild_porter_fts5_table(
    cx: &Cx,
    conn: &AsyncConnection,
    table_name: &str,
) -> SearchResult<()> {
    let table_name = validated_fts5_identifier(table_name)?;
    let fsqlite_cx = fsqlite_cx(cx);
    let metadata = read_persisted_fts5_metadata(&fsqlite_cx, conn, &table_name).await?;
    ensure_rebuildable_porter_fts5(&table_name, &metadata)?;

    match read_porter_fts5_rebuild_marker(&fsqlite_cx, conn, &table_name).await? {
        Some(PORTER_FTS5_REBUILD_VERSION) => return Ok(()),
        Some(version) if version > PORTER_FTS5_REBUILD_VERSION => {
            return Err(SearchError::InvalidConfig {
                field: "fts5.rebuild_version".to_owned(),
                value: version.to_string(),
                reason:
                    "database was rebuilt by a newer Porter FTS5 migration; refusing a downgrade"
                        .to_owned(),
            });
        }
        Some(_) | None => {}
    }

    conn.execute_sync("BEGIN IMMEDIATE;")
        .map_err(|error| map_storage_error_at("begin Porter FTS5 rebuild", error))?;

    let result = catch_unwind(AssertUnwindSafe(|| -> SearchResult<()> {
        let rebuild_sql = format!("INSERT INTO {table_name}({table_name}) VALUES ('rebuild');");
        conn.execute_sync(&rebuild_sql)
            .map_err(|error| map_storage_error_at("rebuild Porter FTS5 table", error))?;

        let params = [
            SqliteValue::Text(table_name.clone().into()),
            SqliteValue::Integer(PORTER_FTS5_REBUILD_VERSION),
        ];
        conn.execute_with_params_sync(
            &format!(
                "INSERT INTO {PORTER_FTS5_REBUILD_TABLE} (table_name, rebuild_version) \
                 VALUES (?1, ?2) \
                 ON CONFLICT(table_name) DO UPDATE SET rebuild_version = excluded.rebuild_version;"
            ),
            &params,
        )
        .map_err(|error| map_storage_error_at("write Porter FTS5 rebuild marker", error))?;
        Ok(())
    }));

    match result {
        Ok(Ok(())) => conn.commit_transaction_sync().map_err(|commit_error| {
            if let Err(rollback_error) = conn.rollback_transaction_sync() {
                warn!(
                    error = %rollback_error,
                    "rollback failed after Porter FTS5 rebuild commit error"
                );
            }
            map_storage_error_at("commit Porter FTS5 rebuild", commit_error)
        }),
        Ok(Err(error)) => {
            if let Err(rollback_error) = conn.rollback_transaction_sync() {
                warn!(
                    error = %rollback_error,
                    "rollback failed after Porter FTS5 rebuild error"
                );
            }
            Err(error)
        }
        Err(payload) => {
            if let Err(rollback_error) = conn.rollback_transaction_sync() {
                warn!(
                    error = %rollback_error,
                    "rollback failed during Porter FTS5 rebuild panic recovery"
                );
            }
            resume_unwind(payload);
        }
    }
}

fn validated_fts5_identifier(table_name: &str) -> SearchResult<String> {
    let mut chars = table_name.chars();
    let valid_start = chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic());
    let valid_rest = chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric());
    if valid_start && valid_rest {
        Ok(table_name.to_owned())
    } else {
        Err(SearchError::InvalidConfig {
            field: "fts5.table_name".to_owned(),
            value: table_name.to_owned(),
            reason: "must be a SQLite ASCII identifier before it is interpolated into a rebuild statement".to_owned(),
        })
    }
}

async fn ensure_porter_fts5_ready(
    fsqlite_cx: &FsqliteCx,
    conn: &AsyncConnection,
    table_name: &str,
) -> SearchResult<PersistedFts5Metadata> {
    let metadata = read_persisted_fts5_metadata(fsqlite_cx, conn, table_name).await?;
    ensure_rebuildable_porter_fts5(table_name, &metadata)?;

    match read_porter_fts5_rebuild_marker(fsqlite_cx, conn, table_name).await? {
        Some(PORTER_FTS5_REBUILD_VERSION) => Ok(metadata),
        Some(version) if version > PORTER_FTS5_REBUILD_VERSION => Err(SearchError::InvalidConfig {
            field: "fts5.rebuild_version".to_owned(),
            value: version.to_string(),
            reason: "database was rebuilt by a newer Porter FTS5 migration; refusing a downgrade".to_owned(),
        }),
        Some(version) => Err(SearchError::InvalidConfig {
            field: "fts5.rebuild_version".to_owned(),
            value: version.to_string(),
            reason: "Porter FTS5 table has an obsolete rebuild marker; rebuild must complete and commit before search".to_owned(),
        }),
        None => Err(SearchError::InvalidConfig {
            field: "fts5.rebuild_version".to_owned(),
            value: table_name.to_owned(),
            reason: "Porter FTS5 table has no committed rebuild marker; refusing potentially stale search results".to_owned(),
        }),
    }
}

async fn read_persisted_fts5_metadata(
    fsqlite_cx: &FsqliteCx,
    conn: &AsyncConnection,
    table_name: &str,
) -> SearchResult<PersistedFts5Metadata> {
    let params = [SqliteValue::Text(table_name.to_owned().into())];
    let rows = conn
        .query_with_params(
            fsqlite_cx,
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?1;",
            &params,
        )
        .await
        .map_err(|error| map_storage_error_at("read persisted FTS5 table metadata", error))?;

    let [row] = rows.as_slice() else {
        return Err(persisted_fts5_metadata_error(
            table_name,
            if rows.is_empty() {
                "table is absent from sqlite_master"
            } else {
                "sqlite_master returned more than one table definition"
            },
        ));
    };
    let Some(SqliteValue::Text(sql)) = row.get(0) else {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "sqlite_master.sql is not text",
        ));
    };

    parse_persisted_fts5_metadata(table_name, sql.as_ref())
}

async fn read_porter_fts5_rebuild_marker(
    fsqlite_cx: &FsqliteCx,
    conn: &AsyncConnection,
    table_name: &str,
) -> SearchResult<Option<i64>> {
    let params = [SqliteValue::Text(table_name.to_owned().into())];
    let rows = conn
        .query_with_params(
            fsqlite_cx,
            &format!(
                "SELECT rebuild_version FROM {PORTER_FTS5_REBUILD_TABLE} WHERE table_name = ?1;"
            ),
            &params,
        )
        .await
        .map_err(|error| map_storage_error_at("read Porter FTS5 rebuild marker", error))?;

    let row = match rows.as_slice() {
        [] => return Ok(None),
        [row] => row,
        _ => {
            return Err(SearchError::InvalidConfig {
                field: "fts5.rebuild_version".to_owned(),
                value: table_name.to_owned(),
                reason: "governed Porter FTS5 marker table contains duplicate rows for one table"
                    .to_owned(),
            });
        }
    };
    match row.get(0) {
        Some(SqliteValue::Integer(version)) => Ok(Some(*version)),
        Some(value) => Err(SearchError::InvalidConfig {
            field: "fts5.rebuild_version".to_owned(),
            value: format!("{table_name}: {value:?}"),
            reason: "refusing to query a Porter FTS5 table whose governed rebuild marker is not an integer".to_owned(),
        }),
        None => Err(SearchError::InvalidConfig {
            field: "fts5.rebuild_version".to_owned(),
            value: format!("{table_name}: missing column"),
            reason: "refusing to query a Porter FTS5 table whose governed rebuild marker row is malformed".to_owned(),
        }),
    }
}

fn ensure_rebuildable_porter_fts5(
    table_name: &str,
    metadata: &PersistedFts5Metadata,
) -> SearchResult<()> {
    if !metadata
        .tokenizer
        .split_ascii_whitespace()
        .next()
        .is_some_and(|tokenizer| tokenizer.eq_ignore_ascii_case("porter"))
    {
        return Err(SearchError::InvalidConfig {
            field: "fts5.tokenize".to_owned(),
            value: metadata.tokenizer.clone(),
            reason: format!(
                "{table_name} is not a Porter FTS5 table according to its persisted sqlite_master definition"
            ),
        });
    }
    if metadata.content_mode == Fts5ContentMode::Contentless {
        return Err(SearchError::InvalidConfig {
            field: "fts5.content_mode".to_owned(),
            value: "contentless".to_owned(),
            reason: "Porter FTS5 rebuild requires authoritative text and original rowids; recreate the contentless table and re-ingest source documents rather than rebuilding from an index or preview".to_owned(),
        });
    }
    Ok(())
}

async fn persisted_fts5_doc_count(
    fsqlite_cx: &FsqliteCx,
    conn: &AsyncConnection,
    table_name: &str,
) -> SearchResult<usize> {
    let rows = conn
        .query(fsqlite_cx, &format!("SELECT COUNT(*) FROM {table_name};"))
        .await
        .map_err(|error| map_storage_error_at("count persisted Porter FTS5 documents", error))?;
    let [row] = rows.as_slice() else {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "COUNT(*) did not return exactly one row",
        ));
    };
    let Some(SqliteValue::Integer(count)) = row.get(0) else {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "COUNT(*) did not return an integer",
        ));
    };
    usize::try_from(*count).map_err(|_| {
        persisted_fts5_metadata_error(table_name, "COUNT(*) is negative or does not fit usize")
    })
}

fn decode_persisted_fts5_row(row: &Row) -> SearchResult<ScoredResult> {
    let Some(SqliteValue::Text(doc_id)) = row.get(0) else {
        return Err(persisted_fts5_result_error("doc_id must be non-NULL TEXT"));
    };
    if doc_id.is_empty() {
        return Err(persisted_fts5_result_error("doc_id must not be empty"));
    }

    let metadata = match row.get(1) {
        None => {
            return Err(persisted_fts5_result_error(
                "metadata_json column is missing",
            ));
        }
        Some(SqliteValue::Null) => None,
        Some(SqliteValue::Text(text)) if text.is_empty() => None,
        Some(SqliteValue::Text(text)) => {
            Some(
                serde_json::from_str(text).map_err(|error| SearchError::InvalidConfig {
                    field: "fts5.metadata_json".to_owned(),
                    value: error.to_string(),
                    reason: "persisted FTS5 metadata_json is not valid JSON".to_owned(),
                })?,
            )
        }
        Some(value) => {
            return Err(persisted_fts5_result_error(&format!(
                "metadata_json must be TEXT or NULL, got {value:?}"
            )));
        }
    };

    let raw_score = match row.get(2) {
        Some(SqliteValue::Float(score)) => *score,
        Some(SqliteValue::Integer(score)) => *score as f64,
        Some(value) => {
            return Err(persisted_fts5_result_error(&format!(
                "bm25 score must be REAL or INTEGER, got {value:?}"
            )));
        }
        None => return Err(persisted_fts5_result_error("bm25 score column is missing")),
    };
    let score = -raw_score;
    if !score.is_finite() || score > f64::from(f32::MAX) {
        return Err(persisted_fts5_result_error(
            "bm25 score is non-finite or does not fit f32",
        ));
    }
    #[allow(clippy::cast_possible_truncation)]
    let score = score as f32;

    Ok(ScoredResult {
        doc_id: doc_id.to_string().into(),
        score,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata,
    })
}

fn parse_persisted_fts5_metadata(
    table_name: &str,
    create_sql: &str,
) -> SearchResult<PersistedFts5Metadata> {
    let mut cursor = Fts5DdlCursor::new(create_sql);
    cursor.expect_keyword("CREATE", table_name)?;
    cursor.expect_keyword("VIRTUAL", table_name)?;
    cursor.expect_keyword("TABLE", table_name)?;
    if cursor.consume_keyword("IF") {
        cursor.expect_keyword("NOT", table_name)?;
        cursor.expect_keyword("EXISTS", table_name)?;
    }
    let declared_table = cursor
        .identifier()
        .ok_or_else(|| persisted_fts5_metadata_error(table_name, "missing virtual table name"))?;
    if !declared_table.eq_ignore_ascii_case(table_name) {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "sqlite_master definition declares a different table name",
        ));
    }
    cursor.expect_keyword("USING", table_name)?;
    cursor.expect_keyword("FTS5", table_name)?;
    let arguments = cursor.parenthesized(table_name)?;
    cursor.finish(table_name)?;

    let mut content_mode = Fts5ContentMode::Stored;
    let mut saw_content = false;
    let mut tokenizer = None;
    for argument in split_fts5_arguments(arguments, table_name)? {
        let Some((key, value)) = split_fts5_option(argument, table_name)? else {
            continue;
        };
        if key.eq_ignore_ascii_case("content") {
            if saw_content {
                return Err(persisted_fts5_metadata_error(
                    table_name,
                    "duplicate content option",
                ));
            }
            saw_content = true;
            content_mode = if parse_fts5_option_value(value, table_name)?.is_empty() {
                Fts5ContentMode::Contentless
            } else {
                Fts5ContentMode::External
            };
        } else if key.eq_ignore_ascii_case("tokenize") {
            if tokenizer.is_some() {
                return Err(persisted_fts5_metadata_error(
                    table_name,
                    "duplicate tokenize option",
                ));
            }
            tokenizer = Some(parse_fts5_option_value(value, table_name)?);
        }
    }

    let tokenizer = tokenizer.ok_or_else(|| {
        persisted_fts5_metadata_error(table_name, "missing explicit tokenize option")
    })?;
    Ok(PersistedFts5Metadata {
        content_mode,
        tokenizer,
    })
}

fn split_fts5_arguments<'a>(arguments: &'a str, table_name: &str) -> SearchResult<Vec<&'a str>> {
    let mut items = Vec::new();
    let mut start = 0;
    let mut depth = 0_u32;
    let mut quote = None;
    let bytes = arguments.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(delimiter) = quote {
            if byte == delimiter {
                if index + 1 < bytes.len() && bytes[index + 1] == delimiter {
                    index += 2;
                    continue;
                }
                quote = None;
            }
        } else {
            match byte {
                b'\'' | b'\"' | b'`' => quote = Some(byte),
                b'[' => quote = Some(b']'),
                b'(' => depth = depth.saturating_add(1),
                b')' => {
                    if depth == 0 {
                        return Err(persisted_fts5_metadata_error(
                            table_name,
                            "unbalanced parenthesis in FTS5 arguments",
                        ));
                    }
                    depth -= 1;
                }
                b',' if depth == 0 => {
                    items.push(arguments[start..index].trim());
                    start = index + 1;
                }
                _ => {}
            }
        }
        index += 1;
    }
    if quote.is_some() || depth != 0 {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "unterminated quote or parenthesis in FTS5 arguments",
        ));
    }
    let final_item = arguments[start..].trim();
    if final_item.is_empty() {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "empty FTS5 argument",
        ));
    }
    items.push(final_item);
    Ok(items)
}

fn split_fts5_option<'a>(
    argument: &'a str,
    table_name: &str,
) -> SearchResult<Option<(&'a str, &'a str)>> {
    let mut quote = None;
    let mut depth = 0_u32;
    let bytes = argument.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(delimiter) = quote {
            if byte == delimiter {
                if index + 1 < bytes.len() && bytes[index + 1] == delimiter {
                    index += 2;
                    continue;
                }
                quote = None;
            }
        } else {
            match byte {
                b'\'' | b'\"' | b'`' => quote = Some(byte),
                b'[' => quote = Some(b']'),
                b'(' => depth = depth.saturating_add(1),
                b')' => {
                    if depth == 0 {
                        return Err(persisted_fts5_metadata_error(
                            table_name,
                            "unbalanced parenthesis in an FTS5 argument",
                        ));
                    }
                    depth -= 1;
                }
                b'=' if depth == 0 => {
                    let key = argument[..index].trim();
                    let value = argument[index + 1..].trim();
                    if key.is_empty() || value.is_empty() {
                        return Err(persisted_fts5_metadata_error(
                            table_name,
                            "FTS5 option has an empty key or value",
                        ));
                    }
                    return Ok(Some((key, value)));
                }
                _ => {}
            }
        }
        index += 1;
    }
    if quote.is_some() || depth != 0 {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "unterminated quote or parenthesis in an FTS5 argument",
        ));
    }
    Ok(None)
}

fn parse_fts5_option_value(value: &str, table_name: &str) -> SearchResult<String> {
    let value = value.trim();
    let Some(quote) = value
        .as_bytes()
        .first()
        .copied()
        .filter(|quote| matches!(quote, b'\'' | b'\"' | b'`'))
    else {
        return Ok(value.to_ascii_lowercase());
    };
    if value.len() < 2 {
        return Err(persisted_fts5_metadata_error(
            table_name,
            "unterminated quoted FTS5 option value",
        ));
    }

    let mut decoded = String::new();
    let bytes = value.as_bytes();
    let mut index = 1;
    while index < bytes.len() {
        if bytes[index] == quote {
            if index + 1 < bytes.len() && bytes[index + 1] == quote {
                decoded.push(quote as char);
                index += 2;
                continue;
            }
            if !value[index + 1..].trim().is_empty() {
                return Err(persisted_fts5_metadata_error(
                    table_name,
                    "trailing text after quoted FTS5 option value",
                ));
            }
            return Ok(decoded.to_ascii_lowercase());
        }
        let Some(character) = value[index..].chars().next() else {
            break;
        };
        decoded.push(character);
        index += character.len_utf8();
    }
    Err(persisted_fts5_metadata_error(
        table_name,
        "unterminated quoted FTS5 option value",
    ))
}

struct Fts5DdlCursor<'a> {
    source: &'a str,
    index: usize,
}

impl<'a> Fts5DdlCursor<'a> {
    const fn new(source: &'a str) -> Self {
        Self { source, index: 0 }
    }

    fn skip_whitespace(&mut self) {
        while self
            .source
            .as_bytes()
            .get(self.index)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.index += 1;
        }
    }

    fn consume_keyword(&mut self, keyword: &str) -> bool {
        self.skip_whitespace();
        let remaining = &self.source[self.index..];
        let Some(candidate) = remaining.get(..keyword.len()) else {
            return false;
        };
        if !candidate.eq_ignore_ascii_case(keyword) {
            return false;
        }
        if remaining
            .as_bytes()
            .get(keyword.len())
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            return false;
        }
        self.index += keyword.len();
        true
    }

    fn expect_keyword(&mut self, keyword: &str, table_name: &str) -> SearchResult<()> {
        if self.consume_keyword(keyword) {
            Ok(())
        } else {
            Err(persisted_fts5_metadata_error(
                table_name,
                &format!("expected {keyword} in CREATE VIRTUAL TABLE definition"),
            ))
        }
    }

    fn identifier(&mut self) -> Option<String> {
        self.skip_whitespace();
        let byte = *self.source.as_bytes().get(self.index)?;
        let closing = match byte {
            b'\"' => Some(b'\"'),
            b'`' => Some(b'`'),
            b'[' => Some(b']'),
            _ => None,
        };
        if let Some(closing) = closing {
            self.index += 1;
            let start = self.index;
            while let Some(current) = self.source.as_bytes().get(self.index).copied() {
                if current == closing {
                    let value = self.source[start..self.index].to_owned();
                    self.index += 1;
                    return Some(value);
                }
                self.index += 1;
            }
            return None;
        }

        if !(byte == b'_' || byte.is_ascii_alphabetic()) {
            return None;
        }
        let start = self.index;
        self.index += 1;
        while self
            .source
            .as_bytes()
            .get(self.index)
            .is_some_and(|current| *current == b'_' || current.is_ascii_alphanumeric())
        {
            self.index += 1;
        }
        Some(self.source[start..self.index].to_owned())
    }

    fn parenthesized(&mut self, table_name: &str) -> SearchResult<&'a str> {
        self.skip_whitespace();
        if self.source.as_bytes().get(self.index) != Some(&b'(') {
            return Err(persisted_fts5_metadata_error(
                table_name,
                "expected FTS5 argument list",
            ));
        }
        self.index += 1;
        let start = self.index;
        let mut depth = 0_u32;
        let mut quote = None;
        while let Some(byte) = self.source.as_bytes().get(self.index).copied() {
            if let Some(delimiter) = quote {
                if byte == delimiter {
                    if self.source.as_bytes().get(self.index + 1) == Some(&delimiter) {
                        self.index += 2;
                        continue;
                    }
                    quote = None;
                }
            } else {
                match byte {
                    b'\'' | b'\"' | b'`' => quote = Some(byte),
                    b'[' => quote = Some(b']'),
                    b'(' => depth = depth.saturating_add(1),
                    b')' if depth == 0 => {
                        let end = self.index;
                        self.index += 1;
                        return Ok(&self.source[start..end]);
                    }
                    b')' => depth -= 1,
                    _ => {}
                }
            }
            self.index += 1;
        }
        Err(persisted_fts5_metadata_error(
            table_name,
            "unterminated FTS5 argument list",
        ))
    }

    fn finish(&mut self, table_name: &str) -> SearchResult<()> {
        self.skip_whitespace();
        if self.source.as_bytes().get(self.index) == Some(&b';') {
            self.index += 1;
            self.skip_whitespace();
        }
        if self.index == self.source.len() {
            Ok(())
        } else {
            Err(persisted_fts5_metadata_error(
                table_name,
                "unexpected trailing text in CREATE VIRTUAL TABLE definition",
            ))
        }
    }
}

fn persisted_fts5_metadata_error(table_name: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "fts5.persisted_metadata".to_owned(),
        value: table_name.to_owned(),
        reason: reason.to_owned(),
    }
}

fn persisted_fts5_result_error(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "fts5.persisted_result".to_owned(),
        value: reason.to_owned(),
        reason: "persisted FTS5 table does not meet the frankensearch result contract".to_owned(),
    }
}

// ─── Split lexical trait implementations ────────────────────────────────────

#[allow(clippy::significant_drop_tightening)]
impl frankensearch_core::LexicalRead for Fts5LexicalSearch {
    #[instrument(skip_all, fields(query = %query, limit = limit))]
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let query = Self::truncate_query(query);

            if query.trim().is_empty() {
                return Ok(Vec::new());
            }

            let table = self.table.lock().map_err(lock_error)?;
            let rowid_map = self.rowid_map.lock().map_err(lock_error)?;

            let search_results = table
                .search(query)
                .map_err(|e| SearchError::QueryParseError {
                    query: query.to_owned(),
                    detail: e.to_string(),
                })?;

            debug!(hits = search_results.len(), "fts5 BM25 search completed");

            let mut results = Vec::with_capacity(search_results.len().min(limit));
            for (rowid, score) in search_results.into_iter().take(limit) {
                let doc_id = rowid_map.get_doc_id(rowid).unwrap_or("").to_owned();

                // FTS5 BM25 scores are negative (lower = better).
                // Negate to produce positive scores (higher = better).
                #[allow(clippy::cast_possible_truncation)]
                let bm25_score = (-score) as f32;

                let metadata = table
                    .get_document(rowid)
                    .and_then(|cols| cols.get(COL_METADATA))
                    .filter(|s| !s.is_empty())
                    .and_then(|s| serde_json::from_str(s).ok());

                results.push(ScoredResult {
                    doc_id: doc_id.into(),
                    score: bm25_score,
                    source: ScoreSource::Lexical,
                    index: None,
                    fast_score: None,
                    quality_score: None,
                    lexical_score: Some(bm25_score),
                    rerank_score: None,
                    explanation: None,
                    metadata,
                });
            }

            Ok(results)
        })
    }

    /// FTS5 attaches full metadata during `search`, so the inherited eager
    /// `search_candidates` and its no-op hydration are exact for this backend:
    /// there is no deferred path to lose and no snapshot to pin.
    fn doc_count(&self) -> usize {
        self.doc_count.load(Ordering::Relaxed)
    }
}

impl frankensearch_core::LexicalWrite for Fts5LexicalSearch {
    fn index_document<'a>(
        &'a self,
        _cx: &'a Cx,
        doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            let mut table = self.table.lock().map_err(lock_error)?;
            let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

            // Upsert: delete existing document with same ID first.
            if let Some(old_rowid) = rowid_map.get_rowid(&doc.id) {
                table.delete_document(old_rowid);
                // Don't decrement doc_count here — it nets out with the add below.
            } else {
                // Only increment if this is truly new.
                self.doc_count.fetch_add(1, Ordering::Relaxed);
            }

            let rowid = rowid_map.get_or_assign(&doc.id);
            let columns = Self::doc_to_columns(doc);
            table.insert_document(rowid, &columns);

            Ok(())
        })
    }

    fn index_documents<'a>(
        &'a self,
        _cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            let mut table = self.table.lock().map_err(lock_error)?;
            let mut rowid_map = self.rowid_map.lock().map_err(lock_error)?;

            for doc in docs {
                if let Some(old_rowid) = rowid_map.get_rowid(&doc.id) {
                    table.delete_document(old_rowid);
                } else {
                    self.doc_count.fetch_add(1, Ordering::Relaxed);
                }

                let rowid = rowid_map.get_or_assign(&doc.id);
                let columns = Self::doc_to_columns(doc);
                table.insert_document(rowid, &columns);
            }

            debug!(count = docs.len(), "fts5: batch indexed documents");
            Ok(())
        })
    }

    fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
        // FTS5 in-memory table has no separate commit phase.
        Box::pin(async { Ok(()) })
    }
}

// ─── Hit type for snippet-aware search ──────────────────────────────────────

/// A hit from FTS5 search with optional snippet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fts5Hit {
    /// Document identifier.
    pub doc_id: String,
    /// BM25 relevance score (higher = better).
    pub bm25_score: f32,
    /// Position in results (0-indexed).
    pub rank: usize,
    /// Highlighted content snippet around matching terms.
    pub snippet: Option<String>,
    /// Document metadata.
    pub metadata: Option<serde_json::Value>,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn lock_error<T>(_: T) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fts5",
        source: Box::new(std::io::Error::other("fts5 mutex poisoned")),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::future::Future;

    use super::*;
    // The split capabilities, so `adapter.search(..)` / `.index_document(..)`
    // / `.doc_count()` resolve now that the combined trait is gone.
    use frankensearch_core::{LexicalRead as _, LexicalWrite as _};

    /// Helper: run async test code with a `Cx` (asupersync, NO tokio).
    fn run_with_cx<F, Fut>(f: F)
    where
        F: FnOnce(Cx) -> Fut,
        Fut: Future<Output = ()>,
    {
        asupersync::test_utils::run_test_with_cx(f);
    }

    fn make_doc(id: &str, content: &str) -> IndexableDocument {
        IndexableDocument::new(id, content)
    }

    fn make_doc_with_title(id: &str, title: &str, content: &str) -> IndexableDocument {
        IndexableDocument::new(id, content).with_title(title)
    }

    fn make_doc_with_metadata(
        id: &str,
        content: &str,
        key: &str,
        value: &str,
    ) -> IndexableDocument {
        IndexableDocument::new(id, content).with_metadata(key, value)
    }

    // -- Construction --

    #[test]
    fn new_instance_is_empty() {
        let search = Fts5LexicalSearch::with_defaults();
        assert_eq!(search.doc_count(), 0);
    }

    #[test]
    fn config_defaults_are_sane() {
        let config = Fts5AdapterConfig::default();
        assert_eq!(config.content_mode, Fts5ContentMode::Stored);
        assert_eq!(config.tokenizer, Fts5TokenizerChoice::Unicode61);
        assert!((config.title_boost - TITLE_BOOST).abs() < f64::EPSILON);
    }

    #[test]
    fn persisted_metadata_uses_ddl_for_content_mode_and_porter() {
        let stored = parse_persisted_fts5_metadata(
            "docs",
            "CREATE VIRTUAL TABLE docs USING fts5(doc_id, metadata_json, tokenize='porter unicode61');",
        )
        .expect("stored Porter definition should parse");
        assert_eq!(stored.content_mode, Fts5ContentMode::Stored);
        ensure_rebuildable_porter_fts5("docs", &stored)
            .expect("stored Porter definition should be rebuildable");

        let external = parse_persisted_fts5_metadata(
            "docs",
            "CREATE VIRTUAL TABLE docs USING fts5(doc_id, metadata_json, content='documents', tokenize='porter');",
        )
        .expect("external Porter definition should parse");
        assert_eq!(external.content_mode, Fts5ContentMode::External);
        ensure_rebuildable_porter_fts5("docs", &external)
            .expect("external Porter definition should be rebuildable");
    }

    #[test]
    fn persisted_metadata_rejects_contentless_or_non_porter_tables() {
        let contentless = parse_persisted_fts5_metadata(
            "docs",
            "CREATE VIRTUAL TABLE docs USING fts5(doc_id, metadata_json, content='', tokenize='porter');",
        )
        .expect("contentless Porter definition should parse before its policy check");
        assert_eq!(contentless.content_mode, Fts5ContentMode::Contentless);
        assert!(ensure_rebuildable_porter_fts5("docs", &contentless).is_err());

        let unicode = parse_persisted_fts5_metadata(
            "docs",
            "CREATE VIRTUAL TABLE docs USING fts5(doc_id, metadata_json, tokenize='unicode61');",
        )
        .expect("non-Porter definition should still parse");
        assert!(ensure_rebuildable_porter_fts5("docs", &unicode).is_err());
    }

    // -- Indexing --

    #[test]
    fn index_single_document() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc("doc1", "hello world of search");
            search.index_document(&cx, &doc).await.unwrap();
            assert_eq!(search.doc_count(), 1);
        });
    }

    #[test]
    fn index_batch_documents() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let docs = vec![
                make_doc("a", "first document"),
                make_doc("b", "second document"),
                make_doc("c", "third document"),
            ];
            search.index_documents(&cx, &docs).await.unwrap();
            assert_eq!(search.doc_count(), 3);
        });
    }

    #[test]
    fn upsert_replaces_existing_document() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc_v1 = make_doc("doc1", "original content");
            search.index_document(&cx, &doc_v1).await.unwrap();
            assert_eq!(search.doc_count(), 1);

            let doc_v2 = make_doc("doc1", "updated content completely different");
            search.index_document(&cx, &doc_v2).await.unwrap();
            assert_eq!(search.doc_count(), 1);

            // Search should find updated content.
            let results = search.search(&cx, "updated", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc1");

            // Old content should not match.
            let results = search.search(&cx, "original", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    // -- Search --

    #[test]
    fn search_finds_matching_document() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "rust programming language"))
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "python programming language"))
                .await
                .unwrap();

            let results = search.search(&cx, "rust", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc1");
            assert_eq!(results[0].source, ScoreSource::Lexical);
            assert!(results[0].lexical_score.is_some());
            assert!(results[0].score > 0.0);
        });
    }

    #[test]
    fn search_returns_results_sorted_by_relevance() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            // doc1 mentions "search" more times -> higher BM25.
            search
                .index_document(
                    &cx,
                    &make_doc("doc1", "search search search algorithms for search"),
                )
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "search algorithms"))
                .await
                .unwrap();

            let results = search.search(&cx, "search", 10).await.unwrap();
            assert_eq!(results.len(), 2);
            // Higher TF should produce higher BM25 score.
            assert!(results[0].score >= results[1].score);
        });
    }

    #[test]
    fn search_empty_query_returns_empty() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello world"))
                .await
                .unwrap();

            let results = search.search(&cx, "", 10).await.unwrap();
            assert!(results.is_empty());

            let results = search.search(&cx, "   ", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn search_no_match_returns_empty() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello world"))
                .await
                .unwrap();

            let results = search.search(&cx, "zzzznonexistent", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn search_respects_limit() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            for i in 0..10 {
                search
                    .index_document(
                        &cx,
                        &make_doc(&format!("doc{i}"), "common term in all docs"),
                    )
                    .await
                    .unwrap();
            }

            let results = search.search(&cx, "common", 3).await.unwrap();
            assert_eq!(results.len(), 3);
        });
    }

    #[test]
    fn search_preserves_metadata() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc_with_metadata("doc1", "searchable content", "category", "test");
            search.index_document(&cx, &doc).await.unwrap();

            let results = search.search(&cx, "searchable", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            let meta = results[0].metadata.as_ref().unwrap();
            assert_eq!(meta["category"], "test");
        });
    }

    #[test]
    fn search_with_title_and_content() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc_with_title("doc1", "Important Title", "body text here");
            search.index_document(&cx, &doc).await.unwrap();

            // Should match on title.
            let results = search.search(&cx, "important", 10).await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "doc1");

            // Should match on content.
            let results = search.search(&cx, "body", 10).await.unwrap();
            assert_eq!(results.len(), 1);
        });
    }

    // -- Snippets --

    #[test]
    #[allow(clippy::significant_drop_tightening)]
    fn search_with_snippets_returns_highlighted_text() {
        let search = Fts5LexicalSearch::with_defaults();

        {
            let mut table = search.table.lock().unwrap();
            let mut rowid_map = search.rowid_map.lock().unwrap();

            let doc = make_doc("doc1", "The quick brown fox jumps over the lazy dog");
            let rowid = rowid_map.get_or_assign(&doc.id);
            let columns = Fts5LexicalSearch::doc_to_columns(&doc);
            table.insert_document(rowid, &columns);
            search.doc_count.fetch_add(1, Ordering::Relaxed);
        }

        let hits = search.search_with_snippets("fox", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc1");
        assert!(hits[0].snippet.is_some());
        let snippet = hits[0].snippet.as_ref().unwrap();
        assert!(
            snippet.contains("<b>fox</b>"),
            "snippet should highlight match: {snippet}"
        );
    }

    // -- Delete --

    #[test]
    fn delete_document_removes_it() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "findable content"))
                .await
                .unwrap();
            assert_eq!(search.doc_count(), 1);

            let removed = search.delete_document("doc1").unwrap();
            assert!(removed);
            assert_eq!(search.doc_count(), 0);

            let results = search.search(&cx, "findable", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let search = Fts5LexicalSearch::with_defaults();
        let removed = search.delete_document("nonexistent").unwrap();
        assert!(!removed);
    }

    // -- Clear --

    #[test]
    fn clear_removes_all_documents() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "hello"))
                .await
                .unwrap();
            search
                .index_document(&cx, &make_doc("doc2", "world"))
                .await
                .unwrap();
            assert_eq!(search.doc_count(), 2);

            search.clear().unwrap();
            assert_eq!(search.doc_count(), 0);

            let results = search.search(&cx, "hello", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    // -- Commit is no-op --

    #[test]
    fn commit_succeeds_without_error() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search.commit(&cx).await.unwrap();
        });
    }

    // -- Edge cases --

    #[test]
    fn document_with_empty_content() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc("doc1", "");
            search.index_document(&cx, &doc).await.unwrap();
            assert_eq!(search.doc_count(), 1);

            let results = search.search(&cx, "anything", 10).await.unwrap();
            assert!(results.is_empty());
        });
    }

    #[test]
    fn document_with_special_characters() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            let doc = make_doc(
                "doc1",
                "error: fn<T>(x: &mut Vec<u8>) -> Result<(), Box<dyn Error>>",
            );
            search.index_document(&cx, &doc).await.unwrap();

            let results = search.search(&cx, "error", 10).await.unwrap();
            assert_eq!(results.len(), 1);
        });
    }

    #[test]
    fn batch_upsert_mixed_new_and_existing() {
        let search = Fts5LexicalSearch::with_defaults();
        run_with_cx(|cx| async move {
            search
                .index_document(&cx, &make_doc("doc1", "original"))
                .await
                .unwrap();
            assert_eq!(search.doc_count(), 1);

            let batch = vec![
                make_doc("doc1", "updated"),   // existing
                make_doc("doc2", "brand new"), // new
            ];
            search.index_documents(&cx, &batch).await.unwrap();
            assert_eq!(search.doc_count(), 2);
        });
    }

    // -- Trait object safety --

    #[test]
    fn fts5_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Fts5LexicalSearch>();
    }

    // -- Config serialization --

    #[test]
    fn config_serde_roundtrip() {
        let config = Fts5AdapterConfig {
            content_mode: Fts5ContentMode::Contentless,
            tokenizer: Fts5TokenizerChoice::Porter,
            title_boost: 3.0,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Fts5AdapterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content_mode, Fts5ContentMode::Contentless);
        assert_eq!(deserialized.tokenizer, Fts5TokenizerChoice::Porter);
        assert!((deserialized.title_boost - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn content_mode_default_is_stored() {
        assert_eq!(Fts5ContentMode::default(), Fts5ContentMode::Stored);
    }

    #[test]
    fn tokenizer_default_is_unicode61() {
        assert_eq!(
            Fts5TokenizerChoice::default(),
            Fts5TokenizerChoice::Unicode61
        );
    }

    // -- Query truncation --

    #[test]
    fn long_query_is_truncated() {
        let long_query = "a".repeat(MAX_QUERY_LENGTH + 100);
        let truncated = Fts5LexicalSearch::truncate_query(&long_query);
        assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
    }

    #[test]
    fn multibyte_query_uses_character_limit() {
        let long_query = "\u{00E9}".repeat(MAX_QUERY_LENGTH + 3);
        let truncated = Fts5LexicalSearch::truncate_query(&long_query);
        assert_eq!(truncated.chars().count(), MAX_QUERY_LENGTH);
        assert_eq!(truncated.len(), MAX_QUERY_LENGTH * '\u{00E9}'.len_utf8());
    }

    #[test]
    fn multibyte_query_within_character_limit_is_unchanged() {
        let query = "\u{00E9}".repeat(MAX_QUERY_LENGTH / 2 + 1);
        assert!(query.len() > MAX_QUERY_LENGTH);
        assert_eq!(Fts5LexicalSearch::truncate_query(&query), query);
    }

    #[test]
    fn normal_query_is_not_truncated() {
        let query = "normal search query";
        let result = Fts5LexicalSearch::truncate_query(query);
        assert_eq!(result, query);
    }

    // -- Debug impl --

    #[test]
    fn debug_format_includes_doc_count() {
        let search = Fts5LexicalSearch::with_defaults();
        let debug = format!("{search:?}");
        assert!(debug.contains("Fts5LexicalSearch"));
        assert!(debug.contains("doc_count"));
    }

    // -- Fts5Hit serde --

    #[test]
    fn fts5_hit_serde_roundtrip() {
        let hit = Fts5Hit {
            doc_id: "doc1".into(),
            bm25_score: 1.5,
            rank: 0,
            snippet: Some("hello <b>world</b>".to_owned()),
            metadata: None,
        };
        let json = serde_json::to_string(&hit).unwrap();
        let deserialized: Fts5Hit = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.doc_id, "doc1");
        assert!((deserialized.bm25_score - 1.5).abs() < f32::EPSILON);
    }

    // -- cass#301 merge/finalize scaling probe --

    /// Generate a deterministic, lexically varied document body of roughly
    /// `target_bytes`. Mixing a finite vocabulary with the doc index keeps the
    /// posting lists realistic (many shared terms, a few doc-unique terms)
    /// without any RNG, so the probe is reproducible across runs and hosts.
    fn synthetic_body(doc_index: usize, target_bytes: usize) -> String {
        const VOCAB: &[&str] = &[
            "retry",
            "backoff",
            "structured",
            "concurrency",
            "channel",
            "reserve",
            "commit",
            "cancel",
            "region",
            "scope",
            "embedder",
            "lexical",
            "semantic",
            "fusion",
            "rank",
            "vector",
            "index",
            "segment",
            "merge",
            "finalize",
            "tokenizer",
            "posting",
            "document",
            "search",
            "query",
            "score",
            "bm25",
        ];
        use std::fmt::Write as _;
        let mut body = String::with_capacity(target_bytes + 32);
        let mut counter = doc_index;
        while body.len() < target_bytes {
            let word = VOCAB[counter % VOCAB.len()];
            body.push_str(word);
            // Sprinkle a doc-unique token so each document has distinct terms.
            if counter % 11 == 0 {
                let _ = write!(body, " d{doc_index}t{counter}");
            }
            body.push(' ');
            counter = counter.wrapping_add(1).wrapping_add(doc_index);
        }
        body
    }

    /// cass#301: scaling probe for the frankensearch FTS5 lexical-index build
    /// path (`Fts5LexicalSearch::index_documents`, the exact path cass drives
    /// during `cass index --full`).
    ///
    /// Feeds an increasing number of documents simulating ~10MB -> ~40MB of
    /// indexed content and prints the wall-time of the index-build (finalize)
    /// phase plus a representative search at each size. The reported
    /// `build_ms_per_mb` (build time normalised by content size) is the
    /// diagnostic: if it is roughly flat across sizes the build is linear; if
    /// it climbs ~linearly with content size the build is O(N^2).
    ///
    /// Run with:
    /// `cargo test -p frankensearch-storage --features fts5 --release \
    ///    fts5_index_build_scaling_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "cass#301 scaling probe: run explicitly with --ignored --nocapture"]
    fn fts5_index_build_scaling_probe() {
        use std::time::Instant;

        // ~50 KB per document. Content megabytes => doc_count = mb * 20.
        const DOC_BYTES: usize = 50 * 1024;
        let content_mbs: Vec<usize> = std::env::var("FTS5_PROBE_MBS")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect()
            })
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| vec![10, 20, 30, 40]);

        run_with_cx(|cx| async move {
            eprintln!(
                "FTS5_PROBE doc_bytes={DOC_BYTES} sizes_mb={content_mbs:?} (cass#301 build/finalize scaling)"
            );
            for &mb in &content_mbs {
                let doc_count = mb * (1024 * 1024) / DOC_BYTES;
                let docs: Vec<IndexableDocument> = (0..doc_count)
                    .map(|i| {
                        IndexableDocument::new(format!("doc-{i}"), synthetic_body(i, DOC_BYTES))
                            .with_title(format!("Document {i}"))
                    })
                    .collect();

                let search = Fts5LexicalSearch::with_defaults();

                // Index-build / finalize phase — the wedge phase in cass#301.
                let build_started = Instant::now();
                search
                    .index_documents(&cx, &docs)
                    .await
                    .expect("index_documents");
                search.commit(&cx).await.expect("commit");
                let build_elapsed = build_started.elapsed();

                // Representative query against the freshly built index.
                let search_started = Instant::now();
                let hits = search
                    .search(&cx, "structured concurrency retry", 10)
                    .await
                    .expect("search");
                let search_elapsed = search_started.elapsed();

                let build_ms = build_elapsed.as_secs_f64() * 1_000.0;
                let search_ms = search_elapsed.as_secs_f64() * 1_000.0;
                eprintln!(
                    "FTS5_PROBE content_mb={mb} docs={doc_count} build_ms={build_ms:.1} \
                     build_ms_per_mb={:.2} search_ms={search_ms:.3} hits={} doc_count={}",
                    build_ms / mb as f64,
                    hits.len(),
                    search.doc_count(),
                );
                assert_eq!(search.doc_count(), doc_count);
            }
        });
    }
}
