//! Search history and bookmark persistence.
//!
//! Stores recent search queries with metadata (result count, latency, query
//! class) and user bookmarks of individual search results. Both are backed
//! by `FrankenSQLite` tables added in schema version 6.

use std::io;

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::Connection;
use fsqlite_types::value::SqliteValue;

use crate::connection::map_storage_error;

// ─── Search History ────────────────────────────────────────────────────────

/// A single entry in the search history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHistoryEntry {
    /// Auto-incremented row ID.
    pub id: i64,
    /// The query text as entered by the user.
    pub query: String,
    /// Classification of the query (e.g. "identifier", "`natural_language`").
    pub query_class: Option<String>,
    /// Number of results returned.
    pub result_count: Option<i64>,
    /// Phase-1 (fast) latency in milliseconds.
    pub phase1_latency_ms: Option<i64>,
    /// Phase-2 (quality) latency in milliseconds.
    pub phase2_latency_ms: Option<i64>,
    /// JSON array of top-3 result doc IDs.
    pub top_results_json: Option<String>,
    /// Unix epoch timestamp (seconds) of when the search was performed.
    pub searched_at: i64,
}

/// Insert a search history entry, deduplicating queries within `dedup_window_secs`.
///
/// If the same query text was searched within the last `dedup_window_secs`
/// seconds, the existing entry is updated instead of creating a duplicate.
#[allow(clippy::too_many_arguments)]
pub fn record_search(
    conn: &Connection,
    query: &str,
    query_class: Option<&str>,
    result_count: Option<i64>,
    phase1_latency_ms: Option<i64>,
    phase2_latency_ms: Option<i64>,
    top_results_json: Option<&str>,
    searched_at: i64,
    dedup_window_secs: i64,
) -> SearchResult<()> {
    // Check for a recent duplicate within the dedup window.
    let window_start = searched_at.saturating_sub(dedup_window_secs);
    let dedup_params = [
        SqliteValue::Text(query.to_owned()),
        SqliteValue::Integer(window_start),
    ];
    let existing = conn
        .query_with_params(
            "SELECT id FROM search_history \
             WHERE query = ?1 \
             AND searched_at >= ?2 \
             ORDER BY searched_at DESC LIMIT 1;",
            &dedup_params,
        )
        .map_err(map_storage_error)?;

    if let Some(row) = existing.first() {
        // Update the existing entry with new metadata.
        let existing_id = match row.get(0) {
            Some(SqliteValue::Integer(id)) => *id,
            _ => {
                return Err(SearchError::SubsystemError {
                    subsystem: "storage",
                    source: Box::new(io::Error::other("unexpected type for search_history.id")),
                });
            }
        };
        let update_params = [
            opt_i64(result_count),
            opt_i64(phase1_latency_ms),
            opt_i64(phase2_latency_ms),
            opt_text(top_results_json),
            SqliteValue::Integer(searched_at),
            SqliteValue::Integer(existing_id),
        ];
        conn.execute_with_params(
            "UPDATE search_history SET \
             result_count = ?1, phase1_latency_ms = ?2, phase2_latency_ms = ?3, \
             top_results_json = ?4, searched_at = ?5 \
             WHERE id = ?6;",
            &update_params,
        )
        .map_err(map_storage_error)?;
    } else {
        // Insert a new entry.
        let insert_params = [
            SqliteValue::Text(query.to_owned()),
            opt_text(query_class),
            opt_i64(result_count),
            opt_i64(phase1_latency_ms),
            opt_i64(phase2_latency_ms),
            opt_text(top_results_json),
            SqliteValue::Integer(searched_at),
        ];
        conn.execute_with_params(
            "INSERT INTO search_history \
             (query, query_class, result_count, phase1_latency_ms, \
              phase2_latency_ms, top_results_json, searched_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);",
            &insert_params,
        )
        .map_err(map_storage_error)?;
    }

    Ok(())
}

/// Retrieve search history in most-recent-first order.
pub fn list_search_history(conn: &Connection, limit: i64) -> SearchResult<Vec<SearchHistoryEntry>> {
    let params = [SqliteValue::Integer(limit)];
    let rows = conn
        .query_with_params(
            "SELECT id, query, query_class, result_count, phase1_latency_ms, \
             phase2_latency_ms, top_results_json, searched_at \
             FROM search_history ORDER BY searched_at DESC LIMIT ?1;",
            &params,
        )
        .map_err(map_storage_error)?;

    let mut entries = Vec::with_capacity(rows.len());
    for row in &rows {
        entries.push(parse_history_row(row)?);
    }
    Ok(entries)
}

/// Find history entries matching a prefix (for auto-suggest).
pub fn search_history_prefix(
    conn: &Connection,
    prefix: &str,
    limit: i64,
) -> SearchResult<Vec<SearchHistoryEntry>> {
    let pattern = format!("{prefix}%");
    let params = [SqliteValue::Text(pattern), SqliteValue::Integer(limit)];
    let rows = conn
        .query_with_params(
            "SELECT id, query, query_class, result_count, phase1_latency_ms, \
             phase2_latency_ms, top_results_json, searched_at \
             FROM search_history WHERE query LIKE ?1 \
             ORDER BY searched_at DESC LIMIT ?2;",
            &params,
        )
        .map_err(map_storage_error)?;

    let mut entries = Vec::with_capacity(rows.len());
    for row in &rows {
        entries.push(parse_history_row(row)?);
    }
    Ok(entries)
}

/// Count total search history entries.
pub fn count_search_history(conn: &Connection) -> SearchResult<i64> {
    let rows = conn
        .query("SELECT COUNT(*) FROM search_history;")
        .map_err(map_storage_error)?;
    match rows.first().and_then(|r| r.get(0)) {
        Some(SqliteValue::Integer(count)) => Ok(*count),
        _ => Ok(0),
    }
}

/// Truncate search history to keep only the most recent `max_entries`.
pub fn truncate_search_history(conn: &Connection, max_entries: i64) -> SearchResult<i64> {
    let count = count_search_history(conn)?;
    if count <= max_entries {
        return Ok(0);
    }

    let to_delete = count - max_entries;
    let params = [SqliteValue::Integer(to_delete)];
    conn.execute_with_params(
        "DELETE FROM search_history WHERE id IN (\
         SELECT id FROM search_history ORDER BY searched_at ASC LIMIT ?1);",
        &params,
    )
    .map_err(map_storage_error)?;

    Ok(to_delete)
}

fn parse_history_row(row: &fsqlite::Row) -> SearchResult<SearchHistoryEntry> {
    let id = match row.get(0) {
        Some(SqliteValue::Integer(v)) => *v,
        _ => {
            return Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other("missing search_history.id")),
            });
        }
    };
    let query = match row.get(1) {
        Some(SqliteValue::Text(v)) => v.clone(),
        _ => {
            return Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other("missing search_history.query")),
            });
        }
    };
    let searched_at = match row.get(7) {
        Some(SqliteValue::Integer(v)) => *v,
        _ => {
            return Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other("missing search_history.searched_at")),
            });
        }
    };

    Ok(SearchHistoryEntry {
        id,
        query,
        query_class: extract_opt_text(row, 2),
        result_count: extract_opt_i64(row, 3),
        phase1_latency_ms: extract_opt_i64(row, 4),
        phase2_latency_ms: extract_opt_i64(row, 5),
        top_results_json: extract_opt_text(row, 6),
        searched_at,
    })
}

// ─── Bookmarks ─────────────────────────────────────────────────────────────

/// A bookmarked search result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark {
    /// Auto-incremented row ID.
    pub id: i64,
    /// Document ID of the bookmarked result.
    pub doc_id: String,
    /// The query that led to this bookmark (may be `None`).
    pub query: Option<String>,
    /// User annotation.
    pub note: Option<String>,
    /// Unix epoch timestamp (seconds) of creation.
    pub created_at: i64,
}

/// Add a bookmark. If the same (`doc_id`, query) pair exists, updates the note.
pub fn add_bookmark(
    conn: &Connection,
    doc_id: &str,
    query: Option<&str>,
    note: Option<&str>,
    created_at: i64,
) -> SearchResult<()> {
    // Check for existing bookmark with same (doc_id, query) pair.
    // Handle NULL query explicitly since SQL NULL != NULL.
    let existing = if let Some(q) = query {
        let check_params = [
            SqliteValue::Text(doc_id.to_owned()),
            SqliteValue::Text(q.to_owned()),
        ];
        conn.query_with_params(
            "SELECT id FROM bookmarks \
             WHERE doc_id = ?1 AND query = ?2 LIMIT 1;",
            &check_params,
        )
        .map_err(map_storage_error)?
    } else {
        let check_params = [SqliteValue::Text(doc_id.to_owned())];
        conn.query_with_params(
            "SELECT id FROM bookmarks \
             WHERE doc_id = ?1 AND query IS NULL LIMIT 1;",
            &check_params,
        )
        .map_err(map_storage_error)?
    };

    if let Some(row) = existing.first() {
        let existing_id = match row.get(0) {
            Some(SqliteValue::Integer(id)) => *id,
            _ => {
                return Err(SearchError::SubsystemError {
                    subsystem: "storage",
                    source: Box::new(io::Error::other("unexpected type for bookmarks.id")),
                });
            }
        };
        let update_params = [
            opt_text(note),
            SqliteValue::Integer(created_at),
            SqliteValue::Integer(existing_id),
        ];
        conn.execute_with_params(
            "UPDATE bookmarks SET note = ?1, created_at = ?2 WHERE id = ?3;",
            &update_params,
        )
        .map_err(map_storage_error)?;
    } else {
        let insert_params = [
            SqliteValue::Text(doc_id.to_owned()),
            opt_text(query),
            opt_text(note),
            SqliteValue::Integer(created_at),
        ];
        conn.execute_with_params(
            "INSERT INTO bookmarks (doc_id, query, note, created_at) \
             VALUES (?1, ?2, ?3, ?4);",
            &insert_params,
        )
        .map_err(map_storage_error)?;
    }
    Ok(())
}

/// Remove a bookmark by its row ID.
pub fn remove_bookmark(conn: &Connection, bookmark_id: i64) -> SearchResult<bool> {
    let params = [SqliteValue::Integer(bookmark_id)];
    let existed = !conn
        .query_with_params("SELECT 1 FROM bookmarks WHERE id = ?1 LIMIT 1;", &params)
        .map_err(map_storage_error)?
        .is_empty();

    if !existed {
        return Ok(false);
    }

    conn.execute_with_params("DELETE FROM bookmarks WHERE id = ?1;", &params)
        .map_err(map_storage_error)?;

    // Verify deletion completed.
    let check = conn
        .query_with_params("SELECT id FROM bookmarks WHERE id = ?1;", &params)
        .map_err(map_storage_error)?;
    Ok(check.is_empty())
}

/// Remove a bookmark by document ID (removes all bookmarks for that doc).
pub fn remove_bookmark_by_doc(conn: &Connection, doc_id: &str) -> SearchResult<()> {
    let params = [SqliteValue::Text(doc_id.to_owned())];
    conn.execute_with_params("DELETE FROM bookmarks WHERE doc_id = ?1;", &params)
        .map_err(map_storage_error)?;
    Ok(())
}

/// List all bookmarks in most-recent-first order.
pub fn list_bookmarks(conn: &Connection, limit: i64) -> SearchResult<Vec<Bookmark>> {
    let params = [SqliteValue::Integer(limit)];
    let rows = conn
        .query_with_params(
            "SELECT id, doc_id, query, note, created_at \
             FROM bookmarks ORDER BY created_at DESC LIMIT ?1;",
            &params,
        )
        .map_err(map_storage_error)?;

    let mut bookmarks = Vec::with_capacity(rows.len());
    for row in &rows {
        bookmarks.push(parse_bookmark_row(row)?);
    }
    Ok(bookmarks)
}

/// Count total bookmarks.
pub fn count_bookmarks(conn: &Connection) -> SearchResult<i64> {
    let rows = conn
        .query("SELECT COUNT(*) FROM bookmarks;")
        .map_err(map_storage_error)?;
    match rows.first().and_then(|r| r.get(0)) {
        Some(SqliteValue::Integer(count)) => Ok(*count),
        _ => Ok(0),
    }
}

/// Check if a document is bookmarked.
pub fn is_bookmarked(conn: &Connection, doc_id: &str) -> SearchResult<bool> {
    let params = [SqliteValue::Text(doc_id.to_owned())];
    let rows = conn
        .query_with_params(
            "SELECT 1 FROM bookmarks WHERE doc_id = ?1 LIMIT 1;",
            &params,
        )
        .map_err(map_storage_error)?;
    Ok(!rows.is_empty())
}

fn parse_bookmark_row(row: &fsqlite::Row) -> SearchResult<Bookmark> {
    let id = match row.get(0) {
        Some(SqliteValue::Integer(v)) => *v,
        _ => {
            return Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other("missing bookmarks.id")),
            });
        }
    };
    let doc_id = match row.get(1) {
        Some(SqliteValue::Text(v)) => v.clone(),
        _ => {
            return Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other("missing bookmarks.doc_id")),
            });
        }
    };
    let created_at = match row.get(4) {
        Some(SqliteValue::Integer(v)) => *v,
        _ => {
            return Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other("missing bookmarks.created_at")),
            });
        }
    };

    Ok(Bookmark {
        id,
        doc_id,
        query: extract_opt_text(row, 2),
        note: extract_opt_text(row, 3),
        created_at,
    })
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn opt_text(value: Option<&str>) -> SqliteValue {
    value.map_or(SqliteValue::Null, |s| SqliteValue::Text(s.to_owned()))
}

fn opt_i64(value: Option<i64>) -> SqliteValue {
    value.map_or(SqliteValue::Null, SqliteValue::Integer)
}

fn extract_opt_text(row: &fsqlite::Row, index: usize) -> Option<String> {
    match row.get(index) {
        Some(SqliteValue::Text(v)) => Some(v.clone()),
        _ => None,
    }
}

fn extract_opt_i64(row: &fsqlite::Row, index: usize) -> Option<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(v)) => Some(*v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema;

    fn test_conn() -> Connection {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        // Apply pragmas for foreign keys.
        conn.execute("PRAGMA foreign_keys=ON;")
            .expect("pragma should succeed");
        schema::bootstrap(&conn).expect("schema bootstrap should succeed");
        conn
    }

    // ─── Search History Tests ──────────────────────────────────────────

    #[test]
    fn record_and_list_search_history() {
        let conn = test_conn();
        record_search(
            &conn,
            "rust async",
            Some("natural_language"),
            Some(42),
            Some(5),
            Some(120),
            Some("[\"doc1\",\"doc2\",\"doc3\"]"),
            1_700_000_000,
            60,
        )
        .expect("record should succeed");

        let entries = list_search_history(&conn, 10).expect("list should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].query, "rust async");
        assert_eq!(entries[0].query_class.as_deref(), Some("natural_language"));
        assert_eq!(entries[0].result_count, Some(42));
        assert_eq!(entries[0].phase1_latency_ms, Some(5));
        assert_eq!(entries[0].phase2_latency_ms, Some(120));
    }

    #[test]
    fn history_dedup_within_window() {
        let conn = test_conn();
        record_search(
            &conn,
            "tokio spawn",
            None,
            Some(10),
            None,
            None,
            None,
            1_700_000_000,
            60,
        )
        .expect("first record");

        // Same query within 60 seconds — should update, not insert.
        record_search(
            &conn,
            "tokio spawn",
            None,
            Some(15),
            None,
            None,
            None,
            1_700_000_030,
            60,
        )
        .expect("dedup record");

        let count = count_search_history(&conn).expect("count");
        assert_eq!(count, 1, "dedup should prevent duplicate entry");

        let entries = list_search_history(&conn, 10).expect("list");
        assert_eq!(
            entries[0].result_count,
            Some(15),
            "dedup should update result_count"
        );
    }

    #[test]
    fn history_no_dedup_outside_window() {
        let conn = test_conn();
        record_search(
            &conn,
            "async fn",
            None,
            Some(5),
            None,
            None,
            None,
            1_700_000_000,
            60,
        )
        .expect("first record");

        // Same query but 2 minutes later — should insert new entry.
        record_search(
            &conn,
            "async fn",
            None,
            Some(8),
            None,
            None,
            None,
            1_700_000_120,
            60,
        )
        .expect("second record");

        let count = count_search_history(&conn).expect("count");
        assert_eq!(count, 2, "entries outside dedup window should both persist");
    }

    #[test]
    fn history_mru_order() {
        let conn = test_conn();
        for (i, ts) in [1_700_000_000i64, 1_700_000_060, 1_700_000_120]
            .iter()
            .enumerate()
        {
            record_search(
                &conn,
                &format!("query-{i}"),
                None,
                None,
                None,
                None,
                None,
                *ts,
                0,
            )
            .expect("record");
        }

        let entries = list_search_history(&conn, 10).expect("list");
        assert_eq!(entries[0].query, "query-2", "most recent first");
        assert_eq!(entries[1].query, "query-1");
        assert_eq!(entries[2].query, "query-0", "oldest last");
    }

    #[test]
    fn history_truncation() {
        let conn = test_conn();
        for i in 0..10 {
            record_search(
                &conn,
                &format!("query-{i}"),
                None,
                None,
                None,
                None,
                None,
                1_700_000_000 + i64::from(i) * 60,
                0,
            )
            .expect("record");
        }

        assert_eq!(count_search_history(&conn).expect("count"), 10);

        let deleted = truncate_search_history(&conn, 5).expect("truncate");
        assert_eq!(deleted, 5);
        assert_eq!(
            count_search_history(&conn).expect("count after truncate"),
            5
        );

        // Remaining entries should be the 5 most recent.
        let entries = list_search_history(&conn, 10).expect("list");
        assert_eq!(entries[0].query, "query-9");
        assert_eq!(entries[4].query, "query-5");
    }

    #[test]
    fn history_truncation_noop_when_under_limit() {
        let conn = test_conn();
        record_search(
            &conn,
            "only-one",
            None,
            None,
            None,
            None,
            None,
            1_700_000_000,
            0,
        )
        .expect("record");

        let deleted = truncate_search_history(&conn, 100).expect("truncate");
        assert_eq!(deleted, 0);
        assert_eq!(count_search_history(&conn).expect("count"), 1);
    }

    #[test]
    fn history_prefix_search() {
        let conn = test_conn();
        for q in ["rust async", "rust ownership", "python async"] {
            record_search(&conn, q, None, None, None, None, None, 1_700_000_000, 0)
                .expect("record");
        }

        let rust_results = search_history_prefix(&conn, "rust", 10).expect("prefix search");
        assert_eq!(rust_results.len(), 2);

        let py_results = search_history_prefix(&conn, "python", 10).expect("prefix search");
        assert_eq!(py_results.len(), 1);

        let no_results = search_history_prefix(&conn, "golang", 10).expect("prefix search");
        assert!(no_results.is_empty());
    }

    // ─── Bookmark Tests ────────────────────────────────────────────────

    #[test]
    fn add_and_list_bookmarks() {
        let conn = test_conn();
        add_bookmark(
            &conn,
            "doc-123",
            Some("rust async"),
            Some("great explanation"),
            1_700_000_000,
        )
        .expect("add bookmark");

        let bookmarks = list_bookmarks(&conn, 10).expect("list");
        assert_eq!(bookmarks.len(), 1);
        assert_eq!(bookmarks[0].doc_id, "doc-123");
        assert_eq!(bookmarks[0].query.as_deref(), Some("rust async"));
        assert_eq!(bookmarks[0].note.as_deref(), Some("great explanation"));
    }

    #[test]
    fn bookmark_same_doc_different_queries() {
        let conn = test_conn();
        add_bookmark(
            &conn,
            "doc-1",
            Some("rust async"),
            Some("note-a"),
            1_700_000_000,
        )
        .expect("first bookmark");

        add_bookmark(
            &conn,
            "doc-1",
            Some("tokio spawn"),
            Some("note-b"),
            1_700_000_060,
        )
        .expect("second bookmark");

        let count = count_bookmarks(&conn).expect("count");
        assert_eq!(
            count, 2,
            "same doc from different queries should create separate bookmarks"
        );
    }

    #[test]
    fn bookmark_upsert_same_doc_and_query() {
        let conn = test_conn();
        add_bookmark(
            &conn,
            "doc-1",
            Some("rust async"),
            Some("old note"),
            1_700_000_000,
        )
        .expect("first");

        add_bookmark(
            &conn,
            "doc-1",
            Some("rust async"),
            Some("updated note"),
            1_700_000_300,
        )
        .expect("upsert");

        let count = count_bookmarks(&conn).expect("count");
        assert_eq!(count, 1, "same (doc_id, query) should upsert");

        let bookmarks = list_bookmarks(&conn, 10).expect("list");
        assert_eq!(
            bookmarks[0].note.as_deref(),
            Some("updated note"),
            "note should be updated"
        );
    }

    #[test]
    fn remove_bookmark_by_id() {
        let conn = test_conn();
        add_bookmark(&conn, "doc-rm", None, None, 1_700_000_000).expect("add");

        let bookmarks = list_bookmarks(&conn, 10).expect("list");
        assert_eq!(bookmarks.len(), 1);

        let removed = remove_bookmark(&conn, bookmarks[0].id).expect("remove");
        assert!(removed);
        assert_eq!(count_bookmarks(&conn).expect("count"), 0);
    }

    #[test]
    fn remove_bookmark_by_id_returns_false_when_missing() {
        let conn = test_conn();
        add_bookmark(&conn, "doc-rm-missing", None, None, 1_700_000_000).expect("add");
        let removed = remove_bookmark(&conn, 999_999).expect("remove missing");
        assert!(!removed, "missing bookmark id should report false");
        assert_eq!(
            count_bookmarks(&conn).expect("count"),
            1,
            "existing bookmark should remain untouched"
        );
    }

    #[test]
    fn remove_bookmark_by_doc_id() {
        let conn = test_conn();
        add_bookmark(&conn, "doc-rm2", Some("q1"), None, 1_700_000_000).expect("add 1");
        add_bookmark(&conn, "doc-rm2", Some("q2"), None, 1_700_000_060).expect("add 2");

        assert_eq!(count_bookmarks(&conn).expect("count"), 2);
        remove_bookmark_by_doc(&conn, "doc-rm2").expect("remove by doc");
        assert_eq!(count_bookmarks(&conn).expect("count"), 0);
    }

    #[test]
    fn is_bookmarked_check() {
        let conn = test_conn();
        assert!(!is_bookmarked(&conn, "doc-x").expect("check"));

        add_bookmark(&conn, "doc-x", None, None, 1_700_000_000).expect("add");
        assert!(is_bookmarked(&conn, "doc-x").expect("check after add"));
    }

    #[test]
    fn bookmark_mru_order() {
        let conn = test_conn();
        for (i, ts) in [1_700_000_000i64, 1_700_000_060, 1_700_000_120]
            .iter()
            .enumerate()
        {
            add_bookmark(&conn, &format!("doc-{i}"), None, None, *ts).expect("add");
        }

        let bookmarks = list_bookmarks(&conn, 10).expect("list");
        assert_eq!(bookmarks[0].doc_id, "doc-2", "most recent first");
        assert_eq!(bookmarks[2].doc_id, "doc-0", "oldest last");
    }
}
