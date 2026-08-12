//! Batch dedup lookup benchmark.
//!
//! `production_original` calls the shipped [`Storage::check_dedup_batch`] path:
//! one `IN (...)` query, materialize matching rows into a
//! `HashMap<String, DedupRow>`, then probe it once per requested doc. `slot_join`
//! embeds the candidate: a `VALUES` relation carries input order into `SQLite` and
//! returns slot-aligned rows. Both arms run in one alternating-round harness,
//! preceded by a production-vs-mirror-vs-candidate equality oracle.
//!
//! Run with:
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!   cargo bench -p frankensearch-storage --features bench-internals \
//!   --profile release --bench dedup_batch
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::hint::black_box;

use frankensearch_core::{SearchError, SearchResult, bench_support::paired_median_ratio};
use frankensearch_storage::{DeduplicationDecision, DocumentRecord, EmbeddingStatus, Storage};
use fsqlite::{AsyncConnection, Row};
use fsqlite_types::value::SqliteValue;

const EMBEDDER_ID: &str = "fast-tier";

#[derive(Debug, Clone, PartialEq, Eq)]
struct LegacyDedupRow {
    content_hash: [u8; 32],
    status: Option<EmbeddingStatus>,
}

struct DedupFixture {
    storage: Storage,
    items: Vec<(String, [u8; 32])>,
}

fn hash_for(index: usize, salt: u8) -> [u8; 32] {
    let mut hash = [salt; 32];
    hash[0..8].copy_from_slice(&(index as u64).to_le_bytes());
    hash
}

fn make_doc(doc_id: &str, index: usize) -> DocumentRecord {
    DocumentRecord::new(
        doc_id,
        "dedup bench preview",
        hash_for(index, 17),
        256,
        1_739_499_200,
        1_739_499_200,
    )
}

fn make_fixture(batch_size: usize) -> DedupFixture {
    let storage = Storage::open_in_memory().expect("storage should open");
    let mut items = Vec::with_capacity(batch_size);

    for index in 0..batch_size {
        let doc_id = format!("doc-{index:06}");
        match index % 5 {
            0 => {
                items.push((doc_id, hash_for(index, 91)));
            }
            1 => {
                let doc = make_doc(&doc_id, index);
                storage
                    .upsert_document(&doc)
                    .expect("embedded doc should insert");
                storage
                    .mark_embedded(&doc_id, EMBEDDER_ID)
                    .expect("embedded status should insert");
                items.push((doc_id, doc.content_hash));
            }
            2 => {
                let doc = make_doc(&doc_id, index);
                storage
                    .upsert_document(&doc)
                    .expect("changed doc should insert");
                items.push((doc_id, hash_for(index, 42)));
            }
            3 => {
                let doc = make_doc(&doc_id, index);
                storage
                    .upsert_document(&doc)
                    .expect("failed doc should insert");
                storage
                    .mark_failed(&doc_id, EMBEDDER_ID, "retry later")
                    .expect("failed status should insert");
                items.push((doc_id, doc.content_hash));
            }
            _ => {
                let doc = make_doc(&doc_id, index);
                storage
                    .upsert_document(&doc)
                    .expect("skipped doc should insert");
                storage
                    .mark_skipped(&doc_id, EMBEDDER_ID, "empty")
                    .expect("skipped status should insert");
                items.push((doc_id, doc.content_hash));
            }
        }
    }

    let production = storage
        .check_dedup_batch(&items, EMBEDDER_ID)
        .expect("production dedup should succeed");
    let legacy = legacy_check_dedup_batch(&storage, &items, EMBEDDER_ID)
        .expect("legacy dedup should succeed");
    let slot_join =
        slot_join_check_dedup_batch(&storage, &items, EMBEDDER_ID).expect("slot join should work");
    assert_eq!(production, legacy, "production and legacy mirror diverged");
    assert_eq!(production, slot_join, "production and slot join diverged");

    DedupFixture { storage, items }
}

fn legacy_check_dedup_batch(
    storage: &Storage,
    items: &[(String, [u8; 32])],
    embedder_id: &str,
) -> SearchResult<Vec<DeduplicationDecision>> {
    let mut seen_doc_ids = HashSet::with_capacity(items.len());
    for (doc_id, _) in items {
        if !seen_doc_ids.insert(doc_id.as_str()) {
            return Err(storage_error("duplicate doc_id in benchmark input"));
        }
    }

    storage.transaction(|conn| {
        let existing = legacy_fetch_existing_dedup_rows(conn, items, embedder_id)?;
        let mut decisions = Vec::with_capacity(items.len());

        for (doc_id, new_hash) in items {
            decisions.push(legacy_build_dedup_decision(
                doc_id,
                *new_hash,
                existing.get(doc_id),
            ));
        }

        Ok(decisions)
    })
}

fn legacy_fetch_existing_dedup_rows(
    conn: &AsyncConnection,
    items: &[(String, [u8; 32])],
    embedder_id: &str,
) -> SearchResult<HashMap<String, LegacyDedupRow>> {
    let mut sql = String::from(
        "SELECT d.doc_id, d.content_hash, e.status \
         FROM documents d \
         LEFT JOIN embedding_status e \
           ON d.doc_id = e.doc_id AND e.embedder_id = ?1 \
         WHERE d.doc_id IN (",
    );

    for index in 0..items.len() {
        if index > 0 {
            sql.push_str(", ");
        }
        let _ = write!(&mut sql, "?{}", index + 2);
    }
    sql.push_str(");");

    let mut params = Vec::with_capacity(items.len() + 1);
    params.push(SqliteValue::Text(embedder_id.to_owned().into()));
    for (doc_id, _) in items {
        params.push(SqliteValue::Text(doc_id.clone().into()));
    }

    let rows = conn
        .query_with_params_sync(&sql, &params)
        .map_err(fsqlite_error)?;
    let mut existing = HashMap::with_capacity(rows.len());

    for row in &rows {
        let doc_id = row_text(row, 0, "documents.doc_id")?.to_owned();
        let content_hash = row_blob_32(row, 1, "documents.content_hash")?;
        let raw_status = row_optional_text(row, 2)?;
        let status = raw_status.as_deref().and_then(parse_status);
        existing.insert(
            doc_id,
            LegacyDedupRow {
                content_hash,
                status,
            },
        );
    }

    Ok(existing)
}

fn slot_join_check_dedup_batch(
    storage: &Storage,
    items: &[(String, [u8; 32])],
    embedder_id: &str,
) -> SearchResult<Vec<DeduplicationDecision>> {
    let mut seen_doc_ids = HashSet::with_capacity(items.len());
    for (doc_id, _) in items {
        if !seen_doc_ids.insert(doc_id.as_str()) {
            return Err(storage_error("duplicate doc_id in benchmark input"));
        }
    }

    storage.transaction(|conn| {
        let existing = slot_join_fetch_existing_dedup_slots(conn, items, embedder_id)?;
        let mut decisions = Vec::with_capacity(items.len());

        for ((doc_id, new_hash), existing_row) in items.iter().zip(&existing) {
            decisions.push(legacy_build_dedup_decision(
                doc_id,
                *new_hash,
                existing_row.as_ref(),
            ));
        }

        Ok(decisions)
    })
}

fn slot_join_fetch_existing_dedup_slots(
    conn: &AsyncConnection,
    items: &[(String, [u8; 32])],
    embedder_id: &str,
) -> SearchResult<Vec<Option<LegacyDedupRow>>> {
    let mut sql = String::from("WITH requested(ord, doc_id) AS (VALUES ");

    for index in 0..items.len() {
        if index > 0 {
            sql.push_str(", ");
        }
        let _ = write!(&mut sql, "({index}, ?{})", index + 2);
    }
    sql.push_str(
        ") \
         SELECT requested.ord, d.content_hash, e.status \
         FROM requested \
         LEFT JOIN documents d ON d.doc_id = requested.doc_id \
         LEFT JOIN embedding_status e \
           ON d.doc_id = e.doc_id AND e.embedder_id = ?1 \
         ORDER BY requested.ord;",
    );

    let mut params = Vec::with_capacity(items.len() + 1);
    params.push(SqliteValue::Text(embedder_id.to_owned().into()));
    for (doc_id, _) in items {
        params.push(SqliteValue::Text(doc_id.clone().into()));
    }

    let rows = conn
        .query_with_params_sync(&sql, &params)
        .map_err(fsqlite_error)?;
    let mut existing = vec![None; items.len()];

    for row in &rows {
        let ord = row_i64(row, 0, "requested.ord")?;
        let index = usize::try_from(ord).map_err(|_| storage_error("negative requested ord"))?;
        if index >= items.len() {
            return Err(storage_error(format!(
                "requested ord {index} out of range for {} items",
                items.len()
            )));
        }

        let Some(content_hash) = row_optional_blob_32(row, 1, "documents.content_hash")? else {
            continue;
        };
        let raw_status = row_optional_text(row, 2)?;
        let status = raw_status.as_deref().and_then(parse_status);
        existing[index] = Some(LegacyDedupRow {
            content_hash,
            status,
        });
    }

    Ok(existing)
}

fn legacy_build_dedup_decision(
    doc_id: &str,
    new_hash: [u8; 32],
    existing: Option<&LegacyDedupRow>,
) -> DeduplicationDecision {
    let Some(existing) = existing else {
        return DeduplicationDecision::New {
            doc_id: doc_id.to_owned(),
        };
    };

    if existing.content_hash != new_hash {
        return DeduplicationDecision::Changed {
            doc_id: doc_id.to_owned(),
            old_hash: existing.content_hash,
            new_hash,
        };
    }

    match existing.status {
        Some(EmbeddingStatus::Embedded) => DeduplicationDecision::Skip {
            doc_id: doc_id.to_owned(),
            reason: "unchanged_content_already_embedded",
        },
        Some(EmbeddingStatus::Pending) => DeduplicationDecision::Skip {
            doc_id: doc_id.to_owned(),
            reason: "unchanged_content_already_pending",
        },
        Some(EmbeddingStatus::Skipped) => DeduplicationDecision::Skip {
            doc_id: doc_id.to_owned(),
            reason: "unchanged_content_previously_skipped",
        },
        Some(EmbeddingStatus::Failed) | None => DeduplicationDecision::New {
            doc_id: doc_id.to_owned(),
        },
    }
}

fn parse_status(value: &str) -> Option<EmbeddingStatus> {
    match value {
        "pending" => Some(EmbeddingStatus::Pending),
        "embedded" => Some(EmbeddingStatus::Embedded),
        "failed" => Some(EmbeddingStatus::Failed),
        "skipped" => Some(EmbeddingStatus::Skipped),
        _ => None,
    }
}

fn row_optional_text(row: &Row, index: usize) -> SearchResult<Option<String>> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(Some(value.to_string())),
        Some(SqliteValue::Null) => Ok(None),
        Some(other) => Err(storage_error(format!(
            "unexpected optional text: {other:?}"
        ))),
        None => Err(storage_error(format!(
            "missing optional text column {index}"
        ))),
    }
}

fn row_blob_32(row: &Row, index: usize, field: &str) -> SearchResult<[u8; 32]> {
    let blob = match row.get(index) {
        Some(SqliteValue::Blob(blob)) => blob,
        Some(other) => {
            return Err(storage_error(format!(
                "unexpected type for {field}: {other:?}"
            )));
        }
        None => return Err(storage_error(format!("missing column for {field}"))),
    };

    if blob.len() != 32 {
        return Err(storage_error(format!(
            "expected 32-byte blob for {field}, found {}",
            blob.len()
        )));
    }

    let mut out = [0_u8; 32];
    out.copy_from_slice(blob);
    Ok(out)
}

fn row_optional_blob_32(row: &Row, index: usize, field: &str) -> SearchResult<Option<[u8; 32]>> {
    match row.get(index) {
        Some(SqliteValue::Null) => Ok(None),
        Some(SqliteValue::Blob(_)) => row_blob_32(row, index, field).map(Some),
        Some(other) => Err(storage_error(format!(
            "unexpected optional blob for {field}: {other:?}"
        ))),
        None => Err(storage_error(format!("missing column for {field}"))),
    }
}

fn row_text<'a>(row: &'a Row, index: usize, field: &str) -> SearchResult<&'a str> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(value),
        Some(other) => Err(storage_error(format!(
            "unexpected type for {field}: {other:?}"
        ))),
        None => Err(storage_error(format!("missing column for {field}"))),
    }
}

fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(storage_error(format!(
            "unexpected type for {field}: {other:?}"
        ))),
        None => Err(storage_error(format!("missing column for {field}"))),
    }
}

fn fsqlite_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(source),
    }
}

fn storage_error(message: impl Into<String>) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(std::io::Error::other(message.into())),
    }
}

fn main() {
    const ROUNDS: usize = 31;
    const INNER: u32 = 1;

    eprintln!("[config] rounds={ROUNDS} inner={INNER} ratio=slot_join/production_original");
    for batch_size in [32_usize, 128, 384] {
        let fixture = make_fixture(batch_size);

        let production = || {
            black_box(
                fixture
                    .storage
                    .check_dedup_batch(black_box(&fixture.items), black_box(EMBEDDER_ID))
                    .expect("production dedup should succeed"),
            );
        };
        let slot_join = || {
            black_box(
                slot_join_check_dedup_batch(
                    black_box(&fixture.storage),
                    black_box(&fixture.items),
                    black_box(EMBEDDER_ID),
                )
                .expect("slot join dedup should succeed"),
            );
        };

        let null = paired_median_ratio(ROUNDS, INNER, production, production);
        let candidate = paired_median_ratio(ROUNDS, INNER, production, slot_join);
        let verdict = if candidate.median < null.p5 {
            "DECIDABLE_WIN"
        } else if candidate.median > null.p95 {
            "DECIDABLE_REGRESSION"
        } else {
            "INSIDE_NULL_FLOOR"
        };
        eprintln!(
            "[paired] batch={batch_size} null_median={:.6} null_p5={:.6} null_p95={:.6} candidate_median={:.6} candidate_p5={:.6} candidate_p95={:.6} verdict={verdict}",
            null.median, null.p5, null.p95, candidate.median, candidate.p5, candidate.p95,
        );
    }
}
