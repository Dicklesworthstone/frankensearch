use std::error::Error;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::PathBuf;
use std::time::Duration;

#[cfg(feature = "fts5")]
use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{AsyncConnection, ConnectionEnv, FrankenError};
#[cfg(feature = "fts5")]
use fsqlite_types::cx::Cx as FsqliteCx;
use serde::{Deserialize, Serialize};

use crate::metrics::{StorageMetrics, StorageMetricsSnapshot};
use crate::schema;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageConfig {
    pub db_path: PathBuf,
    pub wal_mode: bool,
    pub busy_timeout_ms: u64,
    pub raptorq_repair_symbols: u32,
    pub cache_size_pages: i32,
    pub concurrent_transactions: bool,
}

impl StorageConfig {
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            db_path: PathBuf::from(":memory:"),
            ..Self::default()
        }
    }

    fn begin_sql(&self) -> &'static str {
        if self.concurrent_transactions {
            "BEGIN CONCURRENT;"
        } else {
            "BEGIN;"
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("storage.sqlite3"),
            wal_mode: true,
            busy_timeout_ms: 5_000,
            // Disabled by default until upstream frankensqlite reserve-byte
            // balancing issues are resolved.
            raptorq_repair_symbols: 0,
            cache_size_pages: 2_000,
            concurrent_transactions: true,
        }
    }
}

pub struct Storage {
    conn: AsyncConnection,
    config: StorageConfig,
    metrics: StorageMetrics,
}

static FILE_BOOTSTRAP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// FrankenSQLite already retries contention while it opens the pager, but later
// connection-admission stages can still return a typed Busy/Locked error while
// a just-used peer connection releases its namespace state. Keep this outer
// handoff short and bounded: seven waits total 113 ms, enough to cross a normal
// scheduler handoff without turning persistent lock contention into a hang.
const CONNECTION_OPEN_MAX_ATTEMPTS: u32 = 8;
const CONNECTION_OPEN_RETRY_BASE_DELAY_MS: u64 = 1;
const CONNECTION_OPEN_RETRY_MAX_DELAY_MS: u64 = 50;

fn is_retryable_connection_open_error(error: &FrankenError) -> bool {
    // Prefer the engine's published retry family over a hand-rolled Busy
    // subset. BusySnapshot / WriteConflict / SerializationFailure /
    // PageBufferCapacityExhausted are the same transient class as Busy.
    error.is_transient()
}

pub(crate) fn is_transient_storage_error(error: &SearchError) -> bool {
    let mut current: Option<&(dyn Error + 'static)> = Some(error);
    while let Some(err) = current {
        if let Some(franken) = err.downcast_ref::<FrankenError>() {
            return franken.is_transient();
        }
        current = err.source();
    }
    false
}

pub(crate) fn retry_transient_storage<T>(
    mut op: impl FnMut() -> SearchResult<T>,
    context: &'static str,
) -> SearchResult<T> {
    let mut attempt = 1;
    loop {
        match op() {
            Ok(value) => return Ok(value),
            Err(error)
                if is_transient_storage_error(&error) && attempt < CONNECTION_OPEN_MAX_ATTEMPTS =>
            {
                let delay = connection_open_retry_delay(attempt);
                tracing::debug!(
                    target: "frankensearch.storage",
                    context,
                    next_attempt = attempt.saturating_add(1),
                    max_attempts = CONNECTION_OPEN_MAX_ATTEMPTS,
                    ?delay,
                    error = %error,
                    "retrying transient storage contention"
                );
                std::thread::sleep(delay);
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

fn connection_open_retry_delay(failed_attempt: u32) -> Duration {
    let growth = failed_attempt.saturating_sub(1).min(6);
    let delay_ms =
        (CONNECTION_OPEN_RETRY_BASE_DELAY_MS << growth).min(CONNECTION_OPEN_RETRY_MAX_DELAY_MS);
    Duration::from_millis(delay_ms)
}

fn retry_connection_open<T>(
    mut open: impl FnMut() -> Result<T, FrankenError>,
    mut on_retry: impl FnMut(u32, Duration, &FrankenError),
) -> Result<T, FrankenError> {
    let mut attempt = 1;
    loop {
        match open() {
            Ok(connection) => return Ok(connection),
            Err(error)
                if is_retryable_connection_open_error(&error)
                    && attempt < CONNECTION_OPEN_MAX_ATTEMPTS =>
            {
                let delay = connection_open_retry_delay(attempt);
                on_retry(attempt.saturating_add(1), delay, &error);
                std::thread::sleep(delay);
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(feature = "fts5")]
pub(crate) fn fsqlite_cx(cx: &Cx) -> FsqliteCx {
    let fsqlite_cx = FsqliteCx::new();
    fsqlite_cx.set_native_cx(cx.clone());
    fsqlite_cx
}

fn open_connection_with_retry(path: &str) -> Result<AsyncConnection, FrankenError> {
    let env = ConnectionEnv::default();
    retry_connection_open(
        || AsyncConnection::open_sync_with_env(path, env.clone()),
        |next_attempt, delay, error| {
            tracing::debug!(
                target: "frankensearch.storage",
                path,
                next_attempt,
                max_attempts = CONNECTION_OPEN_MAX_ATTEMPTS,
                ?delay,
                error = %error,
                "retrying transient storage connection-open contention"
            );
        },
    )
}

#[derive(Debug)]
struct StorageStageError<E> {
    stage: &'static str,
    source: E,
}

impl<E> fmt::Display for StorageStageError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.stage, self.source)
    }
}

impl<E> Error for StorageStageError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("path", &self.config.db_path)
            .field("wal_mode", &self.config.wal_mode)
            .field("busy_timeout_ms", &self.config.busy_timeout_ms)
            .finish_non_exhaustive()
    }
}

impl Storage {
    pub fn open(config: StorageConfig) -> SearchResult<Self> {
        tracing::debug!(
            target: "frankensearch.storage",
            path = %config.db_path.display(),
            wal_mode = config.wal_mode,
            busy_timeout_ms = config.busy_timeout_ms,
            cache_size_pages = config.cache_size_pages,
            concurrent_transactions = config.concurrent_transactions,
            "opening storage connection"
        );

        let file_bootstrap_guard = if config.db_path.as_os_str() == ":memory:" {
            None
        } else {
            Some(FILE_BOOTSTRAP_LOCK.lock().map_err(|_| {
                map_storage_error_at(
                    "file bootstrap lock",
                    std::io::Error::other("lock poisoned"),
                )
            })?)
        };

        let path = config.db_path.to_string_lossy().to_string();
        let conn = open_connection_with_retry(&path)
            .map_err(|error| map_storage_error_at("connection open", error))?;

        let storage = Self {
            conn,
            config,
            metrics: StorageMetrics::default(),
        };

        storage.metrics.record_open();
        storage.apply_pragmas()?;
        schema::bootstrap(storage.connection())?;
        storage.metrics.record_schema_bootstrap();

        let version =
            schema::current_version_at(storage.connection(), "post-open schema verification")?;
        drop(file_bootstrap_guard);
        tracing::debug!(
            target: "frankensearch.storage",
            schema_version = version,
            "storage bootstrap complete"
        );

        Ok(storage)
    }

    pub fn open_in_memory() -> SearchResult<Self> {
        Self::open(StorageConfig::in_memory())
    }

    /// Open a real in-memory connection without running schema bootstrap.
    ///
    /// Tests which exercise the fail-closed bootstrap gate itself need to
    /// install only the exact schema surface under test. Keeping this seam
    /// test-only prevents production callers from bypassing bootstrap.
    #[cfg(all(test, feature = "fts5"))]
    pub(crate) async fn open_unbootstrapped_in_memory_for_test(_cx: &Cx) -> SearchResult<Self> {
        let config = StorageConfig::in_memory();
        let path = config.db_path.to_string_lossy().to_string();
        let conn = open_connection_with_retry(&path)
            .map_err(|error| map_storage_error_at("test connection open", error))?;
        let storage = Self {
            conn,
            config,
            metrics: StorageMetrics::default(),
        };
        storage.apply_pragmas()?;
        Ok(storage)
    }

    #[must_use]
    pub fn connection(&self) -> &AsyncConnection {
        &self.conn
    }

    #[must_use]
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    #[must_use]
    pub fn metrics_snapshot(&self) -> StorageMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn transaction<F, T>(&self, f: F) -> SearchResult<T>
    where
        F: FnMut(&AsyncConnection) -> SearchResult<T>,
    {
        self.transaction_with_mode(self.config.begin_sql(), f)
    }

    /// Run a closure inside a `BEGIN IMMEDIATE` transaction.
    ///
    /// Unlike [`Storage::transaction`], this acquires a write lock immediately,
    /// preventing concurrent writers from interleaving reads and writes.
    /// Use this when correctness depends on serialized read-then-write
    /// (e.g. `claim_batch`).
    pub fn immediate_transaction<F, T>(&self, f: F) -> SearchResult<T>
    where
        F: FnMut(&AsyncConnection) -> SearchResult<T>,
    {
        self.transaction_with_mode("BEGIN IMMEDIATE;", f)
    }

    fn transaction_with_mode<F, T>(&self, begin_sql: &str, mut f: F) -> SearchResult<T>
    where
        F: FnMut(&AsyncConnection) -> SearchResult<T>,
    {
        retry_transient_storage(
            || self.transaction_with_mode_once(begin_sql, &mut f),
            "storage transaction",
        )
    }

    fn transaction_with_mode_once<F, T>(&self, begin_sql: &str, f: &mut F) -> SearchResult<T>
    where
        F: FnMut(&AsyncConnection) -> SearchResult<T>,
    {
        tracing::trace!(
            target: "frankensearch.storage",
            begin_sql,
            "starting storage transaction"
        );

        self.conn
            .execute_sync(begin_sql)
            .map_err(map_storage_error)?;

        let outcome = catch_unwind(AssertUnwindSafe(|| f(&self.conn)));
        match outcome {
            Ok(Ok(value)) => {
                self.conn.commit_transaction_sync().map_err(|commit_err| {
                    if let Err(rollback_err) = self.conn.rollback_transaction_sync() {
                        tracing::warn!(
                            target: "frankensearch.storage",
                            error = %rollback_err,
                            "rollback failed after commit error"
                        );
                    }
                    map_storage_error(commit_err)
                })?;
                self.metrics.record_commit();
                tracing::trace!(target: "frankensearch.storage", "storage transaction committed");
                Ok(value)
            }
            Ok(Err(error)) => {
                if let Err(rollback_err) = self.conn.rollback_transaction_sync() {
                    tracing::warn!(
                        target: "frankensearch.storage",
                        error = %rollback_err,
                        "rollback failed after closure error"
                    );
                }
                self.metrics.record_rollback();
                tracing::debug!(
                    target: "frankensearch.storage",
                    ?error,
                    "storage transaction rolled back due to closure error"
                );
                Err(error)
            }
            Err(payload) => {
                if let Err(rollback_err) = self.conn.rollback_transaction_sync() {
                    tracing::error!(
                        target: "frankensearch.storage",
                        error = %rollback_err,
                        "critical: rollback failed during panic recovery"
                    );
                }
                self.metrics.record_rollback();
                tracing::error!(
                    target: "frankensearch.storage",
                    "storage transaction rolled back after panic"
                );
                resume_unwind(payload);
            }
        }
    }

    fn apply_pragmas(&self) -> SearchResult<()> {
        tracing::trace!(
            target: "frankensearch.storage",
            wal_mode = self.config.wal_mode,
            busy_timeout_ms = self.config.busy_timeout_ms,
            cache_size_pages = self.config.cache_size_pages,
            raptorq_repair_symbols = self.config.raptorq_repair_symbols,
            "applying storage pragmas"
        );

        self.conn
            .execute_sync("PRAGMA foreign_keys=ON;")
            .map_err(|error| map_storage_error_at("apply foreign_keys pragma", error))?;

        if self.config.wal_mode {
            self.conn
                .execute_sync("PRAGMA journal_mode=WAL;")
                .map_err(|error| map_storage_error_at("apply journal_mode pragma", error))?;
        } else if let Err(error) = self.conn.execute_sync("PRAGMA journal_mode=DELETE;") {
            tracing::warn!(
                target: "frankensearch.storage",
                ?error,
                "journal_mode=DELETE was not accepted by backend; falling back to WAL"
            );
            self.conn
                .execute_sync("PRAGMA journal_mode=WAL;")
                .map_err(|error| {
                    map_storage_error_at("apply journal_mode fallback pragma", error)
                })?;
        }

        self.conn
            .execute_sync(&format!(
                "PRAGMA busy_timeout={};",
                self.config.busy_timeout_ms
            ))
            .map_err(|error| map_storage_error_at("apply busy_timeout pragma", error))?;

        self.conn
            .execute_sync(&format!(
                "PRAGMA cache_size={};",
                self.config.cache_size_pages
            ))
            .map_err(|error| map_storage_error_at("apply cache_size pragma", error))?;

        if self.config.raptorq_repair_symbols > 0 {
            tracing::warn!(
                target: "frankensearch.storage",
                symbols = self.config.raptorq_repair_symbols,
                "raptorq FEC repair symbols requested but currently disabled: \
                 upstream frankensqlite reserve-byte balancing is unresolved; \
                 configure raptorq_repair_symbols=0 to suppress this warning"
            );
        }

        Ok(())
    }
}

pub(crate) fn map_storage_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    map_storage_error_at("storage operation", source)
}

pub(crate) fn map_storage_error_at<E>(stage: &'static str, source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(StorageStageError { stage, source }),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{self, AssertUnwindSafe};
    use std::path::{Path, PathBuf};
    use std::process;
    use std::sync::{Arc, Barrier, mpsc};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use frankensearch_core::{SearchError, SearchResult};
    use fsqlite::FrankenError;
    use fsqlite_types::value::SqliteValue;
    use serde_json::json;

    use crate::document::{DocumentRecord, count_documents, upsert_document};
    use crate::schema::{self, SCHEMA_VERSION};

    use super::{Storage, StorageConfig};

    struct TempDbPath {
        path: PathBuf,
    }

    impl TempDbPath {
        fn new(tag: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "frankensearch-storage-{tag}-{}-{nanos}.sqlite3",
                process::id()
            ));
            Self { path }
        }

        fn config(&self) -> StorageConfig {
            StorageConfig {
                db_path: self.path.clone(),
                ..StorageConfig::default()
            }
        }
    }

    impl Drop for TempDbPath {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let candidate = if suffix.is_empty() {
                    self.path.clone()
                } else {
                    PathBuf::from(format!("{}{}", self.path.display(), suffix))
                };
                let _ = std::fs::remove_file(candidate);
            }
        }
    }

    fn sample_document(id: &str) -> DocumentRecord {
        let mut doc = DocumentRecord::new(
            id,
            "Rust ownership keeps heap usage explicit during indexing.",
            [0x2a; 32],
            128,
            1_739_499_200,
            1_739_499_200,
        );
        doc.source_path = Some("tests://fixture".to_owned());
        doc.metadata = Some(json!({"fixture": true}));
        doc
    }

    const CONCURRENT_OPEN_THREADS: usize = 4;
    const CONCURRENT_OPEN_STRESS_ROUNDS: usize = 100;
    const REPEATED_CURRENT_SCHEMA_OPENS: usize = 32;

    fn run_concurrent_open_round(config: &StorageConfig) -> Vec<SearchResult<i64>> {
        let barrier = Arc::new(Barrier::new(CONCURRENT_OPEN_THREADS));
        let (tx, rx) = mpsc::channel::<SearchResult<i64>>();
        let mut handles = Vec::with_capacity(CONCURRENT_OPEN_THREADS);

        for _ in 0..CONCURRENT_OPEN_THREADS {
            let cfg = config.clone();
            let gate = Arc::clone(&barrier);
            let sender = tx.clone();
            handles.push(thread::spawn(move || {
                gate.wait();
                let open_result = Storage::open(cfg)
                    .and_then(|storage| schema::current_version(storage.connection()));
                sender
                    .send(open_result)
                    .expect("result send should succeed");
            }));
        }
        drop(tx);

        let results: Vec<SearchResult<i64>> = rx.into_iter().collect();
        for handle in handles {
            handle.join().expect("concurrent opener thread should join");
        }
        results
    }

    fn insert_document_minimal(conn: &fsqlite::AsyncConnection, id: &str) -> SearchResult<()> {
        let params = [
            SqliteValue::Text(id.to_owned().into()),
            SqliteValue::Text("fixture-content".to_owned().into()),
            SqliteValue::Blob(vec![0x33; 32].into()),
            SqliteValue::Integer(64),
            SqliteValue::Integer(1_739_499_200),
            SqliteValue::Integer(1_739_499_200),
        ];
        conn.execute_with_params_sync(
            "INSERT INTO documents (\
            doc_id, content_preview, content_hash, content_length, created_at, updated_at\
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6);",
            &params,
        )
        .map_err(super::map_storage_error)?;
        Ok(())
    }

    fn pragma_i64(conn: &fsqlite::AsyncConnection, name: &str) -> SearchResult<i64> {
        let rows = conn
            .query_sync(&format!("PRAGMA {name};"))
            .map_err(super::map_storage_error)?;
        let row = rows.first().ok_or_else(|| SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(std::io::Error::other(format!(
                "PRAGMA {name} returned no row"
            ))),
        })?;
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => Ok(*value),
            Some(SqliteValue::Text(value)) => {
                value
                    .parse::<i64>()
                    .map_err(|parse_error| SearchError::SubsystemError {
                        subsystem: "storage",
                        source: Box::new(parse_error),
                    })
            }
            Some(other) => Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(std::io::Error::other(format!(
                    "unexpected pragma type for {name}: {:?}",
                    other
                ))),
            }),
            None => Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(std::io::Error::other(format!(
                    "missing pragma value for {name}"
                ))),
            }),
        }
    }

    fn pragma_text(conn: &fsqlite::AsyncConnection, name: &str) -> SearchResult<String> {
        let rows = conn
            .query_sync(&format!("PRAGMA {name};"))
            .map_err(super::map_storage_error)?;
        let row = rows.first().ok_or_else(|| SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(std::io::Error::other(format!(
                "PRAGMA {name} returned no row"
            ))),
        })?;
        match row.get(0) {
            Some(SqliteValue::Text(value)) => Ok(value.to_string()),
            Some(SqliteValue::Integer(value)) => Ok(value.to_string()),
            Some(other) => Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(std::io::Error::other(format!(
                    "unexpected pragma type for {name}: {:?}",
                    other
                ))),
            }),
            None => Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(std::io::Error::other(format!(
                    "missing pragma value for {name}"
                ))),
            }),
        }
    }

    fn checkpoint_metric(conn: &fsqlite::AsyncConnection, name: &str) -> SearchResult<i64> {
        let rows = conn
            .query_sync("PRAGMA fsqlite_checkpoint_stats;")
            .map_err(super::map_storage_error)?;
        for row in rows {
            let Some(SqliteValue::Text(metric_name)) = row.get(0) else {
                continue;
            };
            if metric_name.as_ref() != name {
                continue;
            }
            return match row.get(1) {
                Some(SqliteValue::Integer(value)) => Ok(*value),
                Some(other) => Err(SearchError::SubsystemError {
                    subsystem: "storage",
                    source: Box::new(std::io::Error::other(format!(
                        "unexpected checkpoint metric type for {name}: {other:?}"
                    ))),
                }),
                None => Err(SearchError::SubsystemError {
                    subsystem: "storage",
                    source: Box::new(std::io::Error::other(format!(
                        "missing checkpoint metric value for {name}"
                    ))),
                }),
            };
        }
        Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(std::io::Error::other(format!(
                "checkpoint metric {name} was not reported"
            ))),
        })
    }

    fn schema_version_markers(conn: &fsqlite::AsyncConnection) -> SearchResult<Vec<i64>> {
        conn.query_sync("SELECT version FROM schema_version ORDER BY version;")
            .map_err(super::map_storage_error)?
            .iter()
            .map(|row| schema::row_i64(row, 0, "schema_version.version"))
            .collect()
    }

    fn artifact_len(path: &Path) -> u64 {
        match std::fs::metadata(path) {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => panic!(
                "metadata for {} should be readable: {error}",
                path.display()
            ),
        }
    }

    fn wal_artifact_lengths(db_path: &Path) -> [u64; 2] {
        let dot_wal = PathBuf::from(format!("{}.wal", db_path.display()));
        let dash_wal = PathBuf::from(format!("{}-wal", db_path.display()));
        [artifact_len(&dot_wal), artifact_len(&dash_wal)]
    }

    #[test]
    fn connection_open_retry_exhaustion_preserves_typed_busy_error() {
        let mut attempts = 0_u32;
        let mut observed_retries = Vec::new();

        let error = super::retry_connection_open(
            || {
                attempts = attempts.saturating_add(1);
                Err::<(), _>(FrankenError::Busy)
            },
            |next_attempt, delay, error| {
                assert!(matches!(error, FrankenError::Busy));
                observed_retries.push((next_attempt, delay));
            },
        )
        .expect_err("persistent Busy must exhaust the bounded retry budget");

        assert!(matches!(error, FrankenError::Busy));
        assert_eq!(attempts, super::CONNECTION_OPEN_MAX_ATTEMPTS);
        let expected_retries: Vec<_> = (1..super::CONNECTION_OPEN_MAX_ATTEMPTS)
            .map(|failed_attempt| {
                (
                    failed_attempt.saturating_add(1),
                    super::connection_open_retry_delay(failed_attempt),
                )
            })
            .collect();
        assert_eq!(observed_retries, expected_retries);
    }

    #[test]
    fn connection_open_retry_retries_busy_snapshot() {
        let mut attempts = 0_u32;

        let error = super::retry_connection_open(
            || {
                attempts = attempts.saturating_add(1);
                Err::<(), _>(FrankenError::BusySnapshot {
                    conflicting_pages: "1,2".to_owned(),
                })
            },
            |_, _, _| {},
        )
        .expect_err("persistent BusySnapshot must exhaust the bounded retry budget");

        assert!(matches!(error, FrankenError::BusySnapshot { .. }));
        assert_eq!(attempts, super::CONNECTION_OPEN_MAX_ATTEMPTS);
    }

    #[test]
    fn transient_storage_error_detects_wrapped_busy_snapshot() {
        let error = super::map_storage_error(FrankenError::BusySnapshot {
            conflicting_pages: "1,2".to_owned(),
        });
        assert!(super::is_transient_storage_error(&error));
    }

    #[test]
    fn transient_storage_error_rejects_invalid_config() {
        let error = SearchError::InvalidConfig {
            field: "test".to_owned(),
            value: "busy snapshot locked".to_owned(),
            reason: "string match must not classify this as transient".to_owned(),
        };
        assert!(!super::is_transient_storage_error(&error));
    }

    #[test]
    fn retry_transient_storage_retries_busy_snapshot_then_succeeds() {
        let mut attempts = 0_u32;
        let value = super::retry_transient_storage(
            || {
                attempts = attempts.saturating_add(1);
                if attempts < 3 {
                    Err(super::map_storage_error(FrankenError::BusySnapshot {
                        conflicting_pages: "p1".to_owned(),
                    }))
                } else {
                    Ok(7_i32)
                }
            },
            "test",
        )
        .expect("BusySnapshot should succeed after bounded retries");
        assert_eq!(value, 7);
        assert_eq!(attempts, 3);
    }

    #[test]
    fn connection_open_retry_propagates_non_busy_error_without_retry() {
        let mut attempts = 0_u32;
        let mut retries = 0_u32;

        let error = super::retry_connection_open(
            || {
                attempts = attempts.saturating_add(1);
                Err::<(), _>(FrankenError::NoSuchTable {
                    name: "unrelated".to_owned(),
                })
            },
            |_, _, _| retries = retries.saturating_add(1),
        )
        .expect_err("non-Busy connection errors must fail immediately");

        assert!(matches!(error, FrankenError::NoSuchTable { name } if name == "unrelated"));
        assert_eq!(attempts, 1);
        assert_eq!(retries, 0);
    }

    #[test]
    fn open_in_memory_bootstraps_schema() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let version = schema::current_version(storage.connection()).expect("schema version row");
        assert_eq!(version, SCHEMA_VERSION);

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.opens, 1);
        assert_eq!(metrics.schema_bootstraps, 1);
    }

    #[test]
    fn open_applies_configured_pragmas_deterministically() {
        let tmp = TempDbPath::new("pragmas");
        let storage = Storage::open(StorageConfig {
            db_path: tmp.path.clone(),
            wal_mode: true,
            busy_timeout_ms: 1_234,
            cache_size_pages: 321,
            concurrent_transactions: false,
            ..StorageConfig::default()
        })
        .expect("storage should open with configured pragmas");

        assert_eq!(
            pragma_text(storage.connection(), "journal_mode")
                .expect("journal_mode pragma should be queryable")
                .to_ascii_lowercase(),
            "wal"
        );
        assert_eq!(
            pragma_i64(storage.connection(), "busy_timeout")
                .expect("busy_timeout pragma should be queryable"),
            1_234
        );
        assert_eq!(
            pragma_i64(storage.connection(), "cache_size")
                .expect("cache_size pragma should be queryable"),
            321
        );
    }

    #[test]
    fn schema_bootstrap_is_idempotent() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        schema::bootstrap(storage.connection()).expect("second bootstrap should succeed");
        schema::bootstrap(storage.connection()).expect("third bootstrap should succeed");

        let version = schema::current_version(storage.connection()).expect("schema version row");
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn concurrent_open_initializes_schema_consistently() {
        let tmp = TempDbPath::new("concurrent-bootstrap");
        let config = StorageConfig {
            busy_timeout_ms: 0,
            ..tmp.config()
        };
        let results = run_concurrent_open_round(&config);
        assert_eq!(results.len(), CONCURRENT_OPEN_THREADS);
        for (thread_idx, result) in results.into_iter().enumerate() {
            match result {
                Ok(version) => assert_eq!(
                    version, SCHEMA_VERSION,
                    "thread {thread_idx} should observe latest schema version"
                ),
                Err(error) => panic!("thread {thread_idx} failed concurrent open: {error}"),
            }
        }
    }

    #[test]
    fn concurrent_open_stays_stable_across_repeated_rounds() {
        let tmp = TempDbPath::new("concurrent-bootstrap-rounds");
        let config = StorageConfig {
            busy_timeout_ms: 0,
            ..tmp.config()
        };

        let seed = Storage::open(config.clone()).expect("schema-v6 seed should open");
        assert_eq!(
            schema::current_version(seed.connection()).expect("seed version should read"),
            SCHEMA_VERSION
        );
        drop(seed);

        for round in 0..CONCURRENT_OPEN_STRESS_ROUNDS {
            let results = run_concurrent_open_round(&config);
            assert_eq!(
                results.len(),
                CONCURRENT_OPEN_THREADS,
                "round {round} should return one result per thread"
            );
            for (thread_idx, result) in results.into_iter().enumerate() {
                match result {
                    Ok(version) => assert_eq!(
                        version, SCHEMA_VERSION,
                        "round {round}, thread {thread_idx} should observe latest schema version"
                    ),
                    Err(error) => {
                        panic!("round {round}, thread {thread_idx} failed concurrent open: {error}")
                    }
                }
            }
        }
    }

    #[test]
    fn repeated_current_schema_opens_do_not_mutate_marker_or_grow_wal() {
        let tmp = TempDbPath::new("current-schema-read-only");
        let config = StorageConfig {
            busy_timeout_ms: 0,
            ..tmp.config()
        };
        let seed = Storage::open(config.clone()).expect("schema-v6 seed should open");
        seed.connection()
            .execute_sync("PRAGMA wal_autocheckpoint=0;")
            .expect("test should disable automatic checkpoints");

        let before_frames = checkpoint_metric(seed.connection(), "wal_frames_estimate")
            .expect("baseline WAL frame count should be observable");
        let before_lengths = wal_artifact_lengths(&tmp.path);
        let before_markers = schema_version_markers(seed.connection())
            .expect("baseline schema markers should be readable");

        let mut opened = Vec::with_capacity(REPEATED_CURRENT_SCHEMA_OPENS);
        for iteration in 0..REPEATED_CURRENT_SCHEMA_OPENS {
            let storage = Storage::open(config.clone()).unwrap_or_else(|error| {
                panic!("current-schema open {iteration} should succeed: {error}")
            });
            assert_eq!(
                schema::current_version(storage.connection())
                    .expect("current-schema version should remain readable"),
                SCHEMA_VERSION,
                "open {iteration} should observe v6"
            );
            opened.push(storage);
        }

        let after_frames = checkpoint_metric(seed.connection(), "wal_frames_estimate")
            .expect("final WAL frame count should be observable");
        let after_lengths = wal_artifact_lengths(&tmp.path);
        let after_markers = schema_version_markers(seed.connection())
            .expect("final schema markers should be readable");

        assert_eq!(
            after_frames, before_frames,
            "read-only current-schema opens must not append WAL frames"
        );
        assert_eq!(
            after_lengths, before_lengths,
            "read-only current-schema opens must not grow either supported WAL artifact"
        );
        assert_eq!(
            after_markers, before_markers,
            "read-only current-schema opens must not rewrite schema markers"
        );

        drop(opened);
    }

    #[test]
    fn transaction_rolls_back_on_error() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = sample_document("doc-error");

        let result: SearchResult<()> = storage.transaction(|conn| {
            upsert_document(conn, &doc)?;
            Err(SearchError::InvalidConfig {
                field: "test".to_owned(),
                value: "forced".to_owned(),
                reason: "force rollback".to_owned(),
            })
        });

        assert!(result.is_err(), "transaction should return original error");
        assert_eq!(
            count_documents(storage.connection()).expect("count should work"),
            0,
            "document insert should have been rolled back"
        );

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_commits, 0);
        assert_eq!(metrics.tx_rollbacks, 1);
    }

    #[test]
    fn transaction_rolls_back_on_panic_and_connection_stays_usable() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = sample_document("doc-panic");

        let panic_result = panic::catch_unwind(AssertUnwindSafe(|| {
            let _: SearchResult<()> = storage.transaction(|conn| {
                upsert_document(conn, &doc).expect("insert should succeed before panic");
                panic!("forced panic");
            });
        }));

        assert!(panic_result.is_err(), "panic should propagate to caller");
        assert_eq!(
            count_documents(storage.connection()).expect("count should work"),
            0,
            "panic path should rollback transaction"
        );
        assert_eq!(
            schema::current_version(storage.connection()).expect("connection should remain usable"),
            SCHEMA_VERSION
        );

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_rollbacks, 1);
        assert_eq!(metrics.tx_commits, 0);
    }

    #[test]
    fn commit_persists_after_reopen() {
        let tmp = TempDbPath::new("visibility");
        let config = tmp.config();

        let storage_a = Storage::open(config.clone()).expect("writer storage should open");
        let doc = sample_document("doc-visible");

        storage_a
            .transaction(|conn| {
                insert_document_minimal(conn, &doc.doc_id)?;
                assert_eq!(
                    count_documents(conn)?,
                    1,
                    "write should be visible inside writer transaction"
                );
                Ok(())
            })
            .expect("transaction should commit");

        drop(storage_a);

        let storage_c = Storage::open(config).expect("post-commit storage should open");
        assert_eq!(
            count_documents(storage_c.connection()).expect("count after commit"),
            1,
            "committed write should be visible to newly opened connection"
        );
    }

    #[test]
    fn rollback_is_not_persisted_after_reopen() {
        let tmp = TempDbPath::new("rollback");
        let config = tmp.config();

        let storage_a = Storage::open(config).expect("writer storage should open");
        let doc_id = "doc-rollback";

        let tx_result: SearchResult<()> = storage_a.transaction(|conn| {
            insert_document_minimal(conn, doc_id)?;
            Err(SearchError::InvalidConfig {
                field: "test".to_owned(),
                value: "rollback".to_owned(),
                reason: "forced rollback".to_owned(),
            })
        });
        assert!(tx_result.is_err(), "transaction should rollback");
        assert_eq!(
            count_documents(storage_a.connection()).expect("count after rollback"),
            0,
            "rolled back write should not persist in active connection state"
        );
    }

    #[test]
    fn dropped_connection_allows_reopen_and_write() {
        let tmp = TempDbPath::new("drop");
        let config = tmp.config();

        let storage_a = Storage::open(config.clone()).expect("first storage should open");
        drop(storage_a);

        let storage_b = Storage::open(config).expect("second storage should open");

        storage_b
            .transaction(|conn| {
                insert_document_minimal(conn, "doc-after-drop")?;
                Ok(())
            })
            .expect("second connection should remain fully usable after first drop");

        assert_eq!(
            count_documents(storage_b.connection()).expect("count after write"),
            1
        );
    }

    #[test]
    fn nested_transaction_error_rolls_back_outer_transaction() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        let result: SearchResult<()> = storage.transaction(|conn| {
            insert_document_minimal(conn, "doc-nested")?;
            storage.transaction(|_nested_conn| Ok(()))
        });

        assert!(
            result.is_err(),
            "nested transaction should fail and propagate error"
        );
        assert_eq!(
            count_documents(storage.connection()).expect("count after nested failure"),
            0,
            "outer transaction should rollback when nested begin fails"
        );

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_rollbacks, 1);
        assert_eq!(metrics.tx_commits, 0);
    }

    #[test]
    fn uncommitted_writes_are_invisible_to_concurrent_reader() {
        let tmp = TempDbPath::new("snapshot");
        let config = tmp.config();
        let reader = Storage::open(config.clone()).expect("reader storage should open");
        let writer_config = config;

        let (inserted_tx, inserted_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();

        let writer_thread = thread::spawn(move || {
            let writer = Storage::open(writer_config).expect("writer storage should open");
            writer
                .transaction(|conn| {
                    insert_document_minimal(conn, "doc-uncommitted")?;
                    inserted_tx
                        .send(())
                        .expect("writer should signal insert before commit");
                    release_rx
                        .recv()
                        .expect("writer should wait for reader inspection");
                    Ok(())
                })
                .expect("writer transaction should commit");
        });

        inserted_rx
            .recv()
            .expect("reader should observe insert signal");

        assert_eq!(
            count_documents(reader.connection()).expect("reader count during writer transaction"),
            0,
            "reader should not see uncommitted write from concurrent transaction"
        );

        release_tx
            .send(())
            .expect("reader should release writer to commit");
        writer_thread.join().expect("writer thread should join");

        // In WAL mode without an explicit read transaction, each statement
        // sees the latest committed state.  After the writer commits, the
        // reader observes the new row immediately.
        assert_eq!(
            count_documents(reader.connection()).expect("reader count after writer commit"),
            1,
            "reader without explicit transaction should see committed write"
        );
    }

    #[test]
    fn storage_config_serde_roundtrip() {
        let config = StorageConfig {
            db_path: PathBuf::from("/tmp/test.db"),
            wal_mode: false,
            busy_timeout_ms: 10_000,
            raptorq_repair_symbols: 4,
            cache_size_pages: 500,
            concurrent_transactions: false,
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: StorageConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config, deserialized);
    }

    #[test]
    fn storage_config_in_memory_path() {
        let config = StorageConfig::in_memory();
        assert_eq!(config.db_path, PathBuf::from(":memory:"));
        assert!(config.wal_mode);
        assert!(config.concurrent_transactions);
    }

    #[test]
    fn storage_config_begin_sql_concurrent_vs_standard() {
        let concurrent = StorageConfig {
            concurrent_transactions: true,
            ..StorageConfig::default()
        };
        assert_eq!(concurrent.begin_sql(), "BEGIN CONCURRENT;");

        let standard = StorageConfig {
            concurrent_transactions: false,
            ..StorageConfig::default()
        };
        assert_eq!(standard.begin_sql(), "BEGIN;");
    }

    #[test]
    fn immediate_transaction_commits_successfully() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        storage
            .immediate_transaction(|conn| {
                insert_document_minimal(conn, "doc-immediate")?;
                Ok(())
            })
            .expect("immediate transaction should commit");

        assert_eq!(
            count_documents(storage.connection()).expect("count after immediate tx"),
            1
        );

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_commits, 1);
    }

    #[test]
    fn immediate_transaction_rolls_back_on_error() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        let result: SearchResult<()> = storage.immediate_transaction(|conn| {
            insert_document_minimal(conn, "doc-immediate-rollback")?;
            Err(SearchError::InvalidConfig {
                field: "test".to_owned(),
                value: "forced".to_owned(),
                reason: "force rollback".to_owned(),
            })
        });

        assert!(result.is_err());
        assert_eq!(
            count_documents(storage.connection()).expect("count after rollback"),
            0
        );

        let metrics = storage.metrics_snapshot();
        assert_eq!(metrics.tx_rollbacks, 1);
    }

    #[test]
    fn storage_debug_format() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let debug = format!("{storage:?}");
        assert!(debug.contains("Storage"));
        assert!(debug.contains(":memory:"));
    }

    #[test]
    fn multi_threaded_serial_writers_commit_disjoint_documents() {
        let tmp = TempDbPath::new("concurrent-writers");
        let config = tmp.config();
        let (first_done_tx, first_done_rx) = mpsc::channel::<()>();

        let cfg_a = config.clone();
        let first_writer = thread::spawn(move || {
            let storage = Storage::open(cfg_a).expect("first writer storage should open");
            storage
                .transaction(|conn| {
                    insert_document_minimal(conn, "doc-a")?;
                    Ok(())
                })
                .expect("first writer transaction should commit");
            first_done_tx
                .send(())
                .expect("first writer should signal completion");
        });

        let cfg_b = config.clone();
        let second_writer = thread::spawn(move || {
            first_done_rx
                .recv()
                .expect("second writer should wait for first writer");
            let storage = Storage::open(cfg_b).expect("second writer storage should open");
            storage
                .transaction(|conn| {
                    insert_document_minimal(conn, "doc-b")?;
                    Ok(())
                })
                .expect("second writer transaction should commit");
        });

        first_writer
            .join()
            .expect("first writer thread should not panic");
        second_writer
            .join()
            .expect("second writer thread should not panic");

        let verifier = Storage::open(config).expect("verifier storage should open");
        assert_eq!(
            count_documents(verifier.connection()).expect("count after concurrent writes"),
            2,
            "both writer transactions should persist exactly one row"
        );
    }
}
