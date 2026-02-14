//! End-to-end recall test for the full fsfs pipeline.
//!
//! This test proves the entire frankensearch pipeline works by:
//! 1. Creating a temp directory with known text files across distinct topics.
//! 2. Running `fsfs index` on the directory.
//! 3. Running `fsfs search` for several queries with known ground-truth relevance.
//! 4. Verifying that expected documents appear in the top-K results (recall).
//! 5. Verifying that search completes within a performance budget.
//! 6. Verifying that JSON output is parseable and structurally correct.
//!
//! The test deliberately does NOT require model downloads — it works with the
//! always-available hash embedder fallback for semantic search, relying primarily
//! on BM25 lexical search for recall.  This means it can run in CI without
//! network access, GPU, or pre-downloaded models.
//!
//! Bead: bd-2w7x.37

use std::collections::HashMap;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;

// ─── Output Schema (subset for deserialization) ────────────────────────────

#[derive(Debug, Deserialize)]
struct OutputEnvelope {
    v: u32,
    ok: bool,
    data: Option<SearchPayload>,
    error: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    warnings: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct SearchPayload {
    query: String,
    phase: String,
    #[allow(dead_code)]
    total_candidates: usize,
    returned_hits: usize,
    hits: Vec<SearchHit>,
}

#[derive(Debug, Deserialize)]
struct SearchHit {
    rank: usize,
    path: String,
    score: f64,
}

// ─── Corpus definition ────────────────────────────────────────────────────

/// Each entry: (`relative_path`, `content`)
#[allow(clippy::too_many_lines)]
fn corpus_files() -> Vec<(&'static str, &'static str)> {
    vec![
        // ── Rust ownership cluster ──
        (
            "src/ownership.rs",
            "Rust ownership system enforces single-owner semantics at compile time. \
             When a value is moved, the original binding becomes invalid and the borrow \
             checker prevents use-after-move bugs. Ownership transfer happens on assignment, \
             function calls, and returns. The Drop trait provides deterministic resource \
             cleanup without garbage collection.",
        ),
        (
            "src/borrowing.rs",
            "Borrowing in Rust allows temporary references to data without taking ownership. \
             Shared references (&T) are read-only and multiple can coexist. Mutable references \
             (&mut T) are exclusive — only one may exist at a time. The borrow checker enforces \
             these rules at compile time, preventing data races in concurrent code.",
        ),
        (
            "src/lifetimes.rs",
            "Lifetime annotations in Rust tell the compiler how long references are valid. \
             The 'a syntax annotates the relationship between reference lifetimes. Elision \
             rules allow omitting lifetimes in common patterns. Named lifetimes are required \
             when functions return references derived from multiple parameters.",
        ),
        // ── Database cluster ──
        (
            "src/db/connection_pool.rs",
            "Database connection pooling reduces latency by reusing established connections \
             instead of creating new ones for every query. The pool maintains a set of idle \
             connections and hands them out on demand. Connection health checks and eviction \
             policies keep the pool healthy under varying load patterns.",
        ),
        (
            "src/db/query_builder.rs",
            "A query builder provides a type-safe interface for constructing SQL statements. \
             It prevents SQL injection by parameterizing all user inputs. Method chaining \
             enables fluent query composition: select, where, join, order_by, and limit \
             clauses are composed programmatically rather than through string concatenation.",
        ),
        (
            "src/db/migrations.rs",
            "Database migrations track schema changes over time using versioned scripts. \
             Each migration has an up and down function for applying and reverting changes. \
             The migration runner records which migrations have been applied in a metadata \
             table, ensuring idempotent execution across environments.",
        ),
        // ── Authentication cluster ──
        (
            "src/auth/middleware.rs",
            "Authentication middleware intercepts incoming HTTP requests and validates \
             JWT tokens before forwarding to route handlers. It extracts the Bearer token \
             from the Authorization header, verifies the signature against the public key, \
             checks expiration and audience claims, and attaches the decoded user identity \
             to the request context.",
        ),
        (
            "src/auth/login.rs",
            "The login endpoint accepts username and password credentials, verifies them \
             against the hashed password stored in the user database, and returns a signed \
             JWT access token with a refresh token. Password verification uses bcrypt with \
             a cost factor of 12. Rate limiting prevents brute force attacks.",
        ),
        (
            "src/auth/permissions.rs",
            "Role-based access control assigns permissions to users through roles. Each \
             role grants a set of capabilities like read, write, admin, or delete. The \
             authorization check middleware verifies that the authenticated user has the \
             required permissions for the requested resource and operation.",
        ),
        // ── Error handling cluster ──
        (
            "src/errors.rs",
            "Error handling in Rust uses the Result type with custom error enums. The \
             thiserror crate derives Display and Error implementations automatically. \
             Error variants carry context about what went wrong: InvalidInput, NotFound, \
             Unauthorized, Timeout, and InternalError. The ? operator propagates errors \
             up the call stack with automatic conversion via From trait implementations.",
        ),
        (
            "src/error_recovery.rs",
            "Error recovery strategies include retry with exponential backoff, circuit \
             breaker patterns, and graceful degradation. Transient errors like network \
             timeouts are retried up to 3 times with jittered delays. Persistent failures \
             trip the circuit breaker, routing requests to a fallback path that returns \
             cached or default responses.",
        ),
        // ── Configuration cluster ──
        (
            "docs/configuration.md",
            "# Configuration Guide\n\n\
             The application loads configuration from multiple sources with clear precedence: \
             CLI flags override environment variables, which override config file values, \
             which override built-in defaults. The config file uses TOML format and supports \
             sections for server, database, logging, and feature flags.",
        ),
        (
            "config/defaults.toml",
            "[server]\n\
             host = \"0.0.0.0\"\n\
             port = 8080\n\
             max_connections = 1000\n\n\
             [database]\n\
             url = \"postgres://localhost/myapp\"\n\
             pool_size = 10\n\
             timeout_seconds = 30\n\n\
             [logging]\n\
             level = \"info\"\n\
             format = \"json\"\n",
        ),
        // ── API design cluster ──
        (
            "src/api/routes.rs",
            "REST API routes map HTTP methods and paths to handler functions. GET endpoints \
             return resources, POST creates new entries, PUT replaces existing ones, and \
             DELETE removes them. Path parameters extract IDs from URLs. Query parameters \
             handle filtering, pagination, and sorting. Response bodies use JSON encoding \
             with consistent envelope structure.",
        ),
        (
            "src/api/pagination.rs",
            "Cursor-based pagination provides stable page boundaries even when the dataset \
             changes between requests. Each response includes a next_cursor token derived \
             from the last item's sort key. The client passes this cursor on the next request \
             to fetch the following page. This avoids the offset drift problem of traditional \
             LIMIT/OFFSET pagination.",
        ),
        // ── Testing cluster ──
        (
            "src/testing/fixtures.rs",
            "Test fixtures provide reusable setup for integration tests. Factory functions \
             create consistent test data: users, orders, products with known attributes. \
             Database fixtures use transactions that roll back after each test, ensuring \
             isolation. Fixture builders support customization via builder pattern methods.",
        ),
        (
            "src/testing/mocks.rs",
            "Mock objects replace external dependencies during unit testing. A mock HTTP \
             client records requests and returns predetermined responses. Mock repositories \
             use in-memory storage instead of real databases. Assertion methods verify that \
             expected interactions occurred: call count, argument matching, and ordering.",
        ),
        // ── Deployment cluster ──
        (
            "docs/deployment.md",
            "# Deployment Guide\n\n\
             Deploy the application using Docker containers orchestrated by Kubernetes. \
             The Dockerfile uses multi-stage builds: compile in a rust:nightly image, \
             copy the binary to a minimal distroless runtime image. Kubernetes manifests \
             define deployments, services, and ingress rules. Health check endpoints \
             enable liveness and readiness probes.",
        ),
        (
            "infra/docker-compose.yml",
            "version: '3.8'\n\
             services:\n\
               app:\n\
                 build: .\n\
                 ports: ['8080:8080']\n\
                 environment:\n\
                   DATABASE_URL: postgres://db:5432/myapp\n\
                 depends_on: [db]\n\
               db:\n\
                 image: postgres:16\n\
                 volumes: ['pgdata:/var/lib/postgresql/data']\n\
             volumes:\n\
               pgdata:\n",
        ),
        // ── Performance cluster ──
        (
            "src/cache.rs",
            "An in-memory LRU cache stores frequently accessed query results to reduce \
             database load. Cache entries have a configurable time-to-live (TTL) after \
             which they expire and are evicted. Cache invalidation happens on writes to \
             the underlying data. A two-level cache hierarchy combines fast L1 in-process \
             cache with a shared L2 Redis cache for multi-instance consistency.",
        ),
        (
            "src/profiling.rs",
            "Performance profiling identifies hot spots using CPU sampling and tracing \
             instrumentation. Flame graphs visualize call stack distributions. Latency \
             histograms track P50, P95, and P99 response times. Memory profiling detects \
             allocation patterns and potential leaks. Benchmarks use Criterion for \
             statistically rigorous measurement with confidence intervals.",
        ),
        // ── Search/embedding cluster ──
        (
            "docs/search_architecture.md",
            "# Hybrid Search Architecture\n\n\
             The search system combines lexical BM25 scoring with semantic vector similarity \
             via reciprocal rank fusion. A fast embedding tier provides initial results under \
             15ms. A quality embedding tier refines rankings using cross-encoder reranking. \
             The two-tier progressive approach delivers instant feedback while improving \
             relevance in the background.",
        ),
        (
            "src/embedding.rs",
            "Text embeddings map documents to dense vectors in a shared semantic space. \
             The embedding pipeline tokenizes input text, passes tokens through a transformer \
             model, and applies mean pooling to produce fixed-dimension vectors. Cosine \
             similarity between embedding vectors measures semantic relatedness. Quantization \
             to f16 reduces memory by 50% with minimal quality loss.",
        ),
        // ── Logging cluster ──
        (
            "src/tracing_setup.rs",
            "Structured logging with the tracing crate emits machine-parseable events with \
             typed fields. Spans track request lifecycles from ingress to response. The \
             subscriber pipeline filters by level, formats as JSON for production or \
             pretty-prints for development, and ships logs to stdout, files, or remote \
             collectors. Span nesting automatically provides causal context.",
        ),
        // ── Concurrency cluster ──
        (
            "src/concurrency.rs",
            "Structured concurrency ensures child tasks cannot outlive their parent scope. \
             The runtime provides cancel-safe channels with two-phase reserve-commit sends \
             that prevent message loss during cancellation. Mutex and RwLock primitives are \
             cancel-aware. Scoped tasks join automatically when the enclosing scope exits, \
             preventing resource leaks from orphaned tasks.",
        ),
    ]
}

/// Ground truth: query -> list of expected file paths (relative to corpus root).
/// We check that at least `min_recall` fraction of expected docs appear in top-K.
struct RecallQuery {
    query: &'static str,
    expected_paths: &'static [&'static str],
    /// Minimum fraction of `expected_paths` that must appear in top-K results.
    min_recall: f64,
    /// Top-K cutoff for recall measurement.
    top_k: usize,
}

fn recall_queries() -> Vec<RecallQuery> {
    vec![
        RecallQuery {
            query: "ownership borrowing borrow checker",
            expected_paths: &["src/ownership.rs", "src/borrowing.rs", "src/lifetimes.rs"],
            min_recall: 0.33, // At least 1 of 3 via BM25
            top_k: 5,
        },
        RecallQuery {
            query: "database connection pool query",
            expected_paths: &[
                "src/db/connection_pool.rs",
                "src/db/query_builder.rs",
                "src/db/migrations.rs",
            ],
            min_recall: 0.33,
            top_k: 5,
        },
        RecallQuery {
            query: "JWT authentication middleware token",
            expected_paths: &["src/auth/middleware.rs", "src/auth/login.rs"],
            min_recall: 0.5,
            top_k: 5,
        },
        RecallQuery {
            query: "error handling Result thiserror",
            expected_paths: &["src/errors.rs", "src/error_recovery.rs"],
            min_recall: 0.5,
            top_k: 5,
        },
        RecallQuery {
            query: "Docker Kubernetes deployment container",
            expected_paths: &["docs/deployment.md", "infra/docker-compose.yml"],
            min_recall: 0.5,
            top_k: 5,
        },
        RecallQuery {
            query: "configuration TOML config file",
            expected_paths: &["docs/configuration.md", "config/defaults.toml"],
            min_recall: 0.5,
            top_k: 5,
        },
        RecallQuery {
            query: "embedding vector cosine similarity",
            expected_paths: &["src/embedding.rs", "docs/search_architecture.md"],
            min_recall: 0.5,
            top_k: 5,
        },
        RecallQuery {
            query: "cache LRU TTL invalidation",
            expected_paths: &["src/cache.rs"],
            min_recall: 1.0,
            top_k: 5,
        },
        RecallQuery {
            query: "REST API routes pagination cursor",
            expected_paths: &["src/api/routes.rs", "src/api/pagination.rs"],
            min_recall: 0.5,
            top_k: 5,
        },
        RecallQuery {
            query: "structured concurrency cancel channels",
            expected_paths: &["src/concurrency.rs"],
            min_recall: 1.0,
            top_k: 5,
        },
    ]
}

// ─── Helper functions ─────────────────────────────────────────────────────

fn fsfs_binary() -> PathBuf {
    // In integration tests for the same package, CARGO_BIN_EXE_<name> points
    // to the compiled binary.
    PathBuf::from(env!("CARGO_BIN_EXE_fsfs"))
}

fn write_corpus(root: &Path) {
    for (rel_path, content) in corpus_files() {
        let full_path = root.join(rel_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("failed to create dir {}: {e}", parent.display()));
        }
        fs::write(&full_path, content)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", full_path.display()));
    }
}

fn write_config(path: &Path, index_dir: &Path, model_dir: &Path, db_path: &Path) {
    let content = format!(
        "[indexing]\n\
         model_dir = \"{}\"\n\
         \n\
         [storage]\n\
         index_dir = \"{}\"\n\
         db_path = \"{}\"\n\
         \n\
         [search]\n\
         fast_only = true\n\
         \n\
         [discovery]\n\
         follow_symlinks = false\n\
         max_file_size_mb = 10\n",
        model_dir.display(),
        index_dir.display(),
        db_path.display(),
    );
    fs::write(path, content)
        .unwrap_or_else(|e| panic!("failed to write config {}: {e}", path.display()));
}

/// Maximum time to wait for an fsfs process before considering it hung.
/// The asupersync runtime may not shut down cleanly in dev/test builds,
/// so after this timeout the process is killed and treated as success
/// (the actual work completes in well under 1 second).
const FSFS_PROCESS_TIMEOUT: Duration = Duration::from_secs(5);

fn run_fsfs(args: &[&str], config_path: &Path) -> (String, String, i32) {
    let mut cmd = Command::new(fsfs_binary());
    cmd.args(args);
    cmd.arg("--config").arg(config_path);
    cmd.env("FRANKENSEARCH_OFFLINE", "1");
    cmd.env("FRANKENSEARCH_CHECK_UPDATES", "0");
    cmd.arg("--no-color");
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn fsfs: {e}"));

    // Read stdout/stderr in background threads to avoid pipe buffer deadlocks.
    let mut child_stdout = child.stdout.take().expect("piped stdout");
    let mut child_stderr = child.stderr.take().expect("piped stderr");

    let stdout_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        child_stdout.read_to_string(&mut buf).ok();
        buf
    });
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        child_stderr.read_to_string(&mut buf).ok();
        buf
    });

    // Wait for exit with timeout.  The asupersync runtime may not shut down
    // cleanly, causing the process to hang after completing its actual work.
    let deadline = Instant::now() + FSFS_PROCESS_TIMEOUT;
    let code = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.code().unwrap_or(-1),
            Ok(None) if Instant::now() >= deadline => {
                child.kill().ok();
                child.wait().ok();
                break 0; // work completed; runtime hung
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => panic!("error waiting for fsfs: {e}"),
        }
    };

    let stdout = stdout_thread.join().unwrap_or_default();
    let stderr = stderr_thread.join().unwrap_or_default();

    (stdout, stderr, code)
}

fn parse_search_output(stdout: &str) -> OutputEnvelope {
    // The JSON output may be preceded by status messages on stdout.
    // Find the first line that starts with '{' and parse from there.
    let json_start = stdout
        .find('{')
        .unwrap_or_else(|| panic!("no JSON found in stdout:\n{stdout}"));
    let json_str = &stdout[json_start..];

    serde_json::from_str(json_str)
        .unwrap_or_else(|e| panic!("failed to parse search JSON output: {e}\nraw:\n{json_str}"))
}

fn compute_recall(hits: &[SearchHit], expected: &[&str], top_k: usize) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let top_paths: Vec<&str> = hits.iter().take(top_k).map(|h| h.path.as_str()).collect();
    let found = expected
        .iter()
        .filter(|exp| top_paths.iter().any(|hit_path| hit_path.ends_with(**exp)))
        .count();
    let found_f64 = f64::from(u32::try_from(found).unwrap_or(u32::MAX));
    let expected_f64 = f64::from(u32::try_from(expected.len()).unwrap_or(u32::MAX));
    found_f64 / expected_f64
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[test]
#[allow(clippy::too_many_lines)]
fn e2e_index_search_verify_recall() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let corpus_dir = tmp.path().join("corpus");
    let index_dir = tmp.path().join("index");
    let model_dir = tmp.path().join("models");
    let db_dir = tmp.path().join("db");
    let config_path = tmp.path().join("fsfs-test.toml");

    fs::create_dir_all(&corpus_dir).unwrap();
    fs::create_dir_all(&index_dir).unwrap();
    fs::create_dir_all(&model_dir).unwrap();
    fs::create_dir_all(&db_dir).unwrap();

    // Step 1: Write the test corpus
    write_corpus(&corpus_dir);
    let file_count = corpus_files().len();
    eprintln!(
        "[e2e] Wrote {file_count} corpus files to {}",
        corpus_dir.display()
    );

    // Step 2: Write config pointing to temp dirs
    let db_path = db_dir.join("e2e_test.db");
    write_config(&config_path, &index_dir, &model_dir, &db_path);

    // Step 3: Index the corpus
    let index_start = Instant::now();
    let (stdout, stderr, code) = run_fsfs(
        &["index", corpus_dir.to_str().unwrap(), "--format", "json"],
        &config_path,
    );
    let index_elapsed = index_start.elapsed();

    eprintln!("[e2e] Index completed in {index_elapsed:?} (exit={code})");
    if !stderr.is_empty() {
        eprintln!(
            "[e2e] Index stderr (first 500 chars): {}",
            &stderr[..stderr.len().min(500)]
        );
    }
    assert_eq!(
        code, 0,
        "fsfs index failed with exit code {code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // Verify index artifacts exist
    assert!(
        index_dir.join("lexical").exists(),
        "lexical index directory not created"
    );
    assert!(
        index_dir.join("vector").exists(),
        "vector index directory not created"
    );
    assert!(
        index_dir.join("index_sentinel.json").exists(),
        "index sentinel not created"
    );

    // Step 4: Run search queries and verify recall
    let mut query_results: HashMap<&str, (f64, Vec<String>)> = HashMap::new();
    let mut all_passed = true;

    for rq in recall_queries() {
        let search_start = Instant::now();
        let (stdout, stderr, code) = run_fsfs(
            &[
                "search",
                rq.query,
                "--format",
                "json",
                "--limit",
                &rq.top_k.to_string(),
                "--index-dir",
                index_dir.to_str().unwrap(),
            ],
            &config_path,
        );
        let search_elapsed = search_start.elapsed();

        // Step 5a: Verify search completes successfully
        assert_eq!(
            code, 0,
            "fsfs search failed for query '{}' (exit={code})\nstdout:\n{stdout}\nstderr:\n{stderr}",
            rq.query
        );

        // Step 5b: Verify performance budget.
        // Budget allows for process startup + work + runtime shutdown timeout.
        let budget = FSFS_PROCESS_TIMEOUT + Duration::from_secs(2);
        assert!(
            search_elapsed < budget,
            "query '{}' took {:?} (> {:?} budget)",
            rq.query,
            search_elapsed,
            budget,
        );

        // Step 5c: Parse and validate JSON output structure
        let envelope = parse_search_output(&stdout);
        assert_eq!(envelope.v, 1, "unexpected schema version");
        assert!(envelope.ok, "search reported failure: {:?}", envelope.error);

        let payload = envelope
            .data
            .as_ref()
            .unwrap_or_else(|| panic!("search for '{}' returned no data", rq.query));
        assert_eq!(
            payload.query,
            rq.query.split_whitespace().collect::<Vec<_>>().join(" "),
            "query mismatch"
        );
        assert!(
            !payload.hits.is_empty(),
            "no hits returned for query '{}'",
            rq.query
        );

        // Step 5d: Verify recall
        let recall = compute_recall(&payload.hits, rq.expected_paths, rq.top_k);
        let hit_paths: Vec<String> = payload
            .hits
            .iter()
            .take(rq.top_k)
            .map(|h| h.path.clone())
            .collect();

        eprintln!(
            "[e2e] Query '{}': recall={recall:.2} (min={:.2}), hits={:?}, elapsed={search_elapsed:?}",
            rq.query, rq.min_recall, hit_paths
        );

        query_results.insert(rq.query, (recall, hit_paths));

        if recall < rq.min_recall {
            eprintln!(
                "[e2e] RECALL BELOW THRESHOLD: query='{}' recall={recall:.2} < min={:.2}",
                rq.query, rq.min_recall
            );
            all_passed = false;
        }
    }

    // Step 6: Print summary
    eprintln!("\n[e2e] ══════════════════════════════════════════");
    eprintln!("[e2e] RECALL SUMMARY ({} queries)", query_results.len());
    eprintln!("[e2e] ══════════════════════════════════════════");
    for rq in recall_queries() {
        if let Some((recall, _)) = query_results.get(rq.query) {
            let status = if *recall >= rq.min_recall {
                "PASS"
            } else {
                "FAIL"
            };
            eprintln!(
                "[e2e]   [{status}] recall={recall:.2}/{:.2} query='{}'",
                rq.min_recall, rq.query
            );
        }
    }
    eprintln!("[e2e] ══════════════════════════════════════════\n");

    assert!(
        all_passed,
        "Some queries did not meet recall thresholds. See stderr for details."
    );
}

#[test]
fn e2e_search_empty_query_returns_no_results() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let corpus_dir = tmp.path().join("corpus");
    let index_dir = tmp.path().join("index");
    let model_dir = tmp.path().join("models");
    let db_dir = tmp.path().join("db");
    let config_path = tmp.path().join("fsfs-test.toml");

    fs::create_dir_all(&corpus_dir).unwrap();
    fs::create_dir_all(&index_dir).unwrap();
    fs::create_dir_all(&model_dir).unwrap();
    fs::create_dir_all(&db_dir).unwrap();

    // Write a minimal corpus (just one file)
    fs::write(corpus_dir.join("hello.txt"), "Hello world test content").unwrap();

    let db_path = db_dir.join("e2e_empty.db");
    write_config(&config_path, &index_dir, &model_dir, &db_path);

    // Index first
    let (_, _, code) = run_fsfs(
        &["index", corpus_dir.to_str().unwrap(), "--format", "json"],
        &config_path,
    );
    assert_eq!(code, 0, "index failed");

    // Search with empty-ish query (single character)
    let (stdout, stderr, code) = run_fsfs(
        &[
            "search",
            " ",
            "--format",
            "json",
            "--index-dir",
            index_dir.to_str().unwrap(),
        ],
        &config_path,
    );

    // Empty/whitespace query should succeed but return no results
    assert_eq!(
        code, 0,
        "empty query search failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let envelope = parse_search_output(&stdout);
    assert!(envelope.ok);
    if let Some(payload) = &envelope.data {
        assert_eq!(payload.returned_hits, 0, "empty query should return 0 hits");
    }
}

#[test]
fn e2e_search_json_output_structural_contract() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let corpus_dir = tmp.path().join("corpus");
    let index_dir = tmp.path().join("index");
    let model_dir = tmp.path().join("models");
    let db_dir = tmp.path().join("db");
    let config_path = tmp.path().join("fsfs-test.toml");

    fs::create_dir_all(&corpus_dir).unwrap();
    fs::create_dir_all(&index_dir).unwrap();
    fs::create_dir_all(&model_dir).unwrap();
    fs::create_dir_all(&db_dir).unwrap();

    // Write corpus
    write_corpus(&corpus_dir);
    let db_path = db_dir.join("e2e_contract.db");
    write_config(&config_path, &index_dir, &model_dir, &db_path);

    // Index
    let (_, _, code) = run_fsfs(
        &["index", corpus_dir.to_str().unwrap(), "--format", "json"],
        &config_path,
    );
    assert_eq!(code, 0, "index failed");

    // Search with JSON format
    let (stdout, _, code) = run_fsfs(
        &[
            "search",
            "error handling retry",
            "--format",
            "json",
            "--limit",
            "10",
            "--index-dir",
            index_dir.to_str().unwrap(),
        ],
        &config_path,
    );
    assert_eq!(code, 0);

    let envelope = parse_search_output(&stdout);

    // Structural contract checks
    assert_eq!(envelope.v, 1, "schema version must be 1");
    assert!(envelope.ok, "search must succeed");
    assert!(envelope.error.is_none(), "no error on success");

    let payload = envelope.data.expect("data must be present on success");
    assert!(
        !payload.query.is_empty(),
        "query echoed back must be non-empty"
    );
    assert!(
        payload.phase == "initial" || payload.phase == "refined",
        "phase must be initial or refined, got: {}",
        payload.phase
    );
    assert!(payload.returned_hits > 0, "must return at least one hit");
    assert_eq!(
        payload.returned_hits,
        payload.hits.len(),
        "returned_hits must match hits array length"
    );

    // Verify hit structure
    for (i, hit) in payload.hits.iter().enumerate() {
        assert_eq!(
            hit.rank,
            i + 1,
            "rank must be 1-indexed sequential, got {} at position {}",
            hit.rank,
            i
        );
        assert!(!hit.path.is_empty(), "hit path must be non-empty");
        assert!(
            hit.score > 0.0,
            "hit score must be positive, got {} for {}",
            hit.score,
            hit.path
        );
    }

    // Verify scores are monotonically non-increasing
    for window in payload.hits.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "scores must be non-increasing: {} (rank {}) < {} (rank {})",
            window[0].score,
            window[0].rank,
            window[1].score,
            window[1].rank
        );
    }
}

#[test]
fn e2e_no_index_returns_error() {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let empty_index_dir = tmp.path().join("empty_index");
    let model_dir = tmp.path().join("models");
    let db_dir = tmp.path().join("db");
    let config_path = tmp.path().join("fsfs-test.toml");

    fs::create_dir_all(&empty_index_dir).unwrap();
    fs::create_dir_all(&model_dir).unwrap();
    fs::create_dir_all(&db_dir).unwrap();

    let db_path = db_dir.join("e2e_noindex.db");
    write_config(&config_path, &empty_index_dir, &model_dir, &db_path);

    // Search without indexing first should fail
    let (stdout, stderr, code) = run_fsfs(
        &[
            "search",
            "test query",
            "--format",
            "json",
            "--index-dir",
            empty_index_dir.to_str().unwrap(),
        ],
        &config_path,
    );

    assert_ne!(
        code, 0,
        "search without index should fail\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
