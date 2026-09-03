//! Release-profile latency and index-cost receipt for the `fsfs` binary
//! itself (bd-8j5dc, bridge Gap #3, product half).
//!
//! Opt-in: `FRANKENSEARCH_PERF_RECEIPT=1` runs it; anything else skips with a
//! message. Run it as
//!
//! ```text
//! FRANKENSEARCH_PERF_RECEIPT=1 cargo test --release -p frankensearch-fsfs \
//!     --test fsfs_latency_receipt -- --nocapture
//! ```
//!
//! (the gate's opt-in `perf` stage does exactly that, after the library lane).
//! The JSON receipt goes to `FRANKENSEARCH_PERF_RECEIPT_OUT` when set, else
//! stdout.
//!
//! What is measured, with the registered models (`FRANKENSEARCH_MODEL_DIR`,
//! else `~/.local/share/frankensearch/models`) and an isolated HOME:
//! 1. `fsfs index` over a deterministic 1,000-file prose corpus
//!    (`FRANKENSEARCH_PERF_RECEIPT_DOCS` overrides): wall time end to end.
//! 2. Cold start: three `fsfs search --no-daemon` processes on distinct
//!    queries, wall time each (min / median).
//! 3. Daemon-served queries: one `fsfs serve --daemon-socket ...` process,
//!    5 warm-up + 50 timed one-request-per-connection JSON-line searches,
//!    client wall time from connect to full response (p50/p95/p99); the
//!    final phase must be REFINED; daemon cache hits are counted, never
//!    hidden.
//! 4. The same through the request's `rerank` flag (20 timed queries) so the
//!    cross-encoder's cost has a receipt too; skipped with the reason when the
//!    ms-marco model is not installed.
//! 5. Watch-mode freshness: `fsfs index --watch` on the same corpus, 20 files
//!    written one at a time, the watcher's own per-batch `oldest_event_age_ms`
//!    (event observed to batch applied) and the client-side write-to-visible
//!    wall, a graceful stop, and a fresh-process search proving the new files
//!    are searchable once the watcher has exited. Host-pressure sampling is
//!    pinned to one sample per hour for this section (a saturated host would
//!    otherwise pause the watcher by design); the receipt records the pin.
//!
//! `FRANKENSEARCH_PERF_RECEIPT_MAX_DAEMON_P95_MS=<ms>` turns the lane into a
//! threshold check (the planted-regression control).
#![cfg(unix)]

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

const RECEIPT_SCHEMA: &str = "frankensearch-fsfs-latency-receipt-v1";
const DEFAULT_DOCS: usize = 1_000;
const COLD_RUNS: usize = 3;
const WARMUP_QUERIES: usize = 5;
const TIMED_QUERIES: usize = 50;
const RERANK_QUERIES: usize = 20;
const TOP_K: usize = 10;
const DAEMON_IDLE_TIMEOUT_MS: u64 = 120_000;

const TOPICS: &[&[&str]] = &[
    &[
        "distributed",
        "consensus",
        "raft",
        "paxos",
        "leader",
        "election",
        "quorum",
        "replica",
        "log",
        "commit",
        "follower",
        "heartbeat",
        "partition",
        "failover",
        "cluster",
        "node",
    ],
    &[
        "database",
        "index",
        "btree",
        "transaction",
        "isolation",
        "snapshot",
        "wal",
        "checkpoint",
        "vacuum",
        "query",
        "planner",
        "join",
        "scan",
        "row",
        "page",
        "buffer",
    ],
    &[
        "ownership",
        "borrow",
        "lifetime",
        "trait",
        "generic",
        "iterator",
        "closure",
        "async",
        "future",
        "runtime",
        "cancel",
        "mutex",
        "channel",
        "task",
        "region",
        "scope",
    ],
    &[
        "http",
        "tls",
        "socket",
        "stream",
        "multiplex",
        "header",
        "frame",
        "latency",
        "retry",
        "backoff",
        "timeout",
        "proxy",
        "gateway",
        "route",
        "packet",
        "congestion",
    ],
    &[
        "model",
        "embedding",
        "vector",
        "cosine",
        "training",
        "dataset",
        "tokenizer",
        "attention",
        "encoder",
        "quantization",
        "inference",
        "batch",
        "gradient",
        "epoch",
        "layer",
        "logits",
    ],
    &[
        "filesystem",
        "inode",
        "fsync",
        "mmap",
        "durability",
        "sidecar",
        "repair",
        "tombstone",
        "compaction",
        "generation",
        "manifest",
        "lease",
        "watcher",
        "debounce",
        "crawl",
        "publish",
    ],
];

const CONNECTIVES: &[&str] = &[
    "the", "a", "with", "under", "after", "before", "while", "because", "keeps", "handles",
    "requires", "avoids", "ensures", "measures", "records", "moves", "every", "each", "into",
    "from", "between", "without", "then", "and",
];

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn pick<'a>(&mut self, words: &[&'a str]) -> &'a str {
        let index = usize::try_from(self.next()).unwrap_or(0) % words.len();
        words[index]
    }
}

fn write_corpus(root: &Path, docs: usize) -> usize {
    let mut rng = Lcg(0x5EED_F00D_CAFE_BABE);
    let mut total_chars = 0;
    for index in 0..docs {
        let topic = TOPICS[index % TOPICS.len()];
        let words = 60 + usize::try_from(rng.next()).unwrap_or(0) % 60;
        let mut text = String::new();
        for position in 0..words {
            if position > 0 {
                text.push(' ');
            }
            let word = if rng.next() % 10 < 7 {
                rng.pick(topic)
            } else {
                rng.pick(CONNECTIVES)
            };
            text.push_str(word);
            if position % 12 == 11 {
                text.push('.');
            }
        }
        text.push('\n');
        total_chars += text.len();
        std::fs::write(root.join(format!("doc-{index:04}.md")), text).expect("write corpus file");
    }
    total_chars
}

fn synthetic_queries(count: usize) -> Vec<String> {
    let mut rng = Lcg(0xBAAD_CAFE_1234_5678);
    (0..count)
        .map(|index| {
            let topic = TOPICS[index % TOPICS.len()];
            let words = 3 + usize::try_from(rng.next()).unwrap_or(0) % 4;
            (0..words)
                .map(|_| rng.pick(topic))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Percentile summary of one timing series, in milliseconds.
fn distribution(samples: &[f64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let percentile = |q: f64| -> f64 {
        if n == 0 {
            return 0.0;
        }
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let index = (((n - 1) as f64) * q).round() as usize;
        sorted[index.min(n - 1)]
    };
    #[allow(clippy::cast_precision_loss)]
    let mean = if n == 0 {
        0.0
    } else {
        sorted.iter().sum::<f64>() / n as f64
    };
    serde_json::json!({
        "n": n,
        "min_ms": sorted.first().copied().unwrap_or(0.0),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": sorted.last().copied().unwrap_or(0.0),
        "mean_ms": mean,
    })
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn fsfs_binary() -> PathBuf {
    std::env::var_os("FSFS_E2E_BINARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_fsfs")))
}

fn registered_model_root() -> PathBuf {
    std::env::var_os("FRANKENSEARCH_MODEL_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join(".local/share/frankensearch/models"))
        })
        .expect("a model root (FRANKENSEARCH_MODEL_DIR or HOME)")
}

/// A short socket path: `AF_UNIX` paths are capped near 107 bytes, and the
/// runtime directory is the shortest writable place the daemon itself uses.
fn short_socket_path() -> PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|dir| dir.is_dir())
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join(format!("fsfs-receipt-{}.sock", std::process::id()))
}

struct Harness {
    binary: PathBuf,
    home: PathBuf,
    model_root: PathBuf,
}

impl Harness {
    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(&self.binary);
        command
            .args(args)
            .env("HOME", &self.home)
            .env("XDG_CONFIG_HOME", self.home.join("xdg-config"))
            .env("XDG_CACHE_HOME", self.home.join("xdg-cache"))
            .env("XDG_DATA_HOME", self.home.join("xdg-data"))
            .env("FRANKENSEARCH_MODEL_DIR", &self.model_root)
            .env("FRANKENSEARCH_OFFLINE", "1")
            .env("FRANKENSEARCH_ALLOW_DOWNLOAD", "0")
            .env("FRANKENSEARCH_CHECK_UPDATES", "0")
            .env("NO_COLOR", "1")
            .env_remove("FRANKENSEARCH_INDEX_DIR")
            .env_remove("FSFS_INDEX_DIR")
            .env_remove("RUST_LOG");
        command
    }

    fn run_text(&self, label: &str, args: &[&str]) -> (String, Duration) {
        let started = Instant::now();
        let output = self
            .command(args)
            .output()
            .unwrap_or_else(|error| panic!("{label}: spawn fsfs: {error}"));
        let elapsed = started.elapsed();
        assert!(
            output.status.success(),
            "{label} failed (exit {:?})\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        (
            String::from_utf8_lossy(&output.stdout).into_owned(),
            elapsed,
        )
    }

    fn run_json(&self, label: &str, args: &[&str]) -> (Value, Duration) {
        let started = Instant::now();
        let output = self
            .command(args)
            .output()
            .unwrap_or_else(|error| panic!("{label}: spawn fsfs: {error}"));
        let elapsed = started.elapsed();
        assert!(
            output.status.success(),
            "{label} failed (exit {:?})\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let json = serde_json::from_slice::<Value>(&output.stdout).unwrap_or_else(|error| {
            panic!(
                "{label}: invalid JSON: {error}\nstdout:\n{}",
                String::from_utf8_lossy(&output.stdout)
            )
        });
        (json, elapsed)
    }
}

struct DaemonGuard {
    child: Child,
    socket: PathBuf,
}

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// The daemon's own readiness handshake (`:ready`), so the timed queries
/// never include its startup or the resource bind.
fn wait_for_socket(path: &Path, budget: Duration) {
    let started = Instant::now();
    while started.elapsed() < budget {
        if let Ok(mut stream) = UnixStream::connect(path) {
            let ready = stream
                .write_all(b":ready\n")
                .and_then(|()| stream.flush())
                .and_then(|()| stream.shutdown(std::net::Shutdown::Write))
                .and_then(|()| {
                    let mut raw = String::new();
                    stream.read_to_string(&mut raw).map(|_| raw)
                })
                .map(|raw| !raw.trim().is_empty())
                .unwrap_or(false);
            if ready {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!(
        "daemon socket {} not ready within {budget:?}",
        path.display()
    );
}

/// One request on one connection, the way the CLI client talks to the daemon.
fn daemon_query(socket: &Path, query: &str, rerank: bool) -> (Value, Duration) {
    let request = serde_json::json!({
        "query": query,
        "limit": TOP_K,
        "mode": "full",
        "filter": null,
        "rerank": rerank,
    });
    let started = Instant::now();
    let mut stream = UnixStream::connect(socket).expect("connect to the query daemon");
    stream
        .write_all(request.to_string().as_bytes())
        .expect("write request");
    stream.write_all(b"\n").expect("write newline");
    stream.flush().expect("flush request");
    // The daemon reads the request to end-of-stream, exactly like the CLI
    // client: half-close the write side or its 5 s read timeout expires.
    stream
        .shutdown(std::net::Shutdown::Write)
        .expect("half-close the request");
    let mut raw = String::new();
    stream.read_to_string(&mut raw).expect("read response");
    let elapsed = started.elapsed();
    let response = serde_json::from_str::<Value>(raw.trim())
        .unwrap_or_else(|error| panic!("daemon response is not JSON: {error}\n{raw}"));
    (response, elapsed)
}

fn final_phase(response: &Value) -> Option<&str> {
    response
        .get("payloads")
        .and_then(Value::as_array)
        .and_then(|payloads| payloads.last())
        .and_then(|payload| payload.get("phase"))
        .and_then(Value::as_str)
}

/// `(files, reported_ms, index_size_bytes)` from the `fsfs index` summary
/// line: `Indexed N file(s) (discovered N, skipped 0) into <dir> in M ms
/// (index size B bytes)`.
fn parse_index_summary(stdout: &str) -> Option<(u64, u64, u64)> {
    let line = stdout.lines().find(|line| line.starts_with("Indexed "))?;
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let files = tokens.get(1)?.parse().ok()?;
    let after = |marker: &str| -> Option<u64> {
        tokens
            .iter()
            .position(|token| *token == marker)
            .and_then(|at| tokens.get(at + 1))
            .and_then(|token| token.parse().ok())
    };
    Some((files, after("in")?, after("size")?))
}

/// Lines of the watcher's stderr that carry the per-batch freshness record.
fn watch_batch_lines(log: &Path) -> Vec<String> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter(|line| line.contains("fsfs watch batch applied"))
        .map(str::to_owned)
        .collect()
}

/// `key=<digits>` out of a tracing line.
fn tracing_field_u64(line: &str, key: &str) -> Option<u64> {
    let start = line.find(key)? + key.len();
    let digits: String = line[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn wait_for_log_line(log: &Path, needle: &str, budget: Duration) {
    let started = Instant::now();
    while started.elapsed() < budget {
        if std::fs::read_to_string(log)
            .unwrap_or_default()
            .contains(needle)
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!(
        "{needle:?} not seen in {} within {budget:?}:\n{}",
        log.display(),
        std::fs::read_to_string(log).unwrap_or_default()
    );
}

fn host_fingerprint() -> Value {
    let hostname = std::fs::read_to_string("/etc/hostname")
        .map(|value| value.trim().to_owned())
        .ok();
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().to_owned())
        });
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .ok();
    let load = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|value| value.trim().to_owned());
    serde_json::json!({
        "hostname": hostname,
        "cpu_model": cpu_model,
        "logical_cores": cores,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "loadavg_at_start": load,
    })
}

fn git_revision() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn file_sha256(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).ok()?;
    let digest = Sha256::digest(&bytes);
    Some(
        digest
            .iter()
            .fold(String::with_capacity(64), |mut hex, byte| {
                use std::fmt::Write as _;
                let _ = write!(hex, "{byte:02x}");
                hex
            }),
    )
}

#[test]
fn fsfs_latency_receipt() {
    if std::env::var("FRANKENSEARCH_PERF_RECEIPT").as_deref() != Ok("1") {
        eprintln!(
            "SKIPPING fsfs latency receipt: set FRANKENSEARCH_PERF_RECEIPT=1 (and run under --release) to measure"
        );
        return;
    }
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let host = host_fingerprint();

    let temp = tempfile::tempdir().expect("tempdir");
    let corpus = temp.path().join("corpus");
    let index_dir = temp.path().join("index");
    let home = temp.path().join("home");
    for dir in [&corpus, &home] {
        std::fs::create_dir_all(dir).expect("create dir");
    }
    let docs = env_usize("FRANKENSEARCH_PERF_RECEIPT_DOCS", DEFAULT_DOCS);
    let total_chars = write_corpus(&corpus, docs);
    let harness = Harness {
        binary: fsfs_binary(),
        home,
        model_root: registered_model_root(),
    };
    let corpus_arg = corpus.display().to_string();
    let index_arg = index_dir.display().to_string();

    // 1. Index cost, end to end through the binary. `fsfs index` reports a
    //    human summary line ("Indexed N file(s) ... in M ms (index size B
    //    bytes)") rather than an envelope; the wall clock here is the
    //    number that matters, the reported values are cross-checks.
    let (index_text, index_wall) = harness.run_text(
        "fsfs index",
        &[
            "index",
            &corpus_arg,
            "--index-dir",
            &index_arg,
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    let (indexed_files, index_reported_ms, index_size_bytes) = parse_index_summary(&index_text)
        .unwrap_or_else(|| panic!("fsfs index summary line missing:\n{index_text}"));
    assert_eq!(
        indexed_files,
        u64::try_from(docs).unwrap_or(u64::MAX),
        "every corpus file must be indexed:\n{index_text}"
    );
    let status_json = harness
        .run_json(
            "fsfs status",
            &["status", "--index-dir", &index_arg, "--format", "json"],
        )
        .0;
    let quality_tier_present = index_dir.join("vector").join("quality.fsvi").is_file();
    assert!(
        quality_tier_present,
        "the registered models must produce a quality tier: {status_json}"
    );

    // 2. Cold start: whole-process wall for a no-daemon search.
    let queries = synthetic_queries(COLD_RUNS + WARMUP_QUERIES + TIMED_QUERIES + RERANK_QUERIES);
    let mut cold = Vec::with_capacity(COLD_RUNS);
    let mut cold_phases = Vec::with_capacity(COLD_RUNS);
    for query in &queries[..COLD_RUNS] {
        let (json, wall) = harness.run_json(
            "cold fsfs search",
            &[
                "search",
                query,
                "--index-dir",
                &index_arg,
                "--no-daemon",
                "--no-watch-mode",
                "--format",
                "json",
            ],
        );
        cold.push(ms(wall));
        cold_phases.push(
            json.pointer("/data/phase")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_owned(),
        );
    }

    // 3. Daemon-served queries.
    let socket = short_socket_path();
    let child = harness
        .command(&[
            "serve",
            "--daemon-socket",
            &socket.display().to_string(),
            "--idle-timeout-ms",
            &DAEMON_IDLE_TIMEOUT_MS.to_string(),
            "--index-dir",
            &index_arg,
            "--format",
            "jsonl",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the query daemon");
    let daemon = DaemonGuard {
        child,
        socket: socket.clone(),
    };
    wait_for_socket(&daemon.socket, Duration::from_secs(60));

    // The daemon's overhead floor: a `:ready` round trip does no search, so
    // whatever it costs is accept latency plus the handshake itself.
    let mut ready_ms = Vec::with_capacity(10);
    for _ in 0..10 {
        let started = Instant::now();
        let mut stream = UnixStream::connect(&daemon.socket).expect("connect for :ready");
        stream.write_all(b":ready\n").expect("write :ready");
        stream.flush().expect("flush :ready");
        stream
            .shutdown(std::net::Shutdown::Write)
            .expect("half-close :ready");
        let mut raw = String::new();
        stream
            .read_to_string(&mut raw)
            .expect("read :ready response");
        assert!(!raw.trim().is_empty(), "daemon :ready must answer");
        ready_ms.push(ms(started.elapsed()));
    }

    let mut daemon_ms = Vec::with_capacity(TIMED_QUERIES);
    let mut daemon_cache_hits = 0_usize;
    let mut daemon_phases = std::collections::BTreeMap::<String, usize>::new();
    for (position, query) in queries[COLD_RUNS..COLD_RUNS + WARMUP_QUERIES + TIMED_QUERIES]
        .iter()
        .enumerate()
    {
        let (response, wall) = daemon_query(&daemon.socket, query, false);
        assert_eq!(
            response.get("ok").and_then(Value::as_bool),
            Some(true),
            "daemon refused a query: {response}"
        );
        if position < WARMUP_QUERIES {
            continue;
        }
        daemon_ms.push(ms(wall));
        if response.get("cached").and_then(Value::as_bool) == Some(true) {
            daemon_cache_hits += 1;
        }
        *daemon_phases
            .entry(final_phase(&response).unwrap_or("?").to_owned())
            .or_insert(0) += 1;
    }
    assert!(
        daemon_phases.get("refined").copied().unwrap_or(0) > 0,
        "the daemon must serve REFINED answers: {daemon_phases:?}"
    );

    // 4. The same with the cross-encoder requested.
    let mut rerank_ms = Vec::with_capacity(RERANK_QUERIES);
    let mut rerank_statuses = std::collections::BTreeMap::<String, usize>::new();
    for query in &queries[COLD_RUNS + WARMUP_QUERIES + TIMED_QUERIES..] {
        let (response, wall) = daemon_query(&daemon.socket, query, true);
        assert_eq!(
            response.get("ok").and_then(Value::as_bool),
            Some(true),
            "daemon refused a rerank query: {response}"
        );
        rerank_ms.push(ms(wall));
        let status = response
            .get("payloads")
            .and_then(Value::as_array)
            .and_then(|payloads| payloads.last())
            .and_then(|payload| payload.pointer("/rerank/status"))
            .and_then(Value::as_str)
            .unwrap_or("no_rerank_block")
            .to_owned();
        let reason = response
            .get("payloads")
            .and_then(Value::as_array)
            .and_then(|payloads| payloads.last())
            .and_then(|payload| payload.pointer("/rerank/reason_code"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        *rerank_statuses
            .entry(format!("{status}:{reason}"))
            .or_insert(0) += 1;
    }
    drop(daemon);

    // 5. Watch-mode freshness. The daemon is gone (it held a reader lock on
    //    the vector generations; the watcher needs the writer). Start the
    //    watcher, write files one at a time, and read the per-batch
    //    `fsfs watch batch applied` lines it logs: `oldest_event_age_ms` is
    //    the age of the oldest event in the batch when the sink finished
    //    (debounce + queueing + ingest). The client-side wall from the write
    //    to the line becoming visible is kept as an upper bound (it includes
    //    this test's 10 ms poll). Then stop the watcher and prove a new file
    //    is searchable from a fresh process.
    const WATCH_FILES: usize = 20;
    let watch_log_path = temp.path().join("watch.stderr");
    let watch_log = std::fs::File::create(&watch_log_path).expect("create watcher log");
    let watch_started = Instant::now();
    let mut watcher = harness
        .command(&[
            "index",
            &corpus_arg,
            "--index-dir",
            &index_arg,
            "--watch",
            "--format",
            "json",
        ])
        .env("RUST_LOG", "info")
        // Host pressure is not the subject: on a saturated host the pressure
        // controller moves the watcher to Degraded/Emergency, where watching
        // is switched off by design (observed mid-measurement on a loaded
        // 64-core box). One sample per hour keeps the measurement about the
        // watcher; the receipt says so.
        .env("FRANKENSEARCH_PRESSURE_SAMPLE_INTERVAL_MS", "3600000")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(watch_log)
        .spawn()
        .expect("spawn the watcher");
    wait_for_log_line(&watch_log_path, "watcher started", Duration::from_secs(120));
    let watcher_startup_ms = ms(watch_started.elapsed());

    let mut write_to_visible_ms = Vec::with_capacity(WATCH_FILES);
    let mut event_to_applied_ms = Vec::new();
    let mut batch_apply_ms = Vec::new();
    let mut seen_batches = 0_usize;
    for index in 0..WATCH_FILES {
        let path = corpus.join(format!("watch-{index:04}.md"));
        std::fs::write(
            &path,
            format!(
                "watch freshness probe {index}: zorblax calibrates temporal flux with quorum heartbeat.\n"
            ),
        )
        .expect("write watch file");
        let written = Instant::now();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let lines = watch_batch_lines(&watch_log_path);
            if lines.len() > seen_batches {
                for line in &lines[seen_batches..] {
                    if let Some(age) = tracing_field_u64(line, "oldest_event_age_ms=") {
                        #[allow(clippy::cast_precision_loss)]
                        event_to_applied_ms.push(age as f64);
                    }
                    if let Some(apply) = tracing_field_u64(line, "apply_ms=") {
                        #[allow(clippy::cast_precision_loss)]
                        batch_apply_ms.push(apply as f64);
                    }
                }
                seen_batches = lines.len();
                write_to_visible_ms.push(ms(written.elapsed()));
                break;
            }
            assert!(
                Instant::now() < deadline,
                "watcher did not apply a batch for {} within 30 s; log:\n{}",
                path.display(),
                std::fs::read_to_string(&watch_log_path).unwrap_or_default()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let _ = Command::new("kill")
        .args(["-TERM", &watcher.id().to_string()])
        .status();
    let stop_deadline = Instant::now() + Duration::from_secs(60);
    let watcher_exit = loop {
        if let Some(status) = watcher.try_wait().expect("poll watcher") {
            break status.code();
        }
        if Instant::now() >= stop_deadline {
            let _ = watcher.kill();
            let _ = watcher.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let (after_watch, _) = harness.run_json(
        "search after the watcher exits",
        &[
            "search",
            "zorblax calibrates temporal flux",
            "--index-dir",
            &index_arg,
            "--no-daemon",
            "--no-watch-mode",
            "--format",
            "json",
        ],
    );
    let searchable_after_exit = after_watch
        .pointer("/data/hits")
        .and_then(Value::as_array)
        .is_some_and(|hits| {
            hits.iter().any(|hit| {
                hit.get("path")
                    .and_then(Value::as_str)
                    .is_some_and(|path| path.starts_with("watch-"))
            })
        });
    assert!(
        searchable_after_exit,
        "a file ingested by the watcher must be searchable once it exits: {after_watch}"
    );
    let watch_mode = serde_json::json!({
        "files_written": WATCH_FILES,
        "host_pressure_sampling": "pinned to one sample per hour for this measurement (FRANKENSEARCH_PRESSURE_SAMPLE_INTERVAL_MS=3600000); in production a saturated host moves the watcher to Degraded/Emergency, where watching pauses",
        "batches_applied": seen_batches,
        "watcher_startup_ms": watcher_startup_ms,
        "event_to_applied_ms": distribution(&event_to_applied_ms),
        "batch_apply_ms": distribution(&batch_apply_ms),
        "write_to_line_visible_ms": distribution(&write_to_visible_ms),
        "watcher_exit_code": watcher_exit,
        "searchable_after_exit": searchable_after_exit,
    });

    let receipt = serde_json::json!({
        "schema": RECEIPT_SCHEMA,
        "generated_at_unix": started_at,
        "git_revision": git_revision(),
        "cargo_profile": profile,
        "profile_note": "cargo test --release binary (opt-level 3); the shipped fsfs release adds fat LTO and codegen-units=1",
        "fsfs_binary": harness.binary.display().to_string(),
        "fsfs_binary_sha256": file_sha256(&harness.binary),
        "host": host,
        "model_root": harness.model_root.display().to_string(),
        "corpus": { "files": docs, "total_chars": total_chars, "generator": "deterministic LCG prose over 6 topic vocabularies, one file per document" },
        "index": {
            "wall_ms": ms(index_wall),
            "indexed_files": indexed_files,
            "reported_ms": index_reported_ms,
            "index_size_bytes": index_size_bytes,
            "quality_tier": quality_tier_present,
            "status": status_json.get("data").cloned().unwrap_or(Value::Null),
        },
        "cold_start_search_ms": { "runs": COLD_RUNS, "summary": distribution(&cold), "samples": cold, "phases": cold_phases },
        "daemon_ready_roundtrip_ms": distribution(&ready_ms),
        "daemon_query_ms": distribution(&daemon_ms),
        "daemon_queries": { "warmup": WARMUP_QUERIES, "timed": TIMED_QUERIES, "top_k": TOP_K, "cache_hits": daemon_cache_hits, "final_phases": daemon_phases },
        "daemon_rerank_query_ms": distribution(&rerank_ms),
        "daemon_rerank_queries": { "timed": RERANK_QUERIES, "statuses": rerank_statuses },
        "watch_mode": watch_mode,
    });
    let encoded = serde_json::to_string_pretty(&receipt).expect("encode receipt");
    match std::env::var_os("FRANKENSEARCH_PERF_RECEIPT_OUT") {
        Some(path) => {
            let path = PathBuf::from(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create receipt directory");
            }
            std::fs::write(&path, format!("{encoded}\n")).expect("write receipt");
            eprintln!("fsfs latency receipt written to {}", path.display());
        }
        None => println!("{encoded}"),
    }

    if let Ok(bound) = std::env::var("FRANKENSEARCH_PERF_RECEIPT_MAX_DAEMON_P95_MS") {
        let bound: f64 = bound.trim().parse().expect("numeric p95 bound");
        let p95 = receipt["daemon_query_ms"]["p95_ms"]
            .as_f64()
            .expect("daemon p95");
        assert!(
            p95 <= bound,
            "daemon-served query p95 {p95:.1} ms exceeds the bound {bound:.1} ms"
        );
    }
}
