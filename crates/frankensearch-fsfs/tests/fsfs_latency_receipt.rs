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
//! 5. Watch-mode freshness: `fsfs watch` on the same corpus, 20 files written
//!    one at a time. Each file must become searchable through the already-live
//!    daemon after a complete generation is selected, then through a separate
//!    direct process. The watcher must exit cleanly and leave its last change
//!    searchable. The receipt records the polling interval and pressure policy.
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

const RECEIPT_SCHEMA: &str = "frankensearch-fsfs-latency-receipt-v2";
const DEFAULT_DOCS: usize = 1_000;
const COLD_RUNS: usize = 3;
const WARMUP_QUERIES: usize = 5;
const TIMED_QUERIES: usize = 50;
const RERANK_QUERIES: usize = 20;
const TOP_K: usize = 10;
const DAEMON_IDLE_TIMEOUT_MS: u64 = 120_000;
const WATCH_FILES: usize = 20;
const WATCH_POLL: Duration = Duration::from_millis(25);

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
            .env("FSFS_COMPLETE_GENERATIONS", "1")
            .env("NO_COLOR", "1")
            .env_remove("FRANKENSEARCH_INDEX_DIR")
            .env_remove("FSFS_INDEX_DIR")
            .env_remove("RUST_LOG");
        command
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
    daemon_request(socket, &request)
}

fn daemon_request(socket: &Path, request: &Value) -> (Value, Duration) {
    let started = Instant::now();
    let mut stream = UnixStream::connect(socket).expect("connect to the query daemon");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .expect("bound daemon read time");
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

fn has_hit(payload: &Value, path: &str) -> bool {
    payload["hits"]
        .as_array()
        .is_some_and(|hits| hits.iter().any(|hit| hit["path"].as_str() == Some(path)))
}

struct WatchGuard(Child);

impl Drop for WatchGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn measure_watch(harness: &Harness, corpus: &Path, index: &Path, socket: &Path) -> Value {
    let corpus_arg = corpus.to_str().expect("UTF-8 corpus path");
    let index_arg = index.to_str().expect("UTF-8 index path");
    let log_path = harness.home.join("watch.stderr");
    let log = std::fs::File::create(&log_path).expect("create watcher log");
    let mut watcher = WatchGuard(
        harness
            .command(&[
                "watch",
                corpus_arg,
                "--index-dir",
                index_arg,
                "--format",
                "jsonl",
            ])
            .env("RUST_LOG", "info")
            // A pressure pause is outside this steady-load freshness sample.
            .env("FRANKENSEARCH_PRESSURE_SAMPLE_INTERVAL_MS", "3600000")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .expect("spawn the watcher"),
    );
    let selection = index.join("FSFS-CURRENT");
    let mut daemon_visible_ms = Vec::with_capacity(WATCH_FILES);
    let mut direct_confirmed_ms = Vec::with_capacity(WATCH_FILES);
    for edit in 0..=WATCH_FILES {
        let name = format!("watch-{edit:04}.md");
        let token = format!("zorblax{edit:04}");
        let filter = format!("path:{name}");
        let request = serde_json::json!({
            "query": token, "limit": 1, "mode": "lexical", "filter": filter,
        });
        let search = || {
            let response = daemon_request(socket, &request).0;
            assert_eq!(response["ok"].as_bool(), Some(true), "{response}");
            has_hit(&response["payloads"][0], &name)
        };
        assert!(!search(), "daemon returned {name} before it was written");
        let previous = std::fs::read(&selection).expect("read the prior selection");
        let before_write = Instant::now();
        std::fs::write(
            corpus.join(&name),
            format!("Watch marker {token} calibrates temporal flux with quorum heartbeat.\n"),
        )
        .expect("write the watch file");
        loop {
            assert!(watcher.0.try_wait().expect("poll watcher").is_none());
            let selected = std::fs::read(&selection).expect("read the selection");
            if selected != previous && search() {
                break;
            }
            assert!(
                before_write.elapsed() < Duration::from_secs(60),
                "daemon did not see the watch edit {name}:\n{}",
                std::fs::read_to_string(&log_path).unwrap_or_default(),
            );
            std::thread::sleep(WATCH_POLL);
        }
        let daemon_ms = ms(before_write.elapsed());
        let direct = harness
            .run_json(
                "independent search while the watcher is live",
                &[
                    "search",
                    &token,
                    "--index-dir",
                    index_arg,
                    "--no-daemon",
                    "--filter",
                    &filter,
                    "--format",
                    "json",
                ],
            )
            .0;
        assert!(has_hit(&direct["data"], &name), "{name}: {direct}");
        // The first edit proves the watch is active and is a warm-up sample.
        if edit != 0 {
            daemon_visible_ms.push(daemon_ms);
            direct_confirmed_ms.push(ms(before_write.elapsed()));
        }
    }
    assert!(
        Command::new("kill")
            .args(["-TERM", &watcher.0.id().to_string()])
            .status()
            .expect("signal watcher")
            .success()
    );
    let stop_started = Instant::now();
    loop {
        if let Some(status) = watcher.0.try_wait().expect("poll watcher shutdown") {
            assert!(status.success(), "watcher did not exit cleanly: {status}");
            break;
        }
        assert!(stop_started.elapsed() < Duration::from_secs(60));
        std::thread::sleep(WATCH_POLL);
    }
    let last_query = format!("zorblax{WATCH_FILES:04}");
    let last_path = format!("watch-{WATCH_FILES:04}.md");
    let after = harness
        .run_json(
            "search after the watcher exits",
            &[
                "search",
                &last_query,
                "--index-dir",
                index_arg,
                "--no-daemon",
                "--format",
                "json",
            ],
        )
        .0;
    assert!(has_hit(&after["data"], &last_path), "{after}");
    serde_json::json!({
        "warmup_edits": 1,
        "measured_edits": WATCH_FILES,
        "selection_poll_interval_ms": WATCH_POLL.as_millis(),
        "pressure_sampling_interval_ms": 3_600_000,
        "write_to_daemon_visible_ms": distribution(&daemon_visible_ms),
        "write_to_direct_confirmed_after_daemon_ms": distribution(&direct_confirmed_ms),
        "searchable_after_exit": true,
    })
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

    // 1. Index cost through the complete-generation CLI, then an unchanged rerun.
    let index_args = [
        "index",
        &corpus_arg,
        "--index-dir",
        &index_arg,
        "--no-watch-mode",
        "--format",
        "json",
    ];
    let (index, index_wall) = harness.run_json("fsfs index", &index_args);
    let indexed = &index["data"];
    assert_eq!(
        indexed["indexed_files"].as_u64(),
        Some(u64::try_from(docs).expect("corpus size fits u64")),
        "every corpus file must be indexed: {index}"
    );
    assert_eq!(
        indexed["generation_complete"].as_bool(),
        Some(true),
        "{index}"
    );
    let selected = Path::new(indexed["index_root"].as_str().expect("selected index root"));
    assert!(index_dir.join("FSFS-CURRENT").is_file(), "{index}");
    assert!(
        selected.join("vector/quality.fsvi").is_file()
            && indexed["quality_generation"]["is_hash_control"].as_bool() == Some(false),
        "the registered models must produce a quality tier: {index}"
    );
    let (repeat, repeat_wall) = harness.run_json("unchanged fsfs index", &index_args);
    assert_eq!(
        repeat["data"]["unchanged_files"], indexed["indexed_files"],
        "{repeat}"
    );
    let status_json = harness
        .run_json(
            "fsfs status",
            &["status", "--index-dir", &index_arg, "--format", "json"],
        )
        .0;

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
            "--daemon",
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
    let watch_mode = measure_watch(&harness, &corpus, &index_dir, &daemon.socket);
    drop(daemon);

    let receipt = serde_json::json!({
        "schema": RECEIPT_SCHEMA,
        "generated_at_unix": started_at,
        "git_revision": git_revision(),
        "cargo_profile": profile,
        "profile_note": "The Cargo profile describes this harness. The executable may be supplied separately via FSFS_E2E_BINARY.",
        "fsfs_binary": harness.binary.display().to_string(),
        "fsfs_binary_sha256": file_sha256(&harness.binary),
        "host": host,
        "model_root": harness.model_root.display().to_string(),
        "corpus": { "files": docs, "total_chars": total_chars, "generator": "deterministic LCG prose over 6 topic vocabularies, one file per document" },
        "index": {
            "wall_ms": ms(index_wall),
            "indexed_files": indexed["indexed_files"],
            "reported_ms": index["meta"]["duration_ms"],
            "index_size_bytes": indexed["index_size_bytes"],
            "generation_id": indexed["generation_id"],
            "quality_generation": indexed["quality_generation"],
            "unchanged_rerun_ms": ms(repeat_wall),
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
