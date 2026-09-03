//! Release-profile latency and index-cost receipt for the library's two-tier
//! path with the registered models (bd-8s0nf, bridge Gap #3).
//!
//! Opt-in: `FRANKENSEARCH_PERF_RECEIPT=1` runs it; anything else skips with a
//! message so the gate's `facade` stage never pays for it. Debug numbers are
//! meaningless for latency, so run it as
//!
//! ```text
//! FRANKENSEARCH_PERF_RECEIPT=1 cargo test --release -p frankensearch \
//!     --features hybrid --test latency_receipt -- --nocapture
//! ```
//!
//! (the gate's opt-in `perf` stage does exactly that). The JSON receipt is
//! written to `FRANKENSEARCH_PERF_RECEIPT_OUT` when set, else printed.
//!
//! What is measured, on a deterministic 1,000-document synthetic prose corpus
//! (`FRANKENSEARCH_PERF_RECEIPT_DOCS` overrides the size for mechanics checks):
//! `IndexBuilder` build wall/embed/lexical time for both tiers, then 5 warm-up
//! and 50 timed `TwoTierSearcher::search` calls. `initial_ms` is the phase-1
//! latency the library reports at the INITIAL yield (search start to yield);
//! `phase2_ms` is the phase-2 latency it reports at the REFINED yield (quality
//! embed + quality search + blend); `refined_delivery_ms` is their sum, the
//! number the README's "Phase 2 refined delivery" row is about. Stage metrics
//! (`fast_embed_ms`, `vector_search_ms`, `lexical_search_ms`, `quality_embed_ms`,
//! `quality_search_ms`) come from `TwoTierMetrics` for the same queries.
//!
//! `FRANKENSEARCH_PERF_RECEIPT_MAX_REFINED_P95_MS=<ms>` turns the lane into a
//! threshold check (the planted-regression control): it fails when the
//! measured refined-delivery p95 exceeds the bound.
#![cfg(all(feature = "model2vec", feature = "fastembed"))]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use frankensearch::embed::DetectOptions;
use frankensearch::prelude::*;
use frankensearch::{EmbedderStack, IndexBuilder, TwoTierAvailability, TwoTierIndex};
use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::types::SearchPhase;
use sha2::{Digest, Sha256};

const RECEIPT_SCHEMA: &str = "frankensearch-library-two-tier-latency-receipt-v1";
const DEFAULT_DOCS: usize = 1_000;
const WARMUP_QUERIES: usize = 5;
const TIMED_QUERIES: usize = 50;
const TOP_K: usize = 10;

/// Topic vocabularies: each document draws mostly from one topic, so the
/// corpus has real semantic structure for the quality tier to work on.
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

/// Deterministic LCG so the corpus and the queries are the same on every host.
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

fn synthetic_corpus(docs: usize) -> Vec<(String, String)> {
    let mut rng = Lcg(0x5EED_F00D_CAFE_BABE);
    (0..docs)
        .map(|index| {
            let topic = TOPICS[index % TOPICS.len()];
            let words = 60 + usize::try_from(rng.next()).unwrap_or(0) % 60;
            let mut text = String::new();
            for position in 0..words {
                if position > 0 {
                    text.push(' ');
                }
                let roll = rng.next() % 10;
                let word = if roll < 7 {
                    rng.pick(topic)
                } else {
                    rng.pick(CONNECTIVES)
                };
                text.push_str(word);
                if position % 12 == 11 {
                    text.push('.');
                }
            }
            (format!("doc-{index:04}"), text)
        })
        .collect()
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
fn distribution(samples: &[f64]) -> serde_json::Value {
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

fn registered_model_root() -> Option<PathBuf> {
    std::env::var_os("FRANKENSEARCH_MODEL_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join(".local/share/frankensearch/models"))
        })
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn host_fingerprint() -> serde_json::Value {
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

fn rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse().ok())
}

fn git_revision() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn test_binary_sha256() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let bytes = std::fs::read(exe).ok()?;
    let digest = Sha256::digest(&bytes);
    Some(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[test]
fn library_two_tier_latency_receipt() {
    if std::env::var("FRANKENSEARCH_PERF_RECEIPT").as_deref() != Ok("1") {
        eprintln!(
            "SKIPPING latency receipt: set FRANKENSEARCH_PERF_RECEIPT=1 (and run under --release) to measure"
        );
        return;
    }
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let root = registered_model_root().expect("a model root (FRANKENSEARCH_MODEL_DIR or HOME)");
    let stack = EmbedderStack::auto_detect_with_options(
        Some(&root),
        &DetectOptions {
            offline: Some(true),
            ..DetectOptions::default()
        },
    )
    .expect("auto-detect the registered stack");
    assert_eq!(
        stack.availability(),
        TwoTierAvailability::Full,
        "the receipt needs the full semantic stack (potion + MiniLM) under {}",
        root.display()
    );
    let fast = stack.fast_arc();
    let quality = stack
        .quality_arc()
        .expect("a Full stack carries a quality embedder");
    let fast_id = fast.id().to_owned();
    let quality_id = quality.id().to_owned();

    let docs = env_usize("FRANKENSEARCH_PERF_RECEIPT_DOCS", DEFAULT_DOCS);
    let corpus = synthetic_corpus(docs);
    let total_chars: usize = corpus.iter().map(|(_, text)| text.len()).sum();
    let queries = synthetic_queries(WARMUP_QUERIES + TIMED_QUERIES);
    let host = host_fingerprint();
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let receipt_slot: Arc<std::sync::Mutex<Option<serde_json::Value>>> =
        Arc::new(std::sync::Mutex::new(None));
    let slot = Arc::clone(&receipt_slot);
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().expect("tempdir");
        let dir = temp.path().join("index");

        let build_started = Instant::now();
        let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
        for (id, text) in &corpus {
            builder = builder.add_document(id.clone(), text.clone());
        }
        let stats = builder.build(&cx).await.expect("build both tiers");
        let build_wall_ms = ms(build_started.elapsed());
        assert_eq!(stats.doc_count, corpus.len());
        assert!(stats.has_quality_index, "the quality tier must be built");
        assert_eq!(stats.embedder_availability, TwoTierAvailability::Full);

        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let mut searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);
        // The README pipeline is RRF over the lexical AND semantic arms; the
        // builder wrote the Quill arm beside the vector tiers, so the searcher
        // must carry it or this would only measure a vector-only path.
        let lexical_path = stats.lexical.as_ref().map(|receipt| receipt.path.clone());
        let hybrid_lexical = lexical_path.is_some();
        if let Some(path) = lexical_path {
            let lexical =
                frankensearch::QuillIndex::open(&cx, path, frankensearch::QuillConfig::default())
                    .await
                    .expect("open the Quill lexical arm the builder wrote");
            searcher = searcher.with_lexical(Arc::new(lexical));
        }
        assert!(
            hybrid_lexical,
            "the hybrid feature set must build a lexical arm: {stats:?}"
        );

        let mut initial = Vec::with_capacity(TIMED_QUERIES);
        let mut lexical_candidates = Vec::with_capacity(TIMED_QUERIES);
        let mut phase2 = Vec::with_capacity(TIMED_QUERIES);
        let mut refined_delivery = Vec::with_capacity(TIMED_QUERIES);
        let mut wall = Vec::with_capacity(TIMED_QUERIES);
        let mut fast_embed = Vec::with_capacity(TIMED_QUERIES);
        let mut vector_search = Vec::with_capacity(TIMED_QUERIES);
        let mut lexical_search = Vec::with_capacity(TIMED_QUERIES);
        let mut quality_embed = Vec::with_capacity(TIMED_QUERIES);
        let mut quality_search = Vec::with_capacity(TIMED_QUERIES);
        let mut phase2_vectors = Vec::with_capacity(TIMED_QUERIES);
        let mut refined_count = 0_usize;

        for (index, query) in queries.iter().enumerate() {
            let timed = index >= WARMUP_QUERIES;
            let mut initial_latency = None;
            let mut phase2_latency = None;
            let query_started = Instant::now();
            let metrics = searcher
                .search(
                    &cx,
                    query,
                    TOP_K,
                    |_| None,
                    |phase| match phase {
                        SearchPhase::Initial { latency, .. } => initial_latency = Some(latency),
                        SearchPhase::Refined { latency, .. } => phase2_latency = Some(latency),
                        SearchPhase::Reranked { .. } | SearchPhase::RefinementFailed { .. } => {}
                    },
                )
                .await
                .expect("two-tier search");
            let query_wall = query_started.elapsed();
            let initial_latency = initial_latency.expect("INITIAL yield");
            let phase2_latency = phase2_latency.expect("REFINED yield");
            assert!(
                metrics.phase2_vectors_searched > 0,
                "the quality tier must be searched for every query: {metrics:?}"
            );
            if !timed {
                continue;
            }
            refined_count += 1;
            initial.push(ms(initial_latency));
            phase2.push(ms(phase2_latency));
            refined_delivery.push(ms(initial_latency + phase2_latency));
            wall.push(ms(query_wall));
            fast_embed.push(metrics.fast_embed_ms);
            vector_search.push(metrics.vector_search_ms);
            lexical_search.push(metrics.lexical_search_ms);
            quality_embed.push(metrics.quality_embed_ms);
            quality_search.push(metrics.quality_search_ms);
            phase2_vectors.push(metrics.phase2_vectors_searched);
            lexical_candidates.push(metrics.lexical_candidates);
        }
        assert_eq!(refined_count, TIMED_QUERIES);

        let median = |values: &[usize]| -> usize {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        };
        let phase2_vectors_typical = median(&phase2_vectors);
        let lexical_candidates_typical = median(&lexical_candidates);
        assert!(
            lexical_candidates_typical > 0,
            "the lexical arm must contribute candidates on a lexical-friendly corpus"
        );

        let receipt = serde_json::json!({
            "schema": RECEIPT_SCHEMA,
            "generated_at_unix": started_at,
            "git_revision": git_revision(),
            "cargo_profile": profile,
            "profile_note": "cargo test --release (opt-level 3, thin LTO default); the shipped fsfs binary adds fat LTO and codegen-units=1, so its numbers can only be better",
            "test_binary_sha256": test_binary_sha256(),
            "host": host,
            "models": { "fast": fast_id, "quality": quality_id, "model_root": root.display().to_string() },
            "corpus": { "docs": corpus.len(), "total_chars": total_chars, "generator": "deterministic LCG prose over 6 topic vocabularies" },
            "index": {
                "build_wall_ms": build_wall_ms,
                "builder_total_ms": stats.total_ms,
                "embed_ms_both_tiers": stats.embed_ms,
                "lexical_ms": stats.lexical_ms,
                "quality_docs": stats.quality_indexed,
                "size_bytes": {
                    "total": stats.size_bytes.total,
                    "vector_fast": stats.size_bytes.vector_fast,
                    "vector_quality": stats.size_bytes.vector_quality,
                    "lexical": stats.size_bytes.lexical,
                },
                "lexical_backend": stats.lexical.as_ref().map(|receipt| receipt.backend),
            },
            "queries": { "warmup": WARMUP_QUERIES, "timed": TIMED_QUERIES, "top_k": TOP_K },
            "initial_ms": distribution(&initial),
            "phase2_ms": distribution(&phase2),
            "refined_delivery_ms": distribution(&refined_delivery),
            "search_wall_ms": distribution(&wall),
            "stages_ms": {
                "fast_embed": distribution(&fast_embed),
                "vector_search": distribution(&vector_search),
                "lexical_search": distribution(&lexical_search),
                "quality_embed": distribution(&quality_embed),
                "quality_search": distribution(&quality_search),
            },
            "hybrid_lexical": hybrid_lexical,
            "lexical_candidates_typical": lexical_candidates_typical,
            "phase2_vectors_searched_typical": phase2_vectors_typical,
            "rss_kb_at_end": rss_kb(),
        });
        *slot.lock().expect("receipt slot") = Some(receipt);
    });
    let receipt = receipt_slot
        .lock()
        .expect("receipt slot")
        .take()
        .expect("the measurement block produced a receipt");

    let encoded = serde_json::to_string_pretty(&receipt).expect("encode receipt");
    match std::env::var_os("FRANKENSEARCH_PERF_RECEIPT_OUT") {
        Some(path) => {
            let path = PathBuf::from(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create receipt directory");
            }
            std::fs::write(&path, format!("{encoded}\n")).expect("write receipt");
            eprintln!("latency receipt written to {}", path.display());
        }
        None => println!("{encoded}"),
    }

    if let Ok(bound) = std::env::var("FRANKENSEARCH_PERF_RECEIPT_MAX_REFINED_P95_MS") {
        let bound: f64 = bound.trim().parse().expect("numeric p95 bound");
        let p95 = receipt["refined_delivery_ms"]["p95_ms"]
            .as_f64()
            .expect("refined delivery p95");
        assert!(
            p95 <= bound,
            "refined delivery p95 {p95:.1} ms exceeds the bound {bound:.1} ms"
        );
    }
}
