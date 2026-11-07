#![recursion_limit = "256"]
//! Fan-out lease-cut seal A/B: one-at-a-time lease-cut seals (incumbent) vs
//! building every shard's simultaneous lease cut together (shipping).
//!
//! A fan-out batch gives every ingest shard the same share of each wave, so
//! the shards exhaust their 65,536-docid leases in lockstep. Each exhausted
//! lease forces a segment cut. The incumbent sealed those cuts one shard at a
//! time on the writer thread (scalar radix), leaving the other shards idle;
//! with a scribe budget too large to cut first (CASS's bulk rebuilds run at
//! 4 GiB) every mid-ingest seal is such a lease cut. The candidate builds the
//! round's cuts together and installs them in the same shard order.
//!
//! Lockstep holds only while every batch is full. The `cass_page_tails` cell
//! submits CASS's real shape (pages of full batches plus a tail), whose narrow
//! tails let the low shards run ahead, so fewer cuts share a round.
//!
//! Both arms ingest the identical CASS-schema corpus through the shipping
//! `index_schema_documents` + `commit` path (the route CASS takes), with the
//! same rayon pool, shard count, batch size and commit cadence. Before timing,
//! one build of each arm must agree exactly on every published segment
//! (docid range, document count, seal sequence, section payload bytes), the
//! field statistics, the live document count, and every probe query's hits
//! (identifier, global docid, score bits) and exact count. Segment identifiers
//! and creation timestamps are wall-clock derived in both arms and are the only
//! fields not compared.
//!
//! The A/A null is incumbent-vs-incumbent; the lever ratio is
//! `candidate / incumbent` on ingest wall time, so `< 1.0` outside the null
//! band wins. Query latency over the two resulting indexes is compared the
//! same way (paired, alternating, with its own A/A null) as a parity check:
//! the segment sets are identical, so it should not move.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_LEASE_SCALE=full QUILL_LEASE_ROUNDS=10 QUILL_LEASE_THREADS=8 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench lease_cut_seal_ab
//!
//! # Fast parity/harness check (not performance evidence):
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_LEASE_SCALE=smoke QUILL_LEASE_ROUNDS=10 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench lease_cut_seal_ab
//! ```

use std::cell::RefCell;
use std::fmt::Write as _;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::bench_support::{paired_median_ratio, print_bench_elf_sha256};
use frankensearch_quill::cass::CassDocument;
use frankensearch_quill::query::{CassQueryFilters, CassQueryParser};
use frankensearch_quill::schema::CASS_SEMANTIC_SCHEMA;
use frankensearch_quill::scribe::DOC_ORDS_PER_LEASE;
use frankensearch_quill::{KeeperSnapshot, QuillConfig, QuillIndex, SchemaDocument};

/// One ingest shape. Every shard receives `per_shard_per_commit` documents
/// between commits, `commits` times.
struct Cell {
    name: &'static str,
    per_shard_per_commit: usize,
    commits: usize,
    scribe_shard_budget_bytes: usize,
    /// `None`: every batch is full. `Some((lo, hi))`: the documents arrive in
    /// pending pages of `lo..hi` documents per shard, each submitted as full
    /// batches plus one tail batch of whatever is left, as CASS does.
    pending_per_shard: Option<(usize, usize)>,
}

/// CASS's bulk rebuild submits 4,096-message batches at minimum, which a
/// fan-out spreads over up to sixteen shards; larger hosts submit more.
const BATCH_DOCS_PER_SHARD: usize = 512;

static FULL_CELLS: [Cell; 3] = [
    // The live 74.5M-document rebuild: a 4 GiB scribe budget, so only lease
    // exhaustion and commits cut segments, and about 1.5 leases per shard per
    // commit (3.2M messages over 32 shards).
    Cell {
        name: "budget4g_1.5leases_per_commit",
        per_shard_per_commit: 98_304,
        commits: 2,
        scribe_shard_budget_bytes: 4 << 30,
        pending_per_shard: None,
    },
    // CASS main's default 64 MiB budget at its default cadence (800k
    // messages over 32 shards, 25k per shard per commit).
    Cell {
        name: "budget64m_0.4leases_per_commit",
        per_shard_per_commit: 24_576,
        commits: 6,
        scribe_shard_budget_bytes: 64 << 20,
        pending_per_shard: None,
    },
    // The live rebuild's batch shape: CASS flushes a page of about 1,024
    // conversations (about 14,700 messages per shard at 32 shards) as full
    // batches plus a tail. A tail under 256 documents per shard reaches only
    // the low shards, so their leases run ahead and the shards stop crossing
    // lease boundaries in the same round.
    Cell {
        name: "budget4g_cass_page_tails",
        per_shard_per_commit: 98_304,
        commits: 2,
        scribe_shard_budget_bytes: 4 << 30,
        pending_per_shard: Some((12_288, 17_408)),
    },
];

static SMOKE_CELLS: [Cell; 1] = [Cell {
    name: "smoke_budget4g",
    per_shard_per_commit: 40_000,
    commits: 2,
    scribe_shard_budget_bytes: 4 << 30,
    pending_per_shard: None,
}];

const AGENTS: [&str; 3] = ["claude_code", "codex", "gemini"];
const MESSAGES_PER_CONVERSATION: usize = 40;
/// One document in this many carries the probe token.
const NEEDLE_EVERY: usize = 97;

/// Paired rounds of the probe-set query latency comparison.
const QUERY_ROUNDS: usize = 51;

const PROBE_QUERIES: [(&str, &str); 4] = [
    ("needle_term", "quillneedle"),
    ("head_term", "w7"),
    ("union2", "w11 OR w40123"),
    ("phrase", "\"w3 w5\""),
];

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Deterministic CASS-shaped messages: a skewed vocabulary (a 600-term head
/// and a 50,000-term tail), 12-59 words each, 40 messages per conversation.
fn corpus(documents: usize, seed: u64) -> Vec<SchemaDocument> {
    let mut state = seed | 1;
    (0..documents)
        .map(|ordinal| {
            let words = 12 + usize::try_from(xorshift(&mut state) % 48).expect("word count");
            let mut content = String::with_capacity(words * 7);
            for position in 0..words {
                if position != 0 {
                    content.push(' ');
                }
                let roll = xorshift(&mut state);
                let term = if roll % 4 == 0 {
                    roll % 50_000
                } else {
                    roll % 600
                };
                let _ = write!(content, "w{term}");
            }
            if ordinal % NEEDLE_EVERY == 0 {
                content.push_str(" quillneedle");
            }
            let conversation = ordinal / MESSAGES_PER_CONVERSATION;
            let msg_idx = ordinal % MESSAGES_PER_CONVERSATION;
            CassDocument {
                agent: AGENTS[conversation % AGENTS.len()].to_owned(),
                workspace: Some(format!("/work/project{}", conversation % 17)),
                workspace_original: None,
                source_path: format!("/sessions/s{conversation:07}.jsonl"),
                msg_idx: u64::try_from(msg_idx).expect("message index"),
                created_at: Some(1_700_000_000 + i64::try_from(ordinal).expect("timestamp")),
                title: (msg_idx == 0).then(|| format!("session {conversation}")),
                content,
                source_id: "local".to_owned(),
                origin_kind: "local".to_owned(),
                origin_host: None,
                conversation_id: Some(i64::try_from(conversation).expect("conversation id")),
            }
            .to_schema_document()
        })
        .collect()
}

fn config(cell: &Cell, shards: usize) -> QuillConfig {
    // CASS's writer configuration (`cass_quill_config`): explicit publication
    // only and no engine tier merge, so the segments each commit produces stay
    // exactly what the seal path built.
    QuillConfig {
        max_visibility_lag_ms: u64::MAX,
        tier_fanout: usize::MAX,
        max_ingest_shards: shards,
        scribe_shard_budget_bytes: cell.scribe_shard_budget_bytes,
        ..QuillConfig::default()
    }
}

/// Ingest the whole corpus through the shipping path and return the index and
/// the segment count after each commit.
async fn build(
    cx: &Cx,
    cell: &Cell,
    shards: usize,
    documents: &[SchemaDocument],
    serial_lease_cut_seals: bool,
) -> (QuillIndex, Vec<usize>) {
    let index = QuillIndex::in_memory_with_schema(CASS_SEMANTIC_SCHEMA, config(cell, shards))
        .expect("in-memory CASS index");
    index
        .bench_set_serial_lease_cut_seals(cx, serial_lease_cut_seals)
        .await
        .expect("select the lease-cut seal arm");
    let per_commit = cell.per_shard_per_commit * shards;
    let batch = BATCH_DOCS_PER_SHARD * shards;
    let mut segments_after_commit = Vec::with_capacity(cell.commits);
    // Both arms draw the same page sizes.
    let mut page_state = 0x7a11_5eed_u64;
    for commit in documents.chunks(per_commit) {
        let mut rest = commit;
        while !rest.is_empty() {
            let page = cell
                .pending_per_shard
                .map_or(rest.len(), |(lo, hi)| {
                    let span = u64::try_from((hi - lo) * shards).expect("page span");
                    let extra = xorshift(&mut page_state) % span;
                    lo * shards + usize::try_from(extra).expect("page size")
                })
                .min(rest.len());
            let (pending, next) = rest.split_at(page);
            for batch in pending.chunks(batch) {
                index
                    .index_schema_documents(cx, black_box(batch))
                    .await
                    .expect("ingest batch");
            }
            rest = next;
        }
        index.commit(cx).await.expect("commit");
        segments_after_commit.push(index.snapshot().expect("snapshot").segments().len());
    }
    (index, segments_after_commit)
}

/// Assert that two published segment sets are identical in everything a
/// reader can observe: per segment the docid range, document count, seal
/// sequence and the exact bytes of every section, in manifest order. Only the
/// wall-clock-derived segment identifier and creation time are not compared.
fn assert_same_segments(name: &str, incumbent: &KeeperSnapshot, candidate: &KeeperSnapshot) {
    let incumbent = incumbent.segments();
    let candidate = candidate.segments();
    assert_eq!(incumbent.len(), candidate.len(), "{name}: segment count");
    for (position, (left, right)) in incumbent.iter().zip(candidate).enumerate() {
        let (lm, rm) = (left.manifest(), right.manifest());
        assert_eq!(
            (lm.docid_lo, lm.docid_hi, lm.doc_count, lm.seal_seq),
            (rm.docid_lo, rm.docid_hi, rm.doc_count, rm.seal_seq),
            "{name}: segment {position} manifest entry"
        );
        let (ls, rs) = (left.section_entries(), right.section_entries());
        assert_eq!(ls.len(), rs.len(), "{name}: segment {position} sections");
        for (le, re) in ls.iter().zip(rs) {
            assert_eq!(
                le.kind.raw(),
                re.kind.raw(),
                "{name}: segment {position} kind"
            );
            let lo = usize::try_from(le.offset).expect("offset");
            let ro = usize::try_from(re.offset).expect("offset");
            let ll = usize::try_from(le.len).expect("len");
            let rl = usize::try_from(re.len).expect("len");
            assert!(
                left.source_bytes()[lo..lo + ll] == right.source_bytes()[ro..ro + rl],
                "{name}: segment {position} section {} bytes",
                le.kind.raw()
            );
        }
    }
}

/// One probe query's ranked hits (identifier, global docid, score bits) and
/// exact count.
type ProbeResult = (Vec<(String, u32, u32)>, Option<u64>);

fn probe(cx: &Cx, index: &QuillIndex, parser: &CassQueryParser) -> Vec<ProbeResult> {
    PROBE_QUERIES
        .iter()
        .map(|(_, raw)| {
            let parsed = parser.parse(raw, &CassQueryFilters::default());
            let result = index
                .search_preparsed_paginated(cx, &parsed.query, 50, 0, true)
                .expect("probe query");
            let hits = result
                .hits
                .iter()
                .map(|hit| {
                    (
                        hit.document_id.clone(),
                        hit.global_docid,
                        hit.score.to_bits(),
                    )
                })
                .collect();
            (hits, result.total_count)
        })
        .collect()
}

fn summarize(name: &str, arm: &str, index: &QuillIndex, after_commit: &[usize]) {
    let snapshot = index.snapshot().expect("snapshot");
    let manifest = &snapshot.loaded_manifest().manifest;
    let mut sizes: Vec<u32> = manifest.segments.iter().map(|s| s.doc_count).collect();
    sizes.sort_unstable();
    let lease_aligned = manifest
        .segments
        .iter()
        .filter(|s| s.docid_hi % u64::from(DOC_ORDS_PER_LEASE) == 0)
        .count();
    let docs: u64 = sizes.iter().map(|&n| u64::from(n)).sum();
    eprintln!(
        "[layout] cell={name} arm={arm} docs={docs} segments={} per_commit_cumulative={after_commit:?} \
         ending_on_lease_boundary={lease_aligned} docs_per_segment min={} p50={} max={} mean={}",
        sizes.len(),
        sizes.first().copied().unwrap_or(0),
        sizes.get(sizes.len() / 2).copied().unwrap_or(0),
        sizes.last().copied().unwrap_or(0),
        docs / u64::try_from(sizes.len().max(1)).expect("segment count"),
    );
}

/// Paired probe-set latency over the two published indexes, in alternating
/// order, with an incumbent-vs-incumbent null. Every call uses a limit no
/// earlier call used: the ranked-query cache keys on the limit, so it never
/// answers a timed query.
fn query_ratio(
    cx: &Cx,
    name: &str,
    incumbent: &QuillIndex,
    candidate: &QuillIndex,
    parser: &CassQueryParser,
) {
    let parsed: Vec<_> = PROBE_QUERIES
        .iter()
        .map(|(_, raw)| parser.parse(raw, &CassQueryFilters::default()))
        .collect();
    let next_limit = AtomicUsize::new(20);
    let samples = RefCell::new(Vec::new());
    let probe_set = |index: &QuillIndex, arm: Option<bool>| {
        let limit = next_limit.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        for query in &parsed {
            black_box(
                index
                    .search_preparsed_paginated(cx, black_box(&query.query), limit, 0, false)
                    .expect("timed query"),
            );
        }
        if let Some(candidate_arm) = arm {
            samples
                .borrow_mut()
                .push((candidate_arm, started.elapsed().as_secs_f64() * 1e6));
        }
    };
    let null = paired_median_ratio(
        QUERY_ROUNDS,
        1,
        || probe_set(incumbent, None),
        || probe_set(incumbent, None),
    );
    let lever = paired_median_ratio(
        QUERY_ROUNDS,
        1,
        || probe_set(incumbent, Some(false)),
        || probe_set(candidate, Some(true)),
    );
    let p50 = |candidate_arm: bool| {
        let mut arm: Vec<f64> = samples
            .borrow()
            .iter()
            .filter(|(is_candidate, _)| *is_candidate == candidate_arm)
            .map(|(_, us)| *us)
            .collect();
        arm.sort_unstable_by(f64::total_cmp);
        arm[arm.len() / 2]
    };
    eprintln!(
        "[query] cell={name} probe_set rounds={QUERY_ROUNDS} null={:.4} null_ci95=[{:.4}, {:.4}] \
         null_admissible={} lever(candidate/incumbent)={:.4} ci95=[{:.4}, {:.4}] decidable={} \
         incumbent_us p50={:.1} candidate_us p50={:.1}",
        null.median,
        null.median_ci95_low,
        null.median_ci95_high,
        null.is_admissible_null(),
        lever.median,
        lever.median_ci95_low,
        lever.median_ci95_high,
        lever.decidable_against(&null),
        p50(false),
        p50(true),
    );
}

fn run_cell(cx: &Cx, cell: &Cell, shards: usize, rounds: usize) {
    let documents = corpus(
        cell.per_shard_per_commit * shards * cell.commits,
        0x5eed_1ea5,
    );
    let parser = CassQueryParser::new(CASS_SEMANTIC_SCHEMA).expect("CASS parser");

    // Parity before timing.
    let (incumbent, incumbent_commits) = drive(build(cx, cell, shards, &documents, true));
    let (candidate, candidate_commits) = drive(build(cx, cell, shards, &documents, false));
    let incumbent_snapshot = incumbent.snapshot().expect("incumbent snapshot");
    let candidate_snapshot = candidate.snapshot().expect("candidate snapshot");
    assert_eq!(
        incumbent_commits, candidate_commits,
        "{}: segments per commit",
        cell.name
    );
    assert_same_segments(cell.name, &incumbent_snapshot, &candidate_snapshot);
    assert_eq!(
        incumbent_snapshot.loaded_manifest().manifest.field_stats,
        candidate_snapshot.loaded_manifest().manifest.field_stats,
        "{}: field statistics",
        cell.name
    );
    assert_eq!(
        incumbent.doc_count().expect("incumbent docs"),
        candidate.doc_count().expect("candidate docs"),
        "{}: live documents",
        cell.name
    );
    let incumbent_hits = probe(cx, &incumbent, &parser);
    assert!(
        incumbent_hits.iter().all(|(hits, _)| !hits.is_empty()),
        "{}: every probe query must match",
        cell.name
    );
    assert_eq!(
        incumbent_hits,
        probe(cx, &candidate, &parser),
        "{}: probe hits",
        cell.name
    );
    summarize(cell.name, "incumbent", &incumbent, &incumbent_commits);
    summarize(cell.name, "candidate", &candidate, &candidate_commits);
    query_ratio(cx, cell.name, &incumbent, &candidate, &parser);
    drop((incumbent_snapshot, candidate_snapshot, incumbent, candidate));

    let wall = RefCell::new(Vec::new());
    let ingest = |serial: bool| {
        let started = Instant::now();
        let built = drive(build(cx, cell, shards, black_box(&documents), serial));
        wall.borrow_mut()
            .push((serial, started.elapsed().as_secs_f64()));
        black_box(built);
    };
    let null = paired_median_ratio(rounds, 1, || ingest(true), || ingest(true));
    let lever = paired_median_ratio(rounds, 1, || ingest(true), || ingest(false));
    let mut incumbent_s: Vec<f64> = wall
        .borrow()
        .iter()
        .filter(|(serial, _)| *serial)
        .map(|(_, s)| *s)
        .collect();
    let mut candidate_s: Vec<f64> = wall
        .borrow()
        .iter()
        .filter(|(serial, _)| !*serial)
        .map(|(_, s)| *s)
        .collect();
    incumbent_s.sort_unstable_by(f64::total_cmp);
    candidate_s.sort_unstable_by(f64::total_cmp);
    eprintln!(
        "[ingest] cell={} shards={shards} docs={} batch={} commits={} budget_mib={} \
         null={:.4} [p5 {:.4}, p95 {:.4}] null_ci95=[{:.4}, {:.4}] null_admissible={} \
         lever(candidate/incumbent)={:.4} [p5 {:.4}, p95 {:.4}] \
         ci95=[{:.4}, {:.4}] decidable={} incumbent_s p50={:.3} [{:.3}, {:.3}] \
         candidate_s p50={:.3} [{:.3}, {:.3}]",
        cell.name,
        documents.len(),
        BATCH_DOCS_PER_SHARD * shards,
        cell.commits,
        cell.scribe_shard_budget_bytes >> 20,
        null.median,
        null.p5,
        null.p95,
        null.median_ci95_low,
        null.median_ci95_high,
        null.is_admissible_null(),
        lever.median,
        lever.p5,
        lever.p95,
        lever.median_ci95_low,
        lever.median_ci95_high,
        lever.decidable_against(&null),
        incumbent_s[incumbent_s.len() / 2],
        incumbent_s[0],
        incumbent_s[incumbent_s.len() - 1],
        candidate_s[candidate_s.len() / 2],
        candidate_s[0],
        candidate_s[candidate_s.len() - 1],
    );
}

fn drive<T>(future: impl std::future::Future<Output = T>) -> T {
    asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("bench runtime")
        .block_on(future)
}

fn main() {
    let _ = print_bench_elf_sha256();
    let scale = std::env::var("QUILL_LEASE_SCALE").unwrap_or_else(|_| "full".to_owned());
    let rounds: usize = std::env::var("QUILL_LEASE_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);
    let shards: usize = std::env::var("QUILL_LEASE_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let only = std::env::var("QUILL_LEASE_CELL").ok();
    let cells: &'static [Cell] = if scale == "smoke" {
        &SMOKE_CELLS
    } else {
        &FULL_CELLS
    };
    eprintln!(
        "[harness] scale={scale} rounds={rounds} shards={shards} lease={DOC_ORDS_PER_LEASE} \
         batch_per_shard={BATCH_DOCS_PER_SHARD}"
    );
    // The writer sizes its shard table from the pool it is built on, and the
    // fan-out and parallel seals run on the same pool.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(shards)
        .build()
        .expect("bench rayon pool");
    pool.install(|| {
        let cx = Cx::for_testing();
        for cell in cells {
            if only.as_deref().is_some_and(|name| name != cell.name) {
                continue;
            }
            run_cell(&cx, cell, shards, rounds);
        }
    });
}
