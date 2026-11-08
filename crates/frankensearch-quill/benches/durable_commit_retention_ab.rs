//! Durable-writer retention A/B: how much of every segment a long-lived on-disk
//! writer keeps resident, and what each commit costs as segments accumulate.
//!
//! The publication preflight fully verifies every new segment before the
//! MANIFEST may name it. The incumbent hashed it through the writer's mapping,
//! which faulted every page in, and the writer keeps every published segment
//! mapped, so its resident set grew by every byte it ever published. The
//! candidate reads the same bytes through the descriptor that backs the mapping
//! (the same checks, in the same order, with the same errors) and leaves those
//! pages unfaulted.
//!
//! Three durable CASS-schema indexes are built in this one process through the
//! shipping `index_schema_documents` + `commit` route with CASS's writer
//! configuration: the incumbent, a second incumbent (the A/A null), and the
//! candidate. Every commit ingests the same documents into all three and then
//! commits them in a rotating order. Per-commit wall time is paired per commit
//! (`null/incumbent`, `candidate/incumbent`); each index's resident file pages
//! are read from `/proc/self/smaps` (mappings under its own directory) at
//! checkpoints. After the build, all three must agree exactly on every
//! published segment (docid range, document count, seal sequence, section
//! bytes), the field statistics, the live document count, and every probe
//! query's hits and exact count.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 rch exec -- env QUILL_RETAIN_COMMITS=300 \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!     --profile release --bench durable_commit_retention_ab
//! ```

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::bench_support::print_bench_elf_sha256;
use frankensearch_quill::cass::CassDocument;
use frankensearch_quill::query::{CassQueryFilters, CassQueryParser};
use frankensearch_quill::schema::CASS_SEMANTIC_SCHEMA;
use frankensearch_quill::{KeeperSnapshot, QuillConfig, QuillIndex, SchemaDocument};

const AGENTS: [&str; 3] = ["claude_code", "codex", "gemini"];
const MESSAGES_PER_CONVERSATION: usize = 40;
/// One document in this many carries the probe token.
const NEEDLE_EVERY: usize = 97;
const PROBE_QUERIES: [&str; 4] = ["quillneedle", "w7", "w11 OR w40123", "\"w3 w5\""];
const ARMS: [&str; 3] = ["incumbent", "null", "candidate"];
/// Commits whose resident set is read and reported.
const CHECKPOINTS: [usize; 8] = [1, 25, 50, 100, 150, 200, 250, 300];

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Deterministic CASS-shaped messages `first..first + count`: a skewed
/// vocabulary (a 600-term head and a 50,000-term tail), 12-59 words each.
fn corpus(first: usize, count: usize) -> Vec<SchemaDocument> {
    (first..first + count)
        .map(|ordinal| {
            let mut state = u64::try_from(ordinal)
                .expect("ordinal")
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                | 1;
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

/// Process `RssAnon` in KiB.
fn rss_anon_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("RssAnon:"))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(0)
}

/// Resident KiB of every mapping under each directory, from `/proc/self/smaps`.
fn mapped_rss_kib(directories: &[PathBuf]) -> Vec<u64> {
    let mut totals = vec![0_u64; directories.len()];
    let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") else {
        return totals;
    };
    let mut owner = None;
    for line in smaps.lines() {
        let mut fields = line.split_whitespace();
        let Some(first) = fields.next() else {
            continue;
        };
        if first.contains('-') && first.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
            let path = fields.nth(4).map(Path::new);
            owner = path.and_then(|path| {
                directories
                    .iter()
                    .position(|directory| path.starts_with(directory))
            });
        } else if first == "Rss:"
            && let Some(index) = owner
        {
            totals[index] += fields
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
        }
    }
    totals
}

fn config(shards: usize) -> QuillConfig {
    // CASS's writer configuration (`cass_quill_config`) with its bulk
    // rebuild's 4 GiB scribe budget: explicit publication only, no tier merge.
    QuillConfig {
        max_visibility_lag_ms: u64::MAX,
        tier_fanout: usize::MAX,
        max_ingest_shards: shards,
        scribe_shard_budget_bytes: 4 << 30,
        ..QuillConfig::default()
    }
}

/// Every published segment in manifest order: docid range, document count,
/// seal sequence and the exact bytes of every section. Segment identifiers
/// and creation times are wall-clock derived and not compared.
fn assert_same_segments(arm: &str, incumbent: &KeeperSnapshot, other: &KeeperSnapshot) {
    let (left, right) = (incumbent.segments(), other.segments());
    assert_eq!(left.len(), right.len(), "{arm}: segment count");
    for (position, (l, r)) in left.iter().zip(right).enumerate() {
        let (lm, rm) = (l.manifest(), r.manifest());
        assert_eq!(
            (lm.docid_lo, lm.docid_hi, lm.doc_count, lm.seal_seq),
            (rm.docid_lo, rm.docid_hi, rm.doc_count, rm.seal_seq),
            "{arm}: segment {position} manifest entry"
        );
        assert_eq!(
            l.section_entries().len(),
            r.section_entries().len(),
            "{arm}: segment {position} sections"
        );
        let range = |entry: &frankensearch_quill::segment::SectionEntry| {
            let offset = usize::try_from(entry.offset).expect("offset");
            offset..offset + usize::try_from(entry.len).expect("len")
        };
        for (le, re) in l.section_entries().iter().zip(r.section_entries()) {
            assert_eq!(le.kind, re.kind, "{arm}: segment {position} section kind");
            let left_payload = &l.source_bytes()[range(le)];
            let right_payload = &r.source_bytes()[range(re)];
            assert!(
                left_payload == right_payload,
                "{arm}: segment {position} section {} bytes",
                le.kind.raw()
            );
        }
    }
}

type ProbeResult = (Vec<(String, u32, u32)>, Option<u64>);

fn probe(cx: &Cx, index: &QuillIndex, parser: &CassQueryParser) -> Vec<ProbeResult> {
    PROBE_QUERIES
        .iter()
        .map(|raw| {
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

/// Nearest-rank percentile of a sorted, non-empty slice.
fn percentile(sorted: &[f64], percent: usize) -> f64 {
    sorted[((sorted.len() - 1) * percent + 50) / 100]
}

/// p50 [p5, p95] of `values`.
fn spread(values: &[f64]) -> (f64, f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    (
        percentile(&sorted, 50),
        percentile(&sorted, 5),
        percentile(&sorted, 95),
    )
}

fn drive<T>(future: impl std::future::Future<Output = T>) -> T {
    asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("bench runtime")
        .block_on(future)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let _ = print_bench_elf_sha256();
    let commits = env_usize("QUILL_RETAIN_COMMITS", 300);
    let shards = env_usize("QUILL_RETAIN_SHARDS", 8);
    let per_shard = env_usize("QUILL_RETAIN_DOCS_PER_SHARD", 1_000);
    let per_commit = per_shard * shards;
    let batch = 512 * shards;
    let root = tempfile::tempdir().expect("bench root");
    let directories: Vec<PathBuf> = ARMS.iter().map(|arm| root.path().join(arm)).collect();
    eprintln!(
        "[harness] commits={commits} shards={shards} docs_per_commit={per_commit} root={}",
        root.path().display()
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(shards)
        .build()
        .expect("bench rayon pool");
    pool.install(|| {
        let cx = Cx::for_testing();
        drive(async {
            let mut indexes = Vec::new();
            for (arm, directory) in ARMS.iter().zip(&directories) {
                let index = QuillIndex::create_with_schema(
                    &cx,
                    directory.clone(),
                    CASS_SEMANTIC_SCHEMA,
                    config(shards),
                )
                .await
                .expect("durable CASS index");
                index
                    .bench_set_mapped_preflight(&cx, *arm != "candidate")
                    .await
                    .expect("select the preflight arm");
                indexes.push(index);
            }
            // `/proc/self/smaps` names mappings by their resolved paths.
            let directories: Vec<PathBuf> = directories
                .iter()
                .map(|directory| std::fs::canonicalize(directory).expect("index directory"))
                .collect();
            let anon0 = rss_anon_kib();
            // commit_ms[arm][commit]
            let mut commit_ms = vec![Vec::with_capacity(commits); ARMS.len()];
            let mut segments = 0;
            for commit in 0..commits {
                let documents = corpus(commit * per_commit, per_commit);
                for index in &indexes {
                    for chunk in documents.chunks(batch) {
                        index
                            .index_schema_documents(&cx, black_box(chunk))
                            .await
                            .expect("ingest");
                    }
                }
                drop(documents);
                for offset in 0..ARMS.len() {
                    let arm = (commit + offset) % ARMS.len();
                    let started = Instant::now();
                    let snapshot = indexes[arm].commit(&cx).await.expect("commit");
                    commit_ms[arm].push(started.elapsed().as_secs_f64() * 1e3);
                    segments = snapshot.segments().len();
                }
                if CHECKPOINTS.contains(&(commit + 1)) || commit + 1 == commits {
                    let published_kib = indexes[0]
                        .snapshot()
                        .expect("snapshot")
                        .loaded_manifest()
                        .manifest
                        .segments
                        .iter()
                        .map(|segment| segment.file_len)
                        .sum::<u64>()
                        / 1024;
                    let resident = mapped_rss_kib(&directories);
                    let mut line = format!(
                        "[resident] commit={} segments={segments} published_mib={:.1} \
                         process_anon_growth_mib={:.1}",
                        commit + 1,
                        published_kib as f64 / 1024.0,
                        rss_anon_kib().saturating_sub(anon0) as f64 / 1024.0,
                    );
                    for (arm, kib) in ARMS.iter().zip(&resident) {
                        let _ = write!(
                            line,
                            " {arm}_mib={:.1} ({:.1}% of published, {:.1} KiB/segment)",
                            *kib as f64 / 1024.0,
                            100.0 * *kib as f64 / published_kib.max(1) as f64,
                            *kib as f64 / segments.max(1) as f64,
                        );
                    }
                    eprintln!("{line}");
                }
            }

            // Parity before reporting any number.
            let parser = CassQueryParser::new(CASS_SEMANTIC_SCHEMA).expect("CASS parser");
            let incumbent = indexes[0].snapshot().expect("incumbent snapshot");
            let expected_hits = probe(&cx, &indexes[0], &parser);
            assert!(
                expected_hits.iter().all(|(hits, _)| !hits.is_empty()),
                "every probe query must match"
            );
            for (arm, index) in ARMS.iter().zip(&indexes).skip(1) {
                let other = index.snapshot().expect("snapshot");
                assert_same_segments(arm, &incumbent, &other);
                assert_eq!(
                    incumbent.loaded_manifest().manifest.field_stats,
                    other.loaded_manifest().manifest.field_stats,
                    "{arm}: field statistics"
                );
                assert_eq!(
                    indexes[0].doc_count().expect("docs"),
                    index.doc_count().expect("docs"),
                    "{arm}: live documents"
                );
                assert_eq!(
                    expected_hits,
                    probe(&cx, index, &parser),
                    "{arm}: probe hits"
                );
            }
            eprintln!(
                "[parity] arms={} segments={} docs={} probes={} identical",
                ARMS.len(),
                incumbent.segments().len(),
                indexes[0].doc_count().expect("docs"),
                PROBE_QUERIES.len()
            );
            drop(incumbent);

            let windows = [(0, 25), (75, 125), (commits.saturating_sub(25), commits)];
            for (start, end) in windows {
                let end = end.min(commits);
                if start >= end {
                    continue;
                }
                let ratios = |arm: usize| -> Vec<f64> {
                    (start..end)
                        .map(|commit| commit_ms[arm][commit] / commit_ms[0][commit])
                        .collect()
                };
                let (null, null_lo, null_hi) = spread(&ratios(1));
                let (lever, lever_lo, lever_hi) = spread(&ratios(2));
                let (inc, inc_lo, inc_hi) = spread(&commit_ms[0][start..end]);
                let (cand, cand_lo, cand_hi) = spread(&commit_ms[2][start..end]);
                eprintln!(
                    "[commit] commits={}-{end} null={null:.4} [p5 {null_lo:.4}, p95 {null_hi:.4}] \
                     lever(candidate/incumbent)={lever:.4} [p5 {lever_lo:.4}, p95 {lever_hi:.4}] \
                     incumbent_ms p50={inc:.1} [{inc_lo:.1}, {inc_hi:.1}] \
                     candidate_ms p50={cand:.1} [{cand_lo:.1}, {cand_hi:.1}]",
                    start + 1,
                );
            }
            for (arm, times) in ARMS.iter().zip(&commit_ms) {
                let at = |commit: usize| {
                    times
                        .get(commit - 1)
                        .map_or_else(|| "n/a".to_owned(), |ms| format!("{ms:.1}"))
                };
                eprintln!(
                    "[commit_ms] arm={arm} commit1={} commit100={} commit{commits}={}",
                    at(1),
                    at(100),
                    at(commits.max(1)),
                );
            }
            drop(indexes);
        });
    });
}
