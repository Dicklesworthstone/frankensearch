//! Per-commit publication cost as the published segment count grows.
//!
//! A CASS bulk rebuild commits over and over while the segment count climbs
//! into the thousands, and every commit republishes the whole MANIFEST. The
//! publication preflight looked up each proposal segment's retained binding by
//! scanning every retained segment, and the reopened snapshot looked up each
//! MANIFEST segment's verified candidate by scanning every remaining candidate,
//! so both were quadratic in the segment count (incumbent). The candidate finds
//! both by segment id.
//!
//! Three durable CASS-schema indexes are built in this one process through the
//! shipping `index_schema_documents` + `commit` route with CASS's writer
//! configuration (32 shards, explicit publication only, no tier merge): the
//! incumbent, a second incumbent (the A/A null), and the candidate. Every build
//! commit ingests the same documents into all three and adds one small segment
//! per active shard, committing them in a rotating order. At each requested
//! segment count a run of small single-shard commits is timed, again in a
//! rotating order, and paired per round (`null/incumbent`,
//! `candidate/incumbent`); the per-segment slope between scales separates the
//! growth from the fixed cost. After the run all three must agree exactly on
//! every published segment (docid range, document count, seal sequence, file
//! length, tombstones, every section's bytes), the MANIFEST generation,
//! watermark, flags and field statistics, the live document count, and every
//! probe query's hits and exact count before any timing is printed.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 rch exec -- env CARGO_PROFILE_RELEASE_LTO=false \
//!   CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!     --profile release --bench publish_commit_scale_ab
//! ```
//!
//! `QUILL_PUBLISH_PROFILE_SECS=n` records `perf` (Linux, `sudo -n`) over small
//! commits on arm `QUILL_PUBLISH_PROFILE_ARM` at the largest scale and prints
//! the report.

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::PathBuf;
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

fn status_kib(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(field))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(0)
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

/// Every published segment in MANIFEST order: docid range, document count,
/// seal sequence, file length, tombstones and the exact bytes of every
/// section, plus the MANIFEST's generation, watermark, flags and field
/// statistics. Segment identifiers (and so the file witnesses over headers
/// that carry them) and publish times are wall-clock derived and not compared.
fn assert_same_publication(arm: &str, incumbent: &KeeperSnapshot, other: &KeeperSnapshot) {
    let (left, right) = (
        &incumbent.loaded_manifest().manifest,
        &other.loaded_manifest().manifest,
    );
    assert_eq!(
        (left.generation, left.docid_high_watermark, left.flags),
        (right.generation, right.docid_high_watermark, right.flags),
        "{arm}: MANIFEST generation, watermark and flags"
    );
    assert_eq!(
        left.field_stats, right.field_stats,
        "{arm}: field statistics"
    );
    let (left, right) = (incumbent.segments(), other.segments());
    assert_eq!(left.len(), right.len(), "{arm}: segment count");
    for (position, (l, r)) in left.iter().zip(right).enumerate() {
        let (lm, rm) = (l.manifest(), r.manifest());
        assert_eq!(
            (
                lm.docid_lo,
                lm.docid_hi,
                lm.doc_count,
                lm.seal_seq,
                lm.file_len,
                &lm.tombstones
            ),
            (
                rm.docid_lo,
                rm.docid_hi,
                rm.doc_count,
                rm.seal_seq,
                rm.file_len,
                &rm.tombstones
            ),
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
            assert!(
                l.source_bytes()[range(le)] == r.source_bytes()[range(re)],
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

fn env_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .collect()
}

/// Start `perf record` on this process.
fn profile_start(label: &str) -> (std::process::Child, String) {
    let path = format!("/dev/shm/quill-publish-p14-{label}.perf");
    let pid = std::process::id().to_string();
    let perf = std::process::Command::new("sudo")
        .args([
            "-n",
            "perf",
            "record",
            "-F",
            "999",
            "--call-graph",
            "dwarf,32768",
            "-p",
            &pid,
            "-o",
            &path,
        ])
        .spawn()
        .expect("spawn perf record");
    std::thread::sleep(std::time::Duration::from_millis(1_500));
    eprintln!("QUILL_PUBLISH_PROFILE_START {label}");
    (perf, path)
}

/// Stop `perf record` and print flat and inclusive symbol reports.
fn profile_finish(label: &str, (mut perf, path): (std::process::Child, String)) {
    let _ = std::process::Command::new("sudo")
        .args(["-n", "kill", "-INT", &perf.id().to_string()])
        .status();
    let _ = perf.wait();
    for children in ["--no-children", "--children"] {
        let report = std::process::Command::new("sudo")
            .args([
                "-n",
                "perf",
                "report",
                "-i",
                &path,
                "--stdio",
                children,
                "-g",
                "none",
                "--sort",
                "symbol",
                "--percent-limit",
                "0.4",
            ])
            .output()
            .expect("perf report");
        let text = String::from_utf8_lossy(&report.stdout);
        eprintln!("=== perf report {label} {children}");
        for line in text
            .lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        {
            let trimmed = line.trim_end();
            eprintln!("{}", &trimmed[..trimmed.len().min(240)]);
        }
    }
}

/// The arms in round `round`'s commit order.
fn rotation(round: usize) -> impl Iterator<Item = usize> {
    (0..ARMS.len()).map(move |offset| (round + offset) % ARMS.len())
}

/// One scale's timed small commits: `commit_ms[arm][round]`.
struct ScaleRun {
    scale: usize,
    segments_after: usize,
    commit_ms: Vec<Vec<f64>>,
    rss_kib: u64,
    hwm_kib: u64,
}

#[allow(clippy::too_many_lines)]
fn main() {
    let _ = print_bench_elf_sha256();
    let scales = env_list("QUILL_PUBLISH_SCALES", "500,1000,2000,4000");
    let shards = env_usize("QUILL_PUBLISH_SHARDS", 32);
    let per_shard = env_usize("QUILL_PUBLISH_DOCS_PER_SHARD", 64);
    let timed = env_usize("QUILL_PUBLISH_TIMED", 21);
    let small = env_usize("QUILL_PUBLISH_SMALL_DOCS", 16);
    let profile_secs = env_usize("QUILL_PUBLISH_PROFILE_SECS", 0);
    let profile_arm = env_usize("QUILL_PUBLISH_PROFILE_ARM", 2).min(ARMS.len() - 1);
    let per_commit = per_shard * shards;
    let root = tempfile::tempdir().expect("bench root");
    let directories: Vec<PathBuf> = ARMS.iter().map(|arm| root.path().join(arm)).collect();
    eprintln!(
        "[harness] scales={scales:?} arms={ARMS:?} shards={shards} build_docs_per_commit={per_commit} \
         timed={timed} small_docs={small} available_parallelism={} root={}",
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
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
                    .bench_set_linear_segment_reuse(&cx, *arm != "candidate")
                    .await
                    .expect("select the segment-reuse arm");
                indexes.push(index);
            }
            let mut next_doc = 0_usize;
            let mut segments = 0_usize;
            let mut build_commits = 0_usize;
            let build_started = Instant::now();
            let mut build_commit_ms = vec![0.0_f64; ARMS.len()];
            let mut runs = Vec::new();
            let mut counts = Vec::new();
            for &scale in &scales {
                while segments < scale {
                    let documents = corpus(next_doc, per_commit);
                    next_doc += per_commit;
                    for index in &indexes {
                        index
                            .index_schema_documents(&cx, black_box(&documents))
                            .await
                            .expect("build ingest");
                    }
                    counts.clear();
                    for arm in rotation(build_commits) {
                        let started = Instant::now();
                        let snapshot = indexes[arm].commit(&cx).await.expect("build commit");
                        build_commit_ms[arm] += started.elapsed().as_secs_f64() * 1e3;
                        counts.push(snapshot.segments().len());
                    }
                    assert!(
                        counts.windows(2).all(|pair| pair[0] == pair[1]),
                        "build commit {build_commits}: segment counts {counts:?}"
                    );
                    segments = counts[0];
                    build_commits += 1;
                }
                eprintln!(
                    "[progress] build scale={scale} segments={segments} docs={next_doc} \
                     commits={build_commits} wall_s={:.1} build_commit_ms_sum={build_commit_ms:.0?}",
                    build_started.elapsed().as_secs_f64(),
                );
                let mut commit_ms = vec![Vec::with_capacity(timed); ARMS.len()];
                for round in 0..timed {
                    let documents = corpus(next_doc, small);
                    next_doc += small;
                    for index in &indexes {
                        index
                            .index_schema_documents(&cx, black_box(&documents))
                            .await
                            .expect("small ingest");
                    }
                    counts.clear();
                    for arm in rotation(round) {
                        let started = Instant::now();
                        let snapshot = indexes[arm].commit(&cx).await.expect("small commit");
                        commit_ms[arm].push(started.elapsed().as_secs_f64() * 1e3);
                        counts.push(snapshot.segments().len());
                    }
                    assert!(
                        counts.windows(2).all(|pair| pair[0] == pair[1]),
                        "scale {scale} round {round}: segment counts {counts:?}"
                    );
                    segments = counts[0];
                }
                let (incumbent_p50, _, _) = spread(&commit_ms[0]);
                let (candidate_p50, _, _) = spread(&commit_ms[2]);
                eprintln!(
                    "[progress] timed scale={scale} segments_after={segments} \
                     incumbent_p50_ms={incumbent_p50:.3} candidate_p50_ms={candidate_p50:.3}"
                );
                runs.push(ScaleRun {
                    scale,
                    segments_after: segments,
                    commit_ms,
                    rss_kib: status_kib("VmRSS:"),
                    hwm_kib: status_kib("VmHWM:"),
                });
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
                assert_same_publication(arm, &incumbent, &other);
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
                "[parity] arms={} segments={} generation={} docs={} probes={} identical",
                ARMS.len(),
                incumbent.segments().len(),
                incumbent.loaded_manifest().manifest.generation,
                indexes[0].doc_count().expect("docs"),
                PROBE_QUERIES.len()
            );
            drop(incumbent);

            let mut previous: Option<(usize, [f64; 3])> = None;
            for run in &runs {
                let ratio = |arm: usize| -> Vec<f64> {
                    run.commit_ms[arm]
                        .iter()
                        .zip(&run.commit_ms[0])
                        .map(|(value, incumbent)| value / incumbent)
                        .collect()
                };
                let (null, null_lo, null_hi) = spread(&ratio(1));
                let (lever, lever_lo, lever_hi) = spread(&ratio(2));
                let mut p50 = [0.0_f64; 3];
                let mut line = format!(
                    "[small] scale={} segments_after={} rounds={} null={null:.4} \
                     [p5 {null_lo:.4}, p95 {null_hi:.4}] lever(candidate/incumbent)={lever:.4} \
                     [p5 {lever_lo:.4}, p95 {lever_hi:.4}]",
                    run.scale,
                    run.segments_after,
                    run.commit_ms[0].len(),
                );
                for (arm, name) in ARMS.iter().enumerate() {
                    let (median, lo, hi) = spread(&run.commit_ms[arm]);
                    p50[arm] = median;
                    let _ = write!(line, " {name}_ms p50={median:.3} [{lo:.3}, {hi:.3}]");
                }
                let _ = write!(line, " rss_kib={} hwm_kib={}", run.rss_kib, run.hwm_kib);
                eprintln!("{line}");
                if let Some((segments_before, p50_before)) = previous {
                    let added = (run.segments_after - segments_before) as f64;
                    let mut slope = format!(
                        "[slope] segments={segments_before}->{} us_per_segment",
                        run.segments_after
                    );
                    for (arm, name) in ARMS.iter().enumerate() {
                        let _ = write!(
                            slope,
                            " {name}={:.3}",
                            (p50[arm] - p50_before[arm]) * 1e3 / added
                        );
                    }
                    eprintln!("{slope}");
                }
                previous = Some((run.segments_after, p50));
            }
            eprintln!("[build] commits={build_commits} build_commit_ms_sum={build_commit_ms:.0?}");

            if profile_secs > 0 {
                let index = &indexes[profile_arm];
                let label = format!("small_commit_{}", ARMS[profile_arm]);
                let recording = profile_start(&label);
                let started = Instant::now();
                let until = std::time::Duration::from_secs(profile_secs as u64);
                let mut passes = 0_u32;
                while started.elapsed() < until {
                    let documents = corpus(next_doc, small);
                    next_doc += small;
                    index
                        .index_schema_documents(&cx, black_box(&documents))
                        .await
                        .expect("profile ingest");
                    index.commit(&cx).await.expect("profile commit");
                    passes += 1;
                }
                eprintln!(
                    "QUILL_PUBLISH_PROFILE_END passes={passes} per_pass_ms={:.3} segments={}",
                    started.elapsed().as_secs_f64() * 1e3 / f64::from(passes.max(1)),
                    index.snapshot().expect("snapshot").segments().len()
                );
                profile_finish(&label, recording);
            }
            drop(indexes);
        });
    });
}
