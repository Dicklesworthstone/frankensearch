#![recursion_limit = "256"]
//! End-of-rebuild fold A/B: one `concat_merge` per run, one run after another
//! (incumbent, what CASS's `force_merge_with_output_cap` does), against
//! `concat_merge_runs`, which builds several disjoint runs' merged segments at
//! once and publishes them in the same order (candidate).
//!
//! A durable CASS-schema index is built once through the shipping
//! `index_schema_documents` + `commit` route with CASS's writer configuration
//! (explicit publication only, no tier merge), so it holds hundreds of
//! segments. The ingest and commit wall times of that build are reported next
//! to the serial fold's, which is the share each phase takes. The fold is
//! planned with a copy of CASS's `plan_capped_merge_runs` (consecutive runs
//! under a byte cap and the posting-limit document cap) at each requested cap.
//!
//! Every timed fold runs on its own fresh file-by-file copy of that index, so
//! every arm starts from the same bytes. Arms run in a rotating order within
//! each round: the incumbent, a second incumbent (the A/A null), then each
//! candidate concurrency. Ratios are paired per round (`arm / incumbent`).
//! Output segment ids and creation times are fixed per run, so every arm must
//! publish the same bytes: after each fold the final MANIFEST (every field but
//! the wall-clock publish time, so generation, seal sequences, docid ranges,
//! tombstones and file hashes included), the xxh3 of every segment file's full
//! bytes, the live document count and every probe query's hits (identifier,
//! global docid, score bits) and exact count are checked against the first
//! serial fold before the round counts. Peak anonymous memory during each fold
//! is sampled from `/proc/self/status` every 10 ms, and `VmHWM` is reset
//! before each fold through `/proc/self/clear_refs`.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 rch exec -- env QUILL_FOLD_ROUNDS=10 \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!     --profile release --bench concat_merge_runs_ab
//! ```

use std::fmt::Write as _;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::bench_support::print_bench_elf_sha256;
use frankensearch_quill::cass::CassDocument;
use frankensearch_quill::query::{CassQueryFilters, CassQueryParser};
use frankensearch_quill::quiver::DEFAULT_MAX_POSTINGS_PER_TERM;
use frankensearch_quill::schema::CASS_SEMANTIC_SCHEMA;
use frankensearch_quill::{
    ConcatMergeRun, KeeperSnapshot, Manifest, QuillConfig, QuillIndex, SchemaDocument,
};
use xxhash_rust::xxh3::xxh3_64;

const AGENTS: [&str; 3] = ["claude_code", "codex", "gemini"];
const MESSAGES_PER_CONVERSATION: usize = 40;
/// One document in this many carries the probe token.
const NEEDLE_EVERY: usize = 97;
const PROBE_QUERIES: [&str; 4] = ["quillneedle", "w7", "w11 OR w40123", "\"w3 w5\""];
/// CASS batches bulk-rebuild documents in pages of about this many per shard.
const BATCH_DOCS_PER_SHARD: usize = 512;
/// Output segment ids are `OUTPUT_ID_BASE + run`; creation times likewise.
const OUTPUT_ID_BASE: u64 = 0xf01d_0000_0000_0000;
const CREATED_UNIX_S_BASE: i64 = 1_700_000_000;

/// CASS `FOLD_HULL_BYTES_PER_DOCID` (`src/search/quill_bridge.rs`).
const FOLD_HULL_BYTES_PER_DOCID: u64 = 128;
/// CASS `MAX_FOLD_OUTPUT_DOCS`.
const MAX_FOLD_OUTPUT_DOCS: u64 = DEFAULT_MAX_POSTINGS_PER_TERM as u64;

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

fn config(shards: usize, budget_bytes: usize) -> QuillConfig {
    // CASS's writer configuration (`cass_quill_config`): explicit publication
    // only and no engine tier merge.
    QuillConfig {
        max_visibility_lag_ms: u64::MAX,
        tier_fanout: usize::MAX,
        max_ingest_shards: shards,
        scribe_shard_budget_bytes: budget_bytes,
        ..QuillConfig::default()
    }
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

/// Reset `VmHWM` to the current resident set (Linux `clear_refs` value 5).
fn reset_peak_rss() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

/// Samples `RssAnon` every 10 ms until stopped and keeps the peak.
struct AnonSampler {
    stop: Arc<AtomicBool>,
    peak_kib: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl AnonSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak_kib = Arc::new(AtomicU64::new(status_kib("RssAnon:")));
        let handle = {
            let stop = Arc::clone(&stop);
            let peak_kib = Arc::clone(&peak_kib);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    peak_kib.fetch_max(status_kib("RssAnon:"), Ordering::Relaxed);
                    std::thread::sleep(Duration::from_millis(10));
                }
            })
        };
        Self {
            stop,
            peak_kib,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().expect("RSS sampler");
        }
        self.peak_kib
            .load(Ordering::Relaxed)
            .max(status_kib("RssAnon:"))
    }
}

/// One published segment as CASS's fold planner sees it.
#[derive(Clone, Copy)]
struct FoldProfile {
    segment_id: u64,
    file_len: u64,
    docid_lo: u64,
    docid_hi: u64,
    doc_count: u64,
}

fn fold_profile(snapshot: &KeeperSnapshot) -> Vec<FoldProfile> {
    snapshot
        .loaded_manifest()
        .manifest
        .segments
        .iter()
        .map(|segment| FoldProfile {
            segment_id: segment.segment_id,
            file_len: segment.file_len,
            docid_lo: segment.docid_lo,
            docid_hi: segment.docid_hi,
            doc_count: u64::from(segment.doc_count),
        })
        .collect()
}

fn fold_fits(segment: FoldProfile, max_output_bytes: u64) -> bool {
    segment.file_len.saturating_add(
        segment
            .docid_hi
            .saturating_sub(segment.docid_lo)
            .saturating_mul(FOLD_HULL_BYTES_PER_DOCID),
    ) <= max_output_bytes
        && segment.doc_count <= MAX_FOLD_OUTPUT_DOCS
}

/// CASS `plan_capped_merge_runs`, line for line.
fn plan_capped_merge_runs(profile: &[FoldProfile], max_output_bytes: u64) -> Vec<Vec<u64>> {
    let mut runs: Vec<Vec<u64>> = Vec::new();
    let mut current: Vec<FoldProfile> = Vec::new();
    let mut current_bytes = 0_u64;
    let mut current_docs = 0_u64;
    let mut current_lo = 0_u64;
    let mut current_hi = 0_u64;
    let close = |current: &mut Vec<FoldProfile>, runs: &mut Vec<Vec<u64>>| {
        if current.len() >= 2 {
            runs.push(current.iter().map(|segment| segment.segment_id).collect());
        }
        current.clear();
    };
    for &segment in profile {
        if !current.is_empty() {
            let next_bytes = current_bytes.saturating_add(segment.file_len);
            let next_docs = current_docs.saturating_add(segment.doc_count);
            let next_lo = current_lo.min(segment.docid_lo);
            let next_hi = current_hi.max(segment.docid_hi);
            let next_estimate = next_bytes.saturating_add(
                next_hi
                    .saturating_sub(next_lo)
                    .saturating_mul(FOLD_HULL_BYTES_PER_DOCID),
            );
            if next_estimate > max_output_bytes || next_docs > MAX_FOLD_OUTPUT_DOCS {
                close(&mut current, &mut runs);
            } else {
                current.push(segment);
                current_bytes = next_bytes;
                current_docs = next_docs;
                current_lo = next_lo;
                current_hi = next_hi;
                continue;
            }
        }
        if !fold_fits(segment, max_output_bytes) {
            continue;
        }
        current.push(segment);
        current_bytes = segment.file_len;
        current_docs = segment.doc_count;
        current_lo = segment.docid_lo;
        current_hi = segment.docid_hi;
    }
    close(&mut current, &mut runs);
    runs
}

/// Copy every file of a closed index directory into a fresh directory.
fn copy_index(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("copy target");
    for entry in std::fs::read_dir(from).expect("read index directory") {
        let entry = entry.expect("index entry");
        let kind = entry.file_type().expect("entry type");
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            copy_index(&entry.path(), &target);
        } else if kind.is_file() {
            std::fs::copy(entry.path(), &target).expect("copy index file");
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

/// Everything a reader of the folded index can observe.
#[derive(Debug, PartialEq, Eq)]
struct FoldState {
    /// The published MANIFEST with its wall-clock publish time cleared.
    manifest: Manifest,
    /// xxh3 of each segment file's full bytes, in MANIFEST order.
    segment_xxh3: Vec<u64>,
    live_docs: u64,
    probes: Vec<ProbeResult>,
}

fn fold_state(cx: &Cx, index: &QuillIndex, parser: &CassQueryParser) -> FoldState {
    let snapshot = index.snapshot().expect("folded snapshot");
    let mut manifest = snapshot.loaded_manifest().manifest.clone();
    manifest.last_publish_unix_s = 0;
    FoldState {
        manifest,
        segment_xxh3: snapshot
            .segments()
            .iter()
            .map(|segment| xxh3_64(segment.source_bytes()))
            .collect(),
        live_docs: index.doc_count().expect("live documents"),
        probes: probe(cx, index, parser),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Incumbent,
    Null,
    /// `concat_merge_runs` building this many runs at once.
    Candidate(NonZeroUsize),
}

impl Arm {
    fn label(self) -> String {
        match self {
            Self::Incumbent => "incumbent".to_owned(),
            Self::Null => "null".to_owned(),
            Self::Candidate(concurrency) => format!("candidate_c{concurrency}"),
        }
    }
}

/// Run `offset` of the plan, with its fixed output id and creation time.
fn concat_run(offset: usize, sources: &[u64]) -> ConcatMergeRun {
    ConcatMergeRun {
        source_segment_ids: sources.to_vec(),
        output_segment_id: OUTPUT_ID_BASE + u64::try_from(offset).expect("run offset"),
        created_unix_s: CREATED_UNIX_S_BASE + i64::try_from(offset).expect("run offset"),
    }
}

/// What CASS's `force_merge_with_output_cap` does: one `concat_merge` per run.
async fn fold_serial(cx: &Cx, index: &QuillIndex, runs: &[ConcatMergeRun]) {
    for run in runs {
        index
            .concat_merge(
                cx,
                black_box(&run.source_segment_ids),
                run.output_segment_id,
                run.created_unix_s,
            )
            .await
            .expect("serial concat merge");
    }
}

struct Measured {
    seconds: f64,
    peak_anon_growth_mib: f64,
    vm_hwm_mib: f64,
}

/// Copy the template, open it, fold it with `arm` (timed), and check the result.
fn run_arm(
    cx: &Cx,
    template: &Path,
    scratch: &Path,
    arm: Arm,
    runs: &[ConcatMergeRun],
    shards: usize,
    budget_bytes: usize,
    parser: &CassQueryParser,
    expected: Option<&FoldState>,
) -> (Measured, FoldState) {
    let workdir = tempfile::tempdir_in(scratch).expect("arm directory");
    let directory = workdir.path().join("index");
    copy_index(template, &directory);
    drive(async {
        let index = QuillIndex::open_with_schema(
            cx,
            directory.clone(),
            CASS_SEMANTIC_SCHEMA,
            config(shards, budget_bytes),
        )
        .await
        .expect("open the copied index");
        let anon0 = status_kib("RssAnon:");
        reset_peak_rss();
        let sampler = AnonSampler::start();
        let started = Instant::now();
        match arm {
            Arm::Incumbent | Arm::Null => fold_serial(cx, &index, runs).await,
            Arm::Candidate(concurrency) => {
                index
                    .concat_merge_runs(cx, black_box(runs), concurrency)
                    .await
                    .expect("concurrent concat merge runs");
            }
        }
        let seconds = started.elapsed().as_secs_f64();
        let peak_anon = sampler.finish();
        let vm_hwm = status_kib("VmHWM:");
        let state = fold_state(cx, &index, parser);
        if let Some(expected) = expected {
            assert!(
                &state == expected,
                "{}: folded index differs from the first serial fold",
                arm.label()
            );
        }
        (
            Measured {
                seconds,
                peak_anon_growth_mib: peak_anon.saturating_sub(anon0) as f64 / 1024.0,
                vm_hwm_mib: vm_hwm as f64 / 1024.0,
            },
            state,
        )
    })
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

/// Build the pre-fold index through the shipping ingest route; returns
/// (ingest seconds, commit seconds).
fn build_template(
    cx: &Cx,
    directory: &Path,
    shards: usize,
    budget_bytes: usize,
    batches: usize,
    per_shard: usize,
) -> (f64, f64) {
    drive(async {
        let index = QuillIndex::create_with_schema(
            cx,
            directory.to_path_buf(),
            CASS_SEMANTIC_SCHEMA,
            config(shards, budget_bytes),
        )
        .await
        .expect("durable CASS index");
        let per_commit = per_shard * shards;
        let batch = BATCH_DOCS_PER_SHARD * shards;
        let (mut ingest_s, mut commit_s) = (0.0, 0.0);
        for commit in 0..batches {
            let documents = corpus(commit * per_commit, per_commit);
            let started = Instant::now();
            for chunk in documents.chunks(batch) {
                index
                    .index_schema_documents(cx, black_box(chunk))
                    .await
                    .expect("ingest");
            }
            ingest_s += started.elapsed().as_secs_f64();
            let started = Instant::now();
            index.commit(cx).await.expect("commit");
            commit_s += started.elapsed().as_secs_f64();
        }
        (ingest_s, commit_s)
    })
}

fn main() {
    let _ = print_bench_elf_sha256();
    let shards = env_usize("QUILL_FOLD_SHARDS", 16);
    let batches = env_usize("QUILL_FOLD_COMMITS", 16);
    let per_shard = env_usize("QUILL_FOLD_DOCS_PER_SHARD", 2_048);
    let budget_bytes = env_usize("QUILL_FOLD_BUDGET_MIB", 64) << 20;
    let rounds = env_usize("QUILL_FOLD_ROUNDS", 10);
    let caps_mib = env_list("QUILL_FOLD_CAP_MIB", "16,64");
    let concurrencies = env_list("QUILL_FOLD_CONCURRENCY", "2,4,8");
    let scratch = tempfile::tempdir().expect("bench root");
    let template = scratch.path().join("template");
    eprintln!(
        "[harness] shards={shards} commits={batches} docs_per_shard_per_commit={per_shard} \
         budget_mib={} rounds={rounds} caps_mib={caps_mib:?} concurrency={concurrencies:?} \
         available_parallelism={} root={}",
        budget_bytes >> 20,
        std::thread::available_parallelism().map_or(0, usize::from),
        scratch.path().display(),
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(shards)
        .build()
        .expect("bench rayon pool");
    pool.install(|| {
        let cx = Cx::for_testing();
        let parser = CassQueryParser::new(CASS_SEMANTIC_SCHEMA).expect("CASS parser");
        let (ingest_s, commit_s) =
            build_template(&cx, &template, shards, budget_bytes, batches, per_shard);
        let profile = drive(async {
            let index = QuillIndex::open_with_schema(
                &cx,
                template.clone(),
                CASS_SEMANTIC_SCHEMA,
                config(shards, budget_bytes),
            )
            .await
            .expect("open the template");
            let snapshot = index.snapshot().expect("template snapshot");
            fold_profile(&snapshot)
        });
        let index_bytes: u64 = profile.iter().map(|segment| segment.file_len).sum();
        let docs: u64 = profile.iter().map(|segment| segment.doc_count).sum();
        assert!(
            !profile.iter().any(|segment| {
                (OUTPUT_ID_BASE..OUTPUT_ID_BASE + (1 << 32)).contains(&segment.segment_id)
            }),
            "template segment ids must not collide with the fixed output ids"
        );
        eprintln!(
            "[template] docs={docs} segments={} index_mib={:.1} ingest_s={ingest_s:.2} \
             commit_s={commit_s:.2}",
            profile.len(),
            index_bytes as f64 / f64::from(1 << 20),
        );

        for cap_mib in &caps_mib {
            let cap = u64::try_from(*cap_mib).expect("cap") << 20;
            let runs = plan_capped_merge_runs(&profile, cap);
            let mut lengths: Vec<usize> = runs.iter().map(Vec::len).collect();
            lengths.sort_unstable();
            eprintln!(
                "[plan] cap_mib={cap_mib} runs={} sources={} segments_per_run min={} p50={} max={}",
                runs.len(),
                lengths.iter().sum::<usize>(),
                lengths.first().copied().unwrap_or(0),
                lengths.get(lengths.len() / 2).copied().unwrap_or(0),
                lengths.last().copied().unwrap_or(0),
            );
            if runs.is_empty() {
                continue;
            }
            let runs: Vec<ConcatMergeRun> = runs
                .iter()
                .enumerate()
                .map(|(offset, sources)| concat_run(offset, sources))
                .collect();
            let mut arms = vec![Arm::Incumbent, Arm::Null];
            arms.extend(
                concurrencies
                    .iter()
                    .filter_map(|&concurrency| NonZeroUsize::new(concurrency))
                    .map(Arm::Candidate),
            );
            // Reference fold (untimed parity anchor).
            let (_, expected) = run_arm(
                &cx,
                &template,
                scratch.path(),
                Arm::Incumbent,
                &runs,
                shards,
                budget_bytes,
                &parser,
                None,
            );
            assert!(
                expected.probes.iter().all(|(hits, _)| !hits.is_empty()),
                "every probe query must match"
            );
            eprintln!(
                "[parity] cap_mib={cap_mib} segments_after={} generation={} live_docs={}",
                expected.manifest.segments.len(),
                expected.manifest.generation,
                expected.live_docs,
            );
            let mut measured: Vec<Vec<Measured>> = arms.iter().map(|_| Vec::new()).collect();
            for round in 0..rounds {
                for offset in 0..arms.len() {
                    let slot = (round + offset) % arms.len();
                    let (sample, _) = run_arm(
                        &cx,
                        &template,
                        scratch.path(),
                        arms[slot],
                        &runs,
                        shards,
                        budget_bytes,
                        &parser,
                        Some(&expected),
                    );
                    measured[slot].push(sample);
                }
            }
            let seconds = |slot: usize| -> Vec<f64> {
                measured[slot].iter().map(|sample| sample.seconds).collect()
            };
            let incumbent = seconds(0);
            let (fold_p50, _, _) = spread(&incumbent);
            eprintln!(
                "[share] cap_mib={cap_mib} ingest_s={ingest_s:.2} commit_s={commit_s:.2} \
                 serial_fold_s_p50={fold_p50:.2} fold_share={:.3}",
                fold_p50 / (ingest_s + commit_s + fold_p50),
            );
            for (slot, arm) in arms.iter().enumerate() {
                let ratios: Vec<f64> = seconds(slot)
                    .iter()
                    .zip(&incumbent)
                    .map(|(arm_s, incumbent_s)| arm_s / incumbent_s)
                    .collect();
                let (ratio, ratio_lo, ratio_hi) = spread(&ratios);
                let (secs, secs_lo, secs_hi) = spread(&seconds(slot));
                let anon: Vec<f64> = measured[slot]
                    .iter()
                    .map(|sample| sample.peak_anon_growth_mib)
                    .collect();
                let hwm: Vec<f64> = measured[slot]
                    .iter()
                    .map(|sample| sample.vm_hwm_mib)
                    .collect();
                let (anon_p50, _, anon_hi) = spread(&anon);
                let (hwm_p50, _, hwm_hi) = spread(&hwm);
                eprintln!(
                    "[fold] cap_mib={cap_mib} arm={} rounds={rounds} ratio(arm/incumbent)={ratio:.4} \
                     [p5 {ratio_lo:.4}, p95 {ratio_hi:.4}] fold_s p50={secs:.3} [p5 {secs_lo:.3}, \
                     p95 {secs_hi:.3}] peak_anon_growth_mib p50={anon_p50:.1} max={anon_hi:.1} \
                     vm_hwm_mib p50={hwm_p50:.1} max={hwm_hi:.1}",
                    arm.label(),
                );
            }
        }
    });
}
