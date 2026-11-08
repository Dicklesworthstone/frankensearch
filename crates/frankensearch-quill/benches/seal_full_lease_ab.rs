#![recursion_limit = "256"]
//! Full-lease CASS seal A/B: BLOCKMAX fieldnorm ids from per-document columns
//! (shipping) vs `fieldnorm_to_id` binary-searched at every posting
//! (incumbent, `flush_accumulator_with_per_posting_fieldnorms`).
//!
//! One shard holds a full 65,536-docid lease of a deterministic CASS-shaped
//! corpus, accumulated once through the shipping `CassFieldValues` route.
//! The corpus is calibrated against real CASS full-lease segments: about 22M
//! token rows and 300k terms, sealing to about 90 MB, with POSTINGS,
//! POSITIONS, BLOCKMAX, TERMDICT, IDMAP and STOREDMETA in the observed
//! proportions. Both arms seal the same accumulator with the same input.
//! Before timing, the incumbent and candidate segments must be byte-identical
//! (and so share the file xxh3) under the scalar and the automatic radix, and
//! must reopen and verify; every timed seal must reproduce that file xxh3.
//!
//! Two cells, each with an A/A incumbent-vs-incumbent null and the lever
//! ratio `candidate / incumbent` (`< 1.0` outside the null band wins), both
//! from paired rounds whose order alternates:
//!
//! * `scalar_1thread`: one seal on a one-thread pool, `FlushMode::Scalar`
//!   (the lone-seal path).
//! * `concurrent_k`: `k` seals of the shard at once on a `k`-thread pool,
//!   `FlushMode::Automatic`, as a commit seals its shards in parallel.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
//!     QUILL_SEAL_ROUNDS=15 QUILL_SEAL_CONCURRENT=8 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench seal_full_lease_ab
//!
//! # Fast parity/harness check (not performance evidence):
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_SEAL_DOCS=4096 QUILL_SEAL_ROUNDS=10 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench seal_full_lease_ab
//! ```

use std::cell::RefCell;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio, print_bench_elf_sha256};
use frankensearch_quill::cass::{CassDerivedColumns, CassDocument, CassFieldValues};
use frankensearch_quill::schema::CASS_SEMANTIC_SCHEMA;
use frankensearch_quill::scribe::{
    ColumnarAccumulator, DOC_ORDS_PER_LEASE, FlushDocumentInput, FlushMode, FlushSegmentInput,
    flush_accumulator_with_mode, flush_accumulator_with_per_posting_fieldnorms,
};
use frankensearch_quill::segment::SegmentReader;
use frankensearch_quill::{CURRENT_ENGINE_VERSION, EncodedSegment};
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_64;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

const SYLLABLES: [&str; 64] = [
    "a", "e", "i", "o", "u", "ka", "ke", "ki", "ko", "ku", "ra", "re", "ri", "ro", "ru", "ta",
    "te", "ti", "to", "tu", "na", "ne", "ni", "no", "nu", "sa", "se", "si", "so", "su", "ma", "me",
    "mi", "mo", "mu", "la", "le", "li", "lo", "lu", "pa", "pe", "pi", "po", "pu", "da", "de", "di",
    "do", "du", "ga", "ge", "gi", "go", "gu", "ba", "be", "bi", "bo", "bu", "st", "nd", "th", "ch",
];

/// A pronounceable word for vocabulary rank `rank`: frequent ranks are short,
/// and neighbouring ranks share prefixes the way inflected words do.
fn word(rank: u64, out: &mut String) {
    let mut value = rank + 5;
    let mut parts = [0_usize; 8];
    let mut count = 0;
    while value > 0 && count < parts.len() {
        parts[count] = usize::try_from(value % 64).expect("syllable");
        value /= 64;
        count += 1;
    }
    for part in parts[..count].iter().rev() {
        out.push_str(SYLLABLES[*part]);
    }
}

/// Zipf(s) sampler over `n` ranks by inverse CDF.
struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    fn new(n: usize, s: f64) -> Self {
        let mut cdf = Vec::with_capacity(n);
        let mut total = 0.0;
        for rank in 1..=n {
            total += 1.0 / (rank as f64).powf(s);
            cdf.push(total);
        }
        for value in &mut cdf {
            *value /= total;
        }
        Self { cdf }
    }

    fn sample(&self, rng: &mut Rng) -> u64 {
        let target = rng.unit();
        let index = self.cdf.partition_point(|&value| value < target);
        u64::try_from(index.min(self.cdf.len() - 1)).expect("rank")
    }
}

const AGENTS: [&str; 5] = ["claude_code", "codex", "gemini", "cursor", "aider"];
const SEPARATORS: [&str; 16] = [
    " ", " ", " ", " ", " ", " ", " ", " ", "\n", ", ", ". ", " = ", "::", "(", "); ", " `",
];
/// Zipf vocabulary size and exponent.
const VOCABULARY: usize = 40_000;
const ZIPF_S: f64 = 1.0;
/// Log-normal message length: median words and sigma.
const MEDIAN_WORDS: f64 = 50.0;
const SIGMA: f64 = 1.2;
/// Per-mille of tokens that are hex hashes (mostly unique), numbers below
/// 10,000, and `a-b` / `a_b` compounds.
const HEX_PERMILLE: u64 = 2;
const NUMBER_PERMILLE: u64 = 30;
const COMPOUND_PERMILLE: u64 = 40;

/// Deterministic CASS-shaped messages: a Zipf vocabulary of pronounceable
/// words, plus code-like tokens (hyphenated and `snake_case` compounds,
/// numbers, hex hashes that are mostly unique), log-normal message lengths
/// (heavy tail capped at 3,000 words), 1-120 messages per conversation, and
/// Claude-Code-like source paths and workspaces.
fn corpus(documents: usize, seed: u64) -> Vec<CassDocument> {
    let zipf = Zipf::new(VOCABULARY, ZIPF_S);
    let mut rng = Rng(seed | 1);
    let mut out = Vec::with_capacity(documents);
    let mut conversation = 0_u64;
    let mut remaining_in_conversation = 0_u64;
    let mut msg_idx = 0_u64;
    let mut session = String::new();
    while out.len() < documents {
        if remaining_in_conversation == 0 {
            conversation += 1;
            remaining_in_conversation = 1 + rng.below(120);
            msg_idx = 0;
            session.clear();
            let _ = write!(
                session,
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                rng.next() >> 32,
                rng.next() >> 48,
                rng.next() >> 48,
                rng.next() >> 48,
                rng.next() >> 16
            );
        }
        remaining_in_conversation -= 1;
        // Box-Muller normal -> log-normal word count.
        let u1 = rng.unit().max(1e-12);
        let u2 = rng.unit();
        let normal = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let words = (MEDIAN_WORDS * (SIGMA * normal).exp()).clamp(1.0, 3_000.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let words = words as usize;
        let mut content = String::with_capacity(words * 7);
        for position in 0..words {
            if position != 0 {
                // Prose and code punctuation between tokens.
                content.push_str(SEPARATORS[usize::try_from(rng.below(16)).expect("separator")]);
            }
            let roll = rng.below(1_000);
            if roll < HEX_PERMILLE {
                let _ = write!(content, "{:x}", rng.next() >> (16 + rng.below(4) * 8));
            } else if roll < HEX_PERMILLE + NUMBER_PERMILLE {
                let _ = write!(
                    content,
                    "{}",
                    rng.below(if roll % 2 == 0 { 100 } else { 10_000 })
                );
            } else if roll < HEX_PERMILLE + NUMBER_PERMILLE + COMPOUND_PERMILLE {
                word(zipf.sample(&mut rng), &mut content);
                content.push(if roll % 2 == 0 { '-' } else { '_' });
                word(zipf.sample(&mut rng), &mut content);
            } else {
                word(zipf.sample(&mut rng), &mut content);
            }
        }
        let title = (msg_idx == 0).then(|| {
            let mut title = String::new();
            for index in 0..(2 + rng.below(8)) {
                if index != 0 {
                    title.push(' ');
                }
                word(zipf.sample(&mut rng), &mut title);
            }
            title
        });
        let ordinal = u64::try_from(out.len()).expect("ordinal");
        let project = conversation % 53;
        out.push(CassDocument {
            agent: AGENTS[usize::try_from(conversation % 5).expect("agent")].to_owned(),
            workspace: Some(format!("/data/projects/agent_workspace_project_{project}")),
            workspace_original: None,
            source_path: format!(
                "/home/ubuntu/.claude/projects/-data-projects-agent-workspace-project-{project}/{session}.jsonl"
            ),
            msg_idx,
            created_at: Some(1_700_000_000_000 + i64::try_from(ordinal * 1_500).expect("ts")),
            title,
            content,
            source_id: "local".to_owned(),
            origin_kind: "local".to_owned(),
            origin_host: None,
            conversation_id: Some(i64::try_from(conversation).expect("conversation id")),
        });
        msg_idx += 1;
    }
    out
}

struct Shard {
    accumulator: ColumnarAccumulator,
    ids: Vec<String>,
}

fn accumulate(documents: &[CassDocument]) -> Shard {
    let mut accumulator = ColumnarAccumulator::new(CASS_SEMANTIC_SCHEMA).expect("CASS accumulator");
    let mut ids = Vec::with_capacity(documents.len());
    for (ordinal, document) in documents.iter().enumerate() {
        let derived = CassDerivedColumns::derive(document.as_ref());
        let values = CassFieldValues::build(document.as_ref(), &derived);
        accumulator
            .add_document_with_values(
                u32::try_from(ordinal).expect("doc ord"),
                &values.indexed,
                &values.numeric,
                &values.stored,
            )
            .expect("accumulate CASS document");
        ids.push(document.to_schema_document().id);
    }
    Shard { accumulator, ids }
}

/// Which seal an arm runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Incumbent,
    Candidate,
}

fn seal(shard: &Shard, arm: Arm, mode: FlushMode) -> EncodedSegment {
    let documents = shard
        .ids
        .iter()
        .enumerate()
        .map(|(ordinal, id)| {
            FlushDocumentInput::new(
                u32::try_from(ordinal).expect("doc ord"),
                id,
                xxh3_64(id.as_bytes()),
            )
        })
        .collect::<Vec<_>>();
    let input = FlushSegmentInput {
        segment_id: 0x5ea1_0000_0000_0011,
        lease_docid_base: u64::from(DOC_ORDS_PER_LEASE) * 3,
        created_unix_s: 1_762_000_000,
        engine_version: CURRENT_ENGINE_VERSION,
        documents: &documents,
    };
    match arm {
        Arm::Incumbent => {
            flush_accumulator_with_per_posting_fieldnorms(&shard.accumulator, input, mode)
        }
        Arm::Candidate => flush_accumulator_with_mode(&shard.accumulator, input, mode),
    }
    .expect("seal full lease")
}

/// `(VmRSS, VmHWM)` in KiB; zeros where `/proc` is unavailable.
fn rss_kib() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    };
    (field("VmRSS:"), field("VmHWM:"))
}

/// Reset this process's `VmHWM` to its current RSS (Linux `clear_refs` 5).
fn reset_peak_rss() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

/// Peak RSS growth over the starting RSS, in KiB, while `f` runs alone.
fn peak_rss_growth_kib<T>(f: impl FnOnce() -> T) -> (T, u64) {
    reset_peak_rss();
    let (before, _) = rss_kib();
    let value = f();
    let (_, peak) = rss_kib();
    (value, peak.saturating_sub(before))
}

/// One cell: a parity check, then the A/A null and the lever in paired
/// alternating rounds. `run(arm)` performs one timed unit and checks its
/// output's file xxh3 itself.
fn run_cell(name: &str, rounds: usize, run: impl Fn(Arm)) {
    let wall = RefCell::new(Vec::new());
    let timed = |arm: Arm| {
        let started = Instant::now();
        run(arm);
        wall.borrow_mut()
            .push((arm, started.elapsed().as_secs_f64()));
    };
    let null = paired_median_ratio(
        rounds,
        1,
        || timed(Arm::Incumbent),
        || timed(Arm::Incumbent),
    );
    let lever = paired_median_ratio(
        rounds,
        1,
        || timed(Arm::Incumbent),
        || timed(Arm::Candidate),
    );
    let seconds = |arm: Arm| {
        let mut values: Vec<f64> = wall
            .borrow()
            .iter()
            .filter(|(which, _)| *which == arm)
            .map(|(_, s)| *s)
            .collect();
        values.sort_unstable_by(f64::total_cmp);
        values
    };
    let incumbent_s = seconds(Arm::Incumbent);
    let candidate_s = seconds(Arm::Candidate);
    let (rss, hwm) = rss_kib();
    eprintln!(
        "[seal] cell={name} rounds={rounds} {} {} decidable={} \
         incumbent_s p50={:.4} [{:.4}, {:.4}] n={} candidate_s p50={:.4} [{:.4}, {:.4}] n={} \
         rss_kib={rss} hwm_kib={hwm}",
        describe("null", &null),
        describe("lever(candidate/incumbent)", &lever),
        lever.decidable_against(&null),
        incumbent_s[incumbent_s.len() / 2],
        incumbent_s[0],
        incumbent_s[incumbent_s.len() - 1],
        incumbent_s.len(),
        candidate_s[candidate_s.len() / 2],
        candidate_s[0],
        candidate_s[candidate_s.len() - 1],
        candidate_s.len(),
    );
}

fn describe(label: &str, ratio: &PairedRatio) -> String {
    format!(
        "{label}={:.4} [p5 {:.4}, p95 {:.4}] ci95=[{:.4}, {:.4}]{}",
        ratio.median,
        ratio.p5,
        ratio.p95,
        ratio.median_ci95_low,
        ratio.median_ci95_high,
        if label == "null" {
            format!(" null_admissible={}", ratio.is_admissible_null())
        } else {
            String::new()
        },
    )
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn host_line() -> String {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default();
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let model = cpuinfo
        .lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split(':').nth(1))
        .unwrap_or("")
        .trim();
    format!(
        "host={} cpu=\"{model}\" available_parallelism={}",
        host.trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
    )
}

fn main() {
    let _ = print_bench_elf_sha256();
    let docs: usize = env_or(
        "QUILL_SEAL_DOCS",
        usize::try_from(DOC_ORDS_PER_LEASE).expect("lease fits usize"),
    );
    let rounds: usize = env_or("QUILL_SEAL_ROUNDS", 15);
    let concurrent: usize = env_or("QUILL_SEAL_CONCURRENT", 8);
    eprintln!(
        "[harness] {} docs={docs} rounds={rounds} concurrent={concurrent}",
        host_line()
    );

    let started = Instant::now();
    let documents = corpus(docs, 0x5ea1_ca55);
    let generated = started.elapsed();
    let started = Instant::now();
    let shard = accumulate(&documents);
    let accumulated = started.elapsed();
    drop(documents);

    let scalar_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("one-thread seal pool");
    let concurrent_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(concurrent.max(1))
        .build()
        .expect("concurrent seal pool");

    // Parity before timing, with each arm's seal-time peak RSS growth.
    let (incumbent, incumbent_peak) = peak_rss_growth_kib(|| {
        scalar_pool.install(|| seal(&shard, Arm::Incumbent, FlushMode::Scalar))
    });
    let (candidate, candidate_peak) = peak_rss_growth_kib(|| {
        scalar_pool.install(|| seal(&shard, Arm::Candidate, FlushMode::Scalar))
    });
    assert_eq!(
        incumbent.as_bytes(),
        candidate.as_bytes(),
        "scalar: candidate segment bytes must equal the incumbent's"
    );
    assert_eq!(incumbent.file_xxh3(), candidate.file_xxh3());
    for arm in [Arm::Incumbent, Arm::Candidate] {
        let automatic = concurrent_pool.install(|| seal(&shard, arm, FlushMode::Automatic));
        assert_eq!(
            automatic.as_bytes(),
            incumbent.as_bytes(),
            "automatic: segment bytes must equal the scalar incumbent's"
        );
    }
    SegmentReader::from_bytes(candidate.as_bytes(), CASS_SEMANTIC_SCHEMA)
        .expect("candidate segment reopens")
        .verify()
        .expect("candidate segment witnesses verify");
    let reference_xxh3 = incumbent.file_xxh3();
    let mut sections = String::new();
    for entry in candidate.section_entries() {
        let _ = write!(sections, " k{}={}", entry.kind.raw(), entry.len);
    }
    eprintln!(
        "[corpus] docs={docs} tokens={} terms={} generate_s={:.2} accumulate_s={:.2} \
         arena_used={} segment_bytes={} file_xxh3={reference_xxh3:016x} \
         byte_identical=true sections:{sections}",
        shard.accumulator.token_count(),
        shard.accumulator.terms().len(),
        generated.as_secs_f64(),
        accumulated.as_secs_f64(),
        shard.accumulator.bytes_used(),
        candidate.file_len(),
    );
    let (rss, hwm) = rss_kib();
    eprintln!(
        "[rss] after_parity rss_kib={rss} hwm_kib={hwm} \
         scalar_seal_peak_growth_kib incumbent={incumbent_peak} candidate={candidate_peak}"
    );
    drop((incumbent, candidate));

    run_cell("scalar_1thread", rounds, |arm| {
        let encoded = scalar_pool.install(|| seal(black_box(&shard), arm, FlushMode::Scalar));
        assert_eq!(
            encoded.file_xxh3(),
            reference_xxh3,
            "timed scalar seal drift"
        );
        black_box(encoded);
    });
    if concurrent > 0 {
        run_cell(&format!("concurrent_{concurrent}"), rounds, |arm| {
            concurrent_pool.install(|| {
                (0..concurrent).into_par_iter().for_each(|_| {
                    let encoded = seal(black_box(&shard), arm, FlushMode::Automatic);
                    assert_eq!(
                        encoded.file_xxh3(),
                        reference_xxh3,
                        "timed concurrent seal drift"
                    );
                    black_box(encoded);
                });
            });
        });
    }
}
