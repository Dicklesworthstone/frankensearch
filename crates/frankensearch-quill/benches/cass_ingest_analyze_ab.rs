#![recursion_limit = "256"]
//! CASS per-document ingest A/B: CASS fields analyzed through the borrowed
//! byte-indexed `CassAnalyzer::analyze_borrowed` path (shipping) vs the owned
//! char-walk `CassAnalyzer::analyze` re-borrowed per token through the
//! trait's owned-token bridge (incumbent, `OwnedBridgeCassFamily`).
//!
//! A deterministic CASS-shaped corpus (Zipf vocabulary; log-normal message
//! lengths, median ~300 bytes with a tail to tens of KB; prose, code and tool
//! output with capitals, identifiers, paths, URLs, JSON, numbers, hashes and
//! about 1.2% non-ASCII text) is projected onto `SchemaDocument`s exactly as
//! CASS does (`CassDocument::to_schema_document`, outside the timing). Both
//! arms then run the fan-out's per-document accumulate step
//! (`accumulate_shard_run` -> `IngestDoc::accumulate` ->
//! `ColumnarAccumulator::add_document_with_values`) over the same documents,
//! starting a fresh shard accumulator whenever the previous one reaches the
//! engine-default 64 MiB shard budget, as a sealed shard does. Seals are not
//! timed; their input is identical in both arms.
//!
//! Before timing, both arms must intern the same terms and fill the same
//! columns over the whole corpus, and a 16,384-document shard of each must
//! seal byte-identical segments (same file xxh3) under the scalar and the
//! automatic radix that reopen and verify. Every timed run re-checks a
//! fingerprint of each accumulator it builds (terms, tokens, documents).
//!
//! Two cells, each an A/A incumbent null plus the lever ratio
//! `candidate / incumbent` (`< 1.0` outside the null band wins), from paired
//! rounds whose order alternates:
//!
//! * `shard_fill_1thread`: one thread accumulating the corpus as one shard.
//! * `fanout_k`: `k` threads, each accumulating its shard of a fan-out
//!   partition (256-document runs dealt round-robin), as a fan-out wave does.
//!
//! `QUILL_INGEST_PROFILE_SECS=n` instead records `perf` (Linux, `sudo -n`)
//! over the full shipping `index_schema_documents` route on a one-thread pool.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 rch exec -- env CARGO_PROFILE_RELEASE_LTO=false \
//!   CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 QUILL_INGEST_ROUNDS=15 QUILL_INGEST_FANOUT=8 \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!     --profile release --bench cass_ingest_analyze_ab
//! ```

use std::cell::RefCell;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio, print_bench_elf_sha256};
use frankensearch_quill::cass::CassDocument;
use frankensearch_quill::config::DEFAULT_SCRIBE_SHARD_BUDGET_BYTES;
use frankensearch_quill::schema::CASS_SEMANTIC_SCHEMA;
use frankensearch_quill::scribe::{
    ColumnarAccumulator, DOC_ORDS_PER_LEASE, FlushDocumentInput, FlushMode, FlushSegmentInput,
    IndexedFieldValue, IndexedNumericValue, OwnedBridgeCassFamily, StoredFieldValue, TokenAnalyzer,
    flush_accumulator_with_mode,
};
use frankensearch_quill::segment::SegmentReader;
use frankensearch_quill::{
    CURRENT_ENGINE_VERSION, EncodedSegment, QuillConfig, QuillIndex, SchemaDocument,
};
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

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[usize::try_from(self.below(items.len() as u64)).expect("index")]
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

/// Like [`word`], with the first letter upper-cased.
fn capitalized(rank: u64, out: &mut String) {
    let start = out.len();
    word(rank, out);
    out[start..=start].make_ascii_uppercase();
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
const PROSE_SEPARATORS: [&str; 12] = [
    " ", " ", " ", " ", " ", " ", " ", " ", ", ", " (", ") ", "; ",
];
const CODE_KEYWORDS: [&str; 16] = [
    "fn", "let", "mut", "pub", "impl", "self", "return", "if", "else", "match", "for", "in", "use",
    "struct", "Ok", "Err",
];
const CODE_PUNCT: [&str; 12] = [
    " = ", "(", ")", "::", ".", ", ", "; ", " -> ", " { ", " }", "&", "<",
];
const EXTENSIONS: [&str; 6] = ["rs", "py", "ts", "md", "json", "toml"];
const URL_HOSTS: [&str; 4] = [
    "https://github.com",
    "https://docs.rs",
    "https://crates.io/crates",
    "http://localhost:8080",
];
/// Non-ASCII material seen in agent transcripts: CJK prose, accented Latin,
/// Cyrillic, typographic punctuation, arrows, check marks and emoji.
const NON_ASCII: [&str; 16] = [
    "构建索引",
    "搜索结果",
    "错误处理",
    "テスト",
    "ファイル",
    "검색",
    "café",
    "naïve",
    "résumé",
    "Größe",
    "проверка",
    "—",
    "→",
    "✓",
    "“quoted”",
    "🚀",
];

/// Vocabulary size and Zipf exponent (prose and identifiers).
const VOCABULARY: usize = 40_000;
const ZIPF_S: f64 = 1.0;
/// Log-normal message length: median words and sigma (tail capped).
const MEDIAN_WORDS: f64 = 50.0;
const SIGMA: f64 = 1.3;
const MAX_WORDS: f64 = 6_000.0;

#[derive(Clone, Copy)]
enum Style {
    Prose,
    Code,
    ToolOutput,
}

/// Append one token of `style` to `content`.
#[allow(clippy::too_many_lines)]
fn token(style: Style, zipf: &Zipf, rng: &mut Rng, sentence_start: bool, content: &mut String) {
    let roll = rng.below(1_000);
    // Shared rare kinds: non-ASCII runs, hex hashes, numbers.
    if roll < 12 {
        content.push_str(rng.pick(&NON_ASCII));
        return;
    }
    if roll < 16 {
        let _ = write!(content, "{:x}", rng.next() >> (8 * rng.below(4)));
        return;
    }
    if roll < 46 {
        let _ = write!(
            content,
            "{}",
            rng.below(if roll % 2 == 0 { 100 } else { 100_000 })
        );
        return;
    }
    match style {
        Style::Prose => {
            if sentence_start || roll < 70 {
                capitalized(zipf.sample(rng), content);
            } else if roll < 100 {
                word(zipf.sample(rng), content);
                content.push('-');
                word(zipf.sample(rng), content);
            } else {
                word(zipf.sample(rng), content);
            }
        }
        Style::Code => {
            if roll < 200 {
                content.push_str(rng.pick(&CODE_KEYWORDS));
            } else if roll < 330 {
                // CamelCase type name.
                for _ in 0..=rng.below(3) {
                    capitalized(zipf.sample(rng), content);
                }
            } else if roll < 360 {
                // SCREAMING_SNAKE constant.
                let start = content.len();
                word(zipf.sample(rng), content);
                content.push('_');
                word(zipf.sample(rng), content);
                content[start..].make_ascii_uppercase();
            } else if roll < 520 {
                // snake_case identifier.
                word(zipf.sample(rng), content);
                content.push('_');
                word(zipf.sample(rng), content);
            } else if roll < 560 {
                // Module path.
                content.push_str("crate::");
                word(zipf.sample(rng), content);
                content.push_str("::");
                capitalized(zipf.sample(rng), content);
            } else {
                word(zipf.sample(rng), content);
            }
        }
        Style::ToolOutput => {
            if roll < 160 {
                // File path with a line/column suffix.
                content.push_str("/data/projects/");
                word(zipf.sample(rng) % 200, content);
                content.push_str("/src/");
                word(zipf.sample(rng), content);
                content.push('_');
                word(zipf.sample(rng), content);
                content.push('.');
                content.push_str(rng.pick(&EXTENSIONS));
                let _ = write!(content, ":{}:{}", 1 + rng.below(4_000), 1 + rng.below(80));
            } else if roll < 220 {
                content.push_str(rng.pick(&URL_HOSTS));
                content.push('/');
                word(zipf.sample(rng) % 500, content);
                content.push('/');
                word(zipf.sample(rng), content);
                let _ = write!(content, "/pull/{}", rng.below(5_000));
            } else if roll < 250 {
                let _ = write!(
                    content,
                    "2026-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                    1 + rng.below(12),
                    1 + rng.below(28),
                    rng.below(24),
                    rng.below(60),
                    rng.below(60)
                );
            } else if roll < 300 {
                content.push('"');
                word(zipf.sample(rng), content);
                content.push_str("\": \"");
                word(zipf.sample(rng), content);
                content.push('"');
            } else if roll < 340 {
                let start = content.len();
                word(zipf.sample(rng), content);
                content[start..].make_ascii_uppercase();
            } else if roll < 420 {
                word(zipf.sample(rng), content);
                content.push('-');
                word(zipf.sample(rng), content);
            } else {
                word(zipf.sample(rng), content);
            }
        }
    }
}

/// Deterministic CASS-shaped messages. Each conversation has 1-120 messages;
/// each message is prose, code or tool output (60/20/20) with a log-normal
/// word count (median 50, heavy tail to 6,000 words, tens of KB), a Zipf
/// vocabulary with capitalized sentence starts, `CamelCase`/`snake_case`/
/// constant identifiers, module paths, file paths, URLs, timestamps, JSON,
/// hex hashes, numbers, and about 1.2% non-ASCII tokens (CJK, accented Latin,
/// Cyrillic, typographic punctuation, emoji).
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
        let style = match rng.below(10) {
            0..=5 => Style::Prose,
            6 | 7 => Style::Code,
            _ => Style::ToolOutput,
        };
        // Box-Muller normal -> log-normal word count.
        let u1 = rng.unit().max(1e-12);
        let u2 = rng.unit();
        let normal = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let words = (MEDIAN_WORDS * (SIGMA * normal).exp()).clamp(1.0, MAX_WORDS);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let words = words as usize;
        let mut content = String::with_capacity(words * 8);
        let mut sentence_start = true;
        for position in 0..words {
            if position != 0 {
                match style {
                    Style::Prose => {
                        if rng.below(14) == 0 {
                            content.push_str(if rng.below(5) == 0 { ".\n" } else { ". " });
                            sentence_start = true;
                        } else {
                            content.push_str(rng.pick(&PROSE_SEPARATORS));
                        }
                    }
                    Style::Code => {
                        content.push_str(if rng.below(9) == 0 {
                            "\n    "
                        } else {
                            rng.pick(&CODE_PUNCT)
                        });
                    }
                    Style::ToolOutput => {
                        content.push_str(if rng.below(7) == 0 { "\n" } else { " " });
                    }
                }
            }
            token(style, &zipf, &mut rng, sentence_start, &mut content);
            sentence_start = false;
        }
        let title = (msg_idx == 0).then(|| {
            let mut title = String::new();
            for index in 0..(2 + rng.below(8)) {
                if index == 0 {
                    capitalized(zipf.sample(&mut rng), &mut title);
                } else {
                    title.push(' ');
                    word(zipf.sample(&mut rng), &mut title);
                }
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

/// Which analyzer family an arm accumulates with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Incumbent,
    Candidate,
}

/// What every accumulator an arm builds must agree on: distinct terms,
/// admitted token rows and documents.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Fingerprint {
    terms: usize,
    tokens: usize,
    documents: usize,
}

/// One schema document through `IngestDoc::accumulate`'s projection.
fn add_schema_document<A: TokenAnalyzer>(
    accumulator: &mut ColumnarAccumulator<A>,
    doc_ord: u32,
    document: &SchemaDocument,
) {
    let indexed = document
        .indexed
        .iter()
        .map(|(field_ord, text)| IndexedFieldValue::new(*field_ord, text))
        .collect::<Vec<_>>();
    let numeric = document
        .numeric
        .iter()
        .map(|(field_ord, value)| IndexedNumericValue {
            field_ord: *field_ord,
            value: *value,
        })
        .collect::<Vec<_>>();
    let stored = document
        .stored
        .iter()
        .map(|(field_ord, bytes)| StoredFieldValue::new(*field_ord, bytes))
        .collect::<Vec<_>>();
    accumulator
        .add_document_with_values(doc_ord, &indexed, &numeric, &stored)
        .expect("accumulate CASS document");
}

/// Accumulate one shard's documents in order, starting a fresh accumulator
/// whenever the current one reaches `budget_bytes` (where a shard seals) or
/// fills its docid lease; `finished` sees each completed accumulator.
fn accumulate_shard<A: TokenAnalyzer>(
    documents: &[&SchemaDocument],
    budget_bytes: usize,
    new_accumulator: impl Fn() -> ColumnarAccumulator<A>,
    mut finished: impl FnMut(ColumnarAccumulator<A>),
) {
    let mut accumulator = new_accumulator();
    let mut doc_ord = 0_u32;
    for document in documents {
        if doc_ord == DOC_ORDS_PER_LEASE || accumulator.should_flush(budget_bytes) {
            finished(std::mem::replace(&mut accumulator, new_accumulator()));
            doc_ord = 0;
        }
        add_schema_document(&mut accumulator, doc_ord, document);
        doc_ord += 1;
    }
    finished(accumulator);
}

fn fingerprint<A: TokenAnalyzer>(accumulator: &ColumnarAccumulator<A>) -> Fingerprint {
    Fingerprint {
        terms: accumulator.terms().len(),
        tokens: accumulator.token_count(),
        documents: accumulator.document_count(),
    }
}

/// One arm over one shard, returning the fingerprints of its accumulators.
fn run_shard(arm: Arm, documents: &[&SchemaDocument], budget_bytes: usize) -> Vec<Fingerprint> {
    let mut prints = Vec::new();
    match arm {
        Arm::Incumbent => accumulate_shard(
            documents,
            budget_bytes,
            || {
                ColumnarAccumulator::with_analyzer(
                    CASS_SEMANTIC_SCHEMA,
                    OwnedBridgeCassFamily::default(),
                )
                .expect("incumbent CASS accumulator")
            },
            |accumulator| prints.push(fingerprint(&black_box(accumulator))),
        ),
        Arm::Candidate => accumulate_shard(
            documents,
            budget_bytes,
            || ColumnarAccumulator::new(CASS_SEMANTIC_SCHEMA).expect("shipping CASS accumulator"),
            |accumulator| prints.push(fingerprint(&black_box(accumulator))),
        ),
    }
    prints
}

/// The fan-out's contiguous partition: 256-document runs dealt to `shards`
/// shards in ascending order, wave after wave.
fn fanout_partition(documents: &[SchemaDocument], shards: usize) -> Vec<Vec<&SchemaDocument>> {
    let mut partition = vec![Vec::new(); shards];
    for (run, chunk) in documents.chunks(256).enumerate() {
        partition[run % shards].extend(chunk);
    }
    partition
}

/// Both arms over a whole shard: identical terms in id order and identical
/// token, position, length and fieldnorm columns in every accumulator.
fn assert_same_accumulators(documents: &[&SchemaDocument], budget_bytes: usize) -> usize {
    let mut incumbent = Vec::new();
    accumulate_shard(
        documents,
        budget_bytes,
        || {
            ColumnarAccumulator::with_analyzer(
                CASS_SEMANTIC_SCHEMA,
                OwnedBridgeCassFamily::default(),
            )
            .expect("incumbent CASS accumulator")
        },
        |accumulator| incumbent.push(accumulator),
    );
    let mut compared = 0;
    let mut index = 0;
    accumulate_shard(
        documents,
        budget_bytes,
        || ColumnarAccumulator::new(CASS_SEMANTIC_SCHEMA).expect("shipping CASS accumulator"),
        |candidate| {
            let reference = &incumbent[index];
            index += 1;
            assert_eq!(candidate.terms().len(), reference.terms().len());
            for id in 0..u32::try_from(candidate.terms().len()).expect("term ids fit u32") {
                assert_eq!(
                    candidate.terms().composite_key(id),
                    reference.terms().composite_key(id),
                    "term {id}"
                );
            }
            assert_eq!(candidate.document_ords(), reference.document_ords());
            for (left, right) in candidate.fields().iter().zip(reference.fields()) {
                assert_eq!(left.field_ord(), right.field_ord());
                assert_eq!(left.term_ids(), right.term_ids());
                assert_eq!(left.doc_ords(), right.doc_ords());
                assert_eq!(left.positions(), right.positions());
                assert_eq!(left.document_lengths(), right.document_lengths());
                assert_eq!(left.fieldnorm_ids(), right.fieldnorm_ids());
                assert_eq!(left.total_tokens(), right.total_tokens());
            }
            compared += 1;
        },
    );
    assert_eq!(compared, incumbent.len());
    compared
}

fn seal<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
    ids: &[String],
    mode: FlushMode,
) -> EncodedSegment {
    let documents = ids
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
    flush_accumulator_with_mode(
        accumulator,
        FlushSegmentInput {
            segment_id: 0x0ca5_5ade_0000_0012,
            lease_docid_base: 3 << 16,
            created_unix_s: 1_762_000_000,
            engine_version: CURRENT_ENGINE_VERSION,
            documents: &documents,
        },
        mode,
    )
    .expect("seal CASS shard")
}

/// Seal one shard of `documents` from each arm under both radix modes and
/// require byte-identical, verifiable segments. Returns the file xxh3.
fn assert_same_segments(documents: &[SchemaDocument], pool: &rayon::ThreadPool) -> (u64, u64) {
    let mut incumbent =
        ColumnarAccumulator::with_analyzer(CASS_SEMANTIC_SCHEMA, OwnedBridgeCassFamily::default())
            .expect("incumbent CASS accumulator");
    let mut candidate =
        ColumnarAccumulator::new(CASS_SEMANTIC_SCHEMA).expect("shipping CASS accumulator");
    for (ordinal, document) in documents.iter().enumerate() {
        let doc_ord = u32::try_from(ordinal).expect("doc ord");
        add_schema_document(&mut incumbent, doc_ord, document);
        add_schema_document(&mut candidate, doc_ord, document);
    }
    let ids = documents
        .iter()
        .map(|document| document.id.clone())
        .collect::<Vec<_>>();
    let reference = seal(&incumbent, &ids, FlushMode::Scalar);
    for mode in [FlushMode::Scalar, FlushMode::Automatic] {
        for (name, segment) in [
            ("incumbent", pool.install(|| seal(&incumbent, &ids, mode))),
            ("candidate", pool.install(|| seal(&candidate, &ids, mode))),
        ] {
            assert_eq!(
                segment.as_bytes(),
                reference.as_bytes(),
                "{name} {mode:?}: segment bytes must equal the scalar incumbent's"
            );
        }
    }
    SegmentReader::from_bytes(reference.as_bytes(), CASS_SEMANTIC_SCHEMA)
        .expect("CASS segment reopens")
        .verify()
        .expect("CASS segment witnesses verify");
    (reference.file_xxh3(), reference.file_len())
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

/// One cell: the A/A null and the lever in paired alternating rounds.
/// `run(arm)` performs one timed unit and checks its output itself.
fn run_cell(name: &str, rounds: usize, documents: usize, run: impl Fn(Arm)) {
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
    let per_doc = |values: &[f64]| values[values.len() / 2] * 1e6 / documents as f64;
    eprintln!(
        "[ingest] cell={name} rounds={rounds} docs={documents} {} {} decidable={} \
         incumbent_s p50={:.4} [{:.4}, {:.4}] n={} ({:.2} us/doc) \
         candidate_s p50={:.4} [{:.4}, {:.4}] n={} ({:.2} us/doc) rss_kib={rss} hwm_kib={hwm}",
        describe("null", &null),
        describe("lever(candidate/incumbent)", &lever),
        lever.decidable_against(&null),
        incumbent_s[incumbent_s.len() / 2],
        incumbent_s[0],
        incumbent_s[incumbent_s.len() - 1],
        incumbent_s.len(),
        per_doc(&incumbent_s),
        candidate_s[candidate_s.len() / 2],
        candidate_s[0],
        candidate_s[candidate_s.len() - 1],
        candidate_s.len(),
        per_doc(&candidate_s),
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

fn config(shards: usize) -> QuillConfig {
    // CASS's writer configuration (`cass_quill_config`): explicit publication
    // only and no engine tier merge; everything else at the engine default.
    QuillConfig {
        max_visibility_lag_ms: u64::MAX,
        tier_fanout: usize::MAX,
        max_ingest_shards: shards,
        ..QuillConfig::default()
    }
}

fn drive<T>(future: impl std::future::Future<Output = T>) -> T {
    asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(1, 4)
        .build()
        .expect("bench runtime")
        .block_on(future)
}

/// One full-pipeline pass: a fresh durable CASS index, every batch through
/// `index_schema_documents`, no commit. Returns the ingest seconds.
fn ingest_pipeline(cx: &Cx, documents: &[SchemaDocument], shards: usize, batch: usize) -> f64 {
    let scratch = tempfile::tempdir().expect("bench index root");
    drive(async {
        let index = QuillIndex::create_with_schema(
            cx,
            scratch.path().join("index"),
            CASS_SEMANTIC_SCHEMA,
            config(shards),
        )
        .await
        .expect("durable CASS index");
        let started = Instant::now();
        for chunk in documents.chunks(batch) {
            index
                .index_schema_documents(cx, black_box(chunk))
                .await
                .expect("ingest");
        }
        let seconds = started.elapsed().as_secs_f64();
        black_box(&index);
        seconds
    })
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

fn percentile(sorted: &[usize], percent: usize) -> usize {
    sorted[((sorted.len() - 1) * percent + 50) / 100]
}

fn corpus_stats(documents: &[CassDocument], projected: &[SchemaDocument]) {
    let mut lengths = documents
        .iter()
        .map(|document| document.content.len())
        .collect::<Vec<_>>();
    lengths.sort_unstable();
    let content_bytes: usize = lengths.iter().sum();
    let non_ascii_bytes: usize = documents
        .iter()
        .map(|document| {
            document
                .content
                .bytes()
                .filter(|byte| !byte.is_ascii())
                .count()
        })
        .sum();
    let upper_bytes: usize = documents
        .iter()
        .map(|document| {
            document
                .content
                .bytes()
                .filter(u8::is_ascii_uppercase)
                .count()
        })
        .sum();
    let mut field_bytes = [0_usize; 16];
    for document in projected {
        for (field_ord, text) in &document.indexed {
            field_bytes[usize::from(*field_ord)] += text.len();
        }
    }
    eprintln!(
        "[corpus] docs={} content_bytes={content_bytes} content_len p50={} p90={} p99={} max={} \
         non_ascii_byte_frac={:.4} upper_byte_frac={:.4} indexed_bytes_by_field={:?}",
        documents.len(),
        percentile(&lengths, 50),
        percentile(&lengths, 90),
        percentile(&lengths, 99),
        lengths[lengths.len() - 1],
        non_ascii_bytes as f64 / content_bytes as f64,
        upper_bytes as f64 / content_bytes as f64,
        field_bytes,
    );
}

/// Run `work` under `perf record` for at least `seconds`, then print a flat
/// self-time report and the hottest source lines of the top symbols.
fn profile(label: &str, seconds: f64, mut work: impl FnMut()) {
    let path = format!("/dev/shm/quill-ingest-p12-{label}.perf");
    let pid = std::process::id().to_string();
    let mut perf = std::process::Command::new("sudo")
        .args([
            "-n", "perf", "record", "-F", "1999", "-p", &pid, "-o", &path,
        ])
        .spawn()
        .expect("spawn perf record");
    std::thread::sleep(std::time::Duration::from_millis(1_500));
    eprintln!("QUILL_INGEST_PROFILE_START {label}");
    let started = Instant::now();
    let until = std::time::Duration::from_secs_f64(seconds);
    let mut passes = 0;
    while started.elapsed() < until {
        work();
        passes += 1;
    }
    eprintln!(
        "QUILL_INGEST_PROFILE_END {label} passes={passes} per_pass_s={:.3}",
        started.elapsed().as_secs_f64() / f64::from(passes)
    );
    let _ = std::process::Command::new("sudo")
        .args(["-n", "kill", "-INT", &perf.id().to_string()])
        .status();
    let _ = perf.wait();
    let report = std::process::Command::new("sudo")
        .args([
            "-n",
            "perf",
            "report",
            "-i",
            &path,
            "--stdio",
            "--no-children",
            "--sort",
            "symbol",
            "--percent-limit",
            "0.3",
        ])
        .output()
        .expect("perf report");
    let text = String::from_utf8_lossy(&report.stdout);
    eprintln!("=== perf report {label}");
    for line in text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
    {
        let trimmed = line.trim_end();
        eprintln!("{}", &trimmed[..trimmed.len().min(220)]);
    }
}

fn main() {
    let _ = print_bench_elf_sha256();
    let docs: usize = env_or("QUILL_INGEST_DOCS", 65_536);
    let rounds: usize = env_or("QUILL_INGEST_ROUNDS", 15);
    let fanout: usize = env_or("QUILL_INGEST_FANOUT", 8);
    let shards: usize = env_or("QUILL_INGEST_SHARDS", 32);
    let batch: usize = env_or("QUILL_INGEST_BATCH", 6_400);
    let profile_secs: f64 = env_or("QUILL_INGEST_PROFILE_SECS", 0.0);
    let budget = DEFAULT_SCRIBE_SHARD_BUDGET_BYTES;
    eprintln!(
        "[harness] {} docs={docs} rounds={rounds} fanout={fanout} budget_bytes={budget}",
        host_line()
    );

    let started = Instant::now();
    let documents = corpus(docs, 0x5ea1_ca55_0012);
    let generated = started.elapsed().as_secs_f64();
    let projected = documents
        .par_iter()
        .map(CassDocument::to_schema_document)
        .collect::<Vec<_>>();
    corpus_stats(&documents, &projected);
    drop(documents);
    eprintln!("[harness] generate_s={generated:.2}");

    if profile_secs > 0.0 {
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool");
        let cx = Cx::for_testing();
        eprintln!("[profile] shards={shards} batch={batch}");
        profile("pipeline1t", profile_secs, || {
            black_box(one.install(|| ingest_pipeline(&cx, &projected, shards, batch)));
        });
        return;
    }

    // Parity before timing: the whole corpus as one shard, accumulator by
    // accumulator, then one sealed shard per arm under both radix modes.
    let whole = projected.iter().collect::<Vec<_>>();
    let accumulators = assert_same_accumulators(&whole, budget);
    let seal_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("seal pool");
    let seal_docs = projected.len().min(16_384);
    let (file_xxh3, file_len) = assert_same_segments(&projected[..seal_docs], &seal_pool);
    let expected_whole = run_shard(Arm::Incumbent, &whole, budget);
    assert_eq!(run_shard(Arm::Candidate, &whole, budget), expected_whole);
    let tokens: usize = expected_whole.iter().map(|print| print.tokens).sum();
    eprintln!(
        "[parity] accumulators={accumulators} tokens={tokens} columns_equal=true \
         seal_docs={seal_docs} file_len={file_len} file_xxh3={file_xxh3:016x} \
         scalar_eq_automatic=true segments_equal=true verify=ok"
    );

    run_cell("shard_fill_1thread", rounds, docs, |arm| {
        let prints = run_shard(arm, black_box(&whole), budget);
        assert_eq!(prints, expected_whole, "{arm:?} shard fingerprints");
    });

    let partition = fanout_partition(&projected, fanout);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fanout)
        .build()
        .expect("fan-out pool");
    let run_wave = |arm: Arm| {
        pool.install(|| {
            partition
                .par_iter()
                .map(|shard| run_shard(arm, shard, budget))
                .collect::<Vec<_>>()
        })
    };
    let expected_fanout = run_wave(Arm::Incumbent);
    assert_eq!(run_wave(Arm::Candidate), expected_fanout);
    run_cell(&format!("fanout_{fanout}"), rounds, docs, |arm| {
        assert_eq!(
            run_wave(black_box(arm)),
            expected_fanout,
            "{arm:?} fan-out fingerprints"
        );
    });
}
