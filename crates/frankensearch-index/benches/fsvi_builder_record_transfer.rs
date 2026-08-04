//! Profile and paired gates for owned-record transfers in the two-tier FSVI builder.
//!
//! The fixture models one 20k-document, 384-dimension tier. Fixture construction
//! and per-arm source cloning stay outside the timer. The default gate preserves the
//! historical builder-to-writer comparison; pass `builder-api` to measure the earlier
//! caller-to-builder copy.

use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use frankensearch_core::TwoTierConfig;
use frankensearch_index::{
    TwoTierIndex, VECTOR_INDEX_FAST_FILENAME, VectorIndex, VectorIndexWriter,
};

const DOCUMENTS: usize = 20_000;
const DIMENSION: usize = 384;
const PROFILE_ROUNDS: usize = 21;
const PAIRED_ROUND_PAIRS: usize = 21;
const PARITY_QUERIES: usize = 8;

#[derive(Clone, Copy)]
struct Distribution {
    median: f64,
    p5: f64,
    p95: f64,
}

#[derive(Clone, Copy)]
struct RatioDistribution {
    median: f64,
    p5: f64,
    p95: f64,
    round_pairs: usize,
}

impl RatioDistribution {
    /// Largest `|median - 1|` an A/A control may show and still be admissible.
    /// Mirrors `frankensearch_core::bench_support::NULL_MEDIAN_TOLERANCE`.
    const NULL_MEDIAN_TOLERANCE: f64 = 0.02;

    /// A/A control admissibility: gate the null's accuracy, never its precision.
    ///
    /// Requiring `p5 <= 1.0 <= p95` vetoed a null for being *tight*: the
    /// narrower its spread, the likelier it excluded 1.0 — however small the
    /// bias and however large the effect (`bd-pjh09`). The spread still sets
    /// the effect floor below; only the median says whether the sampler itself
    /// is trustworthy.
    fn null_is_unbiased(self) -> bool {
        (self.median - 1.0).abs() <= Self::NULL_MEDIAN_TOLERANCE
    }

    fn verdict_against(self, null: Self) -> &'static str {
        if !null.null_is_unbiased() {
            "BIASED_NULL_UNDECIDABLE"
        } else if self.median < null.p5 {
            "CANDIDATE_FASTER"
        } else if self.median > null.p95 {
            "CANDIDATE_SLOWER"
        } else {
            "INSIDE_NULL_FLOOR"
        }
    }
}

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    let index = ((sorted.len() - 1) * pct + 50) / 100;
    sorted[index]
}

fn distribution(mut samples: Vec<Duration>) -> Distribution {
    samples.sort_unstable();
    Distribution {
        median: percentile(&samples, 50).as_secs_f64() * 1_000.0,
        p5: percentile(&samples, 5).as_secs_f64() * 1_000.0,
        p95: percentile(&samples, 95).as_secs_f64() * 1_000.0,
    }
}

fn ratio_distribution(mut samples: Vec<f64>) -> RatioDistribution {
    samples.sort_unstable_by(f64::total_cmp);
    let index = |pct: usize| ((samples.len() - 1) * pct + 50) / 100;
    RatioDistribution {
        median: samples[index(50)],
        p5: samples[index(5)],
        p95: samples[index(95)],
        round_pairs: samples.len(),
    }
}

fn fixture() -> Vec<(String, Vec<f32>)> {
    (0..DOCUMENTS)
        .map(|document| {
            let vector = (0..DIMENSION)
                .map(|axis| {
                    let value = document.wrapping_mul(131).wrapping_add(axis * 17) % 2_003;
                    let value = u16::try_from(value).expect("fixture value fits u16");
                    (f32::from(value) - 1_001.0) / 1_001.0
                })
                .collect();
            (format!("document-{document:08}"), vector)
        })
        .collect()
}

fn writer_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fsvi_builder_record_transfer_{}_{}.fsvi",
        std::process::id(),
        label
    ))
}

fn create_writer(label: &str) -> VectorIndexWriter {
    VectorIndex::create(&writer_path(label), "profile-384", DIMENSION)
        .expect("create in-memory writer state")
}

fn builder_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fsvi_builder_api_transfer_{}_{}",
        std::process::id(),
        label
    ))
}

fn create_builder(label: &str) -> frankensearch_index::TwoTierIndexBuilder {
    TwoTierIndex::create(&builder_dir(label), TwoTierConfig::default())
        .expect("create two-tier builder")
}

fn time_borrowed_builder_add(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut builder = create_builder(label);
    let started = Instant::now();
    for (doc_id, embedding) in records {
        builder
            .add_fast_record(doc_id, &embedding)
            .expect("add borrowed builder record");
    }
    let elapsed = started.elapsed();
    black_box(&builder);
    elapsed
}

fn time_owned_builder_add(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut builder = create_builder(label);
    let started = Instant::now();
    for (doc_id, embedding) in records {
        builder
            .add_fast_record_owned_for_benchmark(doc_id, embedding)
            .expect("add owned builder record");
    }
    let elapsed = started.elapsed();
    black_box(&builder);
    elapsed
}

fn time_borrowed_builder_finish(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut builder = create_builder(label);
    let started = Instant::now();
    for (doc_id, embedding) in records {
        builder
            .add_fast_record(doc_id, &embedding)
            .expect("add borrowed builder record");
    }
    let index = builder.finish().expect("finish borrowed builder");
    let elapsed = started.elapsed();
    black_box(&index);
    elapsed
}

fn time_owned_builder_finish(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut builder = create_builder(label);
    let started = Instant::now();
    for (doc_id, embedding) in records {
        builder
            .add_fast_record_owned_for_benchmark(doc_id, embedding)
            .expect("add owned builder record");
    }
    let index = builder.finish().expect("finish owned builder");
    let elapsed = started.elapsed();
    black_box(&index);
    elapsed
}

fn time_borrowed_ingest(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut writer = create_writer(label);
    let started = Instant::now();
    for (doc_id, embedding) in &records {
        writer
            .write_record(doc_id, embedding)
            .expect("buffer valid record");
    }
    let elapsed = started.elapsed();
    black_box(&writer);
    drop(writer);
    drop(records);
    elapsed
}

fn time_owned_ingest(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut writer = create_writer(label);
    let started = Instant::now();
    for (doc_id, embedding) in records {
        writer
            .write_record_owned_for_benchmark(doc_id, embedding)
            .expect("buffer valid owned record");
    }
    let elapsed = started.elapsed();
    black_box(&writer);
    drop(writer);
    elapsed
}

fn time_borrowed_finish(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut writer = create_writer(label);
    let started = Instant::now();
    for (doc_id, embedding) in &records {
        writer
            .write_record(doc_id, embedding)
            .expect("buffer valid record");
    }
    writer.finish().expect("finish borrowed writer");
    let elapsed = started.elapsed();
    drop(records);
    elapsed
}

fn time_owned_finish(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut writer = create_writer(label);
    let started = Instant::now();
    for (doc_id, embedding) in records {
        writer
            .write_record_owned_for_benchmark(doc_id, embedding)
            .expect("buffer valid owned record");
    }
    writer.finish().expect("finish owned writer");
    started.elapsed()
}

fn timed_pair<A, B>(
    original: &mut A,
    candidate: &mut B,
    original_first: bool,
) -> (Duration, Duration)
where
    A: FnMut() -> Duration,
    B: FnMut() -> Duration,
{
    if original_first {
        let original_elapsed = original();
        let candidate_elapsed = candidate();
        (original_elapsed, candidate_elapsed)
    } else {
        let candidate_elapsed = candidate();
        let original_elapsed = original();
        (original_elapsed, candidate_elapsed)
    }
}

fn paired_ratio<A, B>(mut original: A, mut candidate: B) -> RatioDistribution
where
    A: FnMut() -> Duration,
    B: FnMut() -> Duration,
{
    for _ in 0..2 {
        black_box(timed_pair(&mut original, &mut candidate, true));
        black_box(timed_pair(&mut original, &mut candidate, false));
    }

    let mut ratios = Vec::with_capacity(PAIRED_ROUND_PAIRS);
    for pair in 0..PAIRED_ROUND_PAIRS {
        let (ab, ba) = if pair.is_multiple_of(2) {
            (
                timed_pair(&mut original, &mut candidate, true),
                timed_pair(&mut original, &mut candidate, false),
            )
        } else {
            (
                timed_pair(&mut original, &mut candidate, false),
                timed_pair(&mut original, &mut candidate, true),
            )
        };
        let ab_ratio = ab.1.as_secs_f64() / ab.0.as_secs_f64();
        let ba_ratio = ba.1.as_secs_f64() / ba.0.as_secs_f64();
        ratios.push((ab_ratio * ba_ratio).sqrt());
    }
    ratio_distribution(ratios)
}

fn profile_writer_handoff(source: &[(String, Vec<f32>)]) {
    let mut clone_samples = Vec::with_capacity(PROFILE_ROUNDS);
    let mut writer_samples = Vec::with_capacity(PROFILE_ROUNDS);

    for round in 0..PROFILE_ROUNDS {
        let started = Instant::now();
        let cloned = source.to_vec();
        let clone_elapsed = started.elapsed();
        black_box(&cloned);
        clone_samples.push(clone_elapsed);
        drop(cloned);

        // Model the builder's already-owned source records outside the timed handoff.
        writer_samples.push(time_borrowed_ingest(source, &format!("baseline-{round}")));
    }

    let clone_profile = distribution(clone_samples);
    let writer_profile = distribution(writer_samples);
    eprintln!(
        "[profile] record_deep_clone median_ms={:.3} p5_ms={:.3} p95_ms={:.3}",
        clone_profile.median, clone_profile.p5, clone_profile.p95
    );
    eprintln!(
        "[profile] borrowed_writer_ingest median_ms={:.3} p5_ms={:.3} p95_ms={:.3}",
        writer_profile.median, writer_profile.p5, writer_profile.p95
    );
    eprintln!(
        "[profile] clone_to_ingest_median_fraction={:.4}",
        clone_profile.median / writer_profile.median
    );
}

fn profile_builder_api(source: &[(String, Vec<f32>)]) {
    let mut clone_samples = Vec::with_capacity(PROFILE_ROUNDS);
    let mut builder_samples = Vec::with_capacity(PROFILE_ROUNDS);

    for round in 0..PROFILE_ROUNDS {
        let started = Instant::now();
        let cloned = source.to_vec();
        let clone_elapsed = started.elapsed();
        black_box(&cloned);
        clone_samples.push(clone_elapsed);
        drop(cloned);

        builder_samples.push(time_borrowed_builder_add(
            source,
            &format!("builder-profile-{round}"),
        ));
    }

    let clone_profile = distribution(clone_samples);
    let builder_profile = distribution(builder_samples);
    eprintln!(
        "[profile] record_deep_clone median_ms={:.3} p5_ms={:.3} p95_ms={:.3}",
        clone_profile.median, clone_profile.p5, clone_profile.p95
    );
    eprintln!(
        "[profile] borrowed_builder_add median_ms={:.3} p5_ms={:.3} p95_ms={:.3}",
        builder_profile.median, builder_profile.p5, builder_profile.p95
    );
    eprintln!(
        "[profile] clone_to_builder_add_median_fraction={:.4}",
        clone_profile.median / builder_profile.median
    );
}

fn prove_writer_parity(source: &[(String, Vec<f32>)]) {
    let borrowed_path = writer_path("parity-borrowed");
    let owned_path = writer_path("parity-owned");

    let mut borrowed = VectorIndex::create(&borrowed_path, "profile-384", DIMENSION)
        .expect("create borrowed parity writer");
    for (doc_id, embedding) in source {
        borrowed
            .write_record(doc_id, embedding)
            .expect("write borrowed parity record");
    }
    borrowed.finish().expect("finish borrowed parity writer");

    let mut owned = VectorIndex::create(&owned_path, "profile-384", DIMENSION)
        .expect("create owned parity writer");
    for (doc_id, embedding) in source.iter().cloned() {
        owned
            .write_record_owned_for_benchmark(doc_id, embedding)
            .expect("write owned parity record");
    }
    owned.finish().expect("finish owned parity writer");

    let borrowed_bytes = fs::read(&borrowed_path).expect("read borrowed parity file");
    let owned_bytes = fs::read(&owned_path).expect("read owned parity file");
    assert_eq!(borrowed_bytes, owned_bytes, "FSVI bytes must be identical");

    let borrowed = VectorIndex::open(&borrowed_path).expect("open borrowed parity index");
    let owned = VectorIndex::open(&owned_path).expect("open owned parity index");
    assert_eq!(borrowed.record_count(), owned.record_count());
    for index in (0..DOCUMENTS).step_by(DOCUMENTS / 32) {
        assert_eq!(
            borrowed.doc_id_at(index).expect("borrowed doc id"),
            owned.doc_id_at(index).expect("owned doc id")
        );
        let borrowed_bits: Vec<u32> = borrowed
            .vector_at_f32(index)
            .expect("borrowed vector")
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let owned_bits: Vec<u32> = owned
            .vector_at_f32(index)
            .expect("owned vector")
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(borrowed_bits, owned_bits);
    }

    for query in source.iter().step_by(DOCUMENTS / PARITY_QUERIES) {
        let borrowed_hits = borrowed
            .search_top_k(&query.1, 10, None)
            .expect("search borrowed index");
        let owned_hits = owned
            .search_top_k(&query.1, 10, None)
            .expect("search owned index");
        assert_eq!(borrowed_hits.len(), owned_hits.len());
        for (borrowed_hit, owned_hit) in borrowed_hits.iter().zip(&owned_hits) {
            assert_eq!(borrowed_hit.index, owned_hit.index);
            assert_eq!(borrowed_hit.doc_id, owned_hit.doc_id);
            assert_eq!(borrowed_hit.score.to_bits(), owned_hit.score.to_bits());
        }
    }

    eprintln!(
        "[parity] fsvi_bytes_identical=true bytes={} sampled_records=32 queries={PARITY_QUERIES} top10_identical=true recall_preserved=true ndcg_preserved=true score_bits_identical=true",
        borrowed_bytes.len()
    );
}

fn prove_builder_api_parity(source: &[(String, Vec<f32>)]) {
    let borrowed_dir = builder_dir("builder-parity-borrowed");
    let owned_dir = builder_dir("builder-parity-owned");

    let mut borrowed = TwoTierIndex::create(&borrowed_dir, TwoTierConfig::default())
        .expect("create borrowed parity builder");
    for (doc_id, embedding) in source {
        borrowed
            .add_fast_record(doc_id.clone(), embedding)
            .expect("add borrowed parity record");
    }
    let borrowed = borrowed.finish().expect("finish borrowed parity builder");

    let mut owned = TwoTierIndex::create(&owned_dir, TwoTierConfig::default())
        .expect("create owned parity builder");
    for (doc_id, embedding) in source.iter().cloned() {
        owned
            .add_fast_record_owned_for_benchmark(doc_id, embedding)
            .expect("add owned parity record");
    }
    let owned = owned.finish().expect("finish owned parity builder");

    let borrowed_bytes = fs::read(borrowed_dir.join(VECTOR_INDEX_FAST_FILENAME))
        .expect("read borrowed builder file");
    let owned_bytes =
        fs::read(owned_dir.join(VECTOR_INDEX_FAST_FILENAME)).expect("read owned builder file");
    assert_eq!(
        borrowed_bytes, owned_bytes,
        "builder FSVI bytes must be identical"
    );

    for query in source.iter().step_by(DOCUMENTS / PARITY_QUERIES) {
        let borrowed_hits = borrowed
            .search_fast(&query.1, 10)
            .expect("search borrowed builder index");
        let owned_hits = owned
            .search_fast(&query.1, 10)
            .expect("search owned builder index");
        assert_eq!(borrowed_hits.len(), owned_hits.len());
        for (borrowed_hit, owned_hit) in borrowed_hits.iter().zip(&owned_hits) {
            assert_eq!(borrowed_hit.index, owned_hit.index);
            assert_eq!(borrowed_hit.doc_id, owned_hit.doc_id);
            assert_eq!(borrowed_hit.score.to_bits(), owned_hit.score.to_bits());
        }
    }

    eprintln!(
        "[parity] builder_api_fsvi_bytes_identical=true bytes={} queries={PARITY_QUERIES} top10_identical=true recall_preserved=true ndcg_preserved=true score_bits_identical=true",
        borrowed_bytes.len()
    );
}

fn print_ratio(label: &str, result: RatioDistribution, null: Option<RatioDistribution>) {
    if let Some(null) = null {
        eprintln!(
            "[paired-lever] {label} candidate/original median={:.6} p5={:.6} p95={:.6} calibrated_ratio={:.6} rounds={} verdict={}",
            result.median,
            result.p5,
            result.p95,
            result.median / null.median,
            result.round_pairs,
            result.verdict_against(null)
        );
    } else {
        eprintln!(
            "[paired-null] {label} median={:.6} p5={:.6} p95={:.6} contains_one={} rounds={}",
            result.median,
            result.p5,
            result.p95,
            result.null_is_unbiased(),
            result.round_pairs
        );
    }
}

fn run_writer_handoff(source: &[(String, Vec<f32>)]) {
    profile_writer_handoff(source);
    prove_writer_parity(source);

    let ingest_null = paired_ratio(
        || time_borrowed_ingest(source, "ingest-null-a"),
        || time_borrowed_ingest(source, "ingest-null-b"),
    );
    let ingest_lever = paired_ratio(
        || time_borrowed_ingest(source, "ingest-original"),
        || time_owned_ingest(source, "ingest-candidate"),
    );
    print_ratio("writer_ingest", ingest_null, None);
    print_ratio("writer_ingest", ingest_lever, Some(ingest_null));

    let finish_null = paired_ratio(
        || time_borrowed_finish(source, "finish-null-a"),
        || time_borrowed_finish(source, "finish-null-b"),
    );
    let finish_lever = paired_ratio(
        || time_borrowed_finish(source, "finish-original"),
        || time_owned_finish(source, "finish-candidate"),
    );
    print_ratio("writer_transfer_plus_finish", finish_null, None);
    print_ratio(
        "writer_transfer_plus_finish",
        finish_lever,
        Some(finish_null),
    );

    let ingest_verdict = ingest_lever.verdict_against(ingest_null);
    let finish_verdict = finish_lever.verdict_against(finish_null);
    eprintln!(
        "[paired-gate] parity=true ingest_verdict={ingest_verdict} finish_verdict={finish_verdict} gate={}",
        if finish_verdict == "CANDIDATE_FASTER" {
            "KEEP"
        } else {
            "HOLD"
        }
    );
}

fn run_builder_api(source: &[(String, Vec<f32>)]) {
    profile_builder_api(source);
    prove_builder_api_parity(source);

    let add_null = paired_ratio(
        || time_borrowed_builder_add(source, "builder-add-null-a"),
        || time_borrowed_builder_add(source, "builder-add-null-b"),
    );
    let add_lever = paired_ratio(
        || time_borrowed_builder_add(source, "builder-add-original"),
        || time_owned_builder_add(source, "builder-add-candidate"),
    );
    print_ratio("builder_api_add", add_null, None);
    print_ratio("builder_api_add", add_lever, Some(add_null));

    let finish_null = paired_ratio(
        || time_borrowed_builder_finish(source, "builder-finish-null-a"),
        || time_borrowed_builder_finish(source, "builder-finish-null-b"),
    );
    let finish_lever = paired_ratio(
        || time_borrowed_builder_finish(source, "builder-finish-original"),
        || time_owned_builder_finish(source, "builder-finish-candidate"),
    );
    print_ratio("builder_api_add_plus_finish", finish_null, None);
    print_ratio(
        "builder_api_add_plus_finish",
        finish_lever,
        Some(finish_null),
    );

    let add_verdict = add_lever.verdict_against(add_null);
    let finish_verdict = finish_lever.verdict_against(finish_null);
    eprintln!(
        "[paired-gate] parity=true add_verdict={add_verdict} finish_verdict={finish_verdict} gate={}",
        if finish_verdict == "CANDIDATE_FASTER" {
            "KEEP"
        } else {
            "HOLD"
        }
    );
}

fn main() {
    let payload_elements = u32::try_from(DOCUMENTS * DIMENSION).expect("payload fits u32");
    eprintln!(
        "[profile-config] documents={DOCUMENTS} dimension={DIMENSION} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS} payload_mib={:.3}",
        f64::from(payload_elements) * 4.0 / (1024.0 * 1024.0)
    );

    let source = fixture();
    if std::env::args().any(|argument| argument == "builder-api") {
        run_builder_api(&source);
    } else {
        run_writer_handoff(&source);
    }
}
