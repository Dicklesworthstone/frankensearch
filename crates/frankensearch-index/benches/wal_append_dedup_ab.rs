//! Profile and paired gate for resident-WAL deduplication in `VectorIndex::append_batch`.
//!
//! The initial profile measures the shipped repeated-retain loop without changing
//! production. Each timed product sample starts from the same 768-entry resident
//! WAL, then appends 256 dimension-384 vectors with the default fsync policy.

use std::collections::HashSet;
use std::fs;
use std::hint::black_box;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use frankensearch_index::{VectorIndex, WalConfig, wal_path_for};

const DIMENSION: usize = 384;
const MAIN_RECORDS: usize = 32;
const RESIDENT_WAL: usize = 768;
const APPEND_BATCH: usize = 256;
const PROFILE_ROUNDS: usize = 21;
const PAIRED_ROUND_PAIRS: usize = 21;

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

#[derive(Clone, Copy)]
enum AppendArm {
    Original,
    SkipRedundantDedup,
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

fn vector(seed: usize) -> Vec<f32> {
    (0..DIMENSION)
        .map(|axis| {
            let raw = seed.wrapping_mul(131).wrapping_add(axis * 17) % 2_003;
            let raw = u16::try_from(raw).expect("fixture value fits u16");
            (f32::from(raw) - 1_001.0) / 1_001.0
        })
        .collect()
}

fn resident_entries() -> Vec<(String, Vec<f32>)> {
    (0..RESIDENT_WAL)
        .map(|index| (format!("resident-{index:06}"), vector(10_000 + index)))
        .collect()
}

fn append_entries(overlap_pct: usize) -> Vec<(String, Vec<f32>)> {
    let replacements = APPEND_BATCH * overlap_pct / 100;
    (0..APPEND_BATCH)
        .map(|index| {
            let doc_id = if index < replacements {
                format!("resident-{index:06}")
            } else {
                format!("new-{overlap_pct:03}-{index:06}")
            };
            (doc_id, vector(20_000 + overlap_pct * APPEND_BATCH + index))
        })
        .collect()
}

fn bench_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "wal_append_dedup_ab_{}_{}.fsvi",
        std::process::id(),
        label
    ))
}

fn create_base(resident: &[(String, Vec<f32>)]) -> PathBuf {
    let path = bench_path("base");
    let mut writer =
        VectorIndex::create(&path, "wal-dedup-profile", DIMENSION).expect("create profile base");
    for index in 0..MAIN_RECORDS {
        writer
            .write_record(&format!("main-{index:06}"), &vector(index))
            .expect("write main record");
    }
    writer.finish().expect("finish profile base");

    let mut index = VectorIndex::open(&path).expect("open profile base");
    index.set_wal_config(WalConfig::default());
    index
        .append_batch(resident)
        .expect("seed resident WAL outside timing");
    assert_eq!(index.wal_record_count(), RESIDENT_WAL);
    drop(index);
    path
}

fn reset_from_base(base: &Path, label: &str) -> PathBuf {
    let path = bench_path(label);
    fs::copy(base, &path).expect("copy base FSVI");
    fs::copy(wal_path_for(base), wal_path_for(&path)).expect("copy base WAL");
    path
}

fn time_product_append(
    base: &Path,
    entries: &[(String, Vec<f32>)],
    label: &str,
    arm: AppendArm,
) -> Duration {
    let path = reset_from_base(base, label);
    let mut index = VectorIndex::open(&path).expect("open reset product index");
    index.set_wal_config(WalConfig::default());
    let started = Instant::now();
    match arm {
        AppendArm::Original => index.append_batch(entries),
        AppendArm::SkipRedundantDedup => index.bench_append_batch_skip_redundant_dedup(entries),
    }
    .expect("append profiled batch");
    let elapsed = started.elapsed();
    black_box(index.wal_record_count());
    elapsed
}

fn paired_ratio(
    base: &Path,
    entries: &[(String, Vec<f32>)],
    overlap_pct: usize,
    label: &str,
    arm_a: AppendArm,
    arm_b: AppendArm,
) -> RatioDistribution {
    let run_pair = |round: usize, record: bool| {
        let a_then_b_a = time_product_append(
            base,
            entries,
            &format!("{label}-{overlap_pct}-{round}-ab-a"),
            arm_a,
        );
        let a_then_b_b = time_product_append(
            base,
            entries,
            &format!("{label}-{overlap_pct}-{round}-ab-b"),
            arm_b,
        );
        let b_then_a_b = time_product_append(
            base,
            entries,
            &format!("{label}-{overlap_pct}-{round}-ba-b"),
            arm_b,
        );
        let b_then_a_a = time_product_append(
            base,
            entries,
            &format!("{label}-{overlap_pct}-{round}-ba-a"),
            arm_a,
        );

        if record {
            let ab = a_then_b_b.as_secs_f64() / a_then_b_a.as_secs_f64();
            let ba = b_then_a_b.as_secs_f64() / b_then_a_a.as_secs_f64();
            Some((ab * ba).sqrt())
        } else {
            black_box((a_then_b_a, a_then_b_b, b_then_a_b, b_then_a_a));
            None
        }
    };

    for warmup in 0..2 {
        let _ = run_pair(PAIRED_ROUND_PAIRS + warmup, false);
    }
    ratio_distribution(
        (0..PAIRED_ROUND_PAIRS)
            .map(|round| run_pair(round, true).expect("recorded round pair"))
            .collect(),
    )
}

fn hit_fingerprints(index: &VectorIndex, overlap_pct: usize) -> Vec<Vec<(u32, String, u32)>> {
    let appended_seed = 20_000 + overlap_pct * APPEND_BATCH;
    [
        0_usize,
        MAIN_RECORDS - 1,
        10_000,
        10_001,
        10_127,
        10_511,
        appended_seed,
        appended_seed + APPEND_BATCH / 2,
        appended_seed + APPEND_BATCH - 1,
    ]
    .into_iter()
    .map(|seed| {
        index
            .search_top_k(&vector(seed), 16, None)
            .expect("parity search")
            .into_iter()
            .map(|hit| (hit.index, hit.doc_id.to_string(), hit.score.to_bits()))
            .collect()
    })
    .collect()
}

fn prove_parity(base: &Path, entries: &[(String, Vec<f32>)], overlap_pct: usize) {
    let original_path = reset_from_base(base, &format!("parity-{overlap_pct}-original"));
    let candidate_path = reset_from_base(base, &format!("parity-{overlap_pct}-candidate"));
    let mut original = VectorIndex::open(&original_path).expect("open parity original");
    let mut candidate = VectorIndex::open(&candidate_path).expect("open parity candidate");
    original.set_wal_config(WalConfig::default());
    candidate.set_wal_config(WalConfig::default());

    original
        .append_batch(entries)
        .expect("append parity original");
    candidate
        .bench_append_batch_skip_redundant_dedup(entries)
        .expect("append parity candidate");
    assert_eq!(original.wal_record_count(), candidate.wal_record_count());
    assert_eq!(
        fs::read(wal_path_for(&original_path)).expect("read original WAL"),
        fs::read(wal_path_for(&candidate_path)).expect("read candidate WAL"),
        "WAL sidecars must be byte-identical"
    );
    assert_eq!(
        hit_fingerprints(&original, overlap_pct),
        hit_fingerprints(&candidate, overlap_pct),
        "immediate hit identity must hold"
    );
    drop(original);
    drop(candidate);

    let mut original = VectorIndex::open(&original_path).expect("reopen parity original");
    let mut candidate = VectorIndex::open(&candidate_path).expect("reopen parity candidate");
    assert_eq!(
        hit_fingerprints(&original, overlap_pct),
        hit_fingerprints(&candidate, overlap_pct),
        "reopened hit identity must hold"
    );
    let original_stats = original.compact().expect("compact parity original");
    let candidate_stats = candidate.compact().expect("compact parity candidate");
    assert_eq!(
        (
            original_stats.main_records_before,
            original_stats.wal_records,
            original_stats.total_records_after,
        ),
        (
            candidate_stats.main_records_before,
            candidate_stats.wal_records,
            candidate_stats.total_records_after,
        )
    );
    assert_eq!(
        fs::read(&original_path).expect("read compacted original FSVI"),
        fs::read(&candidate_path).expect("read compacted candidate FSVI"),
        "compacted FSVI files must be byte-identical"
    );
    eprintln!(
        "[parity] overlap_pct={overlap_pct} wal_bytes_identical=true immediate_hits_identical=true reopened_hits_identical=true score_bits_identical=true compacted_fsvi_bytes_identical=true output_byte_identical=true"
    );
}

fn print_ratio(label: &str, overlap_pct: usize, value: RatioDistribution) {
    eprintln!(
        "[paired] comparison={label} overlap_pct={overlap_pct} candidate_over_original_median={:.6} p5={:.6} p95={:.6} round_pairs={}",
        value.median, value.p5, value.p95, value.round_pairs
    );
}

fn post_soft_delete_ids(
    resident: &[(String, Vec<f32>)],
    entries: &[(String, Vec<f32>)],
) -> Vec<String> {
    let incoming: HashSet<&str> = entries.iter().map(|(doc_id, _)| doc_id.as_str()).collect();
    resident
        .iter()
        .filter(|(doc_id, _)| !incoming.contains(doc_id.as_str()))
        .map(|(doc_id, _)| doc_id.clone())
        .collect()
}

fn time_shipped_retain(mut resident: Vec<String>, entries: &[(String, Vec<f32>)]) -> Duration {
    let started = Instant::now();
    for (doc_id, _) in entries {
        resident.retain(|existing| existing != doc_id);
    }
    let elapsed = started.elapsed();
    black_box(resident);
    elapsed
}

fn print_distribution(label: &str, overlap_pct: usize, value: Distribution) {
    eprintln!(
        "[profile] stage={label} overlap_pct={overlap_pct} median_ms={:.6} p5_ms={:.6} p95_ms={:.6}",
        value.median, value.p5, value.p95
    );
}

fn main() {
    eprintln!(
        "[profile-config] resident_wal={RESIDENT_WAL} append_batch={APPEND_BATCH} dimension={DIMENSION} main_records={MAIN_RECORDS} rounds={PROFILE_ROUNDS} fsync_on_write=true payload_mib={:.3}",
        (APPEND_BATCH * DIMENSION * size_of::<f32>()) as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "[profile-config] binary_path={}",
        std::env::current_exe()
            .expect("resolve measured binary")
            .display()
    );

    let resident = resident_entries();
    let base = create_base(&resident);
    let mut all_shape_gates_pass = true;
    for overlap_pct in [0_usize, 50, 100] {
        let entries = append_entries(overlap_pct);
        let post_delete = post_soft_delete_ids(&resident, &entries);

        for warmup in 0..3 {
            black_box(time_product_append(
                &base,
                &entries,
                &format!("profile-{overlap_pct}-warmup-{warmup}"),
                AppendArm::Original,
            ));
            black_box(time_shipped_retain(post_delete.clone(), &entries));
        }

        let mut product_samples = Vec::with_capacity(PROFILE_ROUNDS);
        let mut retain_samples = Vec::with_capacity(PROFILE_ROUNDS);
        for round in 0..PROFILE_ROUNDS {
            product_samples.push(time_product_append(
                &base,
                &entries,
                &format!("profile-{overlap_pct}-{round}"),
                AppendArm::Original,
            ));
            retain_samples.push(time_shipped_retain(post_delete.clone(), &entries));
        }
        let product = distribution(product_samples);
        let retain = distribution(retain_samples);
        print_distribution("append_batch_product", overlap_pct, product);
        print_distribution("resident_wal_repeated_retain", overlap_pct, retain);
        eprintln!(
            "[profile] overlap_pct={overlap_pct} retain_to_product_median_fraction={:.6}",
            retain.median / product.median
        );

        prove_parity(&base, &entries, overlap_pct);
        let null = paired_ratio(
            &base,
            &entries,
            overlap_pct,
            "null-original-original",
            AppendArm::Original,
            AppendArm::Original,
        );
        let lever = paired_ratio(
            &base,
            &entries,
            overlap_pct,
            "lever-original-skip-redundant-dedup",
            AppendArm::Original,
            AppendArm::SkipRedundantDedup,
        );
        print_ratio("null_original_original", overlap_pct, null);
        print_ratio("skip_redundant_dedup_vs_original", overlap_pct, lever);
        let gate_pass = null.null_is_unbiased() && lever.median < null.p5;
        all_shape_gates_pass &= gate_pass;
        eprintln!(
            "[gate] overlap_pct={overlap_pct} verdict={} median_speedup={:.6}x null_is_unbiased={} candidate_median_below_null_p5={} gate_pass={gate_pass}",
            lever.verdict_against(null),
            1.0 / lever.median,
            null.null_is_unbiased(),
            lever.median < null.p5
        );
    }
    eprintln!(
        "[gate-summary] decision={} all_three_overlap_shapes_clear_null_floor={all_shape_gates_pass}",
        if all_shape_gates_pass { "KEEP" } else { "HOLD" }
    );
}
