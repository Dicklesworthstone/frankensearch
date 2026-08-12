//! `Model2Vec` embed GATHER software-prefetch A/B.
//!
//! `Model2VecEmbedder::embed_sync` mean-pools token vectors: for each token id it
//! indexes a random row `emb[id*DIM .. id*DIM+DIM]` of the `[vocab, DIM]` static
//! table and AVX2-accumulates it into `sum`. For a real model (potion-base-8M:
//! vocab ~30k, DIM 256 → a ~30 MB table) each token's row is a RANDOM location that
//! the hardware stride-prefetcher cannot predict → a first-touch cache miss per
//! token. The accumulate itself is cheap (DIM/8 AVX2 adds), so the loop is
//! MEMORY-LATENCY-bound on the indirect gather — the same shape as the CLS-attention
//! prefetch win, but on an indirect scatter instead of a fixed stride.
//!
//! The whole token-id sequence is known upfront, so `emb`'s row for token `i+PF` can
//! be `_mm_prefetch`'d while token `i`'s row accumulates — hiding the miss latency.
//! Prefetch is a hint → the accumulated `sum` is BIT-IDENTICAL to the base loop
//! (parity asserts max-delta 0), so this is exact and distribution-independent in
//! correctness.
//!
//! Historical Criterion arms retain `base`, `pf_head`, and `pf_row`. The shipping
//! gate additionally compares the exact former production loop with
//! `accumulate_model2vec_rows`: native-256 rows are fused in ordered groups of at
//! most four below 512 tokens, while the existing full-row prefetch remains at
//! 512+. Mean-scaling plus L2 normalization are included. Each timed arm traverses
//! a full 30 MB table copy so repeated sampling cannot turn the long-document
//! workload into an L2-resident microbenchmark.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench -j 4 \
//!   -p frankensearch-embed --features model2vec,bench-internals \
//!   --profile release --bench model2vec_gather_prefetch -- --noplot
//! ```

use std::hint::black_box;
use std::time::Duration;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::{paired_median_ratio, print_bench_elf_sha256};
use frankensearch_embed::Model2VecEmbedder;
use frankensearch_embed::simd::{accumulate_f32_into, accumulate_model2vec_rows};

const VOCAB: usize = 30_000; // potion-base-8M-class vocab
const DIM: usize = 256; // potion-base-8M dimension → 30k*256*4 ≈ 30 MB table
const PF: usize = 4; // prefetch distance (tokens ahead)
const LINE_F32: usize = 16; // 64-byte cache line = 16 f32

/// Prefetch the row starting at `row_start` (f32 offset). `full` = every cache line
/// of the DIM-wide row; else just the first line (HW streams the rest).
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
fn prefetch_row(emb: &[f32], row_start: usize, full: bool) {
    // SAFETY: _mm_prefetch is a hint; any address is architecturally valid to
    // prefetch and we bound the offset by the slice length anyway.
    unsafe {
        if !full {
            if row_start < emb.len() {
                _mm_prefetch(emb.as_ptr().add(row_start).cast::<i8>(), _MM_HINT_T0);
            }
            return;
        }
        let mut off = row_start;
        let end = (row_start + DIM).min(emb.len());
        while off < end {
            _mm_prefetch(emb.as_ptr().add(off).cast::<i8>(), _MM_HINT_T0);
            off += LINE_F32;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn prefetch_row(_emb: &[f32], _row_start: usize, _full: bool) {}

/// ORIGINAL: gather each token's row and accumulate, no prefetch.
fn gather_base(emb: &[f32], ids: &[u32], sum: &mut [f32]) -> usize {
    sum.fill(0.0);
    let mut count = 0_usize;
    for &id in ids {
        let idx = id as usize;
        if idx < VOCAB {
            let start = idx * DIM;
            accumulate_f32_into(sum, &emb[start..start + DIM]);
            count += 1;
        }
    }
    count
}

/// CANDIDATE: prefetch token `i+PF`'s row while accumulating token `i`.
fn gather_prefetch(emb: &[f32], ids: &[u32], sum: &mut [f32], full: bool) {
    sum.fill(0.0);
    let n = ids.len();
    for i in 0..n {
        if i + PF < n {
            let pidx = ids[i + PF] as usize;
            if pidx < VOCAB {
                prefetch_row(emb, pidx * DIM, full);
            }
        }
        let idx = ids[i] as usize;
        if idx < VOCAB {
            let start = idx * DIM;
            accumulate_f32_into(sum, &emb[start..start + DIM]);
        }
    }
}

fn gather_gated(emb: &[f32], ids: &[u32], sum: &mut [f32]) -> usize {
    sum.fill(0.0);
    accumulate_model2vec_rows(sum, emb, ids, VOCAB)
}

fn finish_mean_pool(sum: &mut [f32], count: usize) {
    assert!(
        count > 0,
        "benchmark corpus must contain in-vocabulary rows"
    );
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / count as f32;
    for value in sum.iter_mut() {
        *value *= inv;
    }
    let norm_sq: f32 = sum.iter().map(|value| value * value).sum();
    if norm_sq.is_finite() && norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for value in sum.iter_mut() {
            *value *= inv_norm;
        }
    } else {
        sum.fill(0.0);
    }
}

#[derive(Clone, Copy)]
enum Arm {
    Original,
    ShippingCandidate,
}

#[derive(Clone, Copy)]
enum FullEmbedArm {
    FormerEncode,
    ShippingEncodeFast,
    FormerFinish,
    ShippingFusedFinish,
}

fn fingerprint_document(fingerprint: &mut u64, sum: &[f32], count: usize) {
    for &value in sum {
        *fingerprint ^= u64::from(value.to_bits());
        *fingerprint = fingerprint.wrapping_mul(0x1000_0000_01b3);
    }
    *fingerprint ^= u64::try_from(count).expect("document token count fits u64");
    *fingerprint = fingerprint.wrapping_mul(0x1000_0000_01b3);
}

fn run_corpus(emb: &[f32], ids: &[u32], tokens_per_doc: usize, sum: &mut [f32], arm: Arm) -> u64 {
    let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
    for doc_ids in ids.chunks_exact(tokens_per_doc) {
        let count = match arm {
            Arm::Original => gather_base(emb, doc_ids, sum),
            Arm::ShippingCandidate => gather_gated(emb, doc_ids, sum),
        };
        finish_mean_pool(sum, count);
        fingerprint_document(&mut fingerprint, sum, count);
        black_box(&*sum);
    }
    fingerprint
}

fn corpus_ids(tokens_per_doc: usize) -> Vec<u32> {
    let docs = VOCAB.div_ceil(tokens_per_doc);
    (0..docs * tokens_per_doc)
        .map(|position| u32::try_from((position * 7_919) % VOCAB).expect("VOCAB fits in u32"))
        .collect()
}

fn paired_shipping_gate(emb_original: &[f32], emb_candidate: &[f32]) {
    for &tokens_per_doc in &[1_usize, 2, 3, 4, 8, 16, 32, 64, 128, 256, 511, 512, 513] {
        let ids = corpus_ids(tokens_per_doc);

        let mut parity_original = vec![0.0_f32; DIM];
        let mut parity_candidate = vec![0.0_f32; DIM];
        let original_fingerprint = run_corpus(
            emb_original,
            &ids,
            tokens_per_doc,
            &mut parity_original,
            Arm::Original,
        );
        let candidate_fingerprint = run_corpus(
            emb_original,
            &ids,
            tokens_per_doc,
            &mut parity_candidate,
            Arm::ShippingCandidate,
        );
        assert_eq!(
            original_fingerprint, candidate_fingerprint,
            "gated production path changed a finished pooled document at {tokens_per_doc} tokens"
        );

        let mut null_original = vec![0.0_f32; DIM];
        let mut null_clone = vec![0.0_f32; DIM];
        let null = paired_median_ratio(
            31,
            1,
            || {
                black_box(run_corpus(
                    black_box(emb_original),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut null_original),
                    Arm::Original,
                ));
            },
            || {
                black_box(run_corpus(
                    black_box(emb_candidate),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut null_clone),
                    Arm::Original,
                ));
            },
        );

        assert!(
            null.is_admissible_null(),
            "A/A null is inadmissible at {tokens_per_doc} tokens: {null:?}"
        );

        let mut candidate_a = vec![0.0_f32; DIM];
        let mut candidate_b = vec![0.0_f32; DIM];
        let candidate_null = paired_median_ratio(
            31,
            1,
            || {
                black_box(run_corpus(
                    black_box(emb_original),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut candidate_a),
                    Arm::ShippingCandidate,
                ));
            },
            || {
                black_box(run_corpus(
                    black_box(emb_candidate),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut candidate_b),
                    Arm::ShippingCandidate,
                ));
            },
        );
        assert!(
            candidate_null.is_admissible_null(),
            "B/B null is inadmissible at {tokens_per_doc} tokens: {candidate_null:?}"
        );

        let mut lever_original = vec![0.0_f32; DIM];
        let mut lever_candidate = vec![0.0_f32; DIM];
        let lever = paired_median_ratio(
            31,
            1,
            || {
                black_box(run_corpus(
                    black_box(emb_original),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut lever_original),
                    Arm::Original,
                ));
            },
            || {
                black_box(run_corpus(
                    black_box(emb_candidate),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut lever_candidate),
                    Arm::ShippingCandidate,
                ));
            },
        );
        let decidable = lever.decidable_against(&null) && lever.decidable_against(&candidate_null);
        // This is a regression tripwire, not a win gate: a sub-one median is
        // reported as no-claim until release codegen and live evidence exist.
        assert!(
            !(decidable && lever.median > 1.0),
            "native-256 fused arm regressed at {tokens_per_doc} tokens: A/A={null:?}, B/B={candidate_null:?}, A/B={lever:?}"
        );
        eprintln!(
            "[paired-gate] tokens={tokens_per_doc} AA={null:?} BB={candidate_null:?} AB={lever:?} decidable={decidable} no_claim=true",
        );
    }
}

fn fingerprint_embedding(vector: &[f32]) -> u64 {
    let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
    for &value in vector {
        fingerprint ^= u64::from(value.to_bits());
        fingerprint = fingerprint.wrapping_mul(0x1000_0000_01b3);
    }
    fingerprint
}

fn run_full_embed_sync_corpus(
    embedder: &Model2VecEmbedder,
    texts: &[String],
    arm: FullEmbedArm,
) -> u64 {
    let mut corpus_fingerprint = 0xcbf2_9ce4_8422_2325_u64;
    for text in texts {
        let vector = full_embed_sync_vector(embedder, text, arm);
        let fingerprint = fingerprint_embedding(&vector);
        corpus_fingerprint ^= fingerprint;
        corpus_fingerprint = corpus_fingerprint.wrapping_mul(0x1000_0000_01b3);
        black_box(vector);
    }
    corpus_fingerprint
}

fn full_embed_sync_fingerprints(
    embedder: &Model2VecEmbedder,
    texts: &[String],
    arm: FullEmbedArm,
) -> Vec<u64> {
    texts
        .iter()
        .map(|text| full_embed_sync_vector(embedder, text, arm))
        .map(|vector| fingerprint_embedding(&vector))
        .collect()
}

fn full_embed_sync_vector(embedder: &Model2VecEmbedder, text: &str, arm: FullEmbedArm) -> Vec<f32> {
    match arm {
        FullEmbedArm::FormerEncode => embedder
            .benchmark_embed_sync_former_encode(text)
            .expect("former encode full embed_sync route"),
        FullEmbedArm::ShippingEncodeFast => embedder
            .embed_sync(text)
            .expect("shipping encode_fast full embed_sync route"),
        FullEmbedArm::FormerFinish => embedder
            .benchmark_embed_sync_former_finish(text)
            .expect("former finish full embed_sync route"),
        FullEmbedArm::ShippingFusedFinish => embedder
            .embed_sync(text)
            .expect("shipping fused-finish full embed_sync route"),
    }
}

fn full_embed_sync_corpora() -> Vec<(&'static str, Vec<String>)> {
    let interactive = vec![
        "Rust safe concurrent search index".to_owned(),
        "How should tokenization preserve Unicode combining marks?".to_owned(),
        "café 東京 résumé Model2Vec".to_owned(),
        "exact semantic search result ordering".to_owned(),
    ];

    let mut indexing_batch = (0..16)
        .map(|index| {
            format!(
                "Document {index}: structured concurrency, durable indexing, and lexical semantic fusion."
            )
        })
        .collect::<Vec<_>>();
    indexing_batch.push(
        std::iter::repeat_n("tokenizer gather pooling", 513)
            .collect::<Vec<_>>()
            .join(" "),
    );

    vec![
        ("interactive", interactive),
        ("indexing_batch", indexing_batch),
    ]
}

fn paired_full_embed_sync_gate(
    embedder: &Model2VecEmbedder,
    label: &str,
    texts: &[String],
    former: FullEmbedArm,
    shipping: FullEmbedArm,
    comparison: &str,
    fail_on_decidable_regression: bool,
) {
    let former_fingerprints = full_embed_sync_fingerprints(embedder, texts, former);
    let shipping_fingerprints = full_embed_sync_fingerprints(embedder, texts, shipping);
    assert_eq!(
        shipping_fingerprints, former_fingerprints,
        "full embed_sync output fingerprint drift in {comparison} {label}"
    );

    let aa = paired_median_ratio(
        31,
        1,
        || {
            black_box(run_full_embed_sync_corpus(embedder, texts, former));
        },
        || {
            black_box(run_full_embed_sync_corpus(embedder, texts, former));
        },
    );
    assert!(
        aa.is_admissible_null(),
        "full embed_sync A/A null is inadmissible for {label}: {aa:?}"
    );

    let bb = paired_median_ratio(
        31,
        1,
        || {
            black_box(run_full_embed_sync_corpus(embedder, texts, shipping));
        },
        || {
            black_box(run_full_embed_sync_corpus(embedder, texts, shipping));
        },
    );
    assert!(
        bb.is_admissible_null(),
        "full embed_sync B/B null is inadmissible for {label}: {bb:?}"
    );

    let ab = paired_median_ratio(
        31,
        1,
        || {
            black_box(run_full_embed_sync_corpus(embedder, texts, former));
        },
        || {
            black_box(run_full_embed_sync_corpus(embedder, texts, shipping));
        },
    );
    let decidable = ab.decidable_against(&aa) && ab.decidable_against(&bb);
    if fail_on_decidable_regression {
        assert!(
            !(decidable && ab.median > 1.0),
            "full embed_sync {comparison} regressed in {label}: A/A={aa:?}, B/B={bb:?}, A/B={ab:?}"
        );
    }
    eprintln!(
        "[full-embed-sync] comparison={comparison} distribution={label} AA={aa:?} BB={bb:?} AB={ab:?} \
         decidable={decidable} no_claim=true"
    );
}

fn bench_full_embed_sync(c: &mut Criterion) {
    let model_dir = match std::env::var("POTION_FIXTURE_DIR") {
        Ok(path) => path,
        Err(_) => {
            eprintln!(
                "[full-embed-sync] skipped: set POTION_FIXTURE_DIR to a verified Potion model; \
                 arithmetic-only gather timing is not a tokenizer result"
            );
            return;
        }
    };
    let embedder = Model2VecEmbedder::load(&model_dir)
        .expect("POTION_FIXTURE_DIR must pass the registered Potion manifest verification");
    let mut group = c.benchmark_group("model2vec_full_embed_sync");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));

    for (label, texts) in full_embed_sync_corpora() {
        paired_full_embed_sync_gate(
            &embedder,
            label,
            &texts,
            FullEmbedArm::FormerEncode,
            FullEmbedArm::ShippingEncodeFast,
            "encode_fast",
            false,
        );
        paired_full_embed_sync_gate(
            &embedder,
            label,
            &texts,
            FullEmbedArm::FormerFinish,
            FullEmbedArm::ShippingFusedFinish,
            "mean_norm_fused",
            true,
        );
        group.bench_with_input(
            BenchmarkId::new("former_encode", label),
            &texts,
            |bench, texts| {
                bench.iter(|| {
                    black_box(run_full_embed_sync_corpus(
                        &embedder,
                        texts,
                        FullEmbedArm::FormerEncode,
                    ));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("shipping_encode_fast", label),
            &texts,
            |bench, texts| {
                bench.iter(|| {
                    black_box(run_full_embed_sync_corpus(
                        &embedder,
                        texts,
                        FullEmbedArm::ShippingEncodeFast,
                    ));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("former_mean_norm_finish", label),
            &texts,
            |bench, texts| {
                bench.iter(|| {
                    black_box(run_full_embed_sync_corpus(
                        &embedder,
                        texts,
                        FullEmbedArm::FormerFinish,
                    ));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("shipping_mean_norm_fused", label),
            &texts,
            |bench, texts| {
                bench.iter(|| {
                    black_box(run_full_embed_sync_corpus(
                        &embedder,
                        texts,
                        FullEmbedArm::ShippingFusedFinish,
                    ));
                });
            },
        );
    }
    group.finish();
}

fn emb_fixture() -> Vec<f32> {
    let mut out = vec![0.0f32; VOCAB * DIM];
    let mut s = 0x9e37_79b9_7f4a_7c15_u64;
    for v in &mut out {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
    }
    out
}

fn ids_fixture(t: usize) -> Vec<u32> {
    // Uniform-random token ids = the cache-cold regime (broad-vocab doc embedding
    // over a 30 MB table). Realistic Zipfian text keeps hot tokens cached and would
    // see a smaller gather benefit; this is the gather-heavy case.
    let mut out = Vec::with_capacity(t);
    let mut s = 0x1234_5678_9abc_def0_u64;
    for _ in 0..t {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.push(u32::try_from((s >> 33) % 30_000).expect("modulus fits in u32"));
    }
    out
}

fn bench(c: &mut Criterion) {
    print_bench_elf_sha256().expect("executing benchmark ELF identity");
    let mut group = c.benchmark_group("model2vec_gather_prefetch");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));

    let emb = emb_fixture();
    let emb_candidate = emb.clone();
    paired_shipping_gate(&emb, &emb_candidate);

    for &t in &[16usize, 64, 256] {
        let ids = ids_fixture(t);

        // Parity: prefetch is a hint → sum is bit-identical to the base loop.
        let mut a = vec![0.0f32; DIM];
        let mut bh = vec![0.0f32; DIM];
        let mut br = vec![0.0f32; DIM];
        let _ = gather_base(&emb, &ids, &mut a);
        gather_prefetch(&emb, &ids, &mut bh, false);
        gather_prefetch(&emb, &ids, &mut br, true);
        let dh = a
            .iter()
            .zip(&bh)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let dr = a
            .iter()
            .zip(&br)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            dh == 0.0,
            "pf_head diverged from base by {dh} (t={t}) — must be bit-identical"
        );
        assert!(
            dr == 0.0,
            "pf_row diverged from base by {dr} (t={t}) — must be bit-identical"
        );

        group.bench_with_input(BenchmarkId::new("base", t), &ids, |bn, ids| {
            let mut sum = vec![0.0f32; DIM];
            bn.iter(|| {
                let _ = gather_base(black_box(&emb), black_box(ids), black_box(&mut sum));
                black_box(&sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("pf_head", t), &ids, |bn, ids| {
            let mut sum = vec![0.0f32; DIM];
            bn.iter(|| {
                gather_prefetch(black_box(&emb), black_box(ids), black_box(&mut sum), false);
                black_box(&sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("pf_row", t), &ids, |bn, ids| {
            let mut sum = vec![0.0f32; DIM];
            bn.iter(|| {
                gather_prefetch(black_box(&emb), black_box(ids), black_box(&mut sum), true);
                black_box(&sum);
            });
        });
    }
    group.finish();
    bench_full_embed_sync(c);
}

criterion_group!(benches, bench);
criterion_main!(benches);
