#![recursion_limit = "256"]
//! Scalar batch-admission A/B against a many-segment snapshot: the incumbent
//! all-segment IDHASH probe vs the writer's `PublishedIdFilter` pre-filter.
//!
//! Every scalar `index_documents` batch first runs `validate_batch_admission`,
//! which asks the published snapshot whether each identifier is already live.
//! The incumbent answers by probing every sealed segment's IDHASH, so the
//! per-document cost grows with the segment count. The candidate keeps a
//! split-block Bloom filter over every published IDHASH key and probes only
//! when the filter cannot rule the identifier out. Both arms run in this one
//! binary over the same snapshot and the same batch, through the same
//! admission loop; only the published-snapshot probe differs.
//!
//! The measured batch is fresh identifiers, the bulk-rebuild shape: every
//! one is absent, so the incumbent probes all S segments per document. Parity
//! is asserted before timing: both arms admit the fresh batch, and both refuse
//! the same batch with one published identifier appended with the identical
//! error. The A/A null is incumbent-vs-incumbent; the lever ratio is
//! `filtered / incumbent`, so `< 1.0` outside the null band wins.
//!
//! The `[upsert]` cells time upsert's identity work before admission over the
//! same snapshot: the replacement probe (`batch_replaces_published_document`)
//! and tombstone staging (`delete_document` per identifier), through the same
//! filter or through the incumbent all-segment probes. `fresh` is a batch of
//! new identifiers (new messages, an inline rebuild); `replace` is a batch of
//! published identifiers spread over every segment, where every identifier
//! passes the filter, so it bounds the filter's overhead. Both arms must stage
//! the identical MANIFEST and replaced-id set for fresh, replacing and mixed
//! batches before anything is timed.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_ADMIT_SCALE=full QUILL_ADMIT_ROUNDS=21 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench admission_filter_ab
//!
//! # Fast parity/harness check (not performance evidence):
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- env QUILL_ADMIT_SCALE=smoke QUILL_ADMIT_ROUNDS=5 \
//!     cargo bench -p frankensearch-quill --features bench-internals \
//!       --profile release --bench admission_filter_ab
//! ```

use std::cell::RefCell;
use std::hint::black_box;
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::IndexableDocument;
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_quill::index::BenchAdmissionSession;
use frankensearch_quill::{QuillConfig, QuillIndex};

/// One snapshot shape: sealed segments times documents per segment.
struct Shape {
    name: &'static str,
    segments: usize,
    docs_per_segment: usize,
}

static FULL_SHAPES: [Shape; 4] = [
    Shape {
        name: "mem64x64",
        segments: 64,
        docs_per_segment: 64,
    },
    Shape {
        name: "mem256x64",
        segments: 256,
        docs_per_segment: 64,
    },
    Shape {
        name: "mem1024x64",
        segments: 1_024,
        docs_per_segment: 64,
    },
    Shape {
        name: "mem128x4096",
        segments: 128,
        docs_per_segment: 4_096,
    },
];

static SMOKE_SHAPES: [Shape; 1] = [Shape {
    name: "smoke_mem16x32",
    segments: 16,
    docs_per_segment: 32,
}];

/// Fresh identifiers admitted per timed call.
const PROBE_BATCH_DOCS: usize = 512;
/// Absent identifiers sampled for the filter's false-positive receipt.
const FALSE_POSITIVE_SAMPLE: usize = 65_536;
/// Fixture ingest stays on the scalar route (the bulk planner starts at 64).
const SCALAR_FIXTURE_INGEST_CHUNK_DOCS: usize = 63;

fn fixture_document(segment: usize, ordinal: usize) -> IndexableDocument {
    IndexableDocument::new(
        format!("conv-{segment:05}:msg-{ordinal:05}"),
        format!("admission fixture segment {segment} message {ordinal}"),
    )
}

async fn build_index(cx: &Cx, shape: &Shape) -> QuillIndex {
    let config = QuillConfig {
        deterministic_ingest: true,
        // Explicit commits are the only publication boundaries, and no tier
        // merge may fold the declared segments (CASS runs with the same
        // `tier_fanout: usize::MAX`).
        max_visibility_lag_ms: u64::MAX,
        tier_fanout: usize::MAX,
        ..QuillConfig::default()
    };
    let index = QuillIndex::in_memory(config).expect("in-memory bench index");
    for segment in 0..shape.segments {
        let batch: Vec<IndexableDocument> = (0..shape.docs_per_segment)
            .map(|ordinal| fixture_document(segment, ordinal))
            .collect();
        for chunk in batch.chunks(SCALAR_FIXTURE_INGEST_CHUNK_DOCS) {
            index
                .index_documents(cx, chunk)
                .await
                .expect("accumulate bench fixture chunk");
        }
        index.commit(cx).await.expect("seal bench segment");
    }
    let segments = index
        .snapshot()
        .expect("benchmark snapshot is authoritative")
        .segments()
        .len();
    assert_eq!(
        segments, shape.segments,
        "{}: each commit must seal exactly one segment",
        shape.name
    );
    index
}

fn probe_batch() -> Vec<IndexableDocument> {
    (0..PROBE_BATCH_DOCS)
        .map(|ordinal| {
            IndexableDocument::new(
                format!("fresh-{ordinal:06}:never-published"),
                "fresh admission probe",
            )
        })
        .collect()
}

fn time_per_doc_ns(calls: u32, mut run: impl FnMut()) -> f64 {
    let started = Instant::now();
    for _ in 0..calls {
        run();
    }
    started.elapsed().as_secs_f64() * 1e9 / (f64::from(calls) * PROBE_BATCH_DOCS as f64)
}

fn run_shape(cx: &Cx, shape: &Shape, rounds: usize, session: &mut BenchAdmissionSession) {
    let fresh = probe_batch();

    // Parity before timing: identical admission, identical refusal.
    session
        .validate(cx, &fresh, false)
        .expect("incumbent admits the fresh batch");
    session
        .validate(cx, &fresh, true)
        .expect("filtered probe admits the fresh batch");
    let mut with_duplicate = fresh.clone();
    with_duplicate.push(fixture_document(shape.segments / 2, 0));
    let incumbent_refusal = session
        .validate(cx, &with_duplicate, false)
        .expect_err("incumbent refuses a published id")
        .to_string();
    let filtered_refusal = session
        .validate(cx, &with_duplicate, true)
        .expect_err("filtered probe refuses a published id")
        .to_string();
    assert_eq!(
        incumbent_refusal, filtered_refusal,
        "{}: refusal diverged",
        shape.name
    );

    // Cold synchronization: the first admission after a writer opens.
    session.reset_filter();
    let cold_started = Instant::now();
    session
        .validate(cx, &fresh, true)
        .expect("cold filtered admission");
    let cold_ms = cold_started.elapsed().as_secs_f64() * 1e3;
    // False-positive rate over a larger absent sample than one batch.
    let absent: Vec<String> = (0..FALSE_POSITIVE_SAMPLE)
        .map(|ordinal| format!("absent-{ordinal:08}:never-published"))
        .collect();
    let absent_ids: Vec<&str> = absent.iter().map(String::as_str).collect();
    let (filter_bytes, filter_keys, answers_absence, passed) = session
        .filter_receipt(cx, &absent_ids)
        .expect("filter receipt");
    assert!(answers_absence, "{}: filter must answer misses", shape.name);

    let session = RefCell::new(session);
    let validate = |filtered: bool| {
        session
            .borrow_mut()
            .validate(cx, black_box(&fresh), filtered)
            .expect("timed admission");
    };
    let null = paired_median_ratio(rounds, 1, || validate(false), || validate(false));
    let lever = paired_median_ratio(rounds, 1, || validate(false), || validate(true));
    let absolute_calls = if shape.segments >= 512 { 4 } else { 16 };
    let incumbent_ns = time_per_doc_ns(absolute_calls, || validate(false));
    let filtered_ns = time_per_doc_ns(absolute_calls, || validate(true));
    eprintln!(
        "[cell] shape={} segments={} docs={} batch={PROBE_BATCH_DOCS} \
         null={:.4} [{:.4}, {:.4}] lever(filtered/incumbent)={:.4} [{:.4}, {:.4}] \
         ci95=[{:.4}, {:.4}] decidable={} incumbent_ns_per_doc={incumbent_ns:.1} \
         filtered_ns_per_doc={filtered_ns:.1} cold_sync_ms={cold_ms:.2} \
         filter_bytes={filter_bytes} filter_keys={filter_keys} \
         false_positive_passes={passed}/{FALSE_POSITIVE_SAMPLE}",
        shape.name,
        shape.segments,
        shape.segments * shape.docs_per_segment,
        null.median,
        null.p5,
        null.p95,
        lever.median,
        lever.p5,
        lever.p95,
        lever.median_ci95_low,
        lever.median_ci95_high,
        lever.decidable_against(&null),
    );
    run_upsert_cells(cx, shape, rounds, &fresh, session.into_inner());
}

/// Upsert's pre-admission identity work: filtered vs incumbent, per batch.
fn run_upsert_cells(
    cx: &Cx,
    shape: &Shape,
    rounds: usize,
    fresh: &[IndexableDocument],
    session: &mut BenchAdmissionSession,
) {
    // Published identifiers spread over every segment, all distinct.
    let replacing: Vec<IndexableDocument> = (0..PROBE_BATCH_DOCS)
        .map(|ordinal| {
            fixture_document(
                ordinal % shape.segments,
                ordinal / shape.segments % shape.docs_per_segment,
            )
        })
        .collect();
    let mut mixed: Vec<IndexableDocument> = fresh[..PROBE_BATCH_DOCS / 2].to_vec();
    mixed.extend_from_slice(&replacing[..PROBE_BATCH_DOCS / 8]);

    // Parity before timing: identical replacement answer, staged MANIFEST and
    // replaced identifiers.
    for (batch, replaced) in [
        (fresh, 0),
        (replacing.as_slice(), PROBE_BATCH_DOCS),
        (mixed.as_slice(), PROBE_BATCH_DOCS / 8),
    ] {
        let incumbent = session
            .stage_upsert(cx, batch, false)
            .expect("incumbent upsert staging");
        let filtered = session
            .stage_upsert(cx, batch, true)
            .expect("filtered upsert staging");
        assert!(
            incumbent == filtered,
            "{}: upsert staging diverged",
            shape.name
        );
        assert_eq!(
            incumbent.0,
            replaced != 0,
            "{}: replacement answer",
            shape.name
        );
        assert_eq!(incumbent.2.len(), replaced, "{}: replaced ids", shape.name);
    }

    let session = RefCell::new(session);
    let stage = |batch: &[IndexableDocument], filtered: bool| {
        black_box(
            session
                .borrow_mut()
                .stage_upsert(cx, black_box(batch), filtered)
                .expect("timed upsert staging"),
        );
    };
    for (name, batch) in [("fresh", fresh), ("replace", replacing.as_slice())] {
        let null = paired_median_ratio(rounds, 1, || stage(batch, false), || stage(batch, false));
        let lever = paired_median_ratio(rounds, 1, || stage(batch, false), || stage(batch, true));
        let absolute_calls = if shape.segments >= 512 { 2 } else { 8 };
        let incumbent_ns = time_per_doc_ns(absolute_calls, || stage(batch, false));
        let filtered_ns = time_per_doc_ns(absolute_calls, || stage(batch, true));
        eprintln!(
            "[upsert] shape={} segments={} docs={} batch={name}x{PROBE_BATCH_DOCS} \
             null={:.4} [{:.4}, {:.4}] lever(filtered/incumbent)={:.4} [{:.4}, {:.4}] \
             ci95=[{:.4}, {:.4}] decidable={} incumbent_ns_per_doc={incumbent_ns:.1} \
             filtered_ns_per_doc={filtered_ns:.1}",
            shape.name,
            shape.segments,
            shape.segments * shape.docs_per_segment,
            null.median,
            null.p5,
            null.p95,
            lever.median,
            lever.p5,
            lever.p95,
            lever.median_ci95_low,
            lever.median_ci95_high,
            lever.decidable_against(&null),
        );
    }
}

fn main() {
    let scale = std::env::var("QUILL_ADMIT_SCALE").unwrap_or_else(|_| "full".to_owned());
    let rounds: usize = std::env::var("QUILL_ADMIT_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(21);
    let shapes: &'static [Shape] = if scale == "smoke" {
        &SMOKE_SHAPES
    } else {
        &FULL_SHAPES
    };
    eprintln!("[harness] scale={scale} rounds={rounds} batch={PROBE_BATCH_DOCS}");
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for shape in shapes {
            let built_at = Instant::now();
            let index = build_index(&cx, shape).await;
            eprintln!(
                "[setup] shape={} segments={} docs={} build_ms={:.1}",
                shape.name,
                shape.segments,
                shape.segments * shape.docs_per_segment,
                built_at.elapsed().as_secs_f64() * 1_000.0,
            );
            let mut session = index
                .bench_admission_session(&cx)
                .await
                .expect("bench admission session");
            run_shape(&cx, shape, rounds, &mut session);
        }
    });
}
