#![recursion_limit = "256"]
//! Same-binary IDMAP winner-materialization benchmark.
//!
//! The baseline is the incumbent reopened ordinal sidecar shape: a resident
//! `Vec<DocId>` cloned only for the requested winners. The candidate performs
//! two little-endian IDMAP offset reads, borrows the durable UTF-8 slice, and
//! materializes the same `DocId = CompactString`. Inputs and winner outputs are
//! asserted identical before Criterion starts. This isolates the materialization
//! primitive inside the legacy `reopen_id_materialize` query benchmark; query,
//! scoring, and top-k work common to both formats is deliberately excluded.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 RCH_WORKER=<pinned-worker> \
//!   rch exec -- cargo bench --profile release -p frankensearch-quill \
//!   --features bench-internals --bench id_map_materialize_ab
//! ```

use std::fmt::Display;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use frankensearch_core::DocId;
use frankensearch_quill::quiver::{
    EncodedIdMapSection, IdMapEntryInput, IdMapSection, id_map_content_hash,
};

const DOCUMENT_COUNT: usize = 20_000;
const WINNER_COUNTS: [usize; 2] = [10, 1_000];

struct Fixture {
    ids: Vec<DocId>,
    encoded: EncodedIdMapSection,
}

fn fail(stage: &str, error: &impl Display) -> ! {
    eprintln!("id-map-materialize-ab {stage} failed: {error}");
    std::process::exit(2);
}

fn require<T, E: Display>(stage: &str, result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => fail(stage, &error),
    }
}

fn as_u64(stage: &str, value: usize) -> u64 {
    match u64::try_from(value) {
        Ok(value) => value,
        Err(error) => fail(stage, &error),
    }
}

fn build_fixture() -> Fixture {
    let ids = (0..DOCUMENT_COUNT)
        .map(|index| {
            let text = if index % 17 == 0 {
                format!("document-{index:08}-identifier-longer-than-inline")
            } else {
                format!("d{index:08}")
            };
            DocId::new(&text)
        })
        .collect::<Vec<_>>();
    let entries = ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            Some(IdMapEntryInput::new(
                id,
                id_map_content_hash(&index.to_le_bytes()),
            ))
        })
        .collect::<Vec<_>>();
    let encoded = require(
        "encode",
        EncodedIdMapSection::encode(0, as_u64("document count", DOCUMENT_COUNT), &entries),
    );
    Fixture { ids, encoded }
}

fn winner_indices(count: usize) -> Vec<usize> {
    (0..count)
        .map(|index| index.wrapping_mul(7_919) % DOCUMENT_COUNT)
        .collect()
}

fn mapped_winners(section: IdMapSection<'_>, docids: &[u64]) -> Vec<DocId> {
    docids
        .iter()
        .map(|&docid| {
            section.materialize(docid).unwrap_or_else(|| {
                eprintln!("id-map-materialize-ab missing winner docid {docid}");
                std::process::exit(2);
            })
        })
        .collect()
}

fn checksum_id(checksum: u64, id: &DocId) -> u64 {
    let first = id.as_bytes().first().copied().map_or(0, u64::from);
    checksum
        .wrapping_mul(0x0000_0100_0000_01b3)
        .wrapping_add(as_u64("identifier length", id.len()))
        .wrapping_add(first)
}

#[allow(clippy::significant_drop_tightening)]
fn bench_materialization(c: &mut Criterion) {
    let fixture = build_fixture();
    let section = require("open", fixture.encoded.section());
    let mut group = c.benchmark_group("id_map_winner_materialize");
    for count in WINNER_COUNTS {
        let indices = winner_indices(count);
        let docids = indices
            .iter()
            .map(|&index| as_u64("winner docid", index))
            .collect::<Vec<_>>();
        let incumbent = indices
            .iter()
            .map(|&index| fixture.ids[index].clone())
            .collect::<Vec<_>>();
        let mapped = mapped_winners(section, &docids);
        assert_eq!(mapped, incumbent, "IDMAP and incumbent winner IDs differ");
        group.throughput(Throughput::Elements(as_u64("winner count", count)));
        group.bench_with_input(
            BenchmarkId::new("incumbent_vec_clone", count),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut checksum = 0_u64;
                    for &index in indices {
                        let id = black_box(fixture.ids[index].clone());
                        checksum = checksum_id(checksum, &id);
                        black_box(id);
                    }
                    black_box(checksum)
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("idmap_two_offsets_materialize", count),
            &docids,
            |b, docids| {
                b.iter(|| {
                    let mut checksum = 0_u64;
                    for &docid in docids {
                        let id = section.materialize(black_box(docid)).unwrap_or_else(|| {
                            eprintln!("id-map-materialize-ab missing timed winner docid {docid}");
                            std::process::exit(2);
                        });
                        checksum = checksum_id(checksum, &id);
                        black_box(id);
                    }
                    black_box(checksum)
                });
            },
        );
    }
    group.finish();

    let mut reopen = c.benchmark_group("id_map_reopen_validation");
    reopen.throughput(Throughput::Bytes(as_u64(
        "section byte length",
        fixture.encoded.as_bytes().len(),
    )));
    reopen.bench_function("documents_20000", |b| {
        b.iter(|| {
            let section = require(
                "timed reopen",
                IdMapSection::parse(
                    black_box(fixture.encoded.as_bytes()),
                    0,
                    as_u64("document count", DOCUMENT_COUNT),
                ),
            );
            black_box(section.present_count())
        });
    });
    reopen.finish();
}

criterion_group!(benches, bench_materialization);
criterion_main!(benches);
