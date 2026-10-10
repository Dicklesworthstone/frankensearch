//! Persisted WAL/main visibility regressions for the truncated MRL lane.
//! Control vectors exercise retrieval contracts, not model-quality evidence.

use std::fs;

use frankensearch_core::filter::{BitsetFilter, PredicateFilter, SearchFilter};

use super::*;
use crate::Quantization;
use crate::test_fixtures::TempFixturePath;
use crate::wal::wal_path_for;

fn crash_window(
    quantization: Quantization,
    stale_rows: usize,
    filler_rows: usize,
) -> (TempFixturePath, VectorIndex) {
    let path = TempFixturePath::new("mrl-wal-visibility", "index.fsvi");
    let mut writer = VectorIndex::create_with_revision(
        &path,
        "mrl-visibility-control",
        "r1",
        2,
        quantization,
    )
    .expect("create main index");
    for _ in 0..stale_rows {
        writer.write_record("updated", &[1.0, 0.0]).unwrap();
    }
    writer.write_record("survivor", &[0.5, 0.5]).unwrap();
    writer.write_record("tail", &[-1.0, 0.0]).unwrap();
    for row in 0..filler_rows {
        writer
            .write_record(&format!("filler-{row:05}"), &[-0.5, 0.5])
            .unwrap();
    }
    writer.finish().unwrap();
    let sealed = fs::read(&path).unwrap();
    {
        let mut index = VectorIndex::open(&path).unwrap();
        index.append("updated", &[0.0, 1.0]).unwrap();
        assert_eq!(index.wal_record_count(), 1);
    }
    // Restore the valid pre-tombstone main file only after its mapping closes.
    // The real durable WAL remains intact, exactly as in the crash window.
    fs::write(&path, sealed).unwrap();
    let index = VectorIndex::open_read_only(&path).unwrap();
    assert_eq!(index.wal_record_count(), 1);
    let still_live = (0..index.record_count())
        .filter(|&row| {
            index.doc_id_at(row).unwrap() == "updated" && !index.is_deleted(row)
        })
        .count();
    assert_eq!(still_live, stale_rows, "fixture must expose stale main rows");
    (path, index)
}

fn signature(hits: &[VectorHit]) -> Vec<(String, u32, u32)> {
    hits.iter()
        .map(|hit| (hit.doc_id.to_string(), hit.score.to_bits(), hit.index))
        .collect()
}

fn config(candidates: usize) -> MrlConfig {
    MrlConfig {
        search_dims: 1,
        rescore_dims: 0,
        rescore_top_k: candidates,
    }
}

#[test]
fn persisted_replacements_cannot_consume_mrl_candidate_or_result_slots() {
    for quantization in [Quantization::F16, Quantization::F32] {
        let (_path, index) = crash_window(quantization, 1, 0);
        let query = [1.0, 0.0];
        let all = index.search_top_k(&query, 4, None).unwrap();
        assert_eq!(
            all.iter().map(|hit| hit.doc_id.as_str()).collect::<Vec<_>>(),
            ["survivor", "updated", "tail"]
        );
        for limit in 1..=3 {
            for candidates in [limit, 0, 16] {
                let (hits, stats) = index
                    .mrl_search_with_stats(&query, limit, &config(candidates), None)
                    .unwrap();
                assert_eq!(signature(&hits), signature(&all[..limit]));
                assert!(!stats.fell_back_to_full);
                assert!(stats.zero_signal.is_none());
                assert_eq!(
                    stats.candidates_rescored,
                    if candidates == 0 { 3 } else { candidates.min(3) }
                );
            }
        }
    }
}

#[test]
fn main_exclusion_preserves_predicates_hash_filters_and_live_wal_replacements() {
    let (_path, index) = crash_window(Quantization::F32, 1, 0);
    let predicate = PredicateFilter::new("non-tail", |id| id != "tail");
    let hash_filter = BitsetFilter::from_doc_ids(["survivor", "updated"]);
    let replacement = BitsetFilter::from_doc_ids(["updated"]);
    let empty = BitsetFilter::from_doc_ids(["absent"]);
    let query = [1.0, 0.0];
    for filter in [
        &predicate as &dyn SearchFilter,
        &hash_filter,
        &replacement,
        &empty,
    ] {
        let expected = index.search_top_k(&query, 4, Some(filter)).unwrap();
        for limit in 1..=3 {
            let (actual, stats) = index
                .mrl_search_with_stats(&query, limit, &config(limit), Some(filter))
                .unwrap();
            assert_eq!(
                signature(&actual),
                signature(&expected[..limit.min(expected.len())])
            );
            assert_eq!(stats.zero_signal.is_some(), actual.is_empty());
        }
    }
    let hits = index
        .mrl_search(&query, 1, &config(1), Some(&replacement))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id.as_str(), "updated");
    assert_eq!(hits[0].score.to_bits(), 0.0_f32.to_bits());
    assert!(usize::try_from(hits[0].index).unwrap() >= index.record_count());
}

#[test]
fn one_wal_replacement_excludes_every_stale_main_copy_before_mrl_selection() {
    for quantization in [Quantization::F16, Quantization::F32] {
        let (_path, index) = crash_window(quantization, 8, 0);
        let (hits, stats) = index
            .mrl_search_with_stats(&[1.0, 0.0], 2, &config(2), None)
            .unwrap();
        assert_eq!(
            hits.len(),
            2,
            "WAL-count overfetch cannot repair this fixture"
        );
        assert_eq!(hits[0].doc_id.as_str(), "survivor");
        assert_eq!(hits[1].doc_id.as_str(), "updated");
        assert_eq!(hits[1].score.to_bits(), 0.0_f32.to_bits());
        assert_eq!(stats.candidates_rescored, 2);
    }
}

#[test]
fn parallel_mrl_chunks_apply_the_same_live_main_view() {
    for quantization in [Quantization::F16, Quantization::F32] {
        let (_path, index) = crash_window(quantization, 8, PARALLEL_THRESHOLD + 17);
        assert!(index.record_count() > PARALLEL_THRESHOLD);
        let query = [1.0, 0.0];
        let expected = index.search_top_k(&query, 2, None).unwrap();
        let (hits, stats) = index
            .mrl_search_with_stats(&query, 2, &config(2), None)
            .unwrap();
        assert_eq!(signature(&hits), signature(&expected));
        assert_eq!(stats.candidates_rescored, 2);
        assert_eq!(stats.records_scanned, index.record_count() + 1);
        assert!(!stats.fell_back_to_full);
    }
}

#[test]
fn mrl_visibility_repair_does_not_mutate_main_or_wal_across_reopens() {
    let (path, index) = crash_window(Quantization::F32, 1, 0);
    let wal_path = wal_path_for(&path);
    let main_bytes = fs::read(&path).unwrap();
    let wal_bytes = fs::read(&wal_path).unwrap();
    let query = [1.0, 0.0];
    let expected = index.mrl_search(&query, 3, &config(3), None).unwrap();
    drop(index);
    for _ in 0..3 {
        let index = VectorIndex::open_read_only(&path).unwrap();
        for limit in 1..=3 {
            let hits = index.mrl_search(&query, limit, &config(limit), None).unwrap();
            assert_eq!(signature(&hits), signature(&expected[..limit]));
        }
        drop(index);
        assert_eq!(fs::read(&path).unwrap(), main_bytes);
        assert_eq!(fs::read(&wal_path).unwrap(), wal_bytes);
    }
}

#[test]
fn ordinary_appends_repeated_replacements_and_zero_k_keep_their_semantics() {
    let path = TempFixturePath::new("mrl-wal-ordinary", "index.fsvi");
    let mut writer = VectorIndex::create_with_revision(
        &path,
        "mrl-visibility-control",
        "r1",
        2,
        Quantization::F32,
    )
    .unwrap();
    writer.write_record("main", &[0.5, 0.5]).unwrap();
    writer.finish().unwrap();
    let mut index = VectorIndex::open(&path).unwrap();
    index.append("new", &[1.0, 0.0]).unwrap();
    let query = [1.0, 0.0];
    let hits = index.mrl_search(&query, 2, &config(2), None).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id.as_str(), "new");
    for values in [[0.25, 0.75], [0.75, 0.25]] {
        index.append("main", &values).unwrap();
        let hits = index.mrl_search(&query, 2, &config(2), None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].doc_id.as_str(), "main");
        assert_eq!(hits[1].score.to_bits(), values[0].to_bits());
    }
    let (hits, stats) = index
        .mrl_search_with_stats(&query, 0, &config(0), None)
        .unwrap();
    assert!(hits.is_empty());
    assert_eq!(
        stats.zero_signal,
        Some(ZeroSignalReason::CallerRequestedZeroK)
    );
}
