#[cfg(test)]
mod tests {
    use crate::{Quantization, VectorIndex};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_index_path(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-wal-shadow-repro-{name}-{}-{now}.fsvi",
            std::process::id()
        ))
    }

    /// A WAL append that reuses a sealed record's `doc_id` supersedes that
    /// record: search must score only the live WAL revision, never the stale
    /// main row. The query is aligned exactly with the SUPERSEDED vector, so
    /// a leak is loud — the stale row scores 1.0, the live revision 0.0.
    #[test]
    fn wal_append_shadows_sealed_record_with_same_doc_id() {
        let path = temp_index_path("stale-leak");

        let mut writer =
            VectorIndex::create_with_revision(&path, "test", "r1", 2, Quantization::F32)
                .expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0])
            .expect("write doc-a");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        index
            .append("doc-a", &[0.0, 1.0])
            .expect("append doc-a wal revision");

        let hits = index
            .search_top_k(&[1.0, 0.0], 1, None)
            .expect("search after WAL supersession");
        assert_eq!(hits.len(), 1, "doc-a must remain searchable");
        assert_eq!(hits[0].doc_id, "doc-a");
        assert!(
            hits[0].score.abs() < f32::EPSILON,
            "sealed main record leaked past its WAL supersession: score {}",
            hits[0].score
        );

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(crate::wal::wal_path_for(&path));
    }

    fn crash_window_index(quantization: Quantization) -> (tempfile::TempDir, VectorIndex) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("crash-window.fsvi");
        let mut writer =
            VectorIndex::create_with_revision(&path, "wal-top-k-test", "r1", 2, quantization)
                .unwrap();
        writer.write_record("updated", &[1.0, 0.0]).unwrap();
        writer
            .write_record("survivor", &[0.5, 0.866_025_4])
            .unwrap();
        writer.write_record("tail", &[-1.0, 0.0]).unwrap();
        writer.finish().unwrap();
        let sealed = fs::read(&path).unwrap();
        {
            let mut writer = VectorIndex::open(&path).unwrap();
            writer.append("updated", &[0.0, 1.0]).unwrap();
            assert_eq!(writer.wal_record_count(), 1);
        }
        // Recreate the valid crash window: the append's durable WAL is present,
        // while the main file still has its exact pre-tombstone bytes. No live
        // mapping spans this write, and neither file is artificially corrupted.
        fs::write(&path, &sealed).unwrap();
        let reopened = VectorIndex::open_read_only(&path).unwrap();
        assert_eq!(reopened.wal_record_count(), 1);
        let row = (0..reopened.record_count())
            .find(|&row| reopened.doc_id_at(row).unwrap() == "updated")
            .unwrap();
        assert!(
            !reopened.is_deleted(row),
            "the fixture must expose a live stale main row"
        );
        (directory, reopened)
    }

    fn hit_signature(hits: &[frankensearch_core::VectorHit]) -> Vec<(String, u32, u32)> {
        hits.iter()
            .map(|hit| (hit.doc_id.to_string(), hit.score.to_bits(), hit.index))
            .collect()
    }

    #[test]
    fn persisted_wal_shadow_does_not_consume_top_k_slots() {
        for quantization in [Quantization::F16, Quantization::F32] {
            let (_directory, index) = crash_window_index(quantization);
            let query = [1.0, 0.0];
            let all = index.search_top_k(&query, 4, None).unwrap();
            assert_eq!(
                all.iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                ["survivor", "updated", "tail"]
            );
            assert_eq!(all[1].score.to_bits(), 0.0_f32.to_bits());
            assert!(usize::try_from(all[1].index).unwrap() >= index.record_count());
            for parallel_enabled in [false, true] {
                for limit in 1..=3 {
                    let actual = index
                        .search_top_k_with_params(
                            &query,
                            limit,
                            None,
                            crate::SearchParams {
                                parallel_threshold: 0,
                                parallel_chunk_size: 1,
                                parallel_enabled,
                            },
                        )
                        .unwrap();
                    assert_eq!(hit_signature(&actual), hit_signature(&all[..limit]));
                }
            }
            let classified = index.search_top_k_classified(&query, 1, None).unwrap();
            assert!(classified.zero_signal.is_none());
            assert_eq!(hit_signature(&classified.hits), hit_signature(&all[..1]));
            assert_eq!(
                hit_signature(&index.search_top_k_int8_two_pass(&query, 2, 4).unwrap()),
                hit_signature(&all[..2]),
                "two-pass search must preserve its exact WAL fallback"
            );
        }
    }

    #[test]
    fn wal_shadow_refill_preserves_predicates_hash_gather_and_live_replacement() {
        use frankensearch_core::filter::{BitsetFilter, PredicateFilter, SearchFilter};
        let (_directory, index) = crash_window_index(Quantization::F32);
        let query = [1.0, 0.0];
        let predicate = PredicateFilter::new("non-tail", |id| id != "tail");
        let hashes = BitsetFilter::from_doc_ids(["updated", "survivor"]);
        let replacement_only = BitsetFilter::from_doc_ids(["updated"]);
        let empty = BitsetFilter::from_doc_ids(["absent"]);
        for filter in [
            &predicate as &dyn SearchFilter,
            &hashes,
            &replacement_only,
            &empty,
        ] {
            let all = index.search_top_k(&query, 4, Some(filter)).unwrap();
            for limit in 1..=2 {
                let expected = hit_signature(&all[..limit.min(all.len())]);
                for actual in [
                    index.search_top_k(&query, limit, Some(filter)).unwrap(),
                    index
                        .bench_scan_filtered(&query, limit, Some(filter))
                        .unwrap(),
                    index.bench_gather_filtered(&query, limit, filter).unwrap(),
                    index
                        .search_top_k_with_params(
                            &query,
                            limit,
                            Some(filter),
                            crate::SearchParams {
                                parallel_threshold: 0,
                                parallel_chunk_size: 1,
                                parallel_enabled: true,
                            },
                        )
                        .unwrap(),
                ] {
                    assert_eq!(hit_signature(&actual), expected);
                }
            }
        }
        let replacement = index
            .search_top_k(&query, 1, Some(&replacement_only))
            .unwrap();
        assert_eq!(
            replacement.len(),
            1,
            "the main-only exclusion must not filter the WAL"
        );
        assert_eq!(replacement[0].doc_id.as_str(), "updated");
        assert_eq!(replacement[0].score.to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn reading_the_crash_window_preserves_both_persisted_artifacts() {
        let (directory, index) = crash_window_index(Quantization::F32);
        let path = directory.path().join("crash-window.fsvi");
        let wal_path = crate::wal::wal_path_for(&path);
        let main_before = fs::read(&path).unwrap();
        let wal_before = fs::read(&wal_path).unwrap();
        for _ in 0..3 {
            let hits = index.search_top_k(&[1.0, 0.0], 1, None).unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id.as_str(), "survivor");
        }
        drop(index);
        assert_eq!(fs::read(&path).unwrap(), main_before);
        assert_eq!(fs::read(&wal_path).unwrap(), wal_before);
        let reopened = VectorIndex::open_read_only(&path).unwrap();
        assert_eq!(
            reopened.search_top_k(&[1.0, 0.0], 1, None).unwrap().len(),
            1
        );
    }

    #[test]
    fn ordinary_wal_appends_and_zero_limits_keep_existing_behavior() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ordinary.fsvi");
        let mut writer =
            VectorIndex::create_with_revision(&path, "wal-top-k-test", "r1", 2, Quantization::F32)
                .unwrap();
        writer.write_record("main", &[0.0, 1.0]).unwrap();
        writer.finish().unwrap();
        let mut index = VectorIndex::open(&path).unwrap();
        index.append("new", &[1.0, 0.0]).unwrap();
        let hits = index.search_top_k(&[1.0, 0.0], 1, None).unwrap();
        assert_eq!(hits[0].doc_id.as_str(), "new");
        assert!(index.search_top_k(&[1.0, 0.0], 0, None).unwrap().is_empty());
        // After an ordinary replacement there is no live stale main winner.
        index.append("main", &[1.0, 0.0]).unwrap();
        assert_eq!(index.search_top_k(&[1.0, 0.0], 2, None).unwrap().len(), 2);
    }

    #[test]
    fn one_wal_replacement_excludes_every_superseded_physical_row() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("duplicate-main.fsvi");
        let mut writer =
            VectorIndex::create_with_revision(&path, "wal-top-k-test", "r1", 2, Quantization::F32)
                .unwrap();
        for _ in 0..8 {
            writer.write_record("updated", &[1.0, 0.0]).unwrap();
        }
        writer
            .write_record("survivor", &[0.5, 0.866_025_4])
            .unwrap();
        writer.finish().unwrap();
        let original = fs::read(&path).unwrap();
        {
            let mut index = VectorIndex::open(&path).unwrap();
            index.append("updated", &[0.0, 1.0]).unwrap();
        }
        fs::write(&path, original).unwrap();
        let index = VectorIndex::open_read_only(&path).unwrap();
        assert_eq!(index.wal_record_count(), 1);
        let hits = index.search_top_k(&[1.0, 0.0], 2, None).unwrap();
        assert_eq!(
            hits.len(),
            2,
            "k + WAL-count overfetch is not a sufficient repair"
        );
        assert_eq!(hits[0].doc_id.as_str(), "survivor");
        assert_eq!(hits[1].doc_id.as_str(), "updated");
        assert_eq!(hits[1].score.to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn unrelated_wal_does_not_change_main_duplicate_cutoff_policy() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unchanged-duplicates.fsvi");
        let mut writer =
            VectorIndex::create_with_revision(&path, "wal-top-k-test", "r1", 2, Quantization::F32)
                .unwrap();
        writer.write_record("duplicate", &[1.0, 0.0]).unwrap();
        writer.write_record("duplicate", &[1.0, 0.0]).unwrap();
        writer.write_record("other", &[0.5, 0.866_025_4]).unwrap();
        writer.finish().unwrap();
        let mut index = VectorIndex::open(&path).unwrap();
        let before = index.search_top_k(&[1.0, 0.0], 2, None).unwrap();
        assert_eq!(
            before.len(),
            1,
            "existing main deduplication occurs after top-k"
        );
        index.append("wal-only", &[-1.0, 0.0]).unwrap();
        let after = index.search_top_k(&[1.0, 0.0], 2, None).unwrap();
        assert_eq!(hit_signature(&after), hit_signature(&before));
    }
}
