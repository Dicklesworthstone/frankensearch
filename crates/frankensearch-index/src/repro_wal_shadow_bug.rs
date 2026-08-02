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
}
