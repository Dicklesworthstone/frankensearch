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
            "frankensearch-index-repro-{name}-{}-{now}.fsvi",
            std::process::id()
        ))
    }

    #[test]
    fn soft_delete_rolls_back_main_index_on_wal_failure() {
        // This test simulates a WAL failure by making the WAL file read-only *after* the index is opened but *before* soft_delete is called.
        // Note: This simulation might be OS-dependent. A more robust way would be to inject a fault into the WAL writer, but that requires internal mocking.
        // For now, we'll try a filesystem-level trigger.

        let path = temp_index_path("soft-delete-rollback");
        let mut writer =
            VectorIndex::create_with_revision(&path, "hash", "test", 4, Quantization::F16)
                .expect("writer");
        writer
            .write_record("doc-a", &[1.0, 0.0, 0.0, 0.0])
            .expect("write doc-a");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");

        // Sanity check: doc-a is live
        let idx = index
            .find_index_by_doc_id("doc-a")
            .expect("find")
            .expect("some");
        assert!(!index.is_deleted(idx));

        // Locate the WAL file
        let wal_path = crate::wal::wal_path_for(&path);

        // Create a WAL entry for the same doc so soft_delete must rewrite the WAL sidecar.
        index
            .append("doc-a", &[0.0, 1.0, 0.0, 0.0])
            .expect("append doc-a wal entry");
        assert!(wal_path.exists());

        // Replace the WAL file with a directory to force a deterministic write failure.
        // This avoids platform/user differences around read-only permission semantics.
        fs::remove_file(&wal_path).expect("remove wal file");
        fs::create_dir(&wal_path).expect("create wal path directory");

        // Attempt soft_delete. It should fail due to WAL write error.
        let result = index.soft_delete("doc-a");

        assert!(
            result.is_err(),
            "soft_delete should fail when WAL is unwritable"
        );

        // CRITICAL CHECK: The main index tombstone should have been rolled back.
        let idx_after = index
            .find_index_by_doc_id("doc-a")
            .expect("find")
            .expect("some");
        assert!(
            !index.is_deleted(idx_after),
            "doc-a should NOT be deleted after failed soft_delete"
        );

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&wal_path);
    }
}
