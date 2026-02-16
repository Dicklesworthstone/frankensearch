#[cfg(test)]
mod tests {
    use crate::wal::{append_wal_batch, read_wal, wal_path_for, WalEntry};
    use crate::Quantization;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_wal_path(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-wal-repro-{name}-{}-{now}.fsvi.wal",
            std::process::id()
        ))
    }

    fn make_entry(doc_id: &str, base: f32, dim: usize) -> WalEntry {
        WalEntry {
            doc_id: doc_id.to_owned(),
            doc_id_hash: crate::fnv1a_hash(doc_id.as_bytes()),
            embedding: vec![base; dim],
        }
    }

    #[test]
    fn wal_append_after_corruption_orphans_new_data() {
        let path = temp_wal_path("append-after-corruption");
        let dim = 4;

        // 1. Write Batch A (valid)
        append_wal_batch(
            &path,
            &[make_entry("doc-a", 1.0, dim)],
            dim,
            Quantization::F16,
            0,
            true,
        )
        .unwrap();

        // 2. Simulate a partial write (Batch B crashed halfway)
        let valid_len = fs::metadata(&path).unwrap().len();
        
        // Append a valid batch...
        append_wal_batch(
            &path,
            &[make_entry("doc-b", 2.0, dim)],
            dim,
            Quantization::F16,
            0,
            true,
        )
        .unwrap();
        
        // ...then truncate it to be corrupt/partial
        let full_len = fs::metadata(&path).unwrap().len();
        let corrupted_len = valid_len + (full_len - valid_len) / 2;
        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(corrupted_len).unwrap();
        drop(file);

        // Verify read_wal stops at Batch A and reports valid length
        let (loaded, _, reported_len) = read_wal(&path, dim, Quantization::F16).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].doc_id, "doc-a");
        assert_eq!(reported_len, valid_len);

        // 3. Open the index to trigger auto-truncation
        // We can't easily use VectorIndex::open here without a full FSVI file.
        // Instead, we can simulate what VectorIndex::open does:
        if fs::metadata(&path).unwrap().len() > reported_len {
            let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.set_len(reported_len).unwrap();
        }

        // 4. Write Batch C (valid)
        // This should now append immediately after Batch A, overwriting the garbage.
        append_wal_batch(
            &path,
            &[make_entry("doc-c", 3.0, dim)],
            dim,
            Quantization::F16,
            0,
            true,
        )
        .unwrap();

        // 5. Read WAL again
        // Expected behavior (FIXED): Batch A and Batch C are visible.
        let (loaded_final, _, _) = read_wal(&path, dim, Quantization::F16).unwrap();
        
        assert_eq!(loaded_final.len(), 2, "Batch C should be visible after auto-truncation");
        assert_eq!(loaded_final[0].doc_id, "doc-a");
        assert_eq!(loaded_final[1].doc_id, "doc-c");

        fs::remove_file(&path).ok();
    }
}
