// Included in keeper::tests to use the real FSLX fixtures and hash counters.
// In-place faults happen only after all mappings have been dropped.

fn local_receipt_fixture() -> Result<(tempfile::TempDir, PathBuf, PathBuf, EncodedSegment), Box<dyn std::error::Error>> {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::Builder::new().prefix("quill-gh501-").tempdir_in("/dev/shm")?;
    let index = root.path().join("index");
    std::fs::create_dir(&index)?;
    let cache = root.path().join("cache");
    std::fs::create_dir(&cache)?;
    std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o700))?;
    let encoded = encoded_identity_test_segment(0xea501, 0, &[Some("receipt-one"), Some("receipt-two")])?;
    let segment = index.join(canonical_segment_name(0xea501));
    std::fs::write(&segment, encoded.as_bytes())?;
    assert!(crate::read_open_receipts::supported(&File::open(&segment)?),
        "positive receipt tests must run on supported local storage");
    write_manifest(&index.join("MANIFEST"),
        &durable_test_manifest(1, vec![manifest_segment(&encoded, 1)]))?;
    Ok((root, index, cache, encoded))
}

fn eager_local_open(index: &Path, cache: &Path) -> Result<KeeperSnapshot, KeeperError> {
    KeeperSnapshot::open_local_receipts_once(index, DEFAULT_SCHEMA, cache,
        crate::read_open_receipts::Policy {
            minimum_file_age: Duration::ZERO,
            ..crate::read_open_receipts::Policy::default()
        })
}

#[test]
fn local_receipts_reuse_prefix_but_strict_reader_and_writer_still_hash() -> TestResult {
    let (_root, index, cache, _) = local_receipt_fixture()?;
    let first = eager_local_open(&index, &cache)?;
    assert_eq!(first.segments()[0].authenticated_file_witness_hash_count(), 1);
    let expected = first.resolve_document_id("receipt-two")?.map(|hit| hit.global_docid);
    drop(first);
    let second = eager_local_open(&index, &cache)?;
    assert_eq!(second.segments()[0].authenticated_file_witness_hash_count(), 0);
    assert_eq!(second.resolve_document_id("receipt-two")?.map(|hit| hit.global_docid), expected);
    drop(second);
    let strict = KeeperSnapshot::open(&index, DEFAULT_SCHEMA)?;
    assert_eq!(strict.segments()[0].authenticated_file_witness_hash_count(), 1);
    drop(strict);
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build()?;
    let writer = runtime.block_on(async {
        let cx = Cx::for_request();
        KeeperSnapshot::open_writer(&cx, index.clone(), DEFAULT_SCHEMA).await
    })?;
    assert_eq!(writer.snapshot.segments()[0].authenticated_file_witness_hash_count(), 1,
        "maintenance writer must ignore read-only receipts");
    Ok(())
}

#[test]
fn local_receipts_reverify_byte_identical_rewrites_even_with_restored_mtime() -> TestResult {
    for restore in [false, true] {
        let (_root, index, cache, encoded) = local_receipt_fixture()?;
        drop(eager_local_open(&index, &cache)?);
        let hit = eager_local_open(&index, &cache)?;
        assert_eq!(hit.segments()[0].authenticated_file_witness_hash_count(), 0);
        drop(hit);
        let path = index.join(canonical_segment_name(0xea501));
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mtime = file.metadata()?.modified()?;
        std::thread::sleep(Duration::from_millis(20));
        file.write_all(encoded.as_bytes())?;
        if restore {
            file.set_times(std::fs::FileTimes::new().set_modified(mtime))?;
            assert_eq!(file.metadata()?.modified()?, mtime);
        }
        drop(file);
        let fresh = eager_local_open(&index, &cache)?;
        // Correct bytes cannot trigger a lazy checksum failure: this counter
        // independently proves that identity invalidation really rehashed.
        assert_eq!(fresh.segments()[0].authenticated_file_witness_hash_count(), 1);
    }
    Ok(())
}

#[test]
fn local_receipts_reject_truncation_rewrite_and_restored_mtime_corruption() -> TestResult {
    for fault in ["truncation", "rewrite", "restored-mtime"] {
        let (_root, index, cache, encoded) = local_receipt_fixture()?;
        drop(eager_local_open(&index, &cache)?);
        let hit = eager_local_open(&index, &cache)?;
        assert_eq!(hit.segments()[0].authenticated_file_witness_hash_count(), 0);
        drop(hit);
        let path = index.join(canonical_segment_name(0xea501));
        if fault == "truncation" {
            OpenOptions::new().write(true).open(&path)?.set_len(64)?;
        } else {
            let offset = encoded.section_entries().iter()
                .find(|entry| entry.kind == SectionKind::TERMDICT).expect("termdict").offset;
            let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
            let modified = file.metadata()?.modified()?;
            std::thread::sleep(Duration::from_millis(20));
            let mut byte = [0_u8];
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut byte)?;
            byte[0] ^= 1;
            file.seek(SeekFrom::Start(offset))?;
            file.write_all(&byte)?;
            if fault == "restored-mtime" {
                file.set_times(std::fs::FileTimes::new().set_modified(modified))?;
                assert_eq!(file.metadata()?.modified()?, modified);
            }
        }
        assert!(eager_local_open(&index, &cache).is_err(), "receipted {fault}");
        assert!(KeeperSnapshot::open(&index, DEFAULT_SCHEMA).is_err(), "strict {fault}");
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build()?;
        let writer = runtime.block_on(async {
            let cx = Cx::for_request();
            KeeperSnapshot::open_writer(&cx, index.clone(), DEFAULT_SCHEMA).await
        });
        assert!(writer.is_err(), "maintenance writer {fault}");
    }
    Ok(())
}

#[test]
fn local_receipts_reverify_replacement_and_bind_retained_readers_to_the_old_inode() -> TestResult {
    let (_root, index, cache, encoded) = local_receipt_fixture()?;
    drop(eager_local_open(&index, &cache)?);
    let retained = eager_local_open(&index, &cache)?;
    assert_eq!(retained.segments()[0].authenticated_file_witness_hash_count(), 0);
    let path = index.join(canonical_segment_name(0xea501));
    let staged = index.join("staged");
    std::fs::write(&staged, encoded.as_bytes())?;
    std::fs::rename(&staged, &path)?;
    let replaced = eager_local_open(&index, &cache)?;
    assert_eq!(replaced.segments()[0].authenticated_file_witness_hash_count(), 1);
    drop(replaced);
    let mut damaged = encoded.as_bytes().to_vec();
    let offset = encoded.section_entries().iter()
        .find(|entry| entry.kind == SectionKind::TERMDICT).expect("termdict").offset;
    damaged[usize::try_from(offset)?] ^= 1;
    std::fs::write(&staged, damaged)?;
    std::fs::rename(&staged, &path)?;
    assert!(retained.resolve_document_id("receipt-two")?.is_some(),
        "an atomic path replacement must not redirect an admitted mapping");
    assert!(eager_local_open(&index, &cache).is_err(),
        "a fresh admission must reject the corrupted replacement");
    Ok(())
}

#[test]
fn local_receipt_proof_damage_falls_back_to_full_hashing() -> TestResult {
    let (_root, index, cache, _) = local_receipt_fixture()?;
    drop(eager_local_open(&index, &cache)?);
    let path = cache.join("read-open-receipts-v3");
    let good = std::fs::read(&path)?;
    for damage in [b"".to_vec(), b"foreign-format".to_vec(), good[..good.len() - 1].to_vec(), {
        let mut flipped = good.clone(); flipped[60] ^= 1; flipped
    }, vec![0; (1 << 20) + 1]] {
        std::fs::write(&path, damage)?;
        let fresh = eager_local_open(&index, &cache)?;
        assert_eq!(fresh.segments()[0].authenticated_file_witness_hash_count(), 1);
        drop(fresh);
        let repaired = eager_local_open(&index, &cache)?;
        assert_eq!(repaired.segments()[0].authenticated_file_witness_hash_count(), 0);
        drop(repaired);
    }
    Ok(())
}

#[test]
fn local_receipts_preserve_lazy_section_failures_under_a_valid_prefix() -> TestResult {
    let (_root, index, cache, _) = local_receipt_fixture()?;
    let encoded = encoded_test_segment(0xea501, 10, 20, 1)?;
    let doclen = encoded.section_entries().iter()
        .find(|entry| entry.kind == SectionKind::DOCLEN).expect("doclen");
    let mut bytes = encoded.as_bytes().to_vec();
    bytes[usize::try_from(doclen.offset)?] ^= 0x80;
    let file_xxh3 = reseal_test_segment_file_witness(&mut bytes)?;
    std::fs::write(index.join(canonical_segment_name(0xea501)), &bytes)?;
    let mut record = manifest_segment(&encoded, 1);
    record.file_xxh3 = file_xxh3;
    write_manifest(&index.join("MANIFEST"), &durable_test_manifest(1, vec![record]))?;
    for expected_hashes in [1, 0] {
        let snapshot = eager_local_open(&index, &cache)?;
        assert_eq!(snapshot.segments()[0].authenticated_file_witness_hash_count(), expected_hashes);
        assert!(matches!(snapshot.segments()[0].section(SectionKind::DOCLEN),
            Err(QuillError::IndexCorrupted { .. })));
    }
    Ok(())
}

#[test]
fn local_receipts_never_write_a_cache_inside_the_index_or_for_racy_files() -> TestResult {
    let (_root, index, cache, _) = local_receipt_fixture()?;
    let inside = index.join("reader-cache");
    std::fs::create_dir(&inside)?;
    let before: Vec<_> = std::fs::read_dir(&index)?.collect::<Result<_, _>>()?;
    let snapshot = eager_local_open(&index, &inside)?;
    assert_eq!(snapshot.segments()[0].authenticated_file_witness_hash_count(), 1);
    assert!(!inside.join("read-open-receipts-v3").exists());
    assert_eq!(std::fs::read_dir(&index)?.count(), before.len());
    drop(snapshot);
    for _ in 0..2 {
        let racy = KeeperSnapshot::open_with_local_receipts(&index, DEFAULT_SCHEMA, &cache)?;
        assert_eq!(racy.segments()[0].authenticated_file_witness_hash_count(), 1);
    }
    assert!(!cache.join("read-open-receipts-v3").exists());
    Ok(())
}

#[test]
fn local_receipts_are_not_persisted_when_another_segment_fails() -> TestResult {
    let (_root, index, cache, encoded) = local_receipt_fixture()?;
    let missing = encoded_identity_test_segment(0xea502, 100, &[Some("not-on-disk")])?;
    write_manifest(&index.join("MANIFEST"), &durable_test_manifest(1,
        vec![manifest_segment(&encoded, 1), manifest_segment(&missing, 2)]))?;
    assert!(eager_local_open(&index, &cache).is_err());
    assert!(!cache.join("read-open-receipts-v3").exists());
    Ok(())
}


#[test]
fn public_readers_share_receipt_policy_on_open_and_refresh() -> TestResult {
    use crate::{QuillConfig, QuillSearchIndex};

    let (_root, index, cache, encoded) = local_receipt_fixture()?;
    // Exercise the actual public default policy, not the eager private helper.
    // ctime cannot safely be backdated. Let this tiny immutable tmpfs fixture
    // pass the production 60-second stability floor; never skip the hit case.
    std::thread::sleep(Duration::from_secs(61));
    let config = QuillConfig {
        read_open_receipt_directory: Some(cache.clone()),
        ..QuillConfig::default()
    };
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread().build()?;
    runtime.block_on(async {
        let cx = Cx::for_request();
        let initial = QuillSearchIndex::open(&cx, &index, config.clone()).await?;
        assert_eq!(initial.authenticated_file_witness_hash_count(), 1);
        assert!(cache.join("read-open-receipts-v3").is_file());

        let explicit = QuillSearchIndex::open_with_schema(
            &cx, &index, DEFAULT_SCHEMA, config.clone(),
        )
        .await?;
        assert_eq!(explicit.authenticated_file_witness_hash_count(), 0);
        let ordinary = QuillSearchIndex::open(&cx, &index, config).await?;
        assert_eq!(ordinary.authenticated_file_witness_hash_count(), 0,
            "default-schema open must not silently bypass the configured receipts");
        let strict = QuillSearchIndex::open(&cx, &index, QuillConfig::default()).await?;
        assert_eq!(strict.authenticated_file_witness_hash_count(), 1,
            "a warm receipt must not bypass an explicitly strict open");
        for reader in [&initial, &explicit, &ordinary, &strict] {
            assert_eq!(reader.keeper_generation(), 1);
            assert_eq!(reader.doc_count()?, 2);
        }

        // Publish a new MANIFEST reusing the immutable segment. No in-place
        // segment write occurs while any reader owns its memory mapping.
        write_manifest(
            &index.join("MANIFEST"),
            &durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]),
        )?;
        for reader in [&explicit, &ordinary] {
            assert!(reader.refresh(&cx).await?);
            assert_eq!(reader.keeper_generation(), 2);
            assert_eq!(reader.authenticated_file_witness_hash_count(), 0,
                "refresh must retain the admission policy of both constructors");
        }
        assert!(strict.refresh(&cx).await?);
        assert_eq!(strict.keeper_generation(), 2);
        assert_eq!(strict.authenticated_file_witness_hash_count(), 1);
        assert_eq!(initial.keeper_generation(), 1,
            "another handle's refresh must not move the retained snapshot");

        // A damaged proof changes neither data nor authority: admission must
        // fully hash the real file and may then replace the advisory proof.
        std::fs::write(cache.join("read-open-receipts-v3"), b"damaged-proof")?;
        write_manifest(
            &index.join("MANIFEST"),
            &durable_test_manifest(3, vec![manifest_segment(&encoded, 1)]),
        )?;
        assert!(ordinary.refresh(&cx).await?);
        assert_eq!(ordinary.keeper_generation(), 3);
        assert_eq!(ordinary.authenticated_file_witness_hash_count(), 1);
        assert_eq!(ordinary.doc_count()?, 2);
        assert!(explicit.refresh(&cx).await?);
        assert_eq!(explicit.authenticated_file_witness_hash_count(), 0);
        assert!(!ordinary.refresh(&cx).await?,
            "an unchanged retained publication still requires no reopen");
        Ok(())
    })
}
