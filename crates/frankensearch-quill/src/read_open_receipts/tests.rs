use super::*;

// The store's clock is optional, so the fixture returns what `load_at` takes.
#[cfg(target_os = "linux")]
#[allow(clippy::unnecessary_wraps)]
fn clock(unix_s: i64, boot_s: u64) -> Option<Clock> {
    Some(Clock {
        unix_ns: i128::from(unix_s) * i128::from(NANOS_PER_SECOND),
        boot_ns: boot_s * NANOS_PER_SECOND,
    })
}

fn fixture() -> (Binding, Identity, Receipt) {
    let binding = Binding {
        schema_id: 7,
        segment_id: 8,
        file_len: 100,
        file_xxh3: 9,
    };
    let identity = Identity {
        dev: 1,
        ino: 2,
        len: 100,
        mtime_s: 100,
        mtime_ns: 3,
        ctime_s: 100,
        ctime_ns: 4,
    };
    (
        binding,
        identity,
        Receipt {
            binding,
            identity,
            verified_boot_ns: 200 * NANOS_PER_SECOND,
        },
    )
}

fn encoded_fixture() -> Vec<u8> {
    let (binding, _, receipt) = fixture();
    encode(&BTreeMap::from([(binding, receipt)]), [42; 32]).expect("encode")
}

fn seal(body: &[u8]) -> Vec<u8> {
    let mut bytes = body.to_vec();
    bytes.extend_from_slice(&Sha256::digest(body));
    bytes
}

#[test]
fn round_trip_binds_schema_identity_and_producer() {
    let (binding, _, receipt) = fixture();
    let bytes = encoded_fixture();
    assert_eq!(
        decode(&bytes, [42; 32]),
        Some(BTreeMap::from([(binding, receipt)]))
    );
    assert!(decode(&bytes, [43; 32]).is_none());
}

#[test]
fn every_single_byte_corruption_is_rejected() {
    let good = encoded_fixture();
    for offset in 0..good.len() {
        let mut bad = good.clone();
        bad[offset] ^= 1;
        assert!(decode(&bad, [42; 32]).is_none(), "byte {offset}");
    }
}

#[test]
fn every_truncation_and_oversized_book_is_rejected() {
    let good = encoded_fixture();
    for end in 0..good.len() {
        assert!(decode(&good[..end], [42; 32]).is_none());
    }
    assert!(decode(&vec![0; MAX_BYTES + 1], [42; 32]).is_none());
    let mut extended = good;
    extended.push(0);
    assert!(decode(&extended, [42; 32]).is_none());
}

#[test]
fn checksummed_invalid_fields_and_duplicate_records_fail_closed() {
    let bytes = encoded_fixture();
    let body = &bytes[..bytes.len() - CHECKSUM_BYTES];
    let mut duplicate = body.to_vec();
    duplicate[40..44].copy_from_slice(&2_u32.to_le_bytes());
    duplicate.extend_from_slice(&body[HEADER_BYTES..]);
    assert!(decode(&seal(&duplicate), [42; 32]).is_none());
    for (word, value) in [(6, 101_u64), (8, 1_000_000_000), (10, u64::MAX)] {
        let mut bad = body.to_vec();
        bad[HEADER_BYTES + word * 8..HEADER_BYTES + (word + 1) * 8]
            .copy_from_slice(&value.to_le_bytes());
        assert!(decode(&seal(&bad), [42; 32]).is_none());
    }
    let mut foreign = body.to_vec();
    foreign[7] = b'2';
    assert!(decode(&seal(&foreign), [42; 32]).is_none());
}

#[cfg(target_os = "linux")]
fn private_dir() -> tempfile::TempDir {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).expect("chmod");
    dir
}

#[cfg(target_os = "linux")]
#[test]
fn boot_expiry_cannot_be_extended_by_wall_clock_rollback_or_cache_hits() {
    let dir = private_dir();
    let (binding, identity, _) = fixture();
    let policy = Policy {
        max_age: Duration::from_secs(60),
        minimum_file_age: Duration::from_secs(10),
    };
    let mut first = ReceiptBook::load_at(dir.path(), policy, clock(1000, 200));
    first.keep(
        first
            .verified(binding, identity, Some(identity))
            .expect("receipt"),
    );
    first.persist();
    // Wall time moves backward but the one-minute boot-time lease never renews.
    for (wall, boot) in [(1000, 200), (900, 259), (800, 260)] {
        let mut hit = ReceiptBook::load_at(dir.path(), policy, clock(wall, boot));
        let receipt = hit
            .admitted_at(binding, identity, clock(wall, boot))
            .expect("hit");
        assert_eq!(receipt.verified_boot_ns, 200 * NANOS_PER_SECOND);
        hit.keep(receipt);
        hit.persist();
    }
    for (wall, boot) in [(1000, 199), (800, 261), (100_000, 261)] {
        let miss = ReceiptBook::load_at(dir.path(), policy, clock(wall, boot));
        assert!(
            miss.admitted_at(binding, identity, clock(wall, boot))
                .is_none()
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn every_identity_and_manifest_component_is_required() {
    let dir = private_dir();
    let (binding, identity, _) = fixture();
    let mut first = ReceiptBook::load_at(dir.path(), Policy::default(), clock(200, 200));
    first.keep(
        first
            .verified(binding, identity, Some(identity))
            .expect("receipt"),
    );
    first.persist();
    let book = ReceiptBook::load_at(dir.path(), Policy::default(), clock(201, 201));
    assert!(book.admitted_at(binding, identity, book.clock).is_some());
    for changed in [
        Identity { dev: 9, ..identity },
        Identity { ino: 9, ..identity },
        Identity { len: 9, ..identity },
        Identity {
            mtime_s: 99,
            ..identity
        },
        Identity {
            mtime_ns: 9,
            ..identity
        },
        Identity {
            ctime_s: 99,
            ..identity
        },
        Identity {
            ctime_ns: 9,
            ..identity
        },
    ] {
        assert!(book.admitted_at(binding, changed, book.clock).is_none());
    }
    for changed in [
        Binding {
            schema_id: 99,
            ..binding
        },
        Binding {
            segment_id: 99,
            ..binding
        },
        Binding {
            file_len: 99,
            ..binding
        },
        Binding {
            file_xxh3: 99,
            ..binding
        },
    ] {
        assert!(book.admitted_at(changed, identity, book.clock).is_none());
    }
    assert!(book.verified(binding, identity, None).is_none());
    assert!(
        book.verified(
            binding,
            identity,
            Some(Identity {
                ctime_ns: 9,
                ..identity
            })
        )
        .is_none()
    );
}

#[cfg(target_os = "linux")]
#[test]
fn racy_files_unknown_clock_and_untrusted_directory_cannot_mint() {
    use std::os::unix::fs::PermissionsExt;
    let dir = private_dir();
    let (binding, identity, _) = fixture();
    for now in [clock(100, 200), clock(159, 200), None] {
        let book = ReceiptBook::load_at(dir.path(), Policy::default(), now);
        assert!(book.verified(binding, identity, Some(identity)).is_none());
    }
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).expect("chmod");
    assert!(!ReceiptBook::load(dir.path(), Policy::default()).enabled());
}

#[cfg(target_os = "linux")]
#[test]
fn namespace_or_boot_producer_change_invalidates_proofs() {
    let dir = private_dir();
    let (binding, identity, _) = fixture();
    let mut first = ReceiptBook::load_at(dir.path(), Policy::default(), clock(200, 200));
    first.keep(
        first
            .verified(binding, identity, Some(identity))
            .expect("receipt"),
    );
    let producer = first.producer;
    let mut foreign = producer;
    foreign[0] ^= 1;
    write_book(
        first.directory.as_ref().expect("directory"),
        &encode(&first.kept, foreign).expect("foreign encoding"),
    )
    .expect("write");
    let reopened = ReceiptBook::load_at(dir.path(), Policy::default(), clock(201, 201));
    assert!(reopened.enabled());
    assert!(
        reopened
            .admitted_at(binding, identity, reopened.clock)
            .is_none()
    );
}

#[cfg(target_os = "linux")]
#[test]
fn symlink_hardlink_world_writable_and_fifo_books_are_not_read() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let dir = private_dir();
    let file = dir.path().join(BOOK);
    std::fs::write(&file, encoded_fixture()).expect("write");
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let handle = open_private_directory(dir.path()).expect("open directory");
    assert!(read_book(&handle).is_ok());
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o666)).expect("chmod");
    assert!(read_book(&handle).is_err());
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let other = dir.path().join("hard-linked");
    std::fs::hard_link(&file, &other).expect("hard link");
    assert!(read_book(&handle).is_err());
    std::fs::rename(&file, dir.path().join("original-book")).expect("retain original fixture");
    symlink(&other, &file).expect("symlink");
    assert!(read_book(&handle).is_err());
    std::fs::rename(&file, dir.path().join("symlink-book")).expect("retain symlink fixture");
    rustix::fs::mknodat(
        &handle,
        BOOK,
        rustix::fs::FileType::Fifo,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        0,
    )
    .expect("fifo");
    assert!(read_book(&handle).is_err()); // NONBLOCK prevents hanging.
}

#[cfg(target_os = "linux")]
#[test]
fn descriptor_identity_detects_replacement_truncation_and_restored_mtime() {
    use std::io::{Seek, SeekFrom};
    let dir = private_dir();
    let path = dir.path().join("segment");
    std::fs::write(&path, b"0123456789").expect("write");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("open");
    let before = Identity::of_file(&file).expect("identity");
    let mtime = file
        .metadata()
        .expect("metadata")
        .modified()
        .expect("mtime");
    std::thread::sleep(Duration::from_millis(20));
    file.seek(SeekFrom::Start(5)).expect("seek");
    file.write_all(b"X").expect("write");
    file.set_times(std::fs::FileTimes::new().set_modified(mtime))
        .expect("restore mtime");
    let rewritten = Identity::of_file(&file).expect("rewritten");
    assert_eq!(before.len, rewritten.len);
    assert_eq!(
        (before.mtime_s, before.mtime_ns),
        (rewritten.mtime_s, rewritten.mtime_ns)
    );
    assert_ne!(
        (before.ctime_s, before.ctime_ns),
        (rewritten.ctime_s, rewritten.ctime_ns)
    );
    file.set_len(5).expect("truncate");
    assert_ne!(Identity::of_file(&file), Some(rewritten));
    let replacement = dir.path().join("replacement");
    std::fs::write(&replacement, b"0123456789").expect("replacement");
    std::fs::rename(&replacement, &path).expect("rename");
    let new = File::open(&path).expect("new handle");
    assert_ne!(
        Identity::of_file(&new).expect("new identity").ino,
        before.ino
    );
    assert_eq!(
        Identity::of_file(&file).expect("retained inode").ino,
        before.ino
    );
}

#[cfg(target_os = "linux")]
#[test]
fn receipts_expire_during_one_open_without_reloading_the_book() {
    let dir = private_dir();
    let (binding, identity, _) = fixture();
    let policy = Policy {
        max_age: Duration::from_secs(60),
        minimum_file_age: Duration::from_secs(10),
    };
    let mut first = ReceiptBook::load_at(dir.path(), policy, clock(1000, 200));
    first.keep(
        first
            .verified(binding, identity, Some(identity))
            .expect("fully verified receipt"),
    );
    first.persist();
    let mut book = ReceiptBook::load_at(dir.path(), policy, clock(1000, 250));
    assert!(book.enabled());

    // Simulate later segment workers using the SAME already-loaded book.
    // Backward wall time cannot renew its original boot-time deadline.
    for now in [clock(1000, 250), clock(900, 259), clock(800, 260)] {
        let receipt = book
            .admitted_at(binding, identity, now)
            .expect("live lease");
        assert_eq!(receipt.verified_boot_ns, 200 * NANOS_PER_SECOND);
        book.keep(receipt);
    }
    let mut just_expired = clock(800, 260).expect("boundary clock");
    just_expired.boot_ns += 1;
    for now in [
        Some(just_expired),
        clock(800, 261),
        clock(100_000, 261),
        clock(1000, 249), // Boot time must not regress behind this open.
        clock(109, 260),  // Minimum file age also uses the decision-time clock.
        None,
    ] {
        assert!(
            book.admitted_at(binding, identity, now).is_none(),
            "stale, regressed, too-young or unavailable clock: {now:?}"
        );
    }
    book.persist();
    let reopened = ReceiptBook::load_at(dir.path(), policy, clock(800, 261));
    assert_eq!(
        reopened.loaded[&binding].verified_boot_ns,
        200 * NANOS_PER_SECOND,
        "keeping cache hits must not renew the persisted proof"
    );
    assert!(
        reopened
            .admitted_at(binding, identity, reopened.clock)
            .is_none()
    );
}

#[cfg(target_os = "linux")]
#[test]
fn production_admission_samples_the_clock_instead_of_the_load_timestamp() {
    let dir = private_dir();
    let (binding, identity, _) = fixture();
    let now = Clock::now().expect("Linux boot clock");
    let loaded_at = Some(Clock {
        boot_ns: now.boot_ns.checked_sub(1).expect("positive host uptime"),
        ..now
    });
    let policy = Policy {
        max_age: Duration::ZERO,
        minimum_file_age: Duration::ZERO,
    };
    let mut first = ReceiptBook::load_at(dir.path(), policy, loaded_at);
    first.keep(
        first
            .verified(binding, identity, Some(identity))
            .expect("fully verified receipt"),
    );
    first.persist();
    let mut book = ReceiptBook::load_at(dir.path(), policy, loaded_at);
    assert!(book.admitted_at(binding, identity, loaded_at).is_some());
    // No sleep: the receipt was already older than the real boot clock before
    // the book was loaded. The old implementation admits it indefinitely.
    assert!(
        book.admitted(binding, identity).is_none(),
        "the production decision must not use the frozen load timestamp"
    );
    book.policy.max_age = Duration::from_secs(3600);
    assert!(
        book.admitted(binding, identity).is_some(),
        "a fresh-clock lookup must still reuse a proof within its lease"
    );
}

fn retention_fixture(ordinal: usize) -> Receipt {
    let (_, _, mut receipt) = fixture();
    let ordinal = u64::try_from(ordinal).expect("fixture ordinal");
    receipt.binding.segment_id = ordinal;
    receipt.binding.file_len = ordinal % 29 + 1;
    receipt.identity.ino = ordinal + 1;
    receipt.identity.len = receipt.binding.file_len;
    receipt
}

#[test]
fn retention_is_bounded_cost_prioritized_and_order_independent() {
    let count = MAX_RECEIPTS + 37;
    let candidates: Vec<_> = (0..count).map(retention_fixture).collect();
    // A full sort is an independent oracle for the bounded streaming selector.
    let mut ranked = candidates.clone();
    ranked.sort_unstable_by_key(|receipt| {
        std::cmp::Reverse((receipt.binding.file_len, receipt.binding))
    });
    ranked.truncate(MAX_RECEIPTS);
    let expected: BTreeMap<_, _> = ranked
        .into_iter()
        .map(|receipt| (receipt.binding, receipt))
        .collect();
    let orders: [Vec<usize>; 3] = [
        (0..count).collect(),
        (0..count).rev().collect(),
        (0..count).step_by(2).chain((1..count).step_by(2)).collect(),
    ];
    for order in orders {
        // No directory or platform clock is needed to exercise retention.
        let mut book = ReceiptBook::load_at(Path::new("unused-cache"), Policy::default(), None);
        for ordinal in order {
            book.keep(candidates[ordinal].clone());
            assert!(book.kept.len() <= MAX_RECEIPTS);
            assert_eq!(book.kept.len(), book.kept_by_cost.len());
        }
        assert_eq!(book.kept, expected);
        assert_eq!(
            book.kept_by_cost,
            expected
                .keys()
                .map(|binding| (binding.file_len, *binding))
                .collect::<BTreeSet<_>>()
        );
    }
}

#[test]
fn capacity_overflow_still_encodes_a_complete_v3_book() {
    let uncapped: BTreeMap<_, _> = (0..=MAX_RECEIPTS)
        .map(retention_fixture)
        .map(|receipt| (receipt.binding, receipt))
        .collect();
    assert!(
        encode(&uncapped, [42; 32]).is_none(),
        "the old keep-all policy cannot persist this snapshot"
    );
    let mut book = ReceiptBook::load_at(Path::new("unused-cache"), Policy::default(), None);
    for receipt in uncapped.into_values() {
        book.keep(receipt);
    }
    assert_eq!(book.kept.len(), MAX_RECEIPTS);
    let bytes = encode(&book.kept, [42; 32]).expect("bounded book must encode");
    assert_eq!(
        bytes.len(),
        HEADER_BYTES + MAX_RECEIPTS * RECORD_BYTES + CHECKSUM_BYTES
    );
    assert!(bytes.len() <= MAX_BYTES);
    assert!(bytes.len() + RECORD_BYTES > MAX_BYTES);
    assert_eq!(&bytes[..8], MAGIC);
    assert_eq!(decode(&bytes, [42; 32]), Some(book.kept));
}

#[test]
fn replacing_a_retained_proof_neither_leaks_priority_entries_nor_renews_other_proofs() {
    let mut book = ReceiptBook::load_at(Path::new("unused-cache"), Policy::default(), None);
    for ordinal in 0..MAX_RECEIPTS {
        book.keep(retention_fixture(ordinal));
    }
    let mut expected = book.kept.clone();
    let mut replacement = expected.values().next().expect("full book").clone();
    replacement.identity.ino += 100;
    replacement.verified_boot_ns -= 1;
    expected.insert(replacement.binding, replacement.clone());
    for _ in 0..16 {
        book.keep(replacement.clone());
        assert_eq!(book.kept.len(), MAX_RECEIPTS);
        assert_eq!(book.kept_by_cost.len(), MAX_RECEIPTS);
    }
    assert_eq!(book.kept, expected);

    // Neither extreme length overflows a cost sum: costs are ordered, not added.
    let mut expensive = retention_fixture(MAX_RECEIPTS);
    expensive.binding.file_len = u64::MAX;
    expensive.identity.len = u64::MAX;
    book.keep(expensive.clone());
    assert_eq!(book.kept.get(&expensive.binding), Some(&expensive));
    assert_eq!(book.kept.len(), MAX_RECEIPTS);
    assert_eq!(book.kept_by_cost.len(), MAX_RECEIPTS);
    let before_miss = book.kept.clone();
    let mut cheap = retention_fixture(MAX_RECEIPTS + 1);
    cheap.binding.file_len = 0;
    cheap.identity.len = 0;
    book.keep(cheap);
    assert_eq!(book.kept, before_miss);
    assert_eq!(book.kept_by_cost.len(), MAX_RECEIPTS);
}

#[cfg(target_os = "linux")]
#[test]
fn oversized_snapshots_persist_hits_and_new_proofs_without_renewing_old_leases() {
    let dir = private_dir();
    let policy = Policy {
        max_age: Duration::from_secs(60),
        minimum_file_age: Duration::from_secs(10),
    };
    let mut first = ReceiptBook::load_at(dir.path(), policy, clock(1000, 200));
    assert!(first.enabled());
    let candidates: Vec<_> = (0..=MAX_RECEIPTS)
        .map(|ordinal| {
            let mut receipt = retention_fixture(ordinal);
            receipt.binding.file_len = u64::try_from(ordinal).expect("ordinal") + 100;
            receipt.identity.len = receipt.binding.file_len;
            receipt
        })
        .collect();
    for receipt in &candidates {
        first.keep(
            first
                .verified(receipt.binding, receipt.identity, Some(receipt.identity))
                .expect("fully checked fixture"),
        );
    }
    assert_eq!(first.kept.len(), MAX_RECEIPTS);
    first.persist();

    let mut reopened = ReceiptBook::load_at(dir.path(), policy, clock(1000, 250));
    assert_eq!(reopened.loaded.len(), MAX_RECEIPTS);
    let cheap = &candidates[0];
    let expensive = candidates.last().expect("nonempty snapshot");
    assert!(
        reopened
            .admitted_at(cheap.binding, cheap.identity, reopened.clock)
            .is_none()
    );
    assert!(
        reopened
            .admitted_at(expensive.binding, expensive.identity, reopened.clock)
            .is_some()
    );
    for receipt in reopened.loaded.clone().into_values() {
        let hit = reopened
            .admitted_at(receipt.binding, receipt.identity, reopened.clock)
            .expect("retained live proof");
        assert_eq!(hit.verified_boot_ns, 200 * NANOS_PER_SECOND);
        reopened.keep(hit);
    }
    // A new fully verified segment must still update an already-full book.
    let mut added = retention_fixture(MAX_RECEIPTS + 1);
    added.binding.file_len = u64::MAX;
    added.identity.len = u64::MAX;
    reopened.keep(
        reopened
            .verified(added.binding, added.identity, Some(added.identity))
            .expect("new fully checked fixture"),
    );
    reopened.persist();

    let expired = ReceiptBook::load_at(dir.path(), policy, clock(1000, 261));
    assert_eq!(expired.loaded.len(), MAX_RECEIPTS);
    assert_eq!(
        expired.loaded[&expensive.binding].verified_boot_ns,
        200 * NANOS_PER_SECOND
    );
    assert!(
        expired
            .admitted_at(expensive.binding, expensive.identity, expired.clock)
            .is_none()
    );
    let fresh = expired
        .admitted_at(added.binding, added.identity, expired.clock)
        .expect("only the freshly verified segment has a later lease");
    assert_eq!(fresh.verified_boot_ns, 250 * NANOS_PER_SECOND);
}
