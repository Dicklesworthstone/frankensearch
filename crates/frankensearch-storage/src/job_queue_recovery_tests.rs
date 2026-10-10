// Included by job_queue::tests so the existing real-storage fixtures are reused.

fn recovery_expire_processing(storage: &Storage) {
    storage
        .connection()
        .execute_sync("UPDATE embedding_jobs SET started_at = 0 WHERE status = 'processing';")
        .unwrap();
}

#[test]
fn recovery_stale_attempts_obey_retry_budget_and_fence_expired_workers() {
    for max_retries in [0, 1, 3] {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            max_retries,
            ..JobQueueConfig::default()
        });
        insert_document(&storage, "crash-loop", 91);
        queue.enqueue("crash-loop", "emb", &[91; 32], 0).unwrap();
        let mut previous = None;
        for attempt in 0..=max_retries {
            let claim = claim_single(&queue, "same-worker");
            assert_eq!(claim.retry_count, attempt);
            assert_eq!(claim.claim_epoch, i64::from(attempt) + 1);
            if let Some(old) = previous {
                assert_eq!(
                    queue.complete(&old, &[91; 32]).unwrap(),
                    ClaimOutcome::LostClaim
                );
            }
            recovery_expire_processing(&storage);
            assert_eq!(queue.reclaim_stale_jobs().unwrap(), 1);
            assert_eq!(
                queue.complete(&claim, &[91; 32]).unwrap(),
                ClaimOutcome::LostClaim
            );
            previous = Some(claim);
        }
        let depth = queue.queue_depth().unwrap();
        assert_eq!(depth.pending, 0);
        assert_eq!(depth.processing, 0);
        assert_eq!(depth.failed, 1);
        assert_eq!(storage.count_by_status("emb").unwrap().failed, 1);
        assert!(queue.claim_batch("next-worker", 1).unwrap().is_empty());
        assert_eq!(queue.reclaim_stale_jobs().unwrap(), 0);
        let metrics = queue.metrics().snapshot();
        assert_eq!(metrics.total_retried, u64::from(max_retries));
        assert_eq!(metrics.total_failed, 1);
        let rows = queue_rows(&storage);
        assert_eq!(rows[0][8], SqliteValue::Integer(i64::from(max_retries) + 1));
        assert_eq!(rows[0][12], SqliteValue::Null);
        assert!(matches!(rows[0][6], SqliteValue::Integer(_)));
    }
}

#[test]
fn recovery_exhausted_old_revision_does_not_fail_current_replacement() {
    let (queue, storage) = queue_fixture(JobQueueConfig {
        max_retries: 0,
        ..JobQueueConfig::default()
    });
    insert_document(&storage, "replacement", 92);
    queue.enqueue("replacement", "emb", &[92; 32], 0).unwrap();
    let old = claim_single(&queue, "old-worker");
    insert_document(&storage, "replacement", 93);
    queue.enqueue("replacement", "emb", &[93; 32], 1).unwrap();
    let catalog_before = catalog_status_rows(&storage);
    recovery_expire_processing(&storage);
    assert_eq!(queue.reclaim_stale_jobs().unwrap(), 1);
    assert_eq!(catalog_status_rows(&storage), catalog_before);
    assert_eq!(queue.queue_depth().unwrap().failed, 0);
    assert_eq!(queue.metrics().snapshot().total_failed, 0);
    let current = claim_single(&queue, "new-worker");
    assert_eq!(current.content_hash, Some([93; 32]));
    assert_eq!(
        queue.complete(&old, &[92; 32]).unwrap(),
        ClaimOutcome::LostClaim
    );
    assert_eq!(
        queue.complete(&current, &[93; 32]).unwrap(),
        ClaimOutcome::Applied(())
    );
}

#[test]
fn recovery_terminal_transition_replaces_existing_failed_history() {
    let (queue, storage) = queue_fixture(JobQueueConfig {
        max_retries: 0,
        ..JobQueueConfig::default()
    });
    insert_document(&storage, "history", 94);
    queue.enqueue("history", "emb", &[94; 32], 0).unwrap();
    let old = claim_single(&queue, "first-worker");
    assert!(matches!(
        queue.fail(&old, &[94; 32], "old failure").unwrap(),
        ClaimOutcome::Applied(FailResult::TerminalFailed { .. })
    ));
    queue.enqueue("history", "emb", &[94; 32], 0).unwrap();
    let current = claim_single(&queue, "second-worker");
    recovery_expire_processing(&storage);
    assert_eq!(queue.reclaim_stale_jobs().unwrap(), 1);
    let rows = queue_rows(&storage);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], SqliteValue::Integer(current.job_id));
    assert_eq!(queue.queue_depth().unwrap().failed, 1);
    assert_eq!(queue.metrics().snapshot().total_failed, 2);
}

#[test]
fn recovery_saturated_retry_counter_is_terminal_for_explicit_and_stale_failures() {
    for stale in [false, true] {
        let (queue, storage) = queue_fixture(JobQueueConfig {
            max_retries: u32::MAX,
            ..JobQueueConfig::default()
        });
        insert_document(&storage, "saturated", 95);
        queue.enqueue("saturated", "emb", &[95; 32], 0).unwrap();
        let claim = claim_single(&queue, "worker");
        storage
            .connection()
            .execute_sync("UPDATE embedding_jobs SET retry_count = 4294967295;")
            .unwrap();
        if stale {
            recovery_expire_processing(&storage);
            assert_eq!(queue.reclaim_stale_jobs().unwrap(), 1);
        } else {
            assert_eq!(
                queue.fail(&claim, &[95; 32], "exhausted").unwrap(),
                ClaimOutcome::Applied(FailResult::TerminalFailed {
                    retry_count: u32::MAX
                })
            );
        }
        assert_eq!(queue.queue_depth().unwrap().failed, 1);
        assert_eq!(
            queue_rows(&storage)[0][8],
            SqliteValue::Integer(i64::from(u32::MAX))
        );
        assert_eq!(storage.count_by_status("emb").unwrap().failed, 1);
        assert_eq!(queue.metrics().snapshot().total_failed, 1);
    }
}

#[test]
fn recovery_invalid_later_counter_rolls_back_queue_catalog_and_metrics() {
    let (queue, storage) = queue_fixture(JobQueueConfig {
        max_retries: 0,
        ..JobQueueConfig::default()
    });
    for (doc_id, seed) in [("rollback-first", 96), ("rollback-second", 97)] {
        insert_document(&storage, doc_id, seed);
        queue.enqueue(doc_id, "emb", &[seed; 32], 0).unwrap();
        let _ = claim_single(&queue, "worker");
    }
    recovery_expire_processing(&storage);
    storage
        .connection()
        .execute_sync("UPDATE embedding_jobs SET retry_count = -1 WHERE doc_id = 'rollback-second';")
        .unwrap();
    let rows_before = queue_rows(&storage);
    let catalog_before = catalog_status_rows(&storage);
    let metrics_before = queue.metrics().snapshot();
    assert!(queue.reclaim_stale_jobs().is_err());
    assert_eq!(queue_rows(&storage), rows_before);
    assert_eq!(catalog_status_rows(&storage), catalog_before);
    assert_eq!(queue.metrics().snapshot(), metrics_before);
}
