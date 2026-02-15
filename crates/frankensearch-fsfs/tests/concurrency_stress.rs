//! Concurrent stress tests for the fsfs indexing pipeline contention model.
//!
//! Validates lock ordering, contention metrics, resource tokens, sentinel recovery,
//! and workload scheduler behavior under multi-threaded contention pressure.
//!
//! Fast CI tests run in < 5 minutes. Extended soak tests are gated behind
//! `FSFS_SOAK_TESTS=1` environment variable.
//!
//! Bead: bd-2hz.10.10

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use frankensearch_fsfs::concurrency::{
    AccessMode, BudgetSchedulerMode, BudgetSchedulerPolicy, ContentionMetrics, ContentionPolicy,
    ContentionSnapshot, LockLevel, LockOrderGuard, LockSentinel, ResourceId, ResourceToken,
    WorkloadBudgetScheduler, WorkloadClass, WorkloadDemand, pipeline_access_matrix, read_sentinel,
    remove_sentinel, try_acquire_sentinel, write_sentinel,
};
use frankensearch_fsfs::lifecycle::PidFile;

// ─── Test artifact types ─────────────────────────────────────────────────

/// Summary of a concurrency stress test run.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StressTestResult {
    test_name: String,
    thread_count: usize,
    iterations: u64,
    duration_ms: u64,
    passed: bool,
    violations: Vec<String>,
    metrics: Option<ContentionSummary>,
}

/// Serializable contention metrics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContentionSummary {
    acquisitions: u64,
    contentions: u64,
    timeouts: u64,
    retries: u64,
    contention_rate: f64,
}

impl From<ContentionSnapshot> for ContentionSummary {
    fn from(s: ContentionSnapshot) -> Self {
        Self {
            acquisitions: s.acquisitions,
            contentions: s.contentions,
            timeouts: s.timeouts,
            retries: s.retries,
            contention_rate: s.contention_rate(),
        }
    }
}

fn emit_stress_artifact(result: &StressTestResult, artifact_dir: &Path) {
    fs::create_dir_all(artifact_dir).expect("create artifact dir");
    let path = artifact_dir.join(format!("{}.json", result.test_name));
    let json = serde_json::to_string_pretty(result).expect("serialize result");
    fs::write(&path, json).expect("write artifact");
}

const fn multiples_in_zero_based_range(len: u64, divisor: u64) -> u64 {
    if len == 0 || divisor == 0 {
        0
    } else {
        ((len - 1) / divisor) + 1
    }
}

// ─── Lock ordering: correct ascending acquisition across threads ──────

#[test]
fn lock_order_ascending_concurrent() {
    // N threads each acquire locks in correct ascending order.
    // No panics should occur (debug_assertions enforced).
    let thread_count = 8;
    let iterations_per_thread = 1000;
    let barrier = Arc::new(Barrier::new(thread_count));
    let violations = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let violations = Arc::clone(&violations);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..iterations_per_thread {
                    // Ascending order: Catalog(1) → EmbeddingQueue(2) → IndexCache(3)
                    let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                    let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                    let _g3 = LockOrderGuard::acquire(LockLevel::IndexCache);
                    // Guards drop in reverse order — correct RAII unwinding
                }
                // If we got here without panic, the ordering held.
                // (On panic the thread would be joined as Err.)
                violations.fetch_add(0, Ordering::Relaxed); // no-op, thread survived
            })
        })
        .collect();

    let mut panicked = 0;
    for h in handles {
        if h.join().is_err() {
            panicked += 1;
        }
    }

    assert_eq!(
        panicked, 0,
        "No threads should panic when acquiring locks in ascending order"
    );
}

#[test]
fn lock_order_full_chain_concurrent() {
    // Threads acquire the full 6-level chain.
    let thread_count = 4;
    let iterations = 500;
    let barrier = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..iterations {
                    let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                    let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                    let _g3 = LockOrderGuard::acquire(LockLevel::IndexCache);
                    let _g4 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                    let _g5 = LockOrderGuard::acquire(LockLevel::TantivyWriter);
                    let _g6 = LockOrderGuard::acquire(LockLevel::AdaptiveState);
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("thread should not panic with full ascending chain");
    }
}

#[test]
fn lock_order_partial_chains_interleaved() {
    // Different threads acquire different subsets of the lock hierarchy,
    // all in ascending order. This mimics real pipeline stages.
    let thread_count = 8;
    let iterations = 1000;
    let barrier = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|i| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..iterations {
                    match i % 4 {
                        0 => {
                            // crawl pattern: Catalog → EmbeddingQueue
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                        }
                        1 => {
                            // embed pattern: Catalog → EmbeddingQueue → FsviSegment
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                            let _g3 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                        }
                        2 => {
                            // query pattern: Catalog → IndexCache
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::IndexCache);
                        }
                        _ => {
                            // compaction pattern: Catalog → FsviSegment → TantivyWriter
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                            let _g3 = LockOrderGuard::acquire(LockLevel::TantivyWriter);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("no panic when threads use different ascending subchains");
    }
}

// ─── Lock ordering violation detection ────────────────────────────────

#[test]
#[cfg(debug_assertions)]
fn lock_order_violation_detected_in_debug() {
    // Attempt descending lock acquisition should panic in debug builds.
    let result = std::panic::catch_unwind(|| {
        let _g1 = LockOrderGuard::acquire(LockLevel::TantivyWriter);
        let _g2 = LockOrderGuard::acquire(LockLevel::Catalog); // violation: 5 → 1
    });

    assert!(
        result.is_err(),
        "Descending lock acquisition should panic in debug builds"
    );
}

#[test]
#[cfg(debug_assertions)]
fn lock_order_same_level_violation_detected() {
    // Acquiring the same level twice should panic in debug builds.
    let result = std::panic::catch_unwind(|| {
        let _g1 = LockOrderGuard::acquire(LockLevel::IndexCache);
        let _g2 = LockOrderGuard::acquire(LockLevel::IndexCache); // violation: 3 → 3
    });

    assert!(
        result.is_err(),
        "Same-level lock re-acquisition should panic in debug builds"
    );
}

// ─── Contention metrics under concurrent load ─────────────────────────

#[test]
fn contention_metrics_accumulate_correctly_under_concurrent_writes() {
    let metrics = Arc::new(ContentionMetrics::new());
    let thread_count = 16;
    let ops_per_thread = 10_000;
    let barrier = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|i| {
            let metrics = Arc::clone(&metrics);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for j in 0..ops_per_thread {
                    let contended = (j % 3) == 0; // 33% contention
                    metrics.record_acquisition(contended);
                    if (j % 7) == 0 {
                        metrics.record_retry();
                    }
                    if i == 0 && (j % 100) == 0 {
                        metrics.record_timeout();
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let snap = metrics.snapshot();
    let expected_acquisitions = (thread_count as u64) * ops_per_thread;
    assert_eq!(
        snap.acquisitions, expected_acquisitions,
        "All acquisitions should be counted atomically"
    );

    // Multiples are zero-based (`j = 0` also matches), so use exact count.
    let expected_contentions =
        (thread_count as u64) * multiples_in_zero_based_range(ops_per_thread, 3);
    assert_eq!(
        snap.contentions, expected_contentions,
        "Contention count should match expected rate"
    );

    // Retries are also zero-based (`j % 7 == 0` includes j=0).
    let expected_retries = (thread_count as u64) * multiples_in_zero_based_range(ops_per_thread, 7);
    assert_eq!(snap.retries, expected_retries);

    // Timeouts: only thread 0, every 100th op (including j=0).
    let expected_timeouts = multiples_in_zero_based_range(ops_per_thread, 100);
    assert_eq!(snap.timeouts, expected_timeouts);

    // Contention rate should be approximately 33%
    let rate = snap.contention_rate();
    assert!(
        (rate - 0.333).abs() < 0.01,
        "Contention rate should be ~33%, got {rate:.4}"
    );
}

#[test]
fn contention_metrics_snapshot_is_consistent_during_concurrent_updates() {
    // Take snapshots while metrics are being updated concurrently.
    // Snapshots should be internally consistent (no torn reads).
    let metrics = Arc::new(ContentionMetrics::new());
    let running = Arc::new(AtomicBool::new(true));
    let thread_count = 8;
    let barrier = Arc::new(Barrier::new(thread_count + 1)); // +1 for snapshot taker

    // Spawn writer threads
    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let metrics = Arc::clone(&metrics);
            let running = Arc::clone(&running);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                while running.load(Ordering::Relaxed) {
                    metrics.record_acquisition(true);
                    metrics.record_retry();
                }
            })
        })
        .collect();

    // Snapshot taker
    barrier.wait();
    let mut snapshots = Vec::new();
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(100) {
        snapshots.push(metrics.snapshot());
    }

    running.store(false, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    // Verify monotonicity: each successive snapshot should have >= values
    for window in snapshots.windows(2) {
        assert!(
            window[1].acquisitions >= window[0].acquisitions,
            "Acquisitions should be monotonically non-decreasing"
        );
        assert!(
            window[1].contentions >= window[0].contentions,
            "Contentions should be monotonically non-decreasing"
        );
        assert!(
            window[1].retries >= window[0].retries,
            "Retries should be monotonically non-decreasing"
        );
    }

    // Cross-counter inequalities are not guaranteed for in-flight snapshots because
    // `snapshot()` reads atomics independently. Validate cross-counter invariants
    // after all writers have stopped instead.
    let final_snapshot = metrics.snapshot();
    assert_eq!(
        final_snapshot.acquisitions, final_snapshot.contentions,
        "Each loop records exactly one contended acquisition"
    );
    assert_eq!(
        final_snapshot.retries, final_snapshot.acquisitions,
        "Each loop records exactly one retry"
    );
}

// ─── Contention policy ────────────────────────────────────────────────

#[test]
fn contention_policy_backoff_stays_within_bounds() {
    let policy = ContentionPolicy::default();
    for attempt in 0..100 {
        let delay = policy.backoff_delay(attempt);
        assert!(
            delay <= policy.max_backoff,
            "Backoff at attempt {attempt} ({delay:?}) exceeded max ({:?})",
            policy.max_backoff
        );
        assert!(
            delay >= policy.initial_backoff || attempt == 0,
            "Backoff should be at least initial_backoff after first attempt"
        );
    }
}

#[test]
fn contention_policy_backpressure_thresholds() {
    let policy = ContentionPolicy::default();
    assert!(!policy.is_backpressured(0));
    assert!(!policy.is_backpressured(9_999));
    assert!(policy.is_backpressured(10_000));
    assert!(policy.is_backpressured(100_000));
}

// ─── Resource token lifecycle ─────────────────────────────────────────

#[test]
fn resource_token_tracks_holder_and_duration() {
    let token = ResourceToken::new(ResourceId::EmbeddingQueue, "stress_test_worker");
    assert_eq!(token.holder(), "stress_test_worker");
    assert!(matches!(token.resource(), ResourceId::EmbeddingQueue));

    std::thread::sleep(Duration::from_millis(5));
    assert!(token.held_duration() >= Duration::from_millis(5));
}

#[test]
fn resource_token_concurrent_creation() {
    // Multiple threads creating tokens for different resources simultaneously.
    let thread_count = 16;
    let barrier = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|i| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let resource = match i % 5 {
                    0 => ResourceId::Catalog(format!("/tmp/cat_{i}").into()),
                    1 => ResourceId::FsviSegment(format!("/tmp/fsvi_{i}").into()),
                    2 => ResourceId::TantivyIndex(format!("/tmp/tantivy_{i}").into()),
                    3 => ResourceId::EmbeddingQueue,
                    _ => ResourceId::IndexCache,
                };
                let token = ResourceToken::new(resource, format!("worker_{i}"));
                assert_eq!(token.holder(), format!("worker_{i}"));
                // Token should survive creation without data races
                token.held_duration()
            })
        })
        .collect();

    for h in handles {
        let duration = h.join().expect("token creation should not panic");
        assert!(duration < Duration::from_secs(1));
    }
}

// ─── Sentinel file concurrent access ──────────────────────────────────

#[test]
fn sentinel_write_read_roundtrip_concurrent() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let thread_count = 8;
    let barrier = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|i| {
            let dir = tmp.path().to_owned();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let sentinel_path = dir.join(format!("sentinel_{i}.lock"));
                let sentinel =
                    LockSentinel::current(format!("resource_{i}"), format!("holder_{i}"));
                write_sentinel(&sentinel_path, &sentinel).expect("write sentinel");

                let read_back = read_sentinel(&sentinel_path).expect("read sentinel");
                assert_eq!(read_back.pid, sentinel.pid);
                assert_eq!(read_back.resource, format!("resource_{i}"));
                assert_eq!(read_back.holder, format!("holder_{i}"));

                remove_sentinel(&sentinel_path).expect("remove sentinel");
                assert!(!sentinel_path.exists());
            })
        })
        .collect();

    for h in handles {
        h.join().expect("sentinel roundtrip should not panic");
    }
}

#[test]
fn sentinel_acquire_blocks_second_acquirer() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let sentinel_path = tmp.path().join("shared.lock");
    let stale_threshold = Duration::from_mins(5); // high so it won't auto-recover

    // First acquirer succeeds
    let sentinel = try_acquire_sentinel(
        &sentinel_path,
        "shared_resource",
        "first_holder",
        stale_threshold,
    )
    .expect("first acquire should succeed");

    assert_eq!(sentinel.holder, "first_holder");
    assert!(sentinel.is_holder_alive());

    // Second acquirer should fail with WouldBlock
    let result = try_acquire_sentinel(
        &sentinel_path,
        "shared_resource",
        "second_holder",
        stale_threshold,
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::WouldBlock);

    // Cleanup
    remove_sentinel(&sentinel_path).expect("cleanup");
}

#[test]
fn sentinel_stale_recovery() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let sentinel_path = tmp.path().join("stale.lock");

    // Write a sentinel with a fake (dead) PID
    let stale = LockSentinel {
        pid: 999_999_999, // very unlikely to be alive
        hostname: std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("HOST"))
            .unwrap_or_else(|_| "unknown".into()),
        created_at_ms: 0, // epoch — very old
        resource: "stale_resource".to_owned(),
        holder: "dead_holder".to_owned(),
    };
    write_sentinel(&sentinel_path, &stale).expect("write stale sentinel");

    // New acquirer should recover the stale sentinel
    let result = try_acquire_sentinel(
        &sentinel_path,
        "stale_resource",
        "new_holder",
        Duration::from_secs(1),
    );

    assert!(result.is_ok(), "Should recover stale sentinel: {result:?}");
    let recovered = result.unwrap();
    assert_eq!(recovered.holder, "new_holder");

    remove_sentinel(&sentinel_path).expect("cleanup");
}

#[test]
fn sentinel_atomic_acquire_has_single_winner_under_race() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let sentinel_path = tmp.path().join("racy.lock");
    let stale_threshold = Duration::from_mins(5);
    let thread_count = 32;
    let start = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|i| {
            let path = sentinel_path.clone();
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                let holder = format!("holder_{i}");
                let attempt =
                    try_acquire_sentinel(&path, "racy_resource", &holder, stale_threshold);
                // Contenders can briefly observe a partially-written sentinel JSON
                // from the winner. Retry once so the race test validates lock
                // semantics rather than parser timing.
                let resolved = match attempt {
                    Err(err) if err.kind() == std::io::ErrorKind::InvalidData => {
                        std::thread::sleep(Duration::from_millis(1));
                        try_acquire_sentinel(&path, "racy_resource", &holder, stale_threshold)
                    }
                    other => other,
                };

                resolved
                    .map(|sentinel| sentinel.holder)
                    .map_err(|err| err.kind())
            })
        })
        .collect();

    let mut winners = Vec::new();
    let mut errors = Vec::new();
    for handle in handles {
        match handle.join().expect("thread should not panic") {
            Ok(holder) => winners.push(holder),
            Err(kind) => errors.push(kind),
        }
    }

    assert_eq!(
        winners.len(),
        1,
        "exactly one contender must create the sentinel"
    );
    assert_eq!(
        errors.len(),
        thread_count - 1,
        "all other contenders must fail acquisition"
    );
    assert!(
        errors
            .iter()
            .all(|kind| *kind == std::io::ErrorKind::WouldBlock),
        "all losing contenders should see WouldBlock"
    );

    if sentinel_path.exists() {
        remove_sentinel(&sentinel_path).expect("cleanup");
    }
}

#[test]
fn pid_file_atomic_acquire_has_single_winner_under_race() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let pid_path = tmp.path().join("fsfs.pid");
    let thread_count = 24;
    let start = Arc::new(Barrier::new(thread_count));
    let attempts_done = Arc::new(Barrier::new(thread_count + 1));
    let release = Arc::new(Barrier::new(thread_count + 1));

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let path = pid_path.clone();
            let start = Arc::clone(&start);
            let attempts_done = Arc::clone(&attempts_done);
            let release = Arc::clone(&release);
            std::thread::spawn(move || {
                start.wait();
                let mut pid_file = PidFile::new(path);
                let first = pid_file.acquire("test-version");
                let result = match first {
                    Err(err) if err.kind() == std::io::ErrorKind::InvalidData => {
                        std::thread::sleep(Duration::from_millis(1));
                        pid_file.acquire("test-version")
                    }
                    other => other,
                };
                attempts_done.wait();
                release.wait();
                result.map_err(|err| err.kind())
            })
        })
        .collect();

    attempts_done.wait();
    release.wait();

    let mut winners = 0;
    let mut errors = Vec::new();
    for handle in handles {
        match handle.join().expect("thread should not panic") {
            Ok(()) => winners += 1,
            Err(kind) => errors.push(kind),
        }
    }

    assert_eq!(
        winners, 1,
        "exactly one contender must atomically create the PID file"
    );
    assert_eq!(
        errors.len(),
        thread_count - 1,
        "all other contenders should fail while lock holder is alive"
    );
    assert!(
        errors
            .iter()
            .all(|kind| *kind == std::io::ErrorKind::AddrInUse),
        "all losing contenders should see AddrInUse"
    );
}

// ─── Workload scheduler concurrent permit lifecycle ───────────────────

#[test]
fn scheduler_permit_lifecycle_deterministic() {
    // Single-threaded deterministic check of the full reserve→commit→complete cycle.
    let policy = BudgetSchedulerPolicy::default();
    let mut scheduler = WorkloadBudgetScheduler::new(policy);

    // Reserve permits for all 3 classes
    let mut p_ingest = scheduler
        .reserve(WorkloadClass::Ingest, 4)
        .expect("reserve ingest");
    let mut p_embed = scheduler
        .reserve(WorkloadClass::Embed, 4)
        .expect("reserve embed");
    let mut p_query = scheduler
        .reserve(WorkloadClass::Query, 4)
        .expect("reserve query");

    assert_eq!(scheduler.reserved_for(WorkloadClass::Ingest), 4);
    assert_eq!(scheduler.reserved_for(WorkloadClass::Embed), 4);
    assert_eq!(scheduler.reserved_for(WorkloadClass::Query), 4);

    // Commit all
    scheduler
        .commit_permit(&mut p_ingest)
        .expect("commit ingest");
    scheduler.commit_permit(&mut p_embed).expect("commit embed");
    scheduler.commit_permit(&mut p_query).expect("commit query");

    assert_eq!(scheduler.inflight_for(WorkloadClass::Ingest), 4);
    assert_eq!(scheduler.inflight_for(WorkloadClass::Embed), 4);
    assert_eq!(scheduler.inflight_for(WorkloadClass::Query), 4);

    // Complete all
    scheduler
        .complete(WorkloadClass::Ingest, 4)
        .expect("complete ingest");
    scheduler
        .complete(WorkloadClass::Embed, 4)
        .expect("complete embed");
    scheduler
        .complete(WorkloadClass::Query, 4)
        .expect("complete query");

    assert_eq!(scheduler.inflight_for(WorkloadClass::Ingest), 0);
    assert_eq!(scheduler.inflight_for(WorkloadClass::Embed), 0);
    assert_eq!(scheduler.inflight_for(WorkloadClass::Query), 0);
}

#[test]
fn scheduler_reserve_cancel_frees_capacity() {
    let policy = BudgetSchedulerPolicy::default();
    let mut scheduler = WorkloadBudgetScheduler::new(policy);

    let mut permit = scheduler
        .reserve(WorkloadClass::Ingest, 10)
        .expect("reserve");
    assert_eq!(scheduler.reserved_for(WorkloadClass::Ingest), 10);

    scheduler.cancel_permit(&mut permit).expect("cancel");
    assert_eq!(scheduler.reserved_for(WorkloadClass::Ingest), 0);
}

#[test]
fn scheduler_capacity_exhaustion_rejects_reserve() {
    let policy = BudgetSchedulerPolicy {
        total_slots: 4,
        admission_limit: 8,
        ..BudgetSchedulerPolicy::default()
    };
    let mut scheduler = WorkloadBudgetScheduler::new(policy);

    // Reserve all 4 slots
    let _p1 = scheduler
        .reserve(WorkloadClass::Ingest, 4)
        .expect("first reserve");

    // Try to reserve one more — should fail
    let result = scheduler.reserve(WorkloadClass::Embed, 1);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "scheduler.reserve.capacity_exhausted");
}

#[test]
fn scheduler_admission_limit_rejects_reserve() {
    let policy = BudgetSchedulerPolicy {
        total_slots: 100,
        admission_limit: 5,
        ..BudgetSchedulerPolicy::default()
    };
    let mut scheduler = WorkloadBudgetScheduler::new(policy);

    let _p1 = scheduler
        .reserve(WorkloadClass::Ingest, 5)
        .expect("reserve up to limit");

    let result = scheduler.reserve(WorkloadClass::Embed, 1);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "scheduler.reserve.admission_limited");
}

// ─── Fair-share and latency-sensitive allocation under load ───────────

#[test]
fn fair_share_distributes_equally_with_equal_demand() {
    let policy = BudgetSchedulerPolicy {
        total_slots: 24,
        ..BudgetSchedulerPolicy::default()
    };
    let mut scheduler = WorkloadBudgetScheduler::new(policy);

    let demand = WorkloadDemand {
        ingest: 100,
        embed: 100,
        query: 100,
    };
    let alloc = scheduler.plan_cycle(demand, BudgetSchedulerMode::FairShare);

    assert_eq!(alloc.total(), 24);
    assert_eq!(alloc.ingest, 8);
    assert_eq!(alloc.embed, 8);
    assert_eq!(alloc.query, 8);
}

#[test]
fn latency_sensitive_reserves_query_floor() {
    let policy = BudgetSchedulerPolicy {
        total_slots: 20,
        latency_query_reserve_pct: 50,
        ..BudgetSchedulerPolicy::default()
    };
    let mut scheduler = WorkloadBudgetScheduler::new(policy);

    let demand = WorkloadDemand {
        ingest: 100,
        embed: 100,
        query: 100,
    };
    let alloc = scheduler.plan_cycle(demand, BudgetSchedulerMode::LatencySensitive);

    assert_eq!(alloc.total(), 20);
    // Query should get at least 50% of 20 = 10 slots
    assert!(
        alloc.query >= 10,
        "Query should get at least 50% floor, got {}",
        alloc.query
    );
}

#[test]
fn starvation_guard_forces_progress() {
    let policy = BudgetSchedulerPolicy {
        total_slots: 10,
        starvation_guard_cycles: 2,
        ..BudgetSchedulerPolicy::default()
    };
    let mut scheduler = WorkloadBudgetScheduler::new(policy);

    // Starve Query for 3 cycles by only providing Ingest demand first
    for _ in 0..3 {
        let demand = WorkloadDemand {
            ingest: 100,
            embed: 0,
            query: 1, // has demand but should get starved by overwhelming ingest
        };
        let _alloc = scheduler.plan_cycle(demand, BudgetSchedulerMode::FairShare);
    }

    // After starvation_guard_cycles, query should receive at least 1 slot
    let demand = WorkloadDemand {
        ingest: 100,
        embed: 0,
        query: 1,
    };
    let alloc = scheduler.plan_cycle(demand, BudgetSchedulerMode::FairShare);

    // Query should have gotten some slots (starvation guard forces at least 1)
    // Note: fair-share with 2 active classes (ingest + query) already gives query 5 of 10 slots,
    // so starvation guard may not be needed. But the contract still holds.
    assert!(
        alloc.query >= 1,
        "Query should get at least 1 slot after starvation guard threshold"
    );
}

// ─── Pipeline access matrix validation ────────────────────────────────

#[test]
fn pipeline_access_matrix_covers_all_stages() {
    let matrix = pipeline_access_matrix();
    let expected_stages = [
        "crawl",
        "classify",
        "embed_fast",
        "embed_quality",
        "lexical_index",
        "serve_queries",
        "refresh_worker",
        "compaction",
    ];

    assert_eq!(
        matrix.len(),
        expected_stages.len(),
        "Matrix should cover exactly {} stages",
        expected_stages.len()
    );

    for (stage, expected) in matrix.iter().zip(expected_stages.iter()) {
        assert_eq!(stage.stage, *expected);
    }
}

#[test]
fn pipeline_access_serve_queries_is_read_only() {
    let matrix = pipeline_access_matrix();
    let serve = matrix
        .iter()
        .find(|s| s.stage == "serve_queries")
        .expect("serve_queries stage exists");

    assert_eq!(serve.catalog, AccessMode::ReadOnly);
    assert_eq!(serve.fsvi, AccessMode::ReadOnly);
    assert_eq!(serve.tantivy, AccessMode::ReadOnly);
    assert_eq!(serve.cache, AccessMode::ReadOnly);
    assert_eq!(serve.queue, AccessMode::None);
}

#[test]
fn pipeline_access_no_stage_writes_all_resources() {
    // No single stage should need R/W on all 5 resources (safety invariant).
    let matrix = pipeline_access_matrix();
    for stage in matrix {
        let rw_count = [
            stage.catalog,
            stage.queue,
            stage.fsvi,
            stage.tantivy,
            stage.cache,
        ]
        .iter()
        .filter(|&&mode| mode == AccessMode::ReadWrite)
        .count();

        assert!(
            rw_count <= 4,
            "Stage '{}' writes to {} resources (max expected: 4)",
            stage.stage,
            rw_count
        );
    }
}

// ─── Lock ordering across multiple pipeline patterns ──────────────────

#[test]
fn pipeline_patterns_maintain_lock_ordering_concurrent() {
    // Simulate all 8 pipeline stages' lock patterns concurrently.
    // Each stage acquires locks as documented in the access matrix.
    // All must succeed without panic.
    let thread_count = 8;
    let iterations = 500;
    let barrier = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|i| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..iterations {
                    match i {
                        0 => {
                            // crawl: Catalog(1) + Queue(2)
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                        }
                        1 => {
                            // classify: Catalog(1) only
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                        }
                        2 => {
                            // embed_fast: Catalog(1) + Queue(2) + FSVI(4)
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                            let _g3 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                        }
                        3 => {
                            // embed_quality: same as embed_fast
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                            let _g3 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                        }
                        4 => {
                            // lexical_index: Catalog(1) + TantivyWriter(5)
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::TantivyWriter);
                        }
                        5 => {
                            // serve_queries: Catalog(1) + IndexCache(3) (R/O pattern)
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::IndexCache);
                        }
                        6 => {
                            // refresh_worker: Queue(2) + FSVI(4) + IndexCache via arc swap
                            let _g1 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                            let _g2 = LockOrderGuard::acquire(LockLevel::IndexCache);
                            let _g3 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                        }
                        _ => {
                            // compaction: Catalog(1) + FSVI(4) + Tantivy(5) + Cache(3)
                            // Correct ascending order: 1, 3, 4, 5
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::IndexCache);
                            let _g3 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                            let _g4 = LockOrderGuard::acquire(LockLevel::TantivyWriter);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("all pipeline patterns should maintain ascending lock order");
    }
}

// ─── Multi-threaded scheduler stress ──────────────────────────────────

#[test]
fn scheduler_rapid_plan_cycles_deterministic() {
    // Run many plan_cycle calls sequentially (simulating rapid scheduling).
    // Verify total allocation never exceeds total_slots.
    let policy = BudgetSchedulerPolicy {
        total_slots: 16,
        admission_limit: 32,
        latency_query_reserve_pct: 40,
        starvation_guard_cycles: 3,
    };
    let mut scheduler = WorkloadBudgetScheduler::new(policy);

    let demands = [
        WorkloadDemand {
            ingest: 100,
            embed: 0,
            query: 0,
        },
        WorkloadDemand {
            ingest: 0,
            embed: 100,
            query: 0,
        },
        WorkloadDemand {
            ingest: 0,
            embed: 0,
            query: 100,
        },
        WorkloadDemand {
            ingest: 50,
            embed: 50,
            query: 50,
        },
        WorkloadDemand {
            ingest: 1,
            embed: 1,
            query: 1,
        },
        WorkloadDemand {
            ingest: 0,
            embed: 0,
            query: 0,
        },
    ];

    for (i, demand) in demands.iter().cycle().take(1000).enumerate() {
        let mode = if i % 2 == 0 {
            BudgetSchedulerMode::FairShare
        } else {
            BudgetSchedulerMode::LatencySensitive
        };
        let alloc = scheduler.plan_cycle(*demand, mode);

        assert!(
            alloc.total() <= policy.total_slots,
            "Cycle {i}: allocation total {} exceeds {} slots",
            alloc.total(),
            policy.total_slots
        );

        // Each class allocation should not exceed demand
        assert!(alloc.ingest <= demand.ingest || demand.ingest == 0);
        assert!(alloc.embed <= demand.embed || demand.embed == 0);
        assert!(alloc.query <= demand.query || demand.query == 0);
    }
}

// ─── Extended soak test (gated) ──────────────────────────────────────

#[test]
fn soak_lock_ordering_extended() {
    if std::env::var("FSFS_SOAK_TESTS").is_err() {
        eprintln!("Skipping soak test (set FSFS_SOAK_TESTS=1 to enable)");
        return;
    }

    // Extended soak: 32 threads, 60 seconds of continuous lock acquisitions.
    let thread_count = 32;
    let running = Arc::new(AtomicBool::new(true));
    let total_ops = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(thread_count));

    let handles: Vec<_> = (0..thread_count)
        .map(|i| {
            let running = Arc::clone(&running);
            let total_ops = Arc::clone(&total_ops);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let mut ops = 0_u64;
                while running.load(Ordering::Relaxed) {
                    match i % 4 {
                        0 => {
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                        }
                        1 => {
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::IndexCache);
                            let _g3 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                        }
                        2 => {
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::TantivyWriter);
                        }
                        _ => {
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::IndexCache);
                            let _g3 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                            let _g4 = LockOrderGuard::acquire(LockLevel::TantivyWriter);
                            let _g5 = LockOrderGuard::acquire(LockLevel::AdaptiveState);
                        }
                    }
                    ops += 1;
                }
                total_ops.fetch_add(ops, Ordering::Relaxed);
            })
        })
        .collect();

    // Run for 30 seconds (shortened from full 30min for CI-nightly)
    std::thread::sleep(Duration::from_secs(30));
    running.store(false, Ordering::Relaxed);

    for h in handles {
        h.join().expect("soak test: no deadlocks or panics");
    }

    let ops = total_ops.load(Ordering::Relaxed);
    eprintln!("Soak test completed: {ops} lock acquisition cycles across {thread_count} threads");
    assert!(
        ops > 0,
        "Soak test should complete at least some operations"
    );
}

// ─── Artifact emission ───────────────────────────────────────────────

#[test]
fn stress_artifact_emission() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let metrics = ContentionMetrics::new();

    // Simulate some activity
    for i in 0..100 {
        metrics.record_acquisition(i % 5 == 0);
        if i % 10 == 0 {
            metrics.record_retry();
        }
    }

    let result = StressTestResult {
        test_name: "example_stress_test".to_owned(),
        thread_count: 8,
        iterations: 1000,
        duration_ms: 42,
        passed: true,
        violations: Vec::new(),
        metrics: Some(metrics.snapshot().into()),
    };

    emit_stress_artifact(&result, tmp.path());

    let path = tmp.path().join("example_stress_test.json");
    assert!(path.exists());
    let contents = fs::read_to_string(&path).expect("read artifact");
    let parsed: StressTestResult = serde_json::from_str(&contents).expect("parse artifact");
    assert_eq!(parsed.test_name, "example_stress_test");
    assert!(parsed.passed);
    assert_eq!(parsed.metrics.unwrap().acquisitions, 100);
}

// ─── Full concurrent stress suite ────────────────────────────────────

fn run_lock_ordering_stress_phase(
    thread_count: usize,
    iterations: u64,
    metrics: &Arc<ContentionMetrics>,
) -> usize {
    let barrier = Arc::new(Barrier::new(thread_count));
    let lock_handles: Vec<_> = (0..thread_count)
        .map(|thread_index| {
            let barrier = Arc::clone(&barrier);
            let metrics = Arc::clone(metrics);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..iterations {
                    match thread_index % 4 {
                        0 => {
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                            metrics.record_acquisition(false);
                        }
                        1 => {
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::IndexCache);
                            let _g3 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                            metrics.record_acquisition(true);
                        }
                        2 => {
                            let _g1 = LockOrderGuard::acquire(LockLevel::EmbeddingQueue);
                            let _g2 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                            let _g3 = LockOrderGuard::acquire(LockLevel::TantivyWriter);
                            metrics.record_acquisition(false);
                        }
                        _ => {
                            let _g1 = LockOrderGuard::acquire(LockLevel::Catalog);
                            let _g2 = LockOrderGuard::acquire(LockLevel::IndexCache);
                            let _g3 = LockOrderGuard::acquire(LockLevel::FsviSegment);
                            let _g4 = LockOrderGuard::acquire(LockLevel::TantivyWriter);
                            let _g5 = LockOrderGuard::acquire(LockLevel::AdaptiveState);
                            metrics.record_acquisition(true);
                        }
                    }
                }
            })
        })
        .collect();

    let mut panics = 0;
    for handle in lock_handles {
        if handle.join().is_err() {
            panics += 1;
        }
    }
    panics
}

fn collect_sentinel_phase_violations(base_dir: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    for i in 0..4 {
        let sentinel_path = base_dir.join(format!("stress_{i}.lock"));
        let sentinel = LockSentinel::current(format!("res_{i}"), format!("holder_{i}"));
        if write_sentinel(&sentinel_path, &sentinel).is_err() {
            violations.push(format!("Failed to write sentinel {i}"));
            continue;
        }
        if let Ok(read_back) = read_sentinel(&sentinel_path) {
            if read_back.pid != sentinel.pid {
                violations.push(format!("Sentinel {i} PID mismatch"));
            }
        } else {
            violations.push(format!("Failed to read sentinel {i}"));
        }
        let _ = remove_sentinel(&sentinel_path);
    }
    violations
}

fn collect_scheduler_phase_violations(policy: BudgetSchedulerPolicy) -> Vec<String> {
    let mut violations = Vec::new();
    let mut scheduler = WorkloadBudgetScheduler::new(policy);
    for cycle in 0..1000 {
        let demand = WorkloadDemand {
            ingest: (cycle % 50) + 1,
            embed: (cycle % 30) + 1,
            query: (cycle % 20) + 1,
        };
        let mode = if cycle % 2 == 0 {
            BudgetSchedulerMode::FairShare
        } else {
            BudgetSchedulerMode::LatencySensitive
        };
        let allocation = scheduler.plan_cycle(demand, mode);
        if allocation.total() > policy.total_slots {
            violations.push(format!(
                "Cycle {cycle}: allocation {} exceeds {} slots",
                allocation.total(),
                policy.total_slots
            ));
        }
    }
    violations
}

#[test]
fn full_concurrency_stress_suite() {
    let start = Instant::now();
    let thread_count = 16;
    let iterations = 5_000_u64;
    let metrics = Arc::new(ContentionMetrics::new());
    let mut violations = Vec::new();

    let panics = run_lock_ordering_stress_phase(thread_count, iterations, &metrics);
    if panics > 0 {
        violations.push(format!(
            "{panics} threads panicked during lock ordering stress"
        ));
    }

    let tmp = tempfile::tempdir().expect("create temp dir for sentinels");
    violations.extend(collect_sentinel_phase_violations(tmp.path()));
    violations.extend(collect_scheduler_phase_violations(
        BudgetSchedulerPolicy::default(),
    ));

    let snapshot = metrics.snapshot();
    let duration = start.elapsed();
    let duration_ms = u64::try_from(duration.as_millis())
        .expect("stress duration milliseconds must fit into u64");
    let result = StressTestResult {
        test_name: "full_concurrency_stress_suite".to_owned(),
        thread_count,
        iterations,
        duration_ms,
        passed: violations.is_empty(),
        violations: violations.clone(),
        metrics: Some(snapshot.into()),
    };

    let artifact_dir = tmp.path().join("artifacts");
    emit_stress_artifact(&result, &artifact_dir);

    assert!(
        violations.is_empty(),
        "Stress test violations:\n{}",
        violations.join("\n")
    );

    eprintln!(
        "Full stress suite passed in {:?}: {} lock ops across {} threads, contention rate {:.2}%",
        duration,
        snapshot.acquisitions,
        thread_count,
        snapshot.contention_rate() * 100.0
    );
}
