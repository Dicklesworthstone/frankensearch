use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use frankensearch_ops::storage::SummaryWindow;
use frankensearch_ops::{
    OpsStorage, OpsStorageConfig, SimulatedProject, SimulationBatch, SimulationRun,
    SloMaterializationConfig, SloMaterializationResult, SloScope, TelemetrySimulator,
    TelemetrySimulatorConfig, WorkloadProfile,
};

fn pipeline_config(seed: u64) -> TelemetrySimulatorConfig {
    TelemetrySimulatorConfig {
        seed,
        start_ms: 1_734_503_200_000,
        tick_interval_ms: 1_000,
        ticks: 6,
        projects: vec![
            SimulatedProject {
                project_key: "cass".to_owned(),
                host_name: "cass-itest".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::Steady,
            },
            SimulatedProject {
                project_key: "xf".to_owned(),
                host_name: "xf-itest".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::Burst,
            },
            SimulatedProject {
                project_key: "mail".to_owned(),
                host_name: "mail-itest".to_owned(),
                instance_count: 1,
                workload: WorkloadProfile::EmbeddingWave,
            },
        ],
    }
}

fn apply_pipeline(
    storage: &OpsStorage,
    run: &SimulationRun,
    backpressure_threshold: usize,
) -> SearchResult<SloMaterializationResult> {
    apply_pipeline_batches(storage, &run.batches, backpressure_threshold)
}

fn apply_pipeline_batches(
    storage: &OpsStorage,
    batches: &[SimulationBatch],
    backpressure_threshold: usize,
) -> SearchResult<SloMaterializationResult> {
    let mut last_result = SloMaterializationResult::default();
    for batch in batches {
        let records: Vec<_> = batch
            .search_events
            .iter()
            .map(|event| event.record.clone())
            .collect();
        storage.ingest_search_events_batch(&records, backpressure_threshold)?;

        for sample in &batch.resource_samples {
            storage.upsert_resource_sample(sample)?;
        }

        let now_ms = i64::try_from(batch.now_ms).map_err(|_| SearchError::InvalidConfig {
            field: "now_ms".to_owned(),
            value: batch.now_ms.to_string(),
            reason: "must fit into i64".to_owned(),
        })?;
        let mut pairs = BTreeSet::new();
        for sample in &batch.resource_samples {
            pairs.insert((sample.project_key.clone(), sample.instance_id.clone()));
        }
        for (project_key, instance_id) in &pairs {
            let _ =
                storage.refresh_search_summaries_for_instance(project_key, instance_id, now_ms)?;
        }

        last_result = storage
            .materialize_slo_rollups_and_anomalies(now_ms, SloMaterializationConfig::default())?;
    }
    Ok(last_result)
}

fn temp_ops_db_path(test_name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("wall clock should be >= unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("frankensearch-ops-{test_name}-{nanos}.sqlite3"))
}

#[test]
fn pipeline_ingest_to_aggregation_materializes_expected_views() {
    let simulator = TelemetrySimulator::new(pipeline_config(21)).expect("config should validate");
    let run = simulator.generate().expect("generation should succeed");
    let storage = OpsStorage::open_in_memory().expect("storage should open");

    let materialization = apply_pipeline(&storage, &run, 8_192).expect("pipeline should succeed");
    assert!(
        materialization.rollups_upserted > 0,
        "pipeline should produce SLO rollups"
    );

    let metrics = storage.ingestion_metrics();
    assert_eq!(
        metrics.total_inserted,
        u64::try_from(run.total_search_events()).expect("event count should fit into u64")
    );

    for (project_key, instance_id) in run.instance_pairs() {
        let one_minute = storage
            .latest_search_summary(&project_key, &instance_id, SummaryWindow::OneMinute)
            .expect("summary query should succeed");
        assert!(
            one_minute.is_some(),
            "expected search summary for {project_key}/{instance_id}"
        );

        let trend = storage
            .query_resource_trend(
                &project_key,
                &instance_id,
                SummaryWindow::OneHour,
                i64::try_from(run.config.start_ms + 30_000).expect("timestamp should fit i64"),
                128,
            )
            .expect("resource trend query should succeed");
        assert!(
            !trend.is_empty(),
            "expected resource trend rows for {project_key}/{instance_id}"
        );
    }

    let fleet_rollups = storage
        .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 16)
        .expect("fleet rollup query should succeed");
    assert!(
        !fleet_rollups.is_empty(),
        "expected fleet rollups after materialization"
    );
}

#[test]
fn pipeline_discovery_attribution_aligns_with_storage_rollups_and_anomalies() {
    let simulator =
        TelemetrySimulator::new(pipeline_config(3_137)).expect("config should validate");
    let run = simulator.generate().expect("generation should succeed");
    let storage = OpsStorage::open_in_memory().expect("storage should open");

    let mut discovered_projects = BTreeSet::new();
    let mut discovered_pairs = BTreeSet::new();
    for batch in &run.batches {
        for discovered in &batch.discovered_instances {
            if let Some(project_key_hint) = discovered.project_key_hint.as_ref() {
                discovered_projects.insert(project_key_hint.clone());
                discovered_pairs.insert((project_key_hint.clone(), discovered.instance_id.clone()));
            }
        }
    }

    assert!(
        !discovered_pairs.is_empty(),
        "simulator should emit at least one discovered project/instance pair"
    );
    assert_eq!(
        discovered_pairs,
        run.instance_pairs(),
        "discovery and telemetry views should agree on project/instance attribution"
    );

    let _ = apply_pipeline(&storage, &run, 8_192).expect("pipeline should succeed");

    for project_key in &discovered_projects {
        let project_rollups = storage
            .query_slo_rollups_for_scope(SloScope::Project, project_key, 32)
            .expect("project rollup query should succeed");
        assert!(
            !project_rollups.is_empty(),
            "expected project rollups for discovered project {project_key}"
        );
        assert!(
            project_rollups
                .iter()
                .all(|row| row.scope == SloScope::Project && row.scope_key == *project_key),
            "rollups should remain scoped to discovered project {project_key}"
        );
    }

    for (project_key, instance_id) in &discovered_pairs {
        let summary = storage
            .latest_search_summary(project_key, instance_id, SummaryWindow::OneMinute)
            .expect("summary query should succeed");
        assert!(
            summary.is_some(),
            "expected one-minute summary for discovered pair {project_key}/{instance_id}"
        );
    }

    let anomalies = storage
        .query_anomaly_timeline(None, 256)
        .expect("anomaly timeline query should succeed");
    for anomaly in &anomalies {
        if let Some(project_key) = anomaly.project_key.as_ref() {
            assert!(
                discovered_projects.contains(project_key),
                "anomaly project {project_key} should be present in discovered attribution set"
            );
        }
    }
}

#[test]
fn pipeline_replay_is_deterministic_for_same_seed() {
    let config = pipeline_config(55);
    let run_a = TelemetrySimulator::new(config.clone())
        .expect("config should validate")
        .generate()
        .expect("generation should succeed");
    let run_b = TelemetrySimulator::new(config)
        .expect("config should validate")
        .generate()
        .expect("generation should succeed");
    assert_eq!(run_a.signature(), run_b.signature());

    let storage_a = OpsStorage::open_in_memory().expect("storage should open");
    let storage_b = OpsStorage::open_in_memory().expect("storage should open");
    let _ = apply_pipeline(&storage_a, &run_a, 8_192).expect("pipeline A should succeed");
    let _ = apply_pipeline(&storage_b, &run_b, 8_192).expect("pipeline B should succeed");

    let metrics_a = storage_a.ingestion_metrics();
    let metrics_b = storage_b.ingestion_metrics();
    assert_eq!(metrics_a.total_batches, metrics_b.total_batches);
    assert_eq!(metrics_a.total_inserted, metrics_b.total_inserted);
    assert_eq!(metrics_a.total_deduplicated, metrics_b.total_deduplicated);
    assert_eq!(
        metrics_a.total_failed_records,
        metrics_b.total_failed_records
    );
    assert_eq!(
        metrics_a.total_backpressured_batches,
        metrics_b.total_backpressured_batches
    );
    assert_eq!(metrics_a.pending_events, metrics_b.pending_events);
    assert_eq!(
        metrics_a.high_watermark_pending_events,
        metrics_b.high_watermark_pending_events
    );

    let fleet_a = storage_a
        .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 8)
        .expect("fleet rollup query A should succeed");
    let fleet_b = storage_b
        .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 8)
        .expect("fleet rollup query B should succeed");

    let summary_a: Vec<_> = fleet_a
        .iter()
        .map(|row| {
            (
                row.window,
                row.window_start_ms,
                row.total_requests,
                row.reason_code.clone(),
            )
        })
        .collect();
    let summary_b: Vec<_> = fleet_b
        .iter()
        .map(|row| {
            (
                row.window,
                row.window_start_ms,
                row.total_requests,
                row.reason_code.clone(),
            )
        })
        .collect();
    assert_eq!(summary_a, summary_b);
}

#[test]
fn pipeline_recovers_after_backpressure_rejection() {
    let simulator = TelemetrySimulator::new(pipeline_config(99)).expect("config should validate");
    let run = simulator.generate().expect("generation should succeed");
    let storage = OpsStorage::open_in_memory().expect("storage should open");

    let first_batch = run.batches.first().expect("run should include first batch");
    let first_records: Vec<_> = first_batch
        .search_events
        .iter()
        .map(|event| event.record.clone())
        .collect();
    let err = storage
        .ingest_search_events_batch(&first_records, 1)
        .expect_err("threshold=1 should trigger backpressure for first batch");
    assert!(
        matches!(err, SearchError::QueueFull { .. }),
        "expected QueueFull, got {err:?}"
    );

    let _ = apply_pipeline(&storage, &run, 8_192).expect("pipeline should recover and succeed");

    let metrics = storage.ingestion_metrics();
    assert!(
        metrics.total_backpressured_batches >= 1,
        "expected backpressure counter to record initial rejection"
    );
    assert!(metrics.total_inserted > 0);
}

#[test]
fn pipeline_performance_entrypoint_enforces_deterministic_budgets() {
    let config = TelemetrySimulatorConfig {
        seed: 777,
        tick_interval_ms: 750,
        ticks: 14,
        projects: vec![
            SimulatedProject {
                project_key: "xf".to_owned(),
                host_name: "xf-load".to_owned(),
                instance_count: 3,
                workload: WorkloadProfile::Burst,
            },
            SimulatedProject {
                project_key: "mail".to_owned(),
                host_name: "mail-load".to_owned(),
                instance_count: 2,
                workload: WorkloadProfile::EmbeddingWave,
            },
        ],
        ..TelemetrySimulatorConfig::default()
    };
    let simulator = TelemetrySimulator::new(config).expect("config should validate");
    let storage = OpsStorage::open_in_memory().expect("storage should open");

    let report = simulator
        .run_performance_entrypoint(&storage, 16_384)
        .expect("performance replay should succeed");

    assert!(
        report.events_ingested >= 300,
        "expected sustained load volume, got {} events",
        report.events_ingested
    );
    assert!(
        report.events_per_second >= 50.0,
        "expected deterministic throughput floor, got {} events/sec",
        report.events_per_second
    );
    assert!(
        (80_000..=300_000).contains(&report.p95_event_latency_us),
        "p95 should stay within deterministic simulator load budget, got {}us",
        report.p95_event_latency_us
    );
    assert!(
        report.avg_write_latency_us < 200_000.0,
        "avg write latency budget exceeded: {}us",
        report.avg_write_latency_us
    );
    assert_eq!(report.backpressured_batches, 0);
}

#[test]
fn pipeline_embedding_wave_surfaces_backlog_and_resource_pressure() {
    let config = TelemetrySimulatorConfig {
        seed: 4242,
        ticks: 10,
        projects: vec![SimulatedProject {
            project_key: "mail".to_owned(),
            host_name: "mail-sat".to_owned(),
            instance_count: 2,
            workload: WorkloadProfile::EmbeddingWave,
        }],
        ..TelemetrySimulatorConfig::default()
    };
    let simulator = TelemetrySimulator::new(config).expect("config should validate");
    let run = simulator.generate().expect("generation should succeed");
    let storage = OpsStorage::open_in_memory().expect("storage should open");

    let _ = apply_pipeline(&storage, &run, 8_192).expect("pipeline should succeed");
    let now_ms = i64::try_from(
        run.batches
            .last()
            .expect("run should include at least one batch")
            .now_ms,
    )
    .expect("timestamp should fit into i64");

    for (project_key, instance_id) in run.instance_pairs() {
        let trend = storage
            .query_resource_trend(
                &project_key,
                &instance_id,
                SummaryWindow::OneHour,
                now_ms,
                256,
            )
            .expect("resource trend query should succeed");
        assert!(!trend.is_empty());

        let max_queue_depth = trend
            .iter()
            .filter_map(|point| point.queue_depth)
            .max()
            .unwrap_or_default();
        let max_rss_bytes = trend
            .iter()
            .filter_map(|point| point.rss_bytes)
            .max()
            .unwrap_or_default();
        assert!(
            max_queue_depth >= 90,
            "expected backlog pressure for {project_key}/{instance_id}, got {max_queue_depth}"
        );
        assert!(
            max_rss_bytes >= (150 * 1024 * 1024),
            "expected saturated RSS for {project_key}/{instance_id}, got {max_rss_bytes}"
        );

        let one_minute = storage
            .latest_search_summary(&project_key, &instance_id, SummaryWindow::OneMinute)
            .expect("search summary query should succeed")
            .expect("expected one-minute summary to exist");
        let p95_latency_us = one_minute
            .p95_latency_us
            .expect("expected p95 latency in one-minute summary");
        assert!(
            p95_latency_us >= 8_000,
            "embedding wave should carry elevated p95 latency, got {p95_latency_us}us"
        );
    }
}

#[test]
fn pipeline_recovers_after_restart_with_telemetry_gap() {
    let config = TelemetrySimulatorConfig {
        seed: 9090,
        ticks: 12,
        projects: vec![SimulatedProject {
            project_key: "frankenterm".to_owned(),
            host_name: "term-restart".to_owned(),
            instance_count: 2,
            workload: WorkloadProfile::Restarting,
        }],
        ..TelemetrySimulatorConfig::default()
    };
    let simulator = TelemetrySimulator::new(config).expect("config should validate");
    let run = simulator.generate().expect("generation should succeed");
    assert!(
        run.batches.len() >= 6,
        "expected enough batches for split replay"
    );

    let split = run.batches.len() / 2;
    let early = &run.batches[..split];
    let late = &run.batches[split + 1..];
    let storage_path = temp_ops_db_path("restart-gap");
    let storage_config = OpsStorageConfig {
        db_path: storage_path,
        busy_timeout_ms: 25,
        ..OpsStorageConfig::default()
    };

    let storage_before_restart =
        OpsStorage::open(storage_config.clone()).expect("storage before restart should open");
    let _ = apply_pipeline_batches(&storage_before_restart, early, 8_192)
        .expect("initial replay should succeed");
    drop(storage_before_restart);

    let storage_after_restart =
        OpsStorage::open(storage_config).expect("storage after restart should open");
    let _ = apply_pipeline_batches(&storage_after_restart, late, 8_192)
        .expect("replay after telemetry gap should succeed");

    let metrics = storage_after_restart.ingestion_metrics();
    assert!(metrics.total_inserted > 0);
    assert_eq!(metrics.total_failed_records, 0);

    for (project_key, instance_id) in run.instance_pairs() {
        let summary = storage_after_restart
            .latest_search_summary(&project_key, &instance_id, SummaryWindow::OneMinute)
            .expect("summary query should succeed");
        assert!(
            summary.is_some(),
            "expected summaries to remain materialized for {project_key}/{instance_id}"
        );
    }

    let fleet_rollups = storage_after_restart
        .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 16)
        .expect("fleet rollup query should succeed");
    assert!(
        !fleet_rollups.is_empty(),
        "expected fleet rollups after restart + telemetry gap replay"
    );
}
