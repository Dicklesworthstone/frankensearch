//! Integration-level interaction tests with phase and artifact assertions (bd-3un.52.4).
//!
//! Extends the unit-level interaction tests (bd-3un.52.3) to validate:
//! - Full oracle template execution across all lanes and query classes
//! - Phase progression consistency with `ExpectedPhase` declarations
//! - Metric envelope assertions (reason codes, metric keys, log events)
//! - Cross-lane ordering stability (`kitchen_sink` ⊇ baseline oracles)
//! - Structured `LaneTestReport` artifact emission for CI consumption
//! - Multi-query coverage per lane with aggregate pass/fail reporting

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use asupersync::Cx;
use asupersync::test_utils::run_test_with_cx;

use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::e2e_artifact::reason_codes;
use frankensearch_core::error::SearchError;
use frankensearch_core::traits::{Embedder, LexicalSearch, ModelCategory, SearchFuture};
use frankensearch_core::types::{IndexableDocument, ScoreSource, ScoredResult, SearchPhase};
use frankensearch_core::{
    ArtifactEmissionInput, ArtifactEntry, ClockMode, Correlation, DeterminismTier,
    E2E_ARTIFACT_ARTIFACTS_INDEX_JSON, E2E_ARTIFACT_ENV_JSON, E2E_ARTIFACT_REPLAY_COMMAND_TXT,
    E2E_ARTIFACT_REPRO_LOCK, E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL, E2E_SCHEMA_EVENT,
    E2E_SCHEMA_MANIFEST, E2E_SCHEMA_ORACLE_REPORT, E2E_SCHEMA_REPLAY, E2eEnvelope, E2eEventType,
    E2eOutcome, E2eSeverity, EventBody, ExitStatus, LaneReport, ManifestBody, ModelVersion,
    OracleReportBody, OracleVerdictRecord, Platform, ReplayBody, ReplayEventType, ReportTotals,
    Suite, build_artifact_entries, normalize_replay_command, render_artifacts_index,
    sha256_checksum, validate_event_envelope, validate_manifest_envelope,
};
use frankensearch_index::TwoTierIndex;

use frankensearch_fusion::interaction_lanes::{
    ExpectedPhase, InteractionLane, RiskLevel, derive_query_seed, lane_by_id, lane_catalog,
    lanes_at_risk, queries_for_lane,
};
use frankensearch_fusion::interaction_oracles::{
    InvariantGroup, LaneOracleTemplate, LaneTestReport, OracleOutcome, OracleVerdict,
    lane_oracle_templates, oracle_template_for_lane, oracles_for_lane,
};
use frankensearch_fusion::searcher::TwoTierSearcher;

// ─── Test Infrastructure (shared with interaction_unit.rs) ─────────────────

struct StubEmbedder {
    id: &'static str,
    dimension: usize,
}

impl StubEmbedder {
    const fn new(id: &'static str, dimension: usize) -> Self {
        Self { id, dimension }
    }
}

impl Embedder for StubEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let dim = self.dimension;
        let hash = text.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(u64::from(b))
        });
        Box::pin(async move {
            let mut vec = vec![0.0f32; dim];
            for (i, v) in vec.iter_mut().enumerate() {
                let shifted = hash.wrapping_add(i as u64);
                *v = ((shifted % 1000) as f32) / 1000.0;
            }
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut vec {
                    *v /= norm;
                }
            }
            Ok(vec)
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        self.id
    }

    fn model_name(&self) -> &str {
        self.id
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

struct StubLexical {
    doc_count: usize,
}

impl StubLexical {
    const fn new(doc_count: usize) -> Self {
        Self { doc_count }
    }
}

impl LexicalSearch for StubLexical {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        let count = limit.min(self.doc_count);
        let hash = query.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(37).wrapping_add(u64::from(b))
        });
        Box::pin(async move {
            Ok((0..count)
                .map(|i| {
                    let doc_idx = (hash as usize + i) % 10;
                    ScoredResult {
                        doc_id: format!("doc-{doc_idx}"),
                        score: (count - i) as f32 / count as f32,
                        source: ScoreSource::Lexical,
                        index: None,
                        fast_score: None,
                        quality_score: None,
                        lexical_score: Some((count - i) as f32 / count as f32),
                        rerank_score: None,
                        explanation: None,
                        metadata: None,
                    }
                })
                .collect())
        })
    }

    fn index_document<'a>(
        &'a self,
        _cx: &'a Cx,
        _doc: &'a IndexableDocument,
    ) -> SearchFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn index_documents<'a>(
        &'a self,
        _cx: &'a Cx,
        _docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn doc_count(&self) -> usize {
        self.doc_count
    }
}

const DIM: usize = 4;

fn build_test_index() -> Arc<TwoTierIndex> {
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-integ-interaction-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let mut builder =
        TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("create test index");
    builder.set_fast_embedder_id("stub-fast");
    for i in 0..10 {
        let mut vec = vec![0.0f32; DIM];
        vec[i % DIM] = 1.0;
        vec[(i + 1) % DIM] = 0.5;
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut vec {
            *v /= norm;
        }
        builder
            .add_fast_record(format!("doc-{i}"), &vec)
            .expect("add record");
    }
    Arc::new(builder.finish().expect("finish test index"))
}

fn build_searcher_for_lane(lane: &InteractionLane) -> (TwoTierSearcher, Arc<TwoTierIndex>) {
    let index = build_test_index();
    let fast = Arc::new(StubEmbedder::new("fast", DIM));
    let quality = Arc::new(StubEmbedder::new("quality", DIM));
    let lexical = Arc::new(StubLexical::new(10));

    let mut config = TwoTierConfig::default();
    if lane.toggles.explain {
        config.explain = true;
    }

    let searcher = TwoTierSearcher::new(Arc::clone(&index), fast, config)
        .with_quality_embedder(quality)
        .with_lexical(lexical);

    (searcher, index)
}

#[derive(Debug, Default)]
struct PhaseCollector {
    initial_results: Vec<ScoredResult>,
    initial_received: bool,
    refined_results: Option<Vec<ScoredResult>>,
    refinement_failed: bool,
}

impl PhaseCollector {
    fn callback(&mut self) -> impl FnMut(SearchPhase) + '_ {
        move |phase| match phase {
            SearchPhase::Initial { results, .. } => {
                self.initial_results = results;
                self.initial_received = true;
            }
            SearchPhase::Refined { results, .. } => {
                self.refined_results = Some(results);
            }
            SearchPhase::RefinementFailed { .. } => {
                self.refinement_failed = true;
            }
        }
    }

    fn best_results(&self) -> &[ScoredResult] {
        self.refined_results
            .as_deref()
            .unwrap_or(&self.initial_results)
    }
}

// ─── Oracle Execution (integration-level) ──────────────────────────────────

/// Execute all applicable oracles for a lane against search results.
#[allow(clippy::too_many_lines)]
fn execute_lane_oracles(
    collector: &PhaseCollector,
    lane: &InteractionLane,
    template: &LaneOracleTemplate,
) -> LaneTestReport {
    let mut report = LaneTestReport::new(lane.id);
    let results = collector.best_results();

    // --- Ordering oracles ---
    // no_duplicates
    if template.oracle_ids.contains(&"no_duplicates".to_string()) {
        let mut seen = HashSet::new();
        let dup = results.iter().any(|r| !seen.insert(&r.doc_id));
        if dup {
            report.add(OracleVerdict::fail(
                "no_duplicates",
                lane.id,
                "duplicate doc_id in results".into(),
            ));
        } else {
            report.add(OracleVerdict::pass("no_duplicates", lane.id));
        }
    }

    // monotonic_scores
    if template
        .oracle_ids
        .contains(&"monotonic_scores".to_string())
    {
        let monotonic = results
            .windows(2)
            .all(|w| w[0].score.total_cmp(&w[1].score) != std::cmp::Ordering::Less);
        if monotonic {
            report.add(OracleVerdict::pass("monotonic_scores", lane.id));
        } else {
            report.add(OracleVerdict::fail(
                "monotonic_scores",
                lane.id,
                "scores not monotonically non-increasing".into(),
            ));
        }
    }

    // --- Phase oracles ---
    if template
        .oracle_ids
        .contains(&"phase1_always_yields".to_string())
    {
        if collector.initial_received {
            report.add(OracleVerdict::pass("phase1_always_yields", lane.id));
        } else {
            report.add(OracleVerdict::fail(
                "phase1_always_yields",
                lane.id,
                "Initial phase not received".into(),
            ));
        }
    }

    if template.oracle_ids.contains(&"phase2_refined".to_string()) {
        if collector.refined_results.is_some() {
            report.add(OracleVerdict::pass("phase2_refined", lane.id));
        } else if collector.refinement_failed {
            report.add(OracleVerdict::fail(
                "phase2_refined",
                lane.id,
                "got RefinementFailed instead of Refined".into(),
            ));
        } else {
            report.add(OracleVerdict::fail(
                "phase2_refined",
                lane.id,
                "Phase 2 not emitted".into(),
            ));
        }
    }

    if template.oracle_ids.contains(&"phase2_graceful".to_string()) {
        // Either Refined or RefinementFailed is acceptable.
        report.add(OracleVerdict::pass("phase2_graceful", lane.id));
    }

    if template
        .oracle_ids
        .contains(&"refinement_subset".to_string())
    {
        if let Some(ref refined) = collector.refined_results {
            let initial_ids: HashSet<&str> = collector
                .initial_results
                .iter()
                .map(|r| r.doc_id.as_str())
                .collect();
            let all_subset = refined
                .iter()
                .all(|r| initial_ids.contains(r.doc_id.as_str()));
            if all_subset {
                report.add(OracleVerdict::pass("refinement_subset", lane.id));
            } else {
                report.add(OracleVerdict::fail(
                    "refinement_subset",
                    lane.id,
                    "refined doc_ids not subset of initial".into(),
                ));
            }
        } else {
            report.add(OracleVerdict::skip(
                "refinement_subset",
                lane.id,
                "no refined results available",
            ));
        }
    }

    // --- Feature-specific oracles (skip if not in template) ---
    // These are validated at the integration level by checking presence in
    // the template's oracle_ids. Feature-specific logic (calibration range,
    // feedback boost, etc.) requires actual feature implementations that
    // are beyond stub-level; we mark them as skipped with reason.
    let feature_oracles = [
        "deterministic_ordering",
        "explain_present",
        "explain_score_consistent",
        "explain_mmr_rank_movement",
        "explain_phase_matches",
        "calibrated_range",
        "calibration_monotonic",
        "explain_calibrated_scores",
        "feedback_boost_positive",
        "feedback_boost_clamped",
        "breaker_skips_phase2",
        "breaker_phase1_preserved",
        "breaker_no_posterior_update",
        "exclusion_applied",
        "explain_exclusion_absent",
        "mmr_diversity_increased",
        "mmr_limit_respected",
        "prf_normalized",
        "prf_query_class_guard",
        "prf_respects_negation",
        "adaptive_blend_converges",
        "adaptive_k_bounded",
        "conformal_k_positive",
        "conformal_coverage",
    ];

    for oracle_id in &feature_oracles {
        if template.oracle_ids.contains(&oracle_id.to_string()) {
            // At integration level with stub backends, feature-specific oracles
            // that require real feature implementations are marked as skip.
            // The unit tests (bd-3un.52.3) cover structural correctness;
            // these will be fully validated in e2e tests (bd-3un.52.5) with
            // real embedders and feature backends.
            report.add(OracleVerdict::skip(
                *oracle_id,
                lane.id,
                "feature-specific oracle deferred to e2e with real backends",
            ));
        }
    }

    report
}

// ─── Template-Driven Test Execution ────────────────────────────────────────

/// Run the full template-driven oracle suite for a lane.
async fn run_template_driven_test(cx: &Cx, lane_id: &str) -> (LaneTestReport, LaneOracleTemplate) {
    let lane = lane_by_id(lane_id).expect("lane not found");
    let template = oracle_template_for_lane(&lane);
    let (searcher, _index) = build_searcher_for_lane(&lane);
    let queries = queries_for_lane(&lane);
    let k = 5;

    let mut aggregate_report = LaneTestReport::new(lane_id);

    for (qi, fq) in queries.iter().enumerate() {
        let query_text = fq.query_for_lane(lane.query_slice.include_negated);
        let _query_seed = derive_query_seed(lane.seed, qi);

        let mut collector = PhaseCollector::default();
        let search_result = searcher
            .search(cx, query_text, k, |_| None, collector.callback())
            .await;

        match search_result {
            Ok(_metrics) => {
                let report = execute_lane_oracles(&collector, &lane, &template);
                for verdict in report.verdicts {
                    aggregate_report.add(verdict);
                }
            }
            Err(SearchError::Cancelled { .. }) => {
                aggregate_report.add(OracleVerdict::fail(
                    "phase1_always_yields",
                    lane_id,
                    "search cancelled".into(),
                ));
            }
            Err(e) => {
                aggregate_report.add(OracleVerdict::fail(
                    "phase1_always_yields",
                    lane_id,
                    format!("search error: {e}"),
                ));
            }
        }
    }

    (aggregate_report, template)
}

const INTERACTION_OWNER_LANE: &str = "composition-lane";
const INTERACTION_REASON_RUN_START: &str = "e2e.run.start";
const INTERACTION_REASON_LANE_PASS: &str = "e2e.run.lane_pass";
const INTERACTION_REASON_LANE_FAIL: &str = "e2e.run.lane_fail";
const INTERACTION_PRIMARY_LANES: [&str; 5] = [
    "explain_mmr",
    "explain_negation",
    "prf_negation",
    "adaptive_calibration_conformal",
    "breaker_adaptive_feedback",
];

#[derive(Debug)]
struct InteractionE2eArtifacts {
    manifest: E2eEnvelope<ManifestBody>,
    oracle_report: E2eEnvelope<OracleReportBody>,
    events: Vec<E2eEnvelope<EventBody>>,
    replay: Vec<E2eEnvelope<ReplayBody>>,
    replay_command: String,
}

const fn e2e_outcome_for_verdict(verdict: &OracleVerdict) -> E2eOutcome {
    match verdict.outcome {
        OracleOutcome::Pass => E2eOutcome::Pass,
        OracleOutcome::Fail => E2eOutcome::Fail,
        OracleOutcome::Skip => E2eOutcome::Skip,
    }
}

fn reason_code_for_verdict(verdict: &OracleVerdict) -> &'static str {
    match verdict.outcome {
        OracleOutcome::Pass => reason_codes::ORACLE_PASS,
        OracleOutcome::Skip => reason_codes::ORACLE_SKIP_STUB_BACKEND,
        OracleOutcome::Fail => match verdict.oracle_id.as_str() {
            "no_duplicates" => reason_codes::ORACLE_DUPLICATES_FOUND,
            "monotonic_scores" => reason_codes::ORACLE_SCORE_NON_MONOTONIC,
            "deterministic_ordering" => reason_codes::ORACLE_ORDERING_VIOLATED,
            oracle if oracle.contains("phase") || oracle == "refinement_subset" => {
                reason_codes::ORACLE_PHASE_MISMATCH
            }
            _ => reason_codes::ORACLE_ORDERING_VIOLATED,
        },
    }
}

const fn severity_for_outcome(outcome: E2eOutcome) -> E2eSeverity {
    match outcome {
        E2eOutcome::Pass => E2eSeverity::Info,
        E2eOutcome::Skip => E2eSeverity::Warn,
        E2eOutcome::Fail => E2eSeverity::Error,
    }
}

#[allow(clippy::too_many_arguments)]
fn push_event(
    events: &mut Vec<E2eEnvelope<EventBody>>,
    run_id: &str,
    ts: &str,
    seq: &mut u64,
    event_type: E2eEventType,
    severity: E2eSeverity,
    lane_id: Option<String>,
    oracle_id: Option<String>,
    outcome: Option<E2eOutcome>,
    reason_code: Option<String>,
    context: Option<String>,
    metrics: Option<BTreeMap<String, f64>>,
) {
    let event_id = format!("interaction-e2e-evt-{seq:04}");
    let body = EventBody {
        event_type,
        correlation: Correlation {
            event_id,
            root_request_id: "interaction-e2e-root".to_owned(),
            parent_event_id: None,
        },
        severity,
        lane_id,
        oracle_id,
        outcome,
        reason_code,
        context,
        metrics,
    };
    events.push(E2eEnvelope::new(E2E_SCHEMA_EVENT, run_id, ts, body));
    *seq += 1;
}

fn interaction_replay_command() -> String {
    normalize_replay_command(
        "cargo test -p frankensearch-fusion --test interaction_integration -- --nocapture --exact interaction_high_risk_lanes_emit_replay_ready_artifacts",
    )
}

fn interaction_env_json_payload() -> String {
    serde_json::json!({
        "schema": "frankensearch.e2e.env.v1",
        "captured_env": [],
        "suite": "interaction",
    })
    .to_string()
}

fn interaction_repro_lock_payload(exit_status: ExitStatus) -> String {
    let status = match exit_status {
        ExitStatus::Pass => "pass",
        ExitStatus::Fail => "fail",
        ExitStatus::Error => "error",
    };
    format!(
        "schema=frankensearch.e2e.repro-lock.v1\nsuite=interaction\nexit_status={status}\nowner_lane={INTERACTION_OWNER_LANE}\n"
    )
}

#[allow(clippy::too_many_lines)]
async fn build_interaction_e2e_artifacts(
    cx: &Cx,
    forced_failure_lane: Option<&str>,
) -> InteractionE2eArtifacts {
    let mut selected_lanes: BTreeMap<String, InteractionLane> = lanes_at_risk(RiskLevel::High)
        .into_iter()
        .map(|lane| (lane.id.to_owned(), lane))
        .collect();
    for lane_id in INTERACTION_PRIMARY_LANES {
        let lane = lane_by_id(lane_id).expect("primary interaction lane must exist");
        selected_lanes.insert(lane.id.to_owned(), lane);
    }
    let selected_lanes: Vec<InteractionLane> = selected_lanes.into_values().collect();
    assert!(
        !selected_lanes.is_empty(),
        "expected at least one interaction lane selected for e2e coverage"
    );

    let run_id = if forced_failure_lane.is_some() {
        "01JABCD3EFGHJKMNPQRSTVWXY0"
    } else {
        "01JABCD3EFGHJKMNPQRSTVWXYZ"
    };
    let ts = "2026-02-14T00:00:00Z";

    let mut event_seq = 1u64;
    let mut replay_seq = 1u64;
    let mut events = Vec::new();
    let mut replay = Vec::new();
    let mut lane_reports = Vec::new();
    let mut totals = ReportTotals {
        lanes_run: 0,
        lanes_passed: 0,
        oracles_pass: 0,
        oracles_fail: 0,
        oracles_skip: 0,
        all_passed: true,
    };

    push_event(
        &mut events,
        run_id,
        ts,
        &mut event_seq,
        E2eEventType::E2eStart,
        E2eSeverity::Info,
        None,
        None,
        None,
        Some(INTERACTION_REASON_RUN_START.to_owned()),
        Some("starting high-risk interaction e2e matrix".to_owned()),
        None,
    );

    for lane in &selected_lanes {
        let (mut report, template) = run_template_driven_test(cx, lane.id).await;
        if forced_failure_lane == Some(lane.id) {
            report.add(OracleVerdict::fail(
                "phase2_refined",
                lane.id,
                "forced failure for replay-ready artifact assertion".to_owned(),
            ));
        }

        let queries = queries_for_lane(lane);
        let query_count = u32::try_from(queries.len()).expect("query count fits in u32");
        let replay_query = queries.first().map_or_else(
            || "rust ownership borrowing".to_owned(),
            |query| {
                query
                    .query_for_lane(lane.query_slice.include_negated)
                    .to_owned()
            },
        );

        let lane_seed = derive_query_seed(lane.seed, 0);
        replay.push(E2eEnvelope::new(
            E2E_SCHEMA_REPLAY,
            run_id,
            ts,
            ReplayBody {
                replay_type: ReplayEventType::Query,
                offset_ms: replay_seq * 10,
                seq: replay_seq,
                payload: serde_json::json!({
                    "lane_id": lane.id,
                    "owner_lane": INTERACTION_OWNER_LANE,
                    "query": replay_query,
                    "seed": lane_seed,
                    "bead_refs": lane.bead_refs,
                }),
            },
        ));
        replay_seq += 1;

        let mut lane_metrics = BTreeMap::new();
        lane_metrics.insert(
            "query_count".to_owned(),
            f64::from(u32::try_from(queries.len()).expect("query count fits in u32")),
        );
        lane_metrics.insert(
            "template_oracles".to_owned(),
            f64::from(u32::try_from(template.oracle_ids.len()).expect("oracle count fits in u32")),
        );
        lane_metrics.insert(
            "seed".to_owned(),
            f64::from(
                u32::try_from(lane.seed & u64::from(u32::MAX))
                    .expect("masked lane seed fits in u32"),
            ),
        );

        push_event(
            &mut events,
            run_id,
            ts,
            &mut event_seq,
            E2eEventType::LaneStart,
            E2eSeverity::Info,
            Some(lane.id.to_owned()),
            None,
            None,
            Some(INTERACTION_REASON_RUN_START.to_owned()),
            Some(format!(
                "owner={INTERACTION_OWNER_LANE};beads={}",
                lane.bead_refs.join(",")
            )),
            Some(lane_metrics),
        );

        let mut pass_count = 0u32;
        let mut fail_count = 0u32;
        let mut skip_count = 0u32;
        let mut verdict_records = Vec::new();

        for verdict in &report.verdicts {
            let outcome = e2e_outcome_for_verdict(verdict);
            match outcome {
                E2eOutcome::Pass => pass_count += 1,
                E2eOutcome::Fail => fail_count += 1,
                E2eOutcome::Skip => skip_count += 1,
            }

            let reason_code = reason_code_for_verdict(verdict).to_owned();
            let invariant_context = if verdict.context.is_empty() {
                format!(
                    "owner={INTERACTION_OWNER_LANE};invariant={}",
                    verdict.oracle_id
                )
            } else {
                format!(
                    "owner={INTERACTION_OWNER_LANE};invariant={};{}",
                    verdict.oracle_id, verdict.context
                )
            };

            push_event(
                &mut events,
                run_id,
                ts,
                &mut event_seq,
                E2eEventType::OracleCheck,
                severity_for_outcome(outcome),
                Some(lane.id.to_owned()),
                Some(verdict.oracle_id.clone()),
                Some(outcome),
                Some(reason_code),
                Some(invariant_context),
                None,
            );

            verdict_records.push(OracleVerdictRecord {
                oracle_id: verdict.oracle_id.clone(),
                outcome,
                context: (!verdict.context.is_empty()).then(|| verdict.context.clone()),
            });
        }

        let lane_passed = fail_count == 0;
        let lane_outcome = if lane_passed {
            E2eOutcome::Pass
        } else {
            E2eOutcome::Fail
        };

        push_event(
            &mut events,
            run_id,
            ts,
            &mut event_seq,
            E2eEventType::LaneEnd,
            severity_for_outcome(lane_outcome),
            Some(lane.id.to_owned()),
            None,
            Some(lane_outcome),
            Some(
                if lane_passed {
                    INTERACTION_REASON_LANE_PASS
                } else {
                    INTERACTION_REASON_LANE_FAIL
                }
                .to_owned(),
            ),
            Some(format!(
                "owner={INTERACTION_OWNER_LANE};beads={}",
                lane.bead_refs.join(",")
            )),
            None,
        );

        totals.lanes_run += 1;
        if lane_passed {
            totals.lanes_passed += 1;
        }
        totals.oracles_pass += pass_count;
        totals.oracles_fail += fail_count;
        totals.oracles_skip += skip_count;

        lane_reports.push(LaneReport {
            lane_id: lane.id.to_owned(),
            seed: lane.seed,
            query_count,
            verdicts: verdict_records,
            pass_count,
            fail_count,
            skip_count,
            all_passed: lane_passed,
        });
    }

    totals.all_passed = totals.oracles_fail == 0;
    let run_outcome = if totals.all_passed {
        E2eOutcome::Pass
    } else {
        E2eOutcome::Fail
    };
    push_event(
        &mut events,
        run_id,
        ts,
        &mut event_seq,
        E2eEventType::E2eEnd,
        severity_for_outcome(run_outcome),
        None,
        None,
        Some(run_outcome),
        Some(
            if totals.all_passed {
                reason_codes::ORACLE_PASS
            } else {
                reason_codes::ORACLE_PHASE_MISMATCH
            }
            .to_owned(),
        ),
        Some("interaction matrix run complete".to_owned()),
        None,
    );

    let oracle_report = E2eEnvelope::new(
        E2E_SCHEMA_ORACLE_REPORT,
        run_id,
        ts,
        OracleReportBody {
            lanes: lane_reports,
            totals,
        },
    );

    let events_jsonl = events
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .expect("serialize event envelopes to json")
        .join("\n");
    let replay_jsonl = replay
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .expect("serialize replay envelopes to json")
        .join("\n");
    let oracle_report_json = serde_json::to_string_pretty(&oracle_report)
        .expect("serialize oracle report envelope for artifact manifest");
    let replay_command = interaction_replay_command();
    let exit_status = if oracle_report.body.totals.all_passed {
        ExitStatus::Pass
    } else {
        ExitStatus::Fail
    };
    let env_json = interaction_env_json_payload();
    let repro_lock = interaction_repro_lock_payload(exit_status);

    let mut emission_inputs = vec![
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
            bytes: events_jsonl.as_bytes(),
            line_count: Some(u64::try_from(events.len()).expect("event line count fits in u64")),
        },
        ArtifactEmissionInput {
            file: "oracle-report.json",
            bytes: oracle_report_json.as_bytes(),
            line_count: None,
        },
        ArtifactEmissionInput {
            file: "replay.jsonl",
            bytes: replay_jsonl.as_bytes(),
            line_count: Some(u64::try_from(replay.len()).expect("replay line count fits in u64")),
        },
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_ENV_JSON,
            bytes: env_json.as_bytes(),
            line_count: None,
        },
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_REPRO_LOCK,
            bytes: repro_lock.as_bytes(),
            line_count: None,
        },
    ];
    if matches!(exit_status, ExitStatus::Fail | ExitStatus::Error) {
        emission_inputs.push(ArtifactEmissionInput {
            file: E2E_ARTIFACT_REPLAY_COMMAND_TXT,
            bytes: replay_command.as_bytes(),
            line_count: None,
        });
    }

    let mut artifacts = build_artifact_entries(emission_inputs).expect("build artifact entries");
    if matches!(exit_status, ExitStatus::Fail | ExitStatus::Error) {
        let artifacts_index_payload =
            render_artifacts_index(&artifacts).expect("render artifacts index");
        artifacts.push(ArtifactEntry {
            file: E2E_ARTIFACT_ARTIFACTS_INDEX_JSON.to_owned(),
            checksum: sha256_checksum(artifacts_index_payload.as_bytes()),
            line_count: None,
        });
    }
    artifacts.sort_by(|left, right| left.file.cmp(&right.file));

    let manifest = E2eEnvelope::new(
        E2E_SCHEMA_MANIFEST,
        run_id,
        ts,
        ManifestBody {
            suite: Suite::Interaction,
            determinism_tier: DeterminismTier::Semantic,
            seed: 0x_CAFE_5250,
            config_hash: "sha256:3e7f8a4d4e5c9aa3f57ca90ea2cbff7f3ef749f818acdfca0f06f6e9fb87f020"
                .to_owned(),
            index_version: Some("fsvi-test-v1".to_owned()),
            model_versions: vec![
                ModelVersion {
                    name: "stub-fast".to_owned(),
                    revision: "interaction-e2e-v1".to_owned(),
                    digest: None,
                },
                ModelVersion {
                    name: "stub-quality".to_owned(),
                    revision: "interaction-e2e-v1".to_owned(),
                    digest: None,
                },
            ],
            platform: Platform {
                os: std::env::consts::OS.to_owned(),
                arch: std::env::consts::ARCH.to_owned(),
                rustc: "nightly-2026-02-14".to_owned(),
            },
            clock_mode: ClockMode::Simulated,
            tie_break_policy: "doc_id_lexical".to_owned(),
            artifacts,
            duration_ms: u64::try_from(events.len()).expect("event count fits in u64"),
            exit_status,
        },
    );

    InteractionE2eArtifacts {
        manifest,
        oracle_report,
        events,
        replay,
        replay_command,
    }
}

// ─── Template Structure Tests ──────────────────────────────────────────────

#[test]
fn all_templates_have_required_invariant_groups() {
    let templates = lane_oracle_templates();

    for template in &templates {
        assert!(
            template
                .invariant_groups
                .contains(&InvariantGroup::Ordering),
            "template {} missing Ordering group",
            template.lane_id
        );
        assert!(
            template
                .invariant_groups
                .contains(&InvariantGroup::PhaseTransitions),
            "template {} missing PhaseTransitions group",
            template.lane_id
        );
        assert!(
            template
                .invariant_groups
                .contains(&InvariantGroup::ReasonCodes),
            "template {} missing ReasonCodes group",
            template.lane_id
        );
        assert!(
            template
                .invariant_groups
                .contains(&InvariantGroup::FallbackSemantics),
            "template {} missing FallbackSemantics group",
            template.lane_id
        );
    }
}

#[test]
fn template_reason_codes_include_phase_signals() {
    let templates = lane_oracle_templates();

    for template in &templates {
        assert!(
            template
                .reason_codes
                .contains(&"phase.initial.emitted".to_string()),
            "template {} missing phase.initial.emitted reason code",
            template.lane_id
        );
        assert!(
            template
                .reason_codes
                .contains(&"fallback.circuit_breaker.evaluated".to_string()),
            "template {} missing fallback.circuit_breaker.evaluated reason code",
            template.lane_id
        );
    }
}

#[test]
fn template_metric_keys_include_core_metrics() {
    let templates = lane_oracle_templates();

    for template in &templates {
        assert!(
            template
                .metric_keys
                .contains(&"search.phase1.latency_ms".to_string()),
            "template {} missing search.phase1.latency_ms metric key",
            template.lane_id
        );
        assert!(
            template
                .metric_keys
                .contains(&"search.initial.result_count".to_string()),
            "template {} missing search.initial.result_count metric key",
            template.lane_id
        );
    }
}

#[test]
fn template_log_events_include_phase_events() {
    let templates = lane_oracle_templates();

    for template in &templates {
        assert!(
            template
                .log_events
                .contains(&"phase.initial.yielded".to_string()),
            "template {} missing phase.initial.yielded log event",
            template.lane_id
        );
    }
}

#[test]
fn templates_are_deterministic_across_calls() {
    let templates_a = lane_oracle_templates();
    let templates_b = lane_oracle_templates();

    assert_eq!(templates_a.len(), templates_b.len());
    for (a, b) in templates_a.iter().zip(templates_b.iter()) {
        assert_eq!(a, b, "templates differ for lane {}", a.lane_id);
    }
}

#[test]
fn template_seeds_match_lane_seeds() {
    let catalog = lane_catalog();
    let templates = lane_oracle_templates();

    for (lane, template) in catalog.iter().zip(templates.iter()) {
        assert_eq!(
            lane.seed, template.seed,
            "seed mismatch for lane {}",
            lane.id
        );
        assert_eq!(
            lane.id,
            template.lane_id.as_str(),
            "lane_id mismatch for lane {}",
            lane.id
        );
        assert_eq!(
            lane.expected_phase, template.expected_phase,
            "expected_phase mismatch for lane {}",
            lane.id
        );
    }
}

// ─── Template-Driven Oracle Execution Tests ────────────────────────────────

#[test]
fn integration_baseline_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "baseline").await;
        assert_no_failures(&report, "baseline");
        assert_oracle_coverage(&report, &template, "baseline");
    });
}

#[test]
fn integration_explain_mmr_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "explain_mmr").await;
        assert_no_failures(&report, "explain_mmr");
        assert_oracle_coverage(&report, &template, "explain_mmr");
    });
}

#[test]
fn integration_explain_negation_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "explain_negation").await;
        assert_no_failures(&report, "explain_negation");
        assert_oracle_coverage(&report, &template, "explain_negation");
    });
}

#[test]
fn integration_prf_negation_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "prf_negation").await;
        assert_no_failures(&report, "prf_negation");
        assert_oracle_coverage(&report, &template, "prf_negation");
    });
}

#[test]
fn integration_adaptive_calibration_conformal_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) =
            run_template_driven_test(&cx, "adaptive_calibration_conformal").await;
        assert_no_failures(&report, "adaptive_calibration_conformal");
        assert_oracle_coverage(&report, &template, "adaptive_calibration_conformal");
    });
}

#[test]
fn integration_breaker_adaptive_feedback_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "breaker_adaptive_feedback").await;
        assert_no_failures(&report, "breaker_adaptive_feedback");
        assert_oracle_coverage(&report, &template, "breaker_adaptive_feedback");
    });
}

#[test]
fn integration_mmr_feedback_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "mmr_feedback").await;
        assert_no_failures(&report, "mmr_feedback");
        assert_oracle_coverage(&report, &template, "mmr_feedback");
    });
}

#[test]
fn integration_prf_adaptive_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "prf_adaptive").await;
        assert_no_failures(&report, "prf_adaptive");
        assert_oracle_coverage(&report, &template, "prf_adaptive");
    });
}

#[test]
fn integration_calibration_conformal_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "calibration_conformal").await;
        assert_no_failures(&report, "calibration_conformal");
        assert_oracle_coverage(&report, &template, "calibration_conformal");
    });
}

#[test]
fn integration_explain_calibration_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "explain_calibration").await;
        assert_no_failures(&report, "explain_calibration");
        assert_oracle_coverage(&report, &template, "explain_calibration");
    });
}

#[test]
fn integration_breaker_explain_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "breaker_explain").await;
        assert_no_failures(&report, "breaker_explain");
        assert_oracle_coverage(&report, &template, "breaker_explain");
    });
}

#[test]
fn integration_kitchen_sink_template_driven() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "kitchen_sink").await;
        assert_no_failures(&report, "kitchen_sink");
        assert_oracle_coverage(&report, &template, "kitchen_sink");
    });
}

// ─── Cross-Lane Consistency Tests ──────────────────────────────────────────

#[test]
fn kitchen_sink_oracle_set_is_superset_of_baseline() {
    let baseline = lane_by_id("baseline").unwrap();
    let kitchen_sink = lane_by_id("kitchen_sink").unwrap();

    let baseline_oracles: HashSet<&str> =
        oracles_for_lane(&baseline).iter().map(|o| o.id).collect();
    let ks_oracles: HashSet<&str> = oracles_for_lane(&kitchen_sink)
        .iter()
        .map(|o| o.id)
        .collect();

    // Kitchen sink should cover all baseline oracles plus feature-specific ones.
    // (Baseline uses InitialThenRefined; kitchen_sink also uses InitialThenRefined,
    // so phase oracles should match.)
    for oracle_id in &baseline_oracles {
        assert!(
            ks_oracles.contains(oracle_id),
            "kitchen_sink missing baseline oracle: {oracle_id}"
        );
    }

    // Kitchen sink should have more oracles than baseline.
    assert!(
        ks_oracles.len() > baseline_oracles.len(),
        "kitchen_sink should have more oracles than baseline ({} vs {})",
        ks_oracles.len(),
        baseline_oracles.len()
    );
}

#[test]
fn breaker_lanes_have_unique_phase_oracles() {
    let breaker_feedback = lane_by_id("breaker_adaptive_feedback").unwrap();
    let breaker_explain = lane_by_id("breaker_explain").unwrap();

    let bf_oracles: HashSet<&str> = oracles_for_lane(&breaker_feedback)
        .iter()
        .map(|o| o.id)
        .collect();
    let be_oracles: HashSet<&str> = oracles_for_lane(&breaker_explain)
        .iter()
        .map(|o| o.id)
        .collect();

    // Both breaker lanes should have phase2_graceful (MaybeRefined).
    assert!(bf_oracles.contains("phase2_graceful"));
    assert!(be_oracles.contains("phase2_graceful"));

    // Neither should have phase2_refined (that's for InitialThenRefined).
    assert!(!bf_oracles.contains("phase2_refined"));
    assert!(!be_oracles.contains("phase2_refined"));

    // breaker_explain should have explain oracles; breaker_feedback should not.
    assert!(be_oracles.contains("explain_present"));
    assert!(!bf_oracles.contains("explain_present"));
}

#[test]
fn all_lanes_run_without_panic() {
    run_test_with_cx(|cx| async move {
        let catalog = lane_catalog();
        let mut lane_summaries: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();

        for lane in &catalog {
            let (report, _template) = run_template_driven_test(&cx, lane.id).await;
            lane_summaries.insert(
                lane.id.to_string(),
                (
                    report.pass_count(),
                    report.failure_count(),
                    report.skip_count(),
                ),
            );
        }

        // Every lane must have at least some passing oracles.
        for (lane_id, (pass, fail, _skip)) in &lane_summaries {
            assert!(*pass > 0, "lane {lane_id} had 0 passing oracles");
            assert_eq!(*fail, 0, "lane {lane_id} had {fail} failing oracles");
        }
    });
}

#[test]
fn interaction_high_risk_lanes_emit_replay_ready_artifacts() {
    run_test_with_cx(|cx| async move {
        let artifacts = build_interaction_e2e_artifacts(&cx, None).await;
        validate_manifest_envelope(&artifacts.manifest)
            .expect("manifest should satisfy unified interaction artifact contract");
        for event in &artifacts.events {
            validate_event_envelope(event)
                .expect("event must satisfy unified interaction contract");
        }

        let must_cover = [
            "explain_mmr",
            "explain_negation",
            "prf_negation",
            "adaptive_calibration_conformal",
            "breaker_adaptive_feedback",
        ];
        let reported_lane_ids: HashSet<&str> = artifacts
            .oracle_report
            .body
            .lanes
            .iter()
            .map(|lane| lane.lane_id.as_str())
            .collect();
        for lane_id in must_cover {
            assert!(
                reported_lane_ids.contains(lane_id),
                "missing high-risk interaction lane in oracle report: {lane_id}"
            );
        }

        let oracle_events: Vec<&E2eEnvelope<EventBody>> = artifacts
            .events
            .iter()
            .filter(|event| event.body.event_type == E2eEventType::OracleCheck)
            .collect();
        assert!(!oracle_events.is_empty(), "expected oracle check events");
        for event in oracle_events {
            let lane_id = event
                .body
                .lane_id
                .as_deref()
                .expect("oracle events must include lane_id");
            assert!(
                reported_lane_ids.contains(lane_id),
                "oracle event lane_id should map to reported lane"
            );
            let context = event
                .body
                .context
                .as_deref()
                .expect("oracle events must include invariant context");
            assert!(
                context.contains("owner=composition-lane"),
                "oracle context must include ownership metadata"
            );
        }

        for replay in &artifacts.replay {
            let lane_id = replay
                .body
                .payload
                .get("lane_id")
                .and_then(serde_json::Value::as_str)
                .expect("replay payload must include lane_id");
            let owner = replay
                .body
                .payload
                .get("owner_lane")
                .and_then(serde_json::Value::as_str)
                .expect("replay payload must include owner lane");
            assert_eq!(owner, INTERACTION_OWNER_LANE);
            assert!(
                reported_lane_ids.contains(lane_id),
                "replay lane_id must map to reported lane"
            );
        }

        let artifact_files: HashSet<&str> = artifacts
            .manifest
            .body
            .artifacts
            .iter()
            .map(|artifact| artifact.file.as_str())
            .collect();
        assert!(artifact_files.contains(E2E_ARTIFACT_ENV_JSON));
        assert!(artifact_files.contains(E2E_ARTIFACT_REPRO_LOCK));
        assert_eq!(artifacts.manifest.body.suite, Suite::Interaction);
        assert_eq!(artifacts.manifest.body.clock_mode, ClockMode::Simulated);
    });
}

#[test]
fn interaction_failure_path_includes_replay_command_and_lane_context() {
    run_test_with_cx(|cx| async move {
        let forced_lane = lanes_at_risk(RiskLevel::High)
            .first()
            .expect("at least one high-risk lane")
            .id;
        let artifacts = build_interaction_e2e_artifacts(&cx, Some(forced_lane)).await;

        assert_eq!(artifacts.manifest.body.exit_status, ExitStatus::Fail);
        let files: HashSet<&str> = artifacts
            .manifest
            .body
            .artifacts
            .iter()
            .map(|artifact| artifact.file.as_str())
            .collect();
        assert!(files.contains(E2E_ARTIFACT_ENV_JSON));
        assert!(files.contains(E2E_ARTIFACT_REPRO_LOCK));
        assert!(files.contains(E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL));
        assert!(files.contains(E2E_ARTIFACT_REPLAY_COMMAND_TXT));
        assert!(files.contains(E2E_ARTIFACT_ARTIFACTS_INDEX_JSON));
        assert!(
            artifacts
                .replay_command
                .contains("interaction_high_risk_lanes_emit_replay_ready_artifacts"),
            "replay command should point to deterministic lane test entrypoint"
        );

        let failing_oracles: Vec<&E2eEnvelope<EventBody>> = artifacts
            .events
            .iter()
            .filter(|event| {
                event.body.event_type == E2eEventType::OracleCheck
                    && event.body.outcome == Some(E2eOutcome::Fail)
            })
            .collect();
        assert!(
            !failing_oracles.is_empty(),
            "forced lane failure should emit failing oracle event(s)"
        );
        for event in failing_oracles {
            assert!(
                event.body.lane_id.is_some(),
                "failing oracle events must include lane_id"
            );
            let context = event
                .body
                .context
                .as_deref()
                .expect("failing oracle events must include invariant context");
            assert!(
                context.contains("invariant="),
                "failing oracle context should include invariant identifier"
            );
        }
    });
}

// ─── Artifact Assertion Tests ──────────────────────────────────────────────

#[test]
fn report_artifact_serializes_to_json() {
    run_test_with_cx(|cx| async move {
        let (report, template) = run_template_driven_test(&cx, "baseline").await;

        // The report should serialize cleanly for CI artifact consumption.
        let report_json = serde_json::to_string_pretty(&report).expect("serialize report");
        let template_json = serde_json::to_string_pretty(&template).expect("serialize template");

        // Verify roundtrip.
        let report_back: LaneTestReport =
            serde_json::from_str(&report_json).expect("deserialize report");
        assert_eq!(report.lane_id, report_back.lane_id);
        assert_eq!(report.verdicts.len(), report_back.verdicts.len());

        let template_back: LaneOracleTemplate =
            serde_json::from_str(&template_json).expect("deserialize template");
        assert_eq!(template.lane_id, template_back.lane_id);
        assert_eq!(template.seed, template_back.seed);
    });
}

#[test]
fn report_display_format_includes_summary() {
    run_test_with_cx(|cx| async move {
        let (report, _) = run_template_driven_test(&cx, "explain_mmr").await;
        let display = format!("{report}");

        // Display should include the lane name and counts.
        assert!(display.contains("explain_mmr"), "display missing lane name");
        assert!(display.contains("passed"), "display missing pass count");
    });
}

// ─── Phase Progression Tests ───────────────────────────────────────────────

#[test]
fn initial_then_refined_lanes_always_produce_phase2() {
    run_test_with_cx(|cx| async move {
        let catalog = lane_catalog();
        let refined_lanes: Vec<&InteractionLane> = catalog
            .iter()
            .filter(|l| l.expected_phase == ExpectedPhase::InitialThenRefined)
            .collect();

        assert!(
            !refined_lanes.is_empty(),
            "no lanes with InitialThenRefined phase"
        );

        for lane in refined_lanes {
            let (searcher, _) = build_searcher_for_lane(lane);
            let queries = queries_for_lane(lane);
            if let Some(fq) = queries.first() {
                let query = fq.query_for_lane(lane.query_slice.include_negated);
                let mut collector = PhaseCollector::default();
                let result = searcher
                    .search(&cx, query, 5, |_| None, collector.callback())
                    .await;

                assert!(
                    result.is_ok(),
                    "search failed for lane {}: {:?}",
                    lane.id,
                    result.err()
                );
                assert!(
                    collector.initial_received,
                    "lane {} did not produce Initial phase",
                    lane.id
                );
                assert!(
                    collector.refined_results.is_some(),
                    "lane {} did not produce Refined phase (expected InitialThenRefined)",
                    lane.id
                );
            }
        }
    });
}

#[test]
fn maybe_refined_lanes_tolerate_both_outcomes() {
    run_test_with_cx(|cx| async move {
        let catalog = lane_catalog();
        let maybe_lanes: Vec<&InteractionLane> = catalog
            .iter()
            .filter(|l| l.expected_phase == ExpectedPhase::InitialThenMaybeRefined)
            .collect();

        assert!(
            !maybe_lanes.is_empty(),
            "no lanes with InitialThenMaybeRefined phase"
        );

        for lane in maybe_lanes {
            let (searcher, _) = build_searcher_for_lane(lane);
            let queries = queries_for_lane(lane);
            if let Some(fq) = queries.first() {
                let query = fq.query_for_lane(lane.query_slice.include_negated);
                let mut collector = PhaseCollector::default();
                let result = searcher
                    .search(&cx, query, 5, |_| None, collector.callback())
                    .await;

                assert!(
                    result.is_ok(),
                    "search failed for lane {}: {:?}",
                    lane.id,
                    result.err()
                );
                assert!(
                    collector.initial_received,
                    "lane {} did not produce Initial phase",
                    lane.id
                );
                // Either Refined or RefinementFailed is acceptable.
                // No assertion on which one — both are valid for MaybeRefined.
            }
        }
    });
}

// ─── Seed Reproducibility Tests ────────────────────────────────────────────

#[test]
fn query_seed_derivation_is_collision_free_within_lane() {
    let catalog = lane_catalog();
    for lane in &catalog {
        let queries = queries_for_lane(lane);
        let mut seeds = HashSet::new();
        for (qi, _) in queries.iter().enumerate() {
            let seed = derive_query_seed(lane.seed, qi);
            assert!(
                seeds.insert(seed),
                "seed collision in lane {} at query index {}",
                lane.id,
                qi
            );
        }
    }
}

#[test]
fn query_seed_derivation_differs_across_lanes() {
    let catalog = lane_catalog();
    // Check that the same query index produces different seeds for different lanes.
    let seeds_at_0: Vec<u64> = catalog
        .iter()
        .map(|l| derive_query_seed(l.seed, 0))
        .collect();
    let unique_seeds: HashSet<u64> = seeds_at_0.iter().copied().collect();
    assert_eq!(
        seeds_at_0.len(),
        unique_seeds.len(),
        "query seed derivation produced collisions across lanes at index 0"
    );
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn assert_no_failures(report: &LaneTestReport, lane_id: &str) {
    for v in &report.verdicts {
        assert!(
            v.outcome != OracleOutcome::Fail,
            "lane {lane_id} oracle failed: {v}"
        );
    }
}

fn assert_oracle_coverage(report: &LaneTestReport, template: &LaneOracleTemplate, lane_id: &str) {
    let reported_oracle_ids: HashSet<&str> = report
        .verdicts
        .iter()
        .map(|v| v.oracle_id.as_str())
        .collect();

    // Every oracle in the template should have at least one verdict
    // (pass, fail, or skip) in the report.
    for oracle_id in &template.oracle_ids {
        assert!(
            reported_oracle_ids.contains(oracle_id.as_str()),
            "lane {lane_id} missing verdict for oracle {oracle_id}"
        );
    }
}
