//! Profiling harness contracts for fsfs optimization cycles.
//!
//! This module defines:
//! - a deterministic profiling workflow (flamegraph/heap/syscall),
//! - an impact-confidence-effort opportunity matrix,
//! - a single-lever iteration validator for behavior-preserving optimization.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Schema version for the profiling workflow contract.
pub const PROFILING_WORKFLOW_SCHEMA_VERSION: &str = "fsfs-profiling-workflow-v1";
/// Schema version for the opportunity-matrix contract.
pub const OPPORTUNITY_MATRIX_SCHEMA_VERSION: &str = "fsfs-opportunity-matrix-v1";
/// Schema version for crawl/ingest optimization track planning.
pub const CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION: &str = "fsfs-crawl-ingest-opt-track-v1";

/// Reason code emitted when an optimization iteration is accepted.
pub const ITERATION_REASON_ACCEPTED: &str = "opt.iteration.accepted";
/// Reason code emitted when no lever changed.
pub const ITERATION_REASON_NO_CHANGE: &str = "opt.iteration.invalid.no_change";
/// Reason code emitted when more than one lever changed.
pub const ITERATION_REASON_MULTI_CHANGE: &str = "opt.iteration.invalid.multiple_levers";

/// Profile lane required by the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileKind {
    /// CPU hotspot profile (flamegraph).
    Flamegraph,
    /// Heap/allocation profile.
    Heap,
    /// Syscall profile.
    Syscall,
}

impl std::fmt::Display for ProfileKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flamegraph => write!(f, "flamegraph"),
            Self::Heap => write!(f, "heap"),
            Self::Syscall => write!(f, "syscall"),
        }
    }
}

/// One deterministic profiling step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileStep {
    /// Profile lane this step captures.
    pub kind: ProfileKind,
    /// Human-readable label used in manifests and logs.
    pub label: String,
    /// Command template to run.
    pub command: String,
    /// Artifact path template.
    pub artifact_path: String,
}

/// Deterministic profile workflow for a dataset profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileWorkflow {
    /// Contract schema version.
    pub schema_version: String,
    /// Dataset profile (tiny/small/medium/etc).
    pub dataset_profile: String,
    /// Ordered profile steps.
    pub steps: Vec<ProfileStep>,
}

impl ProfileWorkflow {
    /// Build the default profiling workflow for a dataset profile.
    #[must_use]
    pub fn for_dataset_profile(dataset_profile: &str) -> Self {
        let normalized = dataset_profile.trim();
        let normalized = if normalized.is_empty() {
            "small"
        } else {
            normalized
        };

        Self {
            schema_version: PROFILING_WORKFLOW_SCHEMA_VERSION.to_owned(),
            dataset_profile: normalized.to_owned(),
            steps: vec![
                ProfileStep {
                    kind: ProfileKind::Flamegraph,
                    label: "cpu.hotspots".to_owned(),
                    command: format!(
                        "cargo flamegraph -p frankensearch-fsfs --test benchmark_baseline_matrix -- --profile {normalized}"
                    ),
                    artifact_path: format!("profiles/{normalized}/flamegraph.svg"),
                },
                ProfileStep {
                    kind: ProfileKind::Heap,
                    label: "heap.allocations".to_owned(),
                    command: format!(
                        "heaptrack target/release/fsfs --mode benchmark --profile {normalized}"
                    ),
                    artifact_path: format!("profiles/{normalized}/heaptrack.out"),
                },
                ProfileStep {
                    kind: ProfileKind::Syscall,
                    label: "syscalls.io".to_owned(),
                    command: format!(
                        "strace -ff -ttT -o profiles/{normalized}/syscall target/release/fsfs --mode benchmark --profile {normalized}"
                    ),
                    artifact_path: format!("profiles/{normalized}/syscall.*"),
                },
            ],
        }
    }

    /// Materialize deterministic artifact entries for a run.
    #[must_use]
    pub fn artifact_manifest(&self, run_id: &str) -> Vec<ProfileArtifact> {
        self.steps
            .iter()
            .map(|step| ProfileArtifact {
                kind: step.kind,
                artifact_path: format!("{run_id}/{}", step.artifact_path),
                replay_command: format!(
                    "fsfs profile replay --run-id {run_id} --kind {}",
                    step.kind
                ),
            })
            .collect()
    }
}

/// Artifact descriptor emitted by the harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileArtifact {
    /// Profile lane.
    pub kind: ProfileKind,
    /// Path relative to benchmark artifact root.
    pub artifact_path: String,
    /// Replay command for deterministic triage.
    pub replay_command: String,
}

/// One candidate optimization opportunity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunityCandidate {
    /// Stable candidate id.
    pub id: String,
    /// Candidate summary.
    pub summary: String,
    /// Expected impact (0-100).
    pub impact: u16,
    /// Confidence in estimate (0-100).
    pub confidence: u16,
    /// Effort cost (1-100), lower is better.
    pub effort: u16,
}

impl OpportunityCandidate {
    /// Deterministic ICE score in per-mille units.
    ///
    /// Score = `(impact * confidence * 1000) / effort`.
    #[must_use]
    pub fn ice_score_per_mille(&self) -> u32 {
        let effort = if self.effort == 0 { 1 } else { self.effort };
        (u32::from(self.impact) * u32::from(self.confidence) * 1_000) / u32::from(effort)
    }
}

/// Opportunity scoring table for optimization planning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunityMatrix {
    /// Contract schema version.
    pub schema_version: String,
    /// Candidate table.
    pub candidates: Vec<OpportunityCandidate>,
}

impl OpportunityMatrix {
    /// Build a matrix from candidates.
    #[must_use]
    pub fn new(candidates: Vec<OpportunityCandidate>) -> Self {
        Self {
            schema_version: OPPORTUNITY_MATRIX_SCHEMA_VERSION.to_owned(),
            candidates,
        }
    }

    /// Return deterministically ranked candidates.
    ///
    /// Tie-break order:
    /// 1. ICE score descending
    /// 2. impact descending
    /// 3. confidence descending
    /// 4. effort ascending
    /// 5. id lexicographic ascending
    #[must_use]
    pub fn ranked(&self) -> Vec<RankedOpportunity> {
        let mut ranked: Vec<RankedOpportunity> = self
            .candidates
            .iter()
            .cloned()
            .map(|candidate| RankedOpportunity {
                rank: 0,
                ice_score_per_mille: candidate.ice_score_per_mille(),
                candidate,
            })
            .collect();

        ranked.sort_by(|left, right| {
            right
                .ice_score_per_mille
                .cmp(&left.ice_score_per_mille)
                .then_with(|| right.candidate.impact.cmp(&left.candidate.impact))
                .then_with(|| right.candidate.confidence.cmp(&left.candidate.confidence))
                .then_with(|| left.candidate.effort.cmp(&right.candidate.effort))
                .then_with(|| left.candidate.id.cmp(&right.candidate.id))
        });

        for (index, candidate) in ranked.iter_mut().enumerate() {
            candidate.rank = index + 1;
        }

        ranked
    }
}

/// Ranked matrix row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankedOpportunity {
    /// 1-based rank.
    pub rank: usize,
    /// ICE score in per-mille units.
    pub ice_score_per_mille: u32,
    /// Original candidate row.
    pub candidate: OpportunityCandidate,
}

/// Canonical crawl/ingest stages used by the optimization track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrawlIngestStage {
    DiscoveryWalk,
    Classification,
    CatalogMutation,
    QueueAdmission,
    EmbeddingGate,
}

/// Ranked hotspot entry for the crawl/ingest path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrawlIngestHotspot {
    /// 1-based rank derived from ICE score ordering.
    pub rank: usize,
    /// Stable optimization lever id.
    pub lever_id: String,
    /// Crawl/ingest stage targeted by this lever.
    pub stage: CrawlIngestStage,
    /// Human-readable optimization summary.
    pub summary: String,
    /// Expected p50 latency improvement percentage.
    pub expected_p50_gain_pct: u8,
    /// Expected p95 latency improvement percentage.
    pub expected_p95_gain_pct: u8,
    /// Expected ingest throughput improvement percentage.
    pub expected_throughput_gain_pct: u8,
}

/// Isomorphism proof checklist item for one optimization lever.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsomorphismProofChecklistItem {
    /// Optimization lever being validated.
    pub lever_id: String,
    /// Baseline behavior this lever must preserve.
    pub baseline_comparator: String,
    /// Explicit invariants that must remain true.
    pub required_invariants: Vec<String>,
    /// Deterministic replay command for triage/proof.
    pub replay_command: String,
}

/// Deterministic rollback guardrail for one optimization lever.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackGuardrail {
    /// Optimization lever protected by this guardrail.
    pub lever_id: String,
    /// Rollback command to execute when abort conditions are met.
    pub rollback_command: String,
    /// Reason codes that force rollback.
    pub abort_reason_codes: Vec<String>,
    /// Reason code expected after rollback succeeds.
    pub recovery_reason_code: String,
}

/// Complete crawl/ingest optimization track contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrawlIngestOptimizationTrack {
    /// Contract schema version.
    pub schema_version: String,
    /// Prioritized hotspot list with expected gains.
    pub hotspots: Vec<CrawlIngestHotspot>,
    /// Behavior-preserving proof checklist for every lever.
    pub proof_checklist: Vec<IsomorphismProofChecklistItem>,
    /// Rollback guardrails per optimization class.
    pub rollback_guardrails: Vec<RollbackGuardrail>,
}

/// Build the canonical crawl/ingest optimization opportunity matrix.
#[must_use]
pub fn crawl_ingest_opportunity_matrix() -> OpportunityMatrix {
    OpportunityMatrix::new(vec![
        OpportunityCandidate {
            id: "ingest.catalog.batch_upsert".into(),
            summary: "Batch catalog/changelog writes to reduce transaction overhead".into(),
            impact: 90,
            confidence: 90,
            effort: 18,
        },
        OpportunityCandidate {
            id: "crawl.classification.policy_batching".into(),
            summary: "Classify discovery candidates in batched policy windows".into(),
            impact: 68,
            confidence: 78,
            effort: 14,
        },
        OpportunityCandidate {
            id: "ingest.queue.lane_budget_admission".into(),
            summary: "Use lane-budget aware queue admission to reduce saturation churn".into(),
            impact: 74,
            confidence: 80,
            effort: 16,
        },
        OpportunityCandidate {
            id: "crawl.discovery.path_metadata_cache".into(),
            summary: "Cache path metadata during crawl to cut repeated stat/syscall work".into(),
            impact: 82,
            confidence: 82,
            effort: 20,
        },
        OpportunityCandidate {
            id: "ingest.embed_gate.early_skip".into(),
            summary: "Apply early embedding skip gates for low-signal candidates".into(),
            impact: 76,
            confidence: 88,
            effort: 24,
        },
    ])
}

/// Build the canonical crawl/ingest optimization track with hotspots, proofs,
/// and rollback guardrails.
#[must_use]
pub fn crawl_ingest_optimization_track() -> CrawlIngestOptimizationTrack {
    let ranked = crawl_ingest_opportunity_matrix().ranked();
    let hotspots = ranked
        .iter()
        .map(|entry| {
            let (stage, p50_gain, p95_gain, throughput_gain) =
                hotspot_expectations_for(&entry.candidate.id);
            CrawlIngestHotspot {
                rank: entry.rank,
                lever_id: entry.candidate.id.clone(),
                stage,
                summary: entry.candidate.summary.clone(),
                expected_p50_gain_pct: p50_gain,
                expected_p95_gain_pct: p95_gain,
                expected_throughput_gain_pct: throughput_gain,
            }
        })
        .collect::<Vec<_>>();

    let proof_checklist = hotspots
        .iter()
        .map(|hotspot| IsomorphismProofChecklistItem {
            lever_id: hotspot.lever_id.clone(),
            baseline_comparator: baseline_comparator_for(hotspot.stage).to_owned(),
            required_invariants: invariants_for_stage(hotspot.stage)
                .iter()
                .map(ToString::to_string)
                .collect(),
            replay_command: format!(
                "fsfs profile replay --lane ingest --lever-id {} --compare baseline",
                hotspot.lever_id
            ),
        })
        .collect::<Vec<_>>();

    let rollback_guardrails = hotspots
        .iter()
        .map(|hotspot| RollbackGuardrail {
            lever_id: hotspot.lever_id.clone(),
            rollback_command: format!(
                "fsfs profile rollback --lever-id {} --restore baseline",
                hotspot.lever_id
            ),
            abort_reason_codes: rollback_abort_reason_codes(hotspot.stage)
                .iter()
                .map(ToString::to_string)
                .collect(),
            recovery_reason_code: "opt.rollback.completed".to_owned(),
        })
        .collect::<Vec<_>>();

    CrawlIngestOptimizationTrack {
        schema_version: CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION.to_owned(),
        hotspots,
        proof_checklist,
        rollback_guardrails,
    }
}

fn hotspot_expectations_for(lever_id: &str) -> (CrawlIngestStage, u8, u8, u8) {
    match lever_id {
        "ingest.catalog.batch_upsert" => (CrawlIngestStage::CatalogMutation, 16, 24, 20),
        "crawl.classification.policy_batching" => (CrawlIngestStage::Classification, 10, 16, 12),
        "ingest.queue.lane_budget_admission" => (CrawlIngestStage::QueueAdmission, 9, 14, 11),
        "crawl.discovery.path_metadata_cache" => (CrawlIngestStage::DiscoveryWalk, 8, 13, 10),
        "ingest.embed_gate.early_skip" => (CrawlIngestStage::EmbeddingGate, 7, 11, 9),
        _ => (CrawlIngestStage::DiscoveryWalk, 5, 8, 5),
    }
}

const fn baseline_comparator_for(stage: CrawlIngestStage) -> &'static str {
    match stage {
        CrawlIngestStage::DiscoveryWalk => "baseline.crawl.discovery.sequential_walk",
        CrawlIngestStage::Classification => "baseline.crawl.classification.single_item",
        CrawlIngestStage::CatalogMutation => "baseline.ingest.catalog.single_upsert",
        CrawlIngestStage::QueueAdmission => "baseline.ingest.queue.global_fifo",
        CrawlIngestStage::EmbeddingGate => "baseline.ingest.embed.defer_after_index",
    }
}

const fn invariants_for_stage(stage: CrawlIngestStage) -> &'static [&'static str] {
    match stage {
        CrawlIngestStage::DiscoveryWalk => &[
            "no path omission across mount boundaries",
            "path canonicalization remains deterministic",
            "discovery scope decisions are unchanged",
        ],
        CrawlIngestStage::Classification => &[
            "ingestion_class assignment remains deterministic",
            "skip/index decisions preserve expected-loss ordering",
            "utility-score tie-break semantics remain unchanged",
        ],
        CrawlIngestStage::CatalogMutation => &[
            "catalog revision monotonicity preserved",
            "changelog stream sequence monotonicity preserved",
            "idempotent upsert semantics preserved",
        ],
        CrawlIngestStage::QueueAdmission => &[
            "lane budgets remain within configured hard limit",
            "backpressure transitions preserve reason-code semantics",
            "replay ordering remains monotonic",
        ],
        CrawlIngestStage::EmbeddingGate => &[
            "semantic-vs-lexical gating follows discovery policy",
            "low-signal candidates remain explainable with reason codes",
            "degraded-mode transitions remain reversible",
        ],
    }
}

const fn rollback_abort_reason_codes(stage: CrawlIngestStage) -> &'static [&'static str] {
    match stage {
        CrawlIngestStage::DiscoveryWalk => &[
            "discovery.scope.regression",
            "discovery.path_omission_detected",
            "ingest.replay.sequence_gap",
        ],
        CrawlIngestStage::Classification => &[
            "ingest.classification.regression",
            "ingest.expected_loss_violation",
            "ingest.explainability.missing_reason_code",
        ],
        CrawlIngestStage::CatalogMutation => &[
            "ingest.catalog.revision_non_monotonic",
            "ingest.catalog.idempotency_violation",
            "ingest.catalog.changelog_gap",
        ],
        CrawlIngestStage::QueueAdmission => &[
            "ingest.queue.starvation_detected",
            "ingest.backpressure.unbounded_growth",
            "ingest.replay.reordering_detected",
        ],
        CrawlIngestStage::EmbeddingGate => &[
            "ingest.embed.skip_policy_regression",
            "ingest.degrade.transition_invalid",
            "ingest.embed.queue_loss_detected",
        ],
    }
}

/// Snapshot of tuning levers for a single optimization iteration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeverSnapshot {
    /// Lever name/value pairs (sorted for deterministic output).
    pub values: BTreeMap<String, String>,
}

impl LeverSnapshot {
    /// Build a snapshot from key/value pairs.
    #[must_use]
    pub fn from_pairs<'a, I>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let values = pairs
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect();
        Self { values }
    }

    fn changed_levers(&self, other: &Self) -> Vec<String> {
        let mut key_union = BTreeSet::new();
        key_union.extend(self.values.keys().cloned());
        key_union.extend(other.values.keys().cloned());

        key_union
            .into_iter()
            .filter(|key| self.values.get(key) != other.values.get(key))
            .collect()
    }
}

/// Validation result for a one-lever optimization transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IterationValidation {
    /// Whether the transition is accepted by protocol.
    pub accepted: bool,
    /// Sorted list of changed levers.
    pub changed_levers: Vec<String>,
    /// Machine-readable reason code.
    pub reason_code: String,
}

/// Enforces one-lever-at-a-time optimization discipline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OneLeverIterationProtocol;

impl OneLeverIterationProtocol {
    /// Validate that exactly one lever changed from `baseline` to `candidate`.
    #[must_use]
    pub fn validate(baseline: &LeverSnapshot, candidate: &LeverSnapshot) -> IterationValidation {
        let changed_levers = baseline.changed_levers(candidate);
        let (accepted, reason_code) = match changed_levers.len() {
            1 => (true, ITERATION_REASON_ACCEPTED),
            0 => (false, ITERATION_REASON_NO_CHANGE),
            _ => (false, ITERATION_REASON_MULTI_CHANGE),
        };

        IterationValidation {
            accepted,
            changed_levers,
            reason_code: reason_code.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION, CrawlIngestStage, ITERATION_REASON_ACCEPTED,
        ITERATION_REASON_MULTI_CHANGE, ITERATION_REASON_NO_CHANGE, LeverSnapshot,
        OneLeverIterationProtocol, OpportunityCandidate, OpportunityMatrix,
        PROFILING_WORKFLOW_SCHEMA_VERSION, ProfileKind, ProfileWorkflow,
        crawl_ingest_opportunity_matrix, crawl_ingest_optimization_track,
    };
    use std::collections::BTreeSet;

    #[test]
    fn crawl_ingest_matrix_ranking_is_deterministic() {
        let ranked_first = crawl_ingest_opportunity_matrix().ranked();
        let ranked_second = crawl_ingest_opportunity_matrix().ranked();
        assert_eq!(ranked_first.len(), ranked_second.len());
        for (left, right) in ranked_first.iter().zip(ranked_second.iter()) {
            assert_eq!(left.rank, right.rank);
            assert_eq!(left.candidate.id, right.candidate.id);
            assert_eq!(left.ice_score_per_mille, right.ice_score_per_mille);
        }
    }

    #[test]
    fn crawl_ingest_track_covers_hotspots_proofs_and_rollbacks() {
        let track = crawl_ingest_optimization_track();
        assert_eq!(track.schema_version, CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION);
        assert_eq!(track.hotspots.len(), 5);
        assert_eq!(track.proof_checklist.len(), track.hotspots.len());
        assert_eq!(track.rollback_guardrails.len(), track.hotspots.len());

        let hotspot_ids: BTreeSet<&str> = track
            .hotspots
            .iter()
            .map(|hotspot| hotspot.lever_id.as_str())
            .collect();

        for hotspot in &track.hotspots {
            assert!(hotspot.expected_p50_gain_pct > 0);
            assert!(hotspot.expected_p95_gain_pct >= hotspot.expected_p50_gain_pct);
            assert!(hotspot.expected_throughput_gain_pct > 0);
        }

        for item in &track.proof_checklist {
            assert!(hotspot_ids.contains(item.lever_id.as_str()));
            assert!(!item.required_invariants.is_empty());
            assert!(item.replay_command.contains("--lane ingest"));
        }

        for guardrail in &track.rollback_guardrails {
            assert!(hotspot_ids.contains(guardrail.lever_id.as_str()));
            assert!(guardrail.rollback_command.contains("fsfs profile rollback"));
            assert!(!guardrail.abort_reason_codes.is_empty());
            assert_eq!(guardrail.recovery_reason_code, "opt.rollback.completed");
        }
    }

    #[test]
    fn crawl_ingest_track_includes_all_expected_stages() {
        let track = crawl_ingest_optimization_track();
        let stages: BTreeSet<CrawlIngestStage> =
            track.hotspots.iter().map(|hotspot| hotspot.stage).collect();
        assert_eq!(
            stages,
            BTreeSet::from([
                CrawlIngestStage::DiscoveryWalk,
                CrawlIngestStage::Classification,
                CrawlIngestStage::CatalogMutation,
                CrawlIngestStage::QueueAdmission,
                CrawlIngestStage::EmbeddingGate,
            ])
        );
    }

    #[test]
    fn profiling_workflow_contains_required_lanes() {
        let workflow = ProfileWorkflow::for_dataset_profile("small");
        let kinds: Vec<ProfileKind> = workflow.steps.iter().map(|step| step.kind).collect();

        assert_eq!(workflow.schema_version, PROFILING_WORKFLOW_SCHEMA_VERSION);
        assert_eq!(
            kinds,
            vec![
                ProfileKind::Flamegraph,
                ProfileKind::Heap,
                ProfileKind::Syscall
            ]
        );
    }

    #[test]
    fn opportunity_matrix_ranking_is_deterministic() {
        let matrix = OpportunityMatrix::new(vec![
            OpportunityCandidate {
                id: "query-fusion".into(),
                summary: "Reduce query fusion allocations".into(),
                impact: 85,
                confidence: 90,
                effort: 25,
            },
            OpportunityCandidate {
                id: "crawl-io".into(),
                summary: "Reduce crawl syscall count".into(),
                impact: 75,
                confidence: 80,
                effort: 30,
            },
            OpportunityCandidate {
                id: "tui-diff".into(),
                summary: "Skip unnecessary frame redraws".into(),
                impact: 70,
                confidence: 95,
                effort: 15,
            },
        ]);

        let ranked = matrix.ranked();

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].candidate.id, "tui-diff");
        assert_eq!(ranked[1].candidate.id, "query-fusion");
        assert_eq!(ranked[2].candidate.id, "crawl-io");
        assert!(ranked[0].ice_score_per_mille >= ranked[1].ice_score_per_mille);
        assert!(ranked[1].ice_score_per_mille >= ranked[2].ice_score_per_mille);
    }

    #[test]
    fn one_lever_protocol_accepts_exactly_one_change() {
        let baseline = LeverSnapshot::from_pairs([
            ("query.semantic_fanout", "64"),
            ("crawl.batch_size", "200"),
        ]);

        let accepted_candidate = LeverSnapshot::from_pairs([
            ("query.semantic_fanout", "80"),
            ("crawl.batch_size", "200"),
        ]);
        let accepted = OneLeverIterationProtocol::validate(&baseline, &accepted_candidate);
        assert!(accepted.accepted);
        assert_eq!(accepted.changed_levers, vec!["query.semantic_fanout"]);
        assert_eq!(accepted.reason_code, ITERATION_REASON_ACCEPTED);

        let unchanged = OneLeverIterationProtocol::validate(&baseline, &baseline);
        assert!(!unchanged.accepted);
        assert_eq!(unchanged.changed_levers.len(), 0);
        assert_eq!(unchanged.reason_code, ITERATION_REASON_NO_CHANGE);

        let multi_change_candidate = LeverSnapshot::from_pairs([
            ("query.semantic_fanout", "80"),
            ("crawl.batch_size", "100"),
        ]);
        let multi_change = OneLeverIterationProtocol::validate(&baseline, &multi_change_candidate);
        assert!(!multi_change.accepted);
        assert_eq!(
            multi_change.changed_levers,
            vec!["crawl.batch_size", "query.semantic_fanout"]
        );
        assert_eq!(multi_change.reason_code, ITERATION_REASON_MULTI_CHANGE);
    }
}
