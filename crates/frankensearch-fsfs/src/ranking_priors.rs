//! Ranking priors: recency, path proximity, and project affinity.
//!
//! Priors apply post-fusion multiplicative boosts to `FusedCandidate` scores.
//! All priors are deterministic: same inputs always produce same outputs.
//! Priors are disabled under pressure (Degraded/Emergency modes skip all priors).

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::query_execution::{DegradedRetrievalMode, FusedCandidate};

/// Schema version for the prior configuration format.
pub const PRIOR_CONFIG_SCHEMA_VERSION: u16 = 1;

/// Default half-life for recency decay in days.
pub const DEFAULT_RECENCY_HALF_LIFE_DAYS: f64 = 180.0;

/// Default path proximity radius in directory levels.
pub const DEFAULT_PATH_PROXIMITY_RADIUS: usize = 3;

/// Default maximum boost from any single prior.
pub const DEFAULT_MAX_PRIOR_BOOST: f64 = 2.0;

/// Minimum prior multiplier (never zero out a result).
pub const MIN_PRIOR_MULTIPLIER: f64 = 0.1;

// ---------------------------------------------------------------------------
// Prior family taxonomy
// ---------------------------------------------------------------------------

/// Enumeration of ranking prior families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorFamily {
    /// Exponential decay based on document age.
    Recency,
    /// Boost for documents near the query context path.
    PathProximity,
    /// Boost for documents in the same project as the query context.
    ProjectAffinity,
}

impl fmt::Display for PriorFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Recency => write!(f, "recency"),
            Self::PathProximity => write!(f, "path_proximity"),
            Self::ProjectAffinity => write!(f, "project_affinity"),
        }
    }
}

impl FromStr for PriorFamily {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "recency" => Ok(Self::Recency),
            "path_proximity" => Ok(Self::PathProximity),
            "project_affinity" => Ok(Self::ProjectAffinity),
            _ => Err(()),
        }
    }
}

/// All supported prior families in canonical order.
pub const ALL_PRIOR_FAMILIES: [PriorFamily; 3] = [
    PriorFamily::Recency,
    PriorFamily::PathProximity,
    PriorFamily::ProjectAffinity,
];

// ---------------------------------------------------------------------------
// Prior configuration
// ---------------------------------------------------------------------------

/// Per-family weight and enablement configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorFamilyConfig {
    pub family: PriorFamily,
    pub enabled: bool,
    pub weight: f64,
}

impl PriorFamilyConfig {
    #[must_use]
    pub const fn new(family: PriorFamily, weight: f64) -> Self {
        // const fn cannot call clamp, so manual clamping:
        let w = if weight < 0.0 {
            0.0
        } else if weight > DEFAULT_MAX_PRIOR_BOOST {
            DEFAULT_MAX_PRIOR_BOOST
        } else {
            weight
        };
        Self {
            family,
            enabled: true,
            weight: w,
        }
    }

    #[must_use]
    pub const fn disabled(family: PriorFamily) -> Self {
        Self {
            family,
            enabled: false,
            weight: 0.0,
        }
    }
}

/// Full prior configuration for the ranking pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingPriorConfig {
    pub recency_half_life_days: f64,
    pub path_proximity_radius: usize,
    pub max_boost: f64,
    pub families: Vec<PriorFamilyConfig>,
}

impl Default for RankingPriorConfig {
    fn default() -> Self {
        Self {
            recency_half_life_days: DEFAULT_RECENCY_HALF_LIFE_DAYS,
            path_proximity_radius: DEFAULT_PATH_PROXIMITY_RADIUS,
            max_boost: DEFAULT_MAX_PRIOR_BOOST,
            families: vec![
                PriorFamilyConfig::new(PriorFamily::Recency, 1.0),
                PriorFamilyConfig::new(PriorFamily::PathProximity, 1.0),
                PriorFamilyConfig::new(PriorFamily::ProjectAffinity, 1.0),
            ],
        }
    }
}

impl RankingPriorConfig {
    /// Returns an all-disabled configuration.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            recency_half_life_days: DEFAULT_RECENCY_HALF_LIFE_DAYS,
            path_proximity_radius: DEFAULT_PATH_PROXIMITY_RADIUS,
            max_boost: DEFAULT_MAX_PRIOR_BOOST,
            families: vec![
                PriorFamilyConfig::disabled(PriorFamily::Recency),
                PriorFamilyConfig::disabled(PriorFamily::PathProximity),
                PriorFamilyConfig::disabled(PriorFamily::ProjectAffinity),
            ],
        }
    }

    /// Checks whether any prior family is enabled.
    #[must_use]
    pub fn any_enabled(&self) -> bool {
        self.families.iter().any(|f| f.enabled)
    }

    /// Looks up the config for a given family.
    #[must_use]
    pub fn family_config(&self, family: PriorFamily) -> Option<&PriorFamilyConfig> {
        self.families.iter().find(|f| f.family == family)
    }

    /// Returns an effective config that respects degradation mode.
    #[must_use]
    pub fn effective_for_mode(&self, mode: DegradedRetrievalMode) -> Self {
        match mode {
            DegradedRetrievalMode::Normal => self.clone(),
            DegradedRetrievalMode::EmbedDeferred => {
                // Keep fast priors (recency, project), disable expensive ones (path).
                let mut cfg = self.clone();
                for fam in &mut cfg.families {
                    if fam.family == PriorFamily::PathProximity {
                        fam.enabled = false;
                    }
                }
                cfg
            }
            DegradedRetrievalMode::LexicalOnly
            | DegradedRetrievalMode::MetadataOnly
            | DegradedRetrievalMode::Paused => Self::disabled(),
        }
    }
}

// ---------------------------------------------------------------------------
// Query context: information available at query time for prior evaluation
// ---------------------------------------------------------------------------

/// Context about the query environment used to evaluate priors.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryPriorContext {
    /// Current timestamp as seconds since Unix epoch.
    pub now_unix_seconds: u64,
    /// Path from which the query was issued (working directory).
    pub query_origin_path: Option<String>,
    /// Project identifier for the query context.
    pub query_project_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Document metadata: per-document fields needed for prior evaluation
// ---------------------------------------------------------------------------

/// Per-document metadata needed to evaluate priors.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocumentPriorMetadata {
    /// Document's last-modified time as seconds since Unix epoch.
    pub modified_unix_seconds: Option<u64>,
    /// Document's file path.
    pub file_path: Option<String>,
    /// Document's project identifier.
    pub project_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Prior evidence: auditable record of each prior application
// ---------------------------------------------------------------------------

/// Evidence for one prior's contribution to a document's score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorEvidence {
    pub family: PriorFamily,
    pub multiplier: f64,
    pub reason_code: &'static str,
    pub detail: String,
}

/// Complete prior application result for one document.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorApplicationResult {
    pub doc_id: String,
    pub base_score: f64,
    pub adjusted_score: f64,
    pub combined_multiplier: f64,
    pub evidence: Vec<PriorEvidence>,
}

// ---------------------------------------------------------------------------
// Prior computation functions (pure, deterministic)
// ---------------------------------------------------------------------------

/// Compute recency multiplier using exponential half-life decay.
///
/// Returns 1.0 for a document modified right now, decaying toward
/// `MIN_PRIOR_MULTIPLIER` as the document ages.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn recency_multiplier(
    now_unix_seconds: u64,
    modified_unix_seconds: u64,
    half_life_days: f64,
    weight: f64,
) -> (f64, &'static str, String) {
    if half_life_days <= 0.0 || !half_life_days.is_finite() {
        return (1.0, "prior.recency.disabled_invalid_config", String::new());
    }
    if modified_unix_seconds > now_unix_seconds {
        return (1.0, "prior.recency.future_timestamp", String::new());
    }

    let age_seconds = now_unix_seconds.saturating_sub(modified_unix_seconds);
    let age_days = age_seconds as f64 / 86_400.0;
    let lambda = (2.0_f64).ln() / half_life_days;
    let raw_decay = (-lambda * age_days).exp();
    let multiplier = weight
        .mul_add(raw_decay - 1.0, 1.0)
        .max(MIN_PRIOR_MULTIPLIER);
    let detail = format!("age={age_days:.1}d decay={raw_decay:.4} weight={weight:.2}");
    (multiplier, "prior.recency.applied", detail)
}

/// Compute path proximity multiplier based on shared directory prefix depth.
///
/// Returns a boost proportional to the number of shared path components.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn path_proximity_multiplier(
    query_path: &str,
    doc_path: &str,
    radius: usize,
    weight: f64,
) -> (f64, &'static str, String) {
    if radius == 0 || query_path.is_empty() || doc_path.is_empty() {
        return (
            1.0,
            "prior.path_proximity.disabled_missing_context",
            String::new(),
        );
    }

    let shared = shared_prefix_depth(query_path, doc_path);
    if shared == 0 {
        return (
            1.0,
            "prior.path_proximity.no_shared_prefix",
            "shared=0".to_string(),
        );
    }

    let effective_depth = shared.min(radius);
    let ratio = effective_depth as f64 / radius as f64;
    let multiplier = weight.mul_add(ratio, 1.0);
    let detail = format!("shared={shared} radius={radius} ratio={ratio:.2}");
    (multiplier, "prior.path_proximity.applied", detail)
}

/// Compute project affinity multiplier.
///
/// Returns a simple boost when query and document share the same project.
#[must_use]
pub fn project_affinity_multiplier(
    query_project: &str,
    doc_project: &str,
    weight: f64,
) -> (f64, &'static str, String) {
    if query_project.is_empty() || doc_project.is_empty() {
        return (
            1.0,
            "prior.project_affinity.disabled_missing_context",
            String::new(),
        );
    }

    if query_project == doc_project {
        let multiplier = 1.0 + weight;
        let detail = format!("project={query_project}");
        (multiplier, "prior.project_affinity.same_project", detail)
    } else {
        (
            1.0,
            "prior.project_affinity.different_project",
            format!("query={query_project} doc={doc_project}"),
        )
    }
}

/// Count shared path prefix components (platform-independent).
#[must_use]
pub fn shared_prefix_depth(path_a: &str, path_b: &str) -> usize {
    let components_a: Vec<&str> = path_a.split('/').filter(|c| !c.is_empty()).collect();
    let components_b: Vec<&str> = path_b.split('/').filter(|c| !c.is_empty()).collect();

    components_a
        .iter()
        .zip(components_b.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

// ---------------------------------------------------------------------------
// Prior applier: orchestrates all priors for a set of fused candidates
// ---------------------------------------------------------------------------

/// Applies ranking priors to fused candidates with full audit trail.
#[derive(Debug, Clone)]
pub struct PriorApplier {
    config: RankingPriorConfig,
}

impl PriorApplier {
    #[must_use]
    pub const fn new(config: RankingPriorConfig) -> Self {
        Self { config }
    }

    /// Create an applier that respects the given degradation mode.
    #[must_use]
    pub fn for_mode(config: &RankingPriorConfig, mode: DegradedRetrievalMode) -> Self {
        Self {
            config: config.effective_for_mode(mode),
        }
    }

    /// Returns the effective config (after degradation adjustments).
    #[must_use]
    pub const fn config(&self) -> &RankingPriorConfig {
        &self.config
    }

    /// Apply all enabled priors to a set of fused candidates.
    ///
    /// Returns the re-ranked candidates (sorted by adjusted score) along with
    /// per-document evidence for explanation payloads.
    #[must_use]
    pub fn apply(
        &self,
        candidates: &[FusedCandidate],
        context: &QueryPriorContext,
        metadata: &HashMap<String, DocumentPriorMetadata>,
    ) -> Vec<PriorApplicationResult> {
        let mut results: Vec<PriorApplicationResult> = if self.config.any_enabled() {
            candidates
                .iter()
                .map(|c| {
                    let doc_meta = metadata.get(&c.doc_id);
                    self.apply_to_candidate(c, context, doc_meta)
                })
                .collect()
        } else {
            candidates
                .iter()
                .map(|c| PriorApplicationResult {
                    doc_id: c.doc_id.clone(),
                    base_score: c.fused_score,
                    adjusted_score: c.fused_score,
                    combined_multiplier: 1.0,
                    evidence: vec![],
                })
                .collect()
        };

        // Always re-sort by adjusted score with deterministic tie-breaking.
        results.sort_by(|a, b| {
            b.adjusted_score
                .total_cmp(&a.adjusted_score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        results
    }

    fn apply_to_candidate(
        &self,
        candidate: &FusedCandidate,
        context: &QueryPriorContext,
        doc_meta: Option<&DocumentPriorMetadata>,
    ) -> PriorApplicationResult {
        let base_score = candidate.fused_score;
        let mut combined_multiplier = 1.0;
        let mut evidence = Vec::new();

        let empty_meta = DocumentPriorMetadata::default();
        let meta = doc_meta.unwrap_or(&empty_meta);

        // Recency prior
        if let Some(fc) = self.config.family_config(PriorFamily::Recency)
            && fc.enabled
        {
            if let Some(modified) = meta.modified_unix_seconds {
                let (mult, code, detail) = recency_multiplier(
                    context.now_unix_seconds,
                    modified,
                    self.config.recency_half_life_days,
                    fc.weight,
                );
                combined_multiplier *= mult;
                evidence.push(PriorEvidence {
                    family: PriorFamily::Recency,
                    multiplier: mult,
                    reason_code: code,
                    detail,
                });
            } else {
                evidence.push(PriorEvidence {
                    family: PriorFamily::Recency,
                    multiplier: 1.0,
                    reason_code: "prior.recency.no_timestamp",
                    detail: String::new(),
                });
            }
        }

        // Path proximity prior
        if let Some(fc) = self.config.family_config(PriorFamily::PathProximity)
            && fc.enabled
        {
            if let (Some(query_path), Some(doc_path)) =
                (&context.query_origin_path, &meta.file_path)
            {
                let (mult, code, detail) = path_proximity_multiplier(
                    query_path,
                    doc_path,
                    self.config.path_proximity_radius,
                    fc.weight,
                );
                combined_multiplier *= mult;
                evidence.push(PriorEvidence {
                    family: PriorFamily::PathProximity,
                    multiplier: mult,
                    reason_code: code,
                    detail,
                });
            } else {
                evidence.push(PriorEvidence {
                    family: PriorFamily::PathProximity,
                    multiplier: 1.0,
                    reason_code: "prior.path_proximity.disabled_missing_context",
                    detail: String::new(),
                });
            }
        }

        // Project affinity prior
        if let Some(fc) = self.config.family_config(PriorFamily::ProjectAffinity)
            && fc.enabled
        {
            if let (Some(query_proj), Some(doc_proj)) =
                (&context.query_project_id, &meta.project_id)
            {
                let (mult, code, detail) =
                    project_affinity_multiplier(query_proj, doc_proj, fc.weight);
                combined_multiplier *= mult;
                evidence.push(PriorEvidence {
                    family: PriorFamily::ProjectAffinity,
                    multiplier: mult,
                    reason_code: code,
                    detail,
                });
            } else {
                evidence.push(PriorEvidence {
                    family: PriorFamily::ProjectAffinity,
                    multiplier: 1.0,
                    reason_code: "prior.project_affinity.disabled_missing_context",
                    detail: String::new(),
                });
            }
        }

        // Clamp combined multiplier.
        combined_multiplier =
            combined_multiplier.clamp(MIN_PRIOR_MULTIPLIER, self.config.max_boost);
        let adjusted_score = base_score * combined_multiplier;

        PriorApplicationResult {
            doc_id: candidate.doc_id.clone(),
            base_score,
            adjusted_score,
            combined_multiplier,
            evidence,
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic tie-break contract
// ---------------------------------------------------------------------------

/// Documents the deterministic tie-break rules for prior-adjusted rankings.
///
/// After prior application, candidates are sorted by:
/// 1. Adjusted score (descending, `total_cmp` for NaN safety)
/// 2. Document ID (ascending, lexicographic)
///
/// This ensures that identical inputs always produce identical orderings
/// regardless of `HashMap` iteration order or floating-point platform differences.
pub const PRIOR_TIE_BREAK_CONTRACT: &str =
    "sort(adjusted_score desc total_cmp, doc_id asc lexicographic)";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fused(doc_id: &str, score: f64) -> FusedCandidate {
        FusedCandidate {
            doc_id: doc_id.to_string(),
            fused_score: score,
            prior_boost: 0.0,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        }
    }

    fn make_meta(
        modified: Option<u64>,
        path: Option<&str>,
        project: Option<&str>,
    ) -> DocumentPriorMetadata {
        DocumentPriorMetadata {
            modified_unix_seconds: modified,
            file_path: path.map(ToString::to_string),
            project_id: project.map(ToString::to_string),
        }
    }

    // -- PriorFamily Display / FromStr --

    #[test]
    fn prior_family_display_and_parse() {
        for family in ALL_PRIOR_FAMILIES {
            let s = family.to_string();
            let parsed: PriorFamily = s.parse().unwrap();
            assert_eq!(parsed, family);
        }
        assert!("invalid".parse::<PriorFamily>().is_err());
    }

    // -- Recency multiplier --

    #[test]
    fn recency_multiplier_full_boost_for_fresh_document() {
        let (mult, code, _) = recency_multiplier(1_000_000, 1_000_000, 180.0, 1.0);
        assert!((mult - 1.0).abs() < 1e-9);
        assert_eq!(code, "prior.recency.applied");
    }

    #[test]
    fn recency_multiplier_decays_with_age() {
        let half_life = 180.0;
        let half_life_seconds = 180_u64 * 86_400;
        let now = half_life_seconds + 1_000_000;
        let (mult, _, _) =
            recency_multiplier(now, now.saturating_sub(half_life_seconds), half_life, 1.0);
        // At exactly one half-life, decay = 0.5, so multiplier = 1 + 1.0*(0.5 - 1.0) = 0.5
        assert!((mult - 0.5).abs() < 0.01, "expected ~0.5, got {mult}");
    }

    #[test]
    fn recency_multiplier_never_below_minimum() {
        let now = 1_000_000_000u64;
        let ancient = 0u64;
        let (mult, _, _) = recency_multiplier(now, ancient, 1.0, 1.0);
        assert!(mult >= MIN_PRIOR_MULTIPLIER);
    }

    #[test]
    fn recency_multiplier_handles_future_timestamp() {
        let (mult, code, _) = recency_multiplier(100, 200, 180.0, 1.0);
        assert!((mult - 1.0).abs() < 1e-9);
        assert_eq!(code, "prior.recency.future_timestamp");
    }

    #[test]
    fn recency_multiplier_weight_zero_is_neutral() {
        let (mult, _, _) = recency_multiplier(1_000_000, 0, 180.0, 0.0);
        assert!((mult - 1.0).abs() < 1e-9);
    }

    #[test]
    fn recency_multiplier_invalid_half_life() {
        let (mult, code, _) = recency_multiplier(100, 50, 0.0, 1.0);
        assert!((mult - 1.0).abs() < 1e-9);
        assert_eq!(code, "prior.recency.disabled_invalid_config");

        let (mult2, _, _) = recency_multiplier(100, 50, f64::NAN, 1.0);
        assert!((mult2 - 1.0).abs() < 1e-9);
    }

    // -- Path proximity --

    #[test]
    fn path_proximity_same_directory() {
        let (mult, code, _) = path_proximity_multiplier(
            "/home/user/project/src",
            "/home/user/project/src/lib.rs",
            3,
            1.0,
        );
        assert!(mult > 1.0);
        assert_eq!(code, "prior.path_proximity.applied");
    }

    #[test]
    fn path_proximity_no_shared_prefix() {
        let (mult, code, _) = path_proximity_multiplier("/foo/bar", "/baz/qux", 3, 1.0);
        assert!((mult - 1.0).abs() < 1e-9);
        assert_eq!(code, "prior.path_proximity.no_shared_prefix");
    }

    #[test]
    fn path_proximity_empty_paths() {
        let (mult, code, _) = path_proximity_multiplier("", "/foo", 3, 1.0);
        assert!((mult - 1.0).abs() < 1e-9);
        assert_eq!(code, "prior.path_proximity.disabled_missing_context");
    }

    #[test]
    fn path_proximity_clamped_to_radius() {
        let (mult, _, _) =
            path_proximity_multiplier("/a/b/c/d/e/f/g/h/i/j", "/a/b/c/d/e/f/g/h/i/j/k", 3, 1.0);
        assert!((mult - 2.0).abs() < 1e-9);
    }

    // -- Project affinity --

    #[test]
    fn project_affinity_same_project() {
        let (mult, code, _) = project_affinity_multiplier("myproject", "myproject", 1.0);
        assert!((mult - 2.0).abs() < 1e-9);
        assert_eq!(code, "prior.project_affinity.same_project");
    }

    #[test]
    fn project_affinity_different_project() {
        let (mult, code, _) = project_affinity_multiplier("projA", "projB", 1.0);
        assert!((mult - 1.0).abs() < 1e-9);
        assert_eq!(code, "prior.project_affinity.different_project");
    }

    #[test]
    fn project_affinity_missing_context() {
        let (mult, code, _) = project_affinity_multiplier("", "projB", 1.0);
        assert!((mult - 1.0).abs() < 1e-9);
        assert_eq!(code, "prior.project_affinity.disabled_missing_context");
    }

    // -- Shared prefix depth --

    #[test]
    fn shared_prefix_depth_basic_cases() {
        assert_eq!(shared_prefix_depth("/a/b/c", "/a/b/d"), 2);
        assert_eq!(shared_prefix_depth("/a/b/c", "/a/b/c"), 3);
        assert_eq!(shared_prefix_depth("/a/b", "/x/y"), 0);
        assert_eq!(shared_prefix_depth("", ""), 0);
    }

    // -- Config --

    #[test]
    fn default_config_has_all_families_enabled() {
        let cfg = RankingPriorConfig::default();
        assert!(cfg.any_enabled());
        assert_eq!(cfg.families.len(), 3);
        for fam in &cfg.families {
            assert!(fam.enabled);
            assert!((fam.weight - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn disabled_config_has_no_families_enabled() {
        let cfg = RankingPriorConfig::disabled();
        assert!(!cfg.any_enabled());
    }

    #[test]
    fn effective_config_degrades_under_pressure() {
        let cfg = RankingPriorConfig::default();

        let normal = cfg.effective_for_mode(DegradedRetrievalMode::Normal);
        assert!(normal.any_enabled());

        let deferred = cfg.effective_for_mode(DegradedRetrievalMode::EmbedDeferred);
        assert!(deferred.any_enabled());
        let path_config = deferred.family_config(PriorFamily::PathProximity).unwrap();
        assert!(!path_config.enabled);
        let recency_config = deferred.family_config(PriorFamily::Recency).unwrap();
        assert!(recency_config.enabled);

        let lexical = cfg.effective_for_mode(DegradedRetrievalMode::LexicalOnly);
        assert!(!lexical.any_enabled());

        let metadata = cfg.effective_for_mode(DegradedRetrievalMode::MetadataOnly);
        assert!(!metadata.any_enabled());
    }

    // -- PriorApplier --

    #[test]
    fn applier_returns_unchanged_scores_when_disabled() {
        let applier = PriorApplier::new(RankingPriorConfig::disabled());
        let candidates = vec![make_fused("doc-a", 0.5), make_fused("doc-b", 0.3)];
        let context = QueryPriorContext::default();
        let metadata = HashMap::new();

        let results = applier.apply(&candidates, &context, &metadata);
        assert_eq!(results.len(), 2);
        assert!((results[0].adjusted_score - 0.5).abs() < 1e-9);
        assert!((results[1].adjusted_score - 0.3).abs() < 1e-9);
    }

    #[test]
    fn applier_boosts_recent_document() {
        let applier = PriorApplier::new(RankingPriorConfig::default());
        let candidates = vec![make_fused("old-doc", 0.5), make_fused("new-doc", 0.4)];
        let now = 1_000_000u64;
        let context = QueryPriorContext {
            now_unix_seconds: now,
            query_origin_path: None,
            query_project_id: None,
        };
        let mut metadata = HashMap::new();
        metadata.insert("old-doc".to_string(), make_meta(Some(0), None, None));
        metadata.insert(
            "new-doc".to_string(),
            make_meta(Some(now - 3600), None, None),
        );

        let results = applier.apply(&candidates, &context, &metadata);
        let new_result = results.iter().find(|r| r.doc_id == "new-doc").unwrap();
        let old_result = results.iter().find(|r| r.doc_id == "old-doc").unwrap();
        assert!(
            new_result.combined_multiplier > old_result.combined_multiplier,
            "new should have higher multiplier: {} vs {}",
            new_result.combined_multiplier,
            old_result.combined_multiplier
        );
    }

    #[test]
    fn applier_boosts_nearby_path() {
        let applier = PriorApplier::new(RankingPriorConfig::default());
        let candidates = vec![make_fused("far-doc", 0.5), make_fused("near-doc", 0.4)];
        let context = QueryPriorContext {
            now_unix_seconds: 0,
            query_origin_path: Some("/home/user/project/src".to_string()),
            query_project_id: None,
        };
        let mut metadata = HashMap::new();
        metadata.insert(
            "far-doc".to_string(),
            make_meta(None, Some("/other/path/lib.rs"), None),
        );
        metadata.insert(
            "near-doc".to_string(),
            make_meta(None, Some("/home/user/project/src/mod.rs"), None),
        );

        let results = applier.apply(&candidates, &context, &metadata);
        let near = results.iter().find(|r| r.doc_id == "near-doc").unwrap();
        let far = results.iter().find(|r| r.doc_id == "far-doc").unwrap();
        assert!(near.combined_multiplier > far.combined_multiplier);
    }

    #[test]
    fn applier_boosts_same_project() {
        let applier = PriorApplier::new(RankingPriorConfig::default());
        let candidates = vec![make_fused("other-proj", 0.5), make_fused("same-proj", 0.4)];
        let context = QueryPriorContext {
            now_unix_seconds: 0,
            query_origin_path: None,
            query_project_id: Some("frankensearch".to_string()),
        };
        let mut metadata = HashMap::new();
        metadata.insert("other-proj".to_string(), make_meta(None, None, Some("xf")));
        metadata.insert(
            "same-proj".to_string(),
            make_meta(None, None, Some("frankensearch")),
        );

        let results = applier.apply(&candidates, &context, &metadata);
        let same = results.iter().find(|r| r.doc_id == "same-proj").unwrap();
        let other = results.iter().find(|r| r.doc_id == "other-proj").unwrap();
        assert!(same.combined_multiplier > other.combined_multiplier);
    }

    #[test]
    fn applier_respects_max_boost_cap() {
        let config = RankingPriorConfig {
            max_boost: 1.5,
            ..RankingPriorConfig::default()
        };
        let applier = PriorApplier::new(config);
        let candidates = vec![make_fused("doc-a", 1.0)];
        let context = QueryPriorContext {
            now_unix_seconds: 100,
            query_origin_path: Some("/a/b/c".to_string()),
            query_project_id: Some("proj".to_string()),
        };
        let mut metadata = HashMap::new();
        metadata.insert(
            "doc-a".to_string(),
            make_meta(Some(99), Some("/a/b/c/d"), Some("proj")),
        );

        let results = applier.apply(&candidates, &context, &metadata);
        assert!(results[0].combined_multiplier <= 1.5 + 1e-9);
    }

    #[test]
    fn applier_reranks_based_on_prior_boosts() {
        let applier = PriorApplier::new(RankingPriorConfig::default());
        let candidates = vec![make_fused("doc-a", 0.5), make_fused("doc-b", 0.45)];
        let context = QueryPriorContext {
            now_unix_seconds: 100,
            query_origin_path: None,
            query_project_id: Some("myproj".to_string()),
        };
        let mut metadata = HashMap::new();
        metadata.insert("doc-a".to_string(), make_meta(None, None, Some("other")));
        metadata.insert("doc-b".to_string(), make_meta(None, None, Some("myproj")));

        let results = applier.apply(&candidates, &context, &metadata);
        // doc-b gets 2x project affinity boost: 0.45*2 = 0.9 > doc-a's 0.5*1 = 0.5
        assert_eq!(results[0].doc_id, "doc-b");
        assert_eq!(results[1].doc_id, "doc-a");
    }

    #[test]
    fn applier_evidence_records_all_priors() {
        let applier = PriorApplier::new(RankingPriorConfig::default());
        let candidates = vec![make_fused("doc-a", 0.5)];
        let context = QueryPriorContext {
            now_unix_seconds: 100,
            query_origin_path: Some("/a/b/c".to_string()),
            query_project_id: Some("proj".to_string()),
        };
        let mut metadata = HashMap::new();
        metadata.insert(
            "doc-a".to_string(),
            make_meta(Some(50), Some("/a/b/d"), Some("proj")),
        );

        let results = applier.apply(&candidates, &context, &metadata);
        let evidence = &results[0].evidence;
        assert_eq!(evidence.len(), 3);
        let families: Vec<PriorFamily> = evidence.iter().map(|e| e.family).collect();
        assert!(families.contains(&PriorFamily::Recency));
        assert!(families.contains(&PriorFamily::PathProximity));
        assert!(families.contains(&PriorFamily::ProjectAffinity));
    }

    #[test]
    fn applier_deterministic_tiebreak_by_doc_id() {
        let applier = PriorApplier::new(RankingPriorConfig::disabled());
        let candidates = vec![
            make_fused("doc-z", 0.5),
            make_fused("doc-a", 0.5),
            make_fused("doc-m", 0.5),
        ];
        let context = QueryPriorContext::default();
        let metadata = HashMap::new();

        let results = applier.apply(&candidates, &context, &metadata);
        let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert_eq!(ids, vec!["doc-a", "doc-m", "doc-z"]);
    }

    #[test]
    fn applier_missing_metadata_does_not_panic() {
        let applier = PriorApplier::new(RankingPriorConfig::default());
        let candidates = vec![make_fused("unknown-doc", 0.5)];
        let context = QueryPriorContext {
            now_unix_seconds: 100,
            query_origin_path: Some("/a/b".to_string()),
            query_project_id: Some("proj".to_string()),
        };
        let metadata = HashMap::new();

        let results = applier.apply(&candidates, &context, &metadata);
        assert_eq!(results.len(), 1);
        assert!((results[0].combined_multiplier - 1.0).abs() < 1e-9);
    }

    #[test]
    fn config_serialization_roundtrip() {
        let config = RankingPriorConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: RankingPriorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn prior_family_config_clamps_weight() {
        let fc = PriorFamilyConfig::new(PriorFamily::Recency, 100.0);
        assert!((fc.weight - DEFAULT_MAX_PRIOR_BOOST).abs() < 1e-9);

        let fc2 = PriorFamilyConfig::new(PriorFamily::Recency, -5.0);
        assert!((fc2.weight - 0.0).abs() < 1e-9);
    }
}
