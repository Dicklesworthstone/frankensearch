use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::GauntletError;

pub const SCORE_EPSILON: f32 = 0.0001;

/// Native secondary ordering evidence retained for every ranked hit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeTieKey {
    /// Quill's globally assigned document ID.
    QuillDocId { doc_id: u32 },
    /// Tantivy's full segment-qualified document address.
    TantivyDocAddress { segment_ord: u32, doc_id: u32 },
}

/// One engine-native ranked hit. Scores are stored as raw bits in artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankedHit {
    pub doc_id: String,
    pub score_bits: u32,
    pub native_tie_key: NativeTieKey,
}

impl RankedHit {
    /// Recover the score without changing its bit pattern.
    #[must_use]
    pub fn score(&self) -> f32 {
        f32::from_bits(self.score_bits)
    }
}

/// Complete observable output for one differential query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineObservation {
    /// Native top-k order. The comparator never sorts this vector by external ID.
    pub hits: Vec<RankedHit>,
    /// Full oracle cutoff tie group when top-k cuts an exact-score group.
    pub cutoff_tie_group: Vec<RankedHit>,
    /// Whether `cutoff_tie_group` is proven complete rather than fetch-limited.
    pub cutoff_tie_complete: bool,
    /// Full oracle tie group at the page's leading (offset) boundary, when
    /// the first returned rank cuts an exact-score group. Same evidence
    /// contract as `cutoff_tie_group`, for the offset edge: without it, a
    /// page starting inside a tie cannot prove its membership is order-only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub offset_tie_group: Vec<RankedHit>,
    /// Whether `offset_tie_group` is proven complete rather than
    /// fetch-limited. Defaults to the conservative `false` (no proof) so
    /// legacy artifacts without leading evidence keep their behavior.
    #[serde(default, skip_serializing_if = "is_false")]
    pub offset_tie_complete: bool,
    /// Snippets keyed by external document ID.
    pub snippets: BTreeMap<String, String>,
    /// Exact match count, or an explicit marker that the case did not request it.
    pub match_count: CountState,
    /// Exact live-document count.
    pub doc_count: u64,
    /// AST/diagnostic lowering differences the engine recorded while
    /// executing this query. Result-level equivalence is still proven by the
    /// rank/count comparison; these records make intentional lowerings
    /// (register classes) visible instead of silent. Empty records are
    /// omitted from canonical bytes so artifact hashes are unchanged for
    /// queries with no recorded lowering difference.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ast_differences: Vec<AstDifference>,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde skip_serializing_if protocol
const fn is_false(value: &bool) -> bool {
    !*value
}

/// Stable taxonomy for AST/diagnostic lowering differences. New kinds must
/// land with a divergence-register class in the same commit; the comparator
/// fails closed on kinds it cannot classify.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AstLoweringKind {
    /// An oversized (>65,530-byte) query token lowered to `MatchNone` under
    /// Quill's symmetric admission rule (register DIV-004).
    OversizedQueryToken,
    /// A match- or score-affecting parser/canonicalization shape differs from
    /// the reviewed logical query contract.
    QueryCanonicalization,
    /// Quill deliberately refuses to reproduce behavior proven incorrect in
    /// the pinned oracle.
    OracleBug,
    /// A bounded wildcard expansion selected a different admissible subset.
    GlobExpansionLimit,
    /// Snapshot statistics differ at a reviewed lifecycle boundary.
    StatsSemantics,
    /// Analyzer behavior differs on one reviewed Unicode edge case.
    UnicodeEdge,
}

/// One recorded AST/diagnostic lowering difference between subject and oracle
/// for the same logical query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AstDifference {
    /// Stable lowering class.
    pub kind: AstLoweringKind,
    /// Human-reviewable oracle AST/diagnostic summary.
    pub oracle: String,
    /// Human-reviewable subject AST/diagnostic summary.
    pub subject: String,
}

/// Query-count observation. Missing evidence is never treated as zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CountState {
    NotRequested,
    Value(u64),
}

/// Reviewed reason that permits `ScoreEpsilon` instead of `RankExact`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreEpsilonReason {
    OracleSegmentGeometry,
    PlatformLibm,
}

/// Comparator configuration encoded without JSON floating-point ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparatorConfig {
    pub score_epsilon_bits: u32,
    pub score_epsilon_reason: Option<ScoreEpsilonReason>,
}

impl Default for ComparatorConfig {
    fn default() -> Self {
        Self {
            score_epsilon_bits: 0.0001_f32.to_bits(),
            score_epsilon_reason: None,
        }
    }
}

impl ComparatorConfig {
    /// Construct a comparator configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidComparatorConfig`] unless
    /// `score_epsilon` is the contract-pinned [`SCORE_EPSILON`].
    pub fn new(score_epsilon: f32) -> Result<Self, GauntletError> {
        if score_epsilon.to_bits() != SCORE_EPSILON.to_bits() {
            return Err(GauntletError::InvalidComparatorConfig {
                reason: format!("score epsilon must be the contract-pinned {SCORE_EPSILON}"),
            });
        }
        Ok(Self {
            score_epsilon_bits: score_epsilon.to_bits(),
            score_epsilon_reason: None,
        })
    }

    /// Permit `ScoreEpsilon` with a closed, artifact-visible reason.
    #[must_use]
    pub const fn with_score_epsilon_reason(mut self, reason: ScoreEpsilonReason) -> Self {
        self.score_epsilon_reason = Some(reason);
        self
    }

    #[must_use]
    pub fn score_epsilon(self) -> f32 {
        f32::from_bits(self.score_epsilon_bits)
    }

    pub(crate) fn validate_contract(self) -> Result<(), GauntletError> {
        if self.score_epsilon_bits == SCORE_EPSILON.to_bits() {
            Ok(())
        } else {
            Err(GauntletError::InvalidComparatorConfig {
                reason: format!("score epsilon must be the contract-pinned {SCORE_EPSILON}"),
            })
        }
    }
}

/// Rank-level outcome before snippet and count checks are folded in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankClass {
    RankExact,
    TieOrder,
    ScoreEpsilon,
    RankMismatch,
}

/// Overall comparison posture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonStatus {
    Exact,
    Classified,
    Failed,
}

/// Stable taxonomy used by differential artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceClass {
    TieOrder,
    ScoreEpsilon,
    RankMismatch,
    SnippetMismatch,
    SnippetWindow,
    CountMismatch,
    DocumentCountMismatch,
    GlobExpansionLimit,
    QueryCanonicalization,
    OracleBug,
    StatsSemantics,
    UnicodeEdge,
    OversizedQueryToken,
}

impl DivergenceClass {
    const fn is_failure(self) -> bool {
        match self {
            Self::TieOrder
            | Self::ScoreEpsilon
            | Self::SnippetWindow
            | Self::GlobExpansionLimit
            | Self::QueryCanonicalization
            | Self::OracleBug
            | Self::StatsSemantics
            | Self::UnicodeEdge
            | Self::OversizedQueryToken => false,
            Self::RankMismatch
            | Self::SnippetMismatch
            | Self::CountMismatch
            | Self::DocumentCountMismatch => true,
        }
    }
}

/// First-class, pointer-addressable comparison difference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Divergence {
    pub class: DivergenceClass,
    /// RFC 6901 JSON pointer into the containing `ArtifactObject`.
    pub pointer: String,
    pub oracle: String,
    pub subject: String,
}

/// Pure comparator output, including the native-order evidence it evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub status: ComparisonStatus,
    pub rank_class: RankClass,
    pub score_epsilon_reason: Option<ScoreEpsilonReason>,
    pub divergences: Vec<Divergence>,
    pub first_divergence: Option<String>,
    pub subject: EngineObservation,
    pub oracle: EngineObservation,
}

/// Compare two observations while preserving each engine's native order.
///
/// Exact-score tie reorders are classified only after proving that the change
/// remains inside one oracle score group. A cutoff substitution additionally
/// requires a complete expanded oracle tie group. Insufficient evidence fails
/// closed as [`RankClass::RankMismatch`].
///
/// # Errors
///
/// Returns an error for duplicate document IDs, non-finite scores, invalid
/// score ordering, or invalid comparator configuration.
pub fn compare_observations(
    subject: EngineObservation,
    oracle: EngineObservation,
    config: ComparatorConfig,
) -> Result<ComparisonReport, GauntletError> {
    config.validate_contract()?;
    let epsilon = config.score_epsilon();

    validate_observation("subject", &subject)?;
    validate_observation("oracle", &oracle)?;

    let (rank_class, rank_divergence) = classify_rank(
        &subject,
        &oracle,
        epsilon,
        config.score_epsilon_reason.is_some(),
    );
    let mut divergences = Vec::new();
    if let Some(divergence) = rank_divergence {
        divergences.push(divergence);
    }
    classify_ast_differences(&subject, &oracle, &mut divergences);
    compare_snippets(&subject, &oracle, &mut divergences);
    if subject.match_count != oracle.match_count {
        divergences.push(Divergence {
            class: DivergenceClass::CountMismatch,
            pointer: "/comparison/subject/match_count".to_owned(),
            oracle: describe_count(oracle.match_count),
            subject: describe_count(subject.match_count),
        });
    }
    if subject.doc_count != oracle.doc_count {
        divergences.push(Divergence {
            class: DivergenceClass::DocumentCountMismatch,
            pointer: "/comparison/subject/doc_count".to_owned(),
            oracle: oracle.doc_count.to_string(),
            subject: subject.doc_count.to_string(),
        });
    }

    let status = if divergences.is_empty() {
        ComparisonStatus::Exact
    } else if divergences.iter().any(|item| item.class.is_failure()) {
        ComparisonStatus::Failed
    } else {
        ComparisonStatus::Classified
    };
    let first_divergence = divergences.first().map(|item| item.pointer.clone());
    let score_epsilon_reason = if rank_class == RankClass::ScoreEpsilon {
        config.score_epsilon_reason
    } else {
        None
    };

    Ok(ComparisonReport {
        status,
        rank_class,
        score_epsilon_reason,
        divergences,
        first_divergence,
        subject,
        oracle,
    })
}

fn validate_observation(label: &str, observation: &EngineObservation) -> Result<(), GauntletError> {
    validate_hit_slice(label, "hits", &observation.hits, true)?;
    validate_hit_slice(
        label,
        "cutoff_tie_group",
        &observation.cutoff_tie_group,
        false,
    )?;
    validate_hit_slice(
        label,
        "offset_tie_group",
        &observation.offset_tie_group,
        false,
    )?;
    validate_cross_evidence_identity(label, &observation.hits, &observation.cutoff_tie_group)?;
    validate_cross_evidence_identity(label, &observation.hits, &observation.offset_tie_group)?;
    validate_cross_evidence_identity(
        label,
        &observation.offset_tie_group,
        &observation.cutoff_tie_group,
    )?;
    let hit_key = observation.hits.first().map(|hit| &hit.native_tie_key);
    let cutoff_key = observation
        .cutoff_tie_group
        .first()
        .map(|hit| &hit.native_tie_key);
    let offset_key = observation
        .offset_tie_group
        .first()
        .map(|hit| &hit.native_tie_key);
    if hit_key
        .zip(cutoff_key)
        .is_some_and(|(left, right)| !same_tie_key_family(left, right))
        || hit_key
            .zip(offset_key)
            .is_some_and(|(left, right)| !same_tie_key_family(left, right))
    {
        return Err(GauntletError::InvalidObservation {
            reason: format!("{label} mixes native tie-key families across evidence"),
        });
    }
    let hit_ids = observation
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    if observation
        .snippets
        .keys()
        .any(|doc_id| !hit_ids.contains(doc_id.as_str()))
    {
        return Err(GauntletError::InvalidObservation {
            reason: format!("{label}.snippets contains a document outside top-k hits"),
        });
    }
    Ok(())
}

fn validate_hit_slice(
    label: &str,
    field: &str,
    hits: &[RankedHit],
    require_score_order: bool,
) -> Result<(), GauntletError> {
    let mut ids = BTreeSet::new();
    let mut native_keys = BTreeSet::new();
    let mut previous_score: Option<f32> = None;
    let mut previous_tie_key: Option<&NativeTieKey> = None;
    let mut tie_key_family: Option<&NativeTieKey> = None;
    for hit in hits {
        if hit.doc_id.is_empty() {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} contains an empty document ID"),
            });
        }
        if !ids.insert(hit.doc_id.as_str()) {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} repeats document ID {}", hit.doc_id),
            });
        }
        if !native_keys.insert(&hit.native_tie_key) {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} repeats a native tie key"),
            });
        }
        let score = hit.score();
        if !score.is_finite() {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} has a non-finite score for {}", hit.doc_id),
            });
        }
        if tie_key_family.is_some_and(|key| !same_tie_key_family(key, &hit.native_tie_key)) {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} mixes native tie-key families"),
            });
        }
        if require_score_order
            && let Some(previous) = previous_score
            && previous.total_cmp(&score).is_lt()
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} is not ordered by descending score"),
            });
        }
        if previous_score.is_some_and(|previous| {
            previous.total_cmp(&score).is_eq()
                && previous_tie_key.is_some_and(|key| key >= &hit.native_tie_key)
        }) {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "{label}.{field} is not ordered by ascending native key inside an exact-score tie"
                ),
            });
        }
        tie_key_family.get_or_insert(&hit.native_tie_key);
        previous_score = Some(score);
        previous_tie_key = Some(&hit.native_tie_key);
    }
    Ok(())
}

fn validate_cross_evidence_identity(
    label: &str,
    hits: &[RankedHit],
    cutoff_tie_group: &[RankedHit],
) -> Result<(), GauntletError> {
    let hits_by_doc = hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), (hit.score_bits, &hit.native_tie_key)))
        .collect::<BTreeMap<_, _>>();
    let hits_by_native_key = hits
        .iter()
        .map(|hit| (hit.native_tie_key.clone(), hit.doc_id.as_str()))
        .collect::<BTreeMap<_, _>>();

    for cutoff_hit in cutoff_tie_group {
        if let Some((score_bits, native_tie_key)) = hits_by_doc.get(cutoff_hit.doc_id.as_str())
            && (!scores_exact(*score_bits, cutoff_hit.score_bits)
                || *native_tie_key != &cutoff_hit.native_tie_key)
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "{label} gives document {} inconsistent top-k and cutoff evidence",
                    cutoff_hit.doc_id
                ),
            });
        }
        if let Some(hit_doc_id) = hits_by_native_key.get(&cutoff_hit.native_tie_key)
            && *hit_doc_id != cutoff_hit.doc_id
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "{label} assigns one native tie key to multiple document IDs across evidence"
                ),
            });
        }
    }
    Ok(())
}

fn classify_rank(
    subject: &EngineObservation,
    oracle: &EngineObservation,
    epsilon: f32,
    score_epsilon_allowed: bool,
) -> (RankClass, Option<Divergence>) {
    if sequence_is_exact(&subject.hits, &oracle.hits) {
        return (RankClass::RankExact, None);
    }

    let pointer = first_rank_pointer(&subject.hits, &oracle.hits);
    if is_exact_tie_reorder(&subject.hits, &oracle.hits)
        || is_proven_cutoff_tie_substitution(subject, oracle)
    {
        return (
            RankClass::TieOrder,
            Some(rank_divergence(
                DivergenceClass::TieOrder,
                pointer,
                subject,
                oracle,
            )),
        );
    }

    if score_epsilon_allowed && is_score_epsilon_equivalent(&subject.hits, &oracle.hits, epsilon) {
        return (
            RankClass::ScoreEpsilon,
            Some(rank_divergence(
                DivergenceClass::ScoreEpsilon,
                pointer,
                subject,
                oracle,
            )),
        );
    }

    (
        RankClass::RankMismatch,
        Some(rank_divergence(
            DivergenceClass::RankMismatch,
            pointer,
            subject,
            oracle,
        )),
    )
}

fn same_tie_key_family(left: &NativeTieKey, right: &NativeTieKey) -> bool {
    matches!(
        (left, right),
        (
            NativeTieKey::QuillDocId { .. },
            NativeTieKey::QuillDocId { .. }
        ) | (
            NativeTieKey::TantivyDocAddress { .. },
            NativeTieKey::TantivyDocAddress { .. }
        )
    )
}

fn sequence_is_exact(subject: &[RankedHit], oracle: &[RankedHit]) -> bool {
    subject.len() == oracle.len()
        && subject.iter().zip(oracle).all(|(subject_hit, oracle_hit)| {
            subject_hit.doc_id == oracle_hit.doc_id
                && scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
        })
}

fn is_exact_tie_reorder(subject: &[RankedHit], oracle: &[RankedHit]) -> bool {
    if subject.len() != oracle.len() || subject.is_empty() {
        return false;
    }
    let Some(oracle_map) = score_map(oracle) else {
        return false;
    };
    let Some(subject_map) = score_map(subject) else {
        return false;
    };
    if oracle_map != subject_map {
        return false;
    }

    let groups = exact_group_map(oracle);
    groups_are_nondecreasing(subject, &groups)
}

fn is_proven_cutoff_tie_substitution(
    subject: &EngineObservation,
    oracle: &EngineObservation,
) -> bool {
    is_proven_boundary_tie_substitution(subject, oracle)
}

/// One validated single-score boundary group: the boundary score plus the
/// complete document membership. `None` when the evidence is absent,
/// incomplete, or internally inconsistent (mixed scores or repeated IDs).
fn complete_boundary_group(
    group: &[RankedHit],
    complete: bool,
    boundary_score_bits: u32,
) -> Option<(u32, BTreeSet<&str>)> {
    if !complete || group.is_empty() {
        return None;
    }
    let score_bits = group.first()?.score_bits;
    if !scores_exact(score_bits, boundary_score_bits) {
        return None;
    }
    let mut docs = BTreeSet::new();
    for hit in group {
        if !scores_exact(hit.score_bits, score_bits) || !docs.insert(hit.doc_id.as_str()) {
            return None;
        }
    }
    Some((score_bits, docs))
}

/// Whether both sides of one differing rank are members of the same complete
/// boundary group at their shared exact score.
fn pair_explained_by_boundary_group(
    subject: &RankedHit,
    oracle: &RankedHit,
    group: Option<&(u32, BTreeSet<&str>)>,
) -> bool {
    group.is_some_and(|(score_bits, docs)| {
        scores_exact(subject.score_bits, *score_bits)
            && scores_exact(oracle.score_bits, *score_bits)
            && docs.contains(subject.doc_id.as_str())
            && docs.contains(oracle.doc_id.as_str())
    })
}

/// Generalized tie-substitution proof covering both page boundaries.
///
/// Every position where the subject and oracle pages differ — by document
/// identity or score — must be explained by a complete oracle boundary
/// group: the differing documents on both sides must share one exact score
/// and belong to the complete leading (offset) or trailing (cutoff) tie
/// group. Positions outside the differing span are exact by construction.
/// With only trailing evidence present this reduces to the original cutoff
/// substitution proof; absent or incomplete evidence fails closed.
fn is_proven_boundary_tie_substitution(
    subject: &EngineObservation,
    oracle: &EngineObservation,
) -> bool {
    if subject.hits.len() != oracle.hits.len() || subject.hits.is_empty() {
        return false;
    }

    let leading = oracle.hits.first().and_then(|boundary| {
        complete_boundary_group(
            &oracle.offset_tie_group,
            oracle.offset_tie_complete,
            boundary.score_bits,
        )
    });
    let trailing = oracle.hits.last().and_then(|boundary| {
        complete_boundary_group(
            &oracle.cutoff_tie_group,
            oracle.cutoff_tie_complete,
            boundary.score_bits,
        )
    });

    let mut saw_difference = false;
    for (subject_hit, oracle_hit) in subject.hits.iter().zip(&oracle.hits) {
        if subject_hit.doc_id == oracle_hit.doc_id
            && scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
        {
            continue;
        }
        saw_difference = true;
        if !scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
            || !(pair_explained_by_boundary_group(subject_hit, oracle_hit, leading.as_ref())
                || pair_explained_by_boundary_group(subject_hit, oracle_hit, trailing.as_ref()))
        {
            return false;
        }
    }
    saw_difference
}

fn is_score_epsilon_equivalent(subject: &[RankedHit], oracle: &[RankedHit], epsilon: f32) -> bool {
    if subject.len() != oracle.len() || subject.is_empty() {
        return false;
    }
    let Some(oracle_scores) = score_map(oracle) else {
        return false;
    };
    let subject_ids = subject
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    if subject_ids.len() != subject.len()
        || subject_ids != oracle_scores.keys().copied().collect::<BTreeSet<_>>()
    {
        return false;
    }
    if subject.iter().any(|hit| {
        oracle_scores
            .get(hit.doc_id.as_str())
            .is_none_or(|oracle_bits| {
                !within_relative_epsilon(hit.score(), f32::from_bits(*oracle_bits), epsilon)
            })
    }) {
        return false;
    }

    let groups = epsilon_group_map(oracle, epsilon);
    groups_are_nondecreasing(subject, &groups)
}

fn score_map(hits: &[RankedHit]) -> Option<BTreeMap<&str, u32>> {
    let map = hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
        .collect::<BTreeMap<_, _>>();
    (map.len() == hits.len()).then_some(map)
}

fn exact_group_map(hits: &[RankedHit]) -> BTreeMap<&str, usize> {
    group_map(hits, scores_exact)
}

fn epsilon_group_map(hits: &[RankedHit], epsilon: f32) -> BTreeMap<&str, usize> {
    group_map(hits, |left, right| {
        within_relative_epsilon(f32::from_bits(left), f32::from_bits(right), epsilon)
    })
}

fn group_map(hits: &[RankedHit], adjacent: impl Fn(u32, u32) -> bool) -> BTreeMap<&str, usize> {
    let mut groups = BTreeMap::new();
    let mut group = 0usize;
    let mut previous = None;
    for hit in hits {
        if previous.is_some_and(|score| !adjacent(score, hit.score_bits)) {
            group = group.saturating_add(1);
        }
        groups.insert(hit.doc_id.as_str(), group);
        previous = Some(hit.score_bits);
    }
    groups
}

fn groups_are_nondecreasing(hits: &[RankedHit], groups: &BTreeMap<&str, usize>) -> bool {
    let mut previous = None;
    for hit in hits {
        let Some(group) = groups.get(hit.doc_id.as_str()).copied() else {
            return false;
        };
        if previous.is_some_and(|prior| group < prior) {
            return false;
        }
        previous = Some(group);
    }
    true
}

fn scores_exact(left: u32, right: u32) -> bool {
    f32::from_bits(left)
        .total_cmp(&f32::from_bits(right))
        .is_eq()
}

fn within_relative_epsilon(left: f32, right: f32, epsilon: f32) -> bool {
    let left = f64::from(left);
    let right = f64::from(right);
    let denominator = left.abs().max(right.abs()).max(1e-12);
    (left - right).abs() / denominator <= f64::from(epsilon)
}

fn first_rank_pointer(subject: &[RankedHit], oracle: &[RankedHit]) -> String {
    let index = subject
        .iter()
        .zip(oracle)
        .position(|(subject_hit, oracle_hit)| {
            subject_hit.doc_id != oracle_hit.doc_id
                || !scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
        });
    index.map_or_else(
        || "/comparison/subject/hits".to_owned(),
        |index| format!("/comparison/subject/hits/{index}"),
    )
}

fn rank_divergence(
    class: DivergenceClass,
    pointer: String,
    subject: &EngineObservation,
    oracle: &EngineObservation,
) -> Divergence {
    let index = pointer
        .rsplit('/')
        .next()
        .and_then(|value| value.parse::<usize>().ok());
    Divergence {
        class,
        pointer,
        oracle: index
            .and_then(|index| oracle.hits.get(index))
            .map_or_else(|| oracle.hits.len().to_string(), describe_hit),
        subject: index
            .and_then(|index| subject.hits.get(index))
            .map_or_else(|| subject.hits.len().to_string(), describe_hit),
    }
}

fn describe_hit(hit: &RankedHit) -> String {
    format!("{}@{:08x}", hit.doc_id, hit.score_bits)
}

fn describe_count(count: CountState) -> String {
    match count {
        CountState::NotRequested => "not_requested".to_owned(),
        CountState::Value(value) => value.to_string(),
    }
}

/// Fold recorded AST/diagnostic lowering differences into the divergence
/// list. Every recorded kind maps to a reviewed register class; kinds the
/// comparator does not know cannot reach here because the enum is closed,
/// and adding a kind without its register class fails review.
fn classify_ast_differences(
    subject: &EngineObservation,
    oracle: &EngineObservation,
    divergences: &mut Vec<Divergence>,
) {
    for (index, difference) in subject.ast_differences.iter().enumerate() {
        let class = match difference.kind {
            AstLoweringKind::OversizedQueryToken => DivergenceClass::OversizedQueryToken,
            AstLoweringKind::QueryCanonicalization => DivergenceClass::QueryCanonicalization,
            AstLoweringKind::OracleBug => DivergenceClass::OracleBug,
            AstLoweringKind::GlobExpansionLimit => DivergenceClass::GlobExpansionLimit,
            AstLoweringKind::StatsSemantics => DivergenceClass::StatsSemantics,
            AstLoweringKind::UnicodeEdge => DivergenceClass::UnicodeEdge,
        };
        divergences.push(Divergence {
            class,
            pointer: format!("/comparison/subject/ast_differences/{index}"),
            oracle: difference.oracle.clone(),
            subject: difference.subject.clone(),
        });
    }
    for (index, difference) in oracle.ast_differences.iter().enumerate() {
        divergences.push(Divergence {
            class: DivergenceClass::RankMismatch,
            pointer: format!("/comparison/oracle/ast_differences/{index}"),
            oracle: difference.oracle.clone(),
            subject: difference.subject.clone(),
        });
    }
}

fn compare_snippets(
    subject: &EngineObservation,
    oracle: &EngineObservation,
    divergences: &mut Vec<Divergence>,
) {
    let subject_ids = subject
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    let oracle_ids = oracle
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    let ids = subject_ids.intersection(&oracle_ids);
    for doc_id in ids {
        let subject_snippet = subject.snippets.get(*doc_id);
        let oracle_snippet = oracle.snippets.get(*doc_id);
        if subject_snippet != oracle_snippet {
            divergences.push(Divergence {
                class: DivergenceClass::SnippetMismatch,
                pointer: format!(
                    "/comparison/{}/snippets/{}",
                    if subject_snippet.is_some() {
                        "subject"
                    } else {
                        "oracle"
                    },
                    escape_json_pointer_token(doc_id)
                ),
                oracle: oracle_snippet
                    .cloned()
                    .unwrap_or_else(|| "<missing>".to_owned()),
                subject: subject_snippet
                    .cloned()
                    .unwrap_or_else(|| "<missing>".to_owned()),
            });
            break;
        }
    }
}

fn escape_json_pointer_token(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quill_hit(doc_id: &str, score: f32, native_doc_id: u32) -> RankedHit {
        RankedHit {
            doc_id: doc_id.to_owned(),
            score_bits: score.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId {
                doc_id: native_doc_id,
            },
        }
    }

    fn tantivy_hit(doc_id: &str, score: f32, doc_id_in_segment: u32) -> RankedHit {
        RankedHit {
            doc_id: doc_id.to_owned(),
            score_bits: score.to_bits(),
            native_tie_key: NativeTieKey::TantivyDocAddress {
                segment_ord: 3,
                doc_id: doc_id_in_segment,
            },
        }
    }

    fn observation(hits: Vec<RankedHit>) -> EngineObservation {
        EngineObservation {
            match_count: CountState::Value(u64::try_from(hits.len()).unwrap_or(u64::MAX)),
            doc_count: 9,
            hits,
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets: BTreeMap::new(),
            ast_differences: Vec::new(),
        }
    }

    #[test]
    fn exact_score_tie_order_is_classified_without_rewriting_native_order() {
        let subject = observation(vec![quill_hit("b", 4.0, 1), quill_hit("a", 4.0, 2)]);
        let oracle = observation(vec![tantivy_hit("a", 4.0, 8), tantivy_hit("b", 4.0, 9)]);

        let report =
            compare_observations(subject.clone(), oracle.clone(), ComparatorConfig::default())
                .expect("tie comparison");

        assert_eq!(report.status, ComparisonStatus::Classified);
        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.subject.hits, subject.hits);
        assert_eq!(report.oracle.hits, oracle.hits);
        assert_eq!(
            report.first_divergence.as_deref(),
            Some("/comparison/subject/hits/0")
        );
    }

    #[test]
    fn top_k_tie_substitution_requires_complete_expansion() {
        let subject = observation(vec![quill_hit("a", 5.0, 1), quill_hit("c", 4.0, 3)]);
        let mut oracle = observation(vec![tantivy_hit("a", 5.0, 1), tantivy_hit("b", 4.0, 2)]);
        oracle.match_count = CountState::Value(3);
        oracle.cutoff_tie_group = vec![tantivy_hit("b", 4.0, 2), tantivy_hit("c", 4.0, 3)];

        let classified = compare_observations(
            EngineObservation {
                match_count: CountState::Value(3),
                ..subject.clone()
            },
            oracle.clone(),
            ComparatorConfig::default(),
        )
        .expect("complete tie evidence");
        assert_eq!(classified.rank_class, RankClass::TieOrder);

        oracle.cutoff_tie_complete = false;
        let failed = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("incomplete tie evidence fails closed");
        assert_eq!(failed.rank_class, RankClass::RankMismatch);
        assert_eq!(failed.status, ComparisonStatus::Failed);
    }

    #[test]
    fn epsilon_reorders_only_inside_oracle_connected_components() {
        let subject = observation(vec![
            quill_hit("b", 10.0004, 2),
            quill_hit("a", 9.9996, 1),
            quill_hit("c", 8.0, 3),
        ]);
        let oracle = observation(vec![
            tantivy_hit("a", 10.0, 1),
            tantivy_hit("b", 9.9999, 2),
            tantivy_hit("c", 8.0, 3),
        ]);
        let report = compare_observations(
            subject,
            oracle,
            ComparatorConfig::default()
                .with_score_epsilon_reason(ScoreEpsilonReason::OracleSegmentGeometry),
        )
        .expect("epsilon comparison");
        assert_eq!(report.rank_class, RankClass::ScoreEpsilon);
        assert_eq!(
            report.score_epsilon_reason,
            Some(ScoreEpsilonReason::OracleSegmentGeometry)
        );
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn epsilon_above_contract_or_without_reason_is_rejected() {
        assert!(ComparatorConfig::new(0.001).is_err());
        let subject = observation(vec![quill_hit("a", 1.000_05, 1)]);
        let oracle = observation(vec![tantivy_hit("a", 1.0, 1)]);
        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("unreviewed epsilon fails as a rank mismatch");
        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
        assert_eq!(report.score_epsilon_reason, None);
    }

    #[test]
    fn rank_exact_rejects_out_of_order_native_tie_keys() {
        let subject = observation(vec![quill_hit("a", 1.0, 2), quill_hit("b", 1.0, 1)]);
        let oracle = observation(vec![tantivy_hit("a", 1.0, 1), tantivy_hit("b", 1.0, 2)]);
        assert!(compare_observations(subject, oracle, ComparatorConfig::default()).is_err());
    }

    #[test]
    fn duplicate_native_identity_across_scores_is_rejected() {
        let subject = observation(vec![quill_hit("a", 2.0, 7), quill_hit("b", 1.0, 7)]);
        let oracle = observation(vec![tantivy_hit("a", 2.0, 1), tantivy_hit("b", 1.0, 2)]);

        assert!(matches!(
            compare_observations(subject, oracle, ComparatorConfig::default()),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn top_k_and_cutoff_identity_must_be_consistent() {
        let oracle = observation(vec![tantivy_hit("a", 2.0, 1)]);

        let mut changed_score = observation(vec![quill_hit("a", 2.0, 7)]);
        changed_score.cutoff_tie_group = vec![quill_hit("a", 1.0, 7)];
        assert!(matches!(
            compare_observations(changed_score, oracle.clone(), ComparatorConfig::default()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut changed_doc = observation(vec![quill_hit("a", 2.0, 7)]);
        changed_doc.cutoff_tie_group = vec![quill_hit("b", 2.0, 7)];
        assert!(matches!(
            compare_observations(changed_doc, oracle, ComparatorConfig::default()),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn cutoff_tie_substitution_does_not_compare_unaligned_snippets() {
        let mut subject = observation(vec![quill_hit("a", 5.0, 1), quill_hit("c", 4.0, 3)]);
        let mut oracle = observation(vec![tantivy_hit("a", 5.0, 1), tantivy_hit("b", 4.0, 2)]);
        subject.match_count = CountState::Value(3);
        oracle.match_count = CountState::Value(3);
        oracle.cutoff_tie_group = vec![tantivy_hit("b", 4.0, 2), tantivy_hit("c", 4.0, 3)];
        subject.snippets.insert("c".to_owned(), "c body".to_owned());
        oracle.snippets.insert("b".to_owned(), "b body".to_owned());

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("complete cutoff tie evidence");
        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn snippet_and_counts_have_stable_json_pointers() {
        let mut subject = observation(vec![quill_hit("a/b~c", 1.0, 1)]);
        let mut oracle = observation(vec![tantivy_hit("a/b~c", 1.0, 1)]);
        subject
            .snippets
            .insert("a/b~c".to_owned(), "left".to_owned());
        oracle
            .snippets
            .insert("a/b~c".to_owned(), "right".to_owned());
        subject.match_count = CountState::Value(2);
        subject.doc_count = 8;

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("surface comparison");
        let pointers = report
            .divergences
            .iter()
            .map(|item| item.pointer.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            pointers,
            vec![
                "/comparison/subject/snippets/a~1b~0c",
                "/comparison/subject/match_count",
                "/comparison/subject/doc_count"
            ]
        );
        let projection = serde_json::json!({ "comparison": &report });
        assert!(
            report
                .divergences
                .iter()
                .all(|divergence| projection.pointer(&divergence.pointer).is_some())
        );
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn oversized_query_token_ast_difference_is_classified_not_failed() {
        // Result-equivalent lowering: identical hits, one recorded AST
        // difference for the oversized-token admission (register DIV-004).
        let mut subject = observation(vec![quill_hit("a", 5.0, 1), quill_hit("b", 4.0, 2)]);
        subject.ast_differences.push(AstDifference {
            kind: AstLoweringKind::OversizedQueryToken,
            oracle: "BooleanQuery(TermQuery(content:oversized))".to_owned(),
            subject: "Empty (oversized > 65,530-byte token admitted as MatchNone)".to_owned(),
        });
        let oracle = observation(vec![tantivy_hit("a", 5.0, 1), tantivy_hit("b", 4.0, 2)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("oversized-token comparison");

        assert_eq!(report.status, ComparisonStatus::Classified);
        assert_eq!(report.rank_class, RankClass::RankExact);
        assert_eq!(report.divergences.len(), 1);
        assert_eq!(
            report.divergences[0].class,
            DivergenceClass::OversizedQueryToken
        );
        assert_eq!(
            report.divergences[0].pointer,
            "/comparison/subject/ast_differences/0"
        );
        // The classified pointer resolves in the serialized artifact.
        let projection = serde_json::json!({ "comparison": &report });
        assert!(projection.pointer(&report.divergences[0].pointer).is_some());
    }

    #[test]
    fn reviewed_ast_difference_taxonomy_is_closed_and_classified() {
        let reviewed = [
            (
                AstLoweringKind::QueryCanonicalization,
                DivergenceClass::QueryCanonicalization,
            ),
            (AstLoweringKind::OracleBug, DivergenceClass::OracleBug),
            (
                AstLoweringKind::GlobExpansionLimit,
                DivergenceClass::GlobExpansionLimit,
            ),
            (
                AstLoweringKind::StatsSemantics,
                DivergenceClass::StatsSemantics,
            ),
            (AstLoweringKind::UnicodeEdge, DivergenceClass::UnicodeEdge),
        ];
        for (kind, expected_class) in reviewed {
            let mut subject = observation(vec![quill_hit("a", 5.0, 1)]);
            subject.ast_differences.push(AstDifference {
                kind,
                oracle: "reviewed oracle shape".to_owned(),
                subject: "reviewed subject shape".to_owned(),
            });
            let oracle = observation(vec![tantivy_hit("a", 5.0, 1)]);
            let report = compare_observations(subject, oracle, ComparatorConfig::default())
                .expect("reviewed AST comparison");
            assert_eq!(report.status, ComparisonStatus::Classified);
            assert_eq!(report.rank_class, RankClass::RankExact);
            assert_eq!(report.divergences.len(), 1);
            assert_eq!(report.divergences[0].class, expected_class);
        }
    }

    #[test]
    fn ast_difference_does_not_mask_result_level_failures() {
        // An oversized-token record may accompany only result-equivalent
        // runs; a real rank divergence still fails closed.
        let mut subject = observation(vec![quill_hit("a", 5.0, 1)]);
        subject.ast_differences.push(AstDifference {
            kind: AstLoweringKind::OversizedQueryToken,
            oracle: "TermQuery".to_owned(),
            subject: "Empty".to_owned(),
        });
        let oracle = observation(vec![tantivy_hit("b", 5.0, 1)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("rank divergence with ast record");

        assert_eq!(report.status, ComparisonStatus::Failed);
        assert!(
            report
                .divergences
                .iter()
                .any(|divergence| divergence.class == DivergenceClass::RankMismatch)
        );
    }

    #[test]
    fn oracle_side_ast_difference_fails_closed() {
        let subject = observation(vec![quill_hit("a", 5.0, 1)]);
        let mut oracle = observation(vec![tantivy_hit("a", 5.0, 1)]);
        oracle.ast_differences.push(AstDifference {
            kind: AstLoweringKind::OversizedQueryToken,
            oracle: "unexpected oracle diagnostic".to_owned(),
            subject: "unexpected oracle lowering".to_owned(),
        });

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("oracle-side AST evidence is represented as a failure");

        assert_eq!(report.status, ComparisonStatus::Failed);
        assert_eq!(report.divergences.len(), 1);
        assert_eq!(report.divergences[0].class, DivergenceClass::RankMismatch);
        assert_eq!(
            report.divergences[0].pointer,
            "/comparison/oracle/ast_differences/0"
        );
    }

    #[test]
    fn observation_without_ast_differences_still_deserializes() {
        // Artifacts written before the ast_differences channel existed must
        // keep parsing (serde default).
        let legacy = serde_json::json!({
            "hits": [],
            "cutoff_tie_group": [],
            "cutoff_tie_complete": true,
            "snippets": {},
            "match_count": "not_requested",
            "doc_count": 0,
        });
        let observation: EngineObservation =
            serde_json::from_value(legacy).expect("legacy observation parses");
        assert!(observation.ast_differences.is_empty());
        assert!(observation.offset_tie_group.is_empty());
        assert!(!observation.offset_tie_complete);
    }

    #[test]
    fn leading_offset_tie_substitution_is_classified_tie_order() {
        // Page [C9, D8] at offset 2 inside oracle order A10,B9,C9,D8: the
        // subject's native order walked B9 into the page instead of C9. The
        // complete leading group {B9, C9} proves the membership difference
        // is order-only.
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        let subject = observation(vec![quill_hit("b", 9.0, 1), quill_hit("d", 8.0, 3)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("offset tie comparison");

        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.status, ComparisonStatus::Classified);
        assert_eq!(report.divergences[0].class, DivergenceClass::TieOrder);
    }

    #[test]
    fn leading_offset_tie_substitution_fails_closed_without_complete_group() {
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = false;
        let subject = observation(vec![quill_hit("b", 9.0, 1), quill_hit("d", 8.0, 3)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("incomplete offset evidence fails closed");

        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn leading_and_trailing_substitutions_combine_across_boundaries() {
        // Offset 2 cuts {B9, C9}; the page tail cuts {D8, E8}. Both
        // substitutions are explainable when both complete groups exist.
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        oracle.cutoff_tie_group = vec![tantivy_hit("d", 8.0, 3), tantivy_hit("e", 8.0, 4)];
        oracle.cutoff_tie_complete = true;
        let subject = observation(vec![quill_hit("b", 9.0, 1), quill_hit("e", 8.0, 4)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("two-boundary tie comparison");

        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn two_boundary_substitutions_allow_an_exact_middle_hit() {
        let mut oracle = observation(vec![
            tantivy_hit("c", 9.0, 2),
            tantivy_hit("middle", 8.0, 3),
            tantivy_hit("d", 7.0, 4),
        ]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        oracle.cutoff_tie_group = vec![tantivy_hit("d", 7.0, 4), tantivy_hit("e", 7.0, 5)];
        oracle.cutoff_tie_complete = true;
        let subject = observation(vec![
            quill_hit("b", 9.0, 1),
            quill_hit("middle", 8.0, 3),
            quill_hit("e", 7.0, 5),
        ]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("two boundary substitutions with exact middle");

        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn cross_boundary_score_substitution_fails_closed() {
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        oracle.cutoff_tie_group = vec![tantivy_hit("d", 8.0, 3), tantivy_hit("e", 8.0, 4)];
        oracle.cutoff_tie_complete = true;
        let subject = observation(vec![quill_hit("b", 9.0, 1), quill_hit("c", 9.0, 2)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("cross-boundary substitution is a real mismatch");

        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn unrelated_boundary_membership_fails_closed() {
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("x", 9.0, 8), tantivy_hit("y", 9.0, 9)];
        oracle.offset_tie_complete = true;
        let subject = observation(vec![quill_hit("x", 9.0, 8), quill_hit("d", 8.0, 3)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("unrelated boundary group is not proof");

        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn unexplained_score_difference_at_leading_edge_is_rank_mismatch() {
        // The substitute has a different score than every boundary group:
        // genuinely divergent, not a tie artifact.
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        let subject = observation(vec![quill_hit("b", 8.0, 1), quill_hit("d", 8.0, 3)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("score-mismatched substitute fails closed");

        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn offset_group_inconsistent_with_page_hits_is_rejected() {
        // The same document appears in the page and in the offset group with
        // a different score: the evidence is internally inconsistent.
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("c", 7.0, 2)];
        oracle.offset_tie_complete = true;

        let error = compare_observations(oracle.clone(), oracle, ComparatorConfig::default())
            .expect_err("inconsistent offset evidence is rejected");
        assert!(matches!(error, GauntletError::InvalidObservation { .. }));
    }
}
