//! Source-preserving fast semantic windows for fsfs-owned vector generations.
//!
//! The persisted source manifest describes every window. Generated FSVI keys
//! are internal identities only: filters, budgets, fusion and output always
//! operate on source document IDs.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use frankensearch_core::filter::{PredicateFilter, SearchFilter};
use frankensearch_core::{SearchError, SearchResult, VectorHit};
use frankensearch_index::{ClassifiedHits, VectorIndex};
use serde::{Deserialize, Serialize};

pub(super) const WINDOW_CHARS: usize = 2_000;
const WINDOW_OVERLAP_CHARS: usize = 200;
const WINDOW_STRIDE_CHARS: usize = WINDOW_CHARS - WINDOW_OVERLAP_CHARS;
pub(super) const MAX_WINDOWS_PER_FILE: usize = 128;
const ROW_SEPARATOR: &str = "\0fsfs-window-v1:";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(super) struct FastWindowRange {
    pub ordinal: usize,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(super) struct FastWindowPlan {
    pub max_per_file: usize,
    pub source_chars: usize,
    pub windows: Vec<FastWindowRange>,
}

fn invalid(reason: impl Into<String>) -> SearchError {
    SearchError::InvalidConfig {
        field: "index.fast_windows".to_owned(),
        value: "source_window_mapping".to_owned(),
        reason: reason.into(),
    }
}

pub(super) fn validate_source_id(source: &str) -> SearchResult<()> {
    if source.is_empty() || source.contains('\0') {
        return Err(invalid(
            "source document IDs must be nonempty and cannot contain NUL, which is reserved for internal semantic window rows",
        ));
    }
    Ok(())
}

/// Row zero keeps the ordinary source ID. Additional rows occupy a namespace
/// that cannot collide with a filesystem path or an admitted append source ID.
pub(super) fn row_id(source: &str, ordinal: usize) -> String {
    if ordinal == 0 {
        source.to_owned()
    } else {
        format!("{source}{ROW_SEPARATOR}{ordinal}")
    }
}

fn decoded_row(row: &str) -> Option<(&str, usize)> {
    let (source, suffix) = row.split_once(ROW_SEPARATOR)?;
    if source.is_empty() || source.contains('\0') || suffix.starts_with('0') {
        return None;
    }
    let ordinal = suffix.parse::<usize>().ok()?;
    (ordinal > 0 && ordinal < MAX_WINDOWS_PER_FILE && suffix == ordinal.to_string())
        .then_some((source, ordinal))
}

pub(super) fn source_id(row: &str) -> &str {
    decoded_row(row).map_or(row, |(source, _)| source)
}

#[cfg(test)]
fn window_ordinal(row: &str) -> usize {
    decoded_row(row).map_or(0, |(_, ordinal)| ordinal)
}

#[cfg(test)]
fn is_window_row(row: &str) -> bool {
    decoded_row(row).is_some()
}

fn plan_for_chars(source_chars: usize, max_per_file: usize) -> SearchResult<FastWindowPlan> {
    if !(1..=MAX_WINDOWS_PER_FILE).contains(&max_per_file) {
        return Err(invalid(format!(
            "fast_window_max_per_file must be between 1 and {MAX_WINDOWS_PER_FILE}",
        )));
    }
    let last_start = source_chars.saturating_sub(WINDOW_CHARS);
    let natural_count = last_start.div_ceil(WINDOW_STRIDE_CHARS).saturating_add(1);
    let count = natural_count.min(max_per_file);
    let mut windows = Vec::with_capacity(count);
    for ordinal in 0..count {
        // When capped, retain the first and last natural windows and spread
        // the remaining ones across the document. The gaps remain explicit
        // in the persisted ranges; a cap never claims complete coverage.
        let natural_ordinal = if count <= 1 {
            0
        } else {
            usize::try_from(
                (ordinal as u128) * ((natural_count - 1) as u128) / ((count - 1) as u128),
            )
            .map_err(|_| invalid("window position exceeds the supported address space"))?
        };
        let start_char = natural_ordinal
            .saturating_mul(WINDOW_STRIDE_CHARS)
            .min(last_start);
        windows.push(FastWindowRange {
            ordinal,
            start_char,
            end_char: start_char.saturating_add(WINDOW_CHARS).min(source_chars),
        });
    }
    Ok(FastWindowPlan {
        max_per_file,
        source_chars,
        windows,
    })
}

pub(super) fn plan(full_text: &str, max_per_file: usize) -> SearchResult<FastWindowPlan> {
    plan_for_chars(full_text.chars().count(), max_per_file)
}

impl FastWindowPlan {
    pub fn validate(&self) -> SearchResult<()> {
        let expected = plan_for_chars(self.source_chars, self.max_per_file)?;
        if *self != expected {
            return Err(invalid(
                "persisted semantic windows do not match the versioned bounded selection policy",
            ));
        }
        Ok(())
    }

    pub fn row_ids(&self, source: &str) -> Vec<String> {
        self.windows
            .iter()
            .map(|window| row_id(source, window.ordinal))
            .collect()
    }

    /// Extract all ranges in one character scan without retaining a byte
    /// offset for every character of a potentially large source document.
    pub fn texts<'a>(&self, full_text: &'a str) -> SearchResult<Vec<&'a str>> {
        self.validate()?;
        let endpoints = self
            .windows
            .iter()
            .flat_map(|window| [window.start_char, window.end_char])
            .collect::<BTreeSet<_>>();
        let mut offsets = BTreeMap::new();
        let mut source_chars = 0;
        for (position, (byte, _)) in full_text.char_indices().enumerate() {
            if endpoints.contains(&position) {
                offsets.insert(position, byte);
            }
            source_chars = position.saturating_add(1);
        }
        if source_chars != self.source_chars {
            return Err(invalid(
                "stored semantic window source length does not match the supplied indexed text",
            ));
        }
        offsets.insert(source_chars, full_text.len());
        self.windows
            .iter()
            .map(|window| {
                let start = offsets
                    .get(&window.start_char)
                    .ok_or_else(|| invalid("window start is outside its indexed source"))?;
                let end = offsets
                    .get(&window.end_char)
                    .ok_or_else(|| invalid("window end is outside its indexed source"))?;
                full_text
                    .get(*start..*end)
                    .ok_or_else(|| invalid("window range is not on UTF-8 boundaries"))
            })
            .collect()
    }

    pub fn covered_chars(&self) -> usize {
        let mut covered = 0_usize;
        let mut end = 0_usize;
        for window in &self.windows {
            covered =
                covered.saturating_add(window.end_char.saturating_sub(end.max(window.start_char)));
            end = end.max(window.end_char);
        }
        covered
    }

    pub fn coverage_complete(&self) -> bool {
        self.covered_chars() == self.source_chars
    }
}

#[derive(Debug, Clone)]
struct SourceWindow {
    source: String,
    range: FastWindowRange,
}

#[derive(Debug)]
pub(super) struct FastWindowMapping {
    rows: HashMap<String, SourceWindow>,
    covered_chars: HashMap<String, usize>,
    source_count: usize,
    max_windows: usize,
}

impl FastWindowMapping {
    pub const fn source_count(&self) -> usize {
        self.source_count
    }

    /// The share of the blend's quality weight a source keeps. The quality
    /// tier embeds one prefix of at most `quality_chars`, while windows let the
    /// fast tier see `covered_chars` of the same source, so on a long file the
    /// quality score speaks for a small part of what the fast score does. A
    /// source the fast tier saw no more of (every one-window source) keeps the
    /// whole weight.
    #[allow(clippy::cast_precision_loss)] // A ratio of character counts.
    pub fn quality_weight_share(&self, source: &str, quality_chars: usize) -> f32 {
        self.covered_chars
            .get(source)
            .filter(|&&covered| covered > quality_chars)
            .map_or(1.0, |&covered| quality_chars as f32 / covered as f32)
    }

    fn from_plans<'a>(
        max_per_file: usize,
        plans: impl IntoIterator<Item = (&'a str, &'a FastWindowPlan)>,
        live_rows: &HashSet<String>,
    ) -> SearchResult<Self> {
        let mut rows = HashMap::new();
        let mut covered_chars = HashMap::new();
        let mut source_count = 0;
        let mut max_windows = 1;
        for (source, plan) in plans {
            validate_source_id(source)?;
            plan.validate()?;
            covered_chars.insert(source.to_owned(), plan.covered_chars());
            if plan.max_per_file != max_per_file {
                return Err(invalid(
                    "source window policy disagrees with its generation sentinel",
                ));
            }
            let ids = plan.row_ids(source);
            if ids.iter().any(|id| id.len() > usize::from(u16::MAX)) {
                return Err(invalid(
                    "source document ID exceeds the FSVI limit after adding a semantic window suffix",
                ));
            }
            let present = ids.iter().filter(|id| live_rows.contains(*id)).count();
            if present != ids.len() {
                return Err(invalid(format!(
                    "source {source:?} has only {present} of {} declared fast windows; rebuild the incomplete generation",
                    ids.len(),
                )));
            }
            source_count += 1;
            max_windows = max_windows.max(ids.len());
            for (id, range) in ids.into_iter().zip(&plan.windows) {
                if rows
                    .insert(
                        id,
                        SourceWindow {
                            source: source.to_owned(),
                            range: *range,
                        },
                    )
                    .is_some()
                {
                    return Err(invalid(
                        "semantic window manifests contain duplicate source rows",
                    ));
                }
            }
        }
        if rows.len() != live_rows.len() || live_rows.iter().any(|id| !rows.contains_key(id)) {
            return Err(invalid(
                "fast vector rows do not have exact source-window manifest coverage; rebuild the generation",
            ));
        }
        Ok(Self {
            rows,
            covered_chars,
            source_count,
            max_windows,
        })
    }
}

pub(super) fn load_mapping(
    index_root: &Path,
    index: &VectorIndex,
) -> SearchResult<Option<FastWindowMapping>> {
    let sentinel = super::FsfsRuntime::read_index_sentinel(index_root)?;
    let live_rows = index.live_doc_ids()?;
    let manifests = super::FsfsRuntime::read_matching_manifest_generation(index_root)?;
    if sentinel
        .as_ref()
        .is_none_or(|sentinel| sentinel.fast_window_max_per_file == 1)
    {
        if live_rows.iter().any(|id| id.contains('\0'))
            || manifests.as_ref().is_some_and(|manifests| {
                manifests.values().any(|entry| entry.fast_windows.is_some())
            })
        {
            return Err(invalid(
                "window rows or source plans exist without a matching windowed generation sentinel",
            ));
        }
        return Ok(None);
    }
    let sentinel =
        sentinel.ok_or_else(|| invalid("windowed generation is missing its sentinel"))?;
    if !(2..=MAX_WINDOWS_PER_FILE).contains(&sentinel.fast_window_max_per_file) {
        return Err(invalid("generation has an unsupported fast window policy"));
    }
    let manifests =
        manifests.ok_or_else(|| invalid("windowed generation has no matching source manifests"))?;
    let ordered = manifests.values().cloned().collect::<Vec<_>>();
    if sentinel.indexed_files != manifests.len()
        || sentinel.source_hash_hex != super::index_source_hash_hex(&ordered)
    {
        return Err(invalid(
            "window source manifests do not match their generation sentinel",
        ));
    }
    for manifest in manifests.values() {
        let semantic = manifest.ingestion_class == "full_semantic_lexical";
        if semantic != manifest.fast_windows.is_some() {
            return Err(invalid(
                "windowed generations require a window plan for every semantic source and no plan for lexical-only sources",
            ));
        }
    }
    let plans = manifests.iter().filter_map(|(source, manifest)| {
        manifest
            .fast_windows
            .as_ref()
            .map(|plan| (source.as_str(), plan))
    });
    FastWindowMapping::from_plans(sentinel.fast_window_max_per_file, plans, &live_rows).map(Some)
}

struct SourceFilter<'a> {
    mapping: &'a FastWindowMapping,
    inner: Option<&'a dyn SearchFilter>,
}

impl SearchFilter for SourceFilter<'_> {
    fn matches(&self, row: &str, metadata: Option<&serde_json::Value>) -> bool {
        self.mapping.rows.get(row).is_some_and(|window| {
            self.inner
                .is_none_or(|filter| filter.matches(&window.source, metadata))
        })
    }

    fn name(&self) -> &'static str {
        "fsfs.fast_windows.source_filter"
    }
}

pub(super) struct FastWindowSearchResult {
    pub classified: ClassifiedHits,
    pub winning_windows: HashMap<String, FastWindowRange>,
}

pub(super) fn search_top_k(
    index: &VectorIndex,
    query: &[f32],
    source_limit: usize,
    filter: Option<&dyn SearchFilter>,
    mapping: Option<&FastWindowMapping>,
) -> SearchResult<FastWindowSearchResult> {
    let Some(mapping) = mapping else {
        let classified = index.search_top_k_classified(query, source_limit, filter)?;
        if classified.hits.iter().any(|hit| hit.doc_id.contains('\0')) {
            return Err(invalid(
                "internal window rows cannot be searched without their persisted source mapping",
            ));
        }
        return Ok(FastWindowSearchResult {
            classified,
            winning_windows: HashMap::new(),
        });
    };
    let source_filter = SourceFilter {
        mapping,
        inner: filter,
    };
    // Every source contributes at most max_windows live row IDs. K*N rows
    // therefore contain the best K distinct sources. Reserve every WAL row
    // too: a recovered lower-scored WAL replacement may supersede an old main
    // row only after native heap selection. This keeps source budgets exact
    // without scanning all scores or assuming a fixed duplication heuristic.
    let mut row_limit = if source_limit == 0 {
        0
    } else {
        source_limit
            .saturating_mul(mapping.max_windows)
            .saturating_add(index.wal_record_count())
    };
    let physical_rows = index
        .record_count()
        .saturating_add(index.wal_record_count());
    loop {
        let classified = index.search_top_k_classified(query, row_limit, Some(&source_filter))?;
        let boundary_score = classified.hits.last().map(|hit| hit.score);
        let mut sources = collapse_rows(mapping, classified.hits)?;
        // The native heap breaks ties by physical row index. Widen a tied
        // boundary until every source at the cutoff is available, so source
        // ID and lowest-window-ordinal ties survive compaction and WAL order.
        let cutoff_score = source_limit
            .checked_sub(1)
            .and_then(|position| sources.get(position))
            .map(|(hit, _)| hit.score);
        if classified.zero_signal.is_none()
            && row_limit < physical_rows
            && (sources.len() < source_limit
                || cutoff_score
                    .zip(boundary_score)
                    .is_some_and(|(cutoff, boundary)| !boundary.total_cmp(&cutoff).is_lt()))
        {
            row_limit = row_limit.saturating_mul(2).max(1).min(physical_rows);
            continue;
        }
        sources.truncate(source_limit);
        let winning_windows = sources
            .iter()
            .map(|(hit, window)| (hit.doc_id.to_string(), *window))
            .collect();
        return Ok(FastWindowSearchResult {
            classified: ClassifiedHits {
                hits: sources.into_iter().map(|(hit, _)| hit).collect(),
                zero_signal: classified.zero_signal,
            },
            winning_windows,
        });
    }
}

/// Exact fast-tier scores for named source documents, each collapsed to its
/// best live row as [`search_top_k`] does. A source with no live fast row is
/// absent from the result.
pub(super) fn score_sources(
    index: &VectorIndex,
    query: &[f32],
    sources: HashSet<String>,
    mapping: Option<&FastWindowMapping>,
) -> SearchResult<Vec<VectorHit>> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    // One scan whose budget holds every live row the named sources own (at
    // most max_windows each; native top-k resolves WAL supersession before
    // selection). No named row is evicted, so a truncated source is never
    // mistaken for an absent one.
    let limit = sources
        .len()
        .saturating_mul(mapping.map_or(1, |mapping| mapping.max_windows));
    let named = PredicateFilter::new("fsfs.fast_windows.named_sources", move |source| {
        sources.contains(source)
    });
    let Some(mapping) = mapping else {
        return index.search_top_k(query, limit, Some(&named));
    };
    let filter = SourceFilter {
        mapping,
        inner: Some(&named),
    };
    let rows = index.search_top_k(query, limit, Some(&filter))?;
    Ok(collapse_rows(mapping, rows)?
        .into_iter()
        .map(|(hit, _)| hit)
        .collect())
}

fn collapse_rows(
    mapping: &FastWindowMapping,
    hits: Vec<VectorHit>,
) -> SearchResult<Vec<(VectorHit, FastWindowRange)>> {
    let mut sources = HashMap::<String, (VectorHit, FastWindowRange)>::new();
    for mut hit in hits {
        let window = mapping
            .rows
            .get(hit.doc_id.as_str())
            .ok_or_else(|| invalid("selected vector row has no source-window mapping"))?;
        hit.doc_id = window.source.as_str().into();
        match sources.entry(window.source.clone()) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert((hit, window.range));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let (incumbent, range) = entry.get();
                if hit.score.total_cmp(&incumbent.score).is_gt()
                    || (hit.score.total_cmp(&incumbent.score).is_eq()
                        && window.range.ordinal < range.ordinal)
                {
                    entry.insert((hit, window.range));
                }
            }
        }
    }
    let mut sources = sources.into_values().collect::<Vec<_>>();
    sources.sort_unstable_by(|(left, _), (right, _)| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    Ok(sources)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::{Embedder, EmbeddingIdentityBundleV1, ModelCategory, SearchFuture};
    use std::sync::Arc;

    #[test]
    fn bounded_unicode_window_plan_covers_tail_and_reports_gaps() {
        let text = "λ界🦀".repeat(4_000);
        let complete = plan(&text, 128).unwrap();
        assert!(complete.coverage_complete());
        assert_eq!(complete.windows.last().unwrap().end_char, 12_000);
        let capped = plan(&text, 3).unwrap();
        assert_eq!(capped.windows.len(), 3);
        assert_eq!(capped.windows[0].start_char, 0);
        assert_eq!(capped.windows[2].end_char, 12_000);
        assert!(!capped.coverage_complete());
        assert_eq!(capped.covered_chars(), 6_000);
        let texts = capped.texts(&text).unwrap();
        assert!(
            texts
                .iter()
                .all(|window| window.chars().count() == WINDOW_CHARS)
        );
        assert!(texts.iter().all(|window| !window.contains('\u{fffd}')));
        assert_eq!(plan("", 2).unwrap().texts("").unwrap(), [""]);
        assert!(plan(&text, 0).is_err());
        assert!(plan(&text, 129).is_err());
        let mut forged = capped.clone();
        forged.windows[1].start_char += 1;
        assert!(forged.validate().is_err());
        assert!(capped.texts("changed body").is_err());
    }

    #[test]
    fn internal_window_ids_do_not_alias_source_suffixes() {
        for source in ["src/file#w1.rs", "src/λ:123.rs", "doc\nname"] {
            assert_eq!(row_id(source, 0), source);
            let row = row_id(source, 7);
            assert_eq!(source_id(&row), source);
            assert_eq!(window_ordinal(&row), 7);
            assert!(is_window_row(&row));
        }
        for malformed in [
            "a\0fsfs-window-v1:0",
            "a\0fsfs-window-v1:01",
            "a\0fsfs-window-v1:+1",
            "a\0fsfs-window-v1:128",
            "a\0fsfs-window-v1:1\0b",
            "\0fsfs-window-v1:1",
        ] {
            assert!(!is_window_row(malformed));
            assert_eq!(source_id(malformed), malformed);
            assert!(validate_source_id(malformed).is_err());
        }
    }

    #[test]
    fn source_top_k_uses_max_sim_and_filters_before_window_budget() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("fast.fsvi");
        let plans = [
            ("src/a.rs", plan(&"a".repeat(20_000), 12).unwrap()),
            ("src/b.rs", plan(&"b".repeat(4_000), 12).unwrap()),
            ("excluded/c.rs", plan(&"c".repeat(4_000), 12).unwrap()),
        ];
        let mut writer = VectorIndex::create(&path, "test", 2).unwrap();
        for (source, plan) in &plans {
            for window in &plan.windows {
                let score = if *source == "excluded/c.rs" {
                    1.0
                } else if *source == "src/a.rs" {
                    0.9
                } else if window.ordinal == 2 {
                    0.8
                } else {
                    0.1
                };
                writer
                    .write_record(&row_id(source, window.ordinal), &[score, 0.0])
                    .unwrap();
            }
        }
        writer.finish().unwrap();
        let index = VectorIndex::open_read_only(&path).unwrap();
        let mapping = FastWindowMapping::from_plans(
            12,
            plans.iter().map(|(source, plan)| (*source, plan)),
            &index.live_doc_ids().unwrap(),
        )
        .unwrap();
        let filter = PredicateFilter::new("source.rs", |id| {
            id.starts_with("src/")
                && std::path::Path::new(id)
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
        });
        let result = search_top_k(&index, &[1.0, 0.0], 2, Some(&filter), Some(&mapping)).unwrap();
        assert_eq!(
            result
                .classified
                .hits
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["src/a.rs", "src/b.rs"]
        );
        assert_eq!(result.winning_windows["src/a.rs"].ordinal, 0);
        assert_eq!(result.winning_windows["src/b.rs"].ordinal, 2);
        assert_eq!(mapping.source_count(), 3);
        assert!(result.classified.zero_signal.is_none());
        let mut missing = index.live_doc_ids().unwrap();
        missing.remove(&row_id("src/b.rs", 2));
        assert!(
            FastWindowMapping::from_plans(
                12,
                plans.iter().map(|(source, plan)| (*source, plan)),
                &missing
            )
            .is_err()
        );
        missing.retain(|row| source_id(row) != "src/b.rs");
        assert!(
            FastWindowMapping::from_plans(
                12,
                plans.iter().map(|(source, plan)| (*source, plan)),
                &missing
            )
            .is_err()
        );
        assert!(search_top_k(&index, &[1.0, 0.0], 2, None, None).is_err());
    }

    #[test]
    fn source_top_k_includes_boundary_ties_and_duplicate_physical_rows() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("ties.fsvi");
        let sources = (0..20)
            .map(|position| format!("source-{position:02}.rs"))
            .collect::<Vec<_>>();
        let plan = plan(&"x".repeat(4_000), 3).unwrap();
        let mut writer = VectorIndex::create(&path, "test", 2).unwrap();
        for source in &sources {
            for row in plan.row_ids(source) {
                writer.write_record(&row, &[1.0, 0.0]).unwrap();
            }
        }
        // Legacy v1 can contain duplicate physical IDs even though the live
        // document-ID set has one entry. These cannot consume source slots.
        for _ in 0..40 {
            writer
                .write_record(&row_id("source-19.rs", 2), &[1.0, 0.0])
                .unwrap();
        }
        writer.finish().unwrap();
        let index = VectorIndex::open_read_only(&path).unwrap();
        let mapping = FastWindowMapping::from_plans(
            3,
            sources.iter().map(|source| (source.as_str(), &plan)),
            &index.live_doc_ids().unwrap(),
        )
        .unwrap();
        let hits = search_top_k(&index, &[1.0, 0.0], 2, None, Some(&mapping)).unwrap();
        assert_eq!(
            hits.classified
                .hits
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["source-00.rs", "source-01.rs"]
        );
        assert_eq!(hits.winning_windows["source-00.rs"].ordinal, 0);
        assert_eq!(hits.winning_windows["source-01.rs"].ordinal, 0);
        let zero = search_top_k(&index, &[1.0, 0.0], 0, None, Some(&mapping)).unwrap();
        assert!(zero.classified.hits.is_empty());
        assert!(zero.classified.zero_signal.is_some());
        assert!(zero.winning_windows.is_empty());
        let none = PredicateFilter::new("none", |_| false);
        let filtered = search_top_k(&index, &[1.0, 0.0], 2, Some(&none), Some(&mapping)).unwrap();
        assert!(filtered.classified.hits.is_empty());
        assert!(filtered.classified.zero_signal.is_some());
    }

    #[test]
    fn score_sources_scores_every_named_source_and_omits_absent_ones() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("named.fsvi");
        let plan = plan(&"x".repeat(4_000), 3).unwrap();
        assert_eq!(plan.windows.len(), 3);
        let mut writer = VectorIndex::create(&path, "test", 2).unwrap();
        for ordinal in 0..3 {
            // Every src/b.rs window outranks src/a.rs's best one, so a budget
            // of one row per named source loses src/a.rs entirely.
            let a = if ordinal == 2 { 0.4 } else { 0.2 };
            writer
                .write_record(&row_id("src/a.rs", ordinal), &[a, 0.0])
                .unwrap();
            writer
                .write_record(&row_id("src/b.rs", ordinal), &[0.5, 0.0])
                .unwrap();
            writer
                .write_record(&row_id("src/unnamed.rs", ordinal), &[1.0, 0.0])
                .unwrap();
        }
        writer.finish().unwrap();
        let index = VectorIndex::open_read_only(&path).unwrap();
        let mapping = FastWindowMapping::from_plans(
            3,
            ["src/a.rs", "src/b.rs", "src/unnamed.rs"]
                .into_iter()
                .map(|source| (source, &plan)),
            &index.live_doc_ids().unwrap(),
        )
        .unwrap();
        let named = |ids: &[&str]| {
            ids.iter()
                .map(|id| (*id).to_owned())
                .collect::<HashSet<_>>()
        };
        let scores = |hits: Vec<VectorHit>| {
            hits.into_iter()
                .map(|hit| (hit.doc_id.to_string(), (hit.score * 1_000.0).round()))
                .collect::<Vec<_>>()
        };
        let windowed = score_sources(
            &index,
            &[1.0, 0.0],
            named(&["src/a.rs", "src/b.rs", "src/absent.rs"]),
            Some(&mapping),
        )
        .unwrap();
        assert_eq!(
            scores(windowed),
            [("src/b.rs".to_owned(), 500.0), ("src/a.rs".to_owned(), 400.0)]
        );
        assert!(
            score_sources(&index, &[1.0, 0.0], HashSet::new(), Some(&mapping))
                .unwrap()
                .is_empty()
        );

        // A crash between a WAL append and the main-row tombstone leaves both
        // physical rows live; the named source scores as its WAL replacement.
        let flat_path = temporary.path().join("flat.fsvi");
        let mut writer = VectorIndex::create(&flat_path, "test", 2).unwrap();
        writer.write_record("x.rs", &[1.0, 0.0]).unwrap();
        writer.write_record("y.rs", &[0.3, 0.0]).unwrap();
        writer.write_record("z.rs", &[0.9, 0.0]).unwrap();
        writer.finish().unwrap();
        let prior_main = std::fs::read(&flat_path).unwrap();
        let mut writer = VectorIndex::open(&flat_path).unwrap();
        writer.append("x.rs", &[0.1, 0.0]).unwrap();
        drop(writer);
        std::fs::write(&flat_path, prior_main).unwrap();
        let flat = VectorIndex::open_read_only(&flat_path).unwrap();
        assert_eq!(flat.wal_record_count(), 1);
        let flat_scores = score_sources(&flat, &[1.0, 0.0], named(&["x.rs", "y.rs"]), None).unwrap();
        assert_eq!(
            scores(flat_scores),
            [("y.rs".to_owned(), 300.0), ("x.rs".to_owned(), 100.0)]
        );
    }

    #[test]
    fn long_windowed_sources_keep_their_covered_share_of_the_quality_weight() {
        use super::super::FsfsRuntime;
        use crate::FsfsConfig;
        use crate::query_execution::SemanticCandidate;

        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("share.fsvi");
        let long = plan(&"x".repeat(20_000), 128).unwrap();
        let short = plan(&"y".repeat(1_500), 128).unwrap();
        assert_eq!(long.covered_chars(), 20_000);
        assert_eq!(short.windows.len(), 1);
        let mut writer = VectorIndex::create(&path, "test", 2).unwrap();
        for (source, plan) in [("long.rs", &long), ("short.rs", &short)] {
            for row in plan.row_ids(source) {
                writer.write_record(&row, &[1.0, 0.0]).unwrap();
            }
        }
        writer.finish().unwrap();
        let index = VectorIndex::open_read_only(&path).unwrap();
        let mapping = FastWindowMapping::from_plans(
            128,
            [("long.rs", &long), ("short.rs", &short)],
            &index.live_doc_ids().unwrap(),
        )
        .unwrap();
        assert!((mapping.quality_weight_share("long.rs", 2_000) - 0.1).abs() < 1e-6);
        assert_eq!(mapping.quality_weight_share("short.rs", 2_000), 1.0);
        assert_eq!(mapping.quality_weight_share("unmapped.rs", 2_000), 1.0);

        // The tiers disagree: the windows rank long.rs first, the one prefix
        // vector ranks short.rs first.
        let fast = [
            SemanticCandidate::new("long.rs", 0.9),
            SemanticCandidate::new("short.rs", 0.1),
        ];
        let quality = [
            SemanticCandidate::new("short.rs", 0.9),
            SemanticCandidate::new("long.rs", 0.1),
        ];
        let mut config = FsfsConfig::default();
        config.search.quality_weight = 0.7;
        let runtime = FsfsRuntime::new(config);
        let (uniform, _) = runtime.blend_semantic_candidates(&fast, &quality, None);
        assert_eq!(uniform[0].doc_id, "short.rs");
        let (weighted, details) =
            runtime.blend_semantic_candidates(&fast, &quality, Some(&mapping));
        assert_eq!(weighted[0].doc_id, "long.rs");
        assert!((weighted[0].score - 0.93).abs() < 1e-5);
        let long_detail = details.iter().find(|hit| hit.path == "long.rs").unwrap();
        assert!((long_detail.quality.as_ref().unwrap().weight - 0.07).abs() < 1e-6);
        assert!((long_detail.fast.as_ref().unwrap().weight - 0.93).abs() < 1e-6);
        let short_detail = details.iter().find(|hit| hit.path == "short.rs").unwrap();
        assert!((short_detail.quality.as_ref().unwrap().weight - 0.7).abs() < 1e-6);
    }

    struct DeepPassageEmbedder(EmbeddingIdentityBundleV1);

    impl Embedder for DeepPassageEmbedder {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.0)
        }

        fn embed<'a>(
            &'a self,
            _cx: &'a asupersync::Cx,
            text: &'a str,
        ) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                Ok(
                    if text == "thermal safety" || text.contains("orbitalshield") {
                        vec![1.0, 0.0]
                    } else if text.contains("planetarium") {
                        vec![0.6, 0.8]
                    } else {
                        vec![0.0, 1.0]
                    },
                )
            })
        }

        fn dimension(&self) -> usize {
            2
        }
        fn id(&self) -> &'static str {
            "deep-passage-test"
        }
        fn model_name(&self) -> &str {
            self.id()
        }
        fn is_semantic(&self) -> bool {
            true
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    struct RestoreFast(Option<Arc<dyn Embedder>>);

    impl Drop for RestoreFast {
        fn drop(&mut self) {
            super::super::set_test_fast_embedder(self.0.take());
        }
    }

    #[test]
    fn fast_window_product_search_recalls_deep_source_and_retains_indexed_passage() {
        use super::super::{FsfsRuntime, SearchExecutionMode};
        use crate::{CliCommand, CliInput, FsfsConfig};

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let temporary = tempfile::tempdir().unwrap();
            let source = temporary.path().join("source");
            std::fs::create_dir(&source).unwrap();
            let body = format!(
                "{}{}",
                "prologue background ".repeat(600),
                "orbitalshield ".repeat(160)
            );
            std::fs::write(source.join("deep.md"), &body).unwrap();
            std::fs::write(source.join("distractor.md"), "planetarium backdrop").unwrap();
            let _restore = RestoreFast(super::super::test_fast_embedder_override());
            super::super::set_test_fast_embedder(Some(Arc::new(DeepPassageEmbedder(
                EmbeddingIdentityBundleV1::explicit_test_model("deep-passage-test", 2),
            ))));
            let mut config = FsfsConfig::default();
            config.indexing.offline = true;
            config.indexing.quality_model.clear();
            config.search.fast_only = true;
            config.search.rerank = false;
            "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
            let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Index,
                target_path: Some(source.clone()),
                index_dir: Some(temporary.path().join("prefix")),
                quiet: true,
                ..CliInput::default()
            });
            runtime
                .run_one_shot_index_scaffold_internal(
                    &cx,
                    CliCommand::Index,
                    |_| Ok(()),
                    false,
                    false,
                )
                .await
                .unwrap();
            let prefix = runtime
                .execute_search_phase_artifacts_with_mode(
                    &cx,
                    "thermal safety",
                    1,
                    SearchExecutionMode::FastOnly,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(prefix[0].payload.hits[0].path, "distractor.md");

            runtime.config.indexing.fast_window_max_per_file = 4;
            runtime.cli_input.index_dir = Some(temporary.path().join("windowed"));
            runtime
                .run_one_shot_index_scaffold_internal(
                    &cx,
                    CliCommand::Index,
                    |_| Ok(()),
                    false,
                    false,
                )
                .await
                .unwrap();
            runtime.config.indexing.fast_window_max_per_file = 1;
            let windowed = runtime
                .execute_search_phase_artifacts_with_mode(
                    &cx,
                    "thermal safety",
                    1,
                    SearchExecutionMode::FastOnly,
                    None,
                )
                .await
                .unwrap();
            let hit = &windowed[0].payload.hits[0];
            assert_eq!(hit.path, "deep.md");
            // Payload ranks are 0-based: the collapsed deep source is first.
            assert_eq!(hit.semantic_rank, Some(0));
            assert!(hit.lexical_rank.is_none());
            assert!(hit.snippet.as_deref().unwrap().contains("orbitalshield"));

            std::fs::write(source.join("deep.md"), "changed live source").unwrap();
            let retained = runtime
                .execute_search_phase_artifacts_with_mode(
                    &cx,
                    "thermal safety",
                    1,
                    SearchExecutionMode::FastOnly,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(retained[0].payload.hits[0].snippet, hit.snippet);
            assert_eq!(retained[0].payload.hits[0].line, None);
            runtime.cli_input.filter = Some("path:distractor.md".to_owned());
            let filtered = runtime
                .execute_search_phase_artifacts_with_mode(
                    &cx,
                    "thermal safety",
                    1,
                    SearchExecutionMode::FastOnly,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(filtered[0].payload.hits[0].path, "distractor.md");
        });
    }

    #[test]
    fn fast_window_daemon_rebind_serves_deferred_lexical_and_recovers_semantic_windows() {
        use super::super::{FsfsRuntime, SearchExecutionMode};
        use crate::{CliCommand, CliInput, FsfsConfig};

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let temporary = tempfile::tempdir().unwrap();
            let source = temporary.path().join("source");
            let root = temporary.path().join("windowed");
            std::fs::create_dir(&source).unwrap();
            std::fs::write(
                source.join("deep.md"),
                format!(
                    "{}{}",
                    "prologue background ".repeat(600),
                    "orbitalshield ".repeat(160),
                ),
            )
            .unwrap();
            std::fs::write(source.join("distractor.md"), "planetarium backdrop").unwrap();
            let _restore = RestoreFast(super::super::test_fast_embedder_override());
            super::super::set_test_fast_embedder(Some(Arc::new(DeepPassageEmbedder(
                EmbeddingIdentityBundleV1::explicit_test_model("deep-passage-test", 2),
            ))));
            let mut config = FsfsConfig::default();
            config.indexing.offline = true;
            config.indexing.quality_model.clear();
            config.indexing.fast_window_max_per_file = 4;
            config.search.fast_only = true;
            config.search.rerank = false;
            "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
            let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Index,
                target_path: Some(source),
                index_dir: Some(root.clone()),
                quiet: true,
                ..CliInput::default()
            });
            runtime
                .run_one_shot_index_scaffold_internal(
                    &cx,
                    CliCommand::Index,
                    |_| Ok(()),
                    false,
                    false,
                )
                .await
                .unwrap();
            let request = |query: &str, mode: &str| {
                FsfsRuntime::parse_search_serve_request(
                    &serde_json::json!({
                        "query": query, "limit": 2, "mode": mode,
                    })
                    .to_string(),
                )
                .unwrap()
            };
            let mut resources = runtime.prepare_search_serve_resources(&cx).await.unwrap();
            assert_eq!(
                resources
                    .fast_window_mapping
                    .as_ref()
                    .unwrap()
                    .source_count(),
                2
            );
            let mut hot_cache = HashMap::new();
            let healthy = runtime
                .execute_search_serve_request(
                    &cx,
                    request("thermal safety", "fast_only"),
                    &mut resources,
                    &mut hot_cache,
                    true,
                )
                .await
                .unwrap();
            assert_eq!(healthy.payloads[0].hits[0].path, "deep.md");
            let full_key = runtime
                .search_cache_key("thermal safety", 2, SearchExecutionMode::Full)
                .unwrap();
            let fast_key = runtime
                .search_cache_key("thermal safety", 2, SearchExecutionMode::FastOnly)
                .unwrap();
            assert!(hot_cache.contains_key(&fast_key));
            let healthy_fingerprint = resources.generation_fingerprint.clone();

            // Model work may be incomplete while its lexical publication is
            // durable. Remove a required window so strict mapping admission
            // really fails; changing the completion bit alone is insufficient.
            let mut sentinel = FsfsRuntime::read_index_sentinel(&root).unwrap().unwrap();
            sentinel.generation_complete = false;
            runtime.write_index_sentinel(&root, &sentinel).unwrap();
            let missing_row = row_id("deep.md", 3);
            let vector_path = root.join(super::super::FSFS_VECTOR_INDEX_FILE);
            let vector_wal = frankensearch_index::wal::wal_path_for(&vector_path);
            let deferred_path = temporary.path().join("deferred.fsvi");
            let deferred_wal = frankensearch_index::wal::wal_path_for(&deferred_path);
            std::fs::copy(&vector_path, &deferred_path).unwrap();
            if vector_wal.exists() {
                std::fs::copy(&vector_wal, &deferred_wal).unwrap();
            }
            let replacement = {
                // The warm daemon retains a shared mapping lock on its old
                // inode. Publish a separate replacement, as a generation
                // transition must, rather than trying to mutate that reader.
                let mut index = VectorIndex::open(&deferred_path).unwrap();
                let ordinal = (0..index.record_count())
                    .find(|ordinal| index.doc_id_at(*ordinal).unwrap() == missing_row)
                    .unwrap();
                let vector = index.vector_at_f32(ordinal).unwrap();
                assert!(index.soft_delete(&missing_row).unwrap());
                index.compact().unwrap();
                assert!(load_mapping(&root, &index).is_err());
                vector
            };
            std::fs::rename(&deferred_path, &vector_path).unwrap();
            if deferred_wal.exists() {
                std::fs::rename(&deferred_wal, &vector_wal).unwrap();
            }
            assert!(
                resources
                    .vector_index
                    .as_ref()
                    .unwrap()
                    .live_doc_ids()
                    .unwrap()
                    .contains(&missing_row)
            );
            let lexical = runtime
                .execute_search_serve_request(
                    &cx,
                    request("orbitalshield", "lexical_only"),
                    &mut resources,
                    &mut hot_cache,
                    true,
                )
                .await
                .unwrap();
            assert!(!lexical.cached);
            assert_eq!(lexical.payloads[0].hits[0].path, "deep.md");
            assert!(resources.vector_index.is_none());
            assert!(resources.fast_window_mapping.is_none());
            assert!(resources.quality_vector_index.is_none());
            assert_ne!(resources.generation_fingerprint, healthy_fingerprint);
            assert!(!hot_cache.contains_key(&fast_key));
            let deferred_fingerprint = resources.generation_fingerprint.clone();

            // A stale hot entry must never bypass explicit semantic readiness.
            hot_cache.insert(full_key.clone(), healthy.payloads.clone());
            hot_cache.insert(fast_key.clone(), healthy.payloads);
            for mode in ["full", "fast_only"] {
                let error = runtime
                    .execute_search_serve_request(
                        &cx,
                        request("thermal safety", mode),
                        &mut resources,
                        &mut hot_cache,
                        true,
                    )
                    .await
                    .unwrap_err();
                assert!(
                    matches!(error, SearchError::InvalidConfig { field, value, .. }
                    if field == "semantic.index_generation"
                        && matches!(value.as_str(), "incomplete" | "deferred_rows"))
                );
            }
            let mut fresh = runtime.prepare_search_serve_resources(&cx).await.unwrap();
            assert!(fresh.vector_index.is_none());
            assert!(fresh.fast_window_mapping.is_none());
            let fresh_lexical = runtime
                .execute_search_serve_request(
                    &cx,
                    request("orbitalshield", "lexical_only"),
                    &mut fresh,
                    &mut HashMap::new(),
                    false,
                )
                .await
                .unwrap();
            assert_eq!(fresh_lexical.payloads[0].hits[0].path, "deep.md");

            let repaired_path = temporary.path().join("repaired.fsvi");
            let repaired_wal = frankensearch_index::wal::wal_path_for(&repaired_path);
            std::fs::copy(&vector_path, &repaired_path).unwrap();
            if vector_wal.exists() {
                std::fs::copy(&vector_wal, &repaired_wal).unwrap();
            }
            {
                let mut index = VectorIndex::open(&repaired_path).unwrap();
                index.append(&missing_row, &replacement).unwrap();
                index.compact().unwrap();
            }
            std::fs::rename(&repaired_path, &vector_path).unwrap();
            if repaired_wal.exists() {
                std::fs::rename(&repaired_wal, &vector_wal).unwrap();
            }
            sentinel.generation_complete = true;
            runtime.write_index_sentinel(&root, &sentinel).unwrap();
            let recovered_lexical = runtime
                .execute_search_serve_request(
                    &cx,
                    request("orbitalshield", "lexical_only"),
                    &mut resources,
                    &mut hot_cache,
                    true,
                )
                .await
                .unwrap();
            assert!(!recovered_lexical.cached);
            assert_ne!(resources.generation_fingerprint, deferred_fingerprint);
            assert!(resources.vector_index.is_some());
            assert_eq!(
                resources
                    .fast_window_mapping
                    .as_ref()
                    .unwrap()
                    .source_count(),
                2
            );
            assert!(!hot_cache.contains_key(&full_key));
            assert!(!hot_cache.contains_key(&fast_key));
            let recovered = runtime
                .execute_search_serve_request(
                    &cx,
                    request("thermal safety", "full"),
                    &mut resources,
                    &mut hot_cache,
                    true,
                )
                .await
                .unwrap();
            assert!(!recovered.cached);
            assert_eq!(recovered.payloads[0].hits[0].path, "deep.md");
            assert!(
                recovered.payloads[0].hits[0]
                    .snippet
                    .as_deref()
                    .unwrap()
                    .contains("orbitalshield")
            );
        });
    }
}
