//! Source-preserving fast semantic windows for fsfs-owned vector generations.
//!
//! The persisted source manifest describes every window. Generated FSVI keys
//! are internal identities only: filters, budgets, fusion and output always
//! operate on source document IDs.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use frankensearch_core::filter::SearchFilter;
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
    source_count: usize,
    max_windows: usize,
}

impl FastWindowMapping {
    pub const fn source_count(&self) -> usize {
        self.source_count
    }

    fn from_plans<'a>(
        max_per_file: usize,
        plans: impl IntoIterator<Item = (&'a str, &'a FastWindowPlan)>,
        live_rows: &HashSet<String>,
    ) -> SearchResult<Self> {
        let mut rows = HashMap::new();
        let mut source_count = 0;
        let mut max_windows = 1;
        for (source, plan) in plans {
            validate_source_id(source)?;
            plan.validate()?;
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

    fn name(&self) -> &str {
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
    use frankensearch_core::filter::PredicateFilter;
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
            id.starts_with("src/") && id.ends_with(".rs")
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
        fn id(&self) -> &str {
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
            assert_eq!(hit.semantic_rank, Some(1));
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
}
