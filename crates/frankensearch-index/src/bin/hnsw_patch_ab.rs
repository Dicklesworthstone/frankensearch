//! Evidence-capturing A/B runner for the pinned `hnsw_rs` graph-construction patch.
//!
//! This binary deliberately links the published baseline and the candidate fork
//! into one executable. Build/RSS measurements run in fresh child processes;
//! query measurements use the same corpus, queries, process, and ABBA schedule.
//! Smoke results are provisional. High-scale runs remain fail-closed for a
//! performance decision until `bd-u3wt.1` freezes the experiment contract,
//! replicates query timings across fresh graphs, binds build-time source state,
//! and adds supported tail inference.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::env;
use std::error::Error;
use std::fs;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use frankensearch_index::recall_certificate::mean_recall_lower_bound_bernstein;
use hnsw_rs::prelude::{AnnT as CandidateAnnT, DistDot as CandidateDistDot, Hnsw as CandidateHnsw};
use hnsw_rs_034::prelude::{
    AnnT as BaselineAnnT, DistDot as BaselineDistDot, Hnsw as BaselineHnsw,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "frankensearch-index.hnsw-patch-ab.v4";
const CHILD_SENTINEL: &str = "HNSW_PATCH_AB_CHILD_JSON=";
const CANDIDATE_REV: &str = "18a5a1a9982138822c34d4c3fb29f4c883715069";
const CANDIDATE_REVERSE_EDGE_REV: &str = "461bedfb992f3461a25f6d6daed1e70c2de05f6a";
const CANDIDATE_LOGICAL_LAYER_REV: &str = "18a5a1a9982138822c34d4c3fb29f4c883715069";
const BASELINE_VERSION: &str = "0.3.4";
const WORKSPACE_SOURCE_RECEIPT_SCHEMA: &str = "frankensearch.hnsw-workspace-source-receipt.v1";
const WORKSPACE_SOURCE_FIXED_INPUTS: &[&str] = &[
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    "crates/frankensearch-core/Cargo.toml",
    "crates/frankensearch-index/Cargo.toml",
    "crates/frankensearch-index/build.rs",
];
const QUERY_NULL_TOLERANCE: f64 = 0.05;
const BUILD_NULL_TOLERANCE: f64 = 0.10;
const MAX_MEAN_RECALL_REGRESSION: f64 = 0.005;
const MIN_ABSOLUTE_RECALL: f64 = 0.95;
const PRIMARY_SIZE: usize = 100_000;
const PRIMARY_EF_SEARCH: usize = 100;
const FULL_ADMISSION_HOLD: &str = "bd-u3wt.1 is open: full performance admission requires a \
    frozen corpus/config contract, ABBA-paired multi-graph query timings, build/start/end source \
    binding review, multiplicity control, and supported tail inference";

include!(concat!(
    env!("OUT_DIR"),
    "/hnsw_workspace_source_receipt.rs"
));

type DynError = Box<dyn Error + Send + Sync>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EngineKind {
    Baseline,
    Candidate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PerformanceClaimStatus {
    Allow,
    Block,
    Quarantine,
    NoClaim,
}

impl PerformanceClaimStatus {
    const fn label(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Block => "block",
            Self::Quarantine => "quarantine",
            Self::NoClaim => "no_claim",
        }
    }
}

impl EngineKind {
    fn parse(value: &str) -> Result<Self, DynError> {
        match value {
            "baseline" => Ok(Self::Baseline),
            "candidate" => Ok(Self::Candidate),
            _ => Err(format!("unknown engine {value:?}; expected baseline or candidate").into()),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Candidate => "candidate",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct Config {
    profile: String,
    sizes: Vec<usize>,
    dimension: usize,
    clusters: usize,
    noise: f32,
    m: usize,
    ef_construction: usize,
    max_layer: usize,
    ef_search: Vec<usize>,
    k: usize,
    holdout_queries: usize,
    query_passes: usize,
    build_repetitions: usize,
    warmup_passes: usize,
    corpus_slab: Option<PathBuf>,
    corpus_manifest: Option<PathBuf>,
    corpus_source_manifest: Option<PathBuf>,
    output: Option<PathBuf>,
    child_engine: Option<EngineKind>,
    child_size: Option<usize>,
    child_repetition: Option<usize>,
    child_slot: Option<usize>,
}

impl Config {
    fn smoke() -> Self {
        Self {
            profile: "smoke".to_owned(),
            sizes: vec![1_000],
            dimension: 32,
            clusters: 16,
            noise: 0.15,
            m: 16,
            ef_construction: 100,
            max_layer: 16,
            ef_search: vec![40, 100],
            k: 10,
            holdout_queries: 32,
            query_passes: 2,
            build_repetitions: 1,
            warmup_passes: 1,
            corpus_slab: None,
            corpus_manifest: None,
            corpus_source_manifest: None,
            output: None,
            child_engine: None,
            child_size: None,
            child_repetition: None,
            child_slot: None,
        }
    }

    fn full() -> Self {
        Self {
            profile: "full".to_owned(),
            sizes: vec![50_000, 100_000],
            dimension: 128,
            clusters: 256,
            noise: 0.15,
            m: 16,
            ef_construction: 200,
            max_layer: 16,
            ef_search: vec![40, 100, 200],
            k: 10,
            holdout_queries: 300,
            query_passes: 10,
            build_repetitions: 10,
            warmup_passes: 2,
            corpus_slab: None,
            corpus_manifest: None,
            corpus_source_manifest: None,
            output: None,
            child_engine: None,
            child_size: None,
            child_repetition: None,
            child_slot: None,
        }
    }

    fn parse() -> Result<Self, DynError> {
        let args: Vec<String> = env::args().skip(1).collect();
        if args.iter().any(|arg| arg == "--full") && args.iter().any(|arg| arg == "--smoke") {
            return Err("--smoke and --full are mutually exclusive".into());
        }
        let mut config = if args.iter().any(|arg| arg == "--full") {
            Self::full()
        } else {
            Self::smoke()
        };

        let mut index = 0;
        while index < args.len() {
            let flag = &args[index];
            match flag.as_str() {
                "--smoke" | "--full" => {
                    index += 1;
                }
                "--sizes" => {
                    config.sizes = parse_usize_list(required_value(&args, index)?, "sizes")?;
                    index += 2;
                }
                "--dim" => {
                    config.dimension = parse_usize(required_value(&args, index)?, "dim")?;
                    index += 2;
                }
                "--clusters" => {
                    config.clusters = parse_usize(required_value(&args, index)?, "clusters")?;
                    index += 2;
                }
                "--noise" => {
                    config.noise = required_value(&args, index)?
                        .parse()
                        .map_err(|_| format!("invalid --noise value {:?}", args[index + 1]))?;
                    index += 2;
                }
                "--m" => {
                    config.m = parse_usize(required_value(&args, index)?, "m")?;
                    index += 2;
                }
                "--ef-construction" => {
                    config.ef_construction =
                        parse_usize(required_value(&args, index)?, "ef-construction")?;
                    index += 2;
                }
                "--max-layer" => {
                    config.max_layer = parse_usize(required_value(&args, index)?, "max-layer")?;
                    index += 2;
                }
                "--ef-search" => {
                    config.ef_search =
                        parse_usize_list(required_value(&args, index)?, "ef-search")?;
                    index += 2;
                }
                "--k" => {
                    config.k = parse_usize(required_value(&args, index)?, "k")?;
                    index += 2;
                }
                "--holdout-queries" => {
                    config.holdout_queries =
                        parse_usize(required_value(&args, index)?, "holdout-queries")?;
                    index += 2;
                }
                "--query-passes" => {
                    config.query_passes =
                        parse_usize(required_value(&args, index)?, "query-passes")?;
                    index += 2;
                }
                "--build-repetitions" => {
                    config.build_repetitions =
                        parse_usize(required_value(&args, index)?, "build-repetitions")?;
                    index += 2;
                }
                "--warmup-passes" => {
                    config.warmup_passes =
                        parse_usize(required_value(&args, index)?, "warmup-passes")?;
                    index += 2;
                }
                "--corpus-slab" => {
                    config.corpus_slab = Some(PathBuf::from(required_value(&args, index)?));
                    index += 2;
                }
                "--corpus-manifest" => {
                    config.corpus_manifest = Some(PathBuf::from(required_value(&args, index)?));
                    index += 2;
                }
                "--corpus-source-manifest" => {
                    config.corpus_source_manifest =
                        Some(PathBuf::from(required_value(&args, index)?));
                    index += 2;
                }
                "--output" => {
                    config.output = Some(PathBuf::from(required_value(&args, index)?));
                    index += 2;
                }
                "--child-engine" => {
                    config.child_engine = Some(EngineKind::parse(required_value(&args, index)?)?);
                    index += 2;
                }
                "--child-size" => {
                    config.child_size =
                        Some(parse_usize(required_value(&args, index)?, "child-size")?);
                    index += 2;
                }
                "--child-repetition" => {
                    config.child_repetition = Some(parse_usize(
                        required_value(&args, index)?,
                        "child-repetition",
                    )?);
                    index += 2;
                }
                "--child-slot" => {
                    config.child_slot =
                        Some(parse_usize(required_value(&args, index)?, "child-slot")?);
                    index += 2;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument {flag:?}; use --help").into()),
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), DynError> {
        if self.sizes.is_empty()
            || self.ef_search.is_empty()
            || self.dimension == 0
            || self.clusters == 0
            || self.m == 0
            || self.ef_construction == 0
            || self.max_layer == 0
            || self.k == 0
            || self.holdout_queries == 0
            || self.query_passes == 0
            || self.build_repetitions == 0
            || self.warmup_passes == 0
        {
            return Err("all counts and list-valued settings must be non-zero".into());
        }
        if self.sizes.iter().any(|&size| size < self.k) {
            return Err("every corpus size must be at least k".into());
        }
        if !self.noise.is_finite() || self.noise < 0.0 {
            return Err("noise must be finite and non-negative".into());
        }
        if self.ef_search.iter().any(|&ef| ef < self.k) {
            return Err("every ef-search value must be at least k".into());
        }
        if self.max_layer > 16 {
            return Err("max-layer must not exceed hnsw_rs's 16-layer limit".into());
        }
        if self.m > u8::MAX.into() {
            return Err("m must fit in hnsw_rs's u8 neighborhood limit".into());
        }
        let unique_sizes: HashSet<_> = self.sizes.iter().copied().collect();
        if unique_sizes.len() != self.sizes.len() {
            return Err("sizes must not contain duplicates".into());
        }
        let unique_ef_search: HashSet<_> = self.ef_search.iter().copied().collect();
        if unique_ef_search.len() != self.ef_search.len() {
            return Err("ef-search must not contain duplicates".into());
        }
        if let Some(path) = &self.corpus_slab
            && !path.is_file()
        {
            return Err(format!("requested corpus slab does not exist: {}", path.display()).into());
        }
        if let Some(path) = &self.corpus_manifest
            && !path.is_file()
        {
            return Err(format!(
                "requested corpus manifest does not exist: {}",
                path.display()
            )
            .into());
        }
        if let Some(path) = &self.corpus_source_manifest
            && !path.is_file()
        {
            return Err(format!(
                "requested corpus source manifest does not exist: {}",
                path.display()
            )
            .into());
        }
        if self.corpus_source_manifest.is_some() && self.corpus_manifest.is_none() {
            return Err("--corpus-source-manifest requires --corpus-manifest".into());
        }
        if let Some(path) = &self.output
            && path.exists()
        {
            return Err(format!(
                "refusing to overwrite existing evidence artifact: {}",
                path.display()
            )
            .into());
        }
        if self.profile == "full" {
            if self.corpus_slab.is_none() {
                return Err(
                    "--full is a high-scale evidence run and requires --corpus-slab; use --smoke for synthetic diagnostics"
                        .into(),
                );
            }
            if self.corpus_manifest.is_none() {
                return Err("--full requires --corpus-manifest identity/provenance".into());
            }
            if self.corpus_source_manifest.is_none() {
                return Err(
                    "--full requires --corpus-source-manifest with verified source/model inputs"
                        .into(),
                );
            }
            if self.output.is_none() {
                return Err("--full requires a durable, create-new --output artifact path".into());
            }
            if cfg!(debug_assertions) {
                return Err(
                    "--full must run an optimized build with debug assertions disabled".into(),
                );
            }
            let build = embedded_build_attestation();
            if !build.is_release_perf() {
                return Err(
                    "--full requires an embedded release-perf receipt: build with \
                     --profile release-perf and set Cargo's controlling \
                     CARGO_PROFILE_RELEASE_PERF_{OPT_LEVEL=3,LTO=thin,CODEGEN_UNITS=1} variables"
                        .into(),
                );
            }
            if !self.sizes.contains(&50_000) || !self.sizes.contains(&100_000) {
                return Err("--full must include both 50000- and 100000-vector cells".into());
            }
            if self.holdout_queries < 300
                || self.query_passes < 10
                || self.build_repetitions < 10
                || self.warmup_passes < 2
            {
                return Err(
                    "--full requires at least 300 holdout queries, 10 query passes, 10 build repetitions, and 2 warmups"
                        .into(),
                );
            }
            if !self.ef_search.contains(&PRIMARY_EF_SEARCH) {
                return Err(
                    "--full must include the predeclared primary ef-search=100 cell".into(),
                );
            }
        }
        let child_fields = [
            self.child_engine.is_some(),
            self.child_size.is_some(),
            self.child_repetition.is_some(),
            self.child_slot.is_some(),
        ];
        if child_fields.iter().any(|set| *set) && !child_fields.iter().all(|set| *set) {
            return Err("child mode requires engine, size, repetition, and slot together".into());
        }
        Ok(())
    }

    fn child_args(
        &self,
        engine: EngineKind,
        size: usize,
        repetition: usize,
        slot: usize,
    ) -> Vec<String> {
        let mut args = vec![
            "--smoke".to_owned(),
            "--sizes".to_owned(),
            size.to_string(),
            "--dim".to_owned(),
            self.dimension.to_string(),
            "--clusters".to_owned(),
            self.clusters.to_string(),
            "--noise".to_owned(),
            self.noise.to_string(),
            "--m".to_owned(),
            self.m.to_string(),
            "--ef-construction".to_owned(),
            self.ef_construction.to_string(),
            "--max-layer".to_owned(),
            self.max_layer.to_string(),
            "--ef-search".to_owned(),
            join_usizes(&self.ef_search),
            "--k".to_owned(),
            self.k.to_string(),
            "--holdout-queries".to_owned(),
            self.holdout_queries.to_string(),
            "--query-passes".to_owned(),
            self.query_passes.to_string(),
            "--build-repetitions".to_owned(),
            self.build_repetitions.to_string(),
            "--warmup-passes".to_owned(),
            self.warmup_passes.to_string(),
            "--child-engine".to_owned(),
            engine.label().to_owned(),
            "--child-size".to_owned(),
            size.to_string(),
            "--child-repetition".to_owned(),
            repetition.to_string(),
            "--child-slot".to_owned(),
            slot.to_string(),
        ];
        if let Some(path) = &self.corpus_slab {
            args.push("--corpus-slab".to_owned());
            args.push(path.display().to_string());
        }
        if let Some(path) = &self.corpus_manifest {
            args.push("--corpus-manifest".to_owned());
            args.push(path.display().to_string());
        }
        if let Some(path) = &self.corpus_source_manifest {
            args.push("--corpus-source-manifest".to_owned());
            args.push(path.display().to_string());
        }
        args
    }
}

fn required_value(args: &[String], index: usize) -> Result<&str, DynError> {
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value after {}", args[index]).into())
}

fn parse_usize(value: &str, field: &str) -> Result<usize, DynError> {
    value
        .parse()
        .map_err(|_| format!("invalid --{field} value {value:?}").into())
}

fn parse_usize_list(value: &str, field: &str) -> Result<Vec<usize>, DynError> {
    let parsed: Result<Vec<_>, _> = value.split(',').map(|part| part.parse::<usize>()).collect();
    let values = parsed.map_err(|_| format!("invalid --{field} list {value:?}"))?;
    if values.is_empty() || values.contains(&0) {
        return Err(format!("--{field} must contain non-zero integers").into());
    }
    Ok(values)
}

fn join_usizes(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn print_help() {
    println!(
        "hnsw_patch_ab [--smoke|--full] [--sizes N,N] [--dim N] \
         [--clusters N] [--noise F] [--m N] [--ef-construction N] \
         [--max-layer N] [--ef-search N,N] [--k N] \
         [--holdout-queries N] [--query-passes N] \
         [--build-repetitions N] [--warmup-passes N] \
         [--corpus-slab PATH] [--corpus-manifest PATH] \
         [--corpus-source-manifest PATH] [--output PATH]"
    );
}

enum Graph {
    Baseline(BaselineHnsw<'static, f32, BaselineDistDot>),
    Candidate(CandidateHnsw<'static, f32, CandidateDistDot>),
}

impl Graph {
    fn build(engine: EngineKind, vectors: &[Vec<f32>], config: &Config) -> Self {
        match engine {
            EngineKind::Baseline => {
                let graph = BaselineHnsw::new(
                    config.m,
                    vectors.len(),
                    config.max_layer,
                    config.ef_construction,
                    BaselineDistDot {},
                );
                insert_baseline(&graph, vectors);
                Self::Baseline(graph)
            }
            EngineKind::Candidate => {
                let graph = CandidateHnsw::new(
                    config.m,
                    vectors.len(),
                    config.max_layer,
                    config.ef_construction,
                    CandidateDistDot {},
                );
                insert_candidate(&graph, vectors);
                Self::Candidate(graph)
            }
        }
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<usize> {
        match self {
            Self::Baseline(graph) => graph
                .search(query, k, ef)
                .into_iter()
                .map(|hit| hit.d_id)
                .collect(),
            Self::Candidate(graph) => graph
                .search(query, k, ef)
                .into_iter()
                .map(|hit| hit.d_id)
                .collect(),
        }
    }

    fn topology(&self, expected_points: usize) -> TopologyMetrics {
        macro_rules! inspect {
            ($graph:expr, $actual_entry:expr, $actual_entry_api_available:expr) => {{
                let mut point_index = HashMap::new();
                let mut internal_point_ids = HashSet::new();
                let mut directed_adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();
                let mut weak_adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();
                let mut unique_origins = HashSet::new();
                let mut points_seen = 0;
                let mut duplicate_origin_ids = 0;
                let mut out_of_range_origins = 0;
                let mut duplicate_internal_point_ids = 0;
                let mut negative_internal_slots = 0;
                let mut truncated_neighborhood_tables = 0;
                let mut invalid_above_level_edges = 0;
                let mut invalid_target_level_edges = 0;
                let mut out_of_range_edges = 0;
                let mut missing_target_edges = 0;
                let mut mismatched_target_point_ids = 0;
                let mut non_finite_edge_distances = 0;
                let mut duplicate_edges = 0;
                let mut self_edges = 0;
                let mut layer0_zero_degree_nodes = 0;
                let mut directed_layer0_edges = 0;

                for point in $graph.get_point_indexation().into_iter() {
                    points_seen += 1;
                    let origin = point.get_origin_id();
                    let point_id = point.get_point_id();
                    let level = usize::from(point_id.0);
                    point_index.insert(origin, (level, point_id));
                    if !unique_origins.insert(origin) {
                        duplicate_origin_ids += 1;
                    }
                    if origin >= expected_points {
                        out_of_range_origins += 1;
                    }
                    if point_id.1 < 0 {
                        negative_internal_slots += 1;
                    }
                    if !internal_point_ids.insert((point_id.0, point_id.1)) {
                        duplicate_internal_point_ids += 1;
                    }
                }

                for point in $graph.get_point_indexation().into_iter() {
                    let origin = point.get_origin_id();
                    let level = usize::from(point.get_point_id().0);
                    let neighborhoods = point.get_neighborhood_id();
                    if level >= neighborhoods.len() {
                        truncated_neighborhood_tables += 1;
                    }
                    let layer0 = neighborhoods.first().map_or(&[][..], Vec::as_slice);
                    if layer0.is_empty() {
                        layer0_zero_degree_nodes += 1;
                    }
                    directed_layer0_edges += layer0.len();
                    let mut seen = HashSet::new();
                    for (layer, neighbors) in neighborhoods.iter().enumerate() {
                        for neighbor in neighbors {
                            if layer > level {
                                invalid_above_level_edges += 1;
                            }
                            if neighbor.d_id >= expected_points {
                                out_of_range_edges += 1;
                            }
                            match point_index.get(&neighbor.d_id) {
                                Some((target_level, target_point_id)) => {
                                    if layer > *target_level {
                                        invalid_target_level_edges += 1;
                                    }
                                    if neighbor.p_id != *target_point_id {
                                        mismatched_target_point_ids += 1;
                                    }
                                }
                                None => missing_target_edges += 1,
                            }
                            if !neighbor.distance.is_finite() {
                                non_finite_edge_distances += 1;
                            }
                            if neighbor.d_id == origin {
                                self_edges += 1;
                            }
                            if !seen.insert((layer, neighbor.d_id)) {
                                duplicate_edges += 1;
                            }
                            if layer == 0 && point_index.contains_key(&neighbor.d_id) {
                                directed_adjacency
                                    .entry(origin)
                                    .or_default()
                                    .insert(neighbor.d_id);
                                weak_adjacency
                                    .entry(origin)
                                    .or_default()
                                    .insert(neighbor.d_id);
                                weak_adjacency
                                    .entry(neighbor.d_id)
                                    .or_default()
                                    .insert(origin);
                            }
                        }
                    }
                }
                for origin in &unique_origins {
                    directed_adjacency.entry(*origin).or_default();
                    weak_adjacency.entry(*origin).or_default();
                }
                let (component_count, largest_component) =
                    weak_components(&weak_adjacency, &unique_origins);
                let reciprocal_layer0_edges = directed_adjacency
                    .iter()
                    .map(|(origin, neighbors)| {
                        neighbors
                            .iter()
                            .filter(|neighbor| {
                                directed_adjacency
                                    .get(neighbor)
                                    .is_some_and(|reverse| reverse.contains(origin))
                            })
                            .count()
                    })
                    .sum();
                let max_level = point_index
                    .values()
                    .map(|(level, _)| *level)
                    .max()
                    .unwrap_or_default();
                let max_level_origins: Vec<_> = point_index
                    .iter()
                    .filter_map(|(origin, (level, _))| (*level == max_level).then_some(*origin))
                    .collect();
                let (minimum_reachable_from_max_level, maximum_reachable_from_max_level) =
                    directed_reachability_range(&directed_adjacency, &max_level_origins);
                let actual_entry: Option<(usize, usize, i32)> = $actual_entry;
                let actual_entry_identity_valid =
                    actual_entry.map(|(origin, entry_level, entry_slot)| {
                        point_index.get(&origin).is_some_and(|(level, point_id)| {
                            *level == entry_level && point_id.1 == entry_slot
                        })
                    });
                let actual_entry_is_max_level =
                    actual_entry.map(|(_, entry_level, _)| entry_level == max_level);
                let reachable_from_actual_entry = actual_entry.map(|(origin, _, _)| {
                    directed_reachability_range(&directed_adjacency, &[origin]).1
                });
                TopologyMetrics {
                    points_seen,
                    unique_origin_ids: unique_origins.len(),
                    duplicate_origin_ids,
                    out_of_range_origin_ids: out_of_range_origins,
                    duplicate_internal_point_ids,
                    negative_internal_slots,
                    truncated_neighborhood_tables,
                    invalid_above_level_edges,
                    invalid_target_level_edges,
                    out_of_range_edges,
                    missing_target_edges,
                    mismatched_target_point_ids,
                    non_finite_edge_distances,
                    duplicate_edges,
                    self_edges,
                    layer0_zero_degree_nodes,
                    directed_layer0_edges,
                    reciprocal_layer0_edges,
                    weak_component_count: component_count,
                    largest_weak_component: largest_component,
                    max_level,
                    max_level_nodes: max_level_origins.len(),
                    minimum_reachable_from_max_level,
                    maximum_reachable_from_max_level,
                    actual_entry_api_available: $actual_entry_api_available,
                    actual_entry_origin: actual_entry.map(|(origin, _, _)| origin),
                    actual_entry_identity_valid,
                    actual_entry_is_max_level,
                    reachable_from_actual_entry,
                }
            }};
        }
        match self {
            Self::Baseline(graph) => inspect!(graph, None, false),
            Self::Candidate(graph) => inspect!(
                graph,
                graph.get_entry_point_id().map(|(origin, point_id)| (
                    origin,
                    usize::from(point_id.0),
                    point_id.1
                )),
                true
            ),
        }
    }

    fn dump_bytes(&self) -> Result<ArtifactMetrics, DynError> {
        let directory = tempfile::Builder::new()
            .prefix("hnsw-patch-ab-")
            .tempdir()?;
        let base = match self {
            Self::Baseline(graph) => BaselineAnnT::file_dump(graph, directory.path(), "index")?,
            Self::Candidate(graph) => CandidateAnnT::file_dump(graph, directory.path(), "index")?,
        };
        let graph_path = directory.path().join(format!("{base}.hnsw.graph"));
        let data_path = directory.path().join(format!("{base}.hnsw.data"));
        let graph_bytes = fs::metadata(&graph_path)?.len();
        let data_bytes = fs::metadata(&data_path)?.len();
        let total_bytes = graph_bytes
            .checked_add(data_bytes)
            .ok_or("HNSW artifact byte-count overflow")?;
        Ok(ArtifactMetrics {
            graph_bytes,
            data_bytes,
            total_bytes,
        })
    }
}

fn insert_baseline(graph: &BaselineHnsw<'_, f32, BaselineDistDot>, vectors: &[Vec<f32>]) {
    if let Some(first) = vectors.first() {
        graph.insert((first.as_slice(), 0));
    }
    let remaining: Vec<_> = vectors
        .iter()
        .enumerate()
        .skip(1)
        .map(|(id, vector)| (vector, id))
        .collect();
    graph.parallel_insert(&remaining);
}

fn insert_candidate(graph: &CandidateHnsw<'_, f32, CandidateDistDot>, vectors: &[Vec<f32>]) {
    if let Some(first) = vectors.first() {
        graph.insert((first.as_slice(), 0));
    }
    let remaining: Vec<_> = vectors
        .iter()
        .enumerate()
        .skip(1)
        .map(|(id, vector)| (vector, id))
        .collect();
    graph.parallel_insert(&remaining);
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TopologyMetrics {
    points_seen: usize,
    unique_origin_ids: usize,
    duplicate_origin_ids: usize,
    out_of_range_origin_ids: usize,
    duplicate_internal_point_ids: usize,
    negative_internal_slots: usize,
    truncated_neighborhood_tables: usize,
    invalid_above_level_edges: usize,
    invalid_target_level_edges: usize,
    out_of_range_edges: usize,
    missing_target_edges: usize,
    mismatched_target_point_ids: usize,
    non_finite_edge_distances: usize,
    duplicate_edges: usize,
    self_edges: usize,
    layer0_zero_degree_nodes: usize,
    directed_layer0_edges: usize,
    reciprocal_layer0_edges: usize,
    weak_component_count: usize,
    largest_weak_component: usize,
    max_level: usize,
    max_level_nodes: usize,
    minimum_reachable_from_max_level: usize,
    maximum_reachable_from_max_level: usize,
    actual_entry_api_available: bool,
    actual_entry_origin: Option<usize>,
    actual_entry_identity_valid: Option<bool>,
    actual_entry_is_max_level: Option<bool>,
    reachable_from_actual_entry: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[allow(
    clippy::struct_field_names,
    reason = "serialized evidence field names retain explicit byte units"
)]
struct ArtifactMetrics {
    graph_bytes: u64,
    data_bytes: u64,
    total_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BuildSample {
    engine: EngineKind,
    corpus_size: usize,
    corpus_sha256: String,
    repetition: usize,
    abba_slot: usize,
    elapsed_ns: u128,
    rss_before_kib: Option<u64>,
    rss_after_kib: Option<u64>,
    rss_delta_kib: Option<i128>,
    peak_rss_kib: Option<u64>,
    topology: TopologyMetrics,
    artifact: ArtifactMetrics,
    query_observations: Vec<BuildQueryObservation>,
    executable_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BuildQueryObservation {
    ef_search: usize,
    hits: Vec<Vec<usize>>,
    nondeterministic_repeat_queries: usize,
}

#[derive(Debug, Serialize)]
struct Distribution {
    count: usize,
    min_ns: u128,
    p50_ns: u128,
    p95_ns: Option<u128>,
    p99_ns: Option<u128>,
    max_ns: u128,
    samples_ns: Vec<u128>,
}

#[derive(Debug, Serialize)]
struct PairedRatio {
    count: usize,
    geometric_mean: f64,
    median: f64,
    bootstrap_median_95_low: f64,
    bootstrap_median_95_high: f64,
}

#[derive(Debug, Serialize)]
struct PairedDelta {
    count: usize,
    mean: f64,
    median: f64,
    bootstrap_95_low: f64,
    bootstrap_95_high: f64,
}

#[derive(Debug, Serialize)]
struct QualityMetrics {
    query_count: usize,
    mean_recall_at_k: f64,
    median_recall_at_k: f64,
    p10_recall_at_k: f64,
    worst_recall_at_k: f64,
    underfilled_queries: usize,
    overfilled_queries: usize,
    duplicate_id_queries: usize,
    out_of_range_id_queries: usize,
    nondeterministic_repeat_queries: usize,
    recall_at_k: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct QueryCell {
    corpus_size: usize,
    ef_search: usize,
    baseline_latency: Distribution,
    candidate_latency: Distribution,
    baseline_aa_latency: Distribution,
    candidate_aa_latency: Distribution,
    candidate_over_baseline_p50: f64,
    baseline_aa_null_ratio_p50: f64,
    candidate_aa_null_ratio_p50: f64,
    paired_candidate_over_baseline: PairedRatio,
    paired_baseline_aa_null: PairedRatio,
    paired_candidate_aa_null: PairedRatio,
    paired_candidate_minus_baseline_recall: PairedDelta,
    baseline_quality: QualityMetrics,
    candidate_quality: QualityMetrics,
    baseline_topology: TopologyMetrics,
    candidate_topology: TopologyMetrics,
}

#[derive(Debug, Serialize)]
struct ReplicatedQualityCell {
    corpus_size: usize,
    ef_search: usize,
    baseline: ReplicatedQualityMetrics,
    candidate: ReplicatedQualityMetrics,
    candidate_minus_baseline: ReplicatedQualityDelta,
}

#[derive(Debug, Serialize)]
struct ReplicatedQualityMetrics {
    graph_count: usize,
    queries_per_graph: usize,
    mean_recall_at_k: f64,
    median_graph_mean_recall_at_k: f64,
    hierarchical_mean_95_low: f64,
    hierarchical_mean_95_high: f64,
    distribution_free_mean_95_low: f64,
    worst_recall_at_k: f64,
    underfilled_queries: usize,
    overfilled_queries: usize,
    duplicate_id_queries: usize,
    out_of_range_id_queries: usize,
    nondeterministic_repeat_queries: usize,
    per_graph_recall_at_k: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize)]
struct ReplicatedQualityDelta {
    mean: f64,
    hierarchical_mean_95_low: f64,
    hierarchical_mean_95_high: f64,
}

#[derive(Debug, Serialize)]
struct SizeResult {
    corpus_size: usize,
    corpus_sha256: String,
    build_samples: Vec<BuildSample>,
    build_summary: BuildSummary,
    query_cells: Vec<QueryCell>,
    replicated_quality_cells: Vec<ReplicatedQualityCell>,
}

#[derive(Debug, Serialize)]
struct BuildSummary {
    baseline_latency: Distribution,
    candidate_latency: Distribution,
    baseline_aa_latency: Distribution,
    candidate_aa_latency: Distribution,
    candidate_over_baseline_p50: f64,
    baseline_aa_null_ratio_p50: f64,
    candidate_aa_null_ratio_p50: f64,
    paired_candidate_over_baseline: PairedRatio,
    paired_baseline_aa_null: PairedRatio,
    paired_candidate_aa_null: PairedRatio,
    baseline_median_peak_rss_kib: Option<u64>,
    candidate_median_peak_rss_kib: Option<u64>,
    candidate_over_baseline_peak_rss: Option<f64>,
    paired_candidate_over_baseline_peak_rss: Option<PairedRatio>,
    baseline_median_artifact_bytes: u64,
    candidate_median_artifact_bytes: u64,
    candidate_over_baseline_artifact_bytes: f64,
    paired_candidate_over_baseline_artifact_bytes: PairedRatio,
}

#[derive(Debug, Serialize)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "provenance booleans are independent attestations, not one encoded state"
)]
struct Provenance {
    generated_unix_ms: u128,
    workspace_git: WorkspaceGitProvenance,
    workspace_source_receipt_start: WorkspaceSourceReceipt,
    workspace_source_receipt_end: WorkspaceSourceReceipt,
    cargo_lock_sha256: String,
    resolved_hnsw_packages: Vec<String>,
    executable_sha256_before: String,
    executable_sha256_after: String,
    executable_stable: bool,
    rustc_version: String,
    runtime_rustc_vv_sha256: String,
    hostname: String,
    kernel: String,
    cpu_model: String,
    rayon_threads: usize,
    baseline_package: String,
    candidate_package: String,
    candidate_reverse_edge_revision: String,
    candidate_logical_layer_revision: String,
    candidate_cumulative_revision: String,
    candidate_revision_note: String,
    candidate_checkout_head: String,
    candidate_reverse_edge_is_ancestor: bool,
    candidate_logical_layer_is_ancestor: bool,
    candidate_checkout_tracked_clean: bool,
    candidate_source_sha256: String,
    baseline_source_sha256: String,
    build: BuildAttestation,
    debug_assertions: bool,
    compiled_target_features: Vec<String>,
    relevant_environment: BTreeMap<String, String>,
    command: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct WorkspaceSourceInput {
    path: String,
    byte_len: u64,
    sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct WorkspaceSourceReceipt {
    schema: String,
    aggregate_sha256: String,
    cargo_lock_sha256: String,
    inputs: Vec<WorkspaceSourceInput>,
}

impl WorkspaceSourceReceipt {
    fn is_well_formed(&self) -> bool {
        self.schema == WORKSPACE_SOURCE_RECEIPT_SCHEMA
            && is_sha256_hex(&self.aggregate_sha256)
            && is_sha256_hex(&self.cargo_lock_sha256)
            && !self.inputs.is_empty()
            && self
                .inputs
                .windows(2)
                .all(|pair| pair[0].path < pair[1].path)
            && self
                .inputs
                .iter()
                .all(|input| safe_relative_source_path(&input.path) && is_sha256_hex(&input.sha256))
            && self
                .inputs
                .iter()
                .any(|input| input.path == "Cargo.lock" && input.sha256 == self.cargo_lock_sha256)
            && workspace_source_aggregate_sha256(&self.inputs)
                .is_some_and(|aggregate| aggregate == self.aggregate_sha256)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "availability", rename_all = "snake_case")]
enum WorkspaceGitProvenance {
    Available {
        top_level: String,
        head: String,
        status_porcelain: String,
        diff_sha256: String,
        untracked_file_sha256: BTreeMap<String, String>,
    },
    Unavailable {
        reason: WorkspaceGitUnavailableReason,
        probe_stderr_sha256: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WorkspaceGitUnavailableReason {
    WorkspaceNotAGitCheckout,
}

#[derive(Clone, Debug, Serialize)]
struct BuildAttestation {
    profile_directory: String,
    profile_family: String,
    opt_level: String,
    debug_info: String,
    lto: String,
    codegen_units: String,
    profile_opt_level: String,
    rustc_vv_sha256: String,
    rustflags_sha256: String,
    host: String,
    target: String,
    target_features: Vec<String>,
    candidate_source_sha256: String,
    baseline_source_sha256: String,
    workspace_source_receipt: WorkspaceSourceReceipt,
}

impl BuildAttestation {
    fn is_release_perf(&self) -> bool {
        self.profile_directory == "release-perf"
            && self.profile_family == "release"
            && self.opt_level == "3"
            && self.profile_opt_level == "3"
            && self.lto == "thin"
            && self.codegen_units == "1"
            && is_sha256_hex(&self.rustc_vv_sha256)
            && is_sha256_hex(&self.rustflags_sha256)
            && is_sha256_hex(&self.candidate_source_sha256)
            && is_sha256_hex(&self.baseline_source_sha256)
            && self.workspace_source_receipt.is_well_formed()
            && !self.host.is_empty()
            && !self.target.is_empty()
    }
}

fn embedded_build_attestation() -> BuildAttestation {
    BuildAttestation {
        profile_directory: env!("FRANKENSEARCH_HNSW_PROFILE_DIR").to_owned(),
        profile_family: env!("FRANKENSEARCH_HNSW_PROFILE_FAMILY").to_owned(),
        opt_level: env!("FRANKENSEARCH_HNSW_OPT_LEVEL").to_owned(),
        debug_info: env!("FRANKENSEARCH_HNSW_DEBUG_INFO").to_owned(),
        lto: env!("FRANKENSEARCH_HNSW_LTO").to_owned(),
        codegen_units: env!("FRANKENSEARCH_HNSW_CODEGEN_UNITS").to_owned(),
        profile_opt_level: env!("FRANKENSEARCH_HNSW_PROFILE_OPT_LEVEL").to_owned(),
        rustc_vv_sha256: env!("FRANKENSEARCH_HNSW_RUSTC_VV_SHA256").to_owned(),
        rustflags_sha256: env!("FRANKENSEARCH_HNSW_RUSTFLAGS_SHA256").to_owned(),
        host: env!("FRANKENSEARCH_HNSW_HOST").to_owned(),
        target: env!("FRANKENSEARCH_HNSW_TARGET").to_owned(),
        target_features: env!("FRANKENSEARCH_HNSW_TARGET_FEATURES")
            .split(',')
            .filter(|feature| !feature.is_empty())
            .map(str::to_owned)
            .collect(),
        candidate_source_sha256: env!("FRANKENSEARCH_HNSW_CANDIDATE_SOURCE_SHA256").to_owned(),
        baseline_source_sha256: env!("FRANKENSEARCH_HNSW_BASELINE_SOURCE_SHA256").to_owned(),
        workspace_source_receipt: WorkspaceSourceReceipt {
            schema: EMBEDDED_WORKSPACE_SOURCE_RECEIPT_SCHEMA.to_owned(),
            aggregate_sha256: EMBEDDED_WORKSPACE_SOURCE_AGGREGATE_SHA256.to_owned(),
            cargo_lock_sha256: EMBEDDED_WORKSPACE_CARGO_LOCK_SHA256.to_owned(),
            inputs: EMBEDDED_WORKSPACE_SOURCE_INPUTS
                .iter()
                .map(|&(path, byte_len, sha256)| WorkspaceSourceInput {
                    path: path.to_owned(),
                    byte_len,
                    sha256: sha256.to_owned(),
                })
                .collect(),
        },
    }
}

#[derive(Debug, Serialize)]
struct Validation {
    expected_size_results: usize,
    observed_size_results: usize,
    expected_query_cells: usize,
    observed_query_cells: usize,
    expected_build_samples: usize,
    observed_build_samples: usize,
    complete: bool,
    correctness_passed: bool,
    measurement_admissible: bool,
    performance_claim_status: PerformanceClaimStatus,
    correctness_violations: Vec<String>,
    performance_blockers: Vec<String>,
    performance_claim_reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    provenance: Provenance,
    corpus_manifest: Option<CorpusSlabManifest>,
    corpus_source_manifest: Option<CorpusSourceManifest>,
    config: Config,
    results: Vec<SizeResult>,
    validation: Validation,
}

#[derive(Debug)]
struct Corpus {
    vectors: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    hash: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusSlabManifest {
    schema: String,
    slab_sha256: String,
    model_id: String,
    model_revision: String,
    corpus_id: String,
    corpus_revision: String,
    source_sha256: String,
    dimension: usize,
    rows: usize,
    dtype: String,
    byte_order: String,
    normalization: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusSourceManifest {
    schema: String,
    model_id: String,
    model_revision: String,
    corpus_id: String,
    corpus_revision: String,
    dimension: usize,
    rows: usize,
    generator_id: String,
    generator_revision: String,
    generator_command: Vec<String>,
    inputs: Vec<CorpusSourceInput>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusSourceInput {
    role: String,
    path: PathBuf,
    sha256: String,
}

fn load_corpus(config: &Config, size: usize) -> Result<Corpus, DynError> {
    config.corpus_slab.as_ref().map_or_else(
        || Ok(synthetic_corpus(config, size)),
        |path| {
            load_slab(
                path,
                config.corpus_manifest.as_deref(),
                config.corpus_source_manifest.as_deref(),
                config.dimension,
                size,
                config.holdout_queries,
            )
        },
    )
}

fn synthetic_corpus(config: &Config, size: usize) -> Corpus {
    let centroids: Vec<_> = (0..config.clusters)
        .map(|cluster| unit_vector(config.dimension, 0xc000_0000_u64 + cluster as u64))
        .collect();
    let vectors: Vec<_> = (0..size)
        .map(|index| {
            noisy_vector(
                &centroids[index % centroids.len()],
                0x1000_0000_u64 + index as u64,
                config.noise,
            )
        })
        .collect();
    let queries: Vec<_> = (0..config.holdout_queries)
        .map(|index| {
            noisy_vector(
                &centroids[index % centroids.len()],
                0x401d_0000_u64 + index as u64,
                config.noise,
            )
        })
        .collect();
    let hash = hash_vectors(&vectors, &queries);
    Corpus {
        vectors,
        queries,
        hash,
    }
}

fn load_slab(
    path: &Path,
    manifest_path: Option<&Path>,
    source_manifest_path: Option<&Path>,
    dimension: usize,
    size: usize,
    holdout: usize,
) -> Result<Corpus, DynError> {
    let bytes = fs::read(path)?;
    let row_bytes = dimension
        .checked_mul(4)
        .ok_or("corpus slab row-size overflow")?;
    if bytes.len() % row_bytes != 0 {
        return Err(format!(
            "corpus slab {} has {} bytes, not a whole number of {row_bytes}-byte rows",
            path.display(),
            bytes.len()
        )
        .into());
    }
    let total_rows = bytes.len() / row_bytes;
    if let Some(manifest_path) = manifest_path {
        let manifest = read_corpus_manifest(manifest_path)?;
        let source_manifest = source_manifest_path
            .map(read_corpus_source_manifest)
            .transpose()?;
        validate_corpus_manifest(
            &manifest,
            source_manifest.as_ref(),
            source_manifest_path,
            &bytes,
            dimension,
            total_rows,
        )?;
    }
    let required_rows = size
        .checked_add(holdout)
        .ok_or("corpus slab row-count overflow")?;
    if total_rows < required_rows {
        return Err(format!(
            "corpus slab {} has {total_rows} rows; need at least {}",
            path.display(),
            required_rows
        )
        .into());
    }
    let (scalar_bytes, remainder) = bytes.as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    for (index, chunk) in scalar_bytes.iter().enumerate() {
        let value = f32::from_le_bytes(*chunk);
        if !value.is_finite() {
            return Err(format!("non-finite slab value at scalar offset {index}").into());
        }
    }
    let query_start = total_rows - holdout;
    let mut vectors = Vec::with_capacity(size);
    for row in 0..size {
        vectors.push(decode_slab_row(
            &bytes,
            row,
            dimension,
            manifest_path.is_some(),
        )?);
    }
    let mut queries = Vec::with_capacity(holdout);
    for row in query_start..total_rows {
        queries.push(decode_slab_row(
            &bytes,
            row,
            dimension,
            manifest_path.is_some(),
        )?);
    }
    let hash = hash_vectors(&vectors, &queries);
    Ok(Corpus {
        vectors,
        queries,
        hash,
    })
}

fn read_corpus_manifest(path: &Path) -> Result<CorpusSlabManifest, DynError> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn read_corpus_source_manifest(path: &Path) -> Result<CorpusSourceManifest, DynError> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn validate_corpus_manifest(
    manifest: &CorpusSlabManifest,
    source_manifest: Option<&CorpusSourceManifest>,
    source_manifest_path: Option<&Path>,
    slab: &[u8],
    dimension: usize,
    rows: usize,
) -> Result<(), DynError> {
    if manifest.schema != "frankensearch-index.hnsw-corpus-manifest.v1"
        || manifest.dimension != dimension
        || manifest.rows != rows
        || manifest.dtype != "f32"
        || manifest.byte_order != "little"
        || manifest.normalization != "l2_unit"
        || manifest.slab_sha256 != sha256_bytes(slab)
        || !is_sha256_hex(&manifest.source_sha256)
        || [
            manifest.model_id.as_str(),
            manifest.model_revision.as_str(),
            manifest.corpus_id.as_str(),
            manifest.corpus_revision.as_str(),
        ]
        .iter()
        .any(|value| value.trim().is_empty())
    {
        return Err(format!("corpus manifest identity or digest mismatch: {manifest:?}").into());
    }
    match (source_manifest, source_manifest_path) {
        (Some(source_manifest), Some(source_manifest_path)) => {
            if manifest.source_sha256 != sha256_file(source_manifest_path)? {
                return Err(
                    "corpus manifest source_sha256 does not bind the source manifest bytes".into(),
                );
            }
            validate_corpus_source_manifest(source_manifest, source_manifest_path, manifest)?;
        }
        (None, None) => {}
        _ => return Err("corpus source manifest value/path mismatch".into()),
    }
    Ok(())
}

fn validate_corpus_source_manifest(
    source: &CorpusSourceManifest,
    source_path: &Path,
    slab: &CorpusSlabManifest,
) -> Result<(), DynError> {
    if source.schema != "frankensearch-index.hnsw-corpus-source-manifest.v1"
        || source.model_id != slab.model_id
        || source.model_revision != slab.model_revision
        || source.corpus_id != slab.corpus_id
        || source.corpus_revision != slab.corpus_revision
        || source.dimension != slab.dimension
        || source.rows != slab.rows
        || source.generator_id.trim().is_empty()
        || source.generator_revision.trim().is_empty()
        || source.generator_command.is_empty()
        || source
            .generator_command
            .iter()
            .any(|argument| argument.trim().is_empty())
        || source.inputs.is_empty()
    {
        return Err(format!("corpus source manifest identity mismatch: {source:?}").into());
    }
    let manifest_dir = source_path.parent().unwrap_or_else(|| Path::new("."));
    let mut roles = BTreeSet::new();
    let mut resolved = BTreeSet::new();
    for input in &source.inputs {
        if input.role.trim().is_empty() || !is_sha256_hex(&input.sha256) {
            return Err(format!("invalid corpus source input: {input:?}").into());
        }
        if !roles.insert(input.role.as_str()) {
            return Err(format!("duplicate corpus source input role {:?}", input.role).into());
        }
        let path = if input.path.is_absolute() {
            input.path.clone()
        } else {
            manifest_dir.join(&input.path)
        };
        let canonical = path.canonicalize().map_err(|error| {
            format!(
                "corpus source input {} cannot be resolved: {error}",
                path.display()
            )
        })?;
        if !canonical.is_file() || !resolved.insert(canonical.clone()) {
            return Err(format!(
                "corpus source input is not a distinct regular file: {}",
                canonical.display()
            )
            .into());
        }
        if sha256_file(&canonical)? != input.sha256 {
            return Err(format!(
                "corpus source input digest mismatch: {}",
                canonical.display()
            )
            .into());
        }
    }
    if !roles.contains("corpus_source") || !roles.contains("embedding_model") {
        return Err(
            "corpus source manifest must bind corpus_source and embedding_model inputs".into(),
        );
    }
    Ok(())
}

fn decode_slab_row(
    bytes: &[u8],
    row: usize,
    dimension: usize,
    require_unit_norm: bool,
) -> Result<Vec<f32>, DynError> {
    let mut vector = Vec::with_capacity(dimension);
    for column in 0..dimension {
        let offset = (row * dimension + column) * 4;
        vector.push(f32::from_le_bytes(bytes[offset..offset + 4].try_into()?));
    }
    if require_unit_norm {
        let norm_squared: f64 = vector
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum();
        if (norm_squared - 1.0).abs() > 2.0e-3 {
            return Err(format!(
                "manifest declares l2_unit, but slab row {row} has squared norm {norm_squared}"
            )
            .into());
        }
    }
    normalize(&mut vector).map_err(|error| -> DynError {
        format!("invalid corpus slab row {row}: {error}").into()
    })?;
    Ok(vector)
}

fn unit_vector(dimension: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut vector = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        state = xorshift64(state);
        let unit = (state >> 40) as f32 / ((1_u32 << 24) - 1) as f32;
        vector.push(unit.mul_add(2.0, -1.0));
    }
    normalize(&mut vector).expect("random vector is non-zero");
    vector
}

fn noisy_vector(centroid: &[f32], seed: u64, noise: f32) -> Vec<f32> {
    let perturbation = unit_vector(centroid.len(), seed);
    let mut vector: Vec<_> = centroid
        .iter()
        .zip(perturbation)
        .map(|(&base, delta)| noise.mul_add(delta, base))
        .collect();
    normalize(&mut vector).expect("centroid-plus-noise vector is non-zero");
    vector
}

const fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^ (state << 17)
}

fn normalize(vector: &mut [f32]) -> Result<(), DynError> {
    let norm_squared: f64 = vector
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum();
    if !norm_squared.is_finite() || norm_squared <= f64::EPSILON {
        return Err("cannot normalize a zero or non-finite vector".into());
    }
    let inverse_f64 = norm_squared.sqrt().recip();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "finite positive normalization scale is intentionally rounded to f32 vectors"
    )]
    let inverse = inverse_f64 as f32;
    for value in vector {
        *value *= inverse;
    }
    Ok(())
}

fn hash_vectors(vectors: &[Vec<f32>], queries: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch-index.hnsw-patch-ab.corpus.v1");
    for (label, rows) in [(b"vectors".as_slice(), vectors), (b"queries", queries)] {
        hasher.update(label);
        hasher.update(u64::try_from(rows.len()).unwrap_or(u64::MAX).to_le_bytes());
        for vector in rows {
            hasher.update(
                u64::try_from(vector.len())
                    .unwrap_or(u64::MAX)
                    .to_le_bytes(),
            );
            for value in vector {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
    }
    hex_bytes(&hasher.finalize())
}

fn child_main(config: &Config) -> Result<(), DynError> {
    let engine = config.child_engine.ok_or("missing child engine")?;
    let size = config.child_size.ok_or("missing child size")?;
    let repetition = config.child_repetition.ok_or("missing child repetition")?;
    let slot = config.child_slot.ok_or("missing child slot")?;
    structured_log(
        "child_build_start",
        &format!(
            "\"engine\":\"{}\",\"size\":{size},\"repetition\":{repetition},\"slot\":{slot}",
            engine.label()
        ),
    );
    let corpus = load_corpus(config, size)?;
    let rss_before = proc_status_kib("VmRSS:");
    let started = Instant::now();
    let graph = Graph::build(engine, &corpus.vectors, config);
    let elapsed = started.elapsed();
    let rss_after = proc_status_kib("VmRSS:");
    let peak_rss = proc_status_kib("VmHWM:");
    let topology = graph.topology(size);
    let artifact = graph.dump_bytes()?;
    let query_observations = observe_build_queries(&graph, &corpus.queries, config);
    let sample = BuildSample {
        engine,
        corpus_size: size,
        corpus_sha256: corpus.hash,
        repetition,
        abba_slot: slot,
        elapsed_ns: elapsed.as_nanos(),
        rss_before_kib: rss_before,
        rss_after_kib: rss_after,
        rss_delta_kib: rss_before
            .zip(rss_after)
            .map(|(before, after)| i128::from(after) - i128::from(before)),
        peak_rss_kib: peak_rss,
        topology,
        artifact,
        query_observations,
        executable_sha256: sha256_file(&env::current_exe()?)?,
    };
    validate_build_sample(&sample)?;
    println!("{CHILD_SENTINEL}{}", serde_json::to_string(&sample)?);
    structured_log(
        "child_build_complete",
        &format!(
            "\"engine\":\"{}\",\"size\":{size},\"elapsed_ns\":{}",
            engine.label(),
            elapsed.as_nanos()
        ),
    );
    Ok(())
}

fn observe_build_queries(
    graph: &Graph,
    queries: &[Vec<f32>],
    config: &Config,
) -> Vec<BuildQueryObservation> {
    config
        .ef_search
        .iter()
        .map(|&ef_search| {
            let mut hits = Vec::with_capacity(queries.len());
            let mut nondeterministic_repeat_queries = 0;
            for query in queries {
                let first = graph.search(query, config.k, ef_search);
                let second = graph.search(query, config.k, ef_search);
                if first != second {
                    nondeterministic_repeat_queries += 1;
                }
                hits.push(first);
            }
            BuildQueryObservation {
                ef_search,
                hits,
                nondeterministic_repeat_queries,
            }
        })
        .collect()
}

fn parent_main(config: Config) -> Result<(), DynError> {
    let executable = env::current_exe()?;
    let executable_sha256_before = sha256_file(&executable)?;
    let workspace = workspace_root()?;
    let workspace_source_receipt_start = workspace_source_receipt(&workspace)?;
    let build_source_receipt = embedded_build_attestation().workspace_source_receipt;
    if !workspace_source_binding_is_complete(
        &build_source_receipt,
        &workspace_source_receipt_start,
        &workspace_source_receipt_start,
        &workspace_source_receipt_start.cargo_lock_sha256,
    ) {
        return Err(
            "workspace source/lock state no longer matches the content-addressed build receipt"
                .into(),
        );
    }
    let mut results = Vec::with_capacity(config.sizes.len());
    for &size in &config.sizes {
        structured_log("size_start", &format!("\"size\":{size}"));
        let corpus = load_corpus(&config, size)?;
        let exact = exact_neighbors(&corpus.vectors, &corpus.queries, config.k);
        let build_samples = collect_build_samples(
            &executable,
            &config,
            size,
            &corpus.hash,
            &executable_sha256_before,
        )?;
        let build_summary = summarize_build_samples(&build_samples)?;
        let replicated_quality_cells =
            replicated_quality_cells(size, &build_samples, &exact, &config)?;
        let baseline = Graph::build(EngineKind::Baseline, &corpus.vectors, &config);
        let candidate = Graph::build(EngineKind::Candidate, &corpus.vectors, &config);
        warm_up(&baseline, &candidate, &corpus.queries, &config);
        let baseline_topology = baseline.topology(size);
        let candidate_topology = candidate.topology(size);
        let mut query_cells = Vec::with_capacity(config.ef_search.len());
        for &ef in &config.ef_search {
            let cell = measure_query_cell(
                size,
                ef,
                &baseline,
                &candidate,
                &corpus.queries,
                &exact,
                &baseline_topology,
                &candidate_topology,
                &config,
            )?;
            structured_log(
                "query_cell_complete",
                &format!(
                    "\"size\":{size},\"ef_search\":{ef},\"candidate_over_baseline_p50\":{}",
                    cell.candidate_over_baseline_p50
                ),
            );
            query_cells.push(cell);
        }
        results.push(SizeResult {
            corpus_size: size,
            corpus_sha256: corpus.hash,
            build_samples,
            build_summary,
            query_cells,
            replicated_quality_cells,
        });
    }
    let provenance = provenance(
        &executable,
        &executable_sha256_before,
        workspace_source_receipt_start,
    )?;
    let validation = validate_report(&config, &results, &provenance)?;
    let corpus_manifest = config
        .corpus_manifest
        .as_deref()
        .map(read_corpus_manifest)
        .transpose()?;
    let corpus_source_manifest = config
        .corpus_source_manifest
        .as_deref()
        .map(read_corpus_source_manifest)
        .transpose()?;
    let report = Report {
        schema: SCHEMA,
        provenance,
        corpus_manifest,
        corpus_source_manifest,
        config,
        results,
        validation,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &report.config.output {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        writeln!(file, "{json}")?;
        file.sync_all()?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::File::open(parent)?.sync_all()?;
        }
        structured_log(
            "report_written",
            &format!(
                "\"path\":{}",
                serde_json::to_string(&path.display().to_string())?
            ),
        );
    } else {
        println!("{json}");
    }
    structured_log(
        "report_validated",
        &format!(
            "\"correctness_passed\":{},\"measurement_admissible\":{},\
             \"performance_claim_status\":\"{}\"",
            report.validation.correctness_passed,
            report.validation.measurement_admissible,
            report.validation.performance_claim_status.label()
        ),
    );
    if !report.validation.correctness_passed {
        return Err(format!(
            "candidate correctness gate failed: {}",
            report.validation.correctness_violations.join("; ")
        )
        .into());
    }
    if matches!(
        report.validation.performance_claim_status,
        PerformanceClaimStatus::Block
    ) {
        return Err(format!(
            "valid performance regression blocks admission: {}",
            report.validation.performance_claim_reasons.join("; ")
        )
        .into());
    }
    if report.config.profile == "full"
        && matches!(
            report.validation.performance_claim_status,
            PerformanceClaimStatus::Quarantine
        )
    {
        return Err(format!(
            "full measurement was quarantined: blockers=[{}]; classification=[{}]",
            report.validation.performance_blockers.join("; "),
            report.validation.performance_claim_reasons.join("; ")
        )
        .into());
    }
    Ok(())
}

fn collect_build_samples(
    executable: &Path,
    config: &Config,
    size: usize,
    expected_corpus_sha256: &str,
    expected_executable_sha256: &str,
) -> Result<Vec<BuildSample>, DynError> {
    let mut samples = Vec::with_capacity(config.build_repetitions * 4);
    for repetition in 0..config.build_repetitions {
        let schedule = if repetition % 2 == 0 {
            [
                EngineKind::Baseline,
                EngineKind::Candidate,
                EngineKind::Candidate,
                EngineKind::Baseline,
            ]
        } else {
            [
                EngineKind::Candidate,
                EngineKind::Baseline,
                EngineKind::Baseline,
                EngineKind::Candidate,
            ]
        };
        for (slot, engine) in schedule.into_iter().enumerate() {
            let output = Command::new(executable)
                .args(config.child_args(engine, size, repetition, slot))
                .stdin(Stdio::null())
                .output()?;
            if !output.status.success() {
                return Err(format!(
                    "{} child failed for size {size}, repetition {repetition}, slot {slot}: {}",
                    engine.label(),
                    String::from_utf8_lossy(&output.stderr)
                )
                .into());
            }
            let stdout = String::from_utf8(output.stdout)?;
            let payloads: Vec<_> = stdout
                .lines()
                .filter_map(|line| line.strip_prefix(CHILD_SENTINEL))
                .collect();
            if payloads.len() != 1 {
                return Err(format!(
                    "{} child emitted {} result payloads; expected exactly one",
                    engine.label(),
                    payloads.len()
                )
                .into());
            }
            let sample: BuildSample = serde_json::from_str(payloads[0])?;
            if sample.engine != engine
                || sample.corpus_size != size
                || sample.corpus_sha256 != expected_corpus_sha256
                || sample.executable_sha256 != expected_executable_sha256
                || sample.repetition != repetition
                || sample.abba_slot != slot
            {
                return Err("child result identity does not match requested cell".into());
            }
            let observed_ef: BTreeSet<_> = sample
                .query_observations
                .iter()
                .map(|observation| observation.ef_search)
                .collect();
            let expected_ef: BTreeSet<_> = config.ef_search.iter().copied().collect();
            let ef_cells_match = observed_ef == expected_ef;
            let observation_count_matches =
                sample.query_observations.len() == config.ef_search.len();
            let hit_counts_match = sample
                .query_observations
                .iter()
                .all(|observation| observation.hits.len() == config.holdout_queries);
            if !ef_cells_match || !observation_count_matches || !hit_counts_match {
                return Err("child query observations do not match the requested grid".into());
            }
            validate_build_sample(&sample)?;
            samples.push(sample);
        }
    }
    Ok(samples)
}

fn validate_build_sample(sample: &BuildSample) -> Result<(), DynError> {
    if sample.elapsed_ns == 0
        || sample.topology.points_seen != sample.corpus_size
        || sample.topology.unique_origin_ids != sample.corpus_size
        || sample.corpus_sha256.len() != 64
        || sample.artifact.total_bytes == 0
        || sample.executable_sha256.len() != 64
        || sample.query_observations.is_empty()
    {
        return Err(format!("invalid or incomplete build sample: {sample:?}").into());
    }
    Ok(())
}

fn replicated_quality_cells(
    corpus_size: usize,
    build_samples: &[BuildSample],
    exact: &[Vec<usize>],
    config: &Config,
) -> Result<Vec<ReplicatedQualityCell>, DynError> {
    let mut cells = Vec::with_capacity(config.ef_search.len());
    for &ef_search in &config.ef_search {
        let mut baseline_graphs = Vec::new();
        let mut candidate_graphs = Vec::new();
        for sample in build_samples {
            let observation = sample
                .query_observations
                .iter()
                .find(|observation| observation.ef_search == ef_search)
                .ok_or_else(|| {
                    format!(
                        "{} build sample is missing ef-search {ef_search}",
                        sample.engine.label()
                    )
                })?;
            let metrics = quality_metrics(
                &observation.hits,
                exact,
                config.k,
                corpus_size,
                observation.nondeterministic_repeat_queries,
            )?;
            match sample.engine {
                EngineKind::Baseline => baseline_graphs.push(metrics),
                EngineKind::Candidate => candidate_graphs.push(metrics),
            }
        }
        if baseline_graphs.len() < config.build_repetitions * 2
            || candidate_graphs.len() < config.build_repetitions * 2
        {
            return Err(format!(
                "ef-search {ef_search} lacks the two graph replicas per engine and build block"
            )
            .into());
        }
        let seed = 0x51ed_0000_u64 ^ corpus_size as u64 ^ ((ef_search as u64) << 32);
        let baseline = replicated_quality_metrics(&baseline_graphs, seed ^ 0xbace)?;
        let candidate = replicated_quality_metrics(&candidate_graphs, seed ^ 0xcade)?;
        let candidate_minus_baseline = replicated_quality_delta(
            &candidate.per_graph_recall_at_k,
            &baseline.per_graph_recall_at_k,
            seed ^ 0xde17a,
        )?;
        cells.push(ReplicatedQualityCell {
            corpus_size,
            ef_search,
            baseline,
            candidate,
            candidate_minus_baseline,
        });
    }
    Ok(cells)
}

fn replicated_quality_metrics(
    graphs: &[QualityMetrics],
    seed: u64,
) -> Result<ReplicatedQualityMetrics, DynError> {
    if graphs.is_empty() {
        return Err("replicated quality requires at least one graph".into());
    }
    let queries_per_graph = graphs[0].recall_at_k.len();
    if queries_per_graph == 0
        || graphs
            .iter()
            .any(|graph| graph.recall_at_k.len() != queries_per_graph)
    {
        return Err("replicated graphs have empty or unequal query grids".into());
    }
    let per_graph_recall_at_k: Vec<_> = graphs
        .iter()
        .map(|graph| graph.recall_at_k.clone())
        .collect();
    let graph_means: Vec<_> = per_graph_recall_at_k
        .iter()
        .map(|recalls| recalls.iter().sum::<f64>() / recalls.len() as f64)
        .collect();
    let mean_recall_at_k = graph_means.iter().sum::<f64>() / graph_means.len() as f64;
    let (hierarchical_mean_95_low, hierarchical_mean_95_high) =
        hierarchical_mean_ci(&per_graph_recall_at_k, seed)?;
    let per_query_graph_mean: Vec<_> = (0..queries_per_graph)
        .map(|query_index| {
            per_graph_recall_at_k
                .iter()
                .map(|graph| graph[query_index])
                .sum::<f64>()
                / per_graph_recall_at_k.len() as f64
        })
        .collect();
    let distribution_free_mean_95_low =
        mean_recall_lower_bound_bernstein(&per_query_graph_mean, 0.05);
    let worst_recall_at_k = per_graph_recall_at_k
        .iter()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min);
    Ok(ReplicatedQualityMetrics {
        graph_count: graphs.len(),
        queries_per_graph,
        mean_recall_at_k,
        median_graph_mean_recall_at_k: percentile_f64(&graph_means, 1, 2)?,
        hierarchical_mean_95_low,
        hierarchical_mean_95_high,
        distribution_free_mean_95_low,
        worst_recall_at_k,
        underfilled_queries: checked_metric_sum(graphs, |graph| graph.underfilled_queries)?,
        overfilled_queries: checked_metric_sum(graphs, |graph| graph.overfilled_queries)?,
        duplicate_id_queries: checked_metric_sum(graphs, |graph| graph.duplicate_id_queries)?,
        out_of_range_id_queries: checked_metric_sum(graphs, |graph| graph.out_of_range_id_queries)?,
        nondeterministic_repeat_queries: checked_metric_sum(graphs, |graph| {
            graph.nondeterministic_repeat_queries
        })?,
        per_graph_recall_at_k,
    })
}

fn checked_metric_sum(
    graphs: &[QualityMetrics],
    value: impl Fn(&QualityMetrics) -> usize,
) -> Result<usize, DynError> {
    graphs.iter().try_fold(0_usize, |sum, graph| {
        sum.checked_add(value(graph))
            .ok_or_else(|| "replicated quality counter overflow".into())
    })
}

fn replicated_quality_delta(
    candidate: &[Vec<f64>],
    baseline: &[Vec<f64>],
    seed: u64,
) -> Result<ReplicatedQualityDelta, DynError> {
    let candidate_mean = nested_mean(candidate)?;
    let baseline_mean = nested_mean(baseline)?;
    let (hierarchical_mean_95_low, hierarchical_mean_95_high) =
        hierarchical_delta_ci(candidate, baseline, seed)?;
    Ok(ReplicatedQualityDelta {
        mean: candidate_mean - baseline_mean,
        hierarchical_mean_95_low,
        hierarchical_mean_95_high,
    })
}

fn nested_mean(graphs: &[Vec<f64>]) -> Result<f64, DynError> {
    let values: Vec<_> = graphs.iter().flatten().copied().collect();
    if graphs.is_empty() || values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("hierarchical quality inputs must be non-empty and finite".into());
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn hierarchical_mean_ci(graphs: &[Vec<f64>], mut state: u64) -> Result<(f64, f64), DynError> {
    validate_hierarchical_samples(graphs)?;
    const RESAMPLES: usize = 2_000;
    let mut means = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        means.push(hierarchical_resample_mean(graphs, &mut state)?);
    }
    Ok((
        percentile_f64(&means, 1, 40)?,
        percentile_f64(&means, 39, 40)?,
    ))
}

fn hierarchical_delta_ci(
    candidate: &[Vec<f64>],
    baseline: &[Vec<f64>],
    mut state: u64,
) -> Result<(f64, f64), DynError> {
    validate_hierarchical_samples(candidate)?;
    validate_hierarchical_samples(baseline)?;
    if candidate.len() != baseline.len()
        || candidate
            .iter()
            .zip(baseline)
            .any(|(candidate_queries, baseline_queries)| {
                candidate_queries.len() != baseline_queries.len()
            })
    {
        return Err("paired hierarchical samples must have identical graph/query grids".into());
    }
    const RESAMPLES: usize = 2_000;
    let mut deltas = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        deltas.push(hierarchical_paired_delta_mean(
            candidate, baseline, &mut state,
        )?);
    }
    Ok((
        percentile_f64(&deltas, 1, 40)?,
        percentile_f64(&deltas, 39, 40)?,
    ))
}

fn hierarchical_paired_delta_mean(
    candidate: &[Vec<f64>],
    baseline: &[Vec<f64>],
    state: &mut u64,
) -> Result<f64, DynError> {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for _ in 0..candidate.len() {
        *state = xorshift64((*state).max(1));
        let graph_index = usize::try_from(*state % candidate.len() as u64)
            .map_err(|_| "paired graph bootstrap index does not fit usize")?;
        let candidate_queries = &candidate[graph_index];
        let baseline_queries = &baseline[graph_index];
        for _ in 0..candidate_queries.len() {
            *state = xorshift64((*state).max(1));
            let query_index = usize::try_from(*state % candidate_queries.len() as u64)
                .map_err(|_| "paired query bootstrap index does not fit usize")?;
            sum += candidate_queries[query_index] - baseline_queries[query_index];
            count += 1;
        }
    }
    Ok(sum / count as f64)
}

fn validate_hierarchical_samples(graphs: &[Vec<f64>]) -> Result<(), DynError> {
    if graphs.is_empty()
        || graphs
            .iter()
            .any(|queries| queries.is_empty() || queries.iter().any(|recall| !recall.is_finite()))
    {
        return Err("hierarchical samples must contain finite queries for every graph".into());
    }
    Ok(())
}

fn hierarchical_resample_mean(graphs: &[Vec<f64>], state: &mut u64) -> Result<f64, DynError> {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for _ in 0..graphs.len() {
        *state = xorshift64((*state).max(1));
        let graph_index = usize::try_from(*state % graphs.len() as u64)
            .map_err(|_| "graph bootstrap index does not fit usize")?;
        let queries = &graphs[graph_index];
        for _ in 0..queries.len() {
            *state = xorshift64((*state).max(1));
            let query_index = usize::try_from(*state % queries.len() as u64)
                .map_err(|_| "query bootstrap index does not fit usize")?;
            sum += queries[query_index];
            count += 1;
        }
    }
    Ok(sum / count as f64)
}

fn summarize_build_samples(samples: &[BuildSample]) -> Result<BuildSummary, DynError> {
    let mut baseline_primary = Vec::new();
    let mut baseline_null = Vec::new();
    let mut candidate_primary = Vec::new();
    let mut candidate_null = Vec::new();
    let mut baseline_peak_rss = Vec::new();
    let mut candidate_peak_rss = Vec::new();
    let mut baseline_artifacts = Vec::new();
    let mut candidate_artifacts = Vec::new();
    let mut baseline_primary_peak_rss = Vec::new();
    let mut candidate_primary_peak_rss = Vec::new();
    let mut baseline_primary_artifacts = Vec::new();
    let mut candidate_primary_artifacts = Vec::new();
    let mut primary_rss_complete = true;
    let mut by_repetition: BTreeMap<usize, Vec<&BuildSample>> = BTreeMap::new();
    for sample in samples {
        by_repetition
            .entry(sample.repetition)
            .or_default()
            .push(sample);
        match sample.engine {
            EngineKind::Baseline => {
                if let Some(rss) = sample.peak_rss_kib {
                    baseline_peak_rss.push(rss);
                }
                baseline_artifacts.push(sample.artifact.total_bytes);
            }
            EngineKind::Candidate => {
                if let Some(rss) = sample.peak_rss_kib {
                    candidate_peak_rss.push(rss);
                }
                candidate_artifacts.push(sample.artifact.total_bytes);
            }
        }
    }
    for repetition_samples in by_repetition.values_mut() {
        repetition_samples.sort_unstable_by_key(|sample| sample.abba_slot);
        for engine in [EngineKind::Baseline, EngineKind::Candidate] {
            let engine_samples: Vec<_> = repetition_samples
                .iter()
                .filter(|sample| sample.engine == engine)
                .collect();
            if engine_samples.len() != 2 {
                return Err(format!(
                    "repetition did not contain exactly two {} build samples",
                    engine.label()
                )
                .into());
            }
            match engine {
                EngineKind::Baseline => {
                    baseline_primary.push(engine_samples[0].elapsed_ns);
                    baseline_null.push(engine_samples[1].elapsed_ns);
                    if let Some(rss) = engine_samples[0].peak_rss_kib {
                        baseline_primary_peak_rss.push(u128::from(rss));
                    } else {
                        primary_rss_complete = false;
                    }
                    baseline_primary_artifacts
                        .push(u128::from(engine_samples[0].artifact.total_bytes));
                }
                EngineKind::Candidate => {
                    candidate_primary.push(engine_samples[0].elapsed_ns);
                    candidate_null.push(engine_samples[1].elapsed_ns);
                    if let Some(rss) = engine_samples[0].peak_rss_kib {
                        candidate_primary_peak_rss.push(u128::from(rss));
                    } else {
                        primary_rss_complete = false;
                    }
                    candidate_primary_artifacts
                        .push(u128::from(engine_samples[0].artifact.total_bytes));
                }
            }
        }
    }
    let baseline_latency = distribution(&baseline_primary)?;
    let candidate_latency = distribution(&candidate_primary)?;
    let baseline_aa_latency = distribution(&baseline_null)?;
    let candidate_aa_latency = distribution(&candidate_null)?;
    let paired_candidate_over_baseline =
        paired_ratio(&candidate_primary, &baseline_primary, 0xb17d_0001)?;
    let paired_baseline_aa_null = paired_ratio(&baseline_null, &baseline_primary, 0xb17d_0002)?;
    let paired_candidate_aa_null = paired_ratio(&candidate_null, &candidate_primary, 0xb17d_0003)?;
    let baseline_median_peak_rss_kib = median_u64(&baseline_peak_rss);
    let candidate_median_peak_rss_kib = median_u64(&candidate_peak_rss);
    let candidate_over_baseline_peak_rss =
        match (candidate_median_peak_rss_kib, baseline_median_peak_rss_kib) {
            (Some(candidate), Some(baseline)) if baseline != 0 => {
                Some(candidate as f64 / baseline as f64)
            }
            _ => None,
        };
    let paired_candidate_over_baseline_peak_rss = if primary_rss_complete
        && baseline_primary_peak_rss.len() == baseline_primary.len()
        && candidate_primary_peak_rss.len() == candidate_primary.len()
    {
        Some(paired_ratio(
            &candidate_primary_peak_rss,
            &baseline_primary_peak_rss,
            0xb17d_0004,
        )?)
    } else {
        None
    };
    let baseline_median_artifact_bytes =
        median_u64(&baseline_artifacts).ok_or("missing baseline artifact samples")?;
    let candidate_median_artifact_bytes =
        median_u64(&candidate_artifacts).ok_or("missing candidate artifact samples")?;
    if baseline_median_artifact_bytes == 0 {
        return Err("baseline artifact size is zero".into());
    }
    let paired_candidate_over_baseline_artifact_bytes = paired_ratio(
        &candidate_primary_artifacts,
        &baseline_primary_artifacts,
        0xb17d_0005,
    )?;
    Ok(BuildSummary {
        candidate_over_baseline_p50: ratio(candidate_latency.p50_ns, baseline_latency.p50_ns)?,
        baseline_aa_null_ratio_p50: ratio(baseline_aa_latency.p50_ns, baseline_latency.p50_ns)?,
        candidate_aa_null_ratio_p50: ratio(candidate_aa_latency.p50_ns, candidate_latency.p50_ns)?,
        paired_candidate_over_baseline,
        paired_baseline_aa_null,
        paired_candidate_aa_null,
        baseline_latency,
        candidate_latency,
        baseline_aa_latency,
        candidate_aa_latency,
        baseline_median_peak_rss_kib,
        candidate_median_peak_rss_kib,
        candidate_over_baseline_peak_rss,
        paired_candidate_over_baseline_peak_rss,
        baseline_median_artifact_bytes,
        candidate_median_artifact_bytes,
        candidate_over_baseline_artifact_bytes: candidate_median_artifact_bytes as f64
            / baseline_median_artifact_bytes as f64,
        paired_candidate_over_baseline_artifact_bytes,
    })
}

fn median_u64(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    Some(sorted[(sorted.len() - 1) / 2])
}

fn warm_up(baseline: &Graph, candidate: &Graph, queries: &[Vec<f32>], config: &Config) {
    let ef = *config.ef_search.iter().max().expect("validated non-empty");
    for _ in 0..config.warmup_passes {
        for query in queries {
            let _ = baseline.search(query, config.k, ef);
            let _ = candidate.search(query, config.k, ef);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_query_cell(
    size: usize,
    ef: usize,
    baseline: &Graph,
    candidate: &Graph,
    queries: &[Vec<f32>],
    exact: &[Vec<usize>],
    baseline_topology: &TopologyMetrics,
    candidate_topology: &TopologyMetrics,
    config: &Config,
) -> Result<QueryCell, DynError> {
    let mut baseline_primary = Vec::new();
    let mut baseline_null = Vec::new();
    let mut candidate_primary = Vec::new();
    let mut candidate_null = Vec::new();
    let mut baseline_results = Vec::new();
    let mut candidate_results = Vec::new();
    let mut baseline_nondeterministic = 0;
    let mut candidate_nondeterministic = 0;

    for pass in 0..config.query_passes {
        for (query_index, query) in queries.iter().enumerate() {
            let order = if (pass + query_index) % 2 == 0 {
                [
                    EngineKind::Baseline,
                    EngineKind::Candidate,
                    EngineKind::Candidate,
                    EngineKind::Baseline,
                ]
            } else {
                [
                    EngineKind::Candidate,
                    EngineKind::Baseline,
                    EngineKind::Baseline,
                    EngineKind::Candidate,
                ]
            };
            let mut baseline_pair = Vec::with_capacity(2);
            let mut candidate_pair = Vec::with_capacity(2);
            for engine in order {
                let graph = match engine {
                    EngineKind::Baseline => baseline,
                    EngineKind::Candidate => candidate,
                };
                let started = Instant::now();
                let ids = graph.search(query, config.k, ef);
                let elapsed = started.elapsed().as_nanos();
                match engine {
                    EngineKind::Baseline => baseline_pair.push((elapsed, ids)),
                    EngineKind::Candidate => candidate_pair.push((elapsed, ids)),
                }
            }
            let (baseline_first, baseline_second) = pair(&baseline_pair)?;
            let (candidate_first, candidate_second) = pair(&candidate_pair)?;
            baseline_primary.push(baseline_first.0);
            baseline_null.push(baseline_second.0);
            candidate_primary.push(candidate_first.0);
            candidate_null.push(candidate_second.0);
            if baseline_first.1 != baseline_second.1 {
                baseline_nondeterministic += 1;
            }
            if candidate_first.1 != candidate_second.1 {
                candidate_nondeterministic += 1;
            }
            if pass == 0 {
                baseline_results.push(baseline_first.1.clone());
                candidate_results.push(candidate_first.1.clone());
            }
        }
    }

    let baseline_latency = distribution(&baseline_primary)?;
    let candidate_latency = distribution(&candidate_primary)?;
    let baseline_aa_latency = distribution(&baseline_null)?;
    let candidate_aa_latency = distribution(&candidate_null)?;
    let candidate_over_baseline_p50 = ratio(candidate_latency.p50_ns, baseline_latency.p50_ns)?;
    let baseline_aa_null_ratio_p50 = ratio(baseline_aa_latency.p50_ns, baseline_latency.p50_ns)?;
    let candidate_aa_null_ratio_p50 = ratio(candidate_aa_latency.p50_ns, candidate_latency.p50_ns)?;
    let baseline_quality = quality_metrics(
        &baseline_results,
        exact,
        config.k,
        size,
        baseline_nondeterministic,
    )?;
    let candidate_quality = quality_metrics(
        &candidate_results,
        exact,
        config.k,
        size,
        candidate_nondeterministic,
    )?;
    let paired_candidate_minus_baseline_recall = paired_delta(
        &candidate_quality.recall_at_k,
        &baseline_quality.recall_at_k,
        0x9e37_79b9_u64 ^ size as u64 ^ ((ef as u64) << 32),
    )?;
    Ok(QueryCell {
        corpus_size: size,
        ef_search: ef,
        baseline_latency,
        candidate_latency,
        baseline_aa_latency,
        candidate_aa_latency,
        candidate_over_baseline_p50,
        baseline_aa_null_ratio_p50,
        candidate_aa_null_ratio_p50,
        paired_candidate_over_baseline: paired_query_ratio(
            &candidate_primary,
            &baseline_primary,
            config.holdout_queries,
            config.query_passes,
            0x7175_0001_u64 ^ size as u64 ^ ((ef as u64) << 32),
        )?,
        paired_baseline_aa_null: paired_query_ratio(
            &baseline_null,
            &baseline_primary,
            config.holdout_queries,
            config.query_passes,
            0x7175_0002_u64 ^ size as u64 ^ ((ef as u64) << 32),
        )?,
        paired_candidate_aa_null: paired_query_ratio(
            &candidate_null,
            &candidate_primary,
            config.holdout_queries,
            config.query_passes,
            0x7175_0003_u64 ^ size as u64 ^ ((ef as u64) << 32),
        )?,
        paired_candidate_minus_baseline_recall,
        baseline_quality,
        candidate_quality,
        baseline_topology: baseline_topology.clone(),
        candidate_topology: candidate_topology.clone(),
    })
}

fn pair<T>(values: &[T]) -> Result<(&T, &T), DynError> {
    match values {
        [first, second] => Ok((first, second)),
        _ => Err(format!(
            "ABBA schedule produced {} samples, expected two",
            values.len()
        )
        .into()),
    }
}

fn exact_neighbors(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    queries
        .par_iter()
        .map(|query| {
            let mut scores: Vec<_> = vectors
                .iter()
                .enumerate()
                .map(|(id, vector)| {
                    let score: f32 = query
                        .iter()
                        .zip(vector)
                        .map(|(&left, &right)| left * right)
                        .sum();
                    (id, score)
                })
                .collect();
            scores.sort_unstable_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            scores.truncate(k.min(scores.len()));
            scores.into_iter().map(|(id, _)| id).collect()
        })
        .collect()
}

fn quality_metrics(
    actual: &[Vec<usize>],
    exact: &[Vec<usize>],
    k: usize,
    corpus_size: usize,
    nondeterministic: usize,
) -> Result<QualityMetrics, DynError> {
    if actual.len() != exact.len() || actual.is_empty() {
        return Err("quality inputs are empty or have mismatched query counts".into());
    }
    let expected_len = k.min(corpus_size);
    let mut recalls = Vec::with_capacity(actual.len());
    let mut underfilled = 0;
    let mut overfilled = 0;
    let mut duplicate_queries = 0;
    let mut out_of_range_queries = 0;
    for (observed, oracle) in actual.iter().zip(exact) {
        if observed.len() < expected_len {
            underfilled += 1;
        }
        if observed.len() > expected_len {
            overfilled += 1;
        }
        let unique: HashSet<_> = observed.iter().copied().collect();
        if unique.len() != observed.len() {
            duplicate_queries += 1;
        }
        if observed.iter().any(|&id| id >= corpus_size) {
            out_of_range_queries += 1;
        }
        let oracle: HashSet<_> = oracle.iter().copied().collect();
        recalls.push(unique.intersection(&oracle).count() as f64 / expected_len as f64);
    }
    let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
    let median = percentile_f64(&recalls, 1, 2)?;
    let p10 = percentile_f64(&recalls, 1, 10)?;
    let worst = recalls.iter().copied().fold(f64::INFINITY, f64::min);
    Ok(QualityMetrics {
        query_count: actual.len(),
        mean_recall_at_k: mean,
        median_recall_at_k: median,
        p10_recall_at_k: p10,
        worst_recall_at_k: worst,
        underfilled_queries: underfilled,
        overfilled_queries: overfilled,
        duplicate_id_queries: duplicate_queries,
        out_of_range_id_queries: out_of_range_queries,
        nondeterministic_repeat_queries: nondeterministic,
        recall_at_k: recalls,
    })
}

fn distribution(values: &[u128]) -> Result<Distribution, DynError> {
    if values.is_empty() {
        return Err("cannot summarize an empty latency distribution".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let p95_ns = if sorted.len() >= 20 {
        Some(percentile_sorted_u128(&sorted, 19, 20)?)
    } else {
        None
    };
    let p99_ns = if sorted.len() >= 100 {
        Some(percentile_sorted_u128(&sorted, 99, 100)?)
    } else {
        None
    };
    Ok(Distribution {
        count: sorted.len(),
        min_ns: sorted[0],
        p50_ns: percentile_sorted_u128(&sorted, 1, 2)?,
        p95_ns,
        p99_ns,
        max_ns: *sorted.last().expect("validated non-empty"),
        samples_ns: values.to_vec(),
    })
}

fn percentile_index(
    value_count: usize,
    numerator: usize,
    denominator: usize,
) -> Result<usize, DynError> {
    if value_count == 0 || denominator == 0 || numerator > denominator {
        return Err("percentile rank requires values and a fraction in [0, 1]".into());
    }
    let scaled_rank = value_count
        .saturating_sub(1)
        .checked_mul(numerator)
        .ok_or("percentile rank overflow")?;
    Ok(scaled_rank.div_ceil(denominator))
}

fn percentile_sorted_u128(
    sorted: &[u128],
    numerator: usize,
    denominator: usize,
) -> Result<u128, DynError> {
    Ok(sorted[percentile_index(sorted.len(), numerator, denominator)?])
}

fn percentile_f64(values: &[f64], numerator: usize, denominator: usize) -> Result<f64, DynError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("percentile input must be non-empty and finite".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let index = percentile_index(sorted.len(), numerator, denominator)?;
    Ok(sorted[index])
}

fn paired_ratio(
    numerators: &[u128],
    denominators: &[u128],
    seed: u64,
) -> Result<PairedRatio, DynError> {
    if numerators.len() != denominators.len() || numerators.is_empty() {
        return Err("paired ratio inputs must be non-empty and equally sized".into());
    }
    let mut ratios = Vec::with_capacity(numerators.len());
    for (&numerator, &denominator) in numerators.iter().zip(denominators) {
        let ratio = ratio(numerator, denominator)?;
        if ratio <= 0.0 {
            return Err("paired timing ratio must be positive".into());
        }
        ratios.push(ratio);
    }
    paired_ratio_values(&ratios, seed)
}

fn paired_query_ratio(
    numerators: &[u128],
    denominators: &[u128],
    query_count: usize,
    pass_count: usize,
    seed: u64,
) -> Result<PairedRatio, DynError> {
    let expected = query_count
        .checked_mul(pass_count)
        .ok_or("query/pass sample-count overflow")?;
    if numerators.len() != expected || denominators.len() != expected {
        return Err("query timing vectors do not match the configured query/pass grid".into());
    }
    let mut per_query_ratios = Vec::with_capacity(query_count);
    for query in 0..query_count {
        let mut pass_ratios = Vec::with_capacity(pass_count);
        for pass in 0..pass_count {
            let index = pass * query_count + query;
            pass_ratios.push(ratio(numerators[index], denominators[index])?);
        }
        per_query_ratios.push(percentile_f64(&pass_ratios, 1, 2)?);
    }
    paired_ratio_values(&per_query_ratios, seed)
}

fn paired_ratio_values(ratios: &[f64], seed: u64) -> Result<PairedRatio, DynError> {
    if ratios.is_empty()
        || ratios
            .iter()
            .any(|ratio| !ratio.is_finite() || *ratio <= 0.0)
    {
        return Err("paired ratios must be non-empty, finite, and positive".into());
    }
    let mean_log = ratios.iter().map(|ratio| ratio.ln()).sum::<f64>() / ratios.len() as f64;
    let (bootstrap_median_95_low, bootstrap_median_95_high) = bootstrap_median_ci(ratios, seed)?;
    Ok(PairedRatio {
        count: ratios.len(),
        geometric_mean: mean_log.exp(),
        median: percentile_f64(ratios, 1, 2)?,
        bootstrap_median_95_low,
        bootstrap_median_95_high,
    })
}

fn paired_delta(
    numerators: &[f64],
    denominators: &[f64],
    seed: u64,
) -> Result<PairedDelta, DynError> {
    if numerators.len() != denominators.len() || numerators.is_empty() {
        return Err("paired delta inputs must be non-empty and equally sized".into());
    }
    let deltas: Vec<_> = numerators
        .iter()
        .zip(denominators)
        .map(|(&numerator, &denominator)| numerator - denominator)
        .collect();
    if deltas.iter().any(|value| !value.is_finite()) {
        return Err("paired delta inputs must be finite".into());
    }
    let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let (bootstrap_95_low, bootstrap_95_high) = bootstrap_mean_ci(&deltas, seed)?;
    Ok(PairedDelta {
        count: deltas.len(),
        mean,
        median: percentile_f64(&deltas, 1, 2)?,
        bootstrap_95_low,
        bootstrap_95_high,
    })
}

fn bootstrap_mean_ci(samples: &[f64], mut state: u64) -> Result<(f64, f64), DynError> {
    if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
        return Err("bootstrap samples must be non-empty and finite".into());
    }
    if state == 0 {
        state = 0x6a09_e667_f3bc_c909;
    }
    const RESAMPLES: usize = 10_000;
    let mut means = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut sum = 0.0;
        for _ in 0..samples.len() {
            state = xorshift64(state);
            let index = usize::try_from(state % samples.len() as u64)
                .map_err(|_| "bootstrap index does not fit usize")?;
            sum += samples[index];
        }
        means.push(sum / samples.len() as f64);
    }
    Ok((
        percentile_f64(&means, 1, 40)?,
        percentile_f64(&means, 39, 40)?,
    ))
}

fn bootstrap_median_ci(samples: &[f64], mut state: u64) -> Result<(f64, f64), DynError> {
    if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
        return Err("bootstrap samples must be non-empty and finite".into());
    }
    if state == 0 {
        state = 0xbb67_ae85_84ca_a73b;
    }
    const RESAMPLES: usize = 10_000;
    let mut medians = Vec::with_capacity(RESAMPLES);
    let mut resample = Vec::with_capacity(samples.len());
    for _ in 0..RESAMPLES {
        resample.clear();
        for _ in 0..samples.len() {
            state = xorshift64(state);
            let index = usize::try_from(state % samples.len() as u64)
                .map_err(|_| "bootstrap index does not fit usize")?;
            resample.push(samples[index]);
        }
        medians.push(percentile_f64(&resample, 1, 2)?);
    }
    Ok((
        percentile_f64(&medians, 1, 40)?,
        percentile_f64(&medians, 39, 40)?,
    ))
}

fn ratio(numerator: u128, denominator: u128) -> Result<f64, DynError> {
    if denominator == 0 {
        return Err("cannot compute timing ratio with zero denominator".into());
    }
    let value = numerator as f64 / denominator as f64;
    if !value.is_finite() {
        return Err("timing ratio is not finite".into());
    }
    Ok(value)
}

fn weak_components(
    adjacency: &HashMap<usize, HashSet<usize>>,
    points: &HashSet<usize>,
) -> (usize, usize) {
    let mut unseen = points.clone();
    let mut components = 0;
    let mut largest = 0;
    while let Some(&start) = unseen.iter().next() {
        components += 1;
        let mut size = 0;
        let mut queue = VecDeque::from([start]);
        unseen.remove(&start);
        while let Some(node) = queue.pop_front() {
            size += 1;
            if let Some(neighbors) = adjacency.get(&node) {
                for neighbor in neighbors {
                    if unseen.remove(neighbor) {
                        queue.push_back(*neighbor);
                    }
                }
            }
        }
        largest = largest.max(size);
    }
    (components, largest)
}

fn directed_reachability_range(
    adjacency: &HashMap<usize, HashSet<usize>>,
    starts: &[usize],
) -> (usize, usize) {
    if starts.is_empty() {
        return (0, 0);
    }
    let mut minimum = usize::MAX;
    let mut maximum = 0;
    for &start in starts {
        let mut reached = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    if reached.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        minimum = minimum.min(reached.len());
        maximum = maximum.max(reached.len());
    }
    (minimum, maximum)
}

fn proc_status_kib(label: &str) -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let rest = line.strip_prefix(label)?;
        rest.split_whitespace().next()?.parse().ok()
    })
}

fn validate_report(
    config: &Config,
    results: &[SizeResult],
    provenance: &Provenance,
) -> Result<Validation, DynError> {
    let expected_size_results = config.sizes.len();
    let expected_query_cells = config.sizes.len() * config.ef_search.len();
    let expected_build_samples = config.sizes.len() * config.build_repetitions * 4;
    let observed_query_cells = results.iter().map(|result| result.query_cells.len()).sum();
    let observed_build_samples = results
        .iter()
        .map(|result| result.build_samples.len())
        .sum();
    let expected_query_samples = config.holdout_queries * config.query_passes;
    let configured_sizes: BTreeSet<_> = config.sizes.iter().copied().collect();
    let observed_sizes: BTreeSet<_> = results.iter().map(|result| result.corpus_size).collect();
    let configured_ef_search: BTreeSet<_> = config.ef_search.iter().copied().collect();
    let complete = results.len() == expected_size_results
        && observed_sizes == configured_sizes
        && observed_query_cells == expected_query_cells
        && observed_build_samples == expected_build_samples
        && results.iter().all(|result| {
            result.corpus_sha256.len() == 64
                && result.build_summary.baseline_latency.count == config.build_repetitions
                && result.build_summary.candidate_latency.count == config.build_repetitions
                && result.build_summary.baseline_aa_latency.count == config.build_repetitions
                && result.build_summary.candidate_aa_latency.count == config.build_repetitions
                && result.build_summary.paired_candidate_over_baseline.count
                    == config.build_repetitions
                && distributions_retain_raw_samples([
                    &result.build_summary.baseline_latency,
                    &result.build_summary.candidate_latency,
                    &result.build_summary.baseline_aa_latency,
                    &result.build_summary.candidate_aa_latency,
                ])
                && result.query_cells.iter().all(|cell| {
                    cell.candidate_over_baseline_p50.is_finite()
                        && cell.baseline_aa_null_ratio_p50.is_finite()
                        && cell.candidate_aa_null_ratio_p50.is_finite()
                        && cell.baseline_latency.count == expected_query_samples
                        && cell.candidate_latency.count == expected_query_samples
                        && cell.baseline_aa_latency.count == expected_query_samples
                        && cell.candidate_aa_latency.count == expected_query_samples
                        && cell.paired_candidate_over_baseline.count == config.holdout_queries
                        && distributions_retain_raw_samples([
                            &cell.baseline_latency,
                            &cell.candidate_latency,
                            &cell.baseline_aa_latency,
                            &cell.candidate_aa_latency,
                        ])
                        && cell.baseline_quality.query_count == config.holdout_queries
                        && cell.candidate_quality.query_count == config.holdout_queries
                        && cell.baseline_quality.recall_at_k.len() == config.holdout_queries
                        && cell.candidate_quality.recall_at_k.len() == config.holdout_queries
                })
                && result
                    .query_cells
                    .iter()
                    .map(|cell| cell.ef_search)
                    .collect::<BTreeSet<_>>()
                    == configured_ef_search
                && result.replicated_quality_cells.len() == config.ef_search.len()
                && result
                    .replicated_quality_cells
                    .iter()
                    .map(|cell| cell.ef_search)
                    .collect::<BTreeSet<_>>()
                    == configured_ef_search
                && result.replicated_quality_cells.iter().all(|cell| {
                    cell.corpus_size == result.corpus_size
                        && cell.baseline.graph_count == config.build_repetitions * 2
                        && cell.candidate.graph_count == config.build_repetitions * 2
                        && cell.baseline.queries_per_graph == config.holdout_queries
                        && cell.candidate.queries_per_graph == config.holdout_queries
                })
        });
    if !complete {
        return Err(format!(
            "report is incomplete: expected {expected_size_results} size results, \
             {expected_query_cells} query cells, and {expected_build_samples} build samples; \
             observed {}, {observed_query_cells}, and {observed_build_samples}",
            results.len()
        )
        .into());
    }

    let mut correctness_violations = BTreeSet::new();
    let mut performance_blockers = BTreeSet::new();
    for result in results {
        for sample in result
            .build_samples
            .iter()
            .filter(|sample| sample.engine == EngineKind::Candidate)
        {
            collect_topology_violations(
                &sample.topology,
                result.corpus_size,
                &format!(
                    "size {} candidate build repetition {} slot {}",
                    result.corpus_size, sample.repetition, sample.abba_slot
                ),
                &mut correctness_violations,
            );
        }
        if result.build_summary.baseline_median_peak_rss_kib.is_none()
            || result.build_summary.candidate_median_peak_rss_kib.is_none()
            || result
                .build_summary
                .paired_candidate_over_baseline_peak_rss
                .as_ref()
                .is_none_or(|estimate| estimate.count != config.build_repetitions)
            || result
                .build_samples
                .iter()
                .any(|sample| sample.peak_rss_kib.is_none())
        {
            performance_blockers.insert(format!(
                "size {} does not have complete paired Linux peak-RSS samples",
                result.corpus_size
            ));
        }
        if result
            .build_summary
            .paired_candidate_over_baseline_artifact_bytes
            .count
            != config.build_repetitions
        {
            performance_blockers.insert(format!(
                "size {} does not have complete paired artifact-size samples",
                result.corpus_size
            ));
        }
        if !null_control_within(
            &result.build_summary.paired_baseline_aa_null,
            BUILD_NULL_TOLERANCE,
        ) || !null_control_within(
            &result.build_summary.paired_candidate_aa_null,
            BUILD_NULL_TOLERANCE,
        ) {
            performance_blockers.insert(format!(
                "size {} build A/A null exceeds the ±{:.0}% admission band",
                result.corpus_size,
                BUILD_NULL_TOLERANCE * 100.0
            ));
        }
        if let Some(first_cell) = result.query_cells.first() {
            collect_topology_violations(
                &first_cell.candidate_topology,
                result.corpus_size,
                &format!("size {} candidate query graph", result.corpus_size),
                &mut correctness_violations,
            );
        }
        for cell in &result.query_cells {
            let quality = &cell.candidate_quality;
            if quality.underfilled_queries != 0
                || quality.overfilled_queries != 0
                || quality.duplicate_id_queries != 0
                || quality.out_of_range_id_queries != 0
                || quality.nondeterministic_repeat_queries != 0
            {
                correctness_violations.insert(format!(
                    "size {} ef {} candidate query contract: underfilled={}, overfilled={}, \
                     duplicate={}, out_of_range={}, nondeterministic={}",
                    result.corpus_size,
                    cell.ef_search,
                    quality.underfilled_queries,
                    quality.overfilled_queries,
                    quality.duplicate_id_queries,
                    quality.out_of_range_id_queries,
                    quality.nondeterministic_repeat_queries
                ));
            }
            if !null_control_within(&cell.paired_baseline_aa_null, QUERY_NULL_TOLERANCE)
                || !null_control_within(&cell.paired_candidate_aa_null, QUERY_NULL_TOLERANCE)
            {
                performance_blockers.insert(format!(
                    "size {} ef {} query A/A null exceeds the ±{:.0}% admission band",
                    result.corpus_size,
                    cell.ef_search,
                    QUERY_NULL_TOLERANCE * 100.0
                ));
            }
        }
        for cell in &result.replicated_quality_cells {
            let candidate = &cell.candidate;
            if candidate.underfilled_queries != 0
                || candidate.overfilled_queries != 0
                || candidate.duplicate_id_queries != 0
                || candidate.out_of_range_id_queries != 0
                || candidate.nondeterministic_repeat_queries != 0
            {
                correctness_violations.insert(format!(
                    "size {} ef {} replicated candidate query contract: underfilled={}, \
                     overfilled={}, duplicate={}, out_of_range={}, nondeterministic={}",
                    result.corpus_size,
                    cell.ef_search,
                    candidate.underfilled_queries,
                    candidate.overfilled_queries,
                    candidate.duplicate_id_queries,
                    candidate.out_of_range_id_queries,
                    candidate.nondeterministic_repeat_queries
                ));
            }
            if cell.ef_search == PRIMARY_EF_SEARCH
                && candidate.hierarchical_mean_95_low < MIN_ABSOLUTE_RECALL
            {
                correctness_violations.insert(format!(
                    "size {} ef {} candidate absolute recall lower CI {:.6} is below {MIN_ABSOLUTE_RECALL}",
                    result.corpus_size, cell.ef_search, candidate.hierarchical_mean_95_low
                ));
            }
            if config.profile == "full"
                && cell.ef_search == PRIMARY_EF_SEARCH
                && candidate.distribution_free_mean_95_low < MIN_ABSOLUTE_RECALL
            {
                correctness_violations.insert(format!(
                    "size {} ef {} candidate distribution-free absolute-recall lower bound \
                     {:.6} is below {MIN_ABSOLUTE_RECALL}",
                    result.corpus_size, cell.ef_search, candidate.distribution_free_mean_95_low
                ));
            }
            if cell.candidate_minus_baseline.hierarchical_mean_95_low < -MAX_MEAN_RECALL_REGRESSION
            {
                correctness_violations.insert(format!(
                    "size {} ef {} replicated candidate-minus-baseline recall lower CI {:.6} \
                     is below -{MAX_MEAN_RECALL_REGRESSION}",
                    result.corpus_size,
                    cell.ef_search,
                    cell.candidate_minus_baseline.hierarchical_mean_95_low
                ));
            }
        }
    }
    collect_profile_admission_blockers(config, &mut performance_blockers);
    if config.corpus_slab.is_none() {
        performance_blockers.insert("no real LE-f32 corpus slab was supplied".to_owned());
    }
    if provenance.executable_sha256_before.len() != 64
        || provenance.executable_sha256_after.len() != 64
        || !provenance.executable_stable
        || !workspace_source_binding_is_complete(
            &provenance.build.workspace_source_receipt,
            &provenance.workspace_source_receipt_start,
            &provenance.workspace_source_receipt_end,
            &provenance.cargo_lock_sha256,
        )
        || provenance.resolved_hnsw_packages.len() != 2
        || !provenance.candidate_reverse_edge_is_ancestor
        || !provenance.candidate_logical_layer_is_ancestor
        || !provenance.candidate_checkout_tracked_clean
        || provenance.candidate_source_sha256 != provenance.build.candidate_source_sha256
        || provenance.baseline_source_sha256 != provenance.build.baseline_source_sha256
        || provenance.runtime_rustc_vv_sha256 != provenance.build.rustc_vv_sha256
        || (config.profile == "full" && !provenance.build.is_release_perf())
        || provenance.debug_assertions
    {
        performance_blockers.insert(
            "binary, workspace source/lock, or dependency provenance is incomplete".to_owned(),
        );
    }
    let correctness_violations: Vec<_> = correctness_violations.into_iter().collect();
    let performance_blockers: Vec<_> = performance_blockers.into_iter().collect();
    let correctness_passed = correctness_violations.is_empty();
    let measurement_admissible = complete && correctness_passed && performance_blockers.is_empty();
    let (performance_claim_status, performance_claim_reasons) = if !correctness_passed {
        (
            PerformanceClaimStatus::Quarantine,
            vec!["correctness failed, so no performance claim is admissible".to_owned()],
        )
    } else if !performance_blockers.is_empty() {
        (
            PerformanceClaimStatus::Quarantine,
            vec!["one or more measurement-admission gates failed".to_owned()],
        )
    } else {
        classify_performance(results, config)?
    };
    Ok(Validation {
        expected_size_results,
        observed_size_results: results.len(),
        expected_query_cells,
        observed_query_cells,
        expected_build_samples,
        observed_build_samples,
        complete,
        correctness_passed,
        measurement_admissible,
        performance_claim_status,
        correctness_violations,
        performance_blockers,
        performance_claim_reasons,
    })
}

fn workspace_source_binding_is_complete(
    build: &WorkspaceSourceReceipt,
    start: &WorkspaceSourceReceipt,
    end: &WorkspaceSourceReceipt,
    runtime_cargo_lock_sha256: &str,
) -> bool {
    build.is_well_formed()
        && start.is_well_formed()
        && end.is_well_formed()
        && is_sha256_hex(runtime_cargo_lock_sha256)
        && runtime_cargo_lock_sha256 == build.cargo_lock_sha256
        && start == build
        && end == build
}

fn collect_profile_admission_blockers(
    config: &Config,
    performance_blockers: &mut BTreeSet<String>,
) {
    if config.profile == "full" {
        performance_blockers.insert(FULL_ADMISSION_HOLD.to_owned());
    } else {
        performance_blockers
            .insert("smoke/synthetic profiles are diagnostic, never decision-grade".to_owned());
    }
}

fn distributions_retain_raw_samples<const N: usize>(values: [&Distribution; N]) -> bool {
    values
        .iter()
        .all(|distribution| distribution.count == distribution.samples_ns.len())
}

fn null_control_within(estimate: &PairedRatio, tolerance: f64) -> bool {
    estimate.bootstrap_median_95_low >= 1.0 - tolerance
        && estimate.bootstrap_median_95_low <= 1.0
        && estimate.bootstrap_median_95_high >= 1.0
        && estimate.bootstrap_median_95_high <= 1.0 + tolerance
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetricDecision {
    Win,
    NonRegression,
    Loss,
    Undecidable,
}

fn classify_performance(
    results: &[SizeResult],
    config: &Config,
) -> Result<(PerformanceClaimStatus, Vec<String>), DynError> {
    if !config.sizes.contains(&PRIMARY_SIZE) || !config.ef_search.contains(&PRIMARY_EF_SEARCH) {
        return Err(
            "full claim classification lacks the predeclared 100k/ef=100 primary cell".into(),
        );
    }
    let mut reasons = Vec::new();
    let mut primary_win = false;
    let mut valid_loss = false;
    let mut undecidable = false;
    for result in results {
        let build = classify_timing_ratio(
            &format!("{}-vector build latency", result.corpus_size),
            &result.build_summary.paired_candidate_over_baseline,
            [
                &result.build_summary.paired_baseline_aa_null,
                &result.build_summary.paired_candidate_aa_null,
            ],
            &mut reasons,
        );
        if result.corpus_size == PRIMARY_SIZE && build == MetricDecision::Win {
            primary_win = true;
        }
        accumulate_metric_decision(build, &mut valid_loss, &mut undecidable);

        let rss = classify_safety_ratio(
            &format!("{}-vector peak RSS", result.corpus_size),
            result
                .build_summary
                .paired_candidate_over_baseline_peak_rss
                .as_ref(),
            &mut reasons,
        );
        accumulate_metric_decision(rss, &mut valid_loss, &mut undecidable);
        let artifact = classify_safety_ratio(
            &format!("{}-vector artifact bytes", result.corpus_size),
            Some(
                &result
                    .build_summary
                    .paired_candidate_over_baseline_artifact_bytes,
            ),
            &mut reasons,
        );
        accumulate_metric_decision(artifact, &mut valid_loss, &mut undecidable);

        for query in &result.query_cells {
            let query_decision = classify_timing_ratio(
                &format!(
                    "{}-vector ef={} query latency",
                    result.corpus_size, query.ef_search
                ),
                &query.paired_candidate_over_baseline,
                [
                    &query.paired_baseline_aa_null,
                    &query.paired_candidate_aa_null,
                ],
                &mut reasons,
            );
            if result.corpus_size == PRIMARY_SIZE
                && query.ef_search == PRIMARY_EF_SEARCH
                && query_decision == MetricDecision::Win
            {
                primary_win = true;
            }
            accumulate_metric_decision(query_decision, &mut valid_loss, &mut undecidable);
        }
    }
    let status = finalize_performance_status(primary_win, valid_loss, undecidable, &mut reasons);
    Ok((status, reasons))
}

fn finalize_performance_status(
    primary_win: bool,
    valid_loss: bool,
    undecidable: bool,
    reasons: &mut Vec<String>,
) -> PerformanceClaimStatus {
    if valid_loss {
        PerformanceClaimStatus::Block
    } else if undecidable {
        reasons.push(
            "at least one required full-matrix metric cannot prove non-regression; the run is \
             measured but quarantined from an admission claim"
                .to_owned(),
        );
        PerformanceClaimStatus::Quarantine
    } else if primary_win {
        PerformanceClaimStatus::Allow
    } else {
        reasons.push(
            "all required cells proved non-regression, but neither predeclared 100k build nor \
             100k/ef=100 query latency established a material win"
                .to_owned(),
        );
        PerformanceClaimStatus::NoClaim
    }
}

fn classify_timing_ratio(
    label: &str,
    estimate: &PairedRatio,
    nulls: [&PairedRatio; 2],
    reasons: &mut Vec<String>,
) -> MetricDecision {
    let null_floor = nulls
        .iter()
        .flat_map(|estimate| {
            [
                estimate.bootstrap_median_95_low,
                estimate.bootstrap_median_95_high,
            ]
        })
        .map(|ratio| ratio.ln().abs())
        .fold(0.0, f64::max);
    let clears_null = estimate.median.ln().abs() >= 2.0 * null_floor;
    if estimate.median <= 0.95 && estimate.bootstrap_median_95_high < 1.0 && clears_null {
        reasons.push(format!(
            "{label} valid win: median ratio {:.4}, CI [{:.4}, {:.4}], A/A log floor {:.4}",
            estimate.median,
            estimate.bootstrap_median_95_low,
            estimate.bootstrap_median_95_high,
            null_floor
        ));
        MetricDecision::Win
    } else if estimate.median >= 1.05 && estimate.bootstrap_median_95_low > 1.0 && clears_null {
        reasons.push(format!(
            "{label} valid loss: median ratio {:.4}, CI [{:.4}, {:.4}], A/A log floor {:.4}",
            estimate.median,
            estimate.bootstrap_median_95_low,
            estimate.bootstrap_median_95_high,
            null_floor
        ));
        MetricDecision::Loss
    } else if estimate.bootstrap_median_95_high <= 1.05 {
        reasons.push(format!(
            "{label} proves non-regression: median ratio {:.4}, CI [{:.4}, {:.4}], \
             A/A log floor {:.4}",
            estimate.median,
            estimate.bootstrap_median_95_low,
            estimate.bootstrap_median_95_high,
            null_floor
        ));
        MetricDecision::NonRegression
    } else {
        reasons.push(format!(
            "{label} is undecidable: median ratio {:.4}, CI [{:.4}, {:.4}], A/A log floor \
             {:.4}; a >5% regression remains plausible",
            estimate.median,
            estimate.bootstrap_median_95_low,
            estimate.bootstrap_median_95_high,
            null_floor
        ));
        MetricDecision::Undecidable
    }
}

fn classify_safety_ratio(
    label: &str,
    estimate: Option<&PairedRatio>,
    reasons: &mut Vec<String>,
) -> MetricDecision {
    match estimate {
        Some(estimate) if estimate.bootstrap_median_95_low > 1.05 => {
            reasons.push(format!(
                "{label} valid >5% regression: median ratio {:.4}, CI [{:.4}, {:.4}]",
                estimate.median,
                estimate.bootstrap_median_95_low,
                estimate.bootstrap_median_95_high
            ));
            MetricDecision::Loss
        }
        Some(estimate) if estimate.bootstrap_median_95_high <= 1.05 => {
            reasons.push(format!(
                "{label} proves non-regression: median ratio {:.4}, CI [{:.4}, {:.4}]",
                estimate.median,
                estimate.bootstrap_median_95_low,
                estimate.bootstrap_median_95_high
            ));
            MetricDecision::NonRegression
        }
        Some(estimate) => {
            reasons.push(format!(
                "{label} is undecidable: median ratio {:.4}, CI [{:.4}, {:.4}]; a >5% \
                 regression remains plausible",
                estimate.median,
                estimate.bootstrap_median_95_low,
                estimate.bootstrap_median_95_high
            ));
            MetricDecision::Undecidable
        }
        None => {
            reasons.push(format!("{label} has no complete paired estimate"));
            MetricDecision::Undecidable
        }
    }
}

const fn accumulate_metric_decision(
    decision: MetricDecision,
    valid_loss: &mut bool,
    undecidable: &mut bool,
) {
    match decision {
        MetricDecision::Loss => *valid_loss = true,
        MetricDecision::Undecidable => *undecidable = true,
        MetricDecision::Win | MetricDecision::NonRegression => {}
    }
}

fn collect_topology_violations(
    topology: &TopologyMetrics,
    expected_points: usize,
    context: &str,
    violations: &mut BTreeSet<String>,
) {
    let counters = [
        ("duplicate_origin_ids", topology.duplicate_origin_ids),
        ("out_of_range_origin_ids", topology.out_of_range_origin_ids),
        (
            "duplicate_internal_point_ids",
            topology.duplicate_internal_point_ids,
        ),
        ("negative_internal_slots", topology.negative_internal_slots),
        (
            "truncated_neighborhood_tables",
            topology.truncated_neighborhood_tables,
        ),
        (
            "invalid_above_level_edges",
            topology.invalid_above_level_edges,
        ),
        (
            "invalid_target_level_edges",
            topology.invalid_target_level_edges,
        ),
        ("out_of_range_edges", topology.out_of_range_edges),
        ("missing_target_edges", topology.missing_target_edges),
        (
            "mismatched_target_point_ids",
            topology.mismatched_target_point_ids,
        ),
        (
            "non_finite_edge_distances",
            topology.non_finite_edge_distances,
        ),
        ("duplicate_edges", topology.duplicate_edges),
        ("self_edges", topology.self_edges),
    ];
    if topology.points_seen != expected_points || topology.unique_origin_ids != expected_points {
        violations.insert(format!(
            "{context}: saw {} points and {} unique IDs, expected {expected_points}",
            topology.points_seen, topology.unique_origin_ids
        ));
    }
    for (name, count) in counters {
        if count != 0 {
            violations.insert(format!("{context}: {name}={count}"));
        }
    }
    if expected_points > 1 {
        if topology.directed_layer0_edges == 0 {
            violations.insert(format!("{context}: base layer has no directed edges"));
        }
        if topology.weak_component_count != 1 || topology.largest_weak_component != expected_points
        {
            violations.insert(format!(
                "{context}: weak components={}, largest={}, expected one component of {expected_points}",
                topology.weak_component_count, topology.largest_weak_component
            ));
        }
    }
    if topology.actual_entry_api_available {
        if expected_points == 0 {
            if topology.actual_entry_origin.is_some()
                || topology.actual_entry_identity_valid.is_some()
                || topology.actual_entry_is_max_level.is_some()
                || topology.reachable_from_actual_entry.is_some()
            {
                violations.insert(format!(
                    "{context}: empty candidate graph unexpectedly exposes an actual entry point"
                ));
            }
        } else if topology.actual_entry_origin.is_none()
            || topology.actual_entry_identity_valid != Some(true)
            || topology.actual_entry_is_max_level != Some(true)
            || topology.reachable_from_actual_entry != Some(expected_points)
        {
            violations.insert(format!(
                "{context}: actual entry is origin {:?}, identity_valid={:?}, \
                 max_level={:?}, reaches {:?} points; expected a valid maximum-level entry \
                 reaching all {expected_points}",
                topology.actual_entry_origin,
                topology.actual_entry_identity_valid,
                topology.actual_entry_is_max_level,
                topology.reachable_from_actual_entry
            ));
        }
    }
}

fn provenance(
    executable: &Path,
    executable_sha256_before: &str,
    workspace_source_receipt_start: WorkspaceSourceReceipt,
) -> Result<Provenance, DynError> {
    let workspace = workspace_root()?;
    let workspace_source_receipt_end = workspace_source_receipt(&workspace)?;
    let cargo_lock_path = workspace.join("Cargo.lock");
    let cargo_lock = fs::read_to_string(&cargo_lock_path)?;
    let cargo_lock_sha256 = sha256_file(&cargo_lock_path)?;
    let resolved_hnsw_packages = resolved_hnsw_packages(&cargo_lock)?;
    let candidate_package = resolved_hnsw_packages
        .iter()
        .find(|package| package.contains(CANDIDATE_REV))
        .cloned()
        .ok_or("Cargo.lock does not attest the compiled candidate revision")?;
    let baseline_package = resolved_hnsw_packages
        .iter()
        .find(|package| package.contains("registry+"))
        .cloned()
        .ok_or("Cargo.lock does not attest the published baseline")?;
    let workspace_git = workspace_git_provenance(&workspace)?;
    let dependency_sources = attest_dependency_sources(&workspace)?;
    let build = embedded_build_attestation();
    let executable_sha256_after = sha256_file(executable)?;
    let rustc_vv = command_output_in("rustc", &["-Vv"], &workspace)?;
    let rustc_version = String::from_utf8(rustc_vv.clone())?;
    let hostname = fs::read_to_string("/etc/hostname")?;
    let mut relevant_environment = BTreeMap::new();
    for name in [
        "FRANKENSEARCH_BENCH_PROFILE",
        "RAYON_NUM_THREADS",
        "RUSTFLAGS",
        "CARGO_PROFILE_RELEASE_LTO",
        "CARGO_PROFILE_RELEASE_PERF_LTO",
        "CARGO_PROFILE_RELEASE_PERF_CODEGEN_UNITS",
        "CARGO_PROFILE_RELEASE_PERF_OPT_LEVEL",
        "MALLOC_CONF",
        "GLIBC_TUNABLES",
    ] {
        relevant_environment.insert(
            name.to_owned(),
            env::var(name).unwrap_or_else(|_| "<unset>".to_owned()),
        );
    }
    Ok(Provenance {
        generated_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        workspace_git,
        workspace_source_receipt_start,
        workspace_source_receipt_end,
        cargo_lock_sha256,
        resolved_hnsw_packages,
        executable_sha256_before: executable_sha256_before.to_owned(),
        executable_stable: executable_sha256_before == executable_sha256_after,
        executable_sha256_after,
        rustc_version: nonempty_trimmed(&rustc_version, "rustc version")?,
        runtime_rustc_vv_sha256: sha256_bytes(&rustc_vv),
        hostname: nonempty_trimmed(&hostname, "hostname")?,
        kernel: command_text_in("uname", &["-a"], &workspace, false)?,
        cpu_model: cpu_model()?,
        rayon_threads: rayon::current_num_threads(),
        baseline_package,
        candidate_package,
        candidate_reverse_edge_revision: CANDIDATE_REVERSE_EDGE_REV.to_owned(),
        candidate_logical_layer_revision: CANDIDATE_LOGICAL_LAYER_REV.to_owned(),
        candidate_cumulative_revision: CANDIDATE_REV.to_owned(),
        candidate_revision_note:
            "cumulative candidate through 18a5a1a includes both production topology repairs: \
             461bedf corrects reverse-edge layer placement and 18a5a1a searches logical layers \
             through all participating higher-level points. It also includes the entry-point \
             inspection API, regression tests, and a packaging-only default-FFI opt-out needed \
             to link baseline and candidate into one evidence executable. This runner decides \
             the cumulative candidate; it cannot isolate either repair's causal performance \
             contribution."
                .to_owned(),
        candidate_checkout_head: dependency_sources.candidate_checkout_head,
        candidate_reverse_edge_is_ancestor: dependency_sources.candidate_reverse_edge_is_ancestor,
        candidate_logical_layer_is_ancestor: dependency_sources.candidate_logical_layer_is_ancestor,
        candidate_checkout_tracked_clean: dependency_sources.candidate_checkout_tracked_clean,
        candidate_source_sha256: dependency_sources.candidate_source_sha256,
        baseline_source_sha256: dependency_sources.baseline_source_sha256,
        build,
        debug_assertions: cfg!(debug_assertions),
        compiled_target_features: compiled_target_features(),
        relevant_environment,
        command: env::args().collect(),
    })
}

struct DependencySourceAttestation {
    candidate_checkout_head: String,
    candidate_reverse_edge_is_ancestor: bool,
    candidate_logical_layer_is_ancestor: bool,
    candidate_checkout_tracked_clean: bool,
    candidate_source_sha256: String,
    baseline_source_sha256: String,
}

fn workspace_root() -> Result<PathBuf, DynError> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.join("../..").canonicalize()?;
    if !workspace.join("Cargo.toml").is_file() || !workspace.join("Cargo.lock").is_file() {
        return Err(format!(
            "compiled workspace root lacks Cargo.toml or Cargo.lock: {}",
            workspace.display()
        )
        .into());
    }
    Ok(workspace)
}

fn workspace_source_receipt(workspace: &Path) -> Result<WorkspaceSourceReceipt, DynError> {
    let mut files: Vec<_> = WORKSPACE_SOURCE_FIXED_INPUTS
        .iter()
        .map(|relative| workspace.join(relative))
        .collect();
    collect_dependency_source_files(&workspace.join("crates/frankensearch-core/src"), &mut files)?;
    collect_dependency_source_files(
        &workspace.join("crates/frankensearch-index/src"),
        &mut files,
    )?;
    workspace_source_receipt_from_paths(workspace, files)
}

fn workspace_source_receipt_from_paths(
    workspace: &Path,
    mut files: Vec<PathBuf>,
) -> Result<WorkspaceSourceReceipt, DynError> {
    files.sort();
    files.dedup();
    let mut inputs = Vec::with_capacity(files.len());
    for path in files {
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(format!(
                "workspace build input is not a regular non-symlink file: {}",
                path.display()
            )
            .into());
        }
        let relative = path
            .strip_prefix(workspace)
            .map_err(|_| format!("workspace build input escaped {}", workspace.display()))?
            .to_str()
            .ok_or_else(|| {
                format!(
                    "workspace build input path is not valid UTF-8: {}",
                    path.display()
                )
            })?
            .to_owned();
        if !safe_relative_source_path(&relative) {
            return Err(format!("unsafe workspace build input path: {relative:?}").into());
        }
        let bytes = fs::read(&path)?;
        let byte_len = u64::try_from(bytes.len())?;
        let sha256 = sha256_bytes(&bytes);
        inputs.push(WorkspaceSourceInput {
            path: relative,
            byte_len,
            sha256,
        });
    }
    let cargo_lock_sha256 = inputs
        .iter()
        .find(|input| input.path == "Cargo.lock")
        .map(|input| input.sha256.clone())
        .ok_or("workspace source receipt does not include Cargo.lock")?;
    let aggregate_sha256 = workspace_source_aggregate_sha256(&inputs)
        .ok_or("workspace source receipt input path length overflow")?;
    Ok(WorkspaceSourceReceipt {
        schema: WORKSPACE_SOURCE_RECEIPT_SCHEMA.to_owned(),
        aggregate_sha256,
        cargo_lock_sha256,
        inputs,
    })
}

fn workspace_source_aggregate_sha256(inputs: &[WorkspaceSourceInput]) -> Option<String> {
    let mut aggregate = Sha256::new();
    aggregate.update(WORKSPACE_SOURCE_RECEIPT_SCHEMA.as_bytes());
    for input in inputs {
        aggregate.update(u64::try_from(input.path.len()).ok()?.to_le_bytes());
        aggregate.update(input.path.as_bytes());
        aggregate.update(input.byte_len.to_le_bytes());
        aggregate.update(input.sha256.as_bytes());
    }
    Some(hex_bytes(&aggregate.finalize()))
}

fn safe_relative_source_path(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| !matches!(byte, b'\r' | b'\n' | b'\t'))
        && Path::new(value)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn workspace_git_provenance(workspace: &Path) -> Result<WorkspaceGitProvenance, DynError> {
    let mut probe = Command::new("git");
    probe
        .arg("-c")
        .arg(format!("safe.directory={}", workspace.display()))
        .args(["rev-parse", "--show-toplevel"])
        .env("LC_ALL", "C")
        .current_dir(workspace);
    if let Some(parent) = workspace.parent() {
        probe.env("GIT_CEILING_DIRECTORIES", parent);
    }
    let output = probe.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("not a git repository") {
            return Ok(WorkspaceGitProvenance::Unavailable {
                reason: WorkspaceGitUnavailableReason::WorkspaceNotAGitCheckout,
                probe_stderr_sha256: sha256_bytes(&output.stderr),
            });
        }
        return Err(format!(
            "git rev-parse --show-toplevel failed while probing workspace provenance: {}",
            stderr.trim()
        )
        .into());
    }

    let top_level = nonempty_trimmed(
        &String::from_utf8(output.stdout)?,
        "workspace Git top-level",
    )?;
    let canonical_top_level = Path::new(&top_level).canonicalize()?;
    let canonical_workspace = workspace.canonicalize()?;
    if canonical_top_level != canonical_workspace {
        return Err(format!(
            "workspace Git top-level {} does not match compiled workspace {}",
            canonical_top_level.display(),
            canonical_workspace.display()
        )
        .into());
    }
    let status_porcelain = command_text_in(
        "git",
        &["status", "--porcelain=v1", "--untracked-files=all"],
        workspace,
        true,
    )?;
    let diff = command_output_in("git", &["diff", "--binary", "HEAD", "--"], workspace)?;
    Ok(WorkspaceGitProvenance::Available {
        top_level,
        head: command_text_in("git", &["rev-parse", "HEAD"], workspace, false)?,
        status_porcelain,
        diff_sha256: sha256_bytes(&diff),
        untracked_file_sha256: untracked_file_sha256(workspace)?,
    })
}

fn untracked_file_sha256(workspace: &Path) -> Result<BTreeMap<String, String>, DynError> {
    let output = command_output_in(
        "git",
        &["ls-files", "--others", "--exclude-standard", "-z"],
        workspace,
    )?;
    let mut hashes = BTreeMap::new();
    for raw_path in output
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        let relative = std::str::from_utf8(raw_path)
            .map_err(|_| "untracked source path is not valid UTF-8")?;
        let path = workspace.join(relative);
        let metadata = fs::symlink_metadata(&path)?;
        let digest = if metadata.file_type().is_symlink() {
            sha256_bytes(fs::read_link(&path)?.as_os_str().as_encoded_bytes())
        } else if metadata.is_file() {
            let canonical = path.canonicalize()?;
            let canonical_workspace = workspace.canonicalize()?;
            if !canonical.starts_with(&canonical_workspace) {
                return Err(format!(
                    "untracked source path escapes workspace: {}",
                    path.display()
                )
                .into());
            }
            sha256_file(&canonical)?
        } else {
            return Err(format!(
                "untracked source entry is neither file nor symlink: {}",
                path.display()
            )
            .into());
        };
        hashes.insert(relative.to_owned(), digest);
    }
    Ok(hashes)
}

fn attest_dependency_sources(_workspace: &Path) -> Result<DependencySourceAttestation, DynError> {
    // Do not invoke `cargo metadata --offline` here. Cargo can require source
    // archives for unrelated optional workspace dependencies even after this
    // binary was built, which made an otherwise complete remote smoke run die
    // during provenance collection. The build script already located and
    // hashed the two exact sources; find those same cache roots directly and
    // compare their runtime hashes to the embedded build attestations.
    let (checkout, baseline) = locate_hnsw_dependency_sources()?;
    let head = command_text_in("git", &["rev-parse", "HEAD"], &checkout, false)?;
    if head != CANDIDATE_REV {
        return Err(format!(
            "candidate checkout HEAD {head} does not match pinned {CANDIDATE_REV}"
        )
        .into());
    }
    let reverse_edge_ancestry = command_status_in(
        "git",
        &[
            "merge-base",
            "--is-ancestor",
            CANDIDATE_REVERSE_EDGE_REV,
            CANDIDATE_REV,
        ],
        &checkout,
    )?;
    let logical_layer_ancestry = command_status_in(
        "git",
        &[
            "merge-base",
            "--is-ancestor",
            CANDIDATE_LOGICAL_LAYER_REV,
            CANDIDATE_REV,
        ],
        &checkout,
    )?;
    let tracked_clean = [
        ["diff", "--quiet", "HEAD", "--"].as_slice(),
        ["diff", "--cached", "--quiet", "HEAD", "--"].as_slice(),
    ]
    .iter()
    .all(|args| command_status_in("git", args, &checkout).is_ok_and(|status| status.success()));
    Ok(DependencySourceAttestation {
        candidate_checkout_head: head,
        candidate_reverse_edge_is_ancestor: reverse_edge_ancestry.success(),
        candidate_logical_layer_is_ancestor: logical_layer_ancestry.success(),
        candidate_checkout_tracked_clean: tracked_clean,
        candidate_source_sha256: hash_dependency_source_tree(&checkout)?,
        baseline_source_sha256: hash_dependency_source_tree(&baseline)?,
    })
}

fn locate_hnsw_dependency_sources() -> Result<(PathBuf, PathBuf), DynError> {
    let cargo_home = std::env::var_os("CARGO_HOME").map_or_else(
        || {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".cargo"))
                .ok_or_else(|| -> DynError { "neither CARGO_HOME nor HOME is available".into() })
        },
        |path| Ok(PathBuf::from(path)),
    )?;
    let mut candidate_matches = Vec::new();
    for repository in read_directory_paths(&cargo_home.join("git/checkouts"))? {
        if !repository.is_dir() {
            continue;
        }
        for checkout in read_directory_paths(&repository)? {
            if checkout.join("Cargo.toml").is_file()
                && command_text_in("git", &["rev-parse", "HEAD"], &checkout, false)
                    .is_ok_and(|head| head == CANDIDATE_REV)
            {
                candidate_matches.push(checkout);
            }
        }
    }
    let mut baseline_matches = Vec::new();
    for registry in read_directory_paths(&cargo_home.join("registry/src"))? {
        let baseline = registry.join(format!("hnsw_rs-{BASELINE_VERSION}"));
        if baseline.join("Cargo.toml").is_file() {
            baseline_matches.push(baseline);
        }
    }
    Ok((
        require_unique_path(candidate_matches, "candidate hnsw_rs source")?,
        require_unique_path(baseline_matches, "published hnsw_rs 0.3.4 source")?,
    ))
}

fn read_directory_paths(directory: &Path) -> Result<Vec<PathBuf>, DynError> {
    fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()).map_err(Into::into))
        .collect()
}

fn require_unique_path(mut paths: Vec<PathBuf>, description: &str) -> Result<PathBuf, DynError> {
    paths.sort();
    paths.dedup();
    match paths.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(format!("could not locate {description} in Cargo's source cache").into()),
        _ => Err(format!("found multiple {description} candidates: {paths:?}").into()),
    }
}

fn compiled_target_features() -> Vec<String> {
    let mut features = Vec::new();
    for (name, enabled) in [
        ("avx", cfg!(target_feature = "avx")),
        ("avx2", cfg!(target_feature = "avx2")),
        ("fma", cfg!(target_feature = "fma")),
        ("sse4.1", cfg!(target_feature = "sse4.1")),
        ("sse4.2", cfg!(target_feature = "sse4.2")),
        ("neon", cfg!(target_feature = "neon")),
    ] {
        if enabled {
            features.push(name.to_owned());
        }
    }
    features
}

fn resolved_hnsw_packages(cargo_lock: &str) -> Result<Vec<String>, DynError> {
    let mut packages = Vec::new();
    for block in cargo_lock.split("[[package]]") {
        let name = lock_value(block, "name");
        let version = lock_value(block, "version");
        if name.as_deref() != Some("hnsw_rs") || version.as_deref() != Some(BASELINE_VERSION) {
            continue;
        }
        let source = lock_value(block, "source").ok_or("hnsw_rs lock entry lacks source")?;
        let checksum = lock_value(block, "checksum");
        packages.push(checksum.map_or_else(
            || format!("hnsw_rs {BASELINE_VERSION} {source}"),
            |checksum| format!("hnsw_rs {BASELINE_VERSION} {source} checksum={checksum}"),
        ));
    }
    packages.sort();
    let candidate_source = format!("#{CANDIDATE_REV}");
    if packages.len() != 2
        || !packages.iter().any(|package| {
            package.contains("registry+")
                && package.contains(
                    "checksum=43a5258f079b97bf2e8311ff9579e903c899dcbac0d9a138d62e9a066778bd07",
                )
        })
        || !packages
            .iter()
            .any(|package| package.contains("git+") && package.contains(&candidate_source))
    {
        return Err(format!("unexpected hnsw_rs dependency attestation: {packages:?}").into());
    }
    Ok(packages)
}

fn lock_value(block: &str, key: &str) -> Option<String> {
    let prefix = format!("{key} = \"");
    block.lines().find_map(|line| {
        line.strip_prefix(&prefix)
            .and_then(|value| value.strip_suffix('"'))
            .map(str::to_owned)
    })
}

fn command_output_in(program: &str, args: &[&str], cwd: &Path) -> Result<Vec<u8>, DynError> {
    let mut command = Command::new(program);
    if program == "git" {
        command
            .arg("-c")
            .arg(format!("safe.directory={}", cwd.display()))
            .env("LC_ALL", "C");
    }
    let output = command.args(args).current_dir(cwd).output()?;
    if !output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(output.stdout)
}

fn command_status_in(
    program: &str,
    args: &[&str],
    cwd: &Path,
) -> Result<std::process::ExitStatus, DynError> {
    let mut command = Command::new(program);
    if program == "git" {
        command
            .arg("-c")
            .arg(format!("safe.directory={}", cwd.display()))
            .env("LC_ALL", "C");
    }
    Ok(command.args(args).current_dir(cwd).status()?)
}

fn command_text_in(
    program: &str,
    args: &[&str],
    cwd: &Path,
    allow_empty: bool,
) -> Result<String, DynError> {
    let text = String::from_utf8(command_output_in(program, args, cwd)?)?
        .trim()
        .to_owned();
    if !allow_empty && text.is_empty() {
        return Err(format!("{program} {} returned no provenance", args.join(" ")).into());
    }
    Ok(text)
}

fn nonempty_trimmed(value: &str, field: &str) -> Result<String, DynError> {
    let value = value.trim().to_owned();
    if value.is_empty() {
        return Err(format!("{field} provenance is empty").into());
    }
    Ok(value)
}

fn cpu_model() -> Result<String, DynError> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    cpuinfo
        .lines()
        .find_map(|line| {
            line.strip_prefix("model name")
                .and_then(|line| line.split_once(':'))
        })
        .map(|(_, value)| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "CPU model provenance is unavailable".into())
}

fn hash_dependency_source_tree(root: &Path) -> Result<String, DynError> {
    let mut files = vec![root.join("Cargo.toml")];
    let build_rs = root.join("build.rs");
    if build_rs.is_file() {
        files.push(build_rs);
    }
    collect_dependency_source_files(&root.join("src"), &mut files)?;
    files.sort();
    files.dedup();
    if files.iter().any(|path| !path.is_file()) {
        return Err(format!(
            "dependency source attestation includes a missing file below {}",
            root.display()
        )
        .into());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.hnsw-source-tree.v1");
    for path in files {
        let relative = path
            .strip_prefix(root)
            .map_err(|_| format!("dependency source escaped {}", root.display()))?
            .as_os_str()
            .as_encoded_bytes();
        let bytes = fs::read(&path)?;
        hasher.update(u64::try_from(relative.len())?.to_le_bytes());
        hasher.update(relative);
        hasher.update(u64::try_from(bytes.len())?.to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(hex_bytes(&hasher.finalize()))
}

fn collect_dependency_source_files(
    directory: &Path,
    files: &mut Vec<PathBuf>,
) -> Result<(), DynError> {
    let mut entries: Vec<_> = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()?;
    entries.sort();
    for path in entries {
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Err(format!("dependency source contains a symlink: {}", path.display()).into());
        }
        if metadata.is_dir() {
            collect_dependency_source_files(&path, files)?;
        } else if metadata.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, DynError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_bytes(&hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn structured_log(event: &str, fields: &str) {
    eprintln!(
        "{{\"schema\":\"{SCHEMA}\",\"event\":{},{}\"unix_ms\":{}}}",
        serde_json::to_string(event).unwrap_or_else(|_| "\"serialization_error\"".to_owned()),
        if fields.is_empty() {
            String::new()
        } else {
            format!("{fields},")
        },
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis()
    );
}

fn run() -> Result<(), DynError> {
    let config = Config::parse()?;
    if config.child_engine.is_some() {
        child_main(&config)
    } else {
        parent_main(config)
    }
}

fn main() {
    if let Err(error) = run() {
        structured_log(
            "fatal_error",
            &format!(
                "\"message\":{}",
                serde_json::to_string(&error.to_string())
                    .unwrap_or_else(|_| "\"serialization_error\"".to_owned())
            ),
        );
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TopologyMutation = (&'static str, fn(&mut TopologyMetrics));

    fn assert_f64_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "expected {expected}, observed {actual}"
        );
    }

    fn test_workspace_source_receipt() -> WorkspaceSourceReceipt {
        let inputs = vec![
            WorkspaceSourceInput {
                path: "Cargo.lock".to_owned(),
                byte_len: 17,
                sha256: "a".repeat(64),
            },
            WorkspaceSourceInput {
                path: "Cargo.toml".to_owned(),
                byte_len: 19,
                sha256: "b".repeat(64),
            },
        ];
        WorkspaceSourceReceipt {
            schema: WORKSPACE_SOURCE_RECEIPT_SCHEMA.to_owned(),
            aggregate_sha256: workspace_source_aggregate_sha256(&inputs).unwrap(),
            cargo_lock_sha256: inputs[0].sha256.clone(),
            inputs,
        }
    }

    #[test]
    fn usize_lists_reject_empty_and_zero_values() {
        assert!(parse_usize_list("", "test").is_err());
        assert!(parse_usize_list("1,0,3", "test").is_err());
        assert_eq!(parse_usize_list("1,2,3", "test").unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn synthetic_corpus_is_deterministic_and_normalized() {
        let config = Config::smoke();
        let left = synthetic_corpus(&config, 32);
        let right = synthetic_corpus(&config, 32);
        assert_eq!(left.hash, right.hash);
        assert_eq!(left.vectors, right.vectors);
        for vector in &left.vectors {
            let norm: f32 = vector.iter().map(|value| value * value).sum();
            assert!((norm - 1.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn percentile_uses_nearest_rank_ceiling() {
        let values = [10_u128, 20, 30, 40, 50];
        assert_eq!(percentile_sorted_u128(&values, 1, 2).unwrap(), 30);
        assert_eq!(percentile_sorted_u128(&values, 19, 20).unwrap(), 50);
    }

    #[test]
    fn quality_metrics_detect_underfill_duplicates_and_bad_ids() {
        let actual = vec![vec![0, 0, 3], vec![99]];
        let exact = vec![vec![0, 1], vec![2, 3]];
        let metrics = quality_metrics(&actual, &exact, 2, 4, 1).unwrap();
        assert_eq!(metrics.underfilled_queries, 1);
        assert_eq!(metrics.overfilled_queries, 1);
        assert_eq!(metrics.duplicate_id_queries, 1);
        assert_eq!(metrics.out_of_range_id_queries, 1);
        assert_eq!(metrics.nondeterministic_repeat_queries, 1);
    }

    #[test]
    fn weak_components_include_isolated_points() {
        let points = HashSet::from([0, 1, 2]);
        let adjacency = HashMap::from([
            (0, HashSet::from([1])),
            (1, HashSet::from([0])),
            (2, HashSet::new()),
        ]);
        assert_eq!(weak_components(&adjacency, &points), (2, 2));
    }

    #[test]
    fn directed_reachability_preserves_edge_direction_and_all_start_points() {
        let adjacency = HashMap::from([
            (0, HashSet::from([1])),
            (1, HashSet::from([2])),
            (2, HashSet::new()),
        ]);
        assert_eq!(directed_reachability_range(&adjacency, &[0]), (3, 3));
        assert_eq!(directed_reachability_range(&adjacency, &[0, 2]), (1, 3));
        assert_eq!(directed_reachability_range(&adjacency, &[]), (0, 0));
    }

    #[test]
    fn full_profile_refuses_unmanifested_synthetic_evidence() {
        let error = Config::full().validate().unwrap_err().to_string();
        assert!(error.contains("--corpus-slab"));
    }

    #[test]
    fn every_profile_is_explicitly_held_from_decision_admission() {
        let mut smoke = BTreeSet::new();
        collect_profile_admission_blockers(&Config::smoke(), &mut smoke);
        assert!(smoke.iter().any(|reason| reason.contains("diagnostic")));

        let mut full = BTreeSet::new();
        collect_profile_admission_blockers(&Config::full(), &mut full);
        assert_eq!(full, BTreeSet::from([FULL_ADMISSION_HOLD.to_owned()]));
    }

    #[test]
    fn embedded_receipt_matches_the_runtime_workspace_inputs() {
        let embedded = embedded_build_attestation().workspace_source_receipt;
        assert!(embedded.is_well_formed());
        for expected in WORKSPACE_SOURCE_FIXED_INPUTS {
            assert!(
                embedded.inputs.iter().any(|input| input.path == *expected),
                "embedded receipt omitted {expected}"
            );
        }
        let runtime = workspace_source_receipt(&workspace_root().unwrap()).unwrap();
        assert_eq!(runtime, embedded);
    }

    #[test]
    fn workspace_receipt_is_content_addressed_and_path_order_independent() {
        let directory = tempfile::tempdir().unwrap();
        let lock = directory.path().join("Cargo.lock");
        let manifest = directory.path().join("Cargo.toml");
        fs::write(&lock, b"lock-v1").unwrap();
        fs::write(&manifest, b"manifest-v1").unwrap();
        let left = workspace_source_receipt_from_paths(
            directory.path(),
            vec![manifest.clone(), lock.clone()],
        )
        .unwrap();
        let right = workspace_source_receipt_from_paths(
            directory.path(),
            vec![lock.clone(), manifest.clone()],
        )
        .unwrap();
        assert_eq!(left, right);
        assert!(left.is_well_formed());
        assert_eq!(
            left.inputs
                .iter()
                .map(|input| input.path.as_str())
                .collect::<Vec<_>>(),
            ["Cargo.lock", "Cargo.toml"]
        );

        fs::write(&manifest, b"manifest-v2").unwrap();
        let changed =
            workspace_source_receipt_from_paths(directory.path(), vec![lock, manifest]).unwrap();
        assert_ne!(changed.aggregate_sha256, left.aggregate_sha256);
        assert_ne!(changed.inputs[1].sha256, left.inputs[1].sha256);
    }

    #[cfg(unix)]
    #[test]
    fn workspace_receipt_rejects_symlink_inputs() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let lock = directory.path().join("Cargo.lock");
        let target = directory.path().join("target.txt");
        let link = directory.path().join("Cargo.toml");
        fs::write(&lock, b"lock").unwrap();
        fs::write(&target, b"manifest").unwrap();
        symlink(&target, &link).unwrap();
        let error = workspace_source_receipt_from_paths(directory.path(), vec![lock, link])
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-symlink"));
    }

    #[test]
    fn workspace_source_binding_requires_identical_build_start_end_and_lock() {
        let build = test_workspace_source_receipt();
        assert!(workspace_source_binding_is_complete(
            &build,
            &build,
            &build,
            &build.cargo_lock_sha256
        ));

        let mut changed = build.clone();
        changed.inputs[1].sha256 = "c".repeat(64);
        changed.aggregate_sha256 = workspace_source_aggregate_sha256(&changed.inputs).unwrap();
        assert!(changed.is_well_formed());
        assert!(!workspace_source_binding_is_complete(
            &build,
            &changed,
            &build,
            &build.cargo_lock_sha256
        ));
        assert!(!workspace_source_binding_is_complete(
            &build,
            &build,
            &build,
            &"d".repeat(64)
        ));
        let mut malformed = build.clone();
        malformed.aggregate_sha256 = "e".repeat(64);
        assert!(!malformed.is_well_formed());
    }

    #[test]
    fn gitless_source_bundle_has_typed_supplementary_provenance() {
        let directory = tempfile::tempdir().unwrap();
        let provenance = workspace_git_provenance(directory.path()).unwrap();
        assert!(matches!(
            provenance,
            WorkspaceGitProvenance::Unavailable {
                reason: WorkspaceGitUnavailableReason::WorkspaceNotAGitCheckout,
                ..
            }
        ));

        let mut blockers = BTreeSet::new();
        collect_profile_admission_blockers(&Config::smoke(), &mut blockers);
        assert!(
            blockers
                .iter()
                .any(|reason| reason.contains("never decision-grade"))
        );
    }

    #[test]
    fn local_git_checkout_preserves_head_diff_status_and_untracked_hashes() {
        let directory = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .args(["init", "--quiet"])
                .current_dir(directory.path())
                .status()
                .unwrap()
                .success()
        );
        let tracked = directory.path().join("tracked.txt");
        fs::write(&tracked, b"tracked-v1").unwrap();
        assert!(
            Command::new("git")
                .args(["add", "tracked.txt"])
                .current_dir(directory.path())
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .args([
                    "-c",
                    "user.name=Frankensearch Test",
                    "-c",
                    "user.email=frankensearch-test@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "fixture",
                ])
                .current_dir(directory.path())
                .status()
                .unwrap()
                .success()
        );
        fs::write(&tracked, b"tracked-v2").unwrap();
        let untracked = directory.path().join("untracked.txt");
        fs::write(&untracked, b"untracked").unwrap();

        match workspace_git_provenance(directory.path()).unwrap() {
            WorkspaceGitProvenance::Available {
                head,
                status_porcelain,
                diff_sha256,
                untracked_file_sha256,
                ..
            } => {
                assert_eq!(head.len(), 40);
                assert!(status_porcelain.contains("tracked.txt"));
                assert!(status_porcelain.contains("?? untracked.txt"));
                assert!(is_sha256_hex(&diff_sha256));
                assert_eq!(
                    untracked_file_sha256.get("untracked.txt"),
                    Some(&sha256_file(&untracked).unwrap())
                );
            }
            WorkspaceGitProvenance::Unavailable { .. } => {
                panic!("temporary Git checkout was reported unavailable");
            }
        }
    }

    #[test]
    fn duplicate_ef_cells_are_rejected() {
        let mut config = Config::smoke();
        config.ef_search = vec![40, 40];
        assert!(config.validate().is_err());
    }

    #[test]
    fn distributions_retain_raw_samples_and_suppress_unsupported_tails() {
        let distribution = distribution(&[3, 1, 2]).unwrap();
        assert_eq!(distribution.samples_ns, vec![3, 1, 2]);
        assert_eq!(distribution.p50_ns, 2);
        assert_eq!(distribution.p95_ns, None);
        assert_eq!(distribution.p99_ns, None);
    }

    #[test]
    fn paired_bootstrap_identity_and_null_gate_are_exact() {
        let estimate = paired_ratio(&[10, 20, 30], &[10, 20, 30], 7).unwrap();
        assert_f64_close(estimate.median, 1.0);
        assert_f64_close(estimate.bootstrap_median_95_low, 1.0);
        assert_f64_close(estimate.bootstrap_median_95_high, 1.0);
        assert!(null_control_within(&estimate, 0.05));

        let biased = PairedRatio {
            count: 10,
            geometric_mean: 1.06,
            median: 1.06,
            bootstrap_median_95_low: 1.05,
            bootstrap_median_95_high: 1.08,
        };
        assert!(!null_control_within(&biased, 0.10));
    }

    #[test]
    fn query_ratio_aggregates_passes_before_bootstrap() {
        let estimate =
            paired_query_ratio(&[100, 200, 110, 220], &[100, 100, 100, 100], 2, 2, 11).unwrap();
        assert_eq!(estimate.count, 2);
        assert_f64_close(estimate.median, 2.2);
    }

    #[test]
    fn hierarchical_bootstrap_preserves_identical_quality() {
        let graphs = vec![vec![1.0; 4], vec![1.0; 4]];
        let (low, high) = hierarchical_mean_ci(&graphs, 13).unwrap();
        assert_f64_close(low, 1.0);
        assert_f64_close(high, 1.0);
        let delta = replicated_quality_delta(&graphs, &graphs, 17).unwrap();
        assert_f64_close(delta.mean, 0.0);
        assert_f64_close(delta.hierarchical_mean_95_low, 0.0);
        assert_f64_close(delta.hierarchical_mean_95_high, 0.0);
    }

    #[test]
    fn paired_hierarchical_bootstrap_preserves_a_constant_within_cell_delta() {
        let baseline = vec![vec![0.1, 0.8, 0.4], vec![0.8, 0.1, 0.7]];
        let candidate: Vec<Vec<f64>> = baseline
            .iter()
            .map(|graph| graph.iter().map(|recall| recall + 0.05).collect())
            .collect();
        let (low, high) = hierarchical_delta_ci(&candidate, &baseline, 29).unwrap();
        assert!((low - 0.05).abs() < 1.0e-12);
        assert!((high - 0.05).abs() < 1.0e-12);
        assert!(hierarchical_delta_ci(&candidate[..1], &baseline, 31).is_err());
    }

    #[test]
    fn topology_gate_rejects_edgeless_and_wrong_layer_graphs() {
        let mut topology = valid_topology(3);
        let mut violations = BTreeSet::new();
        collect_topology_violations(&topology, 3, "valid", &mut violations);
        assert!(violations.is_empty());

        topology.layer0_zero_degree_nodes = 3;
        topology.directed_layer0_edges = 0;
        topology.weak_component_count = 3;
        topology.largest_weak_component = 1;
        topology.invalid_target_level_edges = 1;
        topology.reachable_from_actual_entry = Some(1);
        collect_topology_violations(&topology, 3, "invalid", &mut violations);
        assert!(violations.iter().any(|item| item.contains("actual entry")));
        assert!(
            violations
                .iter()
                .any(|item| item.contains("invalid_target_level_edges"))
        );
    }

    #[test]
    fn topology_gate_uses_the_observed_actual_entry_not_every_max_level_node() {
        let mut topology = valid_topology(4);
        topology.max_level_nodes = 2;
        topology.minimum_reachable_from_max_level = 1;
        topology.maximum_reachable_from_max_level = 4;
        let mut violations = BTreeSet::new();
        collect_topology_violations(&topology, 4, "candidate", &mut violations);
        assert!(
            violations.is_empty(),
            "a non-entry maximum-level node need not reach the entire graph: {violations:?}"
        );

        topology.reachable_from_actual_entry = Some(3);
        collect_topology_violations(&topology, 4, "candidate", &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("actual entry")),
            "{violations:?}"
        );
    }

    #[test]
    fn topology_gate_mirrors_internal_identity_and_neighborhood_table_laws() {
        let cases: [TopologyMutation; 3] = [
            (
                "duplicate_internal_point_ids",
                |topology: &mut TopologyMetrics| topology.duplicate_internal_point_ids = 1,
            ),
            ("negative_internal_slots", |topology| {
                topology.negative_internal_slots = 1;
            }),
            ("truncated_neighborhood_tables", |topology| {
                topology.truncated_neighborhood_tables = 1;
            }),
        ];
        for (field, mutate) in cases {
            let mut topology = valid_topology(4);
            mutate(&mut topology);
            let mut violations = BTreeSet::new();
            collect_topology_violations(&topology, 4, field, &mut violations);
            assert!(
                violations.iter().any(|violation| violation.contains(field)),
                "{field}: {violations:?}"
            );
        }
    }

    #[test]
    fn topology_gate_does_not_invent_an_entry_for_the_uninstrumented_baseline() {
        let mut topology = valid_topology(4);
        topology.actual_entry_api_available = false;
        topology.actual_entry_origin = None;
        topology.actual_entry_identity_valid = None;
        topology.actual_entry_is_max_level = None;
        topology.reachable_from_actual_entry = None;
        let mut violations = BTreeSet::new();
        collect_topology_violations(&topology, 4, "baseline", &mut violations);
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn lock_attestation_requires_registry_checksum_and_exact_git_revision() {
        let lock = format!(
            r#"
[[package]]
name = "hnsw_rs"
version = "0.3.4"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "43a5258f079b97bf2e8311ff9579e903c899dcbac0d9a138d62e9a066778bd07"

[[package]]
name = "hnsw_rs"
version = "0.3.4"
source = "git+https://github.com/Dicklesworthstone/hnswlib-rs?rev={CANDIDATE_REV}#{CANDIDATE_REV}"
"#
        );
        assert_eq!(resolved_hnsw_packages(&lock).unwrap().len(), 2);
        assert!(resolved_hnsw_packages(&lock.replace(CANDIDATE_REV, "deadbeef")).is_err());
    }

    #[test]
    fn corpus_manifest_binds_model_corpus_shape_and_bytes() {
        let slab: Vec<_> = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let manifest = CorpusSlabManifest {
            schema: "frankensearch-index.hnsw-corpus-manifest.v1".to_owned(),
            slab_sha256: sha256_bytes(&slab),
            model_id: "test-model".to_owned(),
            model_revision: "test-revision".to_owned(),
            corpus_id: "test-corpus".to_owned(),
            corpus_revision: "test-corpus-revision".to_owned(),
            source_sha256: "0".repeat(64),
            dimension: 2,
            rows: 2,
            dtype: "f32".to_owned(),
            byte_order: "little".to_owned(),
            normalization: "l2_unit".to_owned(),
        };
        validate_corpus_manifest(&manifest, None, None, &slab, 2, 2).unwrap();
        assert!(validate_corpus_manifest(&manifest, None, None, &slab, 4, 1).is_err());
    }

    #[test]
    fn source_manifest_binds_real_input_files_and_model_identity() {
        let directory = tempfile::tempdir().unwrap();
        let corpus = directory.path().join("corpus.txt");
        let model = directory.path().join("model.bin");
        fs::write(&corpus, b"real corpus bytes").unwrap();
        fs::write(&model, b"real model bytes").unwrap();
        let source_path = directory.path().join("source.json");
        let source = CorpusSourceManifest {
            schema: "frankensearch-index.hnsw-corpus-source-manifest.v1".to_owned(),
            model_id: "test-model".to_owned(),
            model_revision: "test-model-revision".to_owned(),
            corpus_id: "test-corpus".to_owned(),
            corpus_revision: "test-corpus-revision".to_owned(),
            dimension: 2,
            rows: 2,
            generator_id: "fixture-generator".to_owned(),
            generator_revision: "generator-revision".to_owned(),
            generator_command: vec!["fixture-generator".to_owned(), "--frozen".to_owned()],
            inputs: vec![
                CorpusSourceInput {
                    role: "corpus_source".to_owned(),
                    path: PathBuf::from("corpus.txt"),
                    sha256: sha256_file(&corpus).unwrap(),
                },
                CorpusSourceInput {
                    role: "embedding_model".to_owned(),
                    path: PathBuf::from("model.bin"),
                    sha256: sha256_file(&model).unwrap(),
                },
            ],
        };
        fs::write(&source_path, serde_json::to_vec(&source).unwrap()).unwrap();
        let slab: Vec<_> = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let manifest = CorpusSlabManifest {
            schema: "frankensearch-index.hnsw-corpus-manifest.v1".to_owned(),
            slab_sha256: sha256_bytes(&slab),
            model_id: source.model_id.clone(),
            model_revision: source.model_revision.clone(),
            corpus_id: source.corpus_id.clone(),
            corpus_revision: source.corpus_revision.clone(),
            source_sha256: sha256_file(&source_path).unwrap(),
            dimension: 2,
            rows: 2,
            dtype: "f32".to_owned(),
            byte_order: "little".to_owned(),
            normalization: "l2_unit".to_owned(),
        };
        validate_corpus_manifest(&manifest, Some(&source), Some(&source_path), &slab, 2, 2)
            .unwrap();

        fs::write(&model, b"tampered model bytes").unwrap();
        assert!(
            validate_corpus_manifest(&manifest, Some(&source), Some(&source_path), &slab, 2, 2,)
                .is_err()
        );
    }

    #[test]
    fn metric_truth_table_never_allows_an_undecided_or_losing_cell() {
        let mut reasons = Vec::new();
        assert_eq!(
            finalize_performance_status(true, false, true, &mut reasons),
            PerformanceClaimStatus::Quarantine
        );
        assert_eq!(
            finalize_performance_status(true, true, false, &mut reasons),
            PerformanceClaimStatus::Block
        );
        assert_eq!(
            finalize_performance_status(true, false, false, &mut reasons),
            PerformanceClaimStatus::Allow
        );
        assert_eq!(
            finalize_performance_status(false, false, false, &mut reasons),
            PerformanceClaimStatus::NoClaim
        );
    }

    #[test]
    fn material_point_regression_with_crossing_ci_is_undecidable() {
        let estimate = test_ratio(1.10, 0.99, 1.20);
        let null = test_ratio(1.0, 0.99, 1.01);
        let mut reasons = Vec::new();
        assert_eq!(
            classify_timing_ratio("regression", &estimate, [&null, &null], &mut reasons),
            MetricDecision::Undecidable
        );
        assert_eq!(
            classify_safety_ratio("missing rss", None, &mut reasons),
            MetricDecision::Undecidable
        );
    }

    #[test]
    fn release_perf_receipt_requires_cargo_controlling_values() {
        let valid = BuildAttestation {
            profile_directory: "release-perf".to_owned(),
            profile_family: "release".to_owned(),
            opt_level: "3".to_owned(),
            debug_info: "true".to_owned(),
            lto: "thin".to_owned(),
            codegen_units: "1".to_owned(),
            profile_opt_level: "3".to_owned(),
            rustc_vv_sha256: "a".repeat(64),
            rustflags_sha256: "b".repeat(64),
            host: "host".to_owned(),
            target: "target".to_owned(),
            target_features: vec![],
            candidate_source_sha256: "c".repeat(64),
            baseline_source_sha256: "d".repeat(64),
            workspace_source_receipt: test_workspace_source_receipt(),
        };
        assert!(valid.is_release_perf());
        let mut wrong_profile = valid.clone();
        wrong_profile.profile_directory = "release".to_owned();
        assert!(!wrong_profile.is_release_perf());
        let mut missing_lto = valid;
        missing_lto.lto = "<unset>".to_owned();
        assert!(!missing_lto.is_release_perf());
    }

    #[test]
    fn sha256_validation_rejects_uppercase_and_wrong_lengths() {
        assert!(is_sha256_hex(&"a".repeat(64)));
        assert!(!is_sha256_hex(&"A".repeat(64)));
        assert!(!is_sha256_hex(&"a".repeat(63)));
    }

    fn valid_topology(points: usize) -> TopologyMetrics {
        TopologyMetrics {
            points_seen: points,
            unique_origin_ids: points,
            duplicate_origin_ids: 0,
            out_of_range_origin_ids: 0,
            duplicate_internal_point_ids: 0,
            negative_internal_slots: 0,
            truncated_neighborhood_tables: 0,
            invalid_above_level_edges: 0,
            invalid_target_level_edges: 0,
            out_of_range_edges: 0,
            missing_target_edges: 0,
            mismatched_target_point_ids: 0,
            non_finite_edge_distances: 0,
            duplicate_edges: 0,
            self_edges: 0,
            layer0_zero_degree_nodes: 0,
            directed_layer0_edges: points.saturating_mul(2),
            reciprocal_layer0_edges: points.saturating_mul(2),
            weak_component_count: usize::from(points != 0),
            largest_weak_component: points,
            max_level: usize::from(points > 1),
            max_level_nodes: usize::from(points != 0),
            minimum_reachable_from_max_level: points,
            maximum_reachable_from_max_level: points,
            actual_entry_api_available: true,
            actual_entry_origin: (points != 0).then_some(0),
            actual_entry_identity_valid: (points != 0).then_some(true),
            actual_entry_is_max_level: (points != 0).then_some(true),
            reachable_from_actual_entry: (points != 0).then_some(points),
        }
    }

    fn test_ratio(median: f64, low: f64, high: f64) -> PairedRatio {
        PairedRatio {
            count: 10,
            geometric_mean: median,
            median,
            bootstrap_median_95_low: low,
            bootstrap_median_95_high: high,
        }
    }
}
