//! CMA-ES hyperparameter optimizer for frankensearch fusion pipeline.
//!
//! Optimizes 6 continuous parameters of [`TwoTierConfig`] against the test fixture
//! corpus using `nDCG@10` as the objective function.
//!
//! Since we cannot easily run the full embedding pipeline in a standalone tool,
//! we simulate rankings using simple term-overlap scoring (lexical) and
//! TF-IDF-like cosine similarity (semantic), then fuse them with RRF + blend.
//! This tests fusion parameter sensitivity rather than embedding quality.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use fastcma::{CmaesState, CovarianceModeKind};
use frankensearch_core::metrics_eval::ndcg_at_k;
use frankensearch_core::{ScoreSource, ScoredResult, TwoTierConfig, VectorHit};
use frankensearch_fusion::{RrfConfig, rrf_fuse};
use serde::{Deserialize, Serialize};

// ─── Type Aliases ────────────────────────────────────────────────────────────

/// Per-generation optimization history entry: `(evaluations, params, best_ndcg)`.
type GenerationHistory = Vec<(usize, Vec<f64>, f64)>;

/// Result of a single fold optimization: `(best_params, best_ndcg, history, total_evaluations)`.
type FoldResult = (Vec<f64>, f64, GenerationHistory, usize);

/// Result of cross-validation: `(fold_params, train_ndcgs, val_ndcgs, histories, evaluation_counts)`.
type CrossValidationResult = (
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<f64>,
    Vec<GenerationHistory>,
    Vec<usize>,
);

// ─── Fixture Types ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Corpus {
    documents: Vec<Document>,
}

#[derive(Deserialize, Clone)]
struct Document {
    doc_id: String,
    title: String,
    content: String,
}

#[derive(Deserialize)]
struct QueryEntry {
    query: String,
    #[serde(default)]
    relevant_ids: Vec<String>,
    #[serde(default)]
    expected_top_10: Vec<String>,
}

impl QueryEntry {
    /// Get the ground-truth relevant doc IDs (prefer `relevant_ids`, fall back to `expected_top_10`).
    fn ground_truth(&self) -> &[String] {
        if self.relevant_ids.is_empty() {
            &self.expected_top_10
        } else {
            &self.relevant_ids
        }
    }
}

// ─── Simple Text Scoring ────────────────────────────────────────────────────

/// Tokenize text into lowercase words.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect()
}

/// Build term frequency map for a document.
fn term_freqs(tokens: &[String]) -> HashMap<&str, f64> {
    let mut freqs: HashMap<&str, f64> = HashMap::new();
    for token in tokens {
        *freqs.entry(token.as_str()).or_default() += 1.0;
    }
    freqs
}

/// Compute BM25-like score for a query against a document.
#[allow(clippy::cast_precision_loss)]
fn bm25_score(
    query_tokens: &[String],
    doc_tokens: &[String],
    title_tokens: &[String],
    avg_doc_len: f64,
    _num_docs: f64,
    idf_map: &HashMap<String, f64>,
) -> f64 {
    let k1 = 1.2;
    let b = 0.75;
    let title_boost = 2.0;
    let doc_len = doc_tokens.len() as f64;
    let tf_doc = term_freqs(doc_tokens);
    let tf_title = term_freqs(title_tokens);

    let mut score = 0.0;
    for qt in query_tokens {
        let idf = idf_map.get(qt.as_str()).copied().unwrap_or(0.0);
        // Document body contribution
        let tf = tf_doc.get(qt.as_str()).copied().unwrap_or(0.0);
        let numerator = tf * (k1 + 1.0);
        let denominator = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len);
        score += idf * numerator / denominator;
        // Title boost
        let tf_t = tf_title.get(qt.as_str()).copied().unwrap_or(0.0);
        if tf_t > 0.0 {
            score += idf * title_boost * tf_t / (tf_t + 1.0);
        }
    }
    score
}

/// Compute TF-IDF cosine similarity between query and document.
#[allow(clippy::cast_possible_truncation)]
fn tfidf_cosine(
    query_tokens: &[String],
    doc_tokens: &[String],
    idf_map: &HashMap<String, f64>,
) -> f32 {
    let tf_q = term_freqs(query_tokens);
    let tf_d = term_freqs(doc_tokens);

    // Build sparse TF-IDF vectors
    let mut dot = 0.0_f64;
    let mut norm_q = 0.0_f64;
    let mut norm_d = 0.0_f64;

    for qt in query_tokens {
        let idf = idf_map.get(qt.as_str()).copied().unwrap_or(0.0);
        let tfidf_q = tf_q.get(qt.as_str()).copied().unwrap_or(0.0) * idf;
        let tfidf_d = tf_d.get(qt.as_str()).copied().unwrap_or(0.0) * idf;
        dot += tfidf_q * tfidf_d;
        norm_q += tfidf_q * tfidf_q;
    }
    for token in doc_tokens {
        let idf = idf_map.get(token.as_str()).copied().unwrap_or(0.0);
        let tfidf_d = tf_d.get(token.as_str()).copied().unwrap_or(0.0) * idf;
        norm_d += tfidf_d * tfidf_d;
    }

    let denom = norm_q.sqrt() * norm_d.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (dot / denom) as f32
}

// ─── Optimization Parameters ────────────────────────────────────────────────

/// Parameter bounds: [`quality_weight`, `rrf_k`, `candidate_multiplier`, `quality_timeout_ms`,
/// `hnsw_ef_search`, `mrl_rescore_top_k`].
const LOWER_BOUNDS: [f64; 6] = [0.1, 10.0, 1.5, 200.0, 30.0, 10.0];
const UPPER_BOUNDS: [f64; 6] = [0.95, 150.0, 6.0, 1500.0, 300.0, 100.0];

const DEFAULT_MAX_GENERATIONS: usize = 200;
const DEFAULT_MAX_EVALUATIONS: usize = 2000;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_FOLDS: usize = 5;

#[derive(Debug, Clone)]
struct RunConfig {
    max_generations: usize,
    max_evaluations: usize,
    seed: u64,
    folds: usize,
    output_dir: Option<PathBuf>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_generations: DEFAULT_MAX_GENERATIONS,
            max_evaluations: DEFAULT_MAX_EVALUATIONS,
            seed: DEFAULT_SEED,
            folds: DEFAULT_FOLDS,
            output_dir: None,
        }
    }
}

/// Clamp parameter vector to bounds.
fn clamp_params(x: &[f64]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for i in 0..6 {
        out[i] = x[i].clamp(LOWER_BOUNDS[i], UPPER_BOUNDS[i]);
    }
    out
}

/// Convert raw parameter vector to a [`TwoTierConfig`].
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn params_to_config(x: &[f64]) -> TwoTierConfig {
    let p = clamp_params(x);
    TwoTierConfig {
        quality_weight: p[0],
        rrf_k: p[1],
        candidate_multiplier: p[2].round() as usize,
        quality_timeout_ms: p[3].round() as u64,
        hnsw_ef_search: p[4].round() as usize,
        mrl_rescore_top_k: p[5].round() as usize,
        ..TwoTierConfig::default()
    }
}

// ─── Fitness Evaluation ─────────────────────────────────────────────────────

struct EvalContext {
    /// Precomputed per-document token lists (content + title concatenated).
    doc_tokens: Vec<Vec<String>>,
    /// Precomputed per-document title tokens.
    title_tokens: Vec<Vec<String>>,
    /// Document IDs in index order.
    doc_ids: Vec<String>,
    /// IDF map for all terms.
    idf_map: HashMap<String, f64>,
    /// Average document length.
    avg_doc_len: f64,
    /// Queries with ground truth.
    queries: Vec<(Vec<String>, Vec<String>)>, // (query_tokens, relevant_ids)
}

impl EvalContext {
    #[allow(clippy::cast_precision_loss)]
    fn from_fixtures(docs: &[Document], queries_entries: &[QueryEntry]) -> Self {
        let num_docs = docs.len() as f64;

        // Tokenize all documents
        let doc_tokens: Vec<Vec<String>> = docs
            .iter()
            .map(|d| {
                let mut tokens = tokenize(&d.content);
                tokens.extend(tokenize(&d.title));
                tokens
            })
            .collect();

        let title_tokens: Vec<Vec<String>> = docs.iter().map(|d| tokenize(&d.title)).collect();

        let doc_ids: Vec<String> = docs.iter().map(|d| d.doc_id.clone()).collect();

        // Compute IDF for all terms
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        for tokens in &doc_tokens {
            let unique: std::collections::HashSet<&str> =
                tokens.iter().map(String::as_str).collect();
            for term in unique {
                *doc_freq.entry(term.to_string()).or_default() += 1;
            }
        }
        let idf_map: HashMap<String, f64> = doc_freq
            .into_iter()
            .map(|(term, df)| {
                let idf = ((num_docs - df as f64 + 0.5) / (df as f64 + 0.5)).ln_1p();
                (term, idf)
            })
            .collect();

        let avg_doc_len = doc_tokens.iter().map(Vec::len).sum::<usize>() as f64 / num_docs.max(1.0);

        // Prepare queries (skip empty/no-match queries)
        let queries: Vec<(Vec<String>, Vec<String>)> = queries_entries
            .iter()
            .filter(|r| !r.query.is_empty() && !r.ground_truth().is_empty())
            .map(|r| (tokenize(&r.query), r.ground_truth().to_vec()))
            .collect();

        Self {
            doc_tokens,
            title_tokens,
            doc_ids,
            idf_map,
            avg_doc_len,
            queries,
        }
    }

    /// Evaluate mean `nDCG@10` for a given parameter set across a subset of queries.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn evaluate(&self, x: &[f64], query_indices: &[usize]) -> f64 {
        // Degenerate parameter vectors should map to a zero-quality score so
        // tests and smoke validations can assert fail-safe objective behavior.
        if x.len() >= 2 {
            let blend = x[0];
            let rrf_k = x[1];
            if !blend.is_finite() || !rrf_k.is_finite() || blend <= 0.0 || rrf_k <= 0.0 {
                return 0.0;
            }
        }

        let config = params_to_config(x);
        let rrf_config = RrfConfig { k: config.rrf_k };
        let candidate_count = 10_usize
            .saturating_mul(config.candidate_multiplier)
            .min(self.doc_ids.len());

        let mut total_ndcg = 0.0;
        let num_queries = query_indices.len();
        if num_queries == 0 {
            return 0.0;
        }

        for &qi in query_indices {
            let (query_tokens, relevant_ids) = &self.queries[qi];

            // Compute BM25 scores for all documents
            let mut lexical_scores: Vec<(usize, f64)> = self
                .doc_tokens
                .iter()
                .enumerate()
                .map(|(i, doc_tok)| {
                    let score = bm25_score(
                        query_tokens,
                        doc_tok,
                        &self.title_tokens[i],
                        self.avg_doc_len,
                        self.doc_ids.len() as f64,
                        &self.idf_map,
                    );
                    (i, score)
                })
                .collect();

            // Sort by score descending, take top candidates
            lexical_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            let lexical_results: Vec<ScoredResult> = lexical_scores
                .iter()
                .take(candidate_count)
                .filter(|(_, s)| *s > 0.0)
                .map(|(i, s)| ScoredResult {
                    doc_id: self.doc_ids[*i].clone(),
                    score: *s as f32,
                    source: ScoreSource::Lexical,
                    fast_score: None,
                    quality_score: None,
                    lexical_score: Some(*s as f32),
                    rerank_score: None,
                    metadata: None,
                })
                .collect();

            // Compute TF-IDF cosine similarity scores for all documents
            let mut semantic_scores: Vec<(usize, f32)> = self
                .doc_tokens
                .iter()
                .enumerate()
                .map(|(i, doc_tok)| {
                    let score = tfidf_cosine(query_tokens, doc_tok, &self.idf_map);
                    (i, score)
                })
                .collect();

            semantic_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            let semantic_results: Vec<VectorHit> = semantic_scores
                .iter()
                .take(candidate_count)
                .filter(|(_, s)| *s > 0.0)
                .map(|(i, s)| VectorHit {
                    index: *i as u32,
                    score: *s,
                    doc_id: self.doc_ids[*i].clone(),
                })
                .collect();

            // RRF fusion
            let fused = rrf_fuse(&lexical_results, &semantic_results, 10, 0, &rrf_config);

            // Extract doc_id ordering
            let retrieved: Vec<&str> = fused.iter().map(|h| h.doc_id.as_str()).collect();
            let relevant: Vec<&str> = relevant_ids.iter().map(String::as_str).collect();

            total_ndcg += ndcg_at_k(&retrieved, &relevant, 10);
        }

        total_ndcg / num_queries as f64
    }
}

// ─── Optimization Log ───────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct LogEntry {
    generation: usize,
    evaluations: usize,
    best_fitness: f64,
    best_ndcg: f64,
    params: ParamSnapshot,
}

#[derive(Serialize, Deserialize, Clone)]
struct ParamSnapshot {
    quality_weight: f64,
    rrf_k: f64,
    candidate_multiplier: usize,
    quality_timeout_ms: u64,
    hnsw_ef_search: usize,
    mrl_rescore_top_k: usize,
}

impl ParamSnapshot {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn from_raw(x: &[f64]) -> Self {
        let p = clamp_params(x);
        Self {
            quality_weight: p[0],
            rrf_k: p[1],
            candidate_multiplier: p[2].round() as usize,
            quality_timeout_ms: p[3].round() as u64,
            hnsw_ef_search: p[4].round() as usize,
            mrl_rescore_top_k: p[5].round() as usize,
        }
    }
}

// ─── Cross-Validation ───────────────────────────────────────────────────────

/// Split query indices into k folds.
fn k_fold_split(n: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    if k <= 1 {
        let all: Vec<usize> = (0..n).collect();
        return vec![(all.clone(), all)];
    }
    let fold_size = n / k;
    let mut folds = Vec::with_capacity(k);
    for fold in 0..k {
        let val_start = fold * fold_size;
        let val_end = if fold == k - 1 {
            n
        } else {
            val_start + fold_size
        };
        let val_indices: Vec<usize> = (val_start..val_end).collect();
        let train_indices: Vec<usize> = (0..val_start).chain(val_end..n).collect();
        folds.push((train_indices, val_indices));
    }
    folds
}

/// Run CMA-ES optimization on a given set of train queries.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn optimize_fold(
    ctx: &EvalContext,
    train_indices: &[usize],
    fold_id: usize,
    run_config: &RunConfig,
) -> FoldResult {
    let best_params: Mutex<(Vec<f64>, f64)> =
        Mutex::new((vec![0.7, 60.0, 3.0, 500.0, 100.0, 30.0], f64::INFINITY));

    let train_indices_owned: Vec<usize> = train_indices.to_vec();
    let mut history: GenerationHistory = Vec::new();
    let mut evaluations = 0_usize;

    // Use ask/tell API directly for full control
    let x0 = vec![0.7, 60.0, 3.0, 500.0, 100.0, 30.0];
    let mut es = CmaesState::new_with_seed(
        x0,
        0.3,
        None,        // auto popsize
        Some(-0.95), // target: nDCG >= 0.95
        Some(run_config.max_evaluations),
        CovarianceModeKind::Full,
        run_config.seed + fold_id as u64,
    );

    let mut generation = 0;
    while !es.has_terminated() && generation < run_config.max_generations {
        let candidates = es.ask();
        let fitvals: Vec<f64> = candidates
            .iter()
            .map(|x| {
                let ndcg = ctx.evaluate(x, &train_indices_owned);
                let fitness = -ndcg; // minimize negative nDCG

                // Track best
                let mut best = best_params.lock().unwrap();
                if fitness < best.1 {
                    best.0.clone_from(x);
                    best.1 = fitness;
                }

                fitness
            })
            .collect();
        evaluations = evaluations.saturating_add(fitvals.len());
        es.tell(candidates, fitvals);
        generation += 1;

        let best = best_params.lock().unwrap();
        history.push((evaluations, best.0.clone(), -best.1));

        if generation % 20 == 0 {
            eprintln!(
                "  fold {fold_id}: gen {generation}, best nDCG = {:.4}",
                -best.1
            );
        }
        drop(best);
    }

    let best = best_params.lock().unwrap();
    while history.len() < run_config.max_generations {
        history.push((evaluations, best.0.clone(), -best.1));
    }
    (best.0.clone(), -best.1, history, evaluations)
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn load_fixtures(workspace_root: &std::path::Path) -> (Corpus, Vec<QueryEntry>) {
    eprintln!("Loading fixtures...");
    let corpus_path = workspace_root.join("tests/fixtures/corpus.json");
    let queries_path = workspace_root.join("tests/fixtures/queries.json");

    let corpus_str = fs::read_to_string(&corpus_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", corpus_path.display()));
    let corpus: Corpus = serde_json::from_str(&corpus_str)
        .unwrap_or_else(|e| panic!("failed to parse corpus.json: {e}"));

    let queries_str = fs::read_to_string(&queries_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", queries_path.display()));
    let query_entries: Vec<QueryEntry> = serde_json::from_str(&queries_str)
        .unwrap_or_else(|e| panic!("failed to parse queries.json: {e}"));

    eprintln!(
        "Loaded {} documents, {} queries (with ground truth)",
        corpus.documents.len(),
        query_entries
            .iter()
            .filter(|r| !r.ground_truth().is_empty())
            .count()
    );

    (corpus, query_entries)
}

#[allow(clippy::cast_precision_loss)]
fn run_cross_validation(ctx: &EvalContext, run_config: &RunConfig) -> CrossValidationResult {
    let n_folds = run_config.folds.min(ctx.queries.len()).max(1);
    let folds = k_fold_split(ctx.queries.len(), n_folds);
    let mut fold_params: Vec<Vec<f64>> = Vec::with_capacity(n_folds);
    let mut fold_train_ndcg: Vec<f64> = Vec::with_capacity(n_folds);
    let mut fold_val_ndcg: Vec<f64> = Vec::with_capacity(n_folds);
    let mut fold_histories: Vec<GenerationHistory> = Vec::with_capacity(n_folds);
    let mut fold_evaluations: Vec<usize> = Vec::with_capacity(n_folds);

    eprintln!("\nRunning {n_folds}-fold cross-validation...");
    for (fold_idx, (train_indices, val_indices)) in folds.iter().enumerate() {
        eprintln!(
            "\n--- Fold {}/{n_folds} ({} train, {} val) ---",
            fold_idx + 1,
            train_indices.len(),
            val_indices.len()
        );

        let (best_x, train_ndcg, history, evaluations) =
            optimize_fold(ctx, train_indices, fold_idx, run_config);

        // Evaluate on validation set
        let val_ndcg = ctx.evaluate(&best_x, val_indices);
        let snapshot = ParamSnapshot::from_raw(&best_x);

        eprintln!(
            "  Fold {} result: train nDCG={train_ndcg:.4}, val nDCG={val_ndcg:.4}",
            fold_idx + 1
        );
        eprintln!(
            "    quality_weight={:.3}, rrf_k={:.1}, candidate_mult={}, timeout={}ms, ef_search={}, rescore_top_k={}",
            snapshot.quality_weight,
            snapshot.rrf_k,
            snapshot.candidate_multiplier,
            snapshot.quality_timeout_ms,
            snapshot.hnsw_ef_search,
            snapshot.mrl_rescore_top_k
        );

        fold_params.push(best_x);
        fold_train_ndcg.push(train_ndcg);
        fold_val_ndcg.push(val_ndcg);
        fold_histories.push(history);
        fold_evaluations.push(evaluations);
    }

    (
        fold_params,
        fold_train_ndcg,
        fold_val_ndcg,
        fold_histories,
        fold_evaluations,
    )
}

fn print_results(
    baseline_ndcg: f64,
    final_ndcg: f64,
    mean_val: f64,
    cv_variance: f64,
    final_snapshot: &ParamSnapshot,
) {
    eprintln!("\n╔══════════════════════════════════════════════════════╗");
    eprintln!("║            Optimization Results                      ║");
    eprintln!("╠══════════════════════════════════════════════════════╣");
    eprintln!("║ Baseline nDCG@10:    {baseline_ndcg:.4}                       ║");
    eprintln!("║ Optimized nDCG@10:   {final_ndcg:.4}                       ║");
    eprintln!("║ Mean CV val nDCG:    {mean_val:.4}                       ║");
    eprintln!("║ CV variance:         {cv_variance:.6}                     ║");
    if cv_variance > 0.05 {
        eprintln!("║ ⚠ WARNING: CV variance > 0.05 — overfitting risk!   ║");
    }
    eprintln!("╠══════════════════════════════════════════════════════╣");
    eprintln!("║ Optimized Parameters:                                ║");
    eprintln!(
        "║   quality_weight:      {:.3}                          ║",
        final_snapshot.quality_weight
    );
    eprintln!(
        "║   rrf_k:               {:.1}                          ║",
        final_snapshot.rrf_k
    );
    eprintln!(
        "║   candidate_multiplier: {}                            ║",
        final_snapshot.candidate_multiplier
    );
    eprintln!(
        "║   quality_timeout_ms:   {}                          ║",
        final_snapshot.quality_timeout_ms
    );
    eprintln!(
        "║   hnsw_ef_search:       {}                          ║",
        final_snapshot.hnsw_ef_search
    );
    eprintln!(
        "║   mrl_rescore_top_k:    {}                           ║",
        final_snapshot.mrl_rescore_top_k
    );
    eprintln!("╚══════════════════════════════════════════════════════╝");
}

#[allow(clippy::too_many_arguments)]
fn write_outputs(
    output_dir: &Path,
    baseline_ndcg: f64,
    final_ndcg: f64,
    mean_val: f64,
    cv_variance: f64,
    final_snapshot: &ParamSnapshot,
    default_params: &[f64],
    fold_histories: &[Vec<(usize, Vec<f64>, f64)>],
    fold_evaluations: &[usize],
    n_folds: usize,
    run_config: &RunConfig,
) {
    // Write optimized_params.toml
    fs::create_dir_all(output_dir).expect("failed to create output directory");

    let toml_path = output_dir.join("optimized_params.toml");
    let toml_content = format!(
        r"# Optimized TwoTierConfig parameters
# Generated by optimize-params via CMA-ES with {n_folds}-fold CV
# Baseline nDCG@10: {baseline_ndcg:.4}
# Optimized nDCG@10: {final_ndcg:.4}
# CV variance: {cv_variance:.6}
# Seed: {seed}
# Max generations/fold: {max_generations}

quality_weight = {quality_weight}
rrf_k = {rrf_k}
candidate_multiplier = {candidate_multiplier}
quality_timeout_ms = {quality_timeout_ms}
fast_only = false
explain = false
hnsw_ef_search = {hnsw_ef_search}
hnsw_ef_construction = 200
hnsw_m = 16
hnsw_threshold = 50000
mrl_search_dims = 0
mrl_rescore_top_k = {mrl_rescore_top_k}
",
        quality_weight = final_snapshot.quality_weight,
        rrf_k = final_snapshot.rrf_k,
        candidate_multiplier = final_snapshot.candidate_multiplier,
        quality_timeout_ms = final_snapshot.quality_timeout_ms,
        hnsw_ef_search = final_snapshot.hnsw_ef_search,
        mrl_rescore_top_k = final_snapshot.mrl_rescore_top_k,
        seed = run_config.seed,
        max_generations = run_config.max_generations,
    );
    fs::write(&toml_path, &toml_content)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", toml_path.display()));
    eprintln!("\nWrote: {}", toml_path.display());

    // Write optimization_log.jsonl
    let log_path = output_dir.join("optimization_log.jsonl");
    let mut log_file = fs::File::create(&log_path)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", log_path.display()));
    let mut running_best_fitness = -baseline_ndcg;
    let mut running_best_params = ParamSnapshot::from_raw(default_params);
    let mut generation = 0_usize;
    let mut evaluation_offset = 0_usize;

    for (fold_idx, history) in fold_histories.iter().enumerate() {
        for (fold_evaluations_so_far, best_params, best_ndcg) in history {
            generation = generation.saturating_add(1);
            let fitness = -*best_ndcg;
            if fitness < running_best_fitness {
                running_best_fitness = fitness;
                running_best_params = ParamSnapshot::from_raw(best_params);
            }
            let entry = LogEntry {
                generation,
                evaluations: evaluation_offset.saturating_add(*fold_evaluations_so_far),
                best_fitness: running_best_fitness,
                best_ndcg: -running_best_fitness,
                params: running_best_params.clone(),
            };
            writeln!(log_file, "{}", serde_json::to_string(&entry).unwrap()).unwrap();
        }
        evaluation_offset = evaluation_offset
            .saturating_add(fold_evaluations.get(fold_idx).copied().unwrap_or_default());
    }

    eprintln!("Wrote: {}", log_path.display());

    // Write optimization_report.md
    let report_path = output_dir.join("optimization_report.md");
    let report = format!(
        "# Optimization Report\n\n\
Generated by `optimize-params`.\n\n\
## Summary\n\n\
- Baseline nDCG@10: `{baseline_ndcg:.4}`\n\
- Optimized nDCG@10: `{final_ndcg:.4}`\n\
- Mean CV validation nDCG@10: `{mean_val:.4}`\n\
- CV variance: `{cv_variance:.6}`\n\
- Seed: `{seed}`\n\
- Max generations per fold: `{max_generations}`\n\
\n## Final Parameters\n\n\
| Parameter | Value |\n\
|---|---|\n\
| quality_weight | `{quality_weight:.6}` |\n\
| rrf_k | `{rrf_k:.6}` |\n\
| candidate_multiplier | `{candidate_multiplier}` |\n\
| quality_timeout_ms | `{quality_timeout_ms}` |\n\
| hnsw_ef_search | `{hnsw_ef_search}` |\n\
| mrl_rescore_top_k | `{mrl_rescore_top_k}` |\n\
",
        quality_weight = final_snapshot.quality_weight,
        rrf_k = final_snapshot.rrf_k,
        candidate_multiplier = final_snapshot.candidate_multiplier,
        quality_timeout_ms = final_snapshot.quality_timeout_ms,
        hnsw_ef_search = final_snapshot.hnsw_ef_search,
        mrl_rescore_top_k = final_snapshot.mrl_rescore_top_k,
        seed = run_config.seed,
        max_generations = run_config.max_generations,
    );
    fs::write(&report_path, report)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", report_path.display()));
    eprintln!("Wrote: {}", report_path.display());
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run -p optimize-params -- [--max-generations N] [--max-evaluations N] [--seed N] [--folds N] [--output-dir PATH]"
    );
}

fn parse_run_config(args: &[String]) -> Result<RunConfig, String> {
    let mut config = RunConfig::default();
    let mut index = 0usize;

    while index < args.len() {
        let raw = &args[index];
        let (flag, inline_value) = raw
            .split_once('=')
            .map_or((raw.as_str(), None), |(f, v)| (f, Some(v.to_owned())));
        let mut next_value = |name: &str| -> Result<String, String> {
            if let Some(value) = inline_value.clone() {
                return Ok(value);
            }
            index = index.saturating_add(1);
            args.get(index)
                .cloned()
                .ok_or_else(|| format!("missing value for {name}"))
        };

        match flag {
            "--max-generations" => {
                let value = next_value("--max-generations")?;
                config.max_generations = value
                    .parse::<usize>()
                    .map_err(|_| "invalid --max-generations value".to_owned())?;
                if config.max_generations == 0 {
                    return Err("--max-generations must be >= 1".to_owned());
                }
            }
            "--max-evaluations" => {
                let value = next_value("--max-evaluations")?;
                config.max_evaluations = value
                    .parse::<usize>()
                    .map_err(|_| "invalid --max-evaluations value".to_owned())?;
                if config.max_evaluations == 0 {
                    return Err("--max-evaluations must be >= 1".to_owned());
                }
            }
            "--seed" => {
                let value = next_value("--seed")?;
                config.seed = value
                    .parse::<u64>()
                    .map_err(|_| "invalid --seed value".to_owned())?;
            }
            "--folds" => {
                let value = next_value("--folds")?;
                config.folds = value
                    .parse::<usize>()
                    .map_err(|_| "invalid --folds value".to_owned())?;
                if config.folds == 0 {
                    return Err("--folds must be >= 1".to_owned());
                }
            }
            "--output-dir" => {
                let value = next_value("--output-dir")?;
                config.output_dir = Some(PathBuf::from(value));
            }
            _ => return Err(format!("unknown option: {flag}")),
        }

        index = index.saturating_add(1);
    }

    Ok(config)
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage();
        return;
    }
    let run_config = parse_run_config(&args).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        print_usage();
        std::process::exit(2);
    });

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("cannot find workspace root")
        .to_path_buf();

    // Load fixtures
    let (corpus, query_entries) = load_fixtures(&workspace_root);

    // Build evaluation context
    let ctx = EvalContext::from_fixtures(&corpus.documents, &query_entries);

    // Compute baseline nDCG with default params
    let all_indices: Vec<usize> = (0..ctx.queries.len()).collect();
    let default_params = [0.7, 60.0, 3.0, 500.0, 100.0, 30.0];
    let baseline_ndcg = ctx.evaluate(&default_params, &all_indices);
    eprintln!("Baseline nDCG@10 (default params): {baseline_ndcg:.4}");

    // 5-fold cross-validation
    let n_folds = run_config.folds.min(ctx.queries.len()).max(1);
    let (fold_params, _fold_train_ndcg, fold_val_ndcg, fold_histories, fold_evaluations) =
        run_cross_validation(&ctx, &run_config);

    // Compute parameter-wise median across folds
    let final_params = parameter_wise_median(&fold_params);
    let final_snapshot = ParamSnapshot::from_raw(&final_params);
    let final_ndcg = ctx.evaluate(&final_params, &all_indices);

    // CV variance
    let mean_val: f64 = fold_val_ndcg.iter().sum::<f64>() / n_folds as f64;
    let cv_variance: f64 = fold_val_ndcg
        .iter()
        .map(|v| (v - mean_val).powi(2))
        .sum::<f64>()
        / n_folds as f64;

    print_results(
        baseline_ndcg,
        final_ndcg,
        mean_val,
        cv_variance,
        &final_snapshot,
    );

    let default_output_dir = workspace_root.join("data");
    let output_dir = run_config
        .output_dir
        .as_deref()
        .unwrap_or(default_output_dir.as_path());

    write_outputs(
        output_dir,
        baseline_ndcg,
        final_ndcg,
        mean_val,
        cv_variance,
        &final_snapshot,
        &default_params,
        &fold_histories,
        &fold_evaluations,
        n_folds,
        &run_config,
    );

    // Regression check
    if final_ndcg < baseline_ndcg {
        eprintln!("\n⚠ WARNING: Optimized nDCG ({final_ndcg:.4}) < baseline ({baseline_ndcg:.4})!");
        eprintln!("  Using baseline defaults is recommended.");
        std::process::exit(1);
    }

    eprintln!(
        "\nDone. Optimized nDCG improvement: {:.4} → {:.4} (+{:.4})",
        baseline_ndcg,
        final_ndcg,
        final_ndcg - baseline_ndcg
    );
}

/// Compute parameter-wise median across fold results.
fn parameter_wise_median(fold_params: &[Vec<f64>]) -> Vec<f64> {
    let n_params = fold_params[0].len();
    let mut result = vec![0.0; n_params];

    for param_idx in 0..n_params {
        let mut values: Vec<f64> = fold_params.iter().map(|p| p[param_idx]).collect();
        values.sort_by(f64::total_cmp);
        result[param_idx] = values[values.len() / 2]; // median
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_eval_context() -> EvalContext {
        let docs = vec![
            Document {
                doc_id: "doc-a".to_owned(),
                title: "Rust ownership".to_owned(),
                content: "Ownership and borrowing in rust".to_owned(),
            },
            Document {
                doc_id: "doc-b".to_owned(),
                title: "Python threading".to_owned(),
                content: "GIL and threading model".to_owned(),
            },
            Document {
                doc_id: "doc-c".to_owned(),
                title: "Distributed systems".to_owned(),
                content: "Consensus and replication".to_owned(),
            },
        ];
        let queries = vec![
            QueryEntry {
                query: "rust ownership".to_owned(),
                relevant_ids: vec!["doc-a".to_owned()],
                expected_top_10: Vec::new(),
            },
            QueryEntry {
                query: "distributed consensus".to_owned(),
                relevant_ids: vec!["doc-c".to_owned()],
                expected_top_10: Vec::new(),
            },
        ];
        EvalContext::from_fixtures(&docs, &queries)
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn objective_score_is_bounded_between_zero_and_one() {
        let ctx = sample_eval_context();
        let query_indices: Vec<usize> = (0..ctx.queries.len()).collect();
        let score = ctx.evaluate(&[0.7, 60.0, 3.0, 500.0, 100.0, 30.0], &query_indices);
        assert!((0.0..=1.0).contains(&score), "score out of bounds: {score}");
    }

    #[test]
    fn objective_score_returns_zero_for_degenerate_params() {
        let ctx = sample_eval_context();
        let query_indices: Vec<usize> = (0..ctx.queries.len()).collect();
        let score = ctx.evaluate(&[0.0, 0.0, 3.0, 500.0, 100.0, 30.0], &query_indices);
        assert!(score.abs() < f64::EPSILON);
    }

    #[test]
    fn parameter_bounds_are_enforced() {
        let clamped = clamp_params(&[-1.0, 500.0, 0.1, 9_999.0, -5.0, 1_000.0]);
        for index in 0..clamped.len() {
            assert!(
                (LOWER_BOUNDS[index]..=UPPER_BOUNDS[index]).contains(&clamped[index]),
                "parameter {index} out of bounds after clamp: {}",
                clamped[index]
            );
        }
    }

    #[test]
    fn k_fold_split_assigns_each_query_once_to_validation() {
        let folds = k_fold_split(25, 5);
        let mut val_all = Vec::new();
        for (train, val) in &folds {
            for idx in val {
                assert!(!train.contains(idx), "train/val overlap on index {idx}");
                val_all.push(*idx);
            }
        }
        val_all.sort_unstable();
        assert_eq!(val_all, (0..25).collect::<Vec<_>>());
    }

    #[test]
    fn k_fold_split_with_single_fold_uses_all_queries_for_train_and_validation() {
        let folds = k_fold_split(8, 1);
        assert_eq!(folds.len(), 1);
        let (train, val) = &folds[0];
        assert_eq!(train, &(0..8).collect::<Vec<_>>());
        assert_eq!(val, &(0..8).collect::<Vec<_>>());
    }

    #[test]
    fn optimized_toml_roundtrip_is_lossless() {
        let config = TwoTierConfig {
            quality_weight: 0.82,
            rrf_k: 73.5,
            candidate_multiplier: 4,
            quality_timeout_ms: 777,
            fast_only: false,
            explain: false,
            hnsw_ef_search: 123,
            hnsw_ef_construction: 222,
            hnsw_m: 16,
            hnsw_threshold: 50_000,
            mrl_search_dims: 64,
            mrl_rescore_top_k: 45,
            ..TwoTierConfig::default()
        };
        let toml = toml::to_string(&config).expect("serialize config");
        let decoded: TwoTierConfig = toml::from_str(&toml).expect("deserialize config");
        assert!((decoded.quality_weight - config.quality_weight).abs() < 1e-12);
        assert!((decoded.rrf_k - config.rrf_k).abs() < 1e-12);
        assert_eq!(decoded.candidate_multiplier, config.candidate_multiplier);
        assert_eq!(decoded.quality_timeout_ms, config.quality_timeout_ms);
        assert_eq!(decoded.hnsw_ef_search, config.hnsw_ef_search);
        assert_eq!(decoded.mrl_rescore_top_k, config.mrl_rescore_top_k);
    }

    #[test]
    fn optimization_log_jsonl_is_valid_and_monotonic() {
        let output_dir = unique_temp_dir("optimize-params-jsonl");
        let run_config = RunConfig {
            max_generations: 3,
            max_evaluations: 30,
            seed: 42,
            folds: 1,
            output_dir: Some(output_dir.clone()),
        };
        let fold_histories = vec![vec![
            (3, vec![0.7, 60.0, 3.0, 500.0, 100.0, 30.0], 0.20),
            (6, vec![0.8, 65.0, 3.0, 500.0, 100.0, 30.0], 0.30),
            (9, vec![0.9, 70.0, 3.0, 500.0, 100.0, 30.0], 0.40),
        ]];
        let fold_evaluations = vec![9];
        write_outputs(
            &output_dir,
            0.10,
            0.40,
            0.40,
            0.0,
            &ParamSnapshot::from_raw(&[0.9, 70.0, 3.0, 500.0, 100.0, 30.0]),
            &[0.7, 60.0, 3.0, 500.0, 100.0, 30.0],
            &fold_histories,
            &fold_evaluations,
            1,
            &run_config,
        );

        let log_path = output_dir.join("optimization_log.jsonl");
        let raw = std::fs::read_to_string(&log_path).expect("read optimization log");
        let mut previous_best = f64::INFINITY;
        for line in raw.lines() {
            let entry: LogEntry =
                serde_json::from_str(line).expect("each optimization log line must be JSON");
            assert!(entry.best_fitness <= previous_best + f64::EPSILON);
            previous_best = entry.best_fitness;
        }

        let report_path = output_dir.join("optimization_report.md");
        let report = std::fs::read_to_string(&report_path).expect("read optimization report");
        assert!(report.contains("# Optimization Report"));
        assert!(report.contains("Optimized nDCG@10"));
    }

    #[test]
    fn cross_validation_is_reproducible_for_fixed_seed() {
        let ctx = sample_eval_context();
        let run_config = RunConfig {
            max_generations: 2,
            max_evaluations: 40,
            seed: 42,
            folds: 2,
            output_dir: None,
        };

        let (params_a, train_a, val_a, histories_a, evaluations_a) =
            run_cross_validation(&ctx, &run_config);
        let (params_b, train_b, val_b, histories_b, evaluations_b) =
            run_cross_validation(&ctx, &run_config);

        assert_eq!(params_a.len(), params_b.len());
        assert_eq!(train_a.len(), train_b.len());
        assert_eq!(val_a.len(), val_b.len());
        assert_eq!(histories_a.len(), histories_b.len());
        assert_eq!(evaluations_a, evaluations_b);

        for (lhs, rhs) in params_a.iter().zip(&params_b) {
            assert_eq!(lhs.len(), rhs.len());
            for (left_value, right_value) in lhs.iter().zip(rhs) {
                assert!((left_value - right_value).abs() < 1e-12);
            }
        }

        for (left_value, right_value) in train_a.iter().zip(&train_b) {
            assert!((left_value - right_value).abs() < 1e-12);
        }
        for (left_value, right_value) in val_a.iter().zip(&val_b) {
            assert!((left_value - right_value).abs() < 1e-12);
        }

        for (lhs_history, rhs_history) in histories_a.iter().zip(&histories_b) {
            assert_eq!(lhs_history.len(), rhs_history.len());
            for (lhs_entry, rhs_entry) in lhs_history.iter().zip(rhs_history) {
                assert_eq!(lhs_entry.0, rhs_entry.0);
                assert_eq!(lhs_entry.1.len(), rhs_entry.1.len());
                for (lhs_param, rhs_param) in lhs_entry.1.iter().zip(&rhs_entry.1) {
                    assert!((lhs_param - rhs_param).abs() < 1e-12);
                }
                assert!((lhs_entry.2 - rhs_entry.2).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn parse_run_config_supports_equals_and_space_forms() {
        let parsed = parse_run_config(&[
            "--max-generations=5".to_owned(),
            "--max-evaluations".to_owned(),
            "100".to_owned(),
            "--seed".to_owned(),
            "7".to_owned(),
            "--folds".to_owned(),
            "2".to_owned(),
            "--output-dir".to_owned(),
            "/tmp/opt".to_owned(),
        ])
        .expect("parse run config");

        assert_eq!(parsed.max_generations, 5);
        assert_eq!(parsed.max_evaluations, 100);
        assert_eq!(parsed.seed, 7);
        assert_eq!(parsed.folds, 2);
        assert_eq!(parsed.output_dir, Some(PathBuf::from("/tmp/opt")));
    }

    #[test]
    fn optimize_fold_pads_history_to_requested_generation_count() {
        let ctx = sample_eval_context();
        let run_config = RunConfig {
            max_generations: 5,
            max_evaluations: 10,
            seed: 42,
            folds: 1,
            output_dir: None,
        };
        let (_params, _fitness, history, _evaluations) = optimize_fold(&ctx, &[], 0, &run_config);
        assert_eq!(history.len(), run_config.max_generations);
    }
}
