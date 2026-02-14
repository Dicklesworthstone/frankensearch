//! Config override example: load from file, then apply env overrides.
//!
//! Run with file values only:
//! - `cargo run --example config_override`
//!
//! Run with env overrides on top:
//! - `FRANKENSEARCH_QUALITY_WEIGHT=0.9 FRANKENSEARCH_FAST_ONLY=1 cargo run --example config_override`

use std::path::Path;
use std::sync::Arc;

use frankensearch::prelude::*;
use frankensearch::{EmbedderStack, HashEmbedder, IndexBuilder, TwoTierIndex};
use frankensearch_core::traits::Embedder;

fn main() {
    let dir = std::env::temp_dir().join(format!("frankensearch-config-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let config_path = dir.join("search.toml");
    std::fs::write(&config_path, sample_config_toml()).expect("write sample config");

    let file_config = load_config_file(&config_path);
    let effective_config = file_config.clone().with_env_overrides();

    println!("config_file={}", config_path.display());
    print_config("from_file", &file_config);
    print_config("effective_after_env", &effective_config);

    let docs = vec![
        (
            "config-priority",
            "CLI flags should override environment and file configuration",
        ),
        (
            "rrf-notes",
            "RRF combines lexical and semantic ranks without score normalization",
        ),
        (
            "watch-mode",
            "Watch mode keeps the local index fresh during edits",
        ),
    ];

    asupersync::test_utils::run_test_with_cx(|cx| {
        let dir = dir.clone();
        let docs = docs.clone();
        async move {
            let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
            let quality = Arc::new(HashEmbedder::default_384()) as Arc<dyn Embedder>;
            let stack = EmbedderStack::from_parts(fast, Some(quality));

            let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
            for (doc_id, body) in docs {
                builder = builder.add_document(doc_id, body);
            }
            let _ = builder.build(&cx).await.expect("build index");
        }
    });

    let fast: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
    let quality: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_384());
    let index = Arc::new(TwoTierIndex::open(&dir, effective_config.clone()).expect("open index"));
    let searcher =
        TwoTierSearcher::new(index, fast, effective_config.clone()).with_quality_embedder(quality);

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (results, metrics) = searcher
            .search_collect(&cx, "configuration override order", 3)
            .await
            .expect("search");
        println!(
            "search_done hits={} phase1_ms={:.2} phase2_ms={:.2} fast_only={}",
            results.len(),
            metrics.phase1_total_ms,
            metrics.phase2_total_ms,
            effective_config.fast_only
        );
    });

    let _ = std::fs::remove_dir_all(&dir);
}

fn load_config_file(path: &Path) -> TwoTierConfig {
    let raw = std::fs::read_to_string(path).expect("read config file");
    toml::from_str::<TwoTierConfig>(&raw).expect("parse config file")
}

fn print_config(label: &str, config: &TwoTierConfig) {
    println!(
        "{label}: quality_weight={:.2} rrf_k={:.1} fast_only={} quality_timeout_ms={} candidate_multiplier={}",
        config.quality_weight,
        config.rrf_k,
        config.fast_only,
        config.quality_timeout_ms,
        config.candidate_multiplier
    );
}

const fn sample_config_toml() -> &'static str {
    r"
quality_weight = 0.55
rrf_k = 42.0
candidate_multiplier = 4
quality_timeout_ms = 800
fast_only = false
explain = true
hnsw_ef_search = 80
hnsw_ef_construction = 160
hnsw_m = 24
hnsw_threshold = 25000
mrl_search_dims = 0
mrl_rescore_top_k = 40
"
}
