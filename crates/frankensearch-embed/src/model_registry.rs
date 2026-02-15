//! Static model registry with runtime availability checks.
//!
//! The registry catalogs known embedders and rerankers, then filters by
//! on-disk model availability under a configured data directory.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use frankensearch_core::error::{SearchError, SearchResult};
use tracing::warn;

/// Bakeoff eligibility cutoff (strictly later than this date).
pub const BAKEOFF_CUTOFF_DATE: &str = "2025-11-01";

const NO_HUGGINGFACE_ID: &str = "builtin";

const MODEL_ONNX_SUBDIR: &str = "onnx/model.onnx";
const MODEL_ONNX_LEGACY: &str = "model.onnx";
const TOKENIZER_JSON: &str = "tokenizer.json";
const MODEL_SAFETENSORS: &str = "model.safetensors";
const CONFIG_JSON: &str = "config.json";
const SPECIAL_TOKENS_JSON: &str = "special_tokens_map.json";
const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
const FRANKENSEARCH_MODEL_DIR_ENV: &str = "FRANKENSEARCH_MODEL_DIR";
const FRANKENSEARCH_DATA_DIR_ENV: &str = "FRANKENSEARCH_DATA_DIR";
const XDG_DATA_HOME_ENV: &str = "XDG_DATA_HOME";
const KNOWN_MODEL_LAYOUT_DIRS: [&str; 4] = [
    "potion-base-128M",
    "potion-multilingual-128M",
    "all-MiniLM-L6-v2",
    "ms-marco-MiniLM-L-6-v2",
];

/// Static embedder metadata entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisteredEmbedder {
    /// Short display name.
    pub name: &'static str,
    /// Stable identifier.
    pub id: &'static str,
    /// Embedding dimension.
    pub dimension: usize,
    /// Semantic capability flag.
    pub is_semantic: bool,
    /// Human-facing description.
    pub description: &'static str,
    /// Whether external model files are required.
    pub requires_model_files: bool,
    /// Model release date in ISO-8601 (`YYYY-MM-DD`).
    pub release_date: &'static str,
    /// Source model repository id (or `builtin`).
    pub huggingface_id: &'static str,
    /// Approximate model size.
    pub size_bytes: u64,
    /// Baseline flag for bakeoff comparisons.
    pub is_baseline: bool,
}

/// Static reranker metadata entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisteredReranker {
    /// Short display name.
    pub name: &'static str,
    /// Stable identifier.
    pub id: &'static str,
    /// Human-facing description.
    pub description: &'static str,
    /// Whether external model files are required.
    pub requires_model_files: bool,
    /// Model release date in ISO-8601 (`YYYY-MM-DD`).
    pub release_date: &'static str,
    /// Source model repository id.
    pub huggingface_id: &'static str,
    /// Approximate model size.
    pub size_bytes: u64,
    /// Baseline flag for bakeoff comparisons.
    pub is_baseline: bool,
}

const REGISTERED_EMBEDDERS: [RegisteredEmbedder; 6] = [
    RegisteredEmbedder {
        name: "minilm",
        id: "minilm-384",
        dimension: 384,
        is_semantic: true,
        description: "MiniLM-L6-v2 transformer quality embedder",
        requires_model_files: true,
        release_date: "2022-08-01",
        huggingface_id: "sentence-transformers/all-MiniLM-L6-v2",
        size_bytes: 90_000_000,
        is_baseline: true,
    },
    RegisteredEmbedder {
        name: "snowflake-arctic-s",
        id: "snowflake-arctic-s-384",
        dimension: 384,
        is_semantic: true,
        description: "Snowflake Arctic small embedding model",
        requires_model_files: true,
        release_date: "2025-11-10",
        huggingface_id: "Snowflake/snowflake-arctic-embed-s",
        size_bytes: 130_000_000,
        is_baseline: false,
    },
    RegisteredEmbedder {
        name: "nomic-embed",
        id: "nomic-embed-768",
        dimension: 768,
        is_semantic: true,
        description: "Nomic high-dimensional semantic embedder",
        requires_model_files: true,
        release_date: "2025-11-05",
        huggingface_id: "nomic-ai/nomic-embed-text-v1.5",
        size_bytes: 280_000_000,
        is_baseline: false,
    },
    RegisteredEmbedder {
        name: "potion-multilingual-128M",
        id: "potion-multilingual-128m-256",
        dimension: 256,
        is_semantic: true,
        description: "Potion multilingual static embedder",
        requires_model_files: true,
        release_date: "2025-10-10",
        huggingface_id: "minishlab/potion-multilingual-128M",
        size_bytes: 128_000_000,
        is_baseline: false,
    },
    RegisteredEmbedder {
        name: "potion-retrieval-32M",
        id: "potion-retrieval-32m-512",
        dimension: 512,
        is_semantic: true,
        description: "Potion retrieval static embedder",
        requires_model_files: true,
        release_date: "2025-10-20",
        huggingface_id: "minishlab/potion-retrieval-32M",
        size_bytes: 32_000_000,
        is_baseline: false,
    },
    RegisteredEmbedder {
        name: "hash/fnv1a",
        id: "fnv1a-384",
        dimension: 384,
        is_semantic: false,
        description: "FNV-1a deterministic hash embedder",
        requires_model_files: false,
        release_date: "2020-01-01",
        huggingface_id: NO_HUGGINGFACE_ID,
        size_bytes: 0,
        is_baseline: true,
    },
];

const REGISTERED_RERANKERS: [RegisteredReranker; 5] = [
    RegisteredReranker {
        name: "ms-marco-minilm",
        id: "ms-marco-minilm",
        description: "MS MARCO MiniLM cross-encoder baseline reranker",
        requires_model_files: true,
        release_date: "2022-01-15",
        huggingface_id: "cross-encoder/ms-marco-MiniLM-L-6-v2",
        size_bytes: 90_000_000,
        is_baseline: true,
    },
    RegisteredReranker {
        name: "flashrank-nano",
        id: "flashrank-nano",
        description: "FlashRank compact ONNX reranker",
        requires_model_files: true,
        release_date: "2024-06-01",
        huggingface_id: "prithivida/flashrank-nano",
        size_bytes: 4_000_000,
        is_baseline: false,
    },
    RegisteredReranker {
        name: "bge-reranker-v2",
        id: "bge-reranker-v2",
        description: "BAAI BGE reranker v2",
        requires_model_files: true,
        release_date: "2024-09-15",
        huggingface_id: "BAAI/bge-reranker-v2-m3",
        size_bytes: 450_000_000,
        is_baseline: false,
    },
    RegisteredReranker {
        name: "jina-reranker-turbo",
        id: "jina-reranker-turbo",
        description: "Jina turbo multilingual reranker",
        requires_model_files: true,
        release_date: "2025-02-01",
        huggingface_id: "jinaai/jina-reranker-v2-base-multilingual",
        size_bytes: 320_000_000,
        is_baseline: false,
    },
    RegisteredReranker {
        name: "mxbai-rerank-xsmall",
        id: "mxbai-rerank-xsmall",
        description: "Mixedbread xsmall reranker",
        requires_model_files: true,
        release_date: "2025-03-20",
        huggingface_id: "mixedbread-ai/mxbai-rerank-xsmall-v1",
        size_bytes: 150_000_000,
        is_baseline: false,
    },
];

/// Return all static embedder entries.
#[must_use]
pub const fn registered_embedders() -> &'static [RegisteredEmbedder] {
    &REGISTERED_EMBEDDERS
}

/// Return all static reranker entries.
#[must_use]
pub const fn registered_rerankers() -> &'static [RegisteredReranker] {
    &REGISTERED_RERANKERS
}

/// Runtime registry wrapper with configured model-data root.
#[derive(Debug, Clone)]
pub struct EmbedderRegistry {
    data_dir: PathBuf,
}

impl Default for EmbedderRegistry {
    fn default() -> Self {
        Self {
            data_dir: default_registry_data_dir(),
        }
    }
}

impl EmbedderRegistry {
    /// Build a registry rooted at a specific data directory.
    #[must_use]
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
        }
    }

    /// Data directory used for runtime availability checks.
    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Return embedders that are currently available on disk.
    #[must_use]
    pub fn available(&self) -> Vec<&'static RegisteredEmbedder> {
        registered_embedders()
            .iter()
            .filter(|entry| embedder_is_available(entry, &self.data_dir))
            .collect()
    }

    /// Look up an embedder by short name or id (case-insensitive).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&'static RegisteredEmbedder> {
        registered_embedders().iter().find(|entry| {
            entry.name.eq_ignore_ascii_case(name) || entry.id.eq_ignore_ascii_case(name)
        })
    }

    /// Return the most preferred available embedder.
    ///
    /// Preference is quality-first, with deterministic fallback to hash.
    #[must_use]
    pub fn best_available(&self) -> &'static RegisteredEmbedder {
        const HASH_ENTRY_INDEX: usize = REGISTERED_EMBEDDERS.len() - 1;
        self.available()
            .into_iter()
            .max_by_key(|entry| embedder_preference_rank(entry.id))
            .unwrap_or(&REGISTERED_EMBEDDERS[HASH_ENTRY_INDEX])
    }

    /// Return available semantic embedders released after the bakeoff cutoff.
    #[must_use]
    pub fn bakeoff_eligible(&self) -> Vec<&'static RegisteredEmbedder> {
        self.available()
            .into_iter()
            .filter(|entry| entry.is_semantic && entry.release_date > BAKEOFF_CUTOFF_DATE)
            .collect()
    }

    /// Return rerankers that appear available on disk.
    #[must_use]
    pub fn available_rerankers(&self) -> Vec<&'static RegisteredReranker> {
        registered_rerankers()
            .iter()
            .filter(|entry| reranker_is_available(entry, &self.data_dir))
            .collect()
    }
}

fn default_registry_data_dir() -> PathBuf {
    ensure_model_storage_layout()
}

pub(crate) fn model_storage_root() -> PathBuf {
    if let Some(path) = std::env::var_os(FRANKENSEARCH_MODEL_DIR_ENV) {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os(FRANKENSEARCH_DATA_DIR_ENV) {
        return PathBuf::from(path).join("models");
    }
    if let Some(path) = std::env::var_os(XDG_DATA_HOME_ENV) {
        return PathBuf::from(path).join("frankensearch").join("models");
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(path) = dirs::data_local_dir() {
            return path.join("frankensearch").join("models");
        }
    }
    if let Some(home) = dirs::home_dir() {
        return home
            .join(".local")
            .join("share")
            .join("frankensearch")
            .join("models");
    }
    if let Some(path) = dirs::data_local_dir() {
        return path.join("frankensearch").join("models");
    }
    PathBuf::from("models")
}

pub(crate) fn ensure_model_storage_layout_checked() -> SearchResult<PathBuf> {
    let root = model_storage_root();
    ensure_model_layout_dirs(&root).map_err(|source| SearchError::ModelLoadFailed {
        path: root.clone(),
        source: Box::new(source),
    })?;
    Ok(root)
}

pub(crate) fn ensure_model_storage_layout() -> PathBuf {
    match ensure_model_storage_layout_checked() {
        Ok(root) => root,
        Err(error) => {
            let root = model_storage_root();
            warn!(
                root = %root.display(),
                error = %error,
                "failed to ensure model storage layout; continuing with unresolved root"
            );
            root
        }
    }
}

fn ensure_model_layout_dirs(root: &Path) -> std::io::Result<()> {
    fs::create_dir_all(root)?;
    for model_dir in KNOWN_MODEL_LAYOUT_DIRS {
        fs::create_dir_all(root.join(model_dir))?;
    }
    Ok(())
}

pub(crate) fn model_directory_variants(model_name: &str) -> Vec<String> {
    let mut variants = vec![model_name.to_owned()];
    if matches!(model_name, "potion-base-128M" | "potion-multilingual-128M") {
        variants.push("potion-base-128M".to_owned());
        variants.push("potion-multilingual-128M".to_owned());
    }
    variants.sort();
    variants.dedup();
    variants
}

fn embedder_preference_rank(id: &str) -> u8 {
    match id {
        "minilm-384" => 100,
        "snowflake-arctic-s-384" => 95,
        "nomic-embed-768" => 90,
        "potion-retrieval-32m-512" => 80,
        "potion-multilingual-128m-256" => 70,
        "fnv1a-384" => 1,
        _ => 0,
    }
}

fn embedder_is_available(entry: &RegisteredEmbedder, data_dir: &Path) -> bool {
    if !entry.requires_model_files {
        return true;
    }
    any_model_candidate_matches(
        data_dir,
        embedder_dir_name(entry),
        entry.huggingface_id,
        |dir| has_embedder_files(entry.id, dir),
    )
}

fn reranker_is_available(entry: &RegisteredReranker, data_dir: &Path) -> bool {
    if !entry.requires_model_files {
        return true;
    }
    any_model_candidate_matches(
        data_dir,
        reranker_dir_name(entry),
        entry.huggingface_id,
        has_reranker_files,
    )
}

fn embedder_dir_name(entry: &RegisteredEmbedder) -> &'static str {
    match entry.id {
        "minilm-384" => "all-MiniLM-L6-v2",
        "snowflake-arctic-s-384" => "snowflake-arctic-embed-s",
        "nomic-embed-768" => "nomic-embed-text-v1.5",
        "potion-multilingual-128m-256" => "potion-multilingual-128M",
        "potion-retrieval-32m-512" => "potion-retrieval-32M",
        "fnv1a-384" => "hash-fallback",
        _ => "unknown-model",
    }
}

fn reranker_dir_name(entry: &RegisteredReranker) -> &'static str {
    match entry.id {
        "ms-marco-minilm" => "ms-marco-MiniLM-L-6-v2",
        "flashrank-nano" => "flashrank",
        "bge-reranker-v2" => "bge-reranker-v2-m3",
        "jina-reranker-turbo" => "jina-reranker-v2-base-multilingual",
        "mxbai-rerank-xsmall" => "mxbai-rerank-xsmall-v1",
        _ => "unknown-reranker",
    }
}

fn has_embedder_files(id: &str, dir: &Path) -> bool {
    match id {
        "minilm-384" => {
            has_any_file(dir, &[MODEL_ONNX_SUBDIR, MODEL_ONNX_LEGACY])
                && has_all_files(
                    dir,
                    &[
                        TOKENIZER_JSON,
                        CONFIG_JSON,
                        SPECIAL_TOKENS_JSON,
                        TOKENIZER_CONFIG_JSON,
                    ],
                )
        }
        "snowflake-arctic-s-384" | "nomic-embed-768" => {
            has_any_file(dir, &[MODEL_ONNX_SUBDIR, MODEL_ONNX_LEGACY])
                && has_all_files(dir, &[TOKENIZER_JSON])
        }
        "potion-multilingual-128m-256" | "potion-retrieval-32m-512" => {
            has_all_files(dir, &[TOKENIZER_JSON, MODEL_SAFETENSORS])
        }
        _ => false,
    }
}

fn has_reranker_files(dir: &Path) -> bool {
    has_any_file(dir, &[MODEL_ONNX_SUBDIR, MODEL_ONNX_LEGACY])
        && has_all_files(dir, &[TOKENIZER_JSON])
}

fn has_any_file(dir: &Path, files: &[&str]) -> bool {
    files.iter().any(|file| dir.join(file).is_file())
}

fn has_all_files(dir: &Path, files: &[&str]) -> bool {
    files.iter().all(|file| dir.join(file).is_file())
}

fn model_candidates(data_dir: &Path, model_dir: &str, huggingface_id: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    push_candidate(&mut out, &mut seen, data_dir.to_path_buf());
    for variant in model_directory_variants(model_dir) {
        push_candidate(&mut out, &mut seen, data_dir.join(&variant));
    }

    if huggingface_id != NO_HUGGINGFACE_ID {
        let hub_root = data_dir
            .join("huggingface/hub")
            .join(format!("models--{}", huggingface_id.replace('/', "--")))
            .join("snapshots");
        if let Ok(entries) = fs::read_dir(hub_root) {
            for entry in entries.flatten() {
                push_candidate(&mut out, &mut seen, entry.path());
            }
        }
    }

    out
}

fn any_model_candidate_matches(
    data_dir: &Path,
    model_dir: &str,
    huggingface_id: &str,
    mut matches: impl FnMut(&Path) -> bool,
) -> bool {
    if matches(data_dir) {
        return true;
    }

    for variant in model_directory_variants(model_dir) {
        if matches(&data_dir.join(&variant)) {
            return true;
        }
    }

    if huggingface_id != NO_HUGGINGFACE_ID {
        let hub_root = data_dir
            .join("huggingface/hub")
            .join(format!("models--{}", huggingface_id.replace('/', "--")))
            .join("snapshots");
        if let Ok(entries) = fs::read_dir(hub_root) {
            for entry in entries.flatten() {
                if matches(&entry.path()) {
                    return true;
                }
            }
        }
    }

    false
}

fn push_candidate(paths: &mut Vec<PathBuf>, seen: &mut BTreeSet<PathBuf>, path: PathBuf) {
    if seen.insert(path.clone()) {
        paths.push(path);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn touch_model_files(root: &Path, model_dir: &str, files: &[&str]) {
        let model_root = root.join(model_dir);
        std::fs::create_dir_all(&model_root).unwrap();
        for file in files {
            let path = model_root.join(file);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, b"stub").unwrap();
        }
    }

    #[test]
    fn get_finds_embedder_by_name_and_id() {
        let registry = EmbedderRegistry::new("unused");
        assert_eq!(registry.get("minilm").map(|m| m.id), Some("minilm-384"));
        assert_eq!(
            registry.get("FNv1A-384").map(|m| m.name),
            Some("hash/fnv1a")
        );
        assert!(registry.get("does-not-exist").is_none());
    }

    #[test]
    fn empty_directory_only_exposes_hash_embedder() {
        let temp = tempfile::tempdir().unwrap();
        let registry = EmbedderRegistry::new(temp.path());
        let available_ids: Vec<_> = registry.available().iter().map(|entry| entry.id).collect();
        assert_eq!(available_ids, vec!["fnv1a-384"]);
    }

    #[test]
    fn availability_detects_minilm_and_potion() {
        let temp = tempfile::tempdir().unwrap();
        touch_model_files(
            temp.path(),
            "all-MiniLM-L6-v2",
            &[
                MODEL_ONNX_SUBDIR,
                TOKENIZER_JSON,
                CONFIG_JSON,
                SPECIAL_TOKENS_JSON,
                TOKENIZER_CONFIG_JSON,
            ],
        );
        touch_model_files(
            temp.path(),
            "potion-multilingual-128M",
            &[TOKENIZER_JSON, MODEL_SAFETENSORS],
        );

        let registry = EmbedderRegistry::new(temp.path());
        let available_ids: Vec<_> = registry.available().iter().map(|entry| entry.id).collect();
        assert!(available_ids.contains(&"minilm-384"));
        assert!(available_ids.contains(&"potion-multilingual-128m-256"));
        assert!(available_ids.contains(&"fnv1a-384"));
    }

    #[test]
    fn model_candidates_include_default_paths_in_order() {
        let temp = tempfile::tempdir().unwrap();
        let candidates = model_candidates(
            temp.path(),
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
        );

        assert!(candidates.len() >= 2);
        assert_eq!(candidates[0], temp.path().to_path_buf());
        assert_eq!(candidates[1], temp.path().join("all-MiniLM-L6-v2"));
    }

    #[test]
    fn model_directory_variants_include_potion_aliases() {
        let variants = model_directory_variants("potion-multilingual-128M");
        assert!(variants.contains(&"potion-base-128M".to_owned()));
        assert!(variants.contains(&"potion-multilingual-128M".to_owned()));
    }

    #[test]
    fn ensure_model_layout_dirs_creates_known_directories() {
        let temp = tempfile::tempdir().unwrap();
        ensure_model_layout_dirs(temp.path()).expect("create layout");

        for expected in KNOWN_MODEL_LAYOUT_DIRS {
            assert!(temp.path().join(expected).is_dir(), "missing {expected}");
        }
    }

    #[test]
    fn availability_detects_legacy_model_onnx_layout() {
        let temp = tempfile::tempdir().unwrap();
        touch_model_files(
            temp.path(),
            "all-MiniLM-L6-v2",
            &[
                MODEL_ONNX_LEGACY,
                TOKENIZER_JSON,
                CONFIG_JSON,
                SPECIAL_TOKENS_JSON,
                TOKENIZER_CONFIG_JSON,
            ],
        );

        let registry = EmbedderRegistry::new(temp.path());
        assert!(
            registry
                .available()
                .iter()
                .map(|entry| entry.id)
                .any(|id| id == "minilm-384")
        );
    }

    #[test]
    fn availability_detects_huggingface_snapshot_layout() {
        let temp = tempfile::tempdir().unwrap();
        touch_model_files(
            temp.path(),
            "huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/abc123",
            &[
                MODEL_ONNX_SUBDIR,
                TOKENIZER_JSON,
                CONFIG_JSON,
                SPECIAL_TOKENS_JSON,
                TOKENIZER_CONFIG_JSON,
            ],
        );

        let registry = EmbedderRegistry::new(temp.path());
        assert!(
            registry
                .available()
                .iter()
                .map(|entry| entry.id)
                .any(|id| id == "minilm-384")
        );
    }

    #[test]
    fn best_available_prefers_quality_model() {
        let temp = tempfile::tempdir().unwrap();
        touch_model_files(
            temp.path(),
            "all-MiniLM-L6-v2",
            &[
                MODEL_ONNX_SUBDIR,
                TOKENIZER_JSON,
                CONFIG_JSON,
                SPECIAL_TOKENS_JSON,
                TOKENIZER_CONFIG_JSON,
            ],
        );
        let registry = EmbedderRegistry::new(temp.path());
        assert_eq!(registry.best_available().id, "minilm-384");
    }

    #[test]
    fn bakeoff_eligibility_respects_cutoff_and_availability() {
        let temp = tempfile::tempdir().unwrap();
        touch_model_files(
            temp.path(),
            "snowflake-arctic-embed-s",
            &[MODEL_ONNX_SUBDIR, TOKENIZER_JSON],
        );
        touch_model_files(
            temp.path(),
            "nomic-embed-text-v1.5",
            &[MODEL_ONNX_SUBDIR, TOKENIZER_JSON],
        );
        touch_model_files(
            temp.path(),
            "all-MiniLM-L6-v2",
            &[
                MODEL_ONNX_SUBDIR,
                TOKENIZER_JSON,
                CONFIG_JSON,
                SPECIAL_TOKENS_JSON,
                TOKENIZER_CONFIG_JSON,
            ],
        );

        let registry = EmbedderRegistry::new(temp.path());
        let eligible_ids: Vec<_> = registry
            .bakeoff_eligible()
            .iter()
            .map(|entry| entry.id)
            .collect();

        assert!(eligible_ids.contains(&"snowflake-arctic-s-384"));
        assert!(eligible_ids.contains(&"nomic-embed-768"));
        assert!(!eligible_ids.contains(&"minilm-384"));
    }

    #[test]
    fn reranker_availability_detects_flashrank_layout() {
        let temp = tempfile::tempdir().unwrap();
        touch_model_files(
            temp.path(),
            "flashrank",
            &[MODEL_ONNX_SUBDIR, TOKENIZER_JSON],
        );
        let registry = EmbedderRegistry::new(temp.path());
        assert!(
            registry
                .available_rerankers()
                .iter()
                .map(|entry| entry.id)
                .any(|id| id == "flashrank-nano")
        );
    }

    // --- Preference rank tests ---

    #[test]
    fn preference_rank_orders_minilm_highest() {
        assert!(
            embedder_preference_rank("minilm-384")
                > embedder_preference_rank("snowflake-arctic-s-384")
        );
        assert!(
            embedder_preference_rank("snowflake-arctic-s-384")
                > embedder_preference_rank("nomic-embed-768")
        );
        assert!(
            embedder_preference_rank("nomic-embed-768")
                > embedder_preference_rank("potion-retrieval-32m-512")
        );
        assert!(
            embedder_preference_rank("potion-retrieval-32m-512")
                > embedder_preference_rank("potion-multilingual-128m-256")
        );
        assert!(
            embedder_preference_rank("potion-multilingual-128m-256")
                > embedder_preference_rank("fnv1a-384")
        );
    }

    #[test]
    fn preference_rank_unknown_id_is_zero() {
        assert_eq!(embedder_preference_rank("nonexistent-model"), 0);
        assert_eq!(embedder_preference_rank(""), 0);
    }

    #[test]
    fn preference_rank_hash_is_minimal_among_known() {
        assert_eq!(embedder_preference_rank("fnv1a-384"), 1);
        // Hash is above unknown but below all real embedders.
        assert!(embedder_preference_rank("fnv1a-384") > embedder_preference_rank("unknown"));
    }

    // --- Directory name mapping tests ---

    #[test]
    fn embedder_dir_names_are_correct() {
        for entry in registered_embedders() {
            let dir = embedder_dir_name(entry);
            assert!(!dir.is_empty(), "empty dir for {}", entry.id);
            if entry.id == "fnv1a-384" {
                assert_eq!(dir, "hash-fallback");
            } else {
                assert_ne!(dir, "unknown-model", "unmapped dir for {}", entry.id);
            }
        }
    }

    #[test]
    fn reranker_dir_names_are_correct() {
        for entry in registered_rerankers() {
            let dir = reranker_dir_name(entry);
            assert!(!dir.is_empty(), "empty dir for {}", entry.id);
            assert_ne!(dir, "unknown-reranker", "unmapped dir for {}", entry.id);
        }
    }

    #[test]
    fn embedder_dir_name_unknown_fallback() {
        let fake = RegisteredEmbedder {
            name: "fake",
            id: "fake-999",
            dimension: 1,
            is_semantic: false,
            description: "test",
            requires_model_files: false,
            release_date: "2020-01-01",
            huggingface_id: "builtin",
            size_bytes: 0,
            is_baseline: false,
        };
        assert_eq!(embedder_dir_name(&fake), "unknown-model");
    }

    #[test]
    fn reranker_dir_name_unknown_fallback() {
        let fake = RegisteredReranker {
            name: "fake",
            id: "fake-999",
            description: "test",
            requires_model_files: false,
            release_date: "2020-01-01",
            huggingface_id: "builtin",
            size_bytes: 0,
            is_baseline: false,
        };
        assert_eq!(reranker_dir_name(&fake), "unknown-reranker");
    }

    // --- Model directory variants tests ---

    #[test]
    fn model_directory_variants_non_potion_returns_single() {
        let variants = model_directory_variants("all-MiniLM-L6-v2");
        assert_eq!(variants, vec!["all-MiniLM-L6-v2"]);
    }

    #[test]
    fn model_directory_variants_potion_base_includes_both() {
        let variants = model_directory_variants("potion-base-128M");
        assert!(variants.contains(&"potion-base-128M".to_owned()));
        assert!(variants.contains(&"potion-multilingual-128M".to_owned()));
    }

    #[test]
    fn model_directory_variants_are_sorted_and_deduped() {
        let variants = model_directory_variants("potion-multilingual-128M");
        let mut sorted = variants.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(variants, sorted);
    }

    // --- Registry static invariant tests ---

    #[test]
    fn embedder_ids_are_unique() {
        let ids: Vec<&str> = registered_embedders().iter().map(|e| e.id).collect();
        let mut deduped = ids.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(ids.len(), deduped.len(), "duplicate embedder IDs");
    }

    #[test]
    fn embedder_names_are_unique() {
        let names: Vec<&str> = registered_embedders().iter().map(|e| e.name).collect();
        let mut deduped = names.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "duplicate embedder names");
    }

    #[test]
    fn reranker_ids_are_unique() {
        let ids: Vec<&str> = registered_rerankers().iter().map(|e| e.id).collect();
        let mut deduped = ids.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(ids.len(), deduped.len(), "duplicate reranker IDs");
    }

    #[test]
    fn reranker_names_are_unique() {
        let names: Vec<&str> = registered_rerankers().iter().map(|e| e.name).collect();
        let mut deduped = names.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "duplicate reranker names");
    }

    #[test]
    fn all_embedders_have_descriptions() {
        for entry in registered_embedders() {
            assert!(
                !entry.description.is_empty(),
                "empty description for {}",
                entry.id
            );
        }
    }

    #[test]
    fn all_rerankers_have_descriptions() {
        for entry in registered_rerankers() {
            assert!(
                !entry.description.is_empty(),
                "empty description for {}",
                entry.id
            );
        }
    }

    #[test]
    fn all_embedders_have_positive_dimensions() {
        for entry in registered_embedders() {
            assert!(entry.dimension > 0, "zero dimension for {}", entry.id);
        }
    }

    #[test]
    fn at_least_one_baseline_embedder_exists() {
        assert!(
            registered_embedders().iter().any(|e| e.is_baseline),
            "no baseline embedder"
        );
    }

    #[test]
    fn at_least_one_baseline_reranker_exists() {
        assert!(
            registered_rerankers().iter().any(|e| e.is_baseline),
            "no baseline reranker"
        );
    }

    #[test]
    fn hash_embedder_does_not_require_model_files() {
        let hash = registered_embedders()
            .iter()
            .find(|e| e.id == "fnv1a-384")
            .expect("hash embedder not found");
        assert!(!hash.requires_model_files);
        assert!(!hash.is_semantic);
        assert_eq!(hash.size_bytes, 0);
    }

    // --- best_available / bakeoff / reranker edge cases ---

    #[test]
    fn best_available_falls_back_to_hash_when_empty() {
        let temp = tempfile::tempdir().unwrap();
        let registry = EmbedderRegistry::new(temp.path());
        assert_eq!(registry.best_available().id, "fnv1a-384");
    }

    #[test]
    fn bakeoff_eligible_empty_with_no_models() {
        let temp = tempfile::tempdir().unwrap();
        let registry = EmbedderRegistry::new(temp.path());
        assert!(
            registry.bakeoff_eligible().is_empty(),
            "bakeoff should be empty with no models on disk"
        );
    }

    #[test]
    fn available_rerankers_empty_with_no_models() {
        let temp = tempfile::tempdir().unwrap();
        let registry = EmbedderRegistry::new(temp.path());
        assert!(
            registry.available_rerankers().is_empty(),
            "all rerankers require model files, so empty dir => no rerankers"
        );
    }

    #[test]
    fn bakeoff_excludes_hash_embedder() {
        // Even if hash is "available", it is not semantic so never bakeoff-eligible.
        let temp = tempfile::tempdir().unwrap();
        let registry = EmbedderRegistry::new(temp.path());
        let eligible_ids: Vec<_> = registry.bakeoff_eligible().iter().map(|e| e.id).collect();
        assert!(!eligible_ids.contains(&"fnv1a-384"));
    }

    // --- Potion retrieval availability ---

    #[test]
    fn availability_detects_potion_retrieval() {
        let temp = tempfile::tempdir().unwrap();
        touch_model_files(
            temp.path(),
            "potion-retrieval-32M",
            &[TOKENIZER_JSON, MODEL_SAFETENSORS],
        );
        let registry = EmbedderRegistry::new(temp.path());
        assert!(
            registry
                .available()
                .iter()
                .any(|e| e.id == "potion-retrieval-32m-512")
        );
    }

    // --- push_candidate dedup ---

    #[test]
    fn push_candidate_deduplicates() {
        let mut paths = Vec::new();
        let mut seen = BTreeSet::new();
        let p = PathBuf::from("/test/path");
        push_candidate(&mut paths, &mut seen, p.clone());
        push_candidate(&mut paths, &mut seen, p.clone());
        push_candidate(&mut paths, &mut seen, p);
        assert_eq!(paths.len(), 1);
    }

    // --- get case-insensitive lookup ---

    #[test]
    fn get_is_case_insensitive() {
        let registry = EmbedderRegistry::new("unused");
        assert!(registry.get("MINILM").is_some());
        assert!(registry.get("MiniLM").is_some());
        assert!(registry.get("MINILM-384").is_some());
    }

    // --- has_any_file / has_all_files edge cases ---

    #[test]
    fn has_any_file_returns_false_for_empty_dir() {
        let temp = tempfile::tempdir().unwrap();
        assert!(!has_any_file(
            temp.path(),
            &[MODEL_ONNX_SUBDIR, MODEL_ONNX_LEGACY]
        ));
    }

    #[test]
    fn has_all_files_returns_true_for_empty_list() {
        let temp = tempfile::tempdir().unwrap();
        assert!(has_all_files(temp.path(), &[]));
    }

    // --- BAKEOFF_CUTOFF_DATE validity ---

    #[test]
    fn bakeoff_cutoff_date_is_valid_iso8601() {
        assert_eq!(BAKEOFF_CUTOFF_DATE.len(), 10);
        assert_eq!(&BAKEOFF_CUTOFF_DATE[4..5], "-");
        assert_eq!(&BAKEOFF_CUTOFF_DATE[7..8], "-");
    }

    // --- model_candidates with builtin huggingface_id ---

    #[test]
    fn model_candidates_skip_huggingface_for_builtin() {
        let temp = tempfile::tempdir().unwrap();
        let candidates = model_candidates(temp.path(), "hash-fallback", NO_HUGGINGFACE_ID);
        // Should only have data_dir root + model_dir variant, no huggingface paths.
        for c in &candidates {
            assert!(
                !c.to_string_lossy().contains("huggingface"),
                "builtin should not probe huggingface: {c:?}"
            );
        }
    }

    #[test]
    fn availability_checks_match_reference_candidate_scan() {
        let temp = tempfile::tempdir().unwrap();
        touch_model_files(
            temp.path(),
            "all-MiniLM-L6-v2",
            &[
                MODEL_ONNX_SUBDIR,
                TOKENIZER_JSON,
                CONFIG_JSON,
                SPECIAL_TOKENS_JSON,
                TOKENIZER_CONFIG_JSON,
            ],
        );
        touch_model_files(
            temp.path(),
            "flashrank",
            &[MODEL_ONNX_SUBDIR, TOKENIZER_JSON],
        );
        touch_model_files(
            temp.path(),
            "huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5/snapshots/snap-a",
            &[MODEL_ONNX_SUBDIR, TOKENIZER_JSON],
        );

        for entry in registered_embedders() {
            let expected = if entry.requires_model_files {
                model_candidates(temp.path(), embedder_dir_name(entry), entry.huggingface_id)
                    .iter()
                    .any(|dir| has_embedder_files(entry.id, dir))
            } else {
                true
            };
            let actual = embedder_is_available(entry, temp.path());
            assert_eq!(
                actual, expected,
                "embedder availability mismatch for {}",
                entry.id
            );
        }

        for entry in registered_rerankers() {
            let expected = if entry.requires_model_files {
                model_candidates(temp.path(), reranker_dir_name(entry), entry.huggingface_id)
                    .iter()
                    .any(|dir| has_reranker_files(dir))
            } else {
                true
            };
            let actual = reranker_is_available(entry, temp.path());
            assert_eq!(
                actual, expected,
                "reranker availability mismatch for {}",
                entry.id
            );
        }
    }

    // --- Registry counts ---

    #[test]
    fn registered_embedder_count() {
        assert_eq!(registered_embedders().len(), 6);
    }

    #[test]
    fn registered_reranker_count() {
        assert_eq!(registered_rerankers().len(), 5);
    }

    // --- data_dir accessor ---

    #[test]
    fn data_dir_returns_configured_path() {
        let registry = EmbedderRegistry::new("/my/custom/path");
        assert_eq!(registry.data_dir(), Path::new("/my/custom/path"));
    }

    #[test]
    #[ignore = "Perf probe for optimization loop: run explicitly with --ignored"]
    fn perf_probe_available_many_hf_snapshots() {
        let snapshot_count = std::env::var("MODEL_REGISTRY_PERF_SNAPSHOTS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(800);
        let iterations = std::env::var("MODEL_REGISTRY_PERF_ITERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(120);

        let temp = tempfile::tempdir().expect("tempdir");
        let hf_snapshots_root = temp.path().join(
            "huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots",
        );
        std::fs::create_dir_all(&hf_snapshots_root).expect("create hf root");
        for i in 0..snapshot_count {
            let dir = hf_snapshots_root.join(format!("snap-{i:04}"));
            std::fs::create_dir_all(&dir).expect("create snapshot");
        }

        // Keep valid files in the final snapshot to force worst-case scan.
        let valid_snapshot = hf_snapshots_root.join(format!("snap-{:04}", snapshot_count - 1));
        for file in [
            MODEL_ONNX_SUBDIR,
            TOKENIZER_JSON,
            CONFIG_JSON,
            SPECIAL_TOKENS_JSON,
            TOKENIZER_CONFIG_JSON,
        ] {
            let path = valid_snapshot.join(file);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create model parent");
            }
            std::fs::write(path, b"stub").expect("write model file");
        }

        let registry = EmbedderRegistry::new(temp.path());
        let started = Instant::now();
        let mut checksum = 0_usize;
        for _ in 0..iterations {
            checksum = checksum.saturating_add(registry.available().len());
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;

        eprintln!(
            "MODEL_REGISTRY_PERF elapsed_ms={elapsed_ms:.3} snapshots={snapshot_count} iterations={iterations} checksum={checksum}"
        );
        assert!(checksum > 0);
    }
}
