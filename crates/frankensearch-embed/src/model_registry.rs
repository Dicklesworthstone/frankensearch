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
    model_candidates(data_dir, embedder_dir_name(entry), entry.huggingface_id)
        .iter()
        .any(|dir| has_embedder_files(entry.id, dir))
}

fn reranker_is_available(entry: &RegisteredReranker, data_dir: &Path) -> bool {
    if !entry.requires_model_files {
        return true;
    }
    model_candidates(data_dir, reranker_dir_name(entry), entry.huggingface_id)
        .iter()
        .any(|dir| has_reranker_files(dir))
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

fn push_candidate(paths: &mut Vec<PathBuf>, seen: &mut BTreeSet<PathBuf>, path: PathBuf) {
    if seen.insert(path.clone()) {
        paths.push(path);
    }
}

#[cfg(test)]
mod tests {
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
}
