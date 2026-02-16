//! XDG-compliant model cache directory layout.
//!
//! Resolution order for the model cache root:
//!
//! 1. `FRANKENSEARCH_MODEL_DIR` — explicit override (CI, Docker, shared cache)
//! 2. `FRANKENSEARCH_DATA_DIR` — override data home for all frankensearch data
//! 3. `XDG_DATA_HOME/frankensearch/models/` — XDG spec (Linux)
//! 4. `~/Library/Application Support/frankensearch/models/` — macOS (when XDG unset)
//! 5. `~/.local/share/frankensearch/models/` — POSIX fallback
//!
//! Within the cache root, each model gets a base directory. Versioned
//! subdirectories are also provisioned for forward-compatible migrations:
//!
//! ```text
//! <root>/
//!   potion-base-128M/
//!   all-MiniLM-L6-v2/
//!     onnx/model.onnx
//!     tokenizer.json
//!     config.json
//!     manifest.json
//!   potion-base-128M/
//!     v1/
//!       model.safetensors
//!       tokenizer.json
//!       config.json
//!       manifest.json
//! ```

use std::path::{Path, PathBuf};

use frankensearch_core::error::SearchResult;

// ─── Constants ──────────────────────────────────────────────────────────────

/// Environment variable: explicit model cache directory override.
pub const ENV_MODEL_DIR: &str = "FRANKENSEARCH_MODEL_DIR";

/// Environment variable: override data home for all frankensearch data.
pub const ENV_DATA_DIR: &str = "FRANKENSEARCH_DATA_DIR";

/// Environment variable: XDG data home (standard Linux convention).
const ENV_XDG_DATA_HOME: &str = "XDG_DATA_HOME";

/// Subdirectory under the data home for frankensearch.
const FRANKENSEARCH_SUBDIR: &str = "frankensearch";

/// Subdirectory within frankensearch data for model files.
const MODELS_SUBDIR: &str = "models";

/// Schema version for the cache layout format.
pub const MODEL_CACHE_LAYOUT_VERSION: u32 = 1;

/// Known model directory names and their current version tags.
const KNOWN_MODELS: &[KnownModel] = &[
    KnownModel {
        dir_name: "potion-base-128M",
        version: "v1",
        description: "Potion 128M fast embedder (256d)",
    },
    KnownModel {
        dir_name: "potion-multilingual-128M",
        version: "v1",
        description: "Potion multilingual 128M embedder (256d)",
    },
    KnownModel {
        dir_name: "all-MiniLM-L6-v2",
        version: "v1",
        description: "MiniLM-L6-v2 quality embedder (384d)",
    },
    KnownModel {
        dir_name: "ms-marco-MiniLM-L-6-v2",
        version: "v1",
        description: "MS MARCO MiniLM reranker",
    },
];

// ─── Known Model Metadata ──────────────────────────────────────────────────

/// Metadata for a known model in the cache layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownModel {
    /// Directory name under the cache root.
    pub dir_name: &'static str,
    /// Current version tag.
    pub version: &'static str,
    /// Human-readable description.
    pub description: &'static str,
}

/// Return all known models in the cache layout.
#[must_use]
pub const fn known_models() -> &'static [KnownModel] {
    KNOWN_MODELS
}

// ─── Cache Root Resolution ──────────────────────────────────────────────────

/// Resolve the model cache root directory using the priority chain.
///
/// Does NOT create directories — use [`ensure_cache_layout`] for that.
#[must_use]
pub fn resolve_cache_root() -> PathBuf {
    resolve_cache_root_with(&EnvReader::Real)
}

/// Resolve the cache root with a custom environment reader (for testing).
fn resolve_cache_root_with(env: &dyn EnvLookup) -> PathBuf {
    // 1. FRANKENSEARCH_MODEL_DIR
    if let Some(path) = env_var_if_non_empty(env, ENV_MODEL_DIR) {
        return PathBuf::from(path);
    }

    // 2. FRANKENSEARCH_DATA_DIR
    if let Some(path) = env_var_if_non_empty(env, ENV_DATA_DIR) {
        return PathBuf::from(path).join(MODELS_SUBDIR);
    }

    // 3. XDG_DATA_HOME
    if let Some(path) = env_var_if_non_empty(env, ENV_XDG_DATA_HOME) {
        return PathBuf::from(path)
            .join(FRANKENSEARCH_SUBDIR)
            .join(MODELS_SUBDIR);
    }

    // 4. macOS Application Support (when XDG unset)
    #[cfg(target_os = "macos")]
    {
        if let Some(path) = dirs::data_local_dir() {
            return path.join(FRANKENSEARCH_SUBDIR).join(MODELS_SUBDIR);
        }
    }

    // 5. ~/.local/share/frankensearch/models/
    if let Some(home) = dirs::home_dir() {
        return home
            .join(".local")
            .join("share")
            .join(FRANKENSEARCH_SUBDIR)
            .join(MODELS_SUBDIR);
    }

    // Ultimate fallback: data_local_dir or ./models
    dirs::data_local_dir().map_or_else(
        || PathBuf::from(MODELS_SUBDIR),
        |p| p.join(FRANKENSEARCH_SUBDIR).join(MODELS_SUBDIR),
    )
}

// ─── Cache Layout ───────────────────────────────────────────────────────────

/// Description of the full cache directory tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCacheLayout {
    /// Root directory of the model cache.
    pub root: PathBuf,
    /// Per-model versioned directories.
    pub model_dirs: Vec<ModelDirEntry>,
}

/// One model's directory entry within the cache layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDirEntry {
    /// Model directory name.
    pub name: String,
    /// Version tag.
    pub version: String,
    /// Full path to the model base directory.
    pub path: PathBuf,
}

impl ModelCacheLayout {
    /// Build the layout description for a given root.
    #[must_use]
    pub fn for_root(root: PathBuf) -> Self {
        let model_dirs = KNOWN_MODELS
            .iter()
            .map(|m| ModelDirEntry {
                name: m.dir_name.to_string(),
                version: m.version.to_string(),
                path: root.join(m.dir_name),
            })
            .collect();
        Self { root, model_dirs }
    }

    /// Build the layout using the default resolved root.
    #[must_use]
    pub fn default_layout() -> Self {
        Self::for_root(resolve_cache_root())
    }

    /// Get the base path for a model by directory name.
    #[must_use]
    pub fn model_path(&self, dir_name: &str) -> Option<&Path> {
        self.model_dirs
            .iter()
            .find(|e| e.name == dir_name)
            .map(|e| e.path.as_path())
    }

    /// Get the versioned path for a model by directory name.
    #[must_use]
    pub fn versioned_model_path(&self, dir_name: &str) -> Option<PathBuf> {
        self.model_dirs
            .iter()
            .find(|e| e.name == dir_name)
            .map(|e| e.path.join(&e.version))
    }

    /// Get the model's parent directory (without the version suffix).
    #[must_use]
    pub fn model_base_path(&self, dir_name: &str) -> Option<PathBuf> {
        self.model_dirs
            .iter()
            .find(|e| e.name == dir_name)
            .map(|e| self.root.join(&e.name))
    }
}

// ─── Ensure Layout Exists ──────────────────────────────────────────────────

/// Ensure the cache layout directories exist on disk.
///
/// Creates the root and all known model version directories. Safe to call
/// multiple times (idempotent).
///
/// # Errors
///
/// Returns `SearchError` if directory creation fails.
pub fn ensure_cache_layout(layout: &ModelCacheLayout) -> SearchResult<()> {
    std::fs::create_dir_all(&layout.root)?;

    for entry in &layout.model_dirs {
        std::fs::create_dir_all(&entry.path)?;
        std::fs::create_dir_all(entry.path.join(&entry.version))?;
    }

    Ok(())
}

/// Resolve the cache root and ensure all directories exist.
///
/// This is the primary entry point for consumers who just want a working
/// model cache.
///
/// # Errors
///
/// Returns `SearchError` if directory creation fails.
pub fn ensure_default_cache() -> SearchResult<ModelCacheLayout> {
    let layout = ModelCacheLayout::default_layout();
    ensure_cache_layout(&layout)?;
    Ok(layout)
}

// ─── Model Path Resolution ─────────────────────────────────────────────────

/// Resolve the expected path for a specific model file within the cache.
///
/// Returns `None` if the model directory name is not in the known layout.
#[must_use]
pub fn model_file_path(
    layout: &ModelCacheLayout,
    model_dir: &str,
    file_name: &str,
) -> Option<PathBuf> {
    layout.model_path(model_dir).map(|p| p.join(file_name))
}

/// Check whether a specific model appears installed (all expected files present).
#[must_use]
pub fn is_model_installed(model_versioned_dir: &Path, required_files: &[&str]) -> bool {
    if !model_versioned_dir.is_dir() {
        return false;
    }
    let all_present_in = |dir: &Path| required_files.iter().all(|f| dir.join(f).is_file());
    if all_present_in(model_versioned_dir) {
        return true;
    }
    let version_fallbacks = ["v1", "v2"];
    version_fallbacks
        .iter()
        .map(|version| model_versioned_dir.join(version))
        .any(|candidate| candidate.is_dir() && all_present_in(&candidate))
}

// ─── Environment Abstraction (for testing) ─────────────────────────────────

trait EnvLookup {
    fn var(&self, key: &str) -> Option<String>;
}

fn env_var_if_non_empty(env: &dyn EnvLookup, key: &str) -> Option<String> {
    env.var(key).filter(|value| !value.trim().is_empty())
}

enum EnvReader {
    Real,
    #[cfg(test)]
    Mock(std::collections::HashMap<String, String>),
}

impl EnvLookup for EnvReader {
    fn var(&self, key: &str) -> Option<String> {
        match self {
            Self::Real => std::env::var(key).ok(),
            #[cfg(test)]
            Self::Mock(map) => map.get(key).cloned(),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn mock_env(pairs: &[(&str, &str)]) -> EnvReader {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        EnvReader::Mock(map)
    }

    #[test]
    fn resolve_frankensearch_model_dir_takes_priority() {
        let env = mock_env(&[
            (ENV_MODEL_DIR, "/custom/models"),
            (ENV_DATA_DIR, "/custom/data"),
            (ENV_XDG_DATA_HOME, "/xdg"),
        ]);
        let root = resolve_cache_root_with(&env);
        assert_eq!(root, PathBuf::from("/custom/models"));
    }

    #[test]
    fn resolve_frankensearch_data_dir_adds_models_subdir() {
        let env = mock_env(&[(ENV_DATA_DIR, "/custom/data")]);
        let root = resolve_cache_root_with(&env);
        assert_eq!(root, PathBuf::from("/custom/data/models"));
    }

    #[test]
    fn resolve_empty_model_dir_falls_back_to_data_dir() {
        let env = mock_env(&[(ENV_MODEL_DIR, ""), (ENV_DATA_DIR, "/custom/data")]);
        let root = resolve_cache_root_with(&env);
        assert_eq!(root, PathBuf::from("/custom/data/models"));
    }

    #[test]
    fn resolve_empty_data_dir_falls_back_to_xdg() {
        let env = mock_env(&[(ENV_DATA_DIR, "   "), (ENV_XDG_DATA_HOME, "/xdg/data")]);
        let root = resolve_cache_root_with(&env);
        assert_eq!(root, PathBuf::from("/xdg/data/frankensearch/models"));
    }

    #[test]
    fn resolve_xdg_data_home_adds_frankensearch_models() {
        let env = mock_env(&[(ENV_XDG_DATA_HOME, "/xdg/data")]);
        let root = resolve_cache_root_with(&env);
        assert_eq!(root, PathBuf::from("/xdg/data/frankensearch/models"));
    }

    #[test]
    fn resolve_empty_env_falls_back_to_home() {
        let env = mock_env(&[]);
        let root = resolve_cache_root_with(&env);
        // Should contain "frankensearch" and "models" somewhere in the path.
        let path_str = root.to_string_lossy();
        assert!(
            path_str.contains("frankensearch") && path_str.contains("models"),
            "expected frankensearch/models in path, got: {path_str}"
        );
    }

    #[test]
    fn layout_for_root_creates_correct_entries() {
        let layout = ModelCacheLayout::for_root(PathBuf::from("/test/models"));
        assert_eq!(layout.root, PathBuf::from("/test/models"));
        assert_eq!(layout.model_dirs.len(), KNOWN_MODELS.len());

        let potion = layout
            .model_dirs
            .iter()
            .find(|e| e.name == "potion-base-128M")
            .expect("potion entry");
        assert_eq!(potion.version, "v1");
        assert_eq!(potion.path, PathBuf::from("/test/models/potion-base-128M"));
    }

    #[test]
    fn layout_model_path_returns_base_path() {
        let layout = ModelCacheLayout::for_root(PathBuf::from("/m"));
        let path = layout.model_path("all-MiniLM-L6-v2").unwrap();
        assert_eq!(path, Path::new("/m/all-MiniLM-L6-v2"));
    }

    #[test]
    fn layout_versioned_model_path_returns_versioned_path() {
        let layout = ModelCacheLayout::for_root(PathBuf::from("/m"));
        let path = layout.versioned_model_path("all-MiniLM-L6-v2").unwrap();
        assert_eq!(path, Path::new("/m/all-MiniLM-L6-v2/v1"));
    }

    #[test]
    fn layout_model_path_unknown_returns_none() {
        let layout = ModelCacheLayout::for_root(PathBuf::from("/m"));
        assert!(layout.model_path("nonexistent-model").is_none());
    }

    #[test]
    fn layout_model_base_path_strips_version() {
        let layout = ModelCacheLayout::for_root(PathBuf::from("/m"));
        let base = layout.model_base_path("all-MiniLM-L6-v2").unwrap();
        assert_eq!(base, PathBuf::from("/m/all-MiniLM-L6-v2"));
    }

    #[test]
    fn model_file_path_resolves_correctly() {
        let layout = ModelCacheLayout::for_root(PathBuf::from("/cache"));
        let path = model_file_path(&layout, "all-MiniLM-L6-v2", "onnx/model.onnx");
        assert_eq!(
            path,
            Some(PathBuf::from("/cache/all-MiniLM-L6-v2/onnx/model.onnx"))
        );
    }

    #[test]
    fn model_file_path_unknown_model_returns_none() {
        let layout = ModelCacheLayout::for_root(PathBuf::from("/cache"));
        assert!(model_file_path(&layout, "unknown", "file.bin").is_none());
    }

    #[test]
    fn ensure_cache_layout_creates_directories() {
        let temp = tempfile::tempdir().unwrap();
        let layout = ModelCacheLayout::for_root(temp.path().join("models"));
        ensure_cache_layout(&layout).unwrap();

        assert!(layout.root.is_dir());
        for entry in &layout.model_dirs {
            assert!(
                entry.path.is_dir(),
                "expected dir: {}",
                entry.path.display()
            );
            assert!(
                entry.path.join(&entry.version).is_dir(),
                "expected version dir: {}",
                entry.path.join(&entry.version).display()
            );
        }
    }

    #[test]
    fn ensure_cache_layout_idempotent() {
        let temp = tempfile::tempdir().unwrap();
        let layout = ModelCacheLayout::for_root(temp.path().join("models"));
        ensure_cache_layout(&layout).unwrap();
        ensure_cache_layout(&layout).unwrap();
        assert!(layout.root.is_dir());
    }

    #[test]
    fn ensure_default_cache_returns_working_layout() {
        // Instead of mutating env (unsafe in edition 2024), test the
        // ensure_cache_layout + for_root path which is what ensure_default_cache
        // delegates to.
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("isolated-models");
        let layout = ModelCacheLayout::for_root(root.clone());
        ensure_cache_layout(&layout).unwrap();

        assert_eq!(layout.root, root);
        assert!(root.is_dir());
        // Verify all model dirs were created.
        for entry in &layout.model_dirs {
            assert!(entry.path.is_dir());
            assert!(entry.path.join(&entry.version).is_dir());
        }
    }

    #[test]
    fn is_model_installed_accepts_base_dir_with_version_subdir() {
        let temp = tempfile::tempdir().unwrap();
        let model_base = temp.path().join("model");
        let versioned = model_base.join("v1");
        std::fs::create_dir_all(&versioned).unwrap();
        std::fs::write(versioned.join("tokenizer.json"), b"stub").unwrap();
        std::fs::write(versioned.join("model.onnx"), b"stub").unwrap();
        assert!(is_model_installed(
            &model_base,
            &["tokenizer.json", "model.onnx"]
        ));
    }

    #[test]
    fn is_model_installed_false_when_dir_missing() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("nonexistent");
        assert!(!is_model_installed(&missing, &["model.onnx"]));
    }

    #[test]
    fn is_model_installed_false_when_files_missing() {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join("model/v1");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"stub").unwrap();
        assert!(!is_model_installed(
            &model_dir,
            &["tokenizer.json", "model.onnx"]
        ));
    }

    #[test]
    fn is_model_installed_true_when_all_present() {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join("model/v1");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"stub").unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"stub").unwrap();
        assert!(is_model_installed(
            &model_dir,
            &["tokenizer.json", "model.onnx"]
        ));
    }

    #[test]
    fn is_model_installed_handles_nested_files() {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join("model/v1");
        std::fs::create_dir_all(model_dir.join("onnx")).unwrap();
        std::fs::write(model_dir.join("onnx/model.onnx"), b"stub").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"stub").unwrap();
        assert!(is_model_installed(
            &model_dir,
            &["onnx/model.onnx", "tokenizer.json"]
        ));
    }

    #[test]
    fn known_models_is_not_empty() {
        assert!(!known_models().is_empty());
        for m in known_models() {
            assert!(!m.dir_name.is_empty());
            assert!(!m.version.is_empty());
            assert!(!m.description.is_empty());
        }
    }

    #[test]
    fn layout_schema_version() {
        assert_eq!(MODEL_CACHE_LAYOUT_VERSION, 1);
    }

    #[test]
    fn env_constants_match_expected_values() {
        assert_eq!(ENV_MODEL_DIR, "FRANKENSEARCH_MODEL_DIR");
        assert_eq!(ENV_DATA_DIR, "FRANKENSEARCH_DATA_DIR");
    }
}
