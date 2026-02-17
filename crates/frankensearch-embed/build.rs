use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const SOURCE_OVERRIDE_ENV: &str = "FRANKENSEARCH_BUNDLED_MODELS_SOURCE_DIR";
const SKIP_DOWNLOAD_ENV: &str = "FRANKENSEARCH_BUNDLED_MODELS_SKIP_DOWNLOAD";

#[derive(Clone, Copy)]
struct FileSpec {
    relative_path: &'static str,
    url: &'static str,
    sha256: &'static str,
    size: u64,
}

#[derive(Clone, Copy)]
struct ModelSpec {
    manifest_id: &'static str,
    install_dir: &'static str,
    files: &'static [FileSpec],
}

const POTION_FILES: &[FileSpec] = &[
    FileSpec {
        relative_path: "tokenizer.json",
        url: "https://huggingface.co/minishlab/potion-multilingual-128M/resolve/a28f4eebecd4dc585034f605e52d414878a0417c/tokenizer.json",
        sha256: "19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1",
        size: 18_616_131,
    },
    FileSpec {
        relative_path: "model.safetensors",
        url: "https://huggingface.co/minishlab/potion-multilingual-128M/resolve/a28f4eebecd4dc585034f605e52d414878a0417c/model.safetensors",
        sha256: "14b5eb39cb4ce5666da8ad1f3dc6be4346e9b2d601c073302fa0a31bf7943397",
        size: 512_361_560,
    },
];

const MINILM_FILES: &[FileSpec] = &[
    FileSpec {
        relative_path: "onnx/model.onnx",
        url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx",
        sha256: "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        size: 90_405_214,
    },
    FileSpec {
        relative_path: "tokenizer.json",
        url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json",
        sha256: "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        size: 466_247,
    },
    FileSpec {
        relative_path: "config.json",
        url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json",
        sha256: "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
        size: 612,
    },
    FileSpec {
        relative_path: "special_tokens_map.json",
        url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/special_tokens_map.json",
        sha256: "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
        size: 112,
    },
    FileSpec {
        relative_path: "tokenizer_config.json",
        url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json",
        sha256: "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
        size: 350,
    },
];

const DEFAULT_MODELS: &[ModelSpec] = &[
    ModelSpec {
        manifest_id: "potion-multilingual-128m",
        install_dir: "potion-multilingual-128M",
        files: POTION_FILES,
    },
    ModelSpec {
        manifest_id: "all-minilm-l6-v2",
        install_dir: "all-MiniLM-L6-v2",
        files: MINILM_FILES,
    },
];

fn main() {
    println!("cargo:rerun-if-env-changed={SOURCE_OVERRIDE_ENV}");
    println!("cargo:rerun-if-env-changed={SKIP_DOWNLOAD_ENV}");
    println!("cargo:rerun-if-env-changed=FRANKENSEARCH_MODEL_DIR");
    println!("cargo:rerun-if-env-changed=FRANKENSEARCH_DATA_DIR");
    println!("cargo:rerun-if-env-changed=XDG_DATA_HOME");
    println!("cargo:rerun-if-env-changed=HOME");

    if env::var_os("CARGO_FEATURE_BUNDLED_DEFAULT_MODELS").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let bundled_root = out_dir.join("bundled-default-models");
    let generated_file = out_dir.join("bundled_default_models_generated.rs");
    let skip_download = env_truthy(SKIP_DOWNLOAD_ENV);
    let source_override = env::var_os(SOURCE_OVERRIDE_ENV).map(PathBuf::from);

    for model in DEFAULT_MODELS {
        for file in model.files {
            let destination = bundled_root
                .join(model.install_dir)
                .join(file.relative_path);
            if destination.is_file()
                && verify_file(&destination, file.size, file.sha256).unwrap_or(false)
            {
                continue;
            }

            fs::create_dir_all(
                destination
                    .parent()
                    .expect("destination path should always have parent"),
            )
            .expect("failed creating bundled-model destination directory");

            if let Some(local_source) =
                find_local_source_file(model, file, source_override.as_deref())
            {
                println!("cargo:rerun-if-changed={}", local_source.display());
                copy_with_validation(&local_source, &destination, file)
                    .expect("failed copying local bundled model file");
                continue;
            }

            assert!(
                !skip_download,
                "missing bundled model file {} for {} and {SKIP_DOWNLOAD_ENV}=1 blocked download",
                file.relative_path, model.install_dir
            );

            download_with_validation(file.url, &destination, file)
                .expect("failed downloading bundled model file");
        }
    }

    fs::write(&generated_file, generate_embedded_source(&bundled_root))
        .expect("failed writing bundled model generated source");
}

fn env_truthy(name: &str) -> bool {
    env::var(name).ok().is_some_and(|value| {
        let normalized = value.trim();
        normalized == "1"
            || normalized.eq_ignore_ascii_case("true")
            || normalized.eq_ignore_ascii_case("yes")
            || normalized.eq_ignore_ascii_case("on")
    })
}

fn find_local_source_file(
    model: &ModelSpec,
    file: &FileSpec,
    source_override: Option<&Path>,
) -> Option<PathBuf> {
    model_root_candidates(source_override)
        .into_iter()
        .map(|root| root.join(model.install_dir).join(file.relative_path))
        .find(|path| verify_file(path, file.size, file.sha256).unwrap_or(false))
}

fn model_root_candidates(source_override: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(path) = source_override {
        candidates.push(path.to_path_buf());
    }

    if let Some(path) = env::var_os("FRANKENSEARCH_MODEL_DIR") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("FRANKENSEARCH_DATA_DIR") {
        candidates.push(PathBuf::from(path).join("models"));
    }
    if let Some(path) = env::var_os("XDG_DATA_HOME") {
        candidates.push(PathBuf::from(path).join("frankensearch").join("models"));
    }
    if let Some(path) = env::var_os("HOME") {
        candidates.push(
            PathBuf::from(path)
                .join(".local")
                .join("share")
                .join("frankensearch")
                .join("models"),
        );
    }

    dedup_paths(candidates)
}

fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if !deduped.iter().any(|existing| existing == &path) {
            deduped.push(path);
        }
    }
    deduped
}

fn copy_with_validation(source: &Path, destination: &Path, file: &FileSpec) -> Result<(), String> {
    let temp = destination.with_extension(format!("tmp.{}", std::process::id()));
    if let Some(parent) = temp.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("mkdir {}: {err}", parent.display()))?;
    }
    fs::copy(source, &temp)
        .map_err(|err| format!("copy {} -> {}: {err}", source.display(), temp.display()))?;
    if !verify_file(&temp, file.size, file.sha256)? {
        return Err(format!(
            "copied file verification failed for {}",
            source.display()
        ));
    }
    if destination.exists() {
        fs::remove_file(destination)
            .map_err(|err| format!("remove stale {}: {err}", destination.display()))?;
    }
    fs::rename(&temp, destination).map_err(|err| {
        format!(
            "rename {} -> {}: {err}",
            temp.display(),
            destination.display()
        )
    })
}

fn download_with_validation(url: &str, destination: &Path, file: &FileSpec) -> Result<(), String> {
    let temp = destination.with_extension(format!("download.{}", std::process::id()));
    if let Some(parent) = temp.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("mkdir {}: {err}", parent.display()))?;
    }

    let status = Command::new("curl")
        .arg("--fail")
        .arg("--location")
        .arg("--silent")
        .arg("--show-error")
        .arg("--retry")
        .arg("8")
        .arg("--retry-delay")
        .arg("1")
        .arg("--retry-all-errors")
        .arg("--output")
        .arg(&temp)
        .arg(url)
        .status()
        .map_err(|err| format!("failed to spawn curl: {err}"))?;

    if !status.success() {
        return Err(format!(
            "curl download failed for {url} with status {status}"
        ));
    }

    if !verify_file(&temp, file.size, file.sha256)? {
        return Err(format!(
            "downloaded file failed verification for {} ({url})",
            file.relative_path
        ));
    }

    if destination.exists() {
        fs::remove_file(destination)
            .map_err(|err| format!("remove stale {}: {err}", destination.display()))?;
    }
    fs::rename(&temp, destination).map_err(|err| {
        format!(
            "rename {} -> {}: {err}",
            temp.display(),
            destination.display()
        )
    })
}

fn verify_file(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<bool, String> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(format!("stat {}: {err}", path.display())),
    };
    if metadata.len() != expected_size {
        return Ok(false);
    }

    let hash = sha256_hex_for_file(path)?;
    Ok(hash.eq_ignore_ascii_case(expected_sha256))
}

fn sha256_hex_for_file(path: &Path) -> Result<String, String> {
    let file = fs::File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8 * 1024];

    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|err| format!("read {}: {err}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn generate_embedded_source(bundled_root: &Path) -> String {
    let mut generated = String::new();
    generated.push_str("#[derive(Debug, Clone, Copy)]\n");
    generated.push_str("pub(crate) struct EmbeddedModelFile {\n");
    generated.push_str("    pub manifest_id: &'static str,\n");
    generated.push_str("    pub relative_path: &'static str,\n");
    generated.push_str("    pub sha256: &'static str,\n");
    generated.push_str("    pub size: u64,\n");
    generated.push_str("    pub bytes: &'static [u8],\n");
    generated.push_str("}\n\n");
    generated.push_str("pub(crate) static EMBEDDED_MODEL_FILES: &[EmbeddedModelFile] = &[\n");

    for model in DEFAULT_MODELS {
        for file in model.files {
            let path = bundled_root
                .join(model.install_dir)
                .join(file.relative_path);
            let path_literal = format!("{:?}", path.to_string_lossy());
            generated.push_str("    EmbeddedModelFile {\n");
            let _ = writeln!(generated, "        manifest_id: {:?},", model.manifest_id);
            let _ = writeln!(
                generated,
                "        relative_path: {:?},",
                file.relative_path
            );
            let _ = writeln!(generated, "        sha256: {:?},", file.sha256);
            let _ = writeln!(
                generated,
                "        size: {},",
                format_u64_with_underscores(file.size)
            );
            let _ = writeln!(generated, "        bytes: include_bytes!({path_literal}),");
            generated.push_str("    },\n");
        }
    }

    generated.push_str("];\n");
    generated
}

fn format_u64_with_underscores(value: u64) -> String {
    let digits = value.to_string();
    let mut with_separators = String::with_capacity(digits.len() + digits.len() / 3);
    for (seen, ch) in digits.chars().rev().enumerate() {
        if seen > 0 && seen.is_multiple_of(3) {
            with_separators.push('_');
        }
        with_separators.push(ch);
    }
    with_separators.chars().rev().collect()
}
