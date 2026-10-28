//! Explicit, offline import of one registered semantic model.
//! Acquisition verifies artifacts and real producer execution before publication.
//! It never selects or rebuilds an index or executes a core `RecoveryPlan`.

#![deny(unsafe_code)]

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs;
use std::io::{self, ErrorKind, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use asupersync::Cx;
use asupersync::runtime::RuntimeBuilder;
#[cfg(any(feature = "semantic-support", feature = "rerank"))]
use frankensearch_core::Embedder;
#[cfg(any(feature = "semantic-support", feature = "rerank"))]
use frankensearch_core::generation::QuantizationFormat;
use frankensearch_core::{SearchError, SearchResult};
#[cfg(feature = "semantic-loaders")]
use frankensearch_embed::FastEmbedEmbedder;
use frankensearch_embed::{
    ConsentSource, DownloadConsent, FrozenModelArtifactManifestV1, ModelAcquisitionReceiptV1,
    ModelAcquisitionRequest, ModelAcquisitionSource, ModelDownloader, ModelLifecycle,
    ModelManifest, RegisteredQualityModel,
};
#[cfg(feature = "semantic-support")]
use frankensearch_embed::{Model2VecEmbedder, RegisteredModel2Vec};
use frankensearch_fsfs::ShutdownCoordinator;
use frankensearch_fsfs::generation_store::COMPLETE_GENERATION_MANIFEST;
use frankensearch_fsfs::output_schema::{error_code_for, exit_code_for};
use serde::Serialize;

const MAX_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const HELP: &str = "\
fsfs-import-model --model MODEL --from-dir BUNDLE --output CACHE_ROOT [OPTIONS]

Import exactly one registered semantic model from a local artifact directory.
No network request, model fallback, configuration change, or index rebuild occurs.
The registered loader must pass its real execution certificate before publication.
A rejected attempt preserves the prior cache; replacements retain a backup.
A verified warm cache is load-tested and reused without reading the supplied bundle.

Required:
  --model MODEL       Exact registered model ID or documented quality-model alias
  --from-dir BUNDLE   Directory containing the model's pinned artifact tree
  --output CACHE_ROOT Model cache root used by fsfs (the model subdirectory is added)

Options:
  --format table|json Output format (default: table)
  --offline           Explicit acknowledgement; this command is always offline
  --help, -h          Show this help

Registered fast models: potion-multilingual-128m, potion-base-8m, potion-base-32m.
Registered quality models: all-minilm-l6-v2, snowflake-arctic-embed-s,
nomic-embed-text-v1.5, all-minilm-l6-v2-native, paraphrase-multilingual-minilm-l12-v2.
The executable must include the selected model's inference backend. Native English
MiniLM uses its separate all-MiniLM-L6-v2-native cache directory.

Example:
  fsfs-import-model --model potion-base-8m --from-dir /media/potion-base-8M \\
    --output /home/me/.local/share/frankensearch/models --offline

Success means acquired-needs-reindex, never semantic-searchable. Existing index
compatibility has not been assessed. Use fsfs-recover --rebuild-source with a query
preview to prepare and publish a compatible source rebuild separately.
This explicit importer does not consume the core planner's scoped authorization;
the RecoveryPlan acquisition capability guards remain in force.
";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Format {
    #[default]
    Table,
    Json,
}

#[derive(Debug)]
struct Options {
    model: String,
    source: PathBuf,
    output: PathBuf,
    format: Format,
}

impl Options {
    fn parse(args: &[OsString]) -> Result<Self, &'static str> {
        if args.len() > 16
            || args
                .iter()
                .any(|arg| arg.as_encoded_bytes().len() > MAX_ARGUMENT_BYTES)
        {
            return Err("The import command exceeds its argument limits.");
        }
        let mut model = None;
        let mut source = None;
        let mut output = None;
        let mut format = Format::Table;
        let mut seen = HashSet::new();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].to_str().ok_or("Option names must be UTF-8.")?;
            if !seen.insert(flag) {
                return Err("Each import option may be supplied only once.");
            }
            if flag == "--offline" {
                index += 1;
                continue;
            }
            if !matches!(flag, "--model" | "--from-dir" | "--output" | "--format") {
                return Err("Unknown import option. Use --help for the supported syntax.");
            }
            let value = args
                .get(index + 1)
                .filter(|value| !value.is_empty())
                .filter(|value| !value.as_encoded_bytes().starts_with(b"--"))
                .ok_or("An import option is missing its value.")?;
            match flag {
                "--model" => {
                    let name = value.to_str().ok_or("The model ID must be UTF-8.")?;
                    if name.trim().is_empty()
                        || name.len() > 128
                        || name.chars().any(char::is_control)
                    {
                        return Err("The model ID must be a bounded registered name.");
                    }
                    model = Some(name.to_owned());
                }
                "--from-dir" => source = Some(PathBuf::from(value)),
                "--output" => output = Some(PathBuf::from(value)),
                "--format" => {
                    format = match value.to_str() {
                        Some("table") => Format::Table,
                        Some("json") => Format::Json,
                        _ => return Err("Import output format must be table or json."),
                    };
                }
                _ => unreachable!("flag was checked above"),
            }
            index += 2;
        }
        Ok(Self {
            model: model.ok_or("--model is required; bulk import is not supported.")?,
            source: source.ok_or("--from-dir is required; imports never use the network.")?,
            output: output.ok_or("--output must name the explicit model cache root.")?,
            format,
        })
    }
}

enum SelectedModel {
    #[cfg(feature = "semantic-support")]
    Fast(Box<RegisteredModel2Vec>),
    Quality(RegisteredQualityModel),
}

impl SelectedModel {
    fn resolve(name: &str) -> SearchResult<Self> {
        #[cfg(feature = "semantic-support")]
        if let Some(registration) = RegisteredModel2Vec::builtin(name) {
            return registration.map(|model| Self::Fast(Box::new(model)));
        }
        #[cfg(not(feature = "semantic-support"))]
        if [
            "potion-multilingual-128m",
            "potion-base-8m",
            "potion-base-32m",
        ]
        .iter()
        .any(|id| name.trim().eq_ignore_ascii_case(id))
        {
            return Err(backend_unavailable());
        }
        let quality = RegisteredQualityModel::from_name(name).ok_or_else(|| {
            invalid("model", "Select one exact registered semantic model; fuzzy, bulk and reranker selections are unsupported.")
        })?;
        let available = if quality.is_native() {
            cfg!(feature = "rerank")
        } else {
            cfg!(feature = "semantic-loaders")
        };
        if !available {
            return Err(backend_unavailable());
        }
        Ok(Self::Quality(quality))
    }

    fn download_manifest(&self) -> ModelManifest {
        match self {
            #[cfg(feature = "semantic-support")]
            Self::Fast(model) => model.download_manifest().clone(),
            Self::Quality(model) => model.download_manifest(),
        }
    }

    fn frozen_manifest(&self) -> SearchResult<FrozenModelArtifactManifestV1> {
        match self {
            #[cfg(feature = "semantic-support")]
            Self::Fast(model) => model.artifact_manifest().freeze(),
            Self::Quality(model) => model.artifact_manifest()?.freeze(),
        }
    }

    fn install_directory(&self) -> &str {
        match self {
            #[cfg(feature = "semantic-support")]
            Self::Fast(model) => match model.download_manifest().id.as_str() {
                "potion-multilingual-128m" => "potion-multilingual-128M",
                "potion-base-8m" => "potion-base-8M",
                "potion-base-32m" => "potion-base-32M",
                // Only tests supply a custom registration to this private type.
                id => id,
            },
            Self::Quality(RegisteredQualityModel::MiniLmOnnx) => "all-MiniLM-L6-v2",
            Self::Quality(RegisteredQualityModel::MiniLmNative) => "all-MiniLM-L6-v2-native",
            Self::Quality(RegisteredQualityModel::MultilingualNative) => {
                "paraphrase-multilingual-MiniLM-L12-v2"
            }
            Self::Quality(RegisteredQualityModel::SnowflakeOnnx) => "snowflake-arctic-embed-s",
            Self::Quality(RegisteredQualityModel::NomicOnnx) => "nomic-embed-text-v1.5",
        }
    }

    fn load_self_test(
        &self,
        directory: &Path,
        frozen: &FrozenModelArtifactManifestV1,
    ) -> SearchResult<()> {
        frozen.validate()?;
        // These constructors execute the registered golden-vector corpus and
        // verify its independent certificate; a byte-only check is insufficient.
        match self {
            #[cfg(feature = "semantic-support")]
            Self::Fast(model) => {
                let loaded = Model2VecEmbedder::load_registered(directory, model)?;
                validate_loaded_identity(&loaded, frozen)
            }
            Self::Quality(
                model @ (RegisteredQualityModel::MiniLmNative
                | RegisteredQualityModel::MultilingualNative),
            ) => {
                #[cfg(feature = "rerank")]
                {
                    use frankensearch_rerank::{NativeEmbedder, NativeEmbeddingModel};
                    let profile = match model {
                        RegisteredQualityModel::MiniLmNative => {
                            NativeEmbeddingModel::AllMiniLmL6V2F32
                        }
                        _ => NativeEmbeddingModel::ParaphraseMultilingualMiniLmL12V2,
                    };
                    let loaded = NativeEmbedder::load_model(directory, profile)?;
                    validate_loaded_identity(&loaded, frozen)
                }
                #[cfg(not(feature = "rerank"))]
                {
                    let _ = (directory, model);
                    Err(backend_unavailable())
                }
            }
            Self::Quality(
                model @ (RegisteredQualityModel::MiniLmOnnx
                | RegisteredQualityModel::SnowflakeOnnx
                | RegisteredQualityModel::NomicOnnx),
            ) => {
                #[cfg(feature = "semantic-loaders")]
                {
                    let loaded = FastEmbedEmbedder::load_with_name(directory, model.manifest_id())?;
                    validate_loaded_identity(&loaded, frozen)
                }
                #[cfg(not(feature = "semantic-loaders"))]
                {
                    let _ = (directory, model);
                    Err(backend_unavailable())
                }
            }
        }
    }
}

#[cfg(any(feature = "semantic-support", feature = "rerank"))]
fn validate_loaded_identity(
    loaded: &dyn Embedder,
    frozen: &FrozenModelArtifactManifestV1,
) -> SearchResult<()> {
    let expected = frozen
        .manifest
        .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")?;
    if loaded.identity()? != &expected {
        return Err(invalid(
            "producer_identity",
            "The loaded producer does not match the selected frozen model contract.",
        ));
    }
    Ok(())
}

fn invalid(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("cli.import_model.{field}"),
        value: "rejected".to_owned(),
        reason: reason.to_owned(),
    }
}

fn backend_unavailable() -> SearchError {
    SearchError::EmbedderUnavailable {
        model: "selected-semantic-model".to_owned(),
        reason: "This executable lacks the selected model's inference backend; use a compatible loader-capable fsfs build.".to_owned(),
    }
}

// Inspect canonical ancestry before mkdir or lock creation. A cache may not
// mutate an immutable fsfs generation, including through a symlink alias.
fn ensure_unsealed_destination(destination: &Path) -> SearchResult<()> {
    let absolute = std::path::absolute(destination)?;
    for ancestor in absolute.ancestors() {
        match fs::canonicalize(ancestor) {
            Ok(existing) => {
                for directory in existing.ancestors() {
                    match fs::symlink_metadata(directory.join(COMPLETE_GENERATION_MANIFEST)) {
                        Ok(_) => {
                            return Err(invalid(
                                "output",
                                "The model cache cannot be inside a sealed search generation.",
                            ));
                        }
                        Err(error) if error.kind() == ErrorKind::NotFound => {}
                        Err(error) => return Err(error.into()),
                    }
                }
                return Ok(());
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                if ancestor.components().next_back() == Some(Component::ParentDir) {
                    return Err(invalid(
                        "output",
                        "The cache path traverses an unresolved parent directory.",
                    ));
                }
            }
            Err(error) => return Err(error.into()),
        }
    }
    Err(invalid(
        "output",
        "The cache path has no existing ancestor.",
    ))
}

async fn import_model(
    cx: &Cx,
    selected: &SelectedModel,
    source: &Path,
    cache_root: &Path,
) -> SearchResult<ModelAcquisitionReceiptV1> {
    let frozen = selected.frozen_manifest()?;
    let mut manifest = selected.download_manifest();
    if matches!(
        selected,
        SelectedModel::Quality(RegisteredQualityModel::MiniLmNative)
    ) {
        // The native catalog id selects a separate installation directory.
        // The pinned weights retain their registered logical model identity;
        // acquisition must bind that identity, not relabel the frozen model.
        // Artifact hashes, sizes, revision, URLs and license remain unchanged.
        manifest.id.clone_from(&frozen.manifest.logical_model_id);
    }
    let source = std::path::absolute(source)?;
    let cache_root = std::path::absolute(cache_root)?;
    let destination = cache_root.join(selected.install_directory());
    ensure_unsealed_destination(&destination)?;
    cx.checkpoint().map_err(|_| SearchError::Cancelled {
        phase: "model_import".to_owned(),
        reason: "Local model import cancelled.".to_owned(),
    })?;
    fs::create_dir_all(&cache_root)?;
    let mut lifecycle = ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    );
    ModelDownloader::with_defaults()
        .acquire_frozen_model(
            cx,
            ModelAcquisitionRequest {
                frozen_manifest: &frozen,
                source: ModelAcquisitionSource::LocalBundle(&source),
                destination_dir: &destination,
            },
            &mut lifecycle,
            |_| {},
            |directory, contract| selected.load_self_test(directory, contract),
        )
        .await
}

fn run(options: Options) -> SearchResult<ModelAcquisitionReceiptV1> {
    // Resolve the exact model and compiled loader before creating any cache.
    let selected = SelectedModel::resolve(&options.model)?;
    let scheduler = RuntimeBuilder::current_thread()
        .blocking_threads(0, 2)
        .build()
        .map_err(|source| SearchError::SubsystemError {
            subsystem: "fsfs.model_import.runtime",
            source: Box::new(io::Error::other(source.to_string())),
        })?;
    let shutdown = Arc::new(ShutdownCoordinator::new());
    shutdown.register_signals()?;
    let worker_shutdown = Arc::clone(&shutdown);
    let task = scheduler.handle().spawn(async move {
        let cx = Cx::current().expect("runtime installs a request context");
        let _cancellation = worker_shutdown.cancellation_scope(&cx);
        import_model(&cx, &selected, &options.source, &options.output).await
    });
    let result = scheduler.block_on(task);
    shutdown.stop_signal_listener();
    // No late cancellation may turn a completed publication into an abort.
    drop(scheduler);
    result
}

#[derive(Serialize)]
struct Failure {
    code: &'static str,
    message: &'static str,
}

#[derive(Serialize)]
struct Response {
    schema_version: u16,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    postcondition: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acquisition: Option<ModelAcquisitionReceiptV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Failure>,
}

fn failure_message(error: &SearchError) -> &'static str {
    match error {
        SearchError::EmbedderUnavailable { .. } => {
            "The selected inference backend is absent. Use a compatible loader-capable fsfs build."
        }
        SearchError::ModelLoadFailed { .. } | SearchError::EmbeddingFailed { .. } => {
            "The registered model failed its load or execution self-test. The prior cache is retained."
        }
        SearchError::HashMismatch { .. } | SearchError::ModelNotFound { .. } => {
            "The local bundle is missing or does not match the exact registered artifacts."
        }
        SearchError::InvalidConfig { .. } => {
            "The model, bundle, destination or execution contract was rejected. Check --help and the exact registered model."
        }
        SearchError::Cancelled { .. } => "Local model import was cancelled.",
        _ => "Model import failed. Reverify the selected cache before retrying.",
    }
}

fn failure_response(error: &SearchError, message: &'static str) -> (Response, u8) {
    (
        Response {
            schema_version: 1,
            ok: false,
            postcondition: None,
            acquisition: None,
            error: Some(Failure {
                code: error_code_for(error),
                message,
            }),
        },
        u8::try_from(exit_code_for(error)).unwrap_or(1),
    )
}

fn render(response: &Response, format: Format) -> io::Result<Vec<u8>> {
    let mut bytes = if format == Format::Json {
        serde_json::to_vec(response).map_err(io::Error::other)?
    } else if let Some(receipt) = &response.acquisition {
        format!(
            "Model: {}\nSource: {:?}\nOutcome: {:?}\nVerified bytes: {}\n\
             Manifest SHA-256: {}\nPrior generation preserved: {}\n\
             State: acquired-needs-reindex. Existing index compatibility has not been assessed.\n",
            receipt.logical_model_id,
            receipt.source,
            receipt.outcome,
            receipt.bytes_verified,
            receipt.artifact_manifest_fingerprint,
            receipt.prior_generation_preserved,
        )
        .into_bytes()
    } else {
        let error = response
            .error
            .as_ref()
            .expect("failure contains a diagnostic");
        format!("fsfs-import-model: {} [{}]\n", error.message, error.code).into_bytes()
    };
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err(io::Error::other(
            "model import response exceeds its byte limit",
        ));
    }
    if !bytes.ends_with(b"\n") {
        bytes.push(b'\n');
    }
    Ok(bytes)
}

fn main() -> ExitCode {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 1 && matches!(args[0].to_str(), Some("--help" | "-h")) {
        return if io::stdout().lock().write_all(HELP.as_bytes()).is_ok() {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }
    let mut format = if args
        .windows(2)
        .any(|pair| pair[0] == "--format" && pair[1] == "json")
    {
        Format::Json
    } else {
        Format::Table
    };
    let (response, exit) = match Options::parse(&args) {
        Ok(options) => {
            format = options.format;
            match run(options) {
                Ok(receipt) => (
                    Response {
                        schema_version: 1,
                        ok: true,
                        postcondition: Some("model_acquired_needs_reindex"),
                        acquisition: Some(receipt),
                        error: None,
                    },
                    0,
                ),
                Err(error) => failure_response(&error, failure_message(&error)),
            }
        }
        Err(message) => failure_response(&invalid("arguments", message), message),
    };
    let emitted = render(&response, format).and_then(|bytes| {
        let mut output = io::stdout().lock();
        output.write_all(&bytes)?;
        output.flush()
    });
    if emitted.is_err() {
        eprintln!(
            "fsfs-import-model: output failed; import may already be visible. Reverify the cache before retrying."
        );
        return ExitCode::FAILURE;
    }
    ExitCode::from(exit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> Vec<OsString> {
        [
            "--model",
            "potion-base-8m",
            "--from-dir",
            "bundle",
            "--output",
            "cache",
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }

    #[test]
    fn explicit_local_import_requires_one_complete_unambiguous_request() {
        let options = Options::parse(&arguments()).unwrap();
        assert_eq!(options.model, "potion-base-8m");
        assert_eq!(options.source, Path::new("bundle"));
        for suffix in [
            vec!["--model", "minilm"],
            vec!["--from-dir", "other"],
            vec!["--force"],
            vec!["--all"],
            vec!["--online"],
            vec!["--verify"],
            vec!["--unknown"],
            vec!["--format", "csv"],
            vec!["--format"],
        ] {
            let mut args = arguments();
            args.extend(suffix.into_iter().map(OsString::from));
            assert!(Options::parse(&args).is_err());
        }
        for index in [1, 3, 5] {
            let mut args = arguments();
            args[index] = OsString::new();
            assert!(Options::parse(&args).is_err());
        }
        assert!(Options::parse(&[]).is_err());
        let mut args = arguments();
        args.extend(["--offline", "--format", "json"].map(OsString::from));
        assert_eq!(Options::parse(&args).unwrap().format, Format::Json);
        for name in ["all", "*", "potion", "MiniLM-L6", "ms-marco-minilm-l-6-v2"] {
            assert!(SelectedModel::resolve(name).is_err());
        }
    }

    #[test]
    fn sealed_generation_and_alias_cannot_be_used_as_a_cache() {
        let temp = tempfile::tempdir().unwrap();
        let generation = temp.path().join("generation");
        fs::create_dir(&generation).unwrap();
        fs::write(generation.join(COMPLETE_GENERATION_MANIFEST), b"sealed").unwrap();
        let output = generation.join("absent-cache/model");
        assert!(ensure_unsealed_destination(&output).is_err());
        assert!(!generation.join("absent-cache").exists());
        #[cfg(unix)]
        {
            let alias = temp.path().join("alias");
            std::os::unix::fs::symlink(&generation, &alias).unwrap();
            assert!(ensure_unsealed_destination(&alias.join("cache/model")).is_err());
        }
    }

    #[test]
    fn failure_output_never_formats_private_error_details() {
        let error = SearchError::InvalidConfig {
            field: "private-path".to_owned(),
            value: "secret-token".to_owned(),
            reason: "sensitive\nprovider body".to_owned(),
        };
        let (response, _) = failure_response(&error, failure_message(&error));
        for format in [Format::Table, Format::Json] {
            let rendered = String::from_utf8(render(&response, format).unwrap()).unwrap();
            for private in ["private-path", "secret-token", "provider body", "sensitive"] {
                assert!(!rendered.contains(private));
            }
        }
    }

    #[test]
    fn native_catalog_alias_is_admitted_as_its_frozen_logical_model() {
        let selected = SelectedModel::Quality(RegisteredQualityModel::MiniLmNative);
        assert_eq!(selected.install_directory(), "all-MiniLM-L6-v2-native");
        assert_eq!(selected.download_manifest().id, "all-minilm-l6-v2-native");
        assert_eq!(
            selected
                .frozen_manifest()
                .unwrap()
                .manifest
                .logical_model_id,
            "all-minilm-l6-v2"
        );
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let scheduler = RuntimeBuilder::current_thread().build().unwrap();
        scheduler.block_on(async {
            let error = import_model(
                &Cx::for_request(),
                &selected,
                &temp.path().join("missing-local-bundle"),
                &cache,
            )
            .await
            .unwrap_err();
            assert!(!matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "model_acquisition.lifecycle_manifest"
            ));
            // The acquisition lock is created only AFTER the exact lifecycle
            // manifest has passed admission. No source bytes were available.
            let destination = cache.join(selected.install_directory());
            let observed = frankensearch_embed::diagnose_model_acquisition(&destination).unwrap();
            assert!(observed.advisory_lock_present);
            assert!(!destination.exists());
        });
    }

    #[cfg(feature = "semantic-support")]
    mod native_fixture {
        use frankensearch_core::generation::GoldenVectorCertificateV1;
        use frankensearch_embed::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;
        use frankensearch_embed::{ModelFile, ModelTier};
        use sha2::{Digest, Sha256};
        use std::fmt::Write as _;

        use super::*;

        // The expected vectors are analytical unit vectors, independent of
        // the real tokenizer/matrix loader and conformance checks under test.
        fn fixture(certificate_axis: usize) -> (tempfile::TempDir, SelectedModel) {
            let source = tempfile::tempdir().unwrap();
            let tokenizer = serde_json::json!({
                "version": "1.0", "truncation": null, "padding": null,
                "added_tokens": [], "normalizer": {"type": "Lowercase"},
                "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
                "decoder": null,
                "model": {"type": "WordLevel",
                          "vocab": {"[UNK]": 0, "hello": 1, "world": 2},
                          "unk_token": "[UNK]"}
            });
            fs::write(
                source.path().join("tokenizer.json"),
                serde_json::to_vec(&tokenizer).unwrap(),
            )
            .unwrap();
            let data = [1.0_f32, 0.0].repeat(3);
            let data = data
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let mut header = serde_json::to_vec(&serde_json::json!({
                "embeddings": {"dtype": "F32", "shape": [3, 2],
                               "data_offsets": [0, data.len()]}
            }))
            .unwrap();
            while header.len() % 8 != 0 {
                header.push(b' ');
            }
            let mut weights = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
            weights.extend_from_slice(&header);
            weights.extend_from_slice(&data);
            fs::write(source.path().join("model.safetensors"), weights).unwrap();
            let files = ["tokenizer.json", "model.safetensors"]
                .into_iter()
                .map(|name| {
                    let bytes = fs::read(source.path().join(name)).unwrap();
                    let sha256 = Sha256::digest(&bytes).iter().fold(
                        String::with_capacity(64),
                        |mut hex, byte| {
                            write!(hex, "{byte:02x}").unwrap();
                            hex
                        },
                    );
                    ModelFile {
                        name: name.to_owned(),
                        sha256,
                        size: u64::try_from(bytes.len()).unwrap(),
                        url: None,
                    }
                })
                .collect::<Vec<_>>();
            let manifest = ModelManifest {
                id: "offline-import-fixture".to_owned(),
                version: "1".to_owned(),
                display_name: None,
                description: None,
                repo: "fixtures/offline-import".to_owned(),
                revision: "a".repeat(40),
                download_size_bytes: files.iter().map(|file| file.size).sum(),
                files,
                license: "MIT".to_owned(),
                dimension: Some(2),
                tier: Some(ModelTier::Fast),
            };
            let mut unit = vec![0.0_f32; 2];
            unit[certificate_axis] = 1.0;
            let expected = vec![unit; MODEL_CONFORMANCE_TEXTS_V1.len()];
            let certificate =
                GoldenVectorCertificateV1::from_exact_f32(&MODEL_CONFORMANCE_TEXTS_V1, &expected)
                    .unwrap();
            let registered =
                RegisteredModel2Vec::new(manifest, "offline-import-fixture", certificate).unwrap();
            (source, SelectedModel::Fast(Box::new(registered)))
        }

        #[test]
        fn real_local_import_loads_then_reuses_cache_without_source() {
            let (source, selected) = fixture(0);
            let temp = tempfile::tempdir().unwrap();
            let cache = temp.path().join("cache");
            let scheduler = RuntimeBuilder::current_thread().build().unwrap();
            scheduler.block_on(async {
                let cx = Cx::for_request();
                let receipt = import_model(&cx, &selected, source.path(), &cache)
                    .await
                    .unwrap();
                assert_eq!(
                    receipt.source,
                    frankensearch_embed::ModelAcquisitionSourceKindV1::LocalBundle
                );
                assert!(receipt.requires_reindex);
                assert!(receipt.source_hosts.is_empty());
                let destination = cache.join(selected.install_directory());
                let original = fs::read(destination.join("model.safetensors")).unwrap();
                let moved = temp.path().join("source-moved-away");
                fs::rename(source.path(), &moved).unwrap();
                let warm = import_model(&cx, &selected, source.path(), &cache)
                    .await
                    .unwrap();
                assert_eq!(
                    warm.source,
                    frankensearch_embed::ModelAcquisitionSourceKindV1::WarmCache
                );
                assert_eq!(
                    warm.outcome,
                    frankensearch_embed::ModelAcquisitionOutcomeV1::VerifiedWarmCache
                );
                assert_eq!(
                    fs::read(destination.join("model.safetensors")).unwrap(),
                    original
                );
            });
        }

        #[test]
        fn bad_execution_certificate_retains_prior_cache_then_valid_import_preserves_backup() {
            let (source, wrong) = fixture(1);
            let temp = tempfile::tempdir().unwrap();
            let cache = temp.path().join("cache");
            let destination = cache.join(wrong.install_directory());
            fs::create_dir_all(&destination).unwrap();
            fs::write(destination.join("prior-bytes"), b"keep exactly").unwrap();
            let scheduler = RuntimeBuilder::current_thread().build().unwrap();
            scheduler.block_on(async {
                let cx = Cx::for_request();
                assert!(
                    import_model(&cx, &wrong, source.path(), &cache)
                        .await
                        .is_err()
                );
                assert_eq!(
                    fs::read(destination.join("prior-bytes")).unwrap(),
                    b"keep exactly"
                );
                let (valid_source, valid) = fixture(0);
                let receipt = import_model(&cx, &valid, valid_source.path(), &cache)
                    .await
                    .unwrap();
                assert!(receipt.prior_generation_preserved);
                let recovery =
                    frankensearch_embed::diagnose_model_acquisition(&destination).unwrap();
                assert_eq!(recovery.preserved_backup_generations, 1);
                assert!(recovery.orphan_staging_generations >= 1);
                let backup = fs::read_dir(&cache)
                    .unwrap()
                    .map(Result::unwrap)
                    .find(|entry| entry.file_name().to_string_lossy().contains(".backup."))
                    .unwrap();
                assert_eq!(
                    fs::read(backup.path().join("prior-bytes")).unwrap(),
                    b"keep exactly"
                );
            });
        }
    }
}
