//! Explicit, offline import of one registered semantic model.
//! Acquisition verifies artifacts and real producer execution before publication.
//! An explicit scoped mode executes a bound core `RecoveryPlan`; neither mode
//! selects or rebuilds an index.

#![recursion_limit = "512"]
#![deny(unsafe_code)]

use std::collections::{HashSet, hash_map::RandomState};
use std::ffi::OsString;
use std::fs;
use std::hash::BuildHasher;
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use asupersync::runtime::RuntimeBuilder;
#[cfg(any(feature = "semantic-support", feature = "rerank"))]
use frankensearch_core::Embedder;
#[cfg(any(feature = "semantic-support", feature = "rerank"))]
use frankensearch_core::generation::QuantizationFormat;
use frankensearch_core::recovery_plan::{
    InteractionPolicy, MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS,
    ModelAcquisitionAuthorization, ModelAcquisitionTarget, NetworkPolicy, RecoveryPlan,
    RecoveryPolicy, RecoveryRequest, RequestMode, SemanticReadiness, TrustedRecoveryContext, plan,
};
use frankensearch_core::{RetrievalTopology, SearchError, SearchResult};
#[cfg(feature = "semantic-loaders")]
use frankensearch_embed::FastEmbedEmbedder;
use frankensearch_embed::{
    ConsentSource, DownloadConsent, FrozenModelArtifactManifestV1, ModelAcquisitionReceiptV1,
    ModelAcquisitionRequest, ModelAcquisitionSource, ModelDownloader, ModelLifecycle,
    ModelManifest, ModelPublicationUncertainOutcomeV1, ModelPublicationUncertaintyV1,
    RegisteredQualityModel, local_model_acquisition_target, model_publication_uncertainty,
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
  --plan              Inspect a missing/corrupt cache and print exact consent; no writes
  --authorization FILE Execute the exact standalone local-import authorization in FILE
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
Scoped recovery:
  Repeat the same model/source/output options with --plan --format json, review
  its authorization object, and save that object alone as consent.json. Then run
  the original command with --authorization consent.json within its expiry.
  --plan and --authorization are mutually exclusive. No grant is inferred from
  --plan. Authorization binds the exact model, revision, producer manifest,
  artifact bytes, semantic space and symlink-free absolute cache destination.
  Corpus and reindex grants are rejected. Use canonical paths (no symlink aliases).
  Byte-valid caches use the ordinary import command for load testing and warm
  reuse; the scoped path cannot infer a backend failure from verified bytes.
  The generic RecoveryPlan acquisition capability guards remain in force; this
  command executes only the separately bound standalone local-import capability.
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
    plan_only: bool,
    authorization: Option<PathBuf>,
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
        let mut plan_only = false;
        let mut authorization = None;
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
            if flag == "--plan" {
                plan_only = true;
                index += 1;
                continue;
            }
            if !matches!(
                flag,
                "--model" | "--from-dir" | "--output" | "--format" | "--authorization"
            ) {
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
                "--authorization" => authorization = Some(PathBuf::from(value)),
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
        if plan_only && authorization.is_some() {
            return Err("--plan cannot execute or be combined with --authorization.");
        }
        Ok(Self {
            model: model.ok_or("--model is required; bulk import is not supported.")?,
            source: source.ok_or("--from-dir is required; imports never use the network.")?,
            output: output.ok_or("--output must name the explicit model cache root.")?,
            format,
            plan_only,
            authorization,
        })
    }
}

enum SelectedModel {
    #[cfg(feature = "semantic-support")]
    Fast(Box<RegisteredModel2Vec>),
    Quality(RegisteredQualityModel),
}

impl SelectedModel {
    const fn tier(&self) -> frankensearch_core::ModelTier {
        match self {
            #[cfg(feature = "semantic-support")]
            Self::Fast(_) => frankensearch_core::ModelTier::Fast,
            Self::Quality(_) => frankensearch_core::ModelTier::Quality,
        }
    }

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
        #[cfg(not(feature = "rerank"))]
        if quality.is_native() {
            return Err(backend_unavailable());
        }
        #[cfg(not(feature = "semantic-loaders"))]
        if !quality.is_native() {
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

#[cfg(any(not(feature = "rerank"), not(feature = "semantic-loaders")))]
fn backend_unavailable() -> SearchError {
    SearchError::EmbedderUnavailable {
        model: "selected-semantic-model".to_owned(),
        reason: "This executable lacks the selected model's inference backend; use a compatible loader-capable fsfs build.".to_owned(),
    }
}

// Inspect canonical ancestry before mkdir or lock creation. A cache may not
// mutate a sealed fsfs generation or fixed-authority root, including through
// a symlink alias.
fn ensure_unsealed_destination(destination: &Path) -> SearchResult<()> {
    let absolute = std::path::absolute(destination)?;
    for ancestor in absolute.ancestors() {
        match fs::canonicalize(ancestor) {
            Ok(existing) => {
                for directory in existing.ancestors() {
                    for marker in [
                        COMPLETE_GENERATION_MANIFEST,
                        frankensearch_index::generation_root::GENERATION_ROOT_AUTHORITY_FILE_NAME,
                    ] {
                        match fs::symlink_metadata(directory.join(marker)) {
                            Ok(_) => {
                                return Err(invalid(
                                    "output",
                                    "The model cache cannot be inside a sealed search generation or fixed-authority root.",
                                ));
                            }
                            Err(error) if error.kind() == ErrorKind::NotFound => {}
                            Err(error) => return Err(error.into()),
                        }
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

fn import_lifecycle(
    selected: &SelectedModel,
    frozen: &FrozenModelArtifactManifestV1,
) -> ModelLifecycle {
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
    ModelLifecycle::new(
        manifest,
        DownloadConsent::granted(ConsentSource::Programmatic),
    )
}

fn current_time() -> SearchResult<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| {
            invalid(
                "clock",
                "A trusted current authorization time is unavailable.",
            )
        })
}

fn read_authorization(path: &Path) -> SearchResult<ModelAcquisitionAuthorization> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(u64::try_from(MAX_RESPONSE_BYTES + 1).expect("fixed bound fits u64"))
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err(invalid(
            "authorization",
            "The authorization exceeds its byte limit.",
        ));
    }
    serde_json::from_slice(&bytes).map_err(|_| {
        invalid(
            "authorization",
            "The authorization must be one exact v3 model-acquisition grant.",
        )
    })
}

fn recovery_import_plan(
    cx: &Cx,
    selected: &SelectedModel,
    cache_root: &Path,
    authorization: Option<ModelAcquisitionAuthorization>,
) -> SearchResult<(RecoveryPlan, ModelAcquisitionTarget)> {
    let now = current_time()?;
    if let Some(grant) = &authorization {
        grant.validate_at(now).map_err(|_| {
            invalid(
                "authorization",
                "The authorization is invalid or expired; inspect a fresh --plan.",
            )
        })?;
    }
    let (issued, expires, nonce) = if let Some(grant) = &authorization {
        (
            grant.issued_at_unix_seconds,
            grant.expires_at_unix_seconds,
            grant.nonce.clone(),
        )
    } else {
        let expires = now
            .checked_add(MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS)
            .ok_or_else(|| invalid("clock", "The authorization window cannot be represented."))?;
        // This public nonce distinguishes scoped grants; it is not a bearer
        // secret. Separate process-randomized hash seeds provide 128 bits.
        let nonce = format!(
            "{:016x}{:016x}",
            RandomState::new().hash_one((now, std::process::id(), 0_u8)),
            RandomState::new().hash_one((now, std::process::id(), 1_u8))
        );
        (now, expires, nonce)
    };
    let frozen = selected.frozen_manifest()?;
    let destination = std::path::absolute(cache_root)?.join(selected.install_directory());
    ensure_unsealed_destination(&destination)?;
    let target = local_model_acquisition_target(
        &frozen,
        selected.tier(),
        &destination,
        issued,
        expires,
        nonce,
    )?;
    cx.checkpoint().map_err(|_| SearchError::Cancelled {
        phase: "model_import_plan".to_owned(),
        reason: "Local model import planning cancelled.".to_owned(),
    })?;
    let state = match fs::symlink_metadata(&destination) {
        Err(error) if error.kind() == ErrorKind::NotFound => SemanticReadiness::ModelMissing {
            tier: selected.tier(),
        },
        Err(error) => return Err(error.into()),
        Ok(_) => {
            // A loader may write a verification marker. Planning only inspects
            // frozen bytes and never invokes it or diagnoses a backend failure.
            let verified = frozen.manifest.verify_dir(&destination).is_ok();
            cx.checkpoint().map_err(|_| SearchError::Cancelled {
                phase: "model_import_plan".to_owned(),
                reason: "Local model import planning cancelled.".to_owned(),
            })?;
            if verified {
                return Err(invalid(
                    "readiness",
                    "The frozen cache bytes are already verified. Use the ordinary import command for load testing and warm reuse.",
                ));
            }
            SemanticReadiness::ModelUnloadable {
                tier: selected.tier(),
            }
        }
    };
    cx.checkpoint().map_err(|_| SearchError::Cancelled {
        phase: "model_import_plan".to_owned(),
        reason: "Local model import planning cancelled.".to_owned(),
    })?;
    let policy = RecoveryPolicy {
        interaction: InteractionPolicy::NonInteractive,
        network: NetworkPolicy::Offline,
        acquisition_authorization: authorization,
    };
    let request = RecoveryRequest {
        mode: RequestMode::ExplicitSemantic,
        requested_topology: match selected.tier() {
            frankensearch_core::ModelTier::Fast => RetrievalTopology::FastOnly,
            frankensearch_core::ModelTier::Quality => RetrievalTopology::QualityOnly,
        },
    };
    let recovery = plan(TrustedRecoveryContext::new(
        &state, request, &policy, Some(&target), current_time()?,
    )).map_err(|_| invalid("authorization", "The authorization does not match this exact local model and cache; inspect a fresh --plan."))?;
    Ok((recovery, target))
}

async fn import_model(
    cx: &Cx,
    selected: &SelectedModel,
    source: &Path,
    cache_root: &Path,
) -> SearchResult<ModelAcquisitionReceiptV1> {
    let frozen = selected.frozen_manifest()?;
    let source = std::path::absolute(source)?;
    let cache_root = std::path::absolute(cache_root)?;
    let destination = cache_root.join(selected.install_directory());
    ensure_unsealed_destination(&destination)?;
    cx.checkpoint().map_err(|_| SearchError::Cancelled {
        phase: "model_import".to_owned(),
        reason: "Local model import cancelled.".to_owned(),
    })?;
    fs::create_dir_all(&cache_root)?;
    let mut lifecycle = import_lifecycle(selected, &frozen);
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

async fn import_authorized_model(
    cx: &Cx,
    selected: &SelectedModel,
    source: &Path,
    cache_root: &Path,
    authorization: ModelAcquisitionAuthorization,
) -> SearchResult<ImportOutcome> {
    let (recovery, target) = recovery_import_plan(cx, selected, cache_root, Some(authorization))?;
    let bound = recovery
        .bind_local_model_acquisition(&target, current_time()?)
        .map_err(|_| {
            invalid(
                "authorization",
                "The plan cannot execute this exact standalone local import.",
            )
        })?;
    let frozen = selected.frozen_manifest()?;
    let source = std::path::absolute(source)?;
    let destination = std::path::absolute(cache_root)?.join(selected.install_directory());
    let mut lifecycle = import_lifecycle(selected, &frozen);
    let result = ModelDownloader::with_defaults()
        .acquire_local_model_recovery(
            cx,
            &bound,
            ModelAcquisitionRequest {
                frozen_manifest: &frozen,
                source: ModelAcquisitionSource::LocalBundle(&source),
                destination_dir: &destination,
            },
            &mut lifecycle,
            |_| {},
            |directory, contract| selected.load_self_test(directory, contract),
            || {
                ensure_unsealed_destination(&destination)?;
                current_time()
            },
        )
        .await;
    Ok(match result {
        Ok(receipt) => ImportOutcome::Imported(receipt, Some(bound.plan().clone())),
        Err(error) => ImportOutcome::FailedAttempt(error, bound.plan().clone()),
    })
}

#[allow(clippy::large_enum_variant)] // Request-local outcome; retain the exact canonical plan.
enum ImportOutcome {
    Planned(RecoveryPlan, ModelAcquisitionAuthorization),
    Imported(ModelAcquisitionReceiptV1, Option<RecoveryPlan>),
    FailedAttempt(SearchError, RecoveryPlan),
}

fn run(options: Options) -> SearchResult<ImportOutcome> {
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
        if options.plan_only {
            let (recovery, _) = recovery_import_plan(&cx, &selected, &options.output, None)?;
            let authorization = recovery
                .action()
                .and_then(|action| action.required_authorization())
                .cloned()
                .ok_or_else(|| invalid("plan", "The exact acquisition scope is unavailable."))?;
            Ok(ImportOutcome::Planned(recovery, authorization))
        } else if let Some(path) = &options.authorization {
            let authorization = read_authorization(path)?;
            import_authorized_model(
                &cx,
                &selected,
                &options.source,
                &options.output,
                authorization,
            )
            .await
        } else {
            import_model(&cx, &selected, &options.source, &options.output)
                .await
                .map(|receipt| ImportOutcome::Imported(receipt, None))
        }
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
    recovery_plan: Option<RecoveryPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    authorization: Option<ModelAcquisitionAuthorization>,
    #[serde(skip_serializing_if = "Option::is_none")]
    publication_uncertainty: Option<ModelPublicationUncertaintyV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Failure>,
}

fn failure_message(error: &SearchError) -> &'static str {
    if let Some(uncertainty) = model_publication_uncertainty(error) {
        return match uncertainty.outcome {
            ModelPublicationUncertainOutcomeV1::SelectedNewDurabilityUnknown => {
                "The verified new cache is visible, but its publication durability is unconfirmed. Preserve all cache generations and verify the selected cache before any retry."
            }
            ModelPublicationUncertainOutcomeV1::SelectionUnknownAfterRestoreFailure => {
                "Publication could not establish or restore the selected cache. Preserve all retained generations and inspect the cache and backup before any retry."
            }
            ModelPublicationUncertainOutcomeV1::PriorRestoredDurabilityUnknown => {
                "The prior cache was restored, but its restoration durability is unconfirmed. Preserve all retained generations and verify the selection before any retry."
            }
        };
    }
    match error {
        SearchError::InvalidConfig { field, .. } if field == "cli.import_model.readiness" => {
            "The cache artifacts already verify. Run the ordinary import command without --plan or --authorization for load testing and warm reuse."
        }
        SearchError::InvalidConfig { field, .. }
            if field == "cli.import_model.authorization"
                || field.starts_with("model_acquisition.recovery.") =>
        {
            "The exact local import grant, scope or readiness was rejected. Run --plan again and approve its current authorization before retrying."
        }
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
    let uncertainty = model_publication_uncertainty(error).cloned();
    (
        Response {
            schema_version: 1,
            ok: false,
            postcondition: uncertainty.as_ref().map(|_| "model_publication_uncertain"),
            acquisition: None,
            recovery_plan: None,
            authorization: None,
            publication_uncertainty: uncertainty.clone(),
            error: Some(Failure {
                code: if uncertainty.is_some() {
                    "model_publication_uncertain"
                } else {
                    error_code_for(error)
                },
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
    } else if let Some(authorization) = &response.authorization {
        let exact = serde_json::to_string_pretty(authorization).map_err(io::Error::other)?;
        format!(
            "Planned standalone offline import. No cache or index was changed.\n\
             Review and save the following object as consent.json, then repeat the same\n\
             command with --authorization consent.json before expiry.\n{exact}\n"
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
                Ok(ImportOutcome::Imported(receipt, recovery)) => (
                    Response {
                        schema_version: 1,
                        ok: true,
                        postcondition: Some("model_acquired_needs_reindex"),
                        acquisition: Some(receipt),
                        recovery_plan: recovery,
                        authorization: None,
                        publication_uncertainty: None,
                        error: None,
                    },
                    0,
                ),
                Ok(ImportOutcome::Planned(recovery, authorization)) => (
                    Response {
                        schema_version: 1,
                        ok: true,
                        postcondition: None,
                        acquisition: None,
                        recovery_plan: Some(recovery),
                        authorization: Some(authorization),
                        publication_uncertainty: None,
                        error: None,
                    },
                    0,
                ),
                Ok(ImportOutcome::FailedAttempt(error, recovery)) => {
                    let (mut response, exit) = failure_response(&error, failure_message(&error));
                    response.recovery_plan = Some(recovery);
                    (response, exit)
                }
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
        let mut preview = arguments();
        preview.push("--plan".into());
        assert!(Options::parse(&preview).unwrap().plan_only);
        preview.extend(["--authorization", "consent.json"].map(OsString::from));
        assert!(Options::parse(&preview).is_err());
        let mut approved = arguments();
        approved.extend(["--authorization", "consent.json"].map(OsString::from));
        assert_eq!(
            Options::parse(&approved).unwrap().authorization.as_deref(),
            Some(Path::new("consent.json"))
        );
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
        let authority = temp.path().join("authority-root");
        fs::create_dir(&authority).unwrap();
        fs::write(
            authority
                .join(frankensearch_index::generation_root::GENERATION_ROOT_AUTHORITY_FILE_NAME),
            b"fixed authority",
        )
        .unwrap();
        assert!(ensure_unsealed_destination(&authority.join("absent-cache/model")).is_err());
        assert!(!authority.join("absent-cache").exists());
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
    fn publication_uncertainty_is_reported_without_claiming_acquisition_or_verification_failure() {
        for outcome in [
            ModelPublicationUncertainOutcomeV1::SelectedNewDurabilityUnknown,
            ModelPublicationUncertainOutcomeV1::SelectionUnknownAfterRestoreFailure,
            ModelPublicationUncertainOutcomeV1::PriorRestoredDurabilityUnknown,
        ] {
            let error = SearchError::SubsystemError {
                subsystem: "model_publication",
                source: Box::new(ModelPublicationUncertaintyV1 {
                    schema_version: 1,
                    outcome,
                    prior_generation_preserved: true,
                    cause_code: "io",
                    io_error_kind: Some("Other".to_owned()),
                }),
            };
            let (response, exit) = failure_response(&error, failure_message(&error));
            assert!(!response.ok);
            assert_ne!(exit, 0);
            assert!(response.acquisition.is_none());
            assert_eq!(response.postcondition, Some("model_publication_uncertain"));
            assert_eq!(
                response.publication_uncertainty.as_ref().unwrap().outcome,
                outcome
            );
            let json: serde_json::Value =
                serde_json::from_slice(&render(&response, Format::Json).unwrap()).unwrap();
            assert_eq!(json["error"]["code"], "model_publication_uncertain");
            assert!(json.get("acquisition").is_none());
            let human = String::from_utf8(render(&response, Format::Table).unwrap()).unwrap();
            assert!(human.contains("before any retry"));
            assert!(!human.contains("verification failed"));
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
        fn scoped_local_import_plans_without_writes_and_executes_the_exact_registered_producer() {
            let (source, selected) = fixture(0);
            let temp = tempfile::tempdir().unwrap();
            let cache = temp.path().canonicalize().unwrap().join("absent-cache");
            let scheduler = RuntimeBuilder::current_thread().build().unwrap();
            scheduler.block_on(async {
                let cx = Cx::for_request();
                let (preview, _) = recovery_import_plan(&cx, &selected, &cache, None).unwrap();
                assert!(
                    !cache.exists(),
                    "planning must not create the cache or lock"
                );
                assert_eq!(
                    preview.retryability(),
                    frankensearch_core::Retryability::BlockedByCapability
                );
                let authorization = preview
                    .action()
                    .unwrap()
                    .required_authorization()
                    .unwrap()
                    .clone();
                assert_eq!(authorization.document_count, 0);
                assert_eq!(authorization.estimated_reindex_duration_ms, 0);
                let grant_file = temp.path().join("consent.json");
                fs::write(&grant_file, serde_json::to_vec(&authorization).unwrap()).unwrap();
                let supplied = read_authorization(&grant_file).unwrap();
                let ImportOutcome::Imported(receipt, Some(executed)) =
                    import_authorized_model(&cx, &selected, source.path(), &cache, supplied)
                        .await
                        .unwrap()
                else {
                    panic!("the scoped real producer import must complete");
                };
                assert_eq!(
                    receipt.outcome,
                    frankensearch_embed::ModelAcquisitionOutcomeV1::Published
                );
                assert!(receipt.requires_reindex);
                assert!(executed.action().unwrap().prerequisites().is_empty());
                assert_eq!(
                    executed.action().unwrap().required_authorization(),
                    Some(&authorization)
                );
                assert!(!executed.semantic_available());
                assert!(executed.response_contract().is_none());
                selected
                    .load_self_test(
                        &cache.join(selected.install_directory()),
                        &selected.frozen_manifest().unwrap(),
                    )
                    .unwrap();
                let serialized = serde_json::to_string(&executed).unwrap();
                assert!(!serialized.contains(&cache.to_string_lossy().to_string()));
                assert!(!serialized.contains(&source.path().to_string_lossy().to_string()));

                // A completed cache changes readiness, so an old mutation grant
                // is refused. The original explicit warm path remains usable.
                assert!(
                    import_authorized_model(&cx, &selected, source.path(), &cache, authorization)
                        .await
                        .is_err()
                );
                let warm = import_model(&cx, &selected, source.path(), &cache)
                    .await
                    .unwrap();
                assert_eq!(
                    warm.source,
                    frankensearch_embed::ModelAcquisitionSourceKindV1::WarmCache
                );
                assert_eq!(
                    frankensearch_embed::diagnose_model_acquisition(
                        &cache.join(selected.install_directory())
                    )
                    .unwrap()
                    .preserved_backup_generations,
                    0
                );
            });
        }

        #[test]
        fn scoped_local_import_rejects_changed_grants_before_cache_creation() {
            let (source, selected) = fixture(0);
            let temp = tempfile::tempdir().unwrap();
            let cache = temp.path().canonicalize().unwrap().join("absent-cache");
            let scheduler = RuntimeBuilder::current_thread().build().unwrap();
            scheduler.block_on(async {
                let cx = Cx::for_request();
                let (preview, _) = recovery_import_plan(&cx, &selected, &cache, None).unwrap();
                let exact = preview
                    .action()
                    .unwrap()
                    .required_authorization()
                    .unwrap()
                    .clone();
                for mutation in 0..6 {
                    let mut grant = exact.clone();
                    match mutation {
                        0 => grant.destination_fingerprint = "1".repeat(64),
                        1 => grant.byte_budget += 1,
                        2 => grant.document_count = 1,
                        3 => grant.license_spdx = "different-license".to_owned(),
                        4 => grant.manifest_fingerprint = "2".repeat(64),
                        _ => {
                            grant.issued_at_unix_seconds = 1;
                            grant.expires_at_unix_seconds = 2;
                        }
                    }
                    assert!(
                        import_authorized_model(&cx, &selected, source.path(), &cache, grant)
                            .await
                            .is_err()
                    );
                    assert!(!cache.exists());
                }
                let grant_file = temp.path().join("oversized.json");
                fs::write(&grant_file, vec![b' '; MAX_RESPONSE_BYTES + 1]).unwrap();
                assert!(read_authorization(&grant_file).is_err());
            });
        }

        #[test]
        fn scoped_local_import_failed_execution_keeps_prior_and_reports_the_executed_plan() {
            let (source, selected) = fixture(1);
            let temp = tempfile::tempdir().unwrap();
            let cache = temp.path().canonicalize().unwrap().join("cache");
            let destination = cache.join(selected.install_directory());
            fs::create_dir_all(&destination).unwrap();
            fs::write(
                destination.join("prior-bytes"),
                b"retain this exact prior cache",
            )
            .unwrap();
            let scheduler = RuntimeBuilder::current_thread().build().unwrap();
            scheduler.block_on(async {
                let cx = Cx::for_request();
                let (preview, _) = recovery_import_plan(&cx, &selected, &cache, None).unwrap();
                let authorization = preview
                    .action()
                    .unwrap()
                    .required_authorization()
                    .unwrap()
                    .clone();
                let ImportOutcome::FailedAttempt(error, executed) = import_authorized_model(
                    &cx,
                    &selected,
                    source.path(),
                    &cache,
                    authorization.clone(),
                )
                .await
                .unwrap() else {
                    panic!("incorrect registered execution certificate must fail");
                };
                assert!(matches!(error, SearchError::ModelLoadFailed { .. }));
                assert_eq!(
                    executed.action().unwrap().required_authorization(),
                    Some(&authorization)
                );
                assert!(executed.action().unwrap().prerequisites().is_empty());
                assert!(!executed.semantic_available());
                assert_eq!(
                    fs::read(destination.join("prior-bytes")).unwrap(),
                    b"retain this exact prior cache"
                );
                assert_eq!(
                    frankensearch_embed::diagnose_model_acquisition(&destination)
                        .unwrap()
                        .preserved_backup_generations,
                    0
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
