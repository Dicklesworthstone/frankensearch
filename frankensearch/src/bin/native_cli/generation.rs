//! Executable build, fixed-authority publication, and retained exact search.
//!
//! Staging creates a fresh private root under a trusted parent. The existing
//! qualified-root publisher independently admits and syncs every staged image
//! before changing AUTHORITY. An external receipt pins the exact authority and
//! vector identity for subsequent processes; it never discovers a replacement.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use asupersync::runtime::RuntimeBuilder;
use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use frankensearch::native_ann::NativeGenerationSnapshot;
use frankensearch::native_ann::builder::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltExactGeneration, NativeIndexBuilder,
};
use frankensearch::{Cx, GenerationActivationV1, GenerationPublisherV1, SearchError};
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, AuthorityRefV1, CommitRange, EmbedderTierTag,
    FrozenEmbeddingIdentityBundleV1, GenerationAuthorityActionV1, GenerationRootSecurityProfileV1,
};
use frankensearch_index::FsviV2IdentityBinding;
use frankensearch_index::generation_root::authority_publisher::{
    AUTHORITY_PUBLISHER_LOCK_BYTES_V1, AttemptPermitV1, ExpectedAuthorityPairV1,
    PublicationOutcomeV1, WriterIdentityV1,
};
use frankensearch_index::generation_root::generation_reader::{
    SnapshotOpenOutcomeV1, component_object_name_v1,
};
use frankensearch_index::generation_root::{
    GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationRootAnchorLayout, QualifiedGenerationRoot,
};
use frankensearch_quill::{DEFAULT_SCHEMA, QuillConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{
    MAX_DOCUMENTS, MAX_RECEIPT_BYTES, MAX_RECORD_BYTES, Result, SCHEMA, bad, emit, encode, models,
    new_generation, new_path, positive as number, read_documents, validate_query, value,
};

const RECEIPT_SCHEMA: &str = "frankensearch.native.authority-selection.v1";
const DEFAULT_IMAGE_BYTES: usize = 256 * 1024 * 1024;
const PROFILE: GenerationRootSecurityProfileV1 = GenerationRootSecurityProfileV1::CooperativeLocal;
const HELP: &str = "Fixed AUTHORITY exact search (one explicit semantic tier)\n\n\
  authority-index --root NEW_ROOT --work-dir NEW_DIR --receipt NEW_JSON\n\
      --tier fast|quality --source-low N --source-high N --cooperative-local\n\
      [--input JSONL] [--model-dir DIR] [--batch-size N]\n\
      [--quality-backend onnx|native-int8|native-f32|native-multilingual]\n\
      [--quality-model-dir DIR] [--max-image-bytes N]\n\n\
  authority-search --receipt JSON --query TEXT [--limit N]\n\
      [--model-dir DIR] [--quality-backend BACKEND] [--quality-model-dir DIR]\n\
      [--max-image-bytes N]\n\n\
This explicit route builds one exact FSVI v2 tier plus a retained Quill image.\n\
Quality-only generations load and query their quality model directly. Models\n\
are local-only; unavailable or mismatched models fail without substitution.\n\
Source-low/high identify the inclusive source checkpoint supplied by the caller.\n\
Roots, work directories and receipts must be new, separate locations.\n\
The root and its trusted parent must satisfy the qualified-root Linux ext4/btrfs\n\
contract. Unsupported filesystems fail before indexing. Cooperative-local is\n\
explicit: this command does not provide an external anti-rollback service.\n\
The receipt is durably written before publication and selects exactly one\n\
authority; its existence alone is not proof that publication committed.\n\
Publication failures report not_committed, unknown, or committed separately\n\
from serving activation. Existing roots and authority heads are never replaced\n\
by authority-index. Search refuses a head different from the receipt.\n\
The default encoded-image budget is 256 MiB, checked before component reads.\n\
Work files remain available after completion or failure. Stdout is JSON.\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Index,
    Search,
}

struct Options {
    command: Command,
    root: Option<PathBuf>,
    work: Option<PathBuf>,
    receipt: PathBuf,
    input: Option<PathBuf>,
    models: Option<PathBuf>,
    quality: models::QualityOptions,
    tier: Option<EmbedderTierTag>,
    source_low: Option<u64>,
    source_high: Option<u64>,
    query: Option<String>,
    limit: usize,
    batch_size: usize,
    max_image_bytes: usize,
    cooperative_local: bool,
}

pub fn is_command(command: &str) -> bool {
    matches!(command, "authority-index" | "authority-search")
}

impl Options {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>> {
        let mut args = args.into_iter();
        let command = match args.next().as_deref() {
            Some("authority-index") => Command::Index,
            Some("authority-search") => Command::Search,
            _ => return Err(bad("unknown authority command")),
        };
        let mut options = Self {
            command,
            root: None,
            work: None,
            receipt: PathBuf::new(),
            input: None,
            models: None,
            quality: models::QualityOptions::default(),
            tier: None,
            source_low: None,
            source_high: None,
            query: None,
            limit: 10,
            batch_size: 64,
            max_image_bytes: DEFAULT_IMAGE_BYTES,
            cooperative_local: false,
        };
        let mut seen = BTreeSet::new();
        while let Some(flag) = args.next() {
            if matches!(flag.as_str(), "--help" | "-h") {
                return Ok(None);
            }
            if !seen.insert(flag.clone()) {
                return Err(bad("an option may only be supplied once"));
            }
            match flag.as_str() {
                "--receipt" => options.receipt = PathBuf::from(value(&mut args)?),
                "--model-dir" => options.models = Some(PathBuf::from(value(&mut args)?)),
                "--quality-backend" => {
                    options.quality.backend =
                        Some(models::QualityBackend::parse(&value(&mut args)?)?);
                }
                "--quality-model-dir" => {
                    options.quality.directory = Some(PathBuf::from(value(&mut args)?));
                }
                "--max-image-bytes" => {
                    options.max_image_bytes = number(&value(&mut args)?, usize::MAX)?;
                }
                "--root" if command == Command::Index => {
                    options.root = Some(PathBuf::from(value(&mut args)?));
                }
                "--work-dir" if command == Command::Index => {
                    options.work = Some(PathBuf::from(value(&mut args)?));
                }
                "--input" if command == Command::Index => {
                    options.input = Some(PathBuf::from(value(&mut args)?));
                }
                "--tier" if command == Command::Index => {
                    options.tier = Some(match value(&mut args)?.as_str() {
                        "fast" => EmbedderTierTag::Fast,
                        "quality" => EmbedderTierTag::Quality,
                        _ => return Err(bad("authority tier must be fast or quality")),
                    });
                }
                "--source-low" if command == Command::Index => {
                    options.source_low = Some(source_sequence(&value(&mut args)?)?);
                }
                "--source-high" if command == Command::Index => {
                    options.source_high = Some(source_sequence(&value(&mut args)?)?);
                }
                "--batch-size" if command == Command::Index => {
                    options.batch_size = number(&value(&mut args)?, MAX_DOCUMENTS)?;
                }
                "--cooperative-local" if command == Command::Index => {
                    options.cooperative_local = true;
                }
                "--query" if command == Command::Search => options.query = Some(value(&mut args)?),
                "--limit" if command == Command::Search => {
                    options.limit = number(&value(&mut args)?, 1_000)?;
                }
                _ => return Err(bad("unknown or inapplicable authority option")),
            }
        }
        if options.receipt.as_os_str().is_empty() {
            return Err(bad("authority commands require --receipt"));
        }
        match command {
            Command::Index => {
                if options.root.is_none() || options.work.is_none() || !options.cooperative_local {
                    return Err(bad(
                        "authority-index requires --root, --work-dir, and --cooperative-local",
                    ));
                }
                let tier = options
                    .tier
                    .ok_or_else(|| bad("authority-index requires --tier"))?;
                options
                    .quality
                    .validate_usage(tier == EmbedderTierTag::Quality)?;
                options.source_cut()?;
            }
            Command::Search => validate_query(
                options
                    .query
                    .as_deref()
                    .ok_or_else(|| bad("authority-search requires --query"))?,
            )?,
        }
        Ok(Some(options))
    }

    fn source_cut(&self) -> Result<CommitRange> {
        let low = self
            .source_low
            .ok_or_else(|| bad("authority-index requires --source-low"))?;
        let high = self
            .source_high
            .ok_or_else(|| bad("authority-index requires --source-high"))?;
        if high
            .checked_sub(low)
            .and_then(|span| span.checked_add(1))
            .is_none()
        {
            return Err(bad(
                "source checkpoint must be a nonempty representable inclusive range",
            ));
        }
        Ok(CommitRange { low, high })
    }
}

fn source_sequence(text: &str) -> Result<u64> {
    text.parse()
        .map_err(|_| bad("source sequence must be an unsigned 64-bit integer"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    root: PathBuf,
    root_id: [u8; 16],
    profile: String,
    authority: Vec<u8>,
    generation: ArtifactGenerationIdentityV1,
    identity: FrozenEmbeddingIdentityBundleV1,
    tier: EmbedderTierTag,
    documents: u64,
    image_bytes: u64,
}

impl Receipt {
    fn read(path: &Path) -> Result<Self> {
        if !fs::symlink_metadata(path)?.is_file() {
            return Err(bad(
                "the authority receipt must be a regular non-symlink file",
            ));
        }
        let mut bytes = Vec::new();
        File::open(path)?
            .take(MAX_RECEIPT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_RECEIPT_BYTES {
            return Err(bad("authority receipt exceeds 64 KiB"));
        }
        let receipt: Self = serde_json::from_slice(&bytes)
            .map_err(|_| bad("invalid authority selection receipt"))?;
        receipt.validate()?;
        Ok(receipt)
    }

    fn validate(&self) -> Result<()> {
        if self.schema != RECEIPT_SCHEMA
            || !self.root.is_absolute()
            || self.root_id == [0; 16]
            || self.profile != PROFILE.as_str()
            || self.documents == 0
            || self.documents > MAX_DOCUMENTS as u64
            || self.image_bytes == 0
        {
            return Err(bad("invalid authority receipt contract"));
        }
        let authority = self.authority()?;
        if self.image_bytes < authority.manifest_len {
            return Err(bad("authority receipt image length is inconsistent"));
        }
        self.binding()?;
        Ok(())
    }

    fn authority(&self) -> Result<AuthorityRefV1> {
        AuthorityRefV1::from_canonical_bytes(&self.authority).map_err(Into::into)
    }

    fn binding(&self) -> Result<FsviV2IdentityBinding> {
        FsviV2IdentityBinding::new(self.generation, self.identity.clone()).map_err(Into::into)
    }

    fn open(&self, cx: &Cx, max_image_bytes: usize) -> Result<NativeGenerationSnapshot> {
        self.validate()?;
        checkpoint(cx)?;
        let budget = u64::try_from(max_image_bytes)?.min(self.image_bytes);
        let root = admit_root(&self.root)?;
        let opened =
            root.open_generation_snapshot_with_budget(self.root_id, PROFILE, None, budget)?;
        checkpoint(cx)?;
        let SnapshotOpenOutcomeV1::Opened(opened) = opened else {
            return Err(bad(&format!(
                "authority snapshot admission refused: {opened:?}"
            )));
        };
        self.admit_opened(cx, opened, usize::try_from(budget)?)
    }

    fn admit_opened(
        &self,
        cx: &Cx,
        opened: Arc<
            frankensearch_index::generation_root::generation_reader::OpenedGenerationSnapshotV1,
        >,
        max_image_bytes: usize,
    ) -> Result<NativeGenerationSnapshot> {
        if opened.root_id() != self.root_id || opened.head().authority != self.authority()? {
            return Err(bad(
                "the fixed authority head differs from the trusted receipt",
            ));
        }
        let snapshot = NativeGenerationSnapshot::admit_exact(
            cx,
            opened,
            &self.binding()?,
            DEFAULT_SCHEMA,
            QuillConfig::default(),
            max_image_bytes,
        )?;
        if snapshot.tier() != self.tier || snapshot.metadata().total_documents != self.documents {
            return Err(bad(
                "the retained generation differs from the selected tier or document count",
            ));
        }
        Ok(snapshot)
    }
}

fn checkpoint(cx: &Cx) -> Result<()> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        return Err(Box::new(SearchError::Cancelled {
            phase: "native_cli.authority".to_owned(),
            reason: "cancellation requested".to_owned(),
        }));
    }
    Ok(())
}

fn admit_root(path: &Path) -> Result<QualifiedGenerationRoot> {
    QualifiedGenerationRoot::admit(
        path,
        GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1)?,
    )
    .map_err(Into::into)
}

fn create_file(path: &Path, mode: u32) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(mode);
    }
    #[cfg(not(unix))]
    let _ = mode;
    options.open(path).map_err(Into::into)
}

fn create_root(path: &Path) -> Result<QualifiedGenerationRoot> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    builder.create(path)?;
    for (name, len) in [
        ("LOCK", AUTHORITY_PUBLISHER_LOCK_BYTES_V1),
        ("AUTHORITY", GENERATION_ROOT_AUTHORITY_FILE_BYTES),
    ] {
        let mut file = create_file(&path.join(name), 0o600)?;
        file.write_all(&vec![0; usize::try_from(len)?])?;
        file.sync_all()?;
    }
    File::open(path)?.sync_all()?;
    File::open(path.parent().ok_or_else(|| bad("root parent is missing"))?)?.sync_all()?;
    admit_root(path)
}

fn write_immutable(cx: &Cx, path: &Path, bytes: &[u8]) -> Result<()> {
    checkpoint(cx)?;
    let mut file = create_file(path, 0o400)?;
    for chunk in bytes.chunks(1024 * 1024) {
        checkpoint(cx)?;
        file.write_all(chunk)?;
    }
    file.sync_all()?;
    Ok(())
}

fn separate_destinations(paths: &[&Path]) -> Result<()> {
    for (index, path) in paths.iter().enumerate() {
        for other in &paths[index + 1..] {
            if path.starts_with(other) || other.starts_with(path) {
                return Err(bad("root, work directory and receipt must not overlap"));
            }
        }
    }
    Ok(())
}

fn save_receipt(receipt: &Receipt, path: &Path) -> Result<()> {
    receipt.validate()?;
    let bytes = encode(receipt, MAX_RECEIPT_BYTES)?;
    let mut file = create_file(path, 0o600)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    File::open(
        path.parent()
            .ok_or_else(|| bad("receipt parent is missing"))?,
    )?
    .sync_all()?;
    Ok(())
}

#[derive(Debug)]
struct PublicationFailure(serde_json::Value);

impl fmt::Display for PublicationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("fixed-authority publication did not complete serving activation")
    }
}

impl Error for PublicationFailure {}

pub fn publication_failure(error: &(dyn Error + 'static)) -> Option<serde_json::Value> {
    error
        .downcast_ref::<PublicationFailure>()
        .map(|error| error.0.clone())
}

fn publication_error(
    receipt: &Receipt,
    publication: &str,
    serving: &str,
    reason: &str,
    attempt: Option<&AttemptPermitV1>,
) -> Box<dyn Error + Send + Sync> {
    let mut payload = serde_json::json!({
        "schema": SCHEMA, "ok": false, "event": "publication",
        "publication": publication, "serving": serving, "reason": reason,
        "root": receipt.root, "root_id": receipt.root_id, "authority": receipt.authority,
    });
    if let Some(permit) = attempt {
        payload["attempt"] = serde_json::json!({
            "root_id": permit.root_id, "writer_id": permit.writer_id,
            "attempt_id": permit.attempt_id, "target_slot": permit.target_slot,
            "candidate": permit.candidate.canonical_bytes().as_slice(),
            "expected_first": permit.expected.first.map(|slot| slot.authority.canonical_bytes().to_vec()),
            "expected_second": permit.expected.second.map(|slot| slot.authority.canonical_bytes().to_vec()),
            "stage": format!("{:?}", permit.stage), "idempotency_key": permit.idempotency_key,
        });
    }
    Box::new(PublicationFailure(payload))
}

fn stage_and_publish(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    root_path: &Path,
    receipt_path: &Path,
    built: &NativeBuiltExactGeneration,
    max_image_bytes: usize,
) -> Result<Receipt> {
    let root_id = new_generation(1)?.nonce;
    let writer = WriterIdentityV1::for_current_process()?;
    let object_id = new_generation(1)?.nonce;
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.native.authority-build.v1");
    hasher.update(writer.writer_id);
    hasher.update(writer.nonce);
    hasher.update(object_id);
    let candidate = built.candidate(
        object_id,
        None,
        GenerationAuthorityActionV1::Activate,
        hasher.finalize().into(),
    )?;
    if candidate.owned_image_bytes() > u64::try_from(max_image_bytes)? {
        return Err(bad("complete generation exceeds the encoded-image budget"));
    }
    built.validate_candidate(cx, &candidate)?;
    let receipt = Receipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        root: root_path.to_path_buf(),
        root_id,
        profile: PROFILE.as_str().to_owned(),
        authority: candidate.authority().canonical_bytes().to_vec(),
        generation: built.binding().generation(),
        identity: built.binding().frozen_identity().clone(),
        tier: built.metadata().vector_artifacts[0].embedder_tier,
        documents: built.metadata().total_documents,
        image_bytes: candidate.owned_image_bytes(),
    };
    // Freeze the recovery/selection inputs before any authority mutation.
    // Encoding is also preflighted before writing any candidate object.
    receipt.validate()?;
    encode(&receipt, MAX_RECEIPT_BYTES)?;
    for (role, component) in candidate.manifest().components().iter() {
        root.revalidate()?;
        let bytes = built
            .component_bytes(role)
            .ok_or_else(|| bad("the exact builder omitted a declared component"))?;
        write_immutable(
            cx,
            &root_path.join(component_object_name_v1(role, component.sha256)),
            &bytes,
        )?;
    }
    root.revalidate()?;
    write_immutable(
        cx,
        &root_path.join(candidate.manifest_name()),
        candidate.manifest_bytes(),
    )?;
    root.revalidate()?;
    save_receipt(&receipt, receipt_path)?;
    let mut publisher = GenerationPublisherV1::new(root, root_id, writer.writer_id, PROFILE)?;
    let result = publisher
        .publish(
            cx,
            &candidate,
            ExpectedAuthorityPairV1::default(),
            None,
            new_generation(1)?.nonce,
            u64::try_from(max_image_bytes)?,
        )
        .map_err(|error| {
            publication_error(
                &receipt,
                "not_committed",
                "not_activated",
                &error.to_string(),
                None,
            )
        })?;
    match result.publication {
        PublicationOutcomeV1::NotCommitted(reason) => {
            return Err(publication_error(
                &receipt,
                "not_committed",
                "not_activated",
                &format!("{reason:?}"),
                None,
            ));
        }
        PublicationOutcomeV1::CommitOutcomeUnknown(permit) => {
            return Err(publication_error(
                &receipt,
                "unknown",
                "not_activated",
                "reconciliation_required",
                Some(&permit),
            ));
        }
        PublicationOutcomeV1::Committed { .. } => {}
    }
    let Some(GenerationActivationV1::Installed(opened)) = result.activation else {
        return Err(publication_error(
            &receipt,
            "committed",
            "not_activated",
            "snapshot_activation_deferred_or_refused",
            None,
        ));
    };
    if receipt.admit_opened(cx, opened, max_image_bytes).is_err() {
        return Err(publication_error(
            &receipt,
            "committed",
            "not_activated",
            "engine_admission_refused",
            None,
        ));
    }
    Ok(receipt)
}

async fn index(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let root_path = new_path(options.root.as_deref().ok_or_else(|| bad("missing root"))?)?;
    let work_path = new_path(
        options
            .work
            .as_deref()
            .ok_or_else(|| bad("missing work directory"))?,
    )?;
    let receipt_path = new_path(&options.receipt)?;
    separate_destinations(&[&root_path, &work_path, &receipt_path])?;
    // Qualification is before input/model work, including unsupported filesystems.
    let root = create_root(&root_path)?;
    let documents = match &options.input {
        Some(path) => read_documents(&mut BufReader::new(File::open(path)?))?,
        None => read_documents(&mut io::stdin().lock())?,
    };
    if documents.is_empty() {
        return Err(bad(
            "the exact generation metadata contract requires a nonempty source cohort",
        ));
    }
    let tier = options.tier.ok_or_else(|| bad("missing tier"))?;
    let model = models::load_single(cx, options.models.as_deref(), tier, &options.quality, pool)?;
    let built = NativeIndexBuilder::new(&work_path, new_generation(1)?, model)?
        .with_batch_size(options.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
        .with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
        .add_documents(documents)
        .build_exact_generation(cx, tier, options.source_cut()?, options.max_image_bytes)
        .await?;
    let receipt = stage_and_publish(
        cx,
        &root,
        &root_path,
        &receipt_path,
        &built,
        options.max_image_bytes,
    )?;
    emit_indexed(output, &receipt, &receipt_path)
}

fn emit_indexed(output: &mut impl Write, receipt: &Receipt, receipt_path: &Path) -> Result<()> {
    emit(
        output,
        &serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "indexed", "publication": "committed",
            "serving": "activated", "retrieval": "exact", "receipt": receipt_path,
            "selection": receipt,
        }),
    )
    .map_err(|_| {
        publication_error(
            receipt,
            "committed",
            "activated",
            "result_delivery_failed",
            None,
        )
    })
}

async fn search(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let receipt = Receipt::read(&options.receipt)?;
    options
        .quality
        .validate_usage(receipt.tier == EmbedderTierTag::Quality)?;
    let snapshot = receipt.open(cx, options.max_image_bytes)?;
    let model = models::load_single(
        cx,
        options.models.as_deref(),
        receipt.tier,
        &options.quality,
        pool,
    )?;
    let results = snapshot
        .search_text(
            cx,
            model.as_ref(),
            options
                .query
                .as_deref()
                .ok_or_else(|| bad("missing query"))?,
            options.limit,
        )
        .await?;
    emit(
        output,
        &serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "results", "retrieval": "exact",
            "tier": receipt.tier, "generation": receipt.generation,
            "authority": snapshot.authority().canonical_bytes().as_slice(),
            "profile": receipt.profile, "results": results,
        }),
    )
}

pub fn run(args: Vec<String>) -> Result<()> {
    let Some(options) = Options::parse(args)? else {
        io::stdout().lock().write_all(HELP.as_bytes())?;
        return Ok(());
    };
    if !cfg!(target_os = "linux") {
        return Err(bad(
            "authority commands require the qualified Linux root provider",
        ));
    }
    let runtime = RuntimeBuilder::current_thread()
        .blocking_threads(0, 2)
        .build()?;
    let pool = runtime.blocking_handle();
    runtime.block_on(async move {
        let cx = Cx::current().ok_or_else(|| bad("runtime did not install a root context"))?;
        let mut output = io::stdout().lock();
        match options.command {
            Command::Index => Box::pin(index(&cx, &options, &mut output, pool)).await,
            Command::Search => Box::pin(search(&cx, &options, &mut output, pool)).await,
        }
    })
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
