//! Executable build, fixed-authority publication, and retained single-tier search.
//!
//! Staging creates a fresh private root under a trusted parent. The existing
//! qualified-root publisher independently admits and syncs every staged image
//! before changing AUTHORITY. An external receipt pins the exact authority and
//! vector identity for subsequent processes; it never discovers a replacement.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use asupersync::runtime::RuntimeBuilder;
use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use frankensearch::native_ann::builder::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltExactGeneration, NativeIndexBuilder,
};
use frankensearch::native_ann::{NativeGenerationSnapshot, NativeRetrievalMode};
use frankensearch::{
    Cx, Embedder, GenerationActivationV1, GenerationPublisherV1, IndexableDocument, SearchError,
};
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, AuthorityRefV1, AuthoritySlotV1, CommitRange, EmbedderTierTag,
    FrozenEmbeddingIdentityBundleV1, GenerationAuthorityActionV1, GenerationRootSecurityProfileV1,
    QuantizationFormat,
};
use frankensearch_index::FsviV2IdentityBinding;
use frankensearch_index::generation_root::authority_publisher::{
    AUTHORITY_PUBLISHER_LOCK_BYTES_V1, AttemptPermitV1, ExpectedAuthorityPairV1,
    PublicationOutcomeV1, ReconcileOutcomeV1, WriterIdentityV1,
};
use frankensearch_index::generation_root::generation_reader::{
    SnapshotOpenOutcomeV1, activation_manifest_path_v1, component_object_path_v1,
};
use frankensearch_index::generation_root::{
    GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationRootAnchorLayout, GenerationRootObjectWitness,
    QualifiedGenerationRoot,
};
use frankensearch_index::native_hnsw::HnswParams;
use frankensearch_quill::{DEFAULT_SCHEMA, QuillConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{
    MAX_DOCUMENTS, MAX_INPUT_BYTES, MAX_RECEIPT_BYTES, MAX_RECORD_BYTES, Result, SCHEMA, bad, emit,
    encode, execution_context, models, new_generation, new_path, positive as number,
    read_documents, update as mutations, validate_query, value,
};

const RECEIPT_SCHEMA: &str = "frankensearch.native.authority-selection.v2";
const SELECTION_ONLY_RECEIPT_SCHEMA: &str = "frankensearch.native.authority-selection.v1";
const DEFAULT_IMAGE_BYTES: usize = 256 * 1024 * 1024;
const PROFILE: GenerationRootSecurityProfileV1 = GenerationRootSecurityProfileV1::CooperativeLocal;
const HELP: &str = "Fixed AUTHORITY search (one explicit semantic tier)\n\n\
  authority-index --root NEW_ROOT --work-dir NEW_DIR --receipt NEW_JSON\n\
      --tier fast|quality --source-low N --source-high N --cooperative-local\n\
      [--input JSONL] [--model-dir DIR] [--batch-size N]\n\
      [--quality-backend onnx|native-int8|native-f32|native-multilingual]\n\
      [--quality-model-dir DIR] [--retrieval exact|hnsw] [--max-image-bytes N]\n\n\
  authority-rebuild --from-receipt OLD_JSON --work-dir NEW_DIR --receipt NEW_JSON\n\
      (--source-low N --source-high N [--input JSONL] | --retained-source)\n\
      [--max-source-bytes N] [--batch-size N]\n\
      [--model-dir DIR] [--quality-backend BACKEND] [--quality-model-dir DIR]\n\
      [--migrate-model] [--retrieval exact|hnsw] [--max-image-bytes N]\n\n\
  authority-update --from-receipt OLD_JSON --work-dir NEW_DIR --receipt NEW_JSON\n\
      --source-low N --source-high N [--input EDITS_JSONL]\n\
      [--max-source-bytes N] [--batch-size N] [--max-image-bytes N]\n\
      [--model-dir DIR] [--quality-backend BACKEND] [--quality-model-dir DIR]\n\n\
  authority-search --receipt JSON --query TEXT [--limit N]\n\
      [--model-dir DIR] [--quality-backend BACKEND] [--quality-model-dir DIR]\n\
      [--max-image-bytes N]\n\n\
  authority-serve --receipt JSON [--limit N] [--allow-activation]\n\
      [--model-dir DIR] [--quality-backend BACKEND] [--quality-model-dir DIR]\n\
      [--max-image-bytes N]\n\n\
  authority-recover --receipt JSON [--max-image-bytes N]\n\n\
This explicit route builds one FSVI v2 tier plus a retained Quill image.\n\
New indexes use exact retrieval unless --retrieval hnsw explicitly selects ANN.\n\
The selected ANN graph and its source identity are authenticated by AUTHORITY;\n\
missing or invalid graphs fail admission without substituting exact retrieval.\n\
Quality-only generations load and query their quality model directly. Models\n\
are local-only; unavailable or mismatched models fail without substitution.\n\
Source-low/high identify the inclusive source checkpoint supplied by the caller.\n\
Roots, work directories and receipts must be new, separate locations.\n\
Rebuild retains the selected root, tier and security profile, replaces the whole\n\
source cohort, and requires a new work directory and receipt outside the root.\n\
Source boundaries cannot regress. A different producer requires --migrate-model.\n\
With --retained-source, rebuild reads complete stored documents from the admitted\n\
generation and preserves its source checkpoint, without reading JSONL or stdin.\n\
This excludes --input and --source-low/high. --max-source-bytes bounds retained\n\
source for this mode and updates, defaulting to 256 MiB of raw stored source bytes.\n\
The existing 100000-document and 16 MiB encoded-record limits still apply.\n\
Rebuild inherits the admitted retrieval policy, including HNSW parameters and\n\
seed, unless --retrieval explicitly selects exact or the default HNSW policy.\n\
Concurrent publication refuses the captured old selection; retry explicitly\n\
from the winning receipt. Rebuild never edits the old generation's images.\n\
Update applies bounded upsert/delete JSONL to the retained source, with the last\n\
operation for each ID winning. It reuses unchanged content vectors and preserves\n\
the selected producer, tier, precision, and retrieval policy. Only changed or new\n\
content is embedded; complete vector, graph, and Quill images are still produced.\n\
The final cohort must be nonempty and satisfy the combined encoded-source limits.\n\
Updates require nonbackward source checkpoints and a new work directory/receipt.\n\
Use rebuild for producer or retrieval changes. Update never changes old images.\n\
The root and its trusted parent must satisfy the qualified-root Linux ext4/btrfs\n\
contract. Unsupported filesystems fail before indexing. Cooperative-local is\n\
explicit: this command does not provide an external anti-rollback service.\n\
The receipt is durably written before publication and selects exactly one\n\
authority; its existence alone is not proof that publication committed.\n\
It also preserves the complete original attempt for recovery after a crash.\n\
authority-recover reconciles that attempt without republishing or loading models.\n\
Recovery reports commit knowledge separately from bounded reader admission.\n\
Publication failures report not_committed, unknown, or committed separately\n\
from serving activation. Existing roots and authority heads are never replaced\n\
by authority-index. Search refuses a head different from the receipt.\n\
The default encoded-image budget is 256 MiB, checked before component reads.\n\
authority-serve retains one admitted generation and its local model for bounded\n\
JSONL requests: {\"id\":\"q1\",\"query\":\"text\",\"limit\":10}. It never follows a new\n\
authority implicitly. With --allow-activation, an explicit request may select a\n\
new receipt: {\"op\":\"activate\",\"receipt\":\"/absolute/new.json\",\n\
\"expected_authority\":[...the current ready/results authority bytes...]}.\n\
Activation keeps the same qualified root, tier, security profile and producer.\n\
It admits the complete currently selected successor before swapping readers;\n\
rejection retains the previous reader and returns a bounded failed response.\n\
Authority sequence must advance, including rollback to older artifact generations.\n\
Model migration requires restarting with the new receipt and local model options.\n\
Requests are limited to 1 MiB; oversized input or output failure ends the session.\n\
Send quit/exit or close stdin to stop. Reads block on the owning execution lane.\n\
Work files remain available after completion or failure. Stdout is JSON.\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Index,
    Rebuild,
    Update,
    Search,
    Serve,
    Recover,
}

struct Options {
    command: Command,
    root: Option<PathBuf>,
    work: Option<PathBuf>,
    receipt: PathBuf,
    from_receipt: Option<PathBuf>,
    input: Option<PathBuf>,
    models: Option<PathBuf>,
    quality: models::QualityOptions,
    tier: Option<EmbedderTierTag>,
    retrieval: Option<NativeBuildRetrieval>,
    source_low: Option<u64>,
    source_high: Option<u64>,
    query: Option<String>,
    limit: usize,
    batch_size: usize,
    max_image_bytes: usize,
    max_source_bytes: usize,
    cooperative_local: bool,
    migrate_model: bool,
    retained_source: bool,
    allow_activation: bool,
}

pub fn is_command(command: &str) -> bool {
    matches!(
        command,
        "authority-index"
            | "authority-rebuild"
            | "authority-update"
            | "authority-search"
            | "authority-serve"
            | "authority-recover"
    )
}

impl Options {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>> {
        let mut args = args.into_iter();
        let command = match args.next().as_deref() {
            Some("authority-index") => Command::Index,
            Some("authority-rebuild") => Command::Rebuild,
            Some("authority-update") => Command::Update,
            Some("authority-search") => Command::Search,
            Some("authority-serve") => Command::Serve,
            Some("authority-recover") => Command::Recover,
            _ => return Err(bad("unknown authority command")),
        };
        let mut options = Self {
            command,
            root: None,
            work: None,
            receipt: PathBuf::new(),
            from_receipt: None,
            input: None,
            models: None,
            quality: models::QualityOptions::default(),
            tier: None,
            retrieval: None,
            source_low: None,
            source_high: None,
            query: None,
            limit: 10,
            batch_size: 64,
            max_image_bytes: DEFAULT_IMAGE_BYTES,
            max_source_bytes: MAX_INPUT_BYTES,
            cooperative_local: false,
            migrate_model: false,
            retained_source: false,
            allow_activation: false,
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
                "--from-receipt" if matches!(command, Command::Rebuild | Command::Update) => {
                    options.from_receipt = Some(PathBuf::from(value(&mut args)?));
                }
                "--migrate-model" if command == Command::Rebuild => options.migrate_model = true,
                "--retained-source" if command == Command::Rebuild => {
                    options.retained_source = true
                }
                "--max-source-bytes" if matches!(command, Command::Rebuild | Command::Update) => {
                    options.max_source_bytes = number(&value(&mut args)?, usize::MAX)?;
                }
                "--allow-activation" if command == Command::Serve => {
                    options.allow_activation = true
                }
                "--model-dir" if command != Command::Recover => {
                    options.models = Some(PathBuf::from(value(&mut args)?));
                }
                "--quality-backend" if command != Command::Recover => {
                    options.quality.backend =
                        Some(models::QualityBackend::parse(&value(&mut args)?)?);
                }
                "--quality-model-dir" if command != Command::Recover => {
                    options.quality.directory = Some(PathBuf::from(value(&mut args)?));
                }
                "--max-image-bytes" => {
                    options.max_image_bytes = number(&value(&mut args)?, usize::MAX)?;
                }
                "--root" if command == Command::Index => {
                    options.root = Some(PathBuf::from(value(&mut args)?));
                }
                "--work-dir"
                    if matches!(command, Command::Index | Command::Rebuild | Command::Update) =>
                {
                    options.work = Some(PathBuf::from(value(&mut args)?));
                }
                "--input"
                    if matches!(command, Command::Index | Command::Rebuild | Command::Update) =>
                {
                    options.input = Some(PathBuf::from(value(&mut args)?));
                }
                "--tier" if command == Command::Index => {
                    options.tier = Some(match value(&mut args)?.as_str() {
                        "fast" => EmbedderTierTag::Fast,
                        "quality" => EmbedderTierTag::Quality,
                        _ => return Err(bad("authority tier must be fast or quality")),
                    });
                }
                "--source-low"
                    if matches!(command, Command::Index | Command::Rebuild | Command::Update) =>
                {
                    options.source_low = Some(source_sequence(&value(&mut args)?)?);
                }
                "--source-high"
                    if matches!(command, Command::Index | Command::Rebuild | Command::Update) =>
                {
                    options.source_high = Some(source_sequence(&value(&mut args)?)?);
                }
                "--batch-size"
                    if matches!(command, Command::Index | Command::Rebuild | Command::Update) =>
                {
                    options.batch_size = number(&value(&mut args)?, MAX_DOCUMENTS)?;
                }
                "--retrieval" if matches!(command, Command::Index | Command::Rebuild) => {
                    options.retrieval = Some(match value(&mut args)?.as_str() {
                        "exact" => NativeBuildRetrieval::Exact,
                        "hnsw" => NativeBuildRetrieval::Hnsw {
                            params: HnswParams::default(),
                            seed: 42,
                        },
                        _ => return Err(bad("authority retrieval must be exact or hnsw")),
                    });
                }
                "--cooperative-local" if command == Command::Index => {
                    options.cooperative_local = true;
                }
                "--query" if command == Command::Search => options.query = Some(value(&mut args)?),
                "--limit" if matches!(command, Command::Search | Command::Serve) => {
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
            Command::Rebuild => {
                if options.from_receipt.is_none() || options.work.is_none() {
                    return Err(bad(
                        "authority-rebuild requires --from-receipt and --work-dir",
                    ));
                }
                if options.retained_source {
                    if options.input.is_some()
                        || options.source_low.is_some()
                        || options.source_high.is_some()
                    {
                        return Err(bad(
                            "--retained-source excludes --input, --source-low, and --source-high",
                        ));
                    }
                } else {
                    if seen.contains("--max-source-bytes") {
                        return Err(bad("--max-source-bytes requires --retained-source"));
                    }
                    options.source_cut()?;
                }
            }
            Command::Update => {
                if options.from_receipt.is_none() || options.work.is_none() {
                    return Err(bad(
                        "authority-update requires --from-receipt and --work-dir",
                    ));
                }
                options.source_cut()?;
            }
            Command::Serve | Command::Recover => {}
        }
        Ok(Some(options))
    }

    fn build_retrieval(&self, previous: Option<NativeBuildRetrieval>) -> NativeBuildRetrieval {
        self.retrieval.or(previous).unwrap_or_default()
    }

    fn rebuild_source_cut(&self, previous: &CommitRange) -> Result<CommitRange> {
        if self.retained_source {
            Ok(previous.clone())
        } else {
            self.source_cut()
        }
    }

    fn source_cut(&self) -> Result<CommitRange> {
        let low = self
            .source_low
            .ok_or_else(|| bad("authority builds require --source-low"))?;
        let high = self
            .source_high
            .ok_or_else(|| bad("authority builds require --source-high"))?;
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

fn retrieval_name(mode: NativeRetrievalMode) -> &'static str {
    match mode {
        NativeRetrievalMode::Ann => "hnsw",
        NativeRetrievalMode::Exact { .. } => "exact",
    }
}

/// Inputs recorded before publication, never an assertion about its outcome.
/// Root and candidate are pinned by the enclosing selection receipt. The
/// original physical slot pair is preserved in full for aborted-attempt proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedAttempt {
    writer_id: [u8; 16],
    attempt_id: [u8; 16],
    expected_first: Option<Vec<u8>>,
    expected_second: Option<Vec<u8>>,
    idempotency_key: [u8; 16],
    requires_floor: bool,
}

impl SavedAttempt {
    fn from_permit(permit: &AttemptPermitV1, requires_floor: bool) -> Self {
        Self {
            writer_id: permit.writer_id,
            attempt_id: permit.attempt_id,
            expected_first: permit
                .expected
                .first
                .map(|slot| slot.authority.canonical_bytes().to_vec()),
            expected_second: permit
                .expected
                .second
                .map(|slot| slot.authority.canonical_bytes().to_vec()),
            idempotency_key: permit.idempotency_key,
            requires_floor,
        }
    }

    fn restore(&self, root_id: [u8; 16], candidate: AuthorityRefV1) -> Result<AttemptPermitV1> {
        let decode = |bytes: &Option<Vec<u8>>, index| -> Result<Option<AuthoritySlotV1>> {
            bytes
                .as_ref()
                .map(|bytes| {
                    let authority = AuthorityRefV1::from_canonical_bytes(bytes)?;
                    AuthoritySlotV1::new(index, root_id, authority).map_err(Into::into)
                })
                .transpose()
        };
        AttemptPermitV1::new(
            root_id,
            self.writer_id,
            self.attempt_id,
            candidate,
            ExpectedAuthorityPairV1 {
                first: decode(&self.expected_first, 0)?,
                second: decode(&self.expected_second, 1)?,
            },
            self.idempotency_key,
        )
        .map_err(Into::into)
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    attempt: Option<SavedAttempt>,
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
        if !self.root.is_absolute()
            || self.root_id == [0; 16]
            || self.profile != PROFILE.as_str()
            || self.documents == 0
            || self.documents > MAX_DOCUMENTS as u64
            || self.image_bytes == 0
        {
            return Err(bad("invalid authority receipt contract"));
        }
        let authority = self.authority()?;
        match (self.schema.as_str(), self.attempt.as_ref()) {
            (RECEIPT_SCHEMA, Some(attempt)) => {
                // This CLI never supplies an external floor. A saved attempt
                // requiring one cannot be downgraded during a process restart.
                if attempt.requires_floor {
                    return Err(bad(
                        "the recorded attempt requires an unavailable anti-rollback floor",
                    ));
                }
                attempt.restore(self.root_id, authority)?;
            }
            (SELECTION_ONLY_RECEIPT_SCHEMA, None) => {}
            _ => {
                return Err(bad(
                    "authority receipt schema and prepared attempt disagree",
                ));
            }
        }
        if self.image_bytes < authority.manifest_len {
            return Err(bad("authority receipt image length is inconsistent"));
        }
        self.binding()?;
        Ok(())
    }

    fn recovery_attempt(&self) -> Result<AttemptPermitV1> {
        self.validate()?;
        self.attempt
            .as_ref()
            .ok_or_else(|| bad("this selection-only receipt has no durable publication attempt"))?
            .restore(self.root_id, self.authority()?)
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
        let root = admit_root(&self.root)?;
        self.open_on_root(cx, &root, max_image_bytes)
    }

    // Rebuild uses the same qualified capability for selection and publication.
    // Reopening the root path between those steps could admit a different root
    // inode containing a copied authority pair.
    fn open_on_root(
        &self,
        cx: &Cx,
        root: &QualifiedGenerationRoot,
        max_image_bytes: usize,
    ) -> Result<NativeGenerationSnapshot> {
        self.validate()?;
        checkpoint(cx)?;
        let budget = u64::try_from(max_image_bytes)?.min(self.image_bytes);
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

#[cfg(test)]
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

fn attempt_report(permit: &AttemptPermitV1) -> serde_json::Value {
    serde_json::json!({
        "root_id": permit.root_id, "writer_id": permit.writer_id,
        "attempt_id": permit.attempt_id, "target_slot": permit.target_slot,
        "candidate": permit.candidate.canonical_bytes().as_slice(),
        "expected_first": permit.expected.first.map(|slot| slot.authority.canonical_bytes().to_vec()),
        "expected_second": permit.expected.second.map(|slot| slot.authority.canonical_bytes().to_vec()),
        "stage": format!("{:?}", permit.stage), "idempotency_key": permit.idempotency_key,
    })
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
        payload["attempt"] = attempt_report(permit);
    }
    Box::new(PublicationFailure(payload))
}

fn stage_and_publish(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    root_path: &Path,
    receipt_path: &Path,
    built: &NativeBuiltExactGeneration,
    predecessor: Option<&NativeGenerationSnapshot>,
    max_image_bytes: usize,
) -> Result<Receipt> {
    let (root_id, expected, previous) = match predecessor {
        Some(snapshot) => {
            validate_successor_source(
                &snapshot.metadata().commit_range,
                &built.metadata().commit_range,
            )?;
            if built.binding().generation().sequence
                <= snapshot
                    .retained_snapshot()
                    .manifest()
                    .generation()
                    .sequence
                || built.metadata().vector_artifacts[0].embedder_tier != snapshot.tier()
            {
                return Err(bad(
                    "a rebuild must advance the artifact generation and preserve its tier",
                ));
            }
            (
                snapshot.retained_snapshot().root_id(),
                snapshot.retained_snapshot().authority_pair(),
                Some(snapshot.authority()),
            )
        }
        None => (
            new_generation(1)?.nonce,
            ExpectedAuthorityPairV1::default(),
            None,
        ),
    };
    let writer = WriterIdentityV1::for_current_process()?;
    let object_id = new_generation(1)?.nonce;
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.native.authority-build.v1");
    hasher.update(writer.writer_id);
    hasher.update(writer.nonce);
    hasher.update(object_id);
    let candidate = built.candidate(
        object_id,
        previous,
        GenerationAuthorityActionV1::Activate,
        hasher.finalize().into(),
    )?;
    if candidate.owned_image_bytes() > u64::try_from(max_image_bytes)? {
        return Err(bad("complete generation exceeds the encoded-image budget"));
    }
    built.validate_candidate(cx, &candidate)?;
    let mut publisher = GenerationPublisherV1::new(root, root_id, writer.writer_id, PROFILE)?;
    let attempt = publisher.prepare_attempt(
        &candidate,
        expected,
        new_generation(1)?.nonce,
        new_generation(1)?.nonce,
    )?;
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
        attempt: Some(SavedAttempt::from_permit(&attempt, false)),
    };
    // Freeze the recovery/selection inputs before any authority mutation.
    // Encoding is also preflighted before writing any candidate object.
    receipt.validate()?;
    encode(&receipt, MAX_RECEIPT_BYTES)?;
    for (role, component) in candidate.manifest().components().iter() {
        checkpoint(cx)?;
        let bytes = built
            .component_bytes(role)
            .ok_or_else(|| bad("the generation builder omitted a declared component"))?;
        root.stage_immutable_file(&component_object_path_v1(role, component.sha256)?, &bytes)?;
    }
    checkpoint(cx)?;
    root.stage_immutable_file(
        &activation_manifest_path_v1(candidate.authority().object_id)?,
        candidate.manifest_bytes(),
    )?;
    root.revalidate()?;
    save_receipt(&receipt, receipt_path)?;
    let result = publisher
        .publish_prepared(
            cx,
            &candidate,
            &attempt,
            None,
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
            "the generation metadata contract requires a nonempty source cohort",
        ));
    }
    let tier = options.tier.ok_or_else(|| bad("missing tier"))?;
    let model = models::load_single(cx, options.models.as_deref(), tier, &options.quality, pool)?;
    let built = NativeIndexBuilder::new(&work_path, new_generation(1)?, model)?
        .with_batch_size(options.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
        .with_fast_storage(NativeBuildPrecision::F32, options.build_retrieval(None))
        .add_documents(documents)
        .build_exact_generation(cx, tier, options.source_cut()?, options.max_image_bytes)
        .await?;
    let receipt = stage_and_publish(
        cx,
        &root,
        &root_path,
        &receipt_path,
        &built,
        None,
        options.max_image_bytes,
    )?;
    emit_indexed(output, &receipt, &receipt_path, built.retrieval_mode())
}

fn emit_indexed(
    output: &mut impl Write,
    receipt: &Receipt,
    receipt_path: &Path,
    retrieval: NativeRetrievalMode,
) -> Result<()> {
    emit(
        output,
        &serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "indexed", "publication": "committed",
            "serving": "activated", "retrieval": retrieval_name(retrieval), "receipt": receipt_path,
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

fn validate_successor_source(previous: &CommitRange, source: &CommitRange) -> Result<()> {
    if source.low < previous.low || source.high < previous.high {
        return Err(bad(
            "a successor cannot move either source checkpoint boundary backward; rollback requires a distinct activation",
        ));
    }
    Ok(())
}

fn retained_rebuild_documents(
    cx: &Cx,
    snapshot: &NativeGenerationSnapshot,
    max_source_bytes: usize,
) -> Result<Vec<IndexableDocument>> {
    checkpoint(cx)?;
    if snapshot.metadata().total_documents > u64::try_from(MAX_DOCUMENTS)? {
        return Err(bad("retained source exceeds the 100000-document limit"));
    }
    let documents = snapshot.source_documents(cx, max_source_bytes)?;
    let mut ids = BTreeSet::new();
    for document in &documents {
        checkpoint(cx)?;
        if document.id.trim().is_empty()
            || document.id.contains('\0')
            || document.id.len() > usize::from(u16::MAX)
            || !ids.insert(document.id.clone())
        {
            return Err(bad(
                "retained source IDs must be unique, nonblank, NUL-free, and at most 65535 bytes",
            ));
        }
        // Apply the CLI's JSONL record bound to the complete decoded document,
        // including escaping and a newline, before model loading or inference.
        encode(document, MAX_RECORD_BYTES)
            .map_err(|_| bad("retained source exceeds the 16 MiB encoded-record limit"))?;
    }
    checkpoint(cx)?;
    Ok(documents)
}

fn rebuild_identity(
    selected: &Receipt,
    model: &dyn Embedder,
    migrate_model: bool,
) -> Result<FrozenEmbeddingIdentityBundleV1> {
    let mut identity = model.identity()?.clone();
    identity.validate()?;
    if usize::try_from(identity.space.dimension)? != model.dimension() {
        return Err(bad(
            "the rebuild provider dimension disagrees with its complete identity",
        ));
    }
    // Match the builder's explicit F32 persistence encoding. Only storage
    // encoding changes here; the space, producer, and input contracts remain
    // exactly those independently supplied by the loaded provider.
    "fsvi-v2".clone_into(&mut identity.storage.format);
    identity.storage.quantization = QuantizationFormat::F32;
    "little-endian".clone_into(&mut identity.storage.endianness);
    let frozen = identity.freeze()?;
    if !migrate_model && frozen != selected.identity {
        return Err(bad(
            "the rebuild producer differs from the selected generation; an intentional full-cohort migration requires --migrate-model",
        ));
    }
    Ok(frozen)
}

async fn rebuild(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let selected = Receipt::read(
        options
            .from_receipt
            .as_deref()
            .ok_or_else(|| bad("missing predecessor receipt"))?,
    )?;
    options
        .quality
        .validate_usage(selected.tier == EmbedderTierTag::Quality)?;
    let work_path = new_path(
        options
            .work
            .as_deref()
            .ok_or_else(|| bad("missing work directory"))?,
    )?;
    let receipt_path = new_path(&options.receipt)?;
    separate_destinations(&[&selected.root, &work_path, &receipt_path])?;
    let root = admit_root(&selected.root)?;
    // Select once, before input or model work, and keep that reader alive until
    // publication finishes. A competitor can win while we build; its head is
    // never substituted into this captured expected pair.
    let snapshot = selected.open_on_root(cx, &root, options.max_image_bytes)?;
    snapshot.authority().next_sequence()?;
    let source = options.rebuild_source_cut(&snapshot.metadata().commit_range)?;
    validate_successor_source(&snapshot.metadata().commit_range, &source)?;
    let sequence = selected
        .generation
        .sequence
        .checked_add(1)
        .ok_or_else(|| bad("the artifact generation sequence is exhausted"))?;
    let documents = if options.retained_source {
        retained_rebuild_documents(cx, &snapshot, options.max_source_bytes)?
    } else {
        match &options.input {
            Some(path) => read_documents(&mut BufReader::new(File::open(path)?))?,
            None => read_documents(&mut io::stdin().lock())?,
        }
    };
    if documents.is_empty() {
        return Err(bad(
            "the generation metadata contract requires a nonempty source cohort",
        ));
    }
    let model = models::load_single(
        cx,
        options.models.as_deref(),
        selected.tier,
        &options.quality,
        pool,
    )?;
    let expected_identity = rebuild_identity(&selected, model.as_ref(), options.migrate_model)?;
    let built = NativeIndexBuilder::new(&work_path, new_generation(sequence)?, model)?
        .with_batch_size(options.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
        .with_fast_storage(
            NativeBuildPrecision::F32,
            options.build_retrieval(Some(snapshot.retrieval_configuration())),
        )
        .add_documents(documents)
        .build_exact_generation(cx, selected.tier, source, options.max_image_bytes)
        .await?;
    if built.binding().frozen_identity() != &expected_identity {
        return Err(bad("the provider changed after rebuild identity admission"));
    }
    let receipt = stage_and_publish(
        cx,
        &root,
        &selected.root,
        &receipt_path,
        &built,
        Some(&snapshot),
        options.max_image_bytes,
    )?;
    emit(
        output,
        &serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "rebuilt", "publication": "committed",
            "serving": "activated", "retrieval": retrieval_name(built.retrieval_mode()),
            "receipt": receipt_path,
            "source": if options.retained_source { "retained_generation" } else { "input" },
            "replaces": selected.authority, "model_migrated": receipt.identity != selected.identity,
            "selection": receipt,
        }),
    )
    .map_err(|_| {
        publication_error(
            &receipt,
            "committed",
            "activated",
            "result_delivery_failed",
            None,
        )
    })
}

async fn update_generation(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let context = execution_context(cx, pool)?;
    Box::pin(update_generation_with_loader(
        &context,
        options,
        output,
        |tier| {
            models::load_single(
                &context,
                options.models.as_deref(),
                tier,
                &options.quality,
                context.blocking_pool_handle(),
            )
        },
    ))
    .await
}

async fn update_generation_with_loader<F>(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    load_model: F,
) -> Result<()>
where
    F: FnOnce(EmbedderTierTag) -> Result<Arc<dyn Embedder>>,
{
    checkpoint(cx)?;
    let selected = Receipt::read(
        options
            .from_receipt
            .as_deref()
            .ok_or_else(|| bad("missing predecessor receipt"))?,
    )?;
    options
        .quality
        .validate_usage(selected.tier == EmbedderTierTag::Quality)?;
    let work_path = new_path(
        options
            .work
            .as_deref()
            .ok_or_else(|| bad("missing work directory"))?,
    )?;
    let receipt_path = new_path(&options.receipt)?;
    separate_destinations(&[&selected.root, &work_path, &receipt_path])?;
    let root = admit_root(&selected.root)?;
    // Admit the receipt's exact current selection before reading input or
    // loading a model, and retain that expected authority through publication.
    let snapshot = selected.open_on_root(cx, &root, options.max_image_bytes)?;
    snapshot.authority().next_sequence()?;
    let source = options.source_cut()?;
    validate_successor_source(&snapshot.metadata().commit_range, &source)?;
    let sequence = selected
        .generation
        .sequence
        .checked_add(1)
        .ok_or_else(|| bad("the artifact generation sequence is exhausted"))?;
    let documents = retained_rebuild_documents(cx, &snapshot, options.max_source_bytes)?;
    let edits = match &options.input {
        Some(path) => mutations::read_edits(&mut BufReader::new(File::open(path)?))?,
        None => mutations::read_edits(&mut io::stdin().lock())?,
    };
    let document_count = mutations::validate_final_source(cx, &documents, &edits)?;
    if document_count == 0 {
        return Err(bad("authority updates require a nonempty final cohort"));
    }
    // Preparation exports from the same retained immutable image. Release the
    // preflight bodies first rather than holding two copies during inference.
    drop(documents);
    checkpoint(cx)?;
    let model = load_model(selected.tier)?;
    let mut transaction = snapshot
        .begin_update(
            cx,
            &work_path,
            new_generation(sequence)?,
            model,
            options.max_source_bytes,
        )?
        .with_batch_size(options.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?;
    let edit_count = edits.len();
    for (id, document) in edits {
        checkpoint(cx)?;
        transaction = match document {
            Some(document) => transaction.upsert_document(document),
            None => transaction.delete_document(id),
        };
    }
    let built = Box::pin(transaction.build_exact_generation(
        cx,
        selected.tier,
        source,
        options.max_image_bytes,
    ))
    .await?;
    if built.binding().frozen_identity() != &selected.identity {
        return Err(bad("the provider changed after update identity admission"));
    }
    let receipt = stage_and_publish(
        cx,
        &root,
        &selected.root,
        &receipt_path,
        &built,
        Some(&snapshot),
        options.max_image_bytes,
    )?;
    emit(
        output,
        &serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "updated", "publication": "committed",
            "serving": "activated", "retrieval": retrieval_name(built.retrieval_mode()),
            "receipt": receipt_path, "replaces": selected.authority,
            "documents": document_count, "edited_ids": edit_count, "selection": receipt,
        }),
    )
    .map_err(|_| {
        publication_error(
            &receipt,
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
            "schema": SCHEMA, "ok": true, "event": "results",
            "retrieval": retrieval_name(snapshot.retrieval_mode()),
            "tier": receipt.tier, "generation": receipt.generation,
            "authority": snapshot.authority().canonical_bytes().as_slice(),
            "profile": receipt.profile, "results": results,
        }),
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoritySearchRequest {
    id: Option<String>,
    query: String,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorityActivateRequest {
    op: String,
    id: Option<String>,
    receipt: PathBuf,
    expected_authority: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AuthorityRequest {
    Search(AuthoritySearchRequest),
    Activate(AuthorityActivateRequest),
}

#[derive(Debug)]
struct AuthorityServeRefusal(&'static str);

impl fmt::Display for AuthorityServeRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "authority serving refused: {}", self.0)
    }
}

impl Error for AuthorityServeRefusal {}

fn serve_refused(code: &'static str) -> Box<dyn Error + Send + Sync> {
    Box::new(AuthorityServeRefusal(code))
}

fn same_security_object(
    first: GenerationRootObjectWitness,
    second: GenerationRootObjectWitness,
) -> bool {
    first.device() == second.device()
        && first.inode() == second.inode()
        && first.mode() == second.mode()
        && first.uid() == second.uid()
        && first.gid() == second.gid()
        && first.mount_identity() == second.mount_identity()
        && first.filesystem() == second.filesystem()
}

fn same_query_contract(
    first: &FrozenEmbeddingIdentityBundleV1,
    second: &FrozenEmbeddingIdentityBundleV1,
) -> bool {
    first.identity.space == second.identity.space
        && first.identity.producer == second.identity.producer
        && first.identity.input == second.identity.input
}

fn admit_serving_model(receipt: &Receipt, model: &dyn Embedder) -> Result<()> {
    let identity = model
        .identity()
        .map_err(|_| serve_refused("producer_identity_unavailable"))?;
    identity
        .validate()
        .map_err(|_| serve_refused("producer_identity_invalid"))?;
    // Queries carry in-memory vectors. Persisted F16/F32 encoding may change
    // between admitted generations without changing their query producer.
    let expected = &receipt.identity.identity;
    if identity.space != expected.space
        || identity.producer != expected.producer
        || identity.input != expected.input
        || usize::try_from(identity.space.dimension).ok() != Some(model.dimension())
    {
        return Err(serve_refused("producer_identity_mismatch"));
    }
    Ok(())
}

/// One admitted search/model selection. Request execution owns this whole
/// value; no selected component or model is rediscovered between query stages.
struct AuthoritySession {
    root: QualifiedGenerationRoot,
    receipt: Receipt,
    snapshot: NativeGenerationSnapshot,
    model: Arc<dyn Embedder>,
    default_limit: usize,
    max_image_bytes: usize,
    allow_activation: bool,
}

impl AuthoritySession {
    fn new(
        cx: &Cx,
        root: QualifiedGenerationRoot,
        receipt: Receipt,
        snapshot: NativeGenerationSnapshot,
        model: Arc<dyn Embedder>,
        limits: (usize, usize),
        allow_activation: bool,
    ) -> Result<Self> {
        checkpoint(cx)?;
        receipt.validate()?;
        let retained = snapshot.retained_snapshot();
        let anchors = root.anchor_witnesses();
        let old_anchors = retained.anchor_witnesses();
        let budget = u64::try_from(limits.1)?;
        let image_bytes = retained.manifest().components().iter().try_fold(
            u64::try_from(retained.manifest_bytes().len())?,
            |total, (_, component)| total.checked_add(component.byte_len),
        );
        let tier_name = match receipt.tier {
            EmbedderTierTag::Fast => "fast",
            EmbedderTierTag::Quality => "quality",
        };
        if limits.0 == 0
            || limits.0 > 1_000
            || limits.1 == 0
            || image_bytes.is_none_or(|bytes| bytes > budget || bytes > receipt.image_bytes)
            || retained.root_id() != receipt.root_id
            || retained.profile().as_str() != receipt.profile
            || snapshot.authority() != receipt.authority()?
            || snapshot.tier() != receipt.tier
            || snapshot.metadata().total_documents != receipt.documents
            || snapshot.metadata().embedders.get(tier_name) != Some(&receipt.identity)
            || !same_security_object(root.witness(), retained.root_witness())
            || !same_security_object(anchors.lock(), old_anchors.lock())
            || !same_security_object(anchors.authority(), old_anchors.authority())
        {
            return Err(serve_refused("session_selection_mismatch"));
        }
        admit_serving_model(&receipt, model.as_ref())?;
        checkpoint(cx)?;
        Ok(Self {
            root,
            receipt,
            snapshot,
            model,
            default_limit: limits.0,
            max_image_bytes: limits.1,
            allow_activation,
        })
    }

    fn response(
        &self,
        event: &'static str,
        ok: bool,
        request: Option<u64>,
        id: Option<&str>,
    ) -> serde_json::Value {
        serde_json::json!({
            "schema": SCHEMA, "event": event, "ok": ok, "request": request, "id": id,
            "authority": self.snapshot.authority().canonical_bytes().as_slice(),
            "generation": self.receipt.generation, "profile": self.receipt.profile,
            "tier": self.snapshot.tier(),
            "retrieval": retrieval_name(self.snapshot.retrieval_mode()),
        })
    }

    fn failure(
        &self,
        ordinal: u64,
        id: Option<&str>,
        error: &(dyn Error + 'static),
        default_code: &'static str,
    ) -> serde_json::Value {
        let mut response = self.response("failed", false, Some(ordinal), id);
        response["status"] = serde_json::json!("failed");
        response["code"] = serde_json::json!(
            error
                .downcast_ref::<AuthorityServeRefusal>()
                .map_or(default_code, |error| error.0)
        );
        // Never forward provider error bodies, model paths, receipt JSON, or
        // unvalidated request IDs into the persistent protocol.
        response
    }

    async fn query(
        &self,
        cx: &Cx,
        request: &AuthoritySearchRequest,
    ) -> Result<Vec<frankensearch::ScoredResult>> {
        checkpoint(cx)?;
        super::serve::validate_id(request.id.as_deref())
            .map_err(|_| serve_refused("invalid_request_id"))?;
        validate_query(&request.query).map_err(|_| serve_refused("invalid_query"))?;
        let limit = request.limit.unwrap_or(self.default_limit);
        if limit == 0 || limit > 1_000 {
            return Err(serve_refused("invalid_limit"));
        }
        admit_serving_model(&self.receipt, self.model.as_ref())?;
        // NativeGenerationSnapshot independently checks the producer before
        // inference and the actual bound output afterward, including zero hits.
        let results = self
            .snapshot
            .search_text(cx, self.model.as_ref(), &request.query, limit)
            .await?;
        checkpoint(cx)?;
        Ok(results)
    }

    fn activate(&mut self, cx: &Cx, request: &AuthorityActivateRequest) -> Result<()> {
        checkpoint(cx)?;
        if !self.allow_activation {
            return Err(serve_refused("activation_disabled"));
        }
        super::serve::validate_id(request.id.as_deref())
            .map_err(|_| serve_refused("invalid_request_id"))?;
        if request.op != "activate" || !request.receipt.is_absolute() {
            return Err(serve_refused("invalid_activation_request"));
        }
        let expected = AuthorityRefV1::from_canonical_bytes(&request.expected_authority)
            .map_err(|_| serve_refused("invalid_expected_authority"))?;
        if expected != self.snapshot.authority() {
            return Err(serve_refused("stale_expected_authority"));
        }
        let candidate = Receipt::read(&request.receipt)
            .map_err(|_| serve_refused("invalid_activation_receipt"))?;
        if candidate.root != self.receipt.root
            || candidate.root_id != self.receipt.root_id
            || candidate.profile != self.receipt.profile
            || candidate.tier != self.receipt.tier
        {
            return Err(serve_refused("activation_root_or_tier_mismatch"));
        }
        if candidate.authority()?.sequence <= self.snapshot.authority().sequence {
            return Err(serve_refused("activation_not_newer"));
        }
        if !same_query_contract(&candidate.identity, &self.receipt.identity) {
            return Err(serve_refused("activation_producer_mismatch"));
        }
        admit_serving_model(&candidate, self.model.as_ref())?;
        checkpoint(cx)?;
        // The receipt selects an authority on the original qualified physical
        // root. Never reopen the candidate's root pathname or adopt a copy.
        let snapshot = candidate
            .open_on_root(cx, &self.root, self.max_image_bytes)
            .map_err(|_| serve_refused("activation_admission_failed"))?;
        admit_serving_model(&candidate, self.model.as_ref())?;
        checkpoint(cx)?;
        // All fallible work is complete. The request loop owns both fields and
        // cannot serve a query between these assignments.
        self.snapshot = snapshot;
        self.receipt = candidate;
        Ok(())
    }
}

async fn serve_session(
    cx: &Cx,
    session: &mut AuthoritySession,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    checkpoint(cx)?;
    admit_serving_model(&session.receipt, session.model.as_ref())?;
    let mut ready = session.response("ready", true, None, None);
    ready["documents"] = serde_json::json!(session.receipt.documents);
    ready["activation_enabled"] = serde_json::json!(session.allow_activation);
    ready["default_limit"] = serde_json::json!(session.default_limit);
    ready["max_request_bytes"] = serde_json::json!(super::serve::MAX_REQUEST_BYTES);
    emit(output, &ready)?;
    let mut record = Vec::new();
    let mut ordinal = 0_u64;
    while super::serve::read_request(cx, input, &mut record)? {
        ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| serve_refused("request_ordinal_exhausted"))?;
        let request = match serde_json::from_slice::<AuthorityRequest>(record.trim_ascii()) {
            Ok(request) => request,
            Err(_) => {
                emit(
                    output,
                    &session.failure(
                        ordinal,
                        None,
                        serve_refused("invalid_request").as_ref(),
                        "invalid_request",
                    ),
                )?;
                continue;
            }
        };
        let (id, result) = match &request {
            AuthorityRequest::Search(request) => {
                let id = request
                    .id
                    .as_deref()
                    .filter(|id| super::serve::validate_id(Some(id)).is_ok());
                let result = session.query(cx, request).await.map(|results| {
                    let mut response = session.response("results", true, Some(ordinal), id);
                    response["results"] = serde_json::json!(results);
                    response
                });
                (id, result)
            }
            AuthorityRequest::Activate(request) => {
                let id = request
                    .id
                    .as_deref()
                    .filter(|id| super::serve::validate_id(Some(id)).is_ok());
                let previous = session.snapshot.authority();
                let result = session.activate(cx, request).map(|()| {
                    let mut response = session.response("activated", true, Some(ordinal), id);
                    response["previous_authority"] =
                        serde_json::json!(previous.canonical_bytes().as_slice());
                    response["documents"] = serde_json::json!(session.receipt.documents);
                    response
                });
                (id, result)
            }
        };
        checkpoint(cx)?;
        match result {
            Ok(response) => emit(output, &response)?,
            Err(error) => {
                if error
                    .downcast_ref::<SearchError>()
                    .is_some_and(|error| matches!(error, SearchError::Cancelled { .. }))
                {
                    return Err(error);
                }
                emit(
                    output,
                    &session.failure(ordinal, id, error.as_ref(), "query_or_activation_refused"),
                )?;
            }
        }
    }
    Ok(())
}

async fn serve(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let receipt = Receipt::read(&options.receipt)?;
    options
        .quality
        .validate_usage(receipt.tier == EmbedderTierTag::Quality)?;
    let root = admit_root(&receipt.root)?;
    let snapshot = receipt.open_on_root(cx, &root, options.max_image_bytes)?;
    // Admission precedes model loading and every subsequent query. The model
    // and complete retained images remain warm until explicit activation/exit.
    let model = models::load_single(
        cx,
        options.models.as_deref(),
        receipt.tier,
        &options.quality,
        pool,
    )?;
    let mut session = AuthoritySession::new(
        cx,
        root,
        receipt,
        snapshot,
        model,
        (options.limit, options.max_image_bytes),
        options.allow_activation,
    )?;
    serve_session(cx, &mut session, &mut io::stdin().lock(), output).await
}

fn reconciliation_report(
    receipt: &Receipt,
    permit: &AttemptPermitV1,
    outcome: &ReconcileOutcomeV1,
) -> serde_json::Value {
    let (publication, reconciliation, serving, resolved) = match outcome {
        ReconcileOutcomeV1::Committed { .. } => ("committed", "committed", "not_activated", true),
        ReconcileOutcomeV1::NotCommitted => {
            ("not_committed", "not_committed", "not_activated", true)
        }
        ReconcileOutcomeV1::StillUnknown(_) => ("unknown", "still_unknown", "not_activated", false),
        ReconcileOutcomeV1::Fork { .. } => ("unknown", "fork", "not_activated", false),
        ReconcileOutcomeV1::RepairRequired(_) => {
            ("unknown", "repair_required", "not_activated", false)
        }
        ReconcileOutcomeV1::CommittedThenSuperseded { .. } => {
            ("committed", "committed_then_superseded", "superseded", true)
        }
    };
    let mut report = serde_json::json!({
        "schema": SCHEMA, "ok": resolved, "event": "recovered",
        "publication": publication, "reconciliation": reconciliation, "serving": serving,
        "root": receipt.root, "root_id": receipt.root_id,
        "authority": receipt.authority, "attempt": attempt_report(permit),
    });
    match outcome {
        ReconcileOutcomeV1::Fork { observed } => {
            report["observed_authority"] =
                serde_json::json!(observed.authority.canonical_bytes().as_slice());
        }
        ReconcileOutcomeV1::RepairRequired(error) => {
            report["reason"] = serde_json::json!(error.to_string());
        }
        ReconcileOutcomeV1::CommittedThenSuperseded { head } => {
            report["current_authority"] =
                serde_json::json!(head.authority.canonical_bytes().as_slice());
        }
        ReconcileOutcomeV1::StillUnknown(current) => {
            report["attempt"] = attempt_report(current);
        }
        ReconcileOutcomeV1::Committed { .. } | ReconcileOutcomeV1::NotCommitted => {}
    }
    report
}

fn emit_recovery(output: &mut impl Write, mut report: serde_json::Value) -> Result<()> {
    if report["ok"] != true {
        return Err(Box::new(PublicationFailure(report)));
    }
    emit(output, &report).map_err(|_| {
        report["ok"] = serde_json::json!(false);
        report["reason"] = serde_json::json!("result_delivery_failed");
        Box::new(PublicationFailure(report)) as Box<dyn Error + Send + Sync>
    })
}

fn recover(cx: &Cx, options: &Options, output: &mut impl Write) -> Result<()> {
    let receipt = Receipt::read(&options.receipt)?;
    let permit = receipt.recovery_attempt()?;
    checkpoint(cx)?;
    let root = admit_root(&receipt.root)?;
    let writer = WriterIdentityV1::for_current_process()?;
    let publisher = root.authority_publisher(receipt.root_id, writer.writer_id, PROFILE)?;
    checkpoint(cx)?;
    // Reconcile the original exact attempt. Never publish, infer a new attempt,
    // or load models here. Reader admission is separate and remains bounded.
    let outcome = publisher.reconcile(&permit, None).map_err(|error| {
        publication_error(
            &receipt,
            "unknown",
            "not_activated",
            &error.to_string(),
            Some(&permit),
        )
    })?;
    let mut report = reconciliation_report(&receipt, &permit, &outcome);
    if matches!(outcome, ReconcileOutcomeV1::Committed { .. }) {
        match receipt.open(cx, options.max_image_bytes) {
            Ok(snapshot) => {
                report["serving"] = serde_json::json!("activated");
                report["retrieval"] = serde_json::json!(retrieval_name(snapshot.retrieval_mode()));
            }
            Err(error) => {
                report["ok"] = serde_json::json!(false);
                report["reason"] = serde_json::json!("snapshot_activation_deferred_or_refused");
                report["detail"] = serde_json::json!(error.to_string());
            }
        }
    }
    emit_recovery(output, report)
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
        let cx = execution_context(&cx, pool.clone())?;
        let mut output = io::stdout().lock();
        match options.command {
            Command::Index => Box::pin(index(&cx, &options, &mut output, pool)).await,
            Command::Rebuild => Box::pin(rebuild(&cx, &options, &mut output, pool)).await,
            Command::Update => Box::pin(update_generation(&cx, &options, &mut output, pool)).await,
            Command::Search => Box::pin(search(&cx, &options, &mut output, pool)).await,
            Command::Serve => Box::pin(serve(&cx, &options, &mut output, pool)).await,
            Command::Recover => recover(&cx, &options, &mut output),
        }
    })
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
