//! Executable build, fixed-authority publication, and retained hybrid search.
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
use frankensearch::native_ann::builder::scope::NativeScopedGenerationSnapshot;
use frankensearch::native_ann::builder::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltExactGeneration, NativeIndexBuilder,
};
use frankensearch::native_ann::{
    NativeGenerationSnapshot, NativeGenerationTopology, NativeRetrievalMode, NativeSearchPhase,
};
use frankensearch::{
    Cx, Embedder, GenerationActivationV1, GenerationCandidateV1, GenerationPublisherV1,
    IndexableDocument, ScoredResult, SearchError,
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
    MAX_DOCUMENTS, MAX_INPUT_BYTES, MAX_RECEIPT_BYTES, MAX_RECORD_BYTES, Models, Result, SCHEMA,
    bad, emit, encode, execution_context, filter::Filter, models, new_generation, new_path,
    positive as number, read_documents, serve::source, update as mutations, validate_query, value,
};

const RECEIPT_SCHEMA: &str = "frankensearch.native.authority-selection.v2";
const PROGRESSIVE_RECEIPT_SCHEMA: &str = "frankensearch.native.authority-selection.v3";
const SELECTION_ONLY_RECEIPT_SCHEMA: &str = "frankensearch.native.authority-selection.v1";
const DEFAULT_IMAGE_BYTES: usize = 256 * 1024 * 1024;
const MAX_AUTHORITY_MUTATIONS: usize = 1_000;
const PROFILE: GenerationRootSecurityProfileV1 = GenerationRootSecurityProfileV1::CooperativeLocal;
const HELP: &str = "Fixed AUTHORITY search (explicit fast, quality, or full topology)\n\n\
  authority-index --root NEW_ROOT --work-dir NEW_DIR --receipt NEW_JSON\n\
      --tier fast|quality|full --source-low N --source-high N --cooperative-local\n\
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
  authority-search --receipt JSON --query TEXT [--limit N] [--lexical-only]\n\
      [--model-dir DIR] [--quality-backend BACKEND] [--quality-model-dir DIR]\n\
      [--mode fast|quality|full|lexical] [--stream] [--filter JSON]\n\
      [--max-source-bytes N] [--max-image-bytes N]\n\n\
  authority-fetch --receipt JSON --doc-id ID [--offset-bytes N] [--max-bytes N]\n\
      [--include-metadata] [--filter JSON] [--max-source-bytes N]\n\
      [--max-image-bytes N]\n\n\
  authority-serve --receipt JSON [--limit N] [--allow-activation] [--allow-updates]\n\
      [--lexical-only]\n\
      [--model-dir DIR] [--quality-backend BACKEND] [--quality-model-dir DIR]\n\
      [--filter JSON] [--max-source-bytes N] [--max-image-bytes N] [--batch-size N]\n\n\
  authority-recover --receipt JSON [--max-image-bytes N]\n\n\
  authority-rollback --from-receipt CURRENT_JSON --target-receipt PREDECESSOR_JSON\n\
      --receipt NEW_JSON [--allow-damaged-current] [--max-image-bytes N]\n\n\
This route builds one or two FSVI v2 tiers plus a retained Quill image.\n\
Full generations bind both producers and complete document coverage to one\n\
authority. Full search independently retrieves quality candidates, including\n\
documents absent from fast results. --stream emits Initial then Refined from\n\
the same retained generation. --mode fast or quality loads only that producer.\n\
Full generations currently support exact retrieval; partial/sharded cohorts\n\
and two-tier HNSW are not accepted by this fixed-authority route.\n\
New indexes use exact retrieval unless --retrieval hnsw explicitly selects ANN.\n\
The selected ANN graph and its source identity are authenticated by AUTHORITY;\n\
missing or invalid graphs fail admission without substituting exact retrieval.\n\
Quality-only generations load and query their quality model directly. Models\n\
are local-only; unavailable or mismatched models fail without substitution.\n\
Source-low/high identify the inclusive source checkpoint supplied by the caller.\n\
Roots, work directories and receipts must be new, separate locations.\n\
Rebuild retains the selected root, topology and security profile, replaces the whole\n\
source cohort, and requires a new work directory and receipt outside the root.\n\
Source boundaries cannot regress. A change to either producer requires --migrate-model.\n\
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
the selected producers, topology, precision, and retrieval policy. Only changed or new\n\
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
authority-rollback restores the immediate authenticated predecessor of a healthy\n\
current generation. It publishes a new higher authority sequence using the\n\
target's original images, producer and source checkpoint, without inference.\n\
Both receipts must name the same root, security profile and topology. The target\n\
must still occupy the predecessor slot; arbitrary history is not discovered.\n\
The new receipt is outside the root and supports authority-recover after a crash.\n\
Rollback normally admits the complete current generation. Explicit\n\
--allow-damaged-current instead authenticates its exact control state and\n\
activation manifest, then fully admits the selected predecessor. This recovers\n\
from damaged or missing current component images without serving them. Corrupt\n\
control anchors, current manifests, unsafe root entries, or pending publication\n\
still refuse. Its image budget covers both manifests and the target components.\n\
Recovery reports commit knowledge separately from bounded reader admission.\n\
Publication failures report not_committed, unknown, or committed separately\n\
from serving activation. Existing roots and authority heads are never replaced\n\
by authority-index. Search refuses a head different from the receipt.\n\
The default encoded-image budget is 256 MiB, checked before component reads.\n\
authority-serve retains one admitted generation and its local models for bounded\n\
JSONL requests: {\"id\":\"q1\",\"query\":\"text\",\"limit\":10}. Requests may select\n\
\"mode\":\"fast\"|\"quality\"|\"full\" and \"stream\":true. Full is the default for\n\
a full generation. Streaming emits separate Initial and Refined frames. It never follows a new\n\
authority implicitly. With --allow-activation, an explicit request may select a\n\
new receipt: {\"op\":\"activate\",\"receipt\":\"/absolute/new.json\",\n\
\"expected_authority\":[...the current ready/results authority bytes...]}.\n\
Activation keeps the same qualified root, topology and security profile. Semantic\n\
sessions also require the same producers; lexical sessions have no loaded models.\n\
It admits the complete currently selected successor before swapping readers;\n\
rejection retains the previous reader and returns a bounded failed response.\n\
Authority sequence must advance, including rollback to older artifact generations.\n\
Model migration requires restarting with the new receipt and local model options.\n\
Explicit --lexical-only on authority-search/serve runs retained Quill keyword\n\
search without loading or invoking a model. It excludes model options and\n\
--allow-updates, but permits source fetch and granted activation. The complete\n\
authority still must pass component admission. Results report mode/retrieval\n\
lexical and raw BM25 scores. Warm requests may choose mode=lexical; a lexical\n\
session refuses fast/quality/full. There is no automatic fallback after a model error.\n\
With --allow-updates, an explicit update request can publish and serve a successor\n\
using the retained model: op=update, expected_authority, work_dir, new_receipt,\n\
source_low, source_high, and changes (upsert/delete objects). Destinations must be\n\
new absolute paths. At most 1000 mutations fit within the 1 MiB request budget;\n\
an empty delta may advance the checkpoint. --batch-size requires this flag.\n\
Updates retain producer, tier, precision and retrieval policy. They save a recovery\n\
receipt before publication, compare the original authority pair, and install the\n\
exact admitted successor. They report publication separately from reader activation.\n\
Queries retain the old selection after refusal or unresolved publication; further\n\
writes require the retained authority to remain current with no unresolved attempt.\n\
Search/fetch/serve --filter accepts exact IDs, a literal id_prefix, and string-valued\n\
metadata. Every present constraint must match the retained source document.\n\
Search requests may include a filter object; it intersects the server's fixed\n\
filter and cannot widen it. Membership filters vector and lexical candidates\n\
before fusion. Hybrid queries, including empty scopes, require the selected producer.\n\
Filtered source export is bounded by --max-source-bytes (default 256 MiB).\n\
The fixed server scope stays on its retained generation. Activation and updates\n\
prepare its successor membership before changing selection or publication.\n\
Unfiltered queries do not export source.\n\
authority-fetch reads one selected document window without loading a model.\n\
Its receipt must still select the exact current authority. Offsets default to\n\
zero, max-bytes to 8192 (allowed 4..65536), and metadata is opt-in. The source\n\
budget covers that document's full stored fields before copying, not the corpus.\n\
Read-only source requests use op=fetch, expected_authority, and documents with\n\
doc_id, offset_bytes, max_bytes and include_metadata. The exact retained authority\n\
must match before source lookup. Requests allow up to 32 UTF-8 windows, at most\n\
64 KiB each; the complete encoded reply is limited to 1 MiB. Fixed and request\n\
filters restrict source access.\n\
Unfiltered fetch materializes only distinct requested documents, bounded by\n\
--max-source-bytes; no model runs and no current path or authority is reopened.\n\
Requests are limited to 1 MiB; oversized input or output failure ends the session.\n\
Send quit/exit or close stdin to stop. Reads block on the owning execution lane.\n\
Work files remain available after completion or failure. Stdout is JSON.\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Index,
    Rebuild,
    Update,
    Search,
    Fetch,
    Serve,
    Recover,
    Rollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum QueryMode {
    Fast,
    Quality,
    Full,
    Lexical,
}

impl QueryMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Quality => "quality",
            Self::Full => "full",
            Self::Lexical => "lexical",
        }
    }
}

struct Options {
    command: Command,
    root: Option<PathBuf>,
    work: Option<PathBuf>,
    receipt: PathBuf,
    from_receipt: Option<PathBuf>,
    target_receipt: Option<PathBuf>,
    input: Option<PathBuf>,
    models: Option<PathBuf>,
    quality: models::QualityOptions,
    tier: Option<EmbedderTierTag>,
    progressive: bool,
    mode: Option<QueryMode>,
    stream: bool,
    retrieval: Option<NativeBuildRetrieval>,
    source_low: Option<u64>,
    source_high: Option<u64>,
    query: Option<String>,
    doc_id: Option<String>,
    offset_bytes: usize,
    max_bytes: usize,
    include_metadata: bool,
    lexical_only: bool,
    filter: Option<Filter>,
    limit: usize,
    batch_size: usize,
    max_image_bytes: usize,
    max_source_bytes: usize,
    cooperative_local: bool,
    migrate_model: bool,
    retained_source: bool,
    allow_activation: bool,
    allow_updates: bool,
    allow_damaged_current: bool,
}

pub fn is_command(command: &str) -> bool {
    matches!(
        command,
        "authority-index"
            | "authority-rebuild"
            | "authority-update"
            | "authority-search"
            | "authority-fetch"
            | "authority-serve"
            | "authority-recover"
            | "authority-rollback"
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
            Some("authority-fetch") => Command::Fetch,
            Some("authority-serve") => Command::Serve,
            Some("authority-recover") => Command::Recover,
            Some("authority-rollback") => Command::Rollback,
            _ => return Err(bad("unknown authority command")),
        };
        let mut options = Self {
            command,
            root: None,
            work: None,
            receipt: PathBuf::new(),
            from_receipt: None,
            target_receipt: None,
            input: None,
            models: None,
            quality: models::QualityOptions::default(),
            tier: None,
            progressive: false,
            mode: None,
            stream: false,
            retrieval: None,
            source_low: None,
            source_high: None,
            query: None,
            doc_id: None,
            offset_bytes: 0,
            max_bytes: 8 * 1024,
            include_metadata: false,
            lexical_only: false,
            filter: None,
            limit: 10,
            batch_size: 64,
            max_image_bytes: DEFAULT_IMAGE_BYTES,
            max_source_bytes: MAX_INPUT_BYTES,
            cooperative_local: false,
            migrate_model: false,
            retained_source: false,
            allow_activation: false,
            allow_updates: false,
            allow_damaged_current: false,
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
                "--from-receipt"
                    if matches!(
                        command,
                        Command::Rebuild | Command::Update | Command::Rollback
                    ) =>
                {
                    options.from_receipt = Some(PathBuf::from(value(&mut args)?));
                }
                "--target-receipt" if command == Command::Rollback => {
                    options.target_receipt = Some(PathBuf::from(value(&mut args)?));
                }
                "--allow-damaged-current" if command == Command::Rollback => {
                    options.allow_damaged_current = true;
                }
                "--migrate-model" if command == Command::Rebuild => options.migrate_model = true,
                "--retained-source" if command == Command::Rebuild => {
                    options.retained_source = true;
                }
                "--max-source-bytes"
                    if matches!(
                        command,
                        Command::Rebuild
                            | Command::Update
                            | Command::Search
                            | Command::Fetch
                            | Command::Serve
                    ) =>
                {
                    options.max_source_bytes = number(&value(&mut args)?, usize::MAX)?;
                }
                "--filter"
                    if matches!(command, Command::Search | Command::Fetch | Command::Serve) =>
                {
                    options.filter = Some(Filter::parse(&value(&mut args)?)?);
                }
                "--allow-activation" if command == Command::Serve => {
                    options.allow_activation = true;
                }
                "--allow-updates" if command == Command::Serve => options.allow_updates = true,
                "--lexical-only" if matches!(command, Command::Search | Command::Serve) => {
                    options.lexical_only = true;
                }
                "--model-dir"
                    if !matches!(
                        command,
                        Command::Fetch | Command::Recover | Command::Rollback
                    ) =>
                {
                    options.models = Some(PathBuf::from(value(&mut args)?));
                }
                "--quality-backend"
                    if !matches!(
                        command,
                        Command::Fetch | Command::Recover | Command::Rollback
                    ) =>
                {
                    options.quality.backend =
                        Some(models::QualityBackend::parse(&value(&mut args)?)?);
                }
                "--quality-model-dir"
                    if !matches!(
                        command,
                        Command::Fetch | Command::Recover | Command::Rollback
                    ) =>
                {
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
                        "full" => {
                            options.progressive = true;
                            EmbedderTierTag::Fast
                        }
                        _ => return Err(bad("authority tier must be fast, quality, or full")),
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
                    if matches!(
                        command,
                        Command::Index | Command::Rebuild | Command::Update | Command::Serve
                    ) =>
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
                "--doc-id" if command == Command::Fetch => {
                    options.doc_id = Some(value(&mut args)?);
                }
                "--offset-bytes" if command == Command::Fetch => {
                    options.offset_bytes = value(&mut args)?
                        .parse()
                        .map_err(|_| bad("fetch offset must be an unsigned byte count"))?;
                }
                "--max-bytes" if command == Command::Fetch => {
                    options.max_bytes = number(&value(&mut args)?, source::MAX_CONTENT_BYTES)?;
                }
                "--include-metadata" if command == Command::Fetch => {
                    options.include_metadata = true;
                }
                "--limit" if matches!(command, Command::Search | Command::Serve) => {
                    options.limit = number(&value(&mut args)?, 1_000)?;
                }
                "--mode" if command == Command::Search => {
                    options.mode = Some(match value(&mut args)?.as_str() {
                        "fast" => QueryMode::Fast,
                        "quality" => QueryMode::Quality,
                        "full" => QueryMode::Full,
                        "lexical" => QueryMode::Lexical,
                        _ => {
                            return Err(bad(
                                "authority search mode must be fast, quality, full, or lexical",
                            ));
                        }
                    });
                }
                "--stream" if command == Command::Search => options.stream = true,
                _ => return Err(bad("unknown or inapplicable authority option")),
            }
        }
        if options.receipt.as_os_str().is_empty() {
            return Err(bad("authority commands require --receipt"));
        }
        if options.lexical_only && options.mode.is_some_and(|mode| mode != QueryMode::Lexical) {
            return Err(bad("--lexical-only conflicts with a semantic --mode"));
        }
        options.lexical_only |= options.mode == Some(QueryMode::Lexical);
        if options.lexical_only
            && (options.models.is_some()
                || options.quality.backend.is_some()
                || options.quality.directory.is_some()
                || options.allow_updates)
        {
            return Err(bad(
                "--lexical-only excludes model options and --allow-updates",
            ));
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
                    .validate_usage(tier == EmbedderTierTag::Quality || options.progressive)?;
                validate_full_retrieval(options.progressive, options.build_retrieval(None))?;
                options.source_cut()?;
            }
            Command::Search => validate_query(
                options
                    .query
                    .as_deref()
                    .ok_or_else(|| bad("authority-search requires --query"))?,
            )?,
            Command::Fetch => {
                options.fetch_document()?;
            }
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
            Command::Rollback => {
                if options.from_receipt.is_none() || options.target_receipt.is_none() {
                    return Err(bad(
                        "authority-rollback requires --from-receipt and --target-receipt",
                    ));
                }
            }
            Command::Serve => {
                if !options.allow_updates && seen.contains("--batch-size") {
                    return Err(bad("authority-serve --batch-size requires --allow-updates"));
                }
            }
            Command::Recover => {}
        }
        Ok(Some(options))
    }

    fn fetch_document(&self) -> Result<source::DocumentRequest> {
        source::DocumentRequest::new(
            self.doc_id
                .clone()
                .ok_or_else(|| bad("authority-fetch requires --doc-id"))?,
            self.offset_bytes,
            self.max_bytes,
            self.include_metadata,
        )
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

fn validate_full_retrieval(progressive: bool, retrieval: NativeBuildRetrieval) -> Result<()> {
    if progressive && !matches!(retrieval, NativeBuildRetrieval::Exact) {
        return Err(bad(
            "full authority generations currently require exact retrieval for both tiers",
        ));
    }
    Ok(())
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    quality_identity: Option<FrozenEmbeddingIdentityBundleV1>,
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
        match (
            self.schema.as_str(),
            self.attempt.as_ref(),
            self.quality_identity.as_ref(),
        ) {
            (RECEIPT_SCHEMA, Some(attempt), None)
            | (PROGRESSIVE_RECEIPT_SCHEMA, Some(attempt), Some(_)) => {
                // This CLI never supplies an external floor. A saved attempt
                // requiring one cannot be downgraded during a process restart.
                if attempt.requires_floor {
                    return Err(bad(
                        "the recorded attempt requires an unavailable anti-rollback floor",
                    ));
                }
                attempt.restore(self.root_id, authority)?;
            }
            (SELECTION_ONLY_RECEIPT_SCHEMA, None, None) => {}
            _ => {
                return Err(bad(
                    "authority receipt schema and prepared attempt disagree",
                ));
            }
        }
        if self.quality_identity.is_some() && self.tier != EmbedderTierTag::Fast {
            return Err(bad(
                "a full authority receipt must bind fast and quality independently",
            ));
        }
        if self.image_bytes < authority.manifest_len {
            return Err(bad("authority receipt image length is inconsistent"));
        }
        self.binding()?;
        self.quality_binding()?;
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

    fn quality_binding(&self) -> Result<Option<FsviV2IdentityBinding>> {
        self.quality_identity
            .as_ref()
            .map(|identity| {
                FsviV2IdentityBinding::new(self.generation, identity.clone()).map_err(Into::into)
            })
            .transpose()
    }

    fn topology(&self) -> NativeGenerationTopology {
        if self.quality_identity.is_some() {
            NativeGenerationTopology::FullProgressive
        } else if self.tier == EmbedderTierTag::Quality {
            NativeGenerationTopology::QualityOnly
        } else {
            NativeGenerationTopology::FastOnly
        }
    }

    fn topology_name(&self) -> &'static str {
        match self.topology() {
            NativeGenerationTopology::FastOnly => "fast_only",
            NativeGenerationTopology::QualityOnly => "quality_only",
            NativeGenerationTopology::FullProgressive => "full_progressive",
        }
    }

    fn query_mode(&self, requested: Option<QueryMode>) -> Result<QueryMode> {
        let default = match self.topology() {
            NativeGenerationTopology::FastOnly => QueryMode::Fast,
            NativeGenerationTopology::QualityOnly => QueryMode::Quality,
            NativeGenerationTopology::FullProgressive => QueryMode::Full,
        };
        let mode = requested.unwrap_or(default);
        if mode == QueryMode::Lexical || mode == default || self.quality_identity.is_some() {
            Ok(mode)
        } else {
            Err(serve_refused("query_tier_unavailable"))
        }
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
        let snapshot = if let Some(quality) = self.quality_binding()? {
            NativeGenerationSnapshot::admit_progressive(
                cx,
                opened,
                &self.binding()?,
                &quality,
                DEFAULT_SCHEMA,
                QuillConfig::default(),
                max_image_bytes,
            )?
        } else {
            NativeGenerationSnapshot::admit_exact(
                cx,
                opened,
                &self.binding()?,
                DEFAULT_SCHEMA,
                QuillConfig::default(),
                max_image_bytes,
            )?
        };
        if snapshot.topology() != self.topology()
            || snapshot.metadata().total_documents != self.documents
        {
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

struct PublishedGeneration {
    receipt: Receipt,
    snapshot: Arc<NativeGenerationSnapshot>,
    scope: Option<NativeScopedGenerationSnapshot>,
}

struct PublicationOptions<'a> {
    max_image_bytes: usize,
    scope: Option<(&'a Filter, usize)>,
}

impl PublicationOptions<'_> {
    fn unscoped(max_image_bytes: usize) -> Self {
        Self {
            max_image_bytes,
            scope: None,
        }
    }
}

fn stage_and_publish(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    root_path: &Path,
    receipt_path: &Path,
    built: &NativeBuiltExactGeneration,
    predecessor: Option<&NativeGenerationSnapshot>,
    options: PublicationOptions<'_>,
) -> Result<PublishedGeneration> {
    let max_image_bytes = options.max_image_bytes;
    let (root_id, expected, previous) = match predecessor {
        Some(snapshot) => {
            if snapshot
                .retained_snapshot()
                .selected_predecessor_of()
                .is_some()
            {
                return Err(serve_refused("update_base_is_selected_predecessor"));
            }
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
                || built.quality_binding().is_some()
                    != (snapshot.topology() == NativeGenerationTopology::FullProgressive)
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
    // Scope preparation uses the actual completed images, including stored
    // metadata encoding. Any budget or predicate refusal precedes publication.
    let prepared_scope = if let Some((filter, budget)) = options.scope {
        Some(built.prepare_scope(cx, &candidate, budget, |document| {
            Ok(filter.matches(document))
        })?)
    } else {
        built.validate_candidate(cx, &candidate)?;
        None
    };
    let mut publisher = GenerationPublisherV1::new(root, root_id, writer.writer_id, PROFILE)?;
    let attempt = publisher.prepare_attempt(
        &candidate,
        expected,
        new_generation(1)?.nonce,
        new_generation(1)?.nonce,
    )?;
    let receipt = Receipt {
        schema: if built.quality_binding().is_some() {
            PROGRESSIVE_RECEIPT_SCHEMA
        } else {
            RECEIPT_SCHEMA
        }
        .to_owned(),
        root: root_path.to_path_buf(),
        root_id,
        profile: PROFILE.as_str().to_owned(),
        authority: candidate.authority().canonical_bytes().to_vec(),
        generation: built.binding().generation(),
        identity: built.binding().frozen_identity().clone(),
        tier: built.metadata().vector_artifacts[0].embedder_tier,
        quality_identity: built
            .quality_binding()
            .map(|binding| binding.frozen_identity().clone()),
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
    let snapshot = publish_and_admit(
        cx,
        &mut publisher,
        &candidate,
        &attempt,
        &receipt,
        max_image_bytes,
    )
    .map_err(|mut error| {
        // Only this boundary follows a successfully synced recovery receipt.
        // Earlier staging failures must not advertise a usable saved attempt.
        if let Some(failure) = error.downcast_mut::<PublicationFailure>() {
            failure.0["receipt"] = serde_json::json!(receipt_path);
        }
        error
    })?;
    let snapshot = Arc::new(snapshot);
    // Binding only checks the sealed candidate identity; source, membership,
    // budgets and predicates are already complete. It performs no checkpoint.
    let scope = prepared_scope
        .map(|prepared| prepared.bind(Arc::clone(&snapshot)))
        .transpose()
        .map_err(|_| {
            let mut error = publication_error(
                &receipt,
                "committed",
                "not_activated",
                "prepared_scope_binding_refused",
                None,
            );
            if let Some(failure) = error.downcast_mut::<PublicationFailure>() {
                failure.0["receipt"] = serde_json::json!(receipt_path);
            }
            error
        })?;
    Ok(PublishedGeneration {
        receipt,
        snapshot,
        scope,
    })
}

fn publish_and_admit(
    cx: &Cx,
    publisher: &mut GenerationPublisherV1<'_>,
    candidate: &GenerationCandidateV1,
    attempt: &AttemptPermitV1,
    receipt: &Receipt,
    max_image_bytes: usize,
) -> Result<NativeGenerationSnapshot> {
    let result = publisher
        .publish_prepared(
            cx,
            candidate,
            attempt,
            None,
            u64::try_from(max_image_bytes)?,
        )
        .map_err(|error| {
            publication_error(
                receipt,
                "not_committed",
                "not_activated",
                &error.to_string(),
                None,
            )
        })?;
    match result.publication {
        PublicationOutcomeV1::NotCommitted(reason) => {
            return Err(publication_error(
                receipt,
                "not_committed",
                "not_activated",
                &format!("{reason:?}"),
                None,
            ));
        }
        PublicationOutcomeV1::CommitOutcomeUnknown(permit) => {
            return Err(publication_error(
                receipt,
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
            receipt,
            "committed",
            "not_activated",
            "snapshot_activation_deferred_or_refused",
            None,
        ));
    };
    receipt
        .admit_opened(cx, opened, max_image_bytes)
        .map_err(|_| {
            publication_error(
                receipt,
                "committed",
                "not_activated",
                "engine_admission_refused",
                None,
            )
        })
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
    let mode = if options.progressive {
        QueryMode::Full
    } else if tier == EmbedderTierTag::Quality {
        QueryMode::Quality
    } else {
        QueryMode::Fast
    };
    let loaded = load_authority_models(cx, options, mode, pool)?;
    let mut builder = NativeIndexBuilder::new(&work_path, new_generation(1)?, loaded.fast)?
        .with_batch_size(options.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
        .with_fast_storage(NativeBuildPrecision::F32, options.build_retrieval(None))
        .add_documents(documents);
    if let Some(quality) = loaded.quality {
        builder = builder
            .with_quality_embedder(quality)?
            .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)?;
    }
    let built = if options.progressive {
        builder
            .build_progressive_generation(cx, options.source_cut()?, options.max_image_bytes)
            .await?
    } else {
        builder
            .build_exact_generation(cx, tier, options.source_cut()?, options.max_image_bytes)
            .await?
    };
    let PublishedGeneration { receipt, .. } = stage_and_publish(
        cx,
        &root,
        &root_path,
        &receipt_path,
        &built,
        None,
        PublicationOptions::unscoped(options.max_image_bytes),
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
    expected: &FrozenEmbeddingIdentityBundleV1,
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
    if !migrate_model && &frozen != expected {
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
    options.quality.validate_usage(
        selected.tier == EmbedderTierTag::Quality || selected.quality_identity.is_some(),
    )?;
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
    let retrieval = options.build_retrieval(Some(snapshot.retrieval_configuration()));
    validate_full_retrieval(selected.quality_identity.is_some(), retrieval)?;
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
    let loaded = load_authority_models(cx, options, selected.query_mode(None)?, pool)?;
    let expected_identity = rebuild_identity(
        &selected.identity,
        loaded.fast.as_ref(),
        options.migrate_model,
    )?;
    let expected_quality = match (&selected.quality_identity, &loaded.quality) {
        (Some(identity), Some(model)) => Some(rebuild_identity(
            identity,
            model.as_ref(),
            options.migrate_model,
        )?),
        (None, None) => None,
        _ => {
            return Err(bad(
                "the rebuild providers do not match the selected topology",
            ));
        }
    };
    let mut builder = NativeIndexBuilder::new(&work_path, new_generation(sequence)?, loaded.fast)?
        .with_batch_size(options.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .add_documents(documents);
    if let Some(quality) = loaded.quality {
        builder = builder
            .with_quality_embedder(quality)?
            .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)?;
    }
    let built = if expected_quality.is_some() {
        builder
            .build_progressive_generation(cx, source, options.max_image_bytes)
            .await?
    } else {
        builder
            .build_exact_generation(cx, selected.tier, source, options.max_image_bytes)
            .await?
    };
    if built.binding().frozen_identity() != &expected_identity
        || built
            .quality_binding()
            .map(FsviV2IdentityBinding::frozen_identity)
            != expected_quality.as_ref()
    {
        return Err(bad("the provider changed after rebuild identity admission"));
    }
    let PublishedGeneration { receipt, .. } = stage_and_publish(
        cx,
        &root,
        &selected.root,
        &receipt_path,
        &built,
        Some(&snapshot),
        PublicationOptions::unscoped(options.max_image_bytes),
    )?;
    emit(
        output,
        &serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "rebuilt", "publication": "committed",
            "serving": "activated", "retrieval": retrieval_name(built.retrieval_mode()),
            "receipt": receipt_path,
            "source": if options.retained_source { "retained_generation" } else { "input" },
            "replaces": selected.authority,
            "model_migrated": receipt.identity != selected.identity || receipt.quality_identity != selected.quality_identity,
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
            // A full update loads the two producers independently. Fast
            // admission must not consume the quality backend's options.
            let fast_options = models::QualityOptions::default();
            models::load_single(
                &context,
                options.models.as_deref(),
                tier,
                if tier == EmbedderTierTag::Fast {
                    &fast_options
                } else {
                    &options.quality
                },
                context.blocking_pool_handle(),
            )
        },
    ))
    .await
}

#[derive(Clone, Copy)]
struct AuthorityUpdateLimits {
    batch_size: usize,
    max_source_bytes: usize,
}

struct AuthorityUpdateSpec {
    work: PathBuf,
    receipt: PathBuf,
    source: CommitRange,
    limits: AuthorityUpdateLimits,
    max_image_bytes: usize,
}

struct PreparedAuthorityUpdate {
    spec: AuthorityUpdateSpec,
    sequence: u64,
    edits: mutations::Edits,
    document_count: usize,
}

fn prepare_authority_update<F>(
    cx: &Cx,
    selected: &Receipt,
    snapshot: &NativeGenerationSnapshot,
    mut spec: AuthorityUpdateSpec,
    read_edits: F,
) -> Result<PreparedAuthorityUpdate>
where
    F: FnOnce() -> Result<mutations::Edits>,
{
    checkpoint(cx)?;
    if snapshot
        .retained_snapshot()
        .selected_predecessor_of()
        .is_some()
        || snapshot.authority() != selected.authority()?
    {
        return Err(serve_refused("update_base_is_selected_predecessor"));
    }
    if spec.limits.batch_size == 0
        || spec.limits.batch_size > MAX_DOCUMENTS
        || spec.limits.max_source_bytes == 0
        || spec.max_image_bytes == 0
    {
        return Err(serve_refused("invalid_update_limits"));
    }
    if spec
        .source
        .high
        .checked_sub(spec.source.low)
        .and_then(|span| span.checked_add(1))
        .is_none()
    {
        return Err(serve_refused("invalid_source_checkpoint"));
    }
    snapshot.authority().next_sequence()?;
    validate_successor_source(&snapshot.metadata().commit_range, &spec.source)
        .map_err(|_| serve_refused("backward_source_checkpoint"))?;
    let sequence = selected
        .generation
        .sequence
        .checked_add(1)
        .ok_or_else(|| bad("the artifact generation sequence is exhausted"))?;
    spec.work = new_path(&spec.work)?;
    spec.receipt = new_path(&spec.receipt)?;
    separate_destinations(&[&selected.root, &spec.work, &spec.receipt])?;
    let documents = retained_rebuild_documents(cx, snapshot, spec.limits.max_source_bytes)?;
    let edits = read_edits()?;
    let document_count = mutations::validate_final_source(cx, &documents, &edits)?;
    if document_count == 0 {
        return Err(bad("authority updates require a nonempty final cohort"));
    }
    // The library prepares reuse from the same immutable image. Let these
    // preflight bodies go before loading a model or materializing that source.
    checkpoint(cx)?;
    Ok(PreparedAuthorityUpdate {
        spec,
        sequence,
        edits,
        document_count,
    })
}

async fn build_and_publish_update(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    snapshot: &NativeGenerationSnapshot,
    prepared: PreparedAuthorityUpdate,
    loaded: &Models,
    scope: Option<(&Filter, usize)>,
) -> Result<PublishedGeneration> {
    let PreparedAuthorityUpdate {
        spec,
        sequence,
        edits,
        document_count,
    } = prepared;
    admit_serving_models(selected, loaded)?;
    let transaction = if let Some(quality) = &loaded.quality {
        snapshot.begin_progressive_update(
            cx,
            &spec.work,
            new_generation(sequence)?,
            Arc::clone(&loaded.fast),
            Arc::clone(quality),
            spec.limits.max_source_bytes,
        )?
    } else {
        snapshot.begin_update(
            cx,
            &spec.work,
            new_generation(sequence)?,
            Arc::clone(&loaded.fast),
            spec.limits.max_source_bytes,
        )?
    };
    let mut transaction = transaction
        .with_batch_size(spec.limits.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?;
    for (id, document) in edits {
        checkpoint(cx)?;
        transaction = match document {
            Some(document) => transaction.upsert_document(document),
            None => transaction.delete_document(id),
        };
    }
    let built = if selected.quality_identity.is_some() {
        Box::pin(transaction.build_progressive_generation(cx, spec.source, spec.max_image_bytes))
            .await?
    } else {
        Box::pin(transaction.build_exact_generation(
            cx,
            selected.tier,
            spec.source,
            spec.max_image_bytes,
        ))
        .await?
    };
    if built.binding().frozen_identity() != &selected.identity
        || built
            .quality_binding()
            .map(FsviV2IdentityBinding::frozen_identity)
            != selected.quality_identity.as_ref()
    {
        return Err(bad("the provider changed after update identity admission"));
    }
    if built.metadata().total_documents != u64::try_from(document_count)? {
        return Err(bad(
            "the built update differs from its admitted source cohort",
        ));
    }
    stage_and_publish(
        cx,
        root,
        &selected.root,
        &spec.receipt,
        &built,
        Some(snapshot),
        PublicationOptions {
            max_image_bytes: spec.max_image_bytes,
            scope,
        },
    )
}

async fn update_generation_with_loader<F>(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    mut load_model: F,
) -> Result<()>
where
    F: FnMut(EmbedderTierTag) -> Result<Arc<dyn Embedder>>,
{
    checkpoint(cx)?;
    let selected = Receipt::read(
        options
            .from_receipt
            .as_deref()
            .ok_or_else(|| bad("missing predecessor receipt"))?,
    )?;
    options.quality.validate_usage(
        selected.tier == EmbedderTierTag::Quality || selected.quality_identity.is_some(),
    )?;
    let root = admit_root(&selected.root)?;
    // Admit the receipt's exact current selection before reading input or
    // loading a model, and retain that expected authority through publication.
    let snapshot = selected.open_on_root(cx, &root, options.max_image_bytes)?;
    let prepared = prepare_authority_update(
        cx,
        &selected,
        &snapshot,
        AuthorityUpdateSpec {
            work: options
                .work
                .clone()
                .ok_or_else(|| bad("missing work directory"))?,
            receipt: options.receipt.clone(),
            source: options.source_cut()?,
            limits: AuthorityUpdateLimits {
                batch_size: options.batch_size,
                max_source_bytes: options.max_source_bytes,
            },
            max_image_bytes: options.max_image_bytes,
        },
        || match &options.input {
            Some(path) => mutations::read_edits(&mut BufReader::new(File::open(path)?)),
            None => mutations::read_edits(&mut io::stdin().lock()),
        },
    )?;
    let document_count = prepared.document_count;
    let edit_count = prepared.edits.len();
    let receipt_path = prepared.spec.receipt.clone();
    checkpoint(cx)?;
    let loaded = Models {
        fast: load_model(selected.tier)?,
        quality: if selected.quality_identity.is_some() {
            Some(load_model(EmbedderTierTag::Quality)?)
        } else {
            None
        },
    };
    let published = Box::pin(build_and_publish_update(
        cx, &root, &selected, &snapshot, prepared, &loaded, None,
    ))
    .await?;
    let receipt = published.receipt;
    emit(
        output,
        &serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "updated", "publication": "committed",
            "serving": "activated", "retrieval": retrieval_name(published.snapshot.retrieval_mode()),
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

fn rollback(cx: &Cx, options: &Options, output: &mut impl Write) -> Result<()> {
    checkpoint(cx)?;
    let selected = Receipt::read(
        options
            .from_receipt
            .as_deref()
            .ok_or_else(|| bad("missing current receipt"))?,
    )?;
    let target = Receipt::read(
        options
            .target_receipt
            .as_deref()
            .ok_or_else(|| bad("missing rollback target receipt"))?,
    )?;
    if selected.root != target.root
        || selected.root_id != target.root_id
        || selected.profile != target.profile
        || selected.topology() != target.topology()
    {
        return Err(bad(
            "rollback receipts must select the same root, security profile, and tier",
        ));
    }
    let receipt_path = new_path(&options.receipt)?;
    separate_destinations(&[&selected.root, &receipt_path])?;
    let root = admit_root(&selected.root)?;
    let expected_current = selected.authority()?;
    expected_current.next_sequence()?;
    // The default route still requires a complete current reader. Recovery is
    // an explicit alternative, never a fallback from a failed ordinary open.
    let current = if options.allow_damaged_current {
        None
    } else {
        Some(selected.open_on_root(cx, &root, options.max_image_bytes)?)
    };
    checkpoint(cx)?;
    let target_budget = u64::try_from(options.max_image_bytes)?.min(target.image_bytes);
    let opened = match &current {
        Some(current) => root.open_predecessor_generation_snapshot_with_budget(
            current.retained_snapshot(),
            target.authority()?,
            target_budget,
        )?,
        None => {
            // The recovery opener charges the current control manifest as well
            // as the target's complete image set. Keep the target receipt's
            // narrower bound without allowing either sum to exceed the CLI cap.
            let recovery_budget = u64::try_from(options.max_image_bytes)?.min(
                target
                    .image_bytes
                    .saturating_add(expected_current.manifest_len),
            );
            root.open_predecessor_generation_for_recovery_with_budget(
                selected.root_id,
                expected_current,
                selected.generation,
                target.authority()?,
                recovery_budget,
            )?
        }
    };
    checkpoint(cx)?;
    let SnapshotOpenOutcomeV1::Opened(opened) = opened else {
        return Err(bad(&format!(
            "rollback predecessor admission refused: {opened:?}"
        )));
    };
    let predecessor = target.admit_opened(cx, opened, usize::try_from(target_budget)?)?;
    let writer = WriterIdentityV1::for_current_process()?;
    let object_id = new_generation(1)?.nonce;
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.native.authority-rollback.v1");
    hasher.update(writer.writer_id);
    hasher.update(writer.nonce);
    hasher.update(object_id);
    let fence = hasher.finalize().into();
    let candidate = match &current {
        Some(current) => predecessor.rollback_candidate(cx, current, object_id, fence)?,
        None => predecessor.rollback_selected_predecessor_candidate(cx, object_id, fence)?,
    };
    if candidate.owned_image_bytes() > u64::try_from(options.max_image_bytes)? {
        return Err(bad("rollback generation exceeds the encoded-image budget"));
    }
    let mut publisher =
        GenerationPublisherV1::new(&root, selected.root_id, writer.writer_id, PROFILE)?;
    let attempt = publisher.prepare_attempt(
        &candidate,
        // The target snapshot carries the actual current pair that authorized
        // predecessor selection, even when current components were unreadable.
        predecessor.retained_snapshot().authority_pair(),
        new_generation(1)?.nonce,
        new_generation(1)?.nonce,
    )?;
    let receipt = Receipt {
        schema: if target.quality_identity.is_some() {
            PROGRESSIVE_RECEIPT_SCHEMA
        } else {
            RECEIPT_SCHEMA
        }
        .to_owned(),
        root: selected.root.clone(),
        root_id: selected.root_id,
        profile: selected.profile.clone(),
        authority: candidate.authority().canonical_bytes().to_vec(),
        generation: target.generation,
        identity: target.identity.clone(),
        tier: target.tier,
        quality_identity: target.quality_identity.clone(),
        documents: target.documents,
        // The new activation manifest can be larger than the target's old
        // genesis manifest even though every component image is unchanged.
        image_bytes: candidate.owned_image_bytes(),
        attempt: Some(SavedAttempt::from_permit(&attempt, false)),
    };
    receipt.validate()?;
    encode(&receipt, MAX_RECEIPT_BYTES)?;
    checkpoint(cx)?;
    // The target's exact component objects were independently admitted above.
    // Only this new activation manifest is staged; no component is rewritten.
    root.stage_immutable_file(
        &activation_manifest_path_v1(candidate.authority().object_id)?,
        candidate.manifest_bytes(),
    )?;
    root.revalidate()?;
    save_receipt(&receipt, &receipt_path)?;
    publish_and_admit(
        cx,
        &mut publisher,
        &candidate,
        &attempt,
        &receipt,
        options.max_image_bytes,
    )?;
    emit(
        output,
        &serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "rolled_back", "publication": "committed",
            "serving": "activated", "retrieval": retrieval_name(predecessor.retrieval_mode()),
            "receipt": receipt_path, "replaces": selected.authority, "restores": target.authority,
            "current_admission": if options.allow_damaged_current {
                "control_manifest_only"
            } else {
                "complete_generation"
            },
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

fn load_authority_models(
    cx: &Cx,
    options: &Options,
    mode: QueryMode,
    pool: Option<BlockingPoolHandle>,
) -> Result<Models> {
    if mode == QueryMode::Lexical {
        return Err(serve_refused("lexical_requires_model_free_execution"));
    }
    if mode == QueryMode::Full {
        models::load(cx, options.models.as_deref(), true, &options.quality, pool)
    } else {
        let tier = if mode == QueryMode::Quality {
            EmbedderTierTag::Quality
        } else {
            EmbedderTierTag::Fast
        };
        Ok(Models {
            fast: models::load_single(cx, options.models.as_deref(), tier, &options.quality, pool)?,
            quality: None,
        })
    }
}

struct QueryPlan<'a> {
    text: &'a str,
    limit: usize,
    mode: QueryMode,
    stream: bool,
}

/// Execute against one retained generation and deliver each completed phase
/// before beginning the next. An output failure drops the lazy query immediately.
async fn query_admitted<F>(
    cx: &Cx,
    snapshot: &NativeGenerationSnapshot,
    scope: Option<&NativeScopedGenerationSnapshot>,
    loaded: &Models,
    plan: QueryPlan<'_>,
    mut deliver: F,
) -> Result<()>
where
    F: FnMut(&'static str, &[ScoredResult]) -> Result<()>,
{
    if plan.mode == QueryMode::Lexical {
        return Err(serve_refused("lexical_requires_model_free_execution"));
    }
    if plan.mode == QueryMode::Full {
        let quality = loaded
            .quality
            .as_deref()
            .ok_or_else(|| serve_refused("quality_provider_missing"))?;
        if plan.stream {
            let mut empty_initial_without_refinement = false;
            let mut query = match scope {
                Some(scope) => scope.search_progressive_with_quality(
                    cx,
                    loaded.fast.as_ref(),
                    quality,
                    plan.text,
                    plan.limit,
                )?,
                None => snapshot.search_progressive_with_quality(
                    cx,
                    loaded.fast.as_ref(),
                    quality,
                    plan.text,
                    plan.limit,
                )?,
            };
            while let Some(phase) = query.next_phase().await? {
                let (name, results) = match phase {
                    NativeSearchPhase::Initial { results, .. } => {
                        empty_initial_without_refinement = results.is_empty();
                        ("initial", results)
                    }
                    NativeSearchPhase::Refined { results, .. } => {
                        empty_initial_without_refinement = false;
                        ("refined", results)
                    }
                    NativeSearchPhase::RefinementFailed { error, .. }
                    | NativeSearchPhase::RerankFailed { error, .. } => return Err(error.into()),
                    NativeSearchPhase::Reranked { .. } => {
                        return Err(serve_refused("unexpected_reranking_phase"));
                    }
                };
                deliver(name, &results)?;
            }
            if empty_initial_without_refinement {
                // A zero-sized scope admits both identities, then the native
                // iterator completes after empty Initial without inference.
                // Finish the CLI's two-frame contract on a persistent stream.
                checkpoint(cx)?;
                deliver("refined", &[])?;
            }
        } else {
            let results = match scope {
                Some(scope) => {
                    Box::pin(scope.search_refined_text(
                        cx,
                        loaded.fast.as_ref(),
                        quality,
                        plan.text,
                        plan.limit,
                    ))
                    .await?
                }
                None => {
                    Box::pin(snapshot.search_refined_text(
                        cx,
                        loaded.fast.as_ref(),
                        quality,
                        plan.text,
                        plan.limit,
                    ))
                    .await?
                }
            };
            deliver("refined", &results)?;
        }
    } else {
        let (phase, results) = if plan.mode == QueryMode::Quality {
            let quality = loaded
                .quality
                .as_deref()
                .unwrap_or_else(|| loaded.fast.as_ref());
            (
                "quality",
                match scope {
                    Some(scope) => {
                        Box::pin(scope.search_quality_text(cx, quality, plan.text, plan.limit))
                            .await?
                    }
                    None => {
                        Box::pin(snapshot.search_quality_text(cx, quality, plan.text, plan.limit))
                            .await?
                    }
                },
            )
        } else {
            (
                "initial",
                match scope {
                    Some(scope) => {
                        Box::pin(scope.search_text(cx, loaded.fast.as_ref(), plan.text, plan.limit))
                            .await?
                    }
                    None => {
                        Box::pin(snapshot.search_text(
                            cx,
                            loaded.fast.as_ref(),
                            plan.text,
                            plan.limit,
                        ))
                        .await?
                    }
                },
            )
        };
        deliver(phase, &results)?;
    }
    Ok(())
}

async fn search(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let context = execution_context(cx, pool)?;
    Box::pin(search_with_loader(&context, options, output, |tier| {
        let fast_options = models::QualityOptions::default();
        models::load_single(
            &context,
            options.models.as_deref(),
            tier,
            if tier == EmbedderTierTag::Fast {
                &fast_options
            } else {
                &options.quality
            },
            context.blocking_pool_handle(),
        )
    }))
    .await
}

async fn search_with_loader<F>(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    mut load_model: F,
) -> Result<()>
where
    F: FnMut(EmbedderTierTag) -> Result<Arc<dyn Embedder>>,
{
    checkpoint(cx)?;
    let query = options
        .query
        .as_deref()
        .ok_or_else(|| bad("missing query"))?;
    validate_query(query)?;
    let receipt = Receipt::read(&options.receipt)?;
    let mode = receipt.query_mode(if options.lexical_only {
        Some(QueryMode::Lexical)
    } else {
        options.mode
    })?;
    options
        .quality
        .validate_usage(matches!(mode, QueryMode::Quality | QueryMode::Full))?;
    let snapshot = Arc::new(receipt.open(cx, options.max_image_bytes)?);
    // Source bounds and filter admission precede model loading. The scope owns
    // the same admitted generation that supplies every vector and lexical hit.
    let scope = prepare_scope(
        cx,
        &snapshot,
        options.filter.as_ref(),
        options.max_source_bytes,
    )?;
    let loaded = if mode == QueryMode::Lexical {
        AuthorityModels::Lexical
    } else {
        Models {
            fast: load_model(if mode == QueryMode::Quality {
                EmbedderTierTag::Quality
            } else {
                EmbedderTierTag::Fast
            })?,
            quality: if mode == QueryMode::Full {
                Some(load_model(EmbedderTierTag::Quality)?)
            } else {
                None
            },
        }
        .into()
    };
    let plan = QueryPlan {
        text: query,
        limit: options.limit,
        mode,
        stream: options.stream,
    };
    query_selected(
        cx,
        &snapshot,
        scope.as_ref(),
        &loaded,
        plan,
        |phase, results| {
            let mut response = serde_json::json!({
                "schema": SCHEMA, "ok": true, "event": "results",
                "retrieval": retrieval_name(snapshot.retrieval_mode()),
                "tier": mode.as_str(), "mode": mode.as_str(), "topology": receipt.topology_name(),
                "phase": phase, "generation": receipt.generation,
                "authority": snapshot.authority().canonical_bytes().as_slice(),
                "profile": receipt.profile, "results": results,
            });
            annotate_scope(
                &mut response,
                scope.as_ref().map(NativeScopedGenerationSnapshot::len),
            );
            annotate_execution(&mut response, mode);
            emit(output, &response)
        },
    )
    .await
}

fn fetch(cx: &Cx, options: &Options, output: &mut impl Write) -> Result<()> {
    checkpoint(cx)?;
    let window = options.fetch_document()?;
    if let Some(filter) = &options.filter {
        filter.validate()?;
    }
    let receipt = Receipt::read(&options.receipt)?;
    // Fresh-process admission must select this exact receipt. Never follow a
    // newer authority, load a model, or interpret the document ID as a path.
    let snapshot = receipt.open(cx, options.max_image_bytes)?;
    let document = snapshot.source_document(cx, window.doc_id(), options.max_source_bytes)?;
    let mut response = serde_json::json!({
        "schema": SCHEMA, "event": "terminal", "operation": "fetch",
        "status": "complete", "ok": true, "partial_results": false,
        "request": 1, "id": null, "seq": 0,
        "authority": snapshot.authority().canonical_bytes().as_slice(),
        "generation": snapshot.retained_snapshot().manifest().generation(),
        "tier": snapshot.tier(), "profile": receipt.profile,
        "retrieval": retrieval_name(snapshot.retrieval_mode()),
        "source": "retained_snapshot",
    });
    if options.filter.is_some() {
        response["scope"] = serde_json::json!({ "filtered": true });
    }
    // Borrow only this selected document's fields into the shared UTF-8 window
    // renderer. The source budget bounds its complete stored fields before
    // copying; the response bound includes metadata, authority and escaping.
    let bytes = source::render_documents(
        cx,
        std::slice::from_ref(&window),
        options.filter.as_ref(),
        None,
        |_| document.as_ref(),
        &response,
    )?;
    checkpoint(cx)?;
    output.write_all(&bytes)?;
    output.flush()?;
    Ok(())
}

fn prepare_scope(
    cx: &Cx,
    snapshot: &Arc<NativeGenerationSnapshot>,
    filter: Option<&Filter>,
    max_source_bytes: usize,
) -> Result<Option<NativeScopedGenerationSnapshot>> {
    checkpoint(cx)?;
    let Some(filter) = filter else {
        // Keep the unfiltered path free of source scans or allocations.
        return Ok(None);
    };
    filter.validate()?;
    Ok(Some(snapshot.scope(cx, max_source_bytes, |document| {
        Ok(filter.matches(document))
    })?))
}

fn annotate_scope(response: &mut serde_json::Value, eligible_documents: Option<usize>) {
    if let Some(eligible_documents) = eligible_documents {
        response["scope"] = serde_json::json!({
            "filtered": true, "eligible_documents": eligible_documents,
        });
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthoritySearchRequest {
    id: Option<String>,
    query: String,
    limit: Option<usize>,
    mode: Option<QueryMode>,
    #[serde(default)]
    stream: bool,
    filter: Option<Filter>,
}

/// Explicit execution policy. A missing or failed model never selects lexical
/// execution; that choice must come from startup options or a query request.
enum AuthorityModels {
    Semantic(Models),
    Lexical,
}

impl From<Models> for AuthorityModels {
    fn from(models: Models) -> Self {
        Self::Semantic(models)
    }
}

impl AuthorityModels {
    const fn is_lexical(&self) -> bool {
        matches!(self, Self::Lexical)
    }

    fn models(&self) -> Result<&Models> {
        match self {
            Self::Semantic(models) => Ok(models),
            Self::Lexical => Err(serve_refused("semantic_search_disabled")),
        }
    }

    fn admit(&self, receipt: &Receipt) -> Result<()> {
        match self {
            Self::Semantic(models) => admit_serving_models(receipt, models),
            Self::Lexical => Ok(()),
        }
    }

    fn query_mode(&self, receipt: &Receipt, requested: Option<QueryMode>) -> Result<QueryMode> {
        let mode = if self.is_lexical() {
            requested.unwrap_or(QueryMode::Lexical)
        } else {
            receipt.query_mode(requested)?
        };
        if mode != QueryMode::Lexical && self.is_lexical() {
            return Err(serve_refused("semantic_search_disabled"));
        }
        receipt.query_mode(Some(mode))
    }
}

async fn query_selected<F>(
    cx: &Cx,
    snapshot: &NativeGenerationSnapshot,
    scope: Option<&NativeScopedGenerationSnapshot>,
    models: &AuthorityModels,
    plan: QueryPlan<'_>,
    mut deliver: F,
) -> Result<()>
where
    F: FnMut(&'static str, &[frankensearch::ScoredResult]) -> Result<()>,
{
    if plan.mode == QueryMode::Lexical {
        let results = match scope {
            Some(scope) => scope.search_lexical_text(cx, plan.text, plan.limit).await?,
            None => {
                snapshot
                    .search_lexical_text(cx, plan.text, plan.limit)
                    .await?
            }
        };
        checkpoint(cx)?;
        deliver("lexical", &results)
    } else {
        query_admitted(cx, snapshot, scope, models.models()?, plan, deliver).await
    }
}

fn annotate_execution(response: &mut serde_json::Value, mode: QueryMode) {
    response["mode"] = serde_json::json!(mode.as_str());
    // Every authority reader still admits its complete component closure.
    // Query execution is separate from whether its stored bytes were verified.
    response["semantic_components_verified"] = serde_json::json!(true);
    if response["event"] == "results" && mode == QueryMode::Lexical {
        response["semantic_used"] = serde_json::json!(false);
    }
    if mode == QueryMode::Lexical {
        response["phase"] = serde_json::json!("lexical");
        response["retrieval"] = serde_json::json!("lexical");
    }
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
#[serde(deny_unknown_fields)]
struct AuthorityUpdateRequest {
    op: String,
    id: Option<String>,
    expected_authority: Vec<u8>,
    work_dir: PathBuf,
    new_receipt: PathBuf,
    source_low: u64,
    source_high: u64,
    changes: Vec<mutations::Mutation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorityFetchRequest {
    op: String,
    id: Option<String>,
    expected_authority: Vec<u8>,
    documents: Vec<source::DocumentRequest>,
    filter: Option<Filter>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AuthorityRequest {
    Search(AuthoritySearchRequest),
    Activate(AuthorityActivateRequest),
    Update(AuthorityUpdateRequest),
    Fetch(AuthorityFetchRequest),
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

fn admit_serving_model(
    expected: &FrozenEmbeddingIdentityBundleV1,
    model: &dyn Embedder,
) -> Result<()> {
    let identity = model
        .identity()
        .map_err(|_| serve_refused("producer_identity_unavailable"))?;
    identity
        .validate()
        .map_err(|_| serve_refused("producer_identity_invalid"))?;
    // Queries carry in-memory vectors. Persisted F16/F32 encoding may change
    // between admitted generations without changing their query producer.
    let expected = &expected.identity;
    if identity.space != expected.space
        || identity.producer != expected.producer
        || identity.input != expected.input
        || usize::try_from(identity.space.dimension).ok() != Some(model.dimension())
    {
        return Err(serve_refused("producer_identity_mismatch"));
    }
    Ok(())
}

fn admit_serving_models(receipt: &Receipt, loaded: &Models) -> Result<()> {
    admit_serving_model(&receipt.identity, loaded.fast.as_ref())?;
    match (&receipt.quality_identity, &loaded.quality) {
        (Some(identity), Some(model)) => admit_serving_model(identity, model.as_ref()),
        (None, None) => Ok(()),
        _ => Err(serve_refused("session_provider_topology_mismatch")),
    }
}

/// One admitted search/model selection. Request execution owns this whole
/// value; no selected component or model is rediscovered between query stages.
struct AuthoritySession {
    root: QualifiedGenerationRoot,
    receipt: Receipt,
    snapshot: Arc<NativeGenerationSnapshot>,
    models: AuthorityModels,
    default_limit: usize,
    max_image_bytes: usize,
    max_source_bytes: usize,
    fixed_filter: Option<Filter>,
    scope: Option<NativeScopedGenerationSnapshot>,
    allow_activation: bool,
    update_limits: Option<AuthorityUpdateLimits>,
}

impl AuthoritySession {
    fn new(
        cx: &Cx,
        root: QualifiedGenerationRoot,
        receipt: Receipt,
        snapshot: impl Into<Arc<NativeGenerationSnapshot>>,
        models: impl Into<AuthorityModels>,
        limits: (usize, usize),
        allow_activation: bool,
    ) -> Result<Self> {
        checkpoint(cx)?;
        let snapshot = snapshot.into();
        let models = models.into();
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
            || retained.selected_predecessor_of().is_some()
            || retained.profile().as_str() != receipt.profile
            || snapshot.authority() != receipt.authority()?
            || snapshot.topology() != receipt.topology()
            || snapshot.metadata().total_documents != receipt.documents
            || snapshot.metadata().embedders.get(tier_name) != Some(&receipt.identity)
            || receipt.quality_identity.as_ref().is_some_and(|quality| {
                snapshot.metadata().embedders.get("quality") != Some(quality)
            })
            || !same_security_object(root.witness(), retained.root_witness())
            || !same_security_object(anchors.lock(), old_anchors.lock())
            || !same_security_object(anchors.authority(), old_anchors.authority())
        {
            return Err(serve_refused("session_selection_mismatch"));
        }
        models.admit(&receipt)?;
        checkpoint(cx)?;
        Ok(Self {
            root,
            receipt,
            snapshot,
            models,
            default_limit: limits.0,
            max_image_bytes: limits.1,
            max_source_bytes: MAX_INPUT_BYTES,
            fixed_filter: None,
            scope: None,
            allow_activation,
            update_limits: None,
        })
    }

    fn with_updates(mut self, batch_size: usize, max_source_bytes: usize) -> Result<Self> {
        self.models.models()?;
        if batch_size == 0 || batch_size > MAX_DOCUMENTS || max_source_bytes == 0 {
            return Err(serve_refused("invalid_update_limits"));
        }
        self.update_limits = Some(AuthorityUpdateLimits {
            batch_size,
            max_source_bytes,
        });
        Ok(self)
    }

    #[cfg(test)]
    fn with_filter(
        mut self,
        cx: &Cx,
        filter: Option<Filter>,
        max_source_bytes: usize,
    ) -> Result<Self> {
        if max_source_bytes == 0 {
            return Err(serve_refused("invalid_source_budget"));
        }
        let scope = prepare_scope(cx, &self.snapshot, filter.as_ref(), max_source_bytes)?;
        checkpoint(cx)?;
        self.scope = scope;
        self.fixed_filter = filter;
        self.max_source_bytes = max_source_bytes;
        Ok(self)
    }

    fn response(
        &self,
        event: &'static str,
        ok: bool,
        request: Option<u64>,
        id: Option<&str>,
    ) -> serde_json::Value {
        let mut response = serde_json::json!({
            "schema": SCHEMA, "event": event, "ok": ok, "request": request, "id": id,
            "authority": self.snapshot.authority().canonical_bytes().as_slice(),
            "generation": self.receipt.generation, "profile": self.receipt.profile,
            "tier": self.snapshot.tier(),
            "topology": self.receipt.topology_name(),
            "retrieval": retrieval_name(self.snapshot.retrieval_mode()),
        });
        annotate_scope(
            &mut response,
            self.scope.as_ref().map(NativeScopedGenerationSnapshot::len),
        );
        if self.models.is_lexical() {
            annotate_execution(&mut response, QueryMode::Lexical);
        }
        response
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

    fn update_failure(
        &self,
        ordinal: u64,
        id: Option<&str>,
        error: &(dyn Error + 'static),
    ) -> serde_json::Value {
        let mut response = self.failure(ordinal, id, error, "update_refused");
        response["publication"] = serde_json::json!("not_committed");
        response["serving"] = serde_json::json!("not_activated");
        if let Some(failure) = publication_failure(error) {
            // This is the private publisher outcome, not provider/request text.
            // Keep the retained serving authority separate from the candidate.
            response["code"] = serde_json::json!("update_publication_failed");
            response["publication"] = failure["publication"].clone();
            response["serving"] = failure["serving"].clone();
            response["candidate_authority"] = failure["authority"].clone();
            if let Some(receipt) = failure.get("receipt") {
                response["receipt"] = receipt.clone();
            }
            if let Some(attempt) = failure.get("attempt") {
                response["attempt"] = attempt.clone();
            }
        }
        response
    }

    async fn query(
        &self,
        cx: &Cx,
        request: &AuthoritySearchRequest,
        ordinal: u64,
        output: &mut impl Write,
    ) -> Result<()> {
        checkpoint(cx)?;
        super::serve::validate_id(request.id.as_deref())
            .map_err(|_| serve_refused("invalid_request_id"))?;
        validate_query(&request.query).map_err(|_| serve_refused("invalid_query"))?;
        let limit = request.limit.unwrap_or(self.default_limit);
        if limit == 0 || limit > 1_000 {
            return Err(serve_refused("invalid_limit"));
        }
        let mode = self.models.query_mode(&self.receipt, request.mode)?;
        if mode != QueryMode::Lexical {
            self.models.admit(&self.receipt)?;
        }
        if let Some(filter) = &request.filter {
            filter
                .validate()
                .map_err(|_| serve_refused("invalid_filter"))?;
        }
        // Narrow the owner's frozen membership; never substitute a request
        // scope for it. Without a fixed filter, only filtered requests export
        // bounded source. No request can mutate the session's retained scope.
        let narrowed = match (self.scope.as_ref(), request.filter.as_ref()) {
            (Some(scope), Some(filter)) => {
                Some(scope.narrow(cx, |document| Ok(filter.matches(document)))?)
            }
            (None, Some(filter)) => {
                prepare_scope(cx, &self.snapshot, Some(filter), self.max_source_bytes)?
            }
            _ => None,
        };
        let scope = narrowed.as_ref().or(self.scope.as_ref());
        // NativeGenerationSnapshot independently checks the producer before
        // inference and the actual bound output afterward, including zero hits.
        query_selected(
            cx,
            &self.snapshot,
            scope,
            &self.models,
            QueryPlan {
                text: &request.query,
                limit,
                mode,
                stream: request.stream,
            },
            |phase, results| {
                let mut response =
                    self.response("results", true, Some(ordinal), request.id.as_deref());
                response["results"] = serde_json::json!(results);
                response["mode"] = serde_json::json!(mode.as_str());
                response["phase"] = serde_json::json!(phase);
                annotate_execution(&mut response, mode);
                annotate_scope(
                    &mut response,
                    scope.map(NativeScopedGenerationSnapshot::len),
                );
                emit(output, &response).map_err(|_| serve_refused("result_delivery_failed"))
            },
        )
        .await?;
        checkpoint(cx)
    }

    fn fetch(&self, cx: &Cx, request: &AuthorityFetchRequest, ordinal: u64) -> Result<Vec<u8>> {
        checkpoint(cx)?;
        super::serve::validate_id(request.id.as_deref())
            .map_err(|_| serve_refused("invalid_request_id"))?;
        if request.op != "fetch" {
            return Err(serve_refused("invalid_fetch_request"));
        }
        let expected = AuthorityRefV1::from_canonical_bytes(&request.expected_authority)
            .map_err(|_| serve_refused("invalid_expected_authority"))?;
        // Artifact identity alone is insufficient: rollback can select the same
        // component images under a different authority. Fence every source read
        // before looking up a document or evaluating its source predicate.
        if expected != self.snapshot.authority() {
            return Err(serve_refused("stale_expected_authority"));
        }
        source::validate_windows(&request.documents)
            .map_err(|_| serve_refused("invalid_fetch_windows"))?;
        if let Some(filter) = &request.filter {
            filter
                .validate()
                .map_err(|_| serve_refused("invalid_filter"))?;
        }
        let mut response = self.response("terminal", true, Some(ordinal), request.id.as_deref());
        response["operation"] = serde_json::json!("fetch");
        response["status"] = serde_json::json!("complete");
        response["partial_results"] = serde_json::json!(false);
        response["seq"] = serde_json::json!(0);
        response["source"] = serde_json::json!("retained_snapshot");
        if self.scope.is_some() || request.filter.is_some() {
            // A selective source lookup does not enumerate the entire request
            // scope, so do not label a window count as global eligibility.
            response["scope"] = serde_json::json!({ "filtered": true });
        }
        if let Some(scope) = &self.scope {
            // The fixed membership and complete bounded source inventory were
            // prepared with this exact retained authority. Excluded IDs remain
            // unavailable without reading their fields or checking their offsets.
            source::render_documents(
                cx,
                &request.documents,
                request.filter.as_ref(),
                None,
                |id| scope.document(id),
                &response,
            )
        } else {
            // Read only requested live IDs, once per distinct document. Preserve
            // window order and duplicates in the borrowed renderer, not by
            // duplicating complete source allocations or charging them twice.
            let ids = request
                .documents
                .iter()
                .map(source::DocumentRequest::doc_id)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let documents =
                self.snapshot
                    .source_documents_by_ids(cx, &ids, self.max_source_bytes)?;
            source::render_documents(
                cx,
                &request.documents,
                request.filter.as_ref(),
                None,
                |id| {
                    documents
                        .iter()
                        .flatten()
                        .find(|document| document.id == id)
                },
                &response,
            )
        }
    }

    async fn update(
        &mut self,
        cx: &Cx,
        request: AuthorityUpdateRequest,
    ) -> Result<serde_json::Value> {
        checkpoint(cx)?;
        let limits = self
            .update_limits
            .ok_or_else(|| serve_refused("updates_disabled"))?;
        super::serve::validate_id(request.id.as_deref())
            .map_err(|_| serve_refused("invalid_request_id"))?;
        if request.op != "update" {
            return Err(serve_refused("invalid_update_request"));
        }
        for path in [&request.work_dir, &request.new_receipt] {
            if !path.is_absolute()
                || path
                    .to_str()
                    .is_none_or(|text| text.len() > 4096 || text.contains('\0'))
            {
                return Err(serve_refused("invalid_update_request"));
            }
        }
        let expected = AuthorityRefV1::from_canonical_bytes(&request.expected_authority)
            .map_err(|_| serve_refused("invalid_expected_authority"))?;
        if expected != self.snapshot.authority() {
            return Err(serve_refused("stale_expected_authority"));
        }
        if request.changes.len() > MAX_AUTHORITY_MUTATIONS {
            return Err(serve_refused("update_mutation_limit"));
        }
        let edits = mutations::from_mutations(request.changes)
            .map_err(|_| serve_refused("invalid_update_mutations"))?;
        // A warm reader can intentionally outlive the head. Confirm that it is
        // still the exact writable base without adopting any refreshed pair.
        // The opener also refuses an unresolved previous publication attempt.
        let current = self
            .receipt
            .open_on_root(cx, &self.root, self.max_image_bytes)
            .map_err(|error| {
                if error
                    .downcast_ref::<SearchError>()
                    .is_some_and(|error| matches!(error, SearchError::Cancelled { .. }))
                {
                    error
                } else {
                    serve_refused("update_base_not_current")
                }
            })?;
        if current
            .retained_snapshot()
            .selected_predecessor_of()
            .is_some()
            || current.retained_snapshot().authority_pair()
                != self.snapshot.retained_snapshot().authority_pair()
        {
            return Err(serve_refused("update_base_changed"));
        }
        drop(current);
        let prepared = prepare_authority_update(
            cx,
            &self.receipt,
            &self.snapshot,
            AuthorityUpdateSpec {
                work: request.work_dir,
                receipt: request.new_receipt,
                source: CommitRange {
                    low: request.source_low,
                    high: request.source_high,
                },
                limits,
                max_image_bytes: self.max_image_bytes,
            },
            || Ok(edits),
        )?;
        let document_count = prepared.document_count;
        let edit_count = prepared.edits.len();
        let receipt_path = prepared.spec.receipt.clone();
        let published = Box::pin(build_and_publish_update(
            cx,
            &self.root,
            &self.receipt,
            &self.snapshot,
            prepared,
            self.models.models()?,
            self.fixed_filter
                .as_ref()
                .map(|filter| (filter, self.max_source_bytes)),
        ))
        .await?;
        // This is the exact admitted result returned by publication. Do not
        // reopen the head, revalidate mutable paths, or checkpoint after commit.
        // The request loop owns all three fields until terminal acknowledgement.
        self.receipt = published.receipt;
        self.snapshot = published.snapshot;
        self.scope = published.scope;
        let mut response = self.response("updated", true, None, request.id.as_deref());
        response["publication"] = serde_json::json!("committed");
        response["serving"] = serde_json::json!("activated");
        response["previous_authority"] = serde_json::json!(expected.canonical_bytes().as_slice());
        response["receipt"] = serde_json::json!(receipt_path);
        response["selection"] = serde_json::json!(self.receipt);
        response["documents"] = serde_json::json!(document_count);
        response["edited_ids"] = serde_json::json!(edit_count);
        Ok(response)
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
            || candidate.topology() != self.receipt.topology()
        {
            return Err(serve_refused("activation_root_or_tier_mismatch"));
        }
        if candidate.authority()?.sequence <= self.snapshot.authority().sequence {
            return Err(serve_refused("activation_not_newer"));
        }
        let same_quality = match (&candidate.quality_identity, &self.receipt.quality_identity) {
            (Some(first), Some(second)) => same_query_contract(first, second),
            (None, None) => true,
            _ => false,
        };
        if !self.models.is_lexical()
            && (!same_query_contract(&candidate.identity, &self.receipt.identity) || !same_quality)
        {
            return Err(serve_refused("activation_producer_mismatch"));
        }
        self.models.admit(&candidate)?;
        checkpoint(cx)?;
        // The receipt selects an authority on the original qualified physical
        // root. Never reopen the candidate's root pathname or adopt a copy.
        let snapshot = Arc::new(
            candidate
                .open_on_root(cx, &self.root, self.max_image_bytes)
                .map_err(|_| serve_refused("activation_admission_failed"))?,
        );
        self.models.admit(&candidate)?;
        // Re-evaluate the fixed policy against the new retained source before
        // installing any part of the selection. A budget or source failure
        // leaves both the old reader and its old membership intact.
        let scope = prepare_scope(
            cx,
            &snapshot,
            self.fixed_filter.as_ref(),
            self.max_source_bytes,
        );
        checkpoint(cx)?;
        let scope = scope.map_err(|_| serve_refused("activation_scope_failed"))?;
        checkpoint(cx)?;
        // All fallible work is complete. The request loop owns both fields and
        // cannot serve a query between these assignments.
        self.snapshot = snapshot;
        self.scope = scope;
        self.receipt = candidate;
        Ok(())
    }
}

fn emit_authority_response(output: &mut impl Write, response: &serde_json::Value) -> Result<()> {
    emit(output, response).map_err(|error| {
        if response.get("publication").is_none() {
            return error;
        }
        // A broken output channel ends the session. Preserve the known effect
        // instead of replacing committed/unknown with an ordinary I/O failure.
        let mut payload = response.clone();
        payload["ok"] = serde_json::json!(false);
        payload["event"] = serde_json::json!("publication");
        payload["reason"] = serde_json::json!("result_delivery_failed");
        Box::new(PublicationFailure(payload)) as Box<dyn Error + Send + Sync>
    })
}

async fn serve_session(
    cx: &Cx,
    session: &mut AuthoritySession,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    checkpoint(cx)?;
    session.models.admit(&session.receipt)?;
    let mut ready = session.response("ready", true, None, None);
    ready["documents"] = serde_json::json!(session.receipt.documents);
    ready["activation_enabled"] = serde_json::json!(session.allow_activation);
    ready["updates_enabled"] = serde_json::json!(session.update_limits.is_some());
    ready["fetch_enabled"] = serde_json::json!(true);
    ready["query_modes"] = if session.models.is_lexical() {
        serde_json::json!(["lexical"])
    } else if session.receipt.quality_identity.is_some() {
        serde_json::json!(["fast", "quality", "full", "lexical"])
    } else {
        serde_json::json!([session.receipt.query_mode(None)?.as_str(), "lexical"])
    };
    ready["max_fetch_documents"] = serde_json::json!(source::MAX_DOCUMENTS);
    ready["max_fetch_content_bytes"] = serde_json::json!(source::MAX_CONTENT_BYTES);
    ready["max_fetch_response_bytes"] = serde_json::json!(source::MAX_RESPONSE_BYTES);
    if let Some(limits) = session.update_limits {
        ready["max_mutations"] = serde_json::json!(MAX_AUTHORITY_MUTATIONS);
        ready["update_batch_size"] = serde_json::json!(limits.batch_size);
        ready["max_source_bytes"] = serde_json::json!(limits.max_source_bytes);
    }
    ready["default_limit"] = serde_json::json!(session.default_limit);
    ready["max_source_bytes"] = serde_json::json!(session.max_source_bytes);
    ready["max_request_bytes"] = serde_json::json!(super::serve::MAX_REQUEST_BYTES);
    ready["default_mode"] =
        serde_json::json!(session.models.query_mode(&session.receipt, None)?.as_str());
    ready["progressive_available"] = serde_json::json!(
        !session.models.is_lexical() && session.receipt.quality_identity.is_some()
    );
    emit(output, &ready)?;
    let mut record = Vec::new();
    let mut ordinal = 0_u64;
    while super::serve::read_request(cx, input, &mut record)? {
        ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| serve_refused("request_ordinal_exhausted"))?;
        let Ok(request) = serde_json::from_slice::<AuthorityRequest>(record.trim_ascii()) else {
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
        };
        let (id, result, is_update) = match request {
            AuthorityRequest::Search(request) => {
                let id = request
                    .id
                    .clone()
                    .filter(|id| super::serve::validate_id(Some(id)).is_ok());
                let result = session
                    .query(cx, &request, ordinal, output)
                    .await
                    .map(|()| None);
                (id, result, false)
            }
            AuthorityRequest::Activate(request) => {
                let id = request
                    .id
                    .clone()
                    .filter(|id| super::serve::validate_id(Some(id)).is_ok());
                let previous = session.snapshot.authority();
                let result = session.activate(cx, &request).map(|()| {
                    let mut response =
                        session.response("activated", true, Some(ordinal), id.as_deref());
                    response["previous_authority"] =
                        serde_json::json!(previous.canonical_bytes().as_slice());
                    response["documents"] = serde_json::json!(session.receipt.documents);
                    Some(response)
                });
                (id, result, false)
            }
            AuthorityRequest::Update(request) => {
                let id = request
                    .id
                    .clone()
                    .filter(|id| super::serve::validate_id(Some(id)).is_ok());
                let result = Box::pin(session.update(cx, request))
                    .await
                    .map(|mut response| {
                        response["request"] = serde_json::json!(ordinal);
                        Some(response)
                    });
                (id, result, true)
            }
            AuthorityRequest::Fetch(request) => {
                let id = request
                    .id
                    .as_deref()
                    .filter(|id| super::serve::validate_id(Some(id)).is_ok());
                let result = session.fetch(cx, &request, ordinal);
                checkpoint(cx)?;
                match result {
                    Ok(bytes) => {
                        // The complete authority envelope and borrowed source
                        // windows already fit the encoded response budget. A
                        // partial write ends serving without a second frame.
                        output.write_all(&bytes)?;
                        output.flush()?;
                    }
                    Err(error) => {
                        if error
                            .downcast_ref::<SearchError>()
                            .is_some_and(|error| matches!(error, SearchError::Cancelled { .. }))
                        {
                            return Err(error);
                        }
                        let mut response =
                            session.failure(ordinal, id, error.as_ref(), "fetch_refused");
                        response["operation"] = serde_json::json!("fetch");
                        response["partial_results"] = serde_json::json!(false);
                        emit(output, &response)?;
                    }
                }
                continue;
            }
        };
        // Updates may already have committed, or returned a typed uncertain
        // publication outcome. Neither can be erased by a late cancellation.
        if !is_update {
            checkpoint(cx)?;
        }
        match result {
            Ok(Some(response)) => emit_authority_response(output, &response)?,
            Ok(None) => {}
            Err(error) => {
                if error
                    .downcast_ref::<SearchError>()
                    .is_some_and(|error| matches!(error, SearchError::Cancelled { .. }))
                    || error
                        .downcast_ref::<AuthorityServeRefusal>()
                        .is_some_and(|error| error.0 == "result_delivery_failed")
                {
                    return Err(error);
                }
                let response = if is_update {
                    session.update_failure(ordinal, id.as_deref(), error.as_ref())
                } else {
                    session.failure(
                        ordinal,
                        id.as_deref(),
                        error.as_ref(),
                        "query_or_activation_refused",
                    )
                };
                emit_authority_response(output, &response)?;
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
    options.quality.validate_usage(
        receipt.tier == EmbedderTierTag::Quality || receipt.quality_identity.is_some(),
    )?;
    let root = admit_root(&receipt.root)?;
    let snapshot = Arc::new(receipt.open_on_root(cx, &root, options.max_image_bytes)?);
    let scope = prepare_scope(
        cx,
        &snapshot,
        options.filter.as_ref(),
        options.max_source_bytes,
    )?;
    // Complete image and source-scope admission precede model loading. Both
    // producers and retained images remain warm until explicit activation/exit.
    let loaded = if options.lexical_only {
        AuthorityModels::Lexical
    } else {
        load_authority_models(cx, options, receipt.query_mode(None)?, pool)?.into()
    };
    let mut session = AuthoritySession::new(
        cx,
        root,
        receipt,
        snapshot,
        loaded,
        (options.limit, options.max_image_bytes),
        options.allow_activation,
    )?;
    if options.allow_updates {
        session = session.with_updates(options.batch_size, options.max_source_bytes)?;
    }
    session.scope = scope;
    session.fixed_filter = options.filter.clone();
    session.max_source_bytes = options.max_source_bytes;
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
            Command::Fetch => fetch(&cx, &options, &mut output),
            Command::Serve => Box::pin(serve(&cx, &options, &mut output, pool)).await,
            Command::Recover => recover(&cx, &options, &mut output),
            Command::Rollback => rollback(&cx, &options, &mut output),
        }
    })
}

#[cfg(test)]
#[path = "generation_tests.rs"]
mod tests;
