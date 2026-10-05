//! Executable entry point for the identity-bound native hybrid pipeline.
//! Selection is an explicit, caller-owned receipt, never directory discovery.

#![forbid(unsafe_code)]

use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use asupersync::runtime::RuntimeBuilder;
use frankensearch::native_ann::builder::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltHybridIndex, NativeIndexBuilder,
};
use frankensearch::{Cx, Embedder, IndexableDocument};
use frankensearch_core::generation::{ArtifactGenerationIdentityV1, GenerationComponentReceiptV1};
use frankensearch_embed::{DetectOptions, EmbedderStack};
use frankensearch_index::native_hnsw::HnswParams;
use serde::{Deserialize, Serialize};

#[path = "native_cli/serve.rs"]
mod serve;

#[path = "native_cli/query.rs"]
mod query;

#[path = "native_cli/update.rs"]
mod update;

#[path = "native_cli/filter.rs"]
mod filter;

#[path = "native_cli/tests.rs"]
#[cfg(test)]
mod tests;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const SCHEMA: &str = "frankensearch.native.cli.v1";
const SELECTION_SCHEMA: &str = "frankensearch.native.selection.v1";
const MAX_RECORD_BYTES: usize = 16 * 1024 * 1024;
const MAX_INPUT_BYTES: usize = 256 * 1024 * 1024;
const MAX_DOCUMENTS: usize = 100_000;
const MAX_RECEIPT_BYTES: usize = 64 * 1024;
const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const HELP: &str = "frankensearch-native: native HNSW + FSVI v2 + Quill\n\n\
  index  --index-dir NEW_DIR --receipt NEW_JSON [--input JSONL]\n\
         [--model-dir DIR] [--fast-only] [--exact] [--batch-size N]\n\
  search --receipt JSON --query TEXT [--model-dir DIR]\n\
         [--mode full|fast|quality] [--limit N] [--stream] [--filter JSON]\n\
         [--timeout-ms N]\n\
  serve  --receipt JSON [--model-dir DIR] [--mode full|fast|quality] [--limit N]\n\
         [--allow-activation] [--filter JSON] [--timeout-ms N]\n\n\
  update --receipt OLD_JSON --index-dir NEW_DIR --new-receipt NEW_JSON\n\
         [--input CHANGES_JSONL] [--model-dir DIR] [--batch-size N]\n\n\
Input: one {\"id\":\"...\",\"content\":\"...\",\"title\":null,\"metadata\":{}} per line.\n\
Updates use op=upsert with those fields, or {\"op\":\"delete\",\"id\":\"...\"}.\n\
The last update for an ID wins; every input record must be valid.\n\
Omit --input to read stdin. Content is passed to the models without hidden\n\
canonicalization; prepare/chunk documents explicitly. IDs must be unique.\n\
Models are local-only; no downloads or hash fallback. Both tiers are required\n\
unless index --fast-only is explicit. Native HNSW is used unless --exact.\n\
Index creates a new directory and a new trusted selection receipt. Keep that\n\
receipt outside the immutable index; search never discovers or repairs one.\n\
Stdout is JSON. This command does not change fsfs stores or CURRENT pointers.\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Index,
    Search,
    Serve,
    Update,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Full,
    Fast,
    Quality,
}

impl Mode {
    fn parse(text: &str) -> Result<Self> {
        match text {
            "full" => Ok(Self::Full),
            "fast" => Ok(Self::Fast),
            "quality" => Ok(Self::Quality),
            _ => Err(bad("mode must be full, fast, or quality")),
        }
    }
}

#[derive(Debug)]
struct Options {
    command: Command,
    receipt: PathBuf,
    new_receipt: Option<PathBuf>,
    directory: Option<PathBuf>,
    input: Option<PathBuf>,
    models: Option<PathBuf>,
    query: Option<String>,
    mode: Mode,
    limit: usize,
    batch_size: usize,
    fast_only: bool,
    exact: bool,
    stream: bool,
    activation: serve::ActivationPermission,
    filter: Option<filter::Filter>,
    timeout_ms: Option<u64>,
}

impl Options {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>> {
        let mut args = args.into_iter();
        let Some(command) = args.next() else {
            return Ok(None);
        };
        if matches!(command.as_str(), "--help" | "-h" | "help") {
            return Ok(None);
        }
        let command = match command.as_str() {
            "index" => Command::Index,
            "search" => Command::Search,
            "serve" => Command::Serve,
            "update" => Command::Update,
            _ => return Err(bad("expected index, search, serve, or update; use --help")),
        };
        let mut options = Self {
            command,
            receipt: PathBuf::new(),
            new_receipt: None,
            directory: None,
            input: None,
            models: None,
            query: None,
            mode: Mode::Full,
            limit: 10,
            batch_size: 16,
            fast_only: false,
            exact: false,
            stream: false,
            activation: serve::ActivationPermission::Disabled,
            filter: None,
            timeout_ms: None,
        };
        let mut seen = BTreeSet::new();
        while let Some(flag) = args.next() {
            if flag == "--help" || flag == "-h" {
                return Ok(None);
            }
            if !seen.insert(flag.clone()) {
                return Err(bad("an option may only be supplied once"));
            }
            match flag.as_str() {
                "--fast-only" if command == Command::Index => options.fast_only = true,
                "--exact" if command == Command::Index => options.exact = true,
                "--stream" if command == Command::Search => options.stream = true,
                "--allow-activation" if command == Command::Serve => {
                    options.activation = serve::ActivationPermission::Enabled;
                }
                "--receipt" => options.receipt = PathBuf::from(value(&mut args)?),
                "--new-receipt" if command == Command::Update => {
                    options.new_receipt = Some(PathBuf::from(value(&mut args)?));
                }
                "--model-dir" => options.models = Some(PathBuf::from(value(&mut args)?)),
                "--index-dir" if matches!(command, Command::Index | Command::Update) => {
                    options.directory = Some(PathBuf::from(value(&mut args)?));
                }
                "--input" if matches!(command, Command::Index | Command::Update) => {
                    options.input = Some(PathBuf::from(value(&mut args)?));
                }
                "--batch-size" if matches!(command, Command::Index | Command::Update) => {
                    options.batch_size = positive(&value(&mut args)?, 256)?;
                }
                "--query" if command == Command::Search => options.query = Some(value(&mut args)?),
                "--mode" if matches!(command, Command::Search | Command::Serve) => {
                    options.mode = Mode::parse(&value(&mut args)?)?;
                }
                "--limit" if matches!(command, Command::Search | Command::Serve) => {
                    options.limit = positive(&value(&mut args)?, 1_000)?;
                }
                "--timeout-ms" if matches!(command, Command::Search | Command::Serve) => {
                    let milliseconds = value(&mut args)?.parse::<u64>()
                        .map_err(|_| bad("timeout must be an integer number of milliseconds"))?;
                    query::validate_timeout(Some(milliseconds))?;
                    options.timeout_ms = Some(milliseconds);
                }
                "--filter" if matches!(command, Command::Search | Command::Serve) => {
                    options.filter = Some(filter::Filter::parse(&value(&mut args)?)?);
                }
                _ => return Err(bad("unknown or inapplicable option; use --help")),
            }
        }
        if options.receipt.as_os_str().is_empty() {
            return Err(bad("--receipt is required"));
        }
        if matches!(command, Command::Index | Command::Update) && options.directory.is_none() {
            return Err(bad("index/update requires --index-dir pointing to a NEW directory"));
        }
        if command == Command::Update && options.new_receipt.is_none() {
            return Err(bad("update requires --new-receipt pointing to a NEW file"));
        }
        if command == Command::Search {
            let query = options
                .query
                .as_deref()
                .ok_or_else(|| bad("search requires --query"))?;
            validate_query(query)?;
        }
        Ok(Some(options))
    }
}

fn value(args: &mut impl Iterator<Item = String>) -> Result<String> {
    let value = args.next().ok_or_else(|| bad("missing option value"))?;
    if value.is_empty() || value.starts_with("--") {
        return Err(bad("missing or empty option value"));
    }
    Ok(value)
}

fn positive(value: &str, maximum: usize) -> Result<usize> {
    let value: usize = value.parse().map_err(|_| bad("expected a positive integer"))?;
    if value == 0 || value > maximum {
        return Err(bad("integer option is outside its supported range"));
    }
    Ok(value)
}

fn validate_query(query: &str) -> Result<()> {
    if query.trim().is_empty() || query.len() > 64 * 1024 || query.contains('\0') {
        return Err(bad("query must be nonblank, NUL-free, and at most 64 KiB"));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentInput {
    id: String,
    content: String,
    title: Option<String>,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

fn read_documents(reader: &mut impl BufRead) -> Result<Vec<IndexableDocument>> {
    let mut documents = Vec::new();
    let mut ids = BTreeSet::new();
    let mut record = Vec::new();
    let mut total = 0_usize;
    let mut line = 0_usize;
    loop {
        record.clear();
        let count = (&mut *reader)
            .take(MAX_RECORD_BYTES as u64 + 1)
            .read_until(b'\n', &mut record)?;
        if count == 0 {
            break;
        }
        line += 1;
        total = total
            .checked_add(count)
            .ok_or_else(|| bad("input byte count overflow"))?;
        if record.len() > MAX_RECORD_BYTES || total > MAX_INPUT_BYTES {
            return Err(bad("input exceeds its 16 MiB record or 256 MiB stream limit"));
        }
        if record.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let input: DocumentInput = serde_json::from_slice(&record).map_err(|error| {
            bad(&format!(
                "invalid document on line {line} ({:?} at column {})",
                error.classify(),
                error.column(),
            ))
        })?;
        if input.id.trim().is_empty()
            || input.id.contains('\0')
            || input.id.len() > usize::from(u16::MAX)
            || !ids.insert(input.id.clone())
        {
            return Err(bad(&format!(
                "line {line}: IDs must be unique, nonblank, NUL-free, and at most 65535 bytes",
            )));
        }
        if documents.len() == MAX_DOCUMENTS {
            return Err(bad("input exceeds the 100000-document limit"));
        }
        let mut document = IndexableDocument::new(input.id, input.content);
        document.title = input.title;
        document.metadata = input.metadata;
        documents.push(document);
    }
    Ok(documents)
}

#[derive(Clone)]
struct Models {
    fast: Arc<dyn Embedder>,
    quality: Option<Arc<dyn Embedder>>,
}

fn load_models(root: Option<&Path>, quality: bool) -> Result<Models> {
    let policy = DetectOptions {
        offline: Some(true),
        ..DetectOptions::default()
    };
    let fast = EmbedderStack::auto_detect_fast_semantic_with_options(root, &policy)?.fast_arc();
    let quality = if quality {
        Some(
            EmbedderStack::auto_detect_quality_with_options(root, &policy)?.ok_or_else(|| {
                bad("the required local quality model is unavailable; install it or explicitly build --fast-only")
            })?,
        )
    } else {
        None
    };
    Ok(Models { fast, quality })
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotReceipt {
    byte_len: u64,
    sha256: [u8; 32],
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    schema: String,
    directory: PathBuf,
    generation: ArtifactGenerationIdentityV1,
    snapshot: SnapshotReceipt,
    documents: usize,
    fast_producer: String,
    quality_producer: Option<String>,
}

impl Selection {
    fn read(path: &Path) -> Result<Self> {
        if !fs::symlink_metadata(path)?.is_file() {
            return Err(bad("the trusted receipt must be a regular non-symlink file"));
        }
        let mut bytes = Vec::new();
        File::open(path)?
            .take(MAX_RECEIPT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_RECEIPT_BYTES {
            return Err(bad("selection receipt exceeds 64 KiB"));
        }
        let selection: Self = serde_json::from_slice(&bytes).map_err(|_| {
            bad("invalid selection receipt; use the original successful index receipt")
        })?;
        if selection.schema != SELECTION_SCHEMA
            || !selection.directory.is_absolute()
            || selection.documents > MAX_DOCUMENTS
        {
            return Err(bad("invalid selection schema, directory, or document count"));
        }
        selection.generation.validate()?;
        Ok(selection)
    }

    async fn open(&self, cx: &Cx, models: Models) -> Result<NativeBuiltHybridIndex> {
        if models.fast.identity()?.fingerprint() != self.fast_producer
            || models
                .quality
                .as_ref()
                .map(|model| model.identity().map(|identity| identity.fingerprint()))
                .transpose()?
                != self.quality_producer
        {
            return Err(bad("local producers differ from the trusted selection; no model substitution is permitted"));
        }
        let expected = GenerationComponentReceiptV1 {
            byte_len: self.snapshot.byte_len,
            sha256: self.snapshot.sha256,
        };
        let index = NativeBuiltHybridIndex::open_selected(
            cx,
            &self.directory,
            &expected,
            models.fast,
            models.quality,
        )
        .await?;
        if index.vectors().documents().len() != self.documents
            || index.vectors().fast().index().owner_witness().generation != self.generation
        {
            return Err(bad("reopened generation differs from the trusted selection"));
        }
        Ok(index)
    }
}

fn new_path(path: &Path) -> Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| bad("destination must name a new file or directory"))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent)?;
    // A NEW leaf can still corrupt an existing sealed ancestor's inventory.
    // Resolve parent aliases first and refuse known native/fsfs seal markers.
    for ancestor in parent.ancestors() {
        for marker in ["native.hybrid.json", "native.snapshot.json", "FSFS-BUNDLE.json"] {
            match fs::symlink_metadata(ancestor.join(marker)) {
                Ok(_) => return Err(bad("destination is inside a sealed index; choose a sibling path")),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    let path = parent.join(name);
    match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(path),
        Err(error) => Err(error.into()),
        Ok(_) => Err(bad(
            "destination already exists; existing generations and receipts are never overwritten",
        )),
    }
}

fn new_generation(sequence: u64) -> Result<ArtifactGenerationIdentityV1> {
    // Native snapshot sealing supports Linux/macOS. Read OS entropy without
    // adding a random-number dependency or deriving identities from wall clocks.
    let mut nonce = [0_u8; 16];
    File::open("/dev/urandom")?.read_exact(&mut nonce)?;
    Ok(ArtifactGenerationIdentityV1::new(sequence, nonce)?)
}

async fn build(
    cx: &Cx,
    options: &Options,
    directory: &Path,
    documents: Vec<IndexableDocument>,
    models: Models,
) -> Result<(NativeBuiltHybridIndex, Selection)> {
    let generation = new_generation(1)?;
    let fast_producer = models.fast.identity()?.fingerprint();
    let quality_producer = models
        .quality
        .as_ref()
        .map(|model| model.identity().map(|identity| identity.fingerprint()))
        .transpose()?;
    let retrieval = if options.exact {
        NativeBuildRetrieval::Exact
    } else {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 42,
        }
    };
    let mut builder = NativeIndexBuilder::new(directory, generation, models.fast)?
        .with_batch_size(options.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .add_documents(documents);
    if let Some(quality) = models.quality {
        builder = builder
            .with_quality_embedder(quality)?
            .with_quality_storage(NativeBuildPrecision::F32, retrieval)?;
    }
    let index = builder.build_hybrid(cx).await?;
    let snapshot = index.seal_for_reopen(cx)?;
    let selection = Selection {
        schema: SELECTION_SCHEMA.to_owned(),
        directory: index.vectors().directory().to_path_buf(),
        generation,
        snapshot: SnapshotReceipt {
            byte_len: snapshot.byte_len,
            sha256: snapshot.sha256,
        },
        documents: index.vectors().documents().len(),
        fast_producer,
        quality_producer,
    };
    Ok((index, selection))
}

fn save_selection(selection: &Selection, path: &Path) -> Result<()> {
    let bytes = encode(selection, MAX_RECEIPT_BYTES)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    let parent = path
        .parent()
        .ok_or_else(|| bad("receipt parent is missing"))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

async fn search(
    index: &NativeBuiltHybridIndex,
    cx: &Cx,
    query: &str,
    mode: Mode,
    limit: usize,
    filters: filter::Filters<'_>,
) -> Result<serde_json::Value> {
    validate_query(query)?;
    filter::Query::prepare(index, cx, filters)?.search(cx, query, mode, limit).await
}

async fn execute(cx: &Cx, options: Options, output: &mut impl Write) -> Result<()> {
    match options.command {
        Command::Update => update::execute(cx, &options, output).await,
        Command::Index => {
            let directory = new_path(
                options
                    .directory
                    .as_deref()
                    .ok_or_else(|| bad("missing index directory"))?,
            )?;
            let receipt = new_path(&options.receipt)?;
            // Freeze both destinations before reading stdin or initializing models.
            if receipt.starts_with(&directory) || directory.starts_with(&receipt) {
                return Err(bad("the trusted receipt and immutable index must not overlap"));
            }
            let documents = match &options.input {
                Some(path) => read_documents(&mut BufReader::new(File::open(path)?))?,
                None => read_documents(&mut io::stdin().lock())?,
            };
            let models = load_models(options.models.as_deref(), !options.fast_only)?;
            let (_index, selection) = build(cx, &options, &directory, documents, models).await?;
            save_selection(&selection, &receipt)?;
            emit(output, &serde_json::json!({
                "schema": SCHEMA, "ok": true, "event": "indexed", "receipt": receipt,
                "selection": selection,
            }))
        }
        Command::Search | Command::Serve => {
            let selection = Selection::read(&options.receipt)?;
            let models = load_models(
                options.models.as_deref(),
                selection.quality_producer.is_some(),
            )?;
            let index = selection.open(cx, models).await?;
            // Index/model admission is startup work. Each accepted query starts
            // its own total budget after that, before scoping or provider work.
            let policy = query::Policy::new(cx, options.timeout_ms)?;
            if options.command == Command::Serve {
                let live = serve::NativeLiveHybridIndex::new(cx, index)?;
                return serve::run(
                    &live,
                    cx,
                    &mut io::stdin().lock(),
                    output,
                    (options.mode, options.limit),
                    options.activation == serve::ActivationPermission::Enabled,
                    options.filter.as_ref(),
                    &policy,
                )
                .await;
            }
            if options.stream {
                let request = serve::Request {
                    id: None,
                    query: options.query.ok_or_else(|| bad("missing query"))?,
                    mode: Some(options.mode),
                    limit: Some(options.limit),
                    filter: options.filter,
                    timeout_ms: None,
                };
                return if serve::stream_one(
                    &index,
                    cx,
                    &request,
                    1,
                    (options.mode, options.limit),
                    output,
                    None,
                    &policy,
                )
                .await?
                {
                    Ok(())
                } else {
                    Err(bad("search did not complete all requested phases; see terminal frame"))
                };
            }
            let deadline = policy.start(cx, None)?;
            let page = query::within(cx, deadline.as_ref(), search(
                &index,
                cx,
                options.query.as_deref().ok_or_else(|| bad("missing query"))?,
                options.mode,
                options.limit,
                [None, options.filter.as_ref()],
            ))
            .await?;
            emit(output, &page)
        }
    }
}

struct LimitedBuffer {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for LimitedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("JSON output exceeds its byte limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encode(value: &impl Serialize, limit: usize) -> Result<Vec<u8>> {
    let mut buffer = LimitedBuffer {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut buffer, value)?;
    buffer.write_all(b"\n")?;
    Ok(buffer.bytes)
}

fn emit(output: &mut impl Write, value: &impl Serialize) -> Result<()> {
    let bytes = encode(value, MAX_OUTPUT_BYTES)?;
    output.write_all(&bytes)?;
    output.flush()?;
    Ok(())
}

fn bad(message: &str) -> Box<dyn Error + Send + Sync> {
    Box::new(io::Error::new(
        io::ErrorKind::InvalidInput,
        message.to_owned(),
    ))
}

fn run() -> Result<()> {
    let args = std::env::args_os()
        .skip(1)
        .map(|value| value.into_string().map_err(|_| bad("command arguments must be UTF-8")))
        .collect::<Result<Vec<_>>>()?;
    let Some(options) = Options::parse(args)? else {
        io::stdout().lock().write_all(HELP.as_bytes())?;
        return Ok(());
    };
    if !cfg!(any(target_os = "linux", target_os = "macos")) {
        return Err(bad("native selected snapshots currently require Linux or macOS"));
    }
    let runtime = RuntimeBuilder::current_thread()
        .blocking_threads(0, 2)
        .build()?;
    runtime.block_on(async move {
        let cx = Cx::current().ok_or_else(|| bad("runtime did not install a root context"))?;
        execute(&cx, options, &mut io::stdout().lock()).await
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let mut payload = query::failure(error.as_ref());
            payload["schema"] = serde_json::json!(SCHEMA);
            payload["ok"] = serde_json::json!(false);
            payload["event"] = serde_json::json!("error");
            let _ = emit(
                &mut io::stderr().lock(),
                &payload,
            );
            ExitCode::FAILURE
        }
    }
}
