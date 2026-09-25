//! Fresh-process lifecycle tests, using the real retained builder and indexer.
//! The counted hash producer is an explicit non-semantic control: these tests
//! establish reuse and invalidation, not real-model relevance or performance.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{
    Embedder, EmbeddingIdentityBundleV1, ModelCategory, SearchFuture, SearchResult,
};
use frankensearch_embed::HashEmbedder;
use serde::{Deserialize, Serialize};

use super::super::super::set_test_fast_embedder;
use super::super::{
    CompleteGenerationStore, FsfsRuntime, LEGACY_RECEIPT_FILE, LegacyReuseReceipt, RECEIPT_FILE,
    ReuseReceipt, proven, read_json, seed_candidate, session_id,
};
use crate::generation_store::{GenerationPublication, PublishedGeneration};
use crate::{CliCommand, CliInput, FsfsConfig};

const CHILD_ROOT: &str = "FSFS_REUSE_CHILD_ROOT";
const CHILD_MODE: &str = "FSFS_REUSE_CHILD_MODE";
const CHILD_REPORT: &str = "FSFS_REUSE_CHILD_REPORT";

struct CountedHash {
    inner: HashEmbedder,
    inputs: Arc<Mutex<Vec<String>>>,
}

impl Embedder for CountedHash {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.inputs.lock().unwrap().push(text.to_owned());
            self.inner.embed(cx, text).await
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            self.inputs
                .lock()
                .unwrap()
                .extend(texts.iter().map(|text| (*text).to_owned()));
            self.inner.embed_batch(cx, texts).await
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.inner.identity()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ChildReport {
    session: String,
    executable: String,
    generation: String,
    receipt_session: String,
    eligible: usize,
    seeded: Option<usize>,
    embedded_inputs: Vec<String>,
    hits: Vec<String>,
}

fn configured(parent: &Path) -> FsfsRuntime {
    let mut config = FsfsConfig::default();
    config.indexing.offline = true;
    config.indexing.quality_model.clear();
    config.indexing.embedding_batch_size = 1;
    config.search.fast_only = true;
    config.search.rerank = false;
    "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
    FsfsRuntime::new(config).with_cli_input(CliInput {
        command: CliCommand::Index,
        target_path: Some(parent.join("source")),
        index_dir: Some(parent.join("store")),
        quiet: true,
        ..CliInput::default()
    })
}

async fn rebuild(runtime: &FsfsRuntime, cx: &Cx, root: &Path) -> PublishedGeneration {
    let publication = runtime.rebuild_retained_generation(cx, root).await.unwrap();
    match publication {
        GenerationPublication::Durable(generation) => generation,
        other @ GenerationPublication::VisibleButDurabilityUncertain { .. } => {
            panic!("fixture publication is not durable: {other:?}")
        }
    }
}

// Only the parent tests launch this helper, with a fresh process and a real
// caller-owned blocking lane. A report file proves the named helper ran: a
// zero-test libtest success cannot satisfy the parent assertion.
#[test]
#[ignore = "internal subprocess helper; run the parent restart tests"]
fn restart_child() {
    let parent = PathBuf::from(std::env::var_os(CHILD_ROOT).expect("parent test root"));
    let mode = std::env::var(CHILD_MODE).expect("parent test mode");
    let report_path = PathBuf::from(std::env::var_os(CHILD_REPORT).expect("parent report path"));
    let scheduler = asupersync::runtime::RuntimeBuilder::current_thread()
        .blocking_threads(0, 2)
        .build()
        .unwrap();
    scheduler.block_on(async move {
        let cx = Cx::current().expect("runtime context");
        let inputs = Arc::new(Mutex::new(Vec::new()));
        set_test_fast_embedder(Some(Arc::new(CountedHash {
            inner: HashEmbedder::default_256(),
            inputs: Arc::clone(&inputs),
        })));
        super::prepare(&cx).await.unwrap();
        let executable = super::fingerprint()
            .expect("Linux restart test requires a provable executing image")
            .to_owned();
        let mut runtime = configured(&parent);
        let root = parent.join("store");
        if mode == "legacy_rebuild" {
            let payload = runtime
                .run_one_shot_index_scaffold_internal(
                    &cx,
                    CliCommand::Index,
                    |_| Ok(()),
                    false,
                    true,
                )
                .await
                .unwrap();
            let embedded_inputs = inputs.lock().unwrap().clone();
            let evidence: LegacyReuseReceipt =
                read_json(&cx, &root.join(LEGACY_RECEIPT_FILE)).unwrap();
            let layout = FsfsRuntime::resolve_lexical_engine(&root).unwrap();
            let lexical = frankensearch_quill::QuillSearchIndex::open(
                &cx,
                layout.engine_dir().unwrap(),
                frankensearch_quill::QuillConfig::default(),
            )
            .await
            .unwrap();
            let mut hits = lexical
                .search_results(&cx, "sharedtoken", 10)
                .unwrap()
                .into_iter()
                .map(|hit| hit.doc_id.to_string())
                .collect::<Vec<_>>();
            hits.sort();
            let report = ChildReport {
                session: session_id().unwrap().to_owned(),
                executable,
                generation: payload.generation.source_hash_hex,
                receipt_session: evidence.receipt.session,
                eligible: evidence
                    .receipt
                    .checkpoint
                    .files
                    .values()
                    .filter(|entry| proven(entry))
                    .count(),
                seeded: None,
                embedded_inputs,
                hits,
            };
            fs::write(report_path, serde_json::to_vec(&report).unwrap()).unwrap();
            return;
        }
        let mut seeded = None;
        let generation = if mode == "rebuild" {
            rebuild(&runtime, &cx, &root).await
        } else {
            if mode == "seed_changed_config" {
                // The default limit is unlimited (usize::MAX); any finite value
                // changes the configuration without overflowing.
                runtime.config.search.default_limit = 17;
            } else if mode == "seed_full" {
                runtime.cli_input.full_reindex = true;
            } else {
                assert_eq!(mode, "seed");
            }
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let selected = store.active(&cx).unwrap().unwrap();
            let build = store.begin(&cx).unwrap();
            let mut input = runtime.cli_input.clone();
            input.index_dir = Some(build.path().to_path_buf());
            let candidate = runtime.clone().with_cli_input(input);
            seeded = Some(seed_candidate(&cx, &candidate, &store, build.path()).unwrap());
            if seeded == Some(0) {
                assert_eq!(fs::read_dir(build.path()).unwrap().count(), 0);
            }
            assert_eq!(store.active(&cx).unwrap(), Some(selected.clone()));
            selected
        };
        // Capture build-only inference before the verification query below.
        let embedded_inputs = inputs.lock().unwrap().clone();
        let receipt: ReuseReceipt = read_json(&cx, &generation.path().join(RECEIPT_FILE)).unwrap();
        let mut reader = runtime.open_retained_search(&cx, &root).await.unwrap();
        let phases = reader.search(&cx, "sharedtoken", 10).await.unwrap();
        let mut hits = phases
            .last()
            .unwrap()
            .hits
            .iter()
            .map(|hit| hit.path.clone())
            .collect::<Vec<_>>();
        hits.sort();
        let report = ChildReport {
            session: session_id().unwrap().to_owned(),
            executable,
            generation: generation.id().to_owned(),
            receipt_session: receipt.session,
            eligible: receipt
                .checkpoint
                .files
                .values()
                .filter(|entry| proven(entry))
                .count(),
            seeded,
            embedded_inputs,
            hits,
        };
        fs::write(report_path, serde_json::to_vec(&report).unwrap()).unwrap();
    });
}

struct OwnedChild(Child);

impl Drop for OwnedChild {
    fn drop(&mut self) {
        // Only reap this test's own subprocess, including assertion unwinding.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn child(executable: &Path, parent: &Path, mode: &str, label: &str) -> ChildReport {
    let report_path = parent.join(format!("{label}.json"));
    let log_path = parent.join(format!("{label}.log"));
    let log = File::create(&log_path).unwrap();
    let test_name = format!(
        "{}::restart_child",
        module_path!().split_once("::").unwrap().1
    );
    let mut child = OwnedChild(
        Command::new(executable)
            .args([
                "--exact",
                &test_name,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ROOT, parent)
            .env(CHILD_MODE, mode)
            .env(CHILD_REPORT, &report_path)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone().unwrap()))
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap(),
    );
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "restart child timed out; log at {}",
            log_path.display()
        );
        // This is a subprocess watchdog, not a timing-based correctness race.
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(
        status.success(),
        "restart child failed: {}",
        fs::read_to_string(log_path).unwrap()
    );
    let report = fs::read(report_path).expect("child ran and produced its report");
    serde_json::from_slice(&report).unwrap()
}

fn source_fixture(parent: &Path) {
    fs::create_dir(parent.join("source")).unwrap();
    for number in 0..4 {
        fs::write(
            parent.join("source").join(format!("doc-{number}.md")),
            format!("sharedtoken document {number}"),
        )
        .unwrap();
    }
}

fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                out.insert(
                    entry.path().strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(entry.path()).unwrap(),
                );
            }
        }
    }
    out
}

fn generation_path(parent: &Path, report: &ChildReport) -> PathBuf {
    parent.join("store/generations").join(&report.generation)
}

/// Embedded source texts, without the one `"probe"` input every index run
/// sends to check its embedder (`probe_indexing_embedder_with_backoffs`).
fn source_inputs(report: &ChildReport) -> Vec<&str> {
    report
        .embedded_inputs
        .iter()
        .map(String::as_str)
        .filter(|text| *text != "probe")
        .collect()
}

#[test]
fn restart_completed_legacy_reuses_inputs_and_rejects_changed_executable() {
    let directory = tempfile::tempdir().unwrap();
    let parent = directory.path();
    source_fixture(parent);
    let executable = std::env::current_exe().unwrap();
    let first = child(&executable, parent, "legacy_rebuild", "first");
    assert_eq!(first.eligible, 4);
    assert_eq!(source_inputs(&first).len(), 4);
    let second = child(&executable, parent, "legacy_rebuild", "second");
    assert_ne!(second.session, first.session);
    assert_eq!(second.executable, first.executable);
    assert_eq!(second.hits, first.hits);
    assert!(source_inputs(&second).is_empty());

    let alternate = parent.join("alternate-legacy-test-image");
    fs::copy(&executable, &alternate).unwrap();
    let mut file = OpenOptions::new().append(true).open(&alternate).unwrap();
    file.write_all(b"\nfsfs legacy restart negative control\n")
        .unwrap();
    file.sync_all().unwrap();
    drop(file);
    let changed = child(&alternate, parent, "legacy_rebuild", "changed-image");
    assert_ne!(changed.executable, first.executable);
    assert_eq!(source_inputs(&changed).len(), 4);
    assert_eq!(changed.hits, first.hits);
}

#[test]
fn restart_same_executable_reuses_proven_inputs_without_reembedding() {
    let directory = tempfile::tempdir().unwrap();
    let parent = directory.path();
    source_fixture(parent);
    let executable = std::env::current_exe().unwrap();
    let first = child(&executable, parent, "rebuild", "first");
    assert_eq!(first.eligible, 4);
    // The probe alone would make the first run's inputs non-empty; the later
    // "nothing re-embedded" check only means something if every source was.
    for number in 0..4 {
        let document = format!("document {number}");
        assert!(
            source_inputs(&first)
                .iter()
                .any(|text| text.contains(&document)),
            "first build did not embed {document}: {:?}",
            first.embedded_inputs
        );
    }
    let retained = generation_path(parent, &first);
    let before = snapshot(&retained);
    let seeded = child(&executable, parent, "seed", "seeded");
    assert_eq!(seeded.seeded, Some(4));
    assert_ne!(seeded.session, first.session);
    assert_eq!(seeded.receipt_session, first.session);
    assert_eq!(seeded.executable, first.executable);
    assert!(seeded.embedded_inputs.is_empty());
    let second = child(&executable, parent, "rebuild", "second");
    assert_ne!(second.session, first.session);
    assert_ne!(second.generation, first.generation);
    assert_eq!(second.executable, first.executable);
    assert_eq!(second.hits, first.hits);
    assert!(
        source_inputs(&second).is_empty(),
        "unchanged sources were embedded again: {:?}",
        second.embedded_inputs
    );
    assert_eq!(snapshot(&retained), before);
}

#[test]
fn restart_reuses_only_unchanged_sources_and_reconciles_membership() {
    let directory = tempfile::tempdir().unwrap();
    let parent = directory.path();
    source_fixture(parent);
    let executable = std::env::current_exe().unwrap();
    let first = child(&executable, parent, "rebuild", "first");
    let retained = generation_path(parent, &first);
    let before = snapshot(&retained);
    let changed = parent.join("source/doc-0.md");
    let old_modified = fs::metadata(&changed).unwrap().modified().unwrap();
    // Same byte length and restored mtime: content hashing, not metadata alone,
    // must prevent reuse of this entry after a genuine process restart.
    fs::write(&changed, "sharedtoken document Z").unwrap();
    File::options()
        .write(true)
        .open(&changed)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(old_modified))
        .unwrap();
    fs::rename(parent.join("source/doc-1.md"), parent.join("removed.md")).unwrap();
    fs::write(parent.join("source/doc-4.md"), "sharedtoken document 4").unwrap();
    let second = child(&executable, parent, "rebuild", "second");
    assert_ne!(second.session, first.session);
    assert_eq!(second.executable, first.executable);
    assert_eq!(
        second.hits,
        ["doc-0.md", "doc-2.md", "doc-3.md", "doc-4.md"]
    );
    assert!(
        second
            .embedded_inputs
            .iter()
            .any(|text| text.contains("document Z"))
    );
    assert!(
        second
            .embedded_inputs
            .iter()
            .any(|text| text.contains("document 4"))
    );
    for unchanged in ["document 2", "document 3"] {
        assert!(
            !second
                .embedded_inputs
                .iter()
                .any(|text| text.contains(unchanged))
        );
    }
    assert_eq!(snapshot(&retained), before);
}

#[test]
fn restart_changed_executable_bytes_cannot_reuse_a_same_version_receipt() {
    let directory = tempfile::tempdir().unwrap();
    let parent = directory.path();
    source_fixture(parent);
    let executable = std::env::current_exe().unwrap();
    let first = child(&executable, parent, "rebuild", "first");
    let retained = generation_path(parent, &first);
    let before = snapshot(&retained);
    let alternate = parent.join("alternate-test-image");
    fs::copy(&executable, &alternate).unwrap();
    {
        // Trailing ELF data changes the complete image, but not its compiled
        // test logic or package/version strings. Do not touch the real binary.
        let mut file = OpenOptions::new().append(true).open(&alternate).unwrap();
        file.write_all(b"\nfsfs restart negative control\n")
            .unwrap();
        file.sync_all().unwrap();
    }
    let refused = child(&alternate, parent, "seed", "different-image");
    assert_ne!(refused.executable, first.executable);
    assert_eq!(refused.seeded, Some(0));
    assert_eq!(refused.generation, first.generation);
    assert_eq!(refused.hits, first.hits);
    assert!(refused.embedded_inputs.is_empty());
    assert_eq!(snapshot(&retained), before);
}

#[test]
fn restart_changed_configuration_and_explicit_full_reindex_bypass_seed() {
    let directory = tempfile::tempdir().unwrap();
    let parent = directory.path();
    source_fixture(parent);
    let executable = std::env::current_exe().unwrap();
    let first = child(&executable, parent, "rebuild", "first");
    let retained = generation_path(parent, &first);
    let before = snapshot(&retained);
    for mode in ["seed_changed_config", "seed_full"] {
        let refused = child(&executable, parent, mode, mode);
        assert_eq!(refused.executable, first.executable);
        assert_eq!(refused.seeded, Some(0));
        assert_eq!(refused.generation, first.generation);
        assert!(refused.embedded_inputs.is_empty());
    }
    assert_eq!(snapshot(&retained), before);
}
