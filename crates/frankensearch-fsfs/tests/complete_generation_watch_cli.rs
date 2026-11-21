//! Actual-executable watch/read/daemon lifecycle, never the library's test embedder.
#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use frankensearch_fsfs::FsfsConfig;
#[cfg(feature = "semantic-support")]
use frankensearch_fsfs::generation_store::{CompleteGenerationStore, PublishedGeneration};
use serde_json::Value;

struct Fixture {
    directory: tempfile::TempDir,
    source: PathBuf,
    store: PathBuf,
    config: PathBuf,
    executable: PathBuf,
    #[cfg(feature = "semantic-support")]
    quality: bool,
    #[cfg(feature = "semantic-support")]
    quality_only: bool,
}

#[cfg(feature = "semantic-support")]
fn complete_stream_frames(output: &Output, format: &str) -> Vec<Value> {
    use frankensearch_fsfs::stream_protocol::{StreamFrame, decode_stream_frame_toon};
    let text = std::str::from_utf8(&output.stdout).expect("UTF-8 stream");
    let frames: Vec<StreamFrame<frankensearch_fsfs::output_schema::SearchHitPayload>> =
        if format == "jsonl" {
            text.lines()
                .map(|line| serde_json::from_str(line).expect("every line is a stream frame"))
                .collect()
        } else {
            assert!(text.starts_with('\u{001e}'), "TOON record separator");
            text.split('\u{001e}')
                .skip(1)
                .map(|record| {
                    let raw: Value = toon_format::decode_default(record.trim())
                        .expect("every TOON record is a complete object");
                    assert!(
                        raw.as_object()
                            .expect("TOON stream frame object")
                            .keys()
                            .all(|key| {
                                [
                                    "v",
                                    "schema_version",
                                    "stream_id",
                                    "seq",
                                    "ts",
                                    "command",
                                    "event",
                                    "payload",
                                ]
                                .contains(&key.as_str())
                            }),
                        "TOON record contains non-frame fields: {raw}"
                    );
                    decode_stream_frame_toon(record.trim())
                        .expect("every TOON record is a stream frame")
                })
                .collect()
        };
    for (sequence, frame) in frames.iter().enumerate() {
        assert!(frankensearch_fsfs::stream_protocol::validate_stream_frame(frame).valid);
        assert_eq!(frame.seq, u64::try_from(sequence).unwrap());
        assert_eq!(frame.stream_id, frames[0].stream_id);
    }
    frames
        .into_iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect()
}

impl Fixture {
    fn new(models: Option<&Path>, quality: bool) -> Self {
        // This fixture also binds the store's real AF_UNIX daemon socket.
        let directory = tempfile::tempdir_in("/tmp").unwrap();
        let source = directory.path().join("source");
        let store = directory.path().join("store");
        let config = directory.path().join("fsfs.toml");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("alpha.md"), "sharedtoken alpha document").unwrap();
        let mut settings = FsfsConfig::default();
        "{index_dir}/catalog.sqlite".clone_into(&mut settings.storage.db_path);
        settings.indexing.offline = true;
        settings.indexing.watch_mode = false;
        if !quality {
            settings.indexing.quality_model.clear();
        }
        settings.search.fast_only = !quality;
        settings.search.rerank = false;
        if let Some(models) = models {
            settings.indexing.model_dir = models.display().to_string();
        }
        fs::write(&config, settings.to_toml().unwrap()).unwrap();
        Self {
            directory,
            source,
            store,
            config,
            executable: PathBuf::from(env!("CARGO_BIN_EXE_fsfs")),
            #[cfg(feature = "semantic-support")]
            quality,
            #[cfg(feature = "semantic-support")]
            quality_only: false,
        }
    }

    #[cfg(all(feature = "semantic-support", feature = "rerank"))]
    fn configure_quality_only(&mut self, models: &Path) {
        let mut settings = FsfsConfig::default();
        "{index_dir}/catalog.sqlite".clone_into(&mut settings.storage.db_path);
        settings.indexing.offline = true;
        settings.indexing.watch_mode = false;
        "unused-invalid-fast-producer".clone_into(&mut settings.indexing.fast_model);
        "all-minilm-l6-v2-native".clone_into(&mut settings.indexing.quality_model);
        settings.indexing.model_dir = models.display().to_string();
        settings.search.fast_only = false;
        settings.search.rerank = false;
        settings.search.quality_timeout_ms = 30_000;
        fs::write(&self.config, settings.to_toml().unwrap()).unwrap();
        self.quality_only = true;
    }

    fn command(&self, command: &str, format: &str) -> Command {
        let mut child = Command::new(&self.executable);
        for (name, _) in std::env::vars_os() {
            let text = name.to_string_lossy();
            if text.starts_with("FSFS_") || text.starts_with("FRANKENSEARCH_") {
                child.env_remove(name);
            }
        }
        child
            .current_dir(self.directory.path())
            .env("FRANKENSEARCH_CHECK_UPDATES", "0")
            .arg(command);
        // Index/watch take their source immediately after the subcommand.
        if matches!(command, "index" | "watch") {
            child.arg(&self.source);
        }
        child
            .arg("--config")
            .arg(&self.config)
            .arg("--index-dir")
            .arg(&self.store)
            .args(["--quiet", "--format", format]);
        child
    }
}

/// Capture to files so an idle read or a full pipe cannot hang test teardown.
struct Process {
    child: Child,
    capture: tempfile::TempDir,
}

impl Process {
    fn start(command: &mut Command) -> Self {
        let capture = tempfile::tempdir().unwrap();
        let child = command
            .stdin(Stdio::null())
            .stdout(fs::File::create(capture.path().join("stdout")).unwrap())
            .stderr(fs::File::create(capture.path().join("stderr")).unwrap())
            .spawn()
            .expect("start production fsfs");
        Self { child, capture }
    }

    fn stdout(&self) -> Vec<u8> {
        fs::read(self.capture.path().join("stdout")).unwrap()
    }

    #[cfg(feature = "semantic-support")]
    fn assert_alive(&mut self) {
        assert!(
            self.child.try_wait().unwrap().is_none(),
            "fsfs subprocess exited\n{}\n{}",
            String::from_utf8_lossy(&self.stdout()),
            fs::read_to_string(self.capture.path().join("stderr")).unwrap()
        );
    }

    fn finish(&mut self, timeout: Duration) -> Output {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return Output {
                    status,
                    stdout: self.stdout(),
                    stderr: fs::read(self.capture.path().join("stderr")).unwrap(),
                };
            }
            assert!(
                Instant::now() < deadline,
                "fsfs subprocess exceeded its deadline"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[cfg(feature = "semantic-support")]
    fn stop_gracefully(&mut self) -> Output {
        self.assert_alive();
        let signal = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .expect("signal own watcher");
        assert!(signal.success());
        let output = self.finish(Duration::from_secs(30));
        assert!(
            output.status.success(),
            "watcher failed to drain: {output:?}"
        );
        output
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

#[test]
fn complete_watch_binary_refuses_unframed_output_before_creating_store() {
    for command in ["watch", "index"] {
        let fixture = Fixture::new(None, false);
        let mut invocation = fixture.command(command, "json");
        invocation.env("FSFS_COMPLETE_GENERATIONS", "1");
        if command == "index" {
            invocation.arg("--watch");
        }
        let output = Process::start(&mut invocation).finish(Duration::from_secs(10));
        assert!(!output.status.success());
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["ok"], false);
        assert!(
            result["error"].to_string().contains("watch_format"),
            "{result}"
        );
        assert!(!fixture.store.exists());
    }
}

#[test]
fn watch_binary_enforces_background_policy_before_creating_an_index() {
    for complete in ["0", "1"] {
        for command in ["watch", "index"] {
            for (profile, hard_pause, expected_field) in [
                ("strict", false, "pressure.profile"),
                ("degraded", false, "pressure.profile"),
                ("performance", true, "pressure.hard_pause_requested"),
            ] {
                let fixture = Fixture::new(None, false);
                let mut invocation = fixture.command(command, "jsonl");
                invocation
                    .env("FSFS_COMPLETE_GENERATIONS", complete)
                    .env("FRANKENSEARCH_PRESSURE_PROFILE", profile)
                    // ForceFull must not override the independent background
                    // capability ceiling or hard pause.
                    .env("FRANKENSEARCH_PRESSURE_DEGRADATION_OVERRIDE", "force_full")
                    .env(
                        "FRANKENSEARCH_PRESSURE_HARD_PAUSE_REQUESTED",
                        hard_pause.to_string(),
                    );
                if command == "index" {
                    invocation.arg("--watch");
                }
                let output = Process::start(&mut invocation).finish(Duration::from_secs(10));
                assert!(
                    !output.status.success(),
                    "{command}, complete={complete}, profile={profile}, pause={hard_pause}"
                );
                let result: Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(result["ok"], false);
                assert!(
                    result["error"].to_string().contains(expected_field),
                    "{command}, complete={complete}, profile={profile}, pause={hard_pause}: {result}"
                );
                assert!(
                    !fixture.store.exists(),
                    "policy refusal must precede the initial indexing run"
                );
                assert_eq!(
                    fs::read_to_string(fixture.source.join("alpha.md")).unwrap(),
                    "sharedtoken alpha document"
                );
            }
        }
    }
}

#[test]
fn later_no_watch_override_keeps_index_binary_one_shot() {
    for complete in [None, Some("0"), Some("1")] {
        let fixture = Fixture::new(None, false);
        let mut invocation = fixture.command("index", "json");
        if let Some(complete) = complete {
            invocation.env("FSFS_COMPLETE_GENERATIONS", complete);
        }
        let output = Process::start(
            invocation
                .env(
                    "FRANKENSEARCH_PRESSURE_DEGRADATION_OVERRIDE",
                    "force_lexical_only",
                )
                .args(["--watch", "--no-watch-mode"]),
        )
        .finish(Duration::from_secs(30));
        assert!(
            output.status.success(),
            "complete={complete:?}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["ok"], true);
        assert!(fixture.store.exists());
        assert_eq!(
            fixture.store.join("FSFS-CURRENT").is_file(),
            complete.map_or(
                cfg!(any(target_os = "linux", target_os = "macos")),
                |value| value == "1"
            ),
            "default selection and explicit opt-out must select the intended layout: {complete:?}"
        );
    }
}

#[cfg(feature = "semantic-support")]
fn wait_for_visible_paths(
    fixture: &Fixture,
    watch: &mut Process,
    expected: &[&str],
    after: Option<&str>,
) -> PublishedGeneration {
    let reader_runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .unwrap();
    let cx = reader_runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut last_generation = String::new();
    loop {
        watch.assert_alive();
        assert!(
            Instant::now() < deadline,
            "watch never made {expected:?} visible"
        );
        // Ignore a partially written last line; only complete receipts admit a search.
        let bytes = watch.stdout();
        let receipt = bytes
            .split_inclusive(|byte| *byte == b'\n')
            .filter(|line| line.ends_with(b"\n"))
            .filter_map(|line| serde_json::from_slice::<Value>(line).ok())
            .rfind(|value| value["ok"] == true && value["data"]["publication"] == "durable");
        if let Some(receipt) = receipt {
            let generation = receipt["data"]["generation_id"].as_str().unwrap();
            if generation != last_generation && after != Some(generation) {
                // Admit and pin the exact selected image before querying. A
                // pathname alone is not a reader and cannot fence collection.
                let store = CompleteGenerationStore::open(&cx, &fixture.store).unwrap();
                let selected = store.active(&cx).unwrap().unwrap();
                if selected.id() != generation {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
                generation.clone_into(&mut last_generation);
                // The synthetic lexical marker can have zero model signal;
                // common words must exercise the actual semantic reader.
                let output = Process::start(fixture.command("search", "json").args([
                    "find a relevant document",
                    if fixture.quality_only {
                        "--quality-only"
                    } else {
                        "--fast-only"
                    },
                    "--no-daemon",
                    "--no-rerank",
                    "--limit",
                    "10",
                ]))
                .finish(Duration::from_secs(60));
                watch.assert_alive();
                assert!(
                    output.status.success(),
                    "search while watcher alive failed: {}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                let result: Value = serde_json::from_slice(&output.stdout).unwrap();
                let hits = result["data"]["hits"].as_array().unwrap();
                if hits.len() == expected.len()
                    && expected.iter().all(|name| {
                        hits.iter().any(|hit| {
                            Path::new(hit["path"].as_str().unwrap())
                                .file_name()
                                .is_some_and(|filename| filename == std::ffi::OsStr::new(name))
                        })
                    })
                    && store.is_selected(&cx, &selected).unwrap()
                {
                    assert_search_hits(&result["data"], expected, "semantic_rank");
                    if fixture.quality_only {
                        assert_eq!(result["data"]["skip_reason"], "quality_only");
                        assert!(selected.path().join("vector/quality.fsvi").is_file());
                        assert!(!selected.path().join("vector/index.fsvi").exists());
                    } else if fixture.quality {
                        let quality = Process::start(fixture.command("search", "json").args([
                            "find a relevant document",
                            "--quality-only",
                            "--no-daemon",
                            "--no-rerank",
                            "--limit",
                            "10",
                        ]))
                        .finish(Duration::from_secs(60));
                        watch.assert_alive();
                        assert!(
                            quality.status.success(),
                            "quality search while watcher alive failed: {}\n{}",
                            String::from_utf8_lossy(&quality.stdout),
                            String::from_utf8_lossy(&quality.stderr)
                        );
                        let quality: Value = serde_json::from_slice(&quality.stdout).unwrap();
                        let payload = &quality["data"];
                        assert_eq!(payload["skip_reason"], "quality_only");
                        assert!(payload["vector_generation_id"].is_string());
                        assert_ne!(
                            payload["vector_generation_id"],
                            result["data"]["vector_generation_id"]
                        );
                        assert_search_hits(payload, expected, "semantic_rank");
                        assert!(store.is_selected(&cx, &selected).unwrap());
                    }
                    assert_eq!(
                        selected.path(),
                        Path::new(receipt["data"]["generation_path"].as_str().unwrap())
                    );
                    eprintln!(
                        "direct readers observed durable generation={generation} after={after:?} paths={expected:?}"
                    );
                    return selected;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(feature = "semantic-support")]
fn assert_search_hits(payload: &Value, expected: &[&str], rank: &str) {
    let hits = payload["hits"].as_array().unwrap();
    assert_eq!(hits.len(), expected.len(), "{payload}");
    for name in expected {
        assert!(
            hits.iter().any(|hit| {
                hit[rank].is_number()
                    && Path::new(hit["path"].as_str().unwrap())
                        .file_name()
                        .is_some_and(|filename| filename == std::ffi::OsStr::new(name))
            }),
            "missing {name} with {rank}: {payload}"
        );
    }
}

#[cfg(feature = "semantic-support")]
fn daemon_query(
    fixture: &Fixture,
    watch: &mut Process,
    daemon: &mut Process,
    query: &str,
    mode: &str,
) -> Value {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::os::unix::net::UnixStream;

    watch.assert_alive();
    daemon.assert_alive();
    let mut peer = UnixStream::connect(fixture.store.join("fsfs-query.sock")).unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    peer.set_write_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    serde_json::to_writer(
        &mut peer,
        &serde_json::json!({"query": query, "mode": mode, "limit": 10, "rerank": false}),
    )
    .unwrap();
    peer.write_all(b"\n").unwrap();
    let mut line = String::new();
    assert_ne!(
        BufReader::new(peer)
            .take(1024 * 1024)
            .read_line(&mut line)
            .unwrap(),
        0,
        "daemon returned premature EOF"
    );
    assert!(line.ends_with('\n'), "daemon returned an incomplete frame");
    watch.assert_alive();
    daemon.assert_alive();
    let response: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(response["ok"], true, "{response}");
    assert_eq!(response["query"], query);
    assert_eq!(response["mode"], mode);
    assert_eq!(response["payloads"].as_array().unwrap().len(), 1);
    eprintln!(
        "daemon query={query} mode={mode} cached={} producer={} hits={}",
        response["cached"],
        response["payloads"][0]["vector_generation_id"],
        response["payloads"][0]["hits"].as_array().unwrap().len()
    );
    response
}

#[cfg(feature = "semantic-support")]
fn assert_daemon_visible_paths(
    fixture: &Fixture,
    watch: &mut Process,
    daemon: &mut Process,
    expected: &[&str],
) {
    // Use the real socket, so a successful in-process CLI fallback cannot
    // impersonate a daemon that remains pinned to an old generation.
    let fast = daemon_query(
        fixture,
        watch,
        daemon,
        "find a relevant document",
        "fast_only",
    );
    let fast = &fast["payloads"][0];
    assert_eq!(fast["phase"], "initial");
    assert!(fast["vector_generation_id"].is_string());
    assert_ne!(fast["vector_generation_is_hash"], true);
    assert_search_hits(fast, expected, "semantic_rank");
    if fixture.quality {
        let quality = daemon_query(
            fixture,
            watch,
            daemon,
            "find a relevant document",
            "quality_only",
        );
        let quality = &quality["payloads"][0];
        assert_eq!(quality["skip_reason"], "quality_only");
        assert!(quality["vector_generation_id"].is_string());
        assert_ne!(quality["vector_generation_is_hash"], true);
        assert_ne!(
            quality["vector_generation_id"],
            fast["vector_generation_id"]
        );
        assert_search_hits(quality, expected, "semantic_rank");
    }
    let lexical = daemon_query(fixture, watch, daemon, "sharedtoken", "lexical_only");
    assert_search_hits(&lexical["payloads"][0], expected, "lexical_rank");
}

#[cfg(feature = "semantic-support")]
fn assert_content_edit_visible(
    fixture: &Fixture,
    watch: &mut Process,
    daemon: &mut Process,
    previous: &str,
) -> PublishedGeneration {
    let original = daemon_query(fixture, watch, daemon, "priorcontenttoken", "lexical_only");
    assert_search_hits(&original["payloads"][0], &["alpha.md"], "lexical_rank");
    // Response delivery precedes the concurrent worker's cache merge. Wait
    // for an observed cache hit before changing the source, rather than
    // assuming the next connection cannot race that merge.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(
            Instant::now() < deadline,
            "daemon never warmed the cache control"
        );
        let warm = daemon_query(fixture, watch, daemon, "priorcontenttoken", "lexical_only");
        assert_search_hits(&warm["payloads"][0], &["alpha.md"], "lexical_rank");
        if warm["cached"] == true {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    eprintln!("watch source content-edit after={previous}");
    fs::write(
        fixture.source.join("alpha.md"),
        "sharedtoken alpha document replacementcontenttoken",
    )
    .unwrap();
    let edited = wait_for_visible_paths(fixture, watch, &["alpha.md"], Some(previous));
    assert_daemon_visible_paths(fixture, watch, daemon, &["alpha.md"]);
    let old = daemon_query(fixture, watch, daemon, "priorcontenttoken", "lexical_only");
    assert_eq!(
        old["cached"], false,
        "old positive cache must be invalidated"
    );
    assert_search_hits(&old["payloads"][0], &[], "lexical_rank");
    let new = daemon_query(
        fixture,
        watch,
        daemon,
        "replacementcontenttoken",
        "lexical_only",
    );
    assert_search_hits(&new["payloads"][0], &["alpha.md"], "lexical_rank");
    edited
}

#[cfg(feature = "semantic-support")]
#[test]
#[ignore = "requires verified models and a source-bound prior executable in FSFS_UPGRADE_REUSE_PRIOR_EXECUTABLE"]
fn complete_watch_binary_reuses_prior_executable_and_restarts_warm() {
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::process::ExitStatusExt;

    use sha2::{Digest, Sha256};

    fn file_sha256(path: &Path) -> String {
        let mut file = fs::File::open(path).unwrap();
        let mut digest = Sha256::new();
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        frankensearch_storage::ContentHasher::to_hex(&digest.finalize().into())
    }

    fn private_models(fixture: &Fixture, original: &Path) -> PathBuf {
        let destination = fixture.directory.path().join("models");
        let mut files = vec![
            "potion-multilingual-128M/tokenizer.json",
            "potion-multilingual-128M/model.safetensors",
        ];
        if fixture.quality {
            files.extend([
                "all-MiniLM-L6-v2/onnx/model.onnx",
                "all-MiniLM-L6-v2/tokenizer.json",
                "all-MiniLM-L6-v2/config.json",
                "all-MiniLM-L6-v2/special_tokens_map.json",
                "all-MiniLM-L6-v2/tokenizer_config.json",
            ]);
        }
        for relative in files {
            let source = original.join(relative);
            let target = destination.join(relative);
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::copy(&source, &target).unwrap();
            let source = fs::metadata(source).unwrap();
            let target = fs::metadata(target).unwrap();
            assert_ne!((source.dev(), source.ino()), (target.dev(), target.ino()));
        }
        frankensearch_embed::verify_dir_and_record(
            &frankensearch_embed::ModelManifest::potion_128m(),
            &destination.join("potion-multilingual-128M"),
        )
        .unwrap();
        if fixture.quality {
            frankensearch_embed::verify_dir_and_record(
                &frankensearch_embed::ModelManifest::minilm_v2(),
                &destination.join("all-MiniLM-L6-v2"),
            )
            .unwrap();
        }
        destination
    }

    fn replace_byte(path: &Path, offset: u64, byte: u8) {
        let mut file = fs::OpenOptions::new().write(true).open(path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[byte]).unwrap();
        file.sync_all().unwrap();
    }

    /// Keep a valid tokenizer tree or F32 tensor layout while changing one byte.
    fn drift_model_artifact(path: &Path, artifact: &str) -> (u64, u8) {
        if artifact != "weights" {
            let mut bytes = fs::read(path).unwrap();
            let mut expected: Value = serde_json::from_slice(&bytes).unwrap();
            let (needle, old, new): (&[u8], u8, u8) = match artifact {
                "tokenizer" => {
                    let content = &mut expected["normalizer"]["normalizers"][1]["content"];
                    assert_eq!(*content, " ! ");
                    *content = Value::String(" ? ".to_owned());
                    (b"\"content\":\" ! \"", b'!', b'?')
                }
                "quality-tokenizer" => {
                    let stride = &mut expected["truncation"]["stride"];
                    assert_eq!(*stride, 0);
                    *stride = Value::from(1);
                    (b"\"stride\":0", b'0', b'1')
                }
                _ => panic!("unknown drift fixture {artifact}"),
            };
            let matches = bytes
                .windows(needle.len())
                .enumerate()
                .filter_map(|(offset, value)| (value == needle).then_some(offset))
                .collect::<Vec<_>>();
            assert_eq!(matches.len(), 1);
            let offset = matches[0] + needle.iter().position(|byte| *byte == old).unwrap();
            bytes[offset] = new;
            assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), expected);
            let offset = u64::try_from(offset).unwrap();
            replace_byte(path, offset, new);
            (offset, old)
        } else {
            let mut file = fs::File::open(path).unwrap();
            let mut prefix = [0_u8; 8];
            file.read_exact(&mut prefix).unwrap();
            let header_len = u64::from_le_bytes(prefix);
            assert!(header_len <= 1_000_000);
            let mut header = vec![0; usize::try_from(header_len).unwrap()];
            file.read_exact(&mut header).unwrap();
            let header: Value = serde_json::from_slice(&header).unwrap();
            let tensor = &header["embeddings"];
            assert_eq!(tensor["dtype"], "F32");
            let shape = tensor["shape"].as_array().unwrap();
            assert_eq!(shape.len(), 2);
            let elements = shape[0]
                .as_u64()
                .unwrap()
                .checked_mul(shape[1].as_u64().unwrap())
                .unwrap();
            let start = tensor["data_offsets"][0].as_u64().unwrap();
            let end = tensor["data_offsets"][1].as_u64().unwrap();
            assert_eq!(
                end.checked_sub(start).unwrap(),
                elements.checked_mul(4).unwrap()
            );
            let payload = 8_u64.checked_add(header_len).unwrap();
            assert_eq!(
                payload.checked_add(end).unwrap(),
                file.metadata().unwrap().len()
            );
            assert!(elements > 0);
            let offset = payload.checked_add(start).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            let mut value = [0_u8; 4];
            file.read_exact(&mut value).unwrap();
            assert!(f32::from_le_bytes(value).is_finite());
            let original = value[0];
            value[0] ^= 1;
            assert!(f32::from_le_bytes(value).is_finite());
            replace_byte(path, offset, value[0]);
            (offset, original)
        }
    }

    fn artifacts(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut result = BTreeMap::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for entry in fs::read_dir(directory).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    pending.push(entry.path());
                } else if matches!(
                    entry.path().extension().and_then(|ext| ext.to_str()),
                    Some("fsvi" | "fslx" | "wal")
                ) {
                    result.insert(
                        entry.path().strip_prefix(root).unwrap().to_path_buf(),
                        fs::read(entry.path()).unwrap(),
                    );
                }
            }
        }
        for extension in ["fsvi", "fslx"] {
            assert!(
                result
                    .keys()
                    .any(|path| path.extension().unwrap() == extension)
            );
        }
        eprintln!(
            "watch upgrade artifacts: {:?}",
            result
                .iter()
                .map(|(path, bytes)| (path, bytes.len()))
                .collect::<Vec<_>>()
        );
        result
    }

    fn ranks(fixture: &Fixture) -> BTreeMap<&'static str, Vec<Value>> {
        let mut result = BTreeMap::new();
        let mut fast_producer = Value::Null;
        for mode in ["--fast-only", "--quality-only"] {
            if mode == "--quality-only" && !fixture.quality {
                continue;
            }
            let output = Process::start(fixture.command("search", "json").args([
                "find a relevant document",
                mode,
                "--no-daemon",
                "--no-rerank",
                "--limit",
                "10",
            ]))
            .finish(Duration::from_secs(60));
            assert!(output.status.success(), "{mode}: {output:?}");
            let response: Value = serde_json::from_slice(&output.stdout).unwrap();
            let payload = &response["data"];
            assert_search_hits(payload, &["alpha.md", "beta.md"], "semantic_rank");
            assert!(payload["vector_generation_id"].is_string());
            if mode == "--fast-only" {
                fast_producer = payload["vector_generation_id"].clone();
            } else {
                assert_eq!(payload["skip_reason"], "quality_only");
                assert_ne!(payload["vector_generation_id"], fast_producer);
            }
            let hits = payload["hits"].as_array().unwrap();
            assert_eq!(
                hits.iter()
                    .map(|hit| hit["path"].as_str().unwrap())
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from(["alpha.md", "beta.md"])
            );
            result.insert(
                mode,
                hits.iter()
                    .map(|hit| {
                        assert!(hit["score"].is_number());
                        serde_json::json!([hit["path"], hit["semantic_rank"], hit["score"]])
                    })
                    .collect(),
            );
            eprintln!(
                "watch upgrade ranking mode={mode} producer={} hits={hits:?}",
                payload["vector_generation_id"]
            );
        }
        result
    }

    fn document_batches(output: &Output) -> usize {
        std::str::from_utf8(&output.stderr)
            .unwrap()
            .lines()
            .filter(|line| line.contains("fsfs admitted indexing embedding batch"))
            .inspect(|line| eprintln!("watch document embedding witness: {line}"))
            .count()
    }

    fn watch(fixture: &Fixture, policy: &str) -> Process {
        Process::start(
            fixture
                .command("watch", "jsonl")
                .env(
                    "FRANKENSEARCH_LOG",
                    "frankensearch_fsfs=info,frankensearch_fsfs::indexing=debug",
                )
                .args(["--no-color", "--upgrade-reuse", policy]),
        )
    }

    fn assert_carried(output: &Output, carried: usize) {
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        let records = stderr
            .lines()
            .filter(|line| line.contains("fsfs classified crash-resumable Quill bulk rows"))
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 1, "one startup indexing pass: {stderr}");
        let expected = format!("carried_files={carried}");
        assert!(
            records[0]
                .split_ascii_whitespace()
                .any(|field| field == expected),
            "{stderr}"
        );
        eprintln!("watch carried-file witness: {}", records[0]);
    }

    let models = fs::canonicalize(
        std::env::var_os("FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR")
            .expect("provide verified real models"),
    )
    .unwrap();
    let prior = fs::canonicalize(
        std::env::var_os("FSFS_UPGRADE_REUSE_PRIOR_EXECUTABLE")
            .expect("provide a genuine source-bound prior executable"),
    )
    .unwrap();
    let current = PathBuf::from(env!("CARGO_BIN_EXE_fsfs"));
    let prior_sha = file_sha256(&prior);
    assert_eq!(
        prior_sha,
        std::env::var("FSFS_UPGRADE_REUSE_PRIOR_SHA256")
            .expect("provide the independently verified prior executable SHA-256")
    );
    let current_sha = file_sha256(&current);
    assert_ne!(prior_sha, current_sha);

    // A direct fast-model directory also contains tokenizer.json. Its verified
    // bytes belong to Potion, so the absent optional MiniLM still permits indexing.
    {
        let fixture = Fixture::new(None, false);
        let models = private_models(&fixture, &models);
        let direct = models.join("direct-model");
        fs::rename(models.join("potion-multilingual-128M"), &direct).unwrap();
        let mut settings: FsfsConfig =
            toml::from_str(&fs::read_to_string(&fixture.config).unwrap()).unwrap();
        settings.indexing.model_dir = direct.display().to_string();
        settings.indexing.quality_model = FsfsConfig::default().indexing.quality_model;
        settings.search.fast_only = false;
        fs::write(&fixture.config, settings.to_toml().unwrap()).unwrap();
        let index = || {
            Process::start(
                fixture
                    .command("index", "json")
                    .env("FSFS_COMPLETE_GENERATIONS", "true")
                    .args(["--upgrade-reuse", "compatible-contract"]),
            )
            .finish(Duration::from_secs(180))
        };
        let output = index();
        assert!(output.status.success(), "{output:?}");
        let response: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(response["data"]["indexed_files"], 1);
        assert_eq!(
            response["data"]["vector_generation"]["id"],
            "potion-multilingual-128M"
        );
        assert!(response["data"]["quality_generation"].is_null());
        let pointer = fixture.store.join("FSFS-CURRENT");
        let selection = fs::read(&pointer).unwrap();
        fs::create_dir(direct.join("all-MiniLM-L6-v2")).unwrap();
        let refused = index();
        assert!(!refused.status.success(), "{refused:?}");
        let response: Value = serde_json::from_slice(&refused.stdout).unwrap();
        assert_eq!(response["error"]["code"], "model_not_found", "{refused:?}");
        assert_eq!(fs::read(pointer).unwrap(), selection);
    }

    // A terminal operator can decline downloads once. A verified fast producer
    // permits promotion; a missing required producer must preserve the legacy.
    for (missing_fast, interrupt) in [(false, false), (true, false), (true, true)] {
        let mut fixture = Fixture::new(None, false);
        let private = private_models(&fixture, &models);
        let mut settings: FsfsConfig =
            toml::from_str(&fs::read_to_string(&fixture.config).unwrap()).unwrap();
        settings.indexing.model_dir = private.display().to_string();
        settings.indexing.quality_model = FsfsConfig::default().indexing.quality_model;
        settings.search.fast_only = false;
        fs::write(&fixture.config, settings.to_toml().unwrap()).unwrap();
        fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
        fixture.executable.clone_from(&prior);
        let seed = Process::start(
            fixture
                .command("index", "json")
                .env("FSFS_COMPLETE_GENERATIONS", "false"),
        )
        .finish(Duration::from_secs(180));
        assert!(seed.status.success(), "{seed:?}");
        let summary: Value = serde_json::from_slice(&seed.stdout).unwrap();
        assert_eq!(summary["data"]["semantic_indexed_files"], 2);
        assert_eq!(summary["data"]["generation_complete"], true);
        assert!(!fixture.store.join("FSFS-CURRENT").exists());
        assert!(!private.join("all-MiniLM-L6-v2").exists());
        let old_artifacts = artifacts(&fixture.store);
        let old_receipt = fs::read(fixture.store.join("FSFS-LEGACY-REUSE.json")).unwrap();
        let old_ranks = ranks(&fixture);

        settings.indexing.offline = false;
        let empty_models = fixture.directory.path().join("missing-models");
        if missing_fast {
            fs::create_dir(&empty_models).unwrap();
            settings.indexing.model_dir = empty_models.display().to_string();
        }
        fs::write(&fixture.config, settings.to_toml().unwrap()).unwrap();
        fixture.executable.clone_from(&current);
        let mut invocation = fixture.command("index", "table");
        invocation.args(["--upgrade-reuse", "compatible-contract", "--no-color"]);
        let arguments = std::iter::once(invocation.get_program())
            .chain(invocation.get_args())
            .map(|argument| argument.to_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        // Python supplies actual terminal descriptors to the production binary;
        // no fsfs terminal check or consent policy is replaced by a test hook.
        let mut terminal = Command::new("python3");
        terminal
            .arg("-c")
            .arg(
                r"import errno, fcntl, json, os, pathlib, pty, select, signal, subprocess, sys, time
master, slave = pty.openpty()
# Darwin records the first write in F_GETFL (FWASWRITTEN). All three child
# streams share this slave, so establish its write history before taking the
# baseline. Both comparisons below still cover every bit, including O_NONBLOCK.
if os.write(slave, b'.') != 1 or os.read(master, 1) != b'.':
    raise RuntimeError('could not prime shared terminal')
original_flags = fcntl.fcntl(slave, fcntl.F_GETFL)
terminal_identity = os.fstat(slave)
child = subprocess.Popen(json.loads(sys.argv[1]), stdin=slave, stdout=slave, stderr=slave)
transcript = bytearray()
answers = 0
interrupt = json.loads(sys.argv[2])
private_readers = None
deadline = time.monotonic() + 120
try:
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('terminal index deadline')
        readable, _, _ = select.select([master], [], [], 0.1)
        if not readable:
            if child.poll() is not None:
                break
            continue
        try:
            chunk = os.read(master, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        transcript.extend(chunk)
        if len(transcript) > 1048576:
            raise RuntimeError('unexpected terminal output volume')
        prompts = transcript.count(b'consent>')
        if prompts > 4:
            raise RuntimeError('unexpected consent loop')
        if prompts > answers:
            if fcntl.fcntl(slave, fcntl.F_GETFL) != original_flags:
                raise RuntimeError('consent changed shared stdin flags')
            if sys.platform.startswith('linux'):
                private_readers = 0
                for descriptor in pathlib.Path(f'/proc/{child.pid}/fd').iterdir():
                    if int(descriptor.name) <= 2:
                        continue
                    try:
                        info = descriptor.stat()
                        flags = next(line.split()[1] for line in pathlib.Path(
                            f'/proc/{child.pid}/fdinfo/{descriptor.name}').read_text().splitlines()
                            if line.startswith('flags:'))
                    except FileNotFoundError:
                        continue
                    same_terminal = (info.st_dev, info.st_ino, info.st_rdev) == (
                        terminal_identity.st_dev, terminal_identity.st_ino, terminal_identity.st_rdev)
                    flags = int(flags, 8)
                    if same_terminal and (flags & os.O_ACCMODE) == os.O_RDONLY and flags & os.O_NONBLOCK:
                        private_readers += 1
                if private_readers == 0:
                    raise RuntimeError('consent has no private nonblocking terminal reader')
            if interrupt:
                # A partial affirmative is not permission to acquire a model.
                os.write(master, b'y')
                child.send_signal(signal.SIGINT)
                deadline = time.monotonic() + 5
            else:
                os.write(master, b'n\n' * (prompts - answers))
            answers = prompts
    status = child.wait(timeout=5)
finally:
    if child.poll() is None:
        child.kill()
        child.wait()
    flags_preserved = fcntl.fcntl(slave, fcntl.F_GETFL) == original_flags
    os.close(slave)
    os.close(master)
    if not flags_preserved:
        raise RuntimeError('consent left shared stdin flags changed')
print(json.dumps({'exit_code': status, 'prompts': answers,
                  'stdin_flags_preserved': flags_preserved, 'private_terminal_readers': private_readers,
                  'transcript': transcript.decode('utf-8', errors='replace')}))
",
            )
            .arg(serde_json::to_string(&arguments).unwrap())
            .arg(serde_json::to_string(&interrupt).unwrap())
            .current_dir(invocation.get_current_dir().unwrap());
        for (name, value) in invocation.get_envs() {
            if let Some(value) = value {
                terminal.env(name, value);
            } else {
                terminal.env_remove(name);
            }
        }
        let output = Process::start(&mut terminal).finish(Duration::from_secs(180));
        assert!(output.status.success(), "{output:?}");
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["prompts"], 1, "{result}");
        assert_eq!(result["stdin_flags_preserved"], true, "{result}");
        #[cfg(target_os = "linux")]
        assert!(
            result["private_terminal_readers"].as_u64().unwrap() > 0,
            "{result}"
        );
        let transcript = result["transcript"].as_str().unwrap();
        if missing_fast {
            assert_eq!(
                result["exit_code"],
                if interrupt {
                    frankensearch_fsfs::adapters::cli::exit_code::INTERRUPTED
                } else {
                    frankensearch_fsfs::adapters::cli::exit_code::MODEL_UNAVAILABLE
                },
                "{result}"
            );
            assert!(transcript.contains("Download 2 ("), "{result}");
            assert!(!fixture.store.join("FSFS-CURRENT").exists());
            assert_eq!(fs::read_dir(&empty_models).unwrap().count(), 0);
            settings.indexing.model_dir = private.display().to_string();
            fs::write(&fixture.config, settings.to_toml().unwrap()).unwrap();
        } else {
            assert_eq!(result["exit_code"], 0, "{result}");
            assert!(transcript.contains("Download 1 ("), "{result}");
            assert!(fixture.store.join("FSFS-CURRENT").is_file());
        }
        assert!(!private.join("all-MiniLM-L6-v2").exists());
        for (relative, bytes) in &old_artifacts {
            assert_eq!(fs::read(fixture.store.join(relative)).unwrap(), *bytes);
        }
        assert_eq!(
            fs::read(fixture.store.join("FSFS-LEGACY-REUSE.json")).unwrap(),
            old_receipt
        );
        assert_eq!(ranks(&fixture), old_ranks);
        eprintln!(
            "legacy consent terminal descriptor missing_fast={missing_fast} interrupt={interrupt} private_readers={} stdin_flags_preserved=true",
            result["private_terminal_readers"]
        );
        eprintln!(
            "legacy consent terminal missing_fast={missing_fast} interrupt={interrupt} prompts=1 exit={} fast_hits=2 optional_quality_installed=false",
            result["exit_code"]
        );
    }

    for (quality, legacy_seed) in [(false, false), (false, true), (true, false), (true, true)] {
        for (policy, drift) in [
            ("same-executable", "none"),
            ("compatible-contract", "none"),
            ("compatible-contract", "window"),
            ("compatible-contract", "tier"),
            ("compatible-contract", "query_limit"),
            ("compatible-contract", "query_explain"),
            ("compatible-contract", "query_rerank"),
        ] {
            eprintln!(
                "watch upgrade prior={prior_sha} current={current_sha} quality={quality} legacy_seed={legacy_seed} policy={policy} drift={drift}"
            );
            let mut fixture = Fixture::new(Some(&models), quality);
            if drift == "tier" && !quality {
                // Keep the selected producer fixed across A and B. Only the
                // effective tier-coverage switch changes after the A build.
                let mut settings: FsfsConfig =
                    toml::from_str(&fs::read_to_string(&fixture.config).unwrap()).unwrap();
                settings.indexing.quality_model = FsfsConfig::default().indexing.quality_model;
                fs::write(&fixture.config, settings.to_toml().unwrap()).unwrap();
            }
            fixture.executable.clone_from(&prior);
            let predecessor = if legacy_seed {
                fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
                let output = Process::start(
                    fixture
                        .command("index", "json")
                        .env("FSFS_COMPLETE_GENERATIONS", "false")
                        .env("FRANKENSEARCH_LOG", "frankensearch_fsfs::indexing=debug")
                        .arg("--no-color"),
                )
                .finish(Duration::from_secs(180));
                assert!(output.status.success(), "{output:?}");
                let result: Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(result["data"]["generation_complete"], true);
                assert_eq!(result["data"]["semantic_indexed_files"], 2);
                assert!(!fixture.store.join("FSFS-CURRENT").exists());
                assert!(document_batches(&output) > 0);
                None
            } else {
                let mut process = watch(&fixture, "same-executable");
                let first = wait_for_visible_paths(&fixture, &mut process, &["alpha.md"], None);
                fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
                let second = wait_for_visible_paths(
                    &fixture,
                    &mut process,
                    &["alpha.md", "beta.md"],
                    Some(first.id()),
                );
                assert_ne!(first.id(), second.id());
                let output = process.stop_gracefully();
                assert!(document_batches(&output) > 0);
                Some(second)
            };
            let old_root = predecessor
                .as_ref()
                .map_or(fixture.store.as_path(), PublishedGeneration::path)
                .to_path_buf();
            let old_artifacts = artifacts(&old_root);
            assert_eq!(
                old_artifacts.contains_key(Path::new("vector/quality.fsvi")),
                quality
            );
            let old_evidence_path = old_root.join(if legacy_seed {
                "FSFS-LEGACY-REUSE.json"
            } else {
                "FSFS-BUNDLE.json"
            });
            let old_evidence = fs::read(&old_evidence_path).unwrap();
            let mut expected_artifacts = old_artifacts.clone();
            let mut expected_ranks = ranks(&fixture);
            if drift != "none" {
                let mut settings: FsfsConfig =
                    toml::from_str(&fs::read_to_string(&fixture.config).unwrap()).unwrap();
                let before = serde_json::to_value(&settings).unwrap();
                let (section, field) = match drift {
                    "window" => {
                        let cap = settings.indexing.fast_window_max_per_file;
                        settings.indexing.fast_window_max_per_file = if cap == 1 { 2 } else { 1 };
                        ("indexing", "fast_window_max_per_file")
                    }
                    "tier" => {
                        assert!(!settings.indexing.quality_model.is_empty());
                        assert_eq!(settings.search.fast_only, !quality);
                        settings.search.fast_only = quality;
                        fixture.quality = !quality;
                        ("search", "fast_only")
                    }
                    "query_limit" => {
                        settings.search.default_limit = 1;
                        ("search", "default_limit")
                    }
                    "query_explain" => {
                        settings.search.explain = true;
                        ("search", "explain")
                    }
                    "query_rerank" => {
                        settings.search.rerank = true;
                        ("search", "rerank")
                    }
                    _ => panic!("unexpected upgrade fixture mutation: {drift}"),
                };
                let after = serde_json::to_value(&settings).unwrap();
                assert_ne!(after, before);
                let mut restored = after.clone();
                restored[section][field] = before[section][field].clone();
                assert_eq!(restored, before, "only {section}.{field} may change");
                fs::write(&fixture.config, settings.to_toml().unwrap()).unwrap();
                eprintln!(
                    "watch upgrade drift {section}.{field}: {} -> {}",
                    before[section][field], after[section][field]
                );
            }
            let mut previous = predecessor
                .as_ref()
                .map(|generation| generation.id().to_owned());
            let mut readers = Vec::new();
            fixture.executable.clone_from(&current);
            for restart in [false, true] {
                let mut process = watch(&fixture, policy);
                let selected = wait_for_visible_paths(
                    &fixture,
                    &mut process,
                    &["alpha.md", "beta.md"],
                    previous.as_deref(),
                );
                let actual_ranks = ranks(&fixture);
                process.assert_alive();
                let output = process.stop_gracefully();
                let query_only = matches!(drift, "query_limit" | "query_explain" | "query_rerank");
                let warm =
                    restart || (policy == "compatible-contract" && (drift == "none" || query_only));
                assert_carried(&output, if warm { 2 } else { 0 });
                let batches = document_batches(&output);
                assert_eq!(batches == 0, warm);
                if !warm {
                    let stderr = std::str::from_utf8(&output.stderr).unwrap();
                    let models = stderr
                        .lines()
                        .filter(|line| line.contains("fsfs admitted indexing embedding batch"))
                        .flat_map(str::split_ascii_whitespace)
                        .filter_map(|field| field.strip_prefix("model="))
                        .collect::<BTreeSet<_>>();
                    let mut expected = BTreeSet::from(["\"potion-multilingual-128M\""]);
                    if fixture.quality {
                        expected.insert("\"all-minilm-l6-v2\"");
                    }
                    assert_eq!(models, expected, "cold document embedding tiers: {stderr}");
                }
                let actual_artifacts = artifacts(selected.path());
                assert_eq!(
                    actual_artifacts.contains_key(Path::new("vector/quality.fsvi")),
                    fixture.quality
                );
                if warm {
                    assert_eq!(actual_artifacts, expected_artifacts);
                    assert_eq!(actual_ranks, expected_ranks);
                }
                if legacy_seed {
                    for (path, bytes) in &old_artifacts {
                        assert_eq!(fs::read(old_root.join(path)).unwrap(), *bytes);
                    }
                } else {
                    assert_eq!(artifacts(&old_root), old_artifacts);
                }
                assert_eq!(fs::read(&old_evidence_path).unwrap(), old_evidence);
                expected_artifacts = actual_artifacts;
                expected_ranks = actual_ranks;
                previous = Some(selected.id().to_owned());
                let manifest = fs::read(selected.path().join("FSFS-BUNDLE.json")).unwrap();
                readers.push((selected, manifest));
                eprintln!(
                    "watch upgrade result quality={quality} requested_quality={} legacy_seed={legacy_seed} policy={policy} drift={drift} restart={restart} carried={} document_batches={batches}",
                    fixture.quality,
                    if warm { 2 } else { 0 }
                );
            }
            for (reader, manifest) in readers {
                assert_eq!(
                    fs::read(reader.path().join("FSFS-BUNDLE.json")).unwrap(),
                    manifest
                );
            }
        }
    }

    // Real artifact drift must not borrow the predecessor's verified producer,
    // even when its display ID, configuration, size and mtime still agree.
    for quality in [false, true] {
        for legacy_seed in [false, true] {
            for artifact in ["tokenizer", "weights", "quality-tokenizer"] {
                let quality_drift = artifact == "quality-tokenizer";
                if quality_drift && !quality {
                    continue;
                }
                let mut fixture = Fixture::new(None, quality);
                let private = private_models(&fixture, &models);
                let mut settings: FsfsConfig =
                    toml::from_str(&fs::read_to_string(&fixture.config).unwrap()).unwrap();
                settings.indexing.model_dir = private.display().to_string();
                fs::write(&fixture.config, settings.to_toml().unwrap()).unwrap();
                let configuration = fs::read(&fixture.config).unwrap();
                fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
                fixture.executable.clone_from(&prior);
                let predecessor = if legacy_seed {
                    let output = Process::start(
                        fixture
                            .command("index", "json")
                            .env("FSFS_COMPLETE_GENERATIONS", "false")
                            .env("FRANKENSEARCH_LOG", "frankensearch_fsfs::indexing=debug"),
                    )
                    .finish(Duration::from_secs(180));
                    assert!(output.status.success(), "{output:?}");
                    assert!(document_batches(&output) > 0);
                    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
                    assert_eq!(response["data"]["semantic_indexed_files"], 2);
                    assert_eq!(response["data"]["generation_complete"], true);
                    None
                } else {
                    let mut process = watch(&fixture, "same-executable");
                    let generation = wait_for_visible_paths(
                        &fixture,
                        &mut process,
                        &["alpha.md", "beta.md"],
                        None,
                    );
                    let output = process.stop_gracefully();
                    assert!(document_batches(&output) > 0);
                    Some(generation)
                };
                let old_root = predecessor
                    .as_ref()
                    .map_or(fixture.store.as_path(), PublishedGeneration::path);
                let old_artifacts = artifacts(old_root);
                let old_ranks = ranks(&fixture);
                let evidence_path = old_root.join(if legacy_seed {
                    "FSFS-LEGACY-REUSE.json"
                } else {
                    "FSFS-BUNDLE.json"
                });
                let evidence = fs::read(&evidence_path).unwrap();
                let pointer = fixture.store.join("FSFS-CURRENT");
                let selection = pointer.exists().then(|| fs::read(&pointer).unwrap());
                let (producer, manifest) = if quality_drift {
                    (
                        "all-MiniLM-L6-v2",
                        frankensearch_embed::ModelManifest::minilm_v2(),
                    )
                } else {
                    (
                        "potion-multilingual-128M",
                        frankensearch_embed::ModelManifest::potion_128m(),
                    )
                };
                let model = private.join(producer);
                assert!(frankensearch_embed::is_verification_cached(
                    &manifest, &model
                ));
                let receipt = fs::read(model.join(".verified")).unwrap();
                let filename = if artifact == "weights" {
                    "model.safetensors"
                } else {
                    "tokenizer.json"
                };
                let path = model.join(filename);
                let metadata = fs::metadata(&path).unwrap();
                let original_sha = file_sha256(&path);
                let source_path = models.join(producer).join(filename);
                assert_eq!(file_sha256(&source_path), original_sha);
                let (offset, byte) = drift_model_artifact(&path, artifact);
                fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_times(fs::FileTimes::new().set_modified(metadata.modified().unwrap()))
                    .unwrap();
                let changed = fs::metadata(&path).unwrap();
                assert_eq!(
                    (
                        changed.len(),
                        changed.modified().unwrap(),
                        changed.dev(),
                        changed.ino()
                    ),
                    (
                        metadata.len(),
                        metadata.modified().unwrap(),
                        metadata.dev(),
                        metadata.ino()
                    )
                );
                let drifted_sha = file_sha256(&path);
                assert_ne!(drifted_sha, original_sha);
                assert!(!frankensearch_embed::is_verification_cached(
                    &manifest, &model
                ));
                let error = frankensearch_embed::verify_dir_cached(&manifest, &model).unwrap_err();
                assert!(
                    matches!(&error, frankensearch_core::SearchError::HashMismatch { path: refused, .. } if refused == &path),
                    "{error:?}"
                );
                fixture.executable.clone_from(&current);
                let expected_code = if quality_drift {
                    frankensearch_fsfs::output_schema::OutputErrorCode::MODEL_LOAD_FAILED
                } else {
                    frankensearch_fsfs::output_schema::OutputErrorCode::EMBEDDER_UNAVAILABLE
                };
                if quality_drift {
                    let output = Process::start(
                        fixture
                            .command("index", "json")
                            .env("FSFS_COMPLETE_GENERATIONS", "true")
                            .env("FRANKENSEARCH_LOG", "frankensearch_fsfs::indexing=debug")
                            .args(["--upgrade-reuse", "compatible-contract"]),
                    )
                    .finish(Duration::from_secs(180));
                    assert!(!output.status.success(), "{output:?}");
                    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
                    assert_eq!(response["error"]["code"], expected_code, "{output:?}");
                    assert_eq!(document_batches(&output), 0);
                    assert_eq!(
                        pointer.exists().then(|| fs::read(&pointer).unwrap()),
                        selection
                    );
                    if legacy_seed {
                        assert!(!fixture.store.join("generations").try_exists().unwrap());
                    }
                }
                let output =
                    watch(&fixture, "compatible-contract").finish(Duration::from_secs(180));
                assert!(!output.status.success(), "{output:?}");
                assert_eq!(document_batches(&output), 0);
                let responses = std::str::from_utf8(&output.stdout)
                    .unwrap()
                    .lines()
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .collect::<Vec<_>>();
                assert!(
                    responses.iter().any(|response| response["ok"] == false
                        && response["error"]["code"] == expected_code),
                    "{output:?}"
                );
                assert!(
                    !responses
                        .iter()
                        .any(|response| response["data"]["publication"] == "durable")
                );
                assert_eq!(
                    pointer.exists().then(|| fs::read(&pointer).unwrap()),
                    selection
                );
                if legacy_seed {
                    assert!(!fixture.store.join("generations").try_exists().unwrap());
                }
                assert_eq!(fs::read(&fixture.config).unwrap(), configuration);
                assert_eq!(fs::read(model.join(".verified")).unwrap(), receipt);
                for (relative, bytes) in &old_artifacts {
                    assert_eq!(fs::read(old_root.join(relative)).unwrap(), *bytes);
                }
                assert_eq!(fs::read(&evidence_path).unwrap(), evidence);
                eprintln!(
                    "producer drift refused quality={quality} legacy_seed={legacy_seed} artifact={artifact} original={original_sha} drifted={drifted_sha} size={} mtime_unchanged=true",
                    metadata.len()
                );

                replace_byte(&path, offset, byte);
                assert_eq!(file_sha256(&path), original_sha);
                assert_eq!(
                    file_sha256(&source_path),
                    original_sha,
                    "original cache remains untouched"
                );
                frankensearch_embed::verify_dir_and_record(&manifest, &model).unwrap();
                assert_eq!(ranks(&fixture), old_ranks);
                let mut retry = watch(&fixture, "compatible-contract");
                let recovered = wait_for_visible_paths(
                    &fixture,
                    &mut retry,
                    &["alpha.md", "beta.md"],
                    predecessor.as_ref().map(PublishedGeneration::id),
                );
                let output = retry.stop_gracefully();
                assert_carried(&output, 2);
                assert_eq!(document_batches(&output), 0);
                assert_eq!(artifacts(recovered.path()), old_artifacts);
                assert_eq!(ranks(&fixture), old_ranks);
                for (relative, bytes) in &old_artifacts {
                    assert_eq!(fs::read(old_root.join(relative)).unwrap(), *bytes);
                }
                assert_eq!(fs::read(&evidence_path).unwrap(), evidence);
                eprintln!(
                    "producer drift recovered quality={quality} legacy_seed={legacy_seed} artifact={artifact} carried=2 document_batches=0"
                );
            }
        }
    }

    // Interrupt real document work after a genuine executable upgrade. Model
    // health probes do not emit this indexing-only event. No timing sleep or
    // destructor kill is evidence that cancellation reached a build.
    let pending = [
        "pending-00.md",
        "pending-01.md",
        "pending-02.md",
        "pending-03.md",
        "pending-04.md",
        "pending-05.md",
        "pending-06.md",
        "pending-07.md",
    ];
    eprintln!("watch interrupt prior={prior_sha} current={current_sha}");
    let mut fixture = Fixture::new(Some(&models), true);
    fixture.executable.clone_from(&prior);
    fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
    let mut seed = watch(&fixture, "same-executable");
    let original = wait_for_visible_paths(&fixture, &mut seed, &["alpha.md", "beta.md"], None);
    seed.stop_gracefully();
    let original_artifacts = artifacts(original.path());
    assert!(original_artifacts.contains_key(Path::new("vector/quality.fsvi")));
    let pointer = fs::read(fixture.store.join("FSFS-CURRENT")).unwrap();
    let manifest = fs::read(original.path().join("FSFS-BUNDLE.json")).unwrap();
    fixture.executable.clone_from(&current);
    let original_ranks = ranks(&fixture);
    let paragraph = "A relevant document explains transactional indexing, durable snapshots, cancellation, file discovery, semantic retrieval, lexical ranking, and stable search results.\n";
    for (ordinal, path) in pending.iter().enumerate() {
        fs::write(
            fixture.source.join(path),
            format!("Pending document {ordinal}\n{}", paragraph.repeat(256)),
        )
        .unwrap();
    }
    let reader_runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .unwrap();
    let cx = reader_runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
    let store = CompleteGenerationStore::open(&cx, &fixture.store).unwrap();
    for (signal_flag, termination_signal) in [("-INT", None), ("-KILL", Some(9))] {
        let mut interrupted = watch(&fixture, "compatible-contract");
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            interrupted.assert_alive();
            let stderr = fs::read_to_string(interrupted.capture.path().join("stderr")).unwrap();
            if let Some(witness) = stderr
                .split_inclusive('\n')
                .filter(|line| line.ends_with('\n'))
                .find(|line| {
                    line.contains("fsfs admitted indexing embedding batch")
                        && line.contains("model=\"all-minilm-l6-v2\"")
                })
            {
                assert!(
                    witness
                        .split_ascii_whitespace()
                        .any(|field| field == "inputs=8")
                );
                assert_eq!(store.active(&cx).unwrap(), Some(original.clone()));
                eprintln!("watch interrupt document batch signal={signal_flag}: {witness}");
                break;
            }
            assert!(
                Instant::now() < deadline,
                "no quality document batch: {stderr}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let signal = Command::new("kill")
            .args([signal_flag, &interrupted.child.id().to_string()])
            .status()
            .expect("stop own indexing watcher");
        assert!(signal.success());
        let stopped = interrupted.finish(Duration::from_secs(30));
        if let Some(expected) = termination_signal {
            assert!(!stopped.status.success(), "{stopped:?}");
            assert_eq!(stopped.status.signal(), Some(expected), "{stopped:?}");
        } else {
            assert!(stopped.status.success(), "{stopped:?}");
            assert!(
                std::str::from_utf8(&stopped.stderr)
                    .unwrap()
                    .lines()
                    .any(
                        |line| line.contains("complete-generation command stopped on request")
                            && line.contains("Signal(2)")
                    ),
                "{stopped:?}"
            );
        }
        for line in stopped
            .stdout
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
        {
            let receipt: Value = serde_json::from_slice(line).unwrap();
            assert_ne!(receipt["data"]["publication"], "durable");
        }
        assert_eq!(store.active(&cx).unwrap(), Some(original.clone()));
        assert_eq!(
            fs::read(fixture.store.join("FSFS-CURRENT")).unwrap(),
            pointer
        );
        assert_eq!(artifacts(original.path()), original_artifacts);
        assert_eq!(
            fs::read(original.path().join("FSFS-BUNDLE.json")).unwrap(),
            manifest
        );
        assert_eq!(ranks(&fixture), original_ranks);
        eprintln!(
            "watch interrupt preserved original={} signal={signal_flag} pending=8 status={:?}",
            original.id(),
            stopped.status
        );
    }

    let mut retry = watch(&fixture, "compatible-contract");
    let mut expected = vec!["alpha.md", "beta.md"];
    expected.extend(pending);
    let recovered = wait_for_visible_paths(&fixture, &mut retry, &expected, Some(original.id()));
    let recovered_output = retry.stop_gracefully();
    assert!(document_batches(&recovered_output) > 0);
    assert_eq!(artifacts(original.path()), original_artifacts);
    assert_eq!(
        fs::read(original.path().join("FSFS-BUNDLE.json")).unwrap(),
        manifest
    );
    assert_eq!(store.active(&cx).unwrap(), Some(recovered.clone()));
    eprintln!(
        "watch interrupt recovered generation={} visible_documents=10",
        recovered.id()
    );
}

#[cfg(feature = "semantic-support")]
#[test]
#[ignore = "requires FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR with verified Potion and MiniLM caches"]
fn complete_watch_binary_keeps_search_available() {
    let models = fs::canonicalize(
        std::env::var_os("FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR")
            .expect("provide a verified semantic model-cache root"),
    )
    .unwrap();
    for legacy_seed in [false, true] {
        for index_watch in [false, true] {
            for quality in [false, true] {
                assert_default_complete_watch_lifecycle(&models, legacy_seed, index_watch, quality);
            }
        }
    }
}

#[cfg(all(feature = "semantic-support", feature = "rerank"))]
#[test]
#[allow(clippy::too_many_lines)]
#[ignore = "requires FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR with verified native MiniLM artifacts"]
fn complete_quality_only_watch_binary_updates_and_restarts_without_a_fast_model() {
    let models = fs::canonicalize(
        std::env::var_os("FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR")
            .expect("provide a verified native MiniLM model-cache root"),
    )
    .unwrap();
    for index_watch in [false, true] {
        let mut fixture = Fixture::new(Some(&models), true);
        fixture.configure_quality_only(&models);
        let mut command = fixture.command(if index_watch { "index" } else { "watch" }, "jsonl");
        command.arg("--quality-only");
        if index_watch {
            command.arg("--watch");
        }
        let mut watching = Process::start(&mut command);
        let first = wait_for_visible_paths(&fixture, &mut watching, &["alpha.md"], None);
        let first_vectors = fs::read(first.path().join("vector/quality.fsvi")).unwrap();
        let first_manifest = fs::read(first.path().join("FSFS-BUNDLE.json")).unwrap();
        fs::write(
            fixture.source.join("alpha.md"),
            "sharedtoken revised alpha document",
        )
        .unwrap();
        fs::write(
            fixture.source.join("beta.md"),
            "sharedtoken new beta document",
        )
        .unwrap();
        let edited = wait_for_visible_paths(
            &fixture,
            &mut watching,
            &["alpha.md", "beta.md"],
            Some(first.id()),
        );
        fs::rename(
            fixture.source.join("alpha.md"),
            fixture.source.join("renamed.md"),
        )
        .unwrap();
        fs::rename(
            fixture.source.join("beta.md"),
            fixture.directory.path().join("removed-beta.md"),
        )
        .unwrap();
        let renamed =
            wait_for_visible_paths(&fixture, &mut watching, &["renamed.md"], Some(edited.id()));
        assert_eq!(
            fs::read(first.path().join("vector/quality.fsvi")).unwrap(),
            first_vectors
        );
        assert_eq!(
            fs::read(first.path().join("FSFS-BUNDLE.json")).unwrap(),
            first_manifest
        );
        watching.stop_gracefully();

        // Fresh executable, same standalone quality topology and native producer.
        let mut restarted = Process::start(fixture.command("watch", "jsonl").arg("--quality-only"));
        let warm = wait_for_visible_paths(
            &fixture,
            &mut restarted,
            &["renamed.md"],
            Some(renamed.id()),
        );
        restarted.stop_gracefully();
        let selection = fs::read(fixture.store.join("FSFS-CURRENT")).unwrap();
        let vectors = fs::read(warm.path().join("vector/quality.fsvi")).unwrap();
        let missing = fixture.directory.path().join("missing-models");
        fixture.configure_quality_only(&missing);
        let refused = Process::start(fixture.command("watch", "jsonl").arg("--quality-only"))
            .finish(Duration::from_secs(60));
        assert!(
            !refused.status.success(),
            "required quality cannot degrade into fast or lexical output"
        );
        assert_eq!(
            fs::read(fixture.store.join("FSFS-CURRENT")).unwrap(),
            selection
        );
        assert_eq!(
            fs::read(warm.path().join("vector/quality.fsvi")).unwrap(),
            vectors
        );
    }
}

#[cfg(feature = "semantic-support")]
fn assert_default_complete_watch_lifecycle(
    models: &Path,
    legacy_seed: bool,
    index_watch: bool,
    quality: bool,
) {
    eprintln!(
        "default complete-watch lifecycle: legacy_seed={legacy_seed} index_watch={index_watch} quality={quality}"
    );
    let fixture = Fixture::new(Some(models), quality);
    // Keep cache canaries local to this lifecycle case; refusal probes retain
    // the shared fixture's original source bytes.
    fs::write(
        fixture.source.join("alpha.md"),
        "sharedtoken alpha document priorcontenttoken",
    )
    .unwrap();
    // A completed legacy root must promote on ordinary watch startup, without
    // an opt-in flag. Keep the explicit opt-out's legacy behavior observable.
    if legacy_seed {
        let legacy = Process::start(
            fixture
                .command("index", "json")
                .env("FSFS_COMPLETE_GENERATIONS", "0"),
        )
        .finish(Duration::from_secs(180));
        assert!(
            legacy.status.success(),
            "{}",
            String::from_utf8_lossy(&legacy.stderr)
        );
        assert!(!fixture.store.join("FSFS-CURRENT").exists());
        let sentinel: Value =
            serde_json::from_slice(&fs::read(fixture.store.join("index_sentinel.json")).unwrap())
                .unwrap();
        assert_eq!(sentinel["generation_complete"], true);
    }
    let mut command = fixture.command(if index_watch { "index" } else { "watch" }, "jsonl");
    if index_watch {
        command.arg("--watch");
    }
    let mut watch = Process::start(&mut command);
    let first = wait_for_visible_paths(&fixture, &mut watch, &["alpha.md"], None);
    let original_manifest = fs::read(first.path().join("FSFS-BUNDLE.json")).unwrap();
    let socket = fixture.store.join("fsfs-query.sock");
    let mut daemon = Process::start(
        fixture
            .command("daemon", "json")
            .args(["--idle-timeout-ms", "0"]),
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    while !socket.exists() {
        watch.assert_alive();
        daemon.assert_alive();
        assert!(Instant::now() < deadline, "daemon never opened its socket");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_daemon_visible_paths(&fixture, &mut watch, &mut daemon, &["alpha.md"]);
    let edited = assert_content_edit_visible(&fixture, &mut watch, &mut daemon, first.id());
    fs::write(fixture.source.join("beta.md"), "sharedtoken beta document").unwrap();
    let second = wait_for_visible_paths(
        &fixture,
        &mut watch,
        &["alpha.md", "beta.md"],
        Some(edited.id()),
    );
    assert_ne!(first, second);
    drop(edited);
    assert_daemon_visible_paths(&fixture, &mut watch, &mut daemon, &["alpha.md", "beta.md"]);
    fs::rename(
        fixture.source.join("alpha.md"),
        fixture.source.join("renamed.md"),
    )
    .unwrap();
    let renamed = wait_for_visible_paths(
        &fixture,
        &mut watch,
        &["renamed.md", "beta.md"],
        Some(second.id()),
    );
    assert_daemon_visible_paths(
        &fixture,
        &mut watch,
        &mut daemon,
        &["renamed.md", "beta.md"],
    );
    fs::remove_file(fixture.source.join("beta.md")).unwrap();
    let deleted = wait_for_visible_paths(&fixture, &mut watch, &["renamed.md"], Some(renamed.id()));
    drop(renamed);
    assert_daemon_visible_paths(&fixture, &mut watch, &mut daemon, &["renamed.md"]);
    watch.assert_alive();
    assert_eq!(
        fs::read(first.path().join("FSFS-BUNDLE.json")).unwrap(),
        original_manifest
    );
    assert!(second.path().is_dir());

    // The same default collector must reclaim old images after their actual
    // readers release them. Disabling collection cannot satisfy this test.
    let first_path = first.path().to_path_buf();
    let second_path = second.path().to_path_buf();
    drop(first);
    drop(second);
    fs::write(
        fixture.source.join("gamma.md"),
        "sharedtoken gamma document",
    )
    .unwrap();
    let _after_release = wait_for_visible_paths(
        &fixture,
        &mut watch,
        &["renamed.md", "gamma.md"],
        Some(deleted.id()),
    );
    assert_daemon_visible_paths(
        &fixture,
        &mut watch,
        &mut daemon,
        &["renamed.md", "gamma.md"],
    );
    for path in [&first_path, &second_path] {
        assert_eq!(
            fs::symlink_metadata(path).unwrap_err().kind(),
            std::io::ErrorKind::NotFound,
            "released old generation must be collected: {}",
            path.display()
        );
    }
    let stopped = Process::start(fixture.command("daemon", "json").arg("--stop"))
        .finish(Duration::from_secs(10));
    assert!(
        stopped.status.success(),
        "{}",
        String::from_utf8_lossy(&stopped.stderr)
    );
    let daemon_exit = daemon.finish(Duration::from_secs(10));
    assert!(
        daemon_exit.status.success(),
        "{}",
        String::from_utf8_lossy(&daemon_exit.stderr)
    );
    watch.assert_alive();
    // Drop bounds watcher cleanup; it is not a graceful-SIGINT assertion.
}

#[cfg(feature = "semantic-support")]
fn assert_streamed_query(socket: &Path) {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let mut peer = UnixStream::connect(socket).unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    peer.set_write_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    peer.write_all(b"{\"query\":\"sharedtoken\",\"stream\":true}\n")
        .unwrap();
    let mut reader = BufReader::new(peer);
    let mut frames = Vec::new();
    loop {
        assert!(frames.len() < 256, "missing stream terminal");
        let mut line = String::new();
        assert_ne!(
            reader.read_line(&mut line).unwrap(),
            0,
            "premature stream EOF"
        );
        let frame: Value = serde_json::from_str(&line).unwrap();
        let _: frankensearch_fsfs::stream_protocol::StreamFrame<
            frankensearch_fsfs::output_schema::SearchHitPayload,
        > = serde_json::from_value(frame.clone()).unwrap();
        let terminal = frame["event"] == "terminal";
        frames.push(frame);
        if terminal {
            break;
        }
    }
    assert_eq!(frames.first().unwrap()["event"], "started");
    assert_eq!(frames.last().unwrap()["payload"]["status"], "completed");
    assert_eq!(
        frames
            .iter()
            .filter(|frame| frame["event"] == "result")
            .count(),
        1
    );
    for pair in frames.windows(2) {
        assert_eq!(
            pair[1]["seq"].as_u64().unwrap(),
            pair[0]["seq"].as_u64().unwrap() + 1
        );
    }
}

#[cfg(feature = "semantic-support")]
#[test]
#[ignore = "requires FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR with a verified Potion cache"]
fn complete_daemon_binary_serves_and_stops_with_corrupt_selection() {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let models = fs::canonicalize(
        std::env::var_os("FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR")
            .expect("provide a verified semantic model-cache root"),
    )
    .unwrap();
    let fixture = Fixture::new(Some(&models), false);
    let indexed =
        Process::start(&mut fixture.command("index", "json")).finish(Duration::from_secs(180));
    assert!(
        indexed.status.success(),
        "{}",
        String::from_utf8_lossy(&indexed.stderr)
    );
    let socket = fixture.store.join("fsfs-query.sock");
    let mut daemon = Process::start(
        fixture
            .command("daemon", "json")
            .args(["--idle-timeout-ms", "0"]),
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    while !socket.exists() {
        daemon.assert_alive();
        assert!(
            Instant::now() < deadline,
            "daemon never opened its owned socket"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut peer = UnixStream::connect(&socket).unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    peer.set_write_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    peer.write_all(b"sharedtoken\n").unwrap();
    let mut response = String::new();
    BufReader::new(peer).read_line(&mut response).unwrap();
    let response: Value = serde_json::from_str(&response).unwrap();
    assert_eq!(response["ok"], true, "{response}");
    let phases = response["payloads"].as_array().unwrap();
    let hits = phases.last().unwrap()["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0]["path"].as_str().unwrap().ends_with("alpha.md"));
    assert_streamed_query(&socket);

    let forwarded = Process::start(fixture.command("search", "json").args([
        "sharedtoken",
        "--daemon",
        "--limit",
        "1",
    ]))
    .finish(Duration::from_secs(60));
    assert!(
        forwarded.status.success(),
        "forwarded CLI search failed: {}",
        String::from_utf8_lossy(&forwarded.stderr)
    );
    let forwarded: Value = serde_json::from_slice(&forwarded.stdout).unwrap();
    assert_eq!(forwarded["ok"], true, "{forwarded}");
    let forwarded_hits = forwarded["data"]["hits"].as_array().unwrap();
    assert_eq!(forwarded_hits.len(), 1);
    assert!(
        forwarded_hits[0]["path"]
            .as_str()
            .unwrap()
            .ends_with("alpha.md")
    );

    for format in ["jsonl", "toon"] {
        for transport in ["--no-daemon", "--daemon"] {
            let streamed = Process::start(fixture.command("search", format).args([
                "sharedtoken",
                transport,
                "--stream",
                "--limit",
                "1",
            ]))
            .finish(Duration::from_secs(60));
            assert!(
                streamed.status.success(),
                "{}",
                String::from_utf8_lossy(&streamed.stderr)
            );
            let frames = complete_stream_frames(&streamed, format);
            assert_eq!(frames.first().unwrap()["event"], "started");
            assert_eq!(frames.last().unwrap()["payload"]["status"], "completed");
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| frame["event"] == "result")
                    .count(),
                1
            );
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| frame["event"] == "terminal")
                    .count(),
                1
            );
        }
    }

    // Stop dispatch must not first admit the selected generation or its models.
    let pointer = fixture
        .store
        .join(frankensearch_fsfs::generation_store::COMPLETE_GENERATION_POINTER);
    fs::write(&pointer, b"corrupt selection for stop dispatch").unwrap();
    let refused = Process::start(
        fixture
            .command("search", "json")
            .args(["sharedtoken", "--daemon"]),
    )
    .finish(Duration::from_secs(30));
    assert!(!refused.status.success(), "corrupt selection was served");
    assert_eq!(
        fs::read(&pointer).unwrap(),
        b"corrupt selection for stop dispatch"
    );
    for format in ["jsonl", "toon"] {
        for transport in ["--no-daemon", "--daemon"] {
            let refused = Process::start(fixture.command("search", format).args([
                "sharedtoken",
                transport,
                "--stream",
            ]))
            .finish(Duration::from_secs(30));
            assert!(!refused.status.success());
            let frames = complete_stream_frames(&refused, format);
            assert_eq!(frames.len(), 2, "one Started and final failure: {frames:?}");
            assert_eq!(frames[0]["event"], "started");
            assert_ne!(frames[0]["payload"]["semantic_admitted"], true);
            assert_eq!(frames[1]["event"], "terminal");
            assert_eq!(frames[1]["payload"]["status"], "failed");
            assert_eq!(
                frames[1]["payload"]["exit_code"],
                refused.status.code().unwrap()
            );
            assert_eq!(
                fs::read(&pointer).unwrap(),
                b"corrupt selection for stop dispatch"
            );
        }
    }
    daemon.assert_alive();
    let stopped = Process::start(fixture.command("daemon", "json").arg("--stop"))
        .finish(Duration::from_secs(10));
    assert!(
        stopped.status.success(),
        "{}",
        String::from_utf8_lossy(&stopped.stderr)
    );
    let exited = daemon.finish(Duration::from_secs(10));
    assert!(
        exited.status.success(),
        "{}",
        String::from_utf8_lossy(&exited.stderr)
    );
    assert!(!socket.exists());
    assert_eq!(
        fs::read(pointer).unwrap(),
        b"corrupt selection for stop dispatch"
    );
}
