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
    #[cfg(feature = "semantic-support")]
    quality: bool,
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
            #[cfg(feature = "semantic-support")]
            quality,
        }
    }

    fn command(&self, command: &str, format: &str) -> Command {
        let mut child = Command::new(env!("CARGO_BIN_EXE_fsfs"));
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
                    "--fast-only",
                    "--no-daemon",
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
                    if fixture.quality {
                        let quality = Process::start(fixture.command("search", "json").args([
                            "find a relevant document",
                            "--quality-only",
                            "--no-daemon",
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
