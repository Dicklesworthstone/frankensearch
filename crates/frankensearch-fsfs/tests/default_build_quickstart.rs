//! Executable contract for the loader-capable default `fsfs` build.
//!
//! The always-run lane proves that an offline machine with no models fails
//! closed and explains the explicit provisioning step. The ignored lane is the
//! mock-free release/CI check: it verifies the pinned model bytes, indexes a
//! real corpus with the production binary, and searches the durable generation.

#![forbid(unsafe_code)]

#[cfg(not(feature = "embedded-models"))]
mod loader_only {
    use std::ffi::OsStr;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};
    use std::process::{Command, ExitStatus, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};

    use frankensearch_embed::model_manifest::{ModelManifest, is_verification_cached};
    use frankensearch_index::VectorIndex;
    use serde_json::Value;

    const FAILURE_TIMEOUT: Duration = Duration::from_secs(30);
    const QUICKSTART_TIMEOUT: Duration = Duration::from_secs(240);
    const QUICKSTART_DOCUMENT_COUNT: usize = 10;
    const RETRY_DOCUMENT: &str = "Recover transient network failures with exponential backoff, bounded retries, and random jitter.";
    const SEMANTIC_PARAPHRASES: [&str; 2] = [
        "staggered reconnects after brief outages",
        "delayed reattempts following temporary disruptions",
    ];
    const SEARCH_LIMIT: &str = "3";

    #[derive(Debug)]
    struct CommandOutcome {
        status: ExitStatus,
        elapsed: Duration,
        timed_out: bool,
        stdout: String,
        stderr: String,
    }

    fn fsfs_binary() -> PathBuf {
        std::env::var_os("FSFS_E2E_BINARY").map_or_else(
            || PathBuf::from(env!("CARGO_BIN_EXE_fsfs")),
            |value| {
                let path = PathBuf::from(value);
                assert!(
                    path.is_absolute(),
                    "FSFS_E2E_BINARY must identify an absolute installed-binary path: {}",
                    path.display()
                );
                assert!(
                    path.is_file(),
                    "FSFS_E2E_BINARY does not identify a regular file: {}",
                    path.display()
                );
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;

                    let mode = fs::metadata(&path)
                        .expect("read FSFS_E2E_BINARY metadata")
                        .permissions()
                        .mode();
                    assert_ne!(
                        mode & 0o111,
                        0,
                        "FSFS_E2E_BINARY is not executable: {}",
                        path.display()
                    );
                }
                path
            },
        )
    }

    impl CommandOutcome {
        fn combined_output(&self) -> String {
            format!("{}\n{}", self.stdout, self.stderr)
        }
    }

    fn log_binary_profile(lane: &str) {
        let verified_executable = fsfs_binary();
        let harness_profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        let binary_origin = if std::env::var_os("FSFS_E2E_BINARY").is_some() {
            "explicit-installed-override"
        } else {
            "cargo-test-target"
        };
        eprintln!(
            "[default-build-e2e] stage=stock-default-contract event=start lane={lane} binary={} binary_origin={binary_origin} harness_profile={harness_profile} semantic_loaders={} embedded_models={}",
            verified_executable.display(),
            cfg!(feature = "semantic-loaders"),
            cfg!(feature = "embedded-models")
        );
    }

    #[derive(Debug)]
    struct IsolatedFsfs {
        home: PathBuf,
        xdg_config: PathBuf,
        xdg_cache: PathBuf,
        xdg_data: PathBuf,
        model_root: PathBuf,
        log_root: PathBuf,
    }

    impl IsolatedFsfs {
        fn new(root: &Path, model_root: PathBuf) -> Self {
            let home = root.join("home");
            let xdg_config = root.join("xdg-config");
            let xdg_cache = root.join("xdg-cache");
            let xdg_data = root.join("xdg-data");
            let log_root = root.join("logs");
            for path in [
                &home,
                &xdg_config,
                &xdg_cache,
                &xdg_data,
                &model_root,
                &log_root,
            ] {
                fs::create_dir_all(path).expect("create isolated fsfs test directory");
            }

            Self {
                home,
                xdg_config,
                xdg_cache,
                xdg_data,
                model_root,
                log_root,
            }
        }

        fn run<I, S>(&self, cwd: &Path, label: &str, args: I, timeout: Duration) -> CommandOutcome
        where
            I: IntoIterator<Item = S>,
            S: AsRef<OsStr>,
        {
            self.run_with_env(cwd, label, args, timeout, &[])
        }

        fn command(&self, cwd: &Path) -> Command {
            let mut command = Command::new(fsfs_binary());
            command
                .current_dir(cwd)
                .env("HOME", &self.home)
                .env("XDG_CONFIG_HOME", &self.xdg_config)
                .env("XDG_CACHE_HOME", &self.xdg_cache)
                .env("XDG_DATA_HOME", &self.xdg_data)
                .env("FRANKENSEARCH_MODEL_DIR", &self.model_root)
                .env("FRANKENSEARCH_OFFLINE", "1")
                .env("FRANKENSEARCH_ALLOW_DOWNLOAD", "0")
                .env("RUST_LOG", "warn")
                .env("NO_COLOR", "1")
                .env("RAYON_NUM_THREADS", "1")
                .env_remove("FSFS_CONFIG")
                .env_remove("FRANKENSEARCH_CONFIG")
                .env_remove("FSFS_INDEX_DIR")
                .env_remove("FRANKENSEARCH_INDEX_DIR")
                .env_remove("FSFS_STORAGE_INDEX_DIR")
                .env_remove("FRANKENSEARCH_STORAGE_INDEX_DIR")
                .env_remove("HF_HOME")
                .env_remove("HUGGINGFACE_HUB_CACHE");
            command
        }

        fn run_with_env<I, S>(
            &self,
            cwd: &Path,
            label: &str,
            args: I,
            timeout: Duration,
            extra_env: &[(&str, &str)],
        ) -> CommandOutcome
        where
            I: IntoIterator<Item = S>,
            S: AsRef<OsStr>,
        {
            let verified_executable = fsfs_binary();
            let stdout_path = self.log_root.join(format!("{label}.stdout.log"));
            let stderr_path = self.log_root.join(format!("{label}.stderr.log"));
            let stdout_file = File::create(&stdout_path).expect("create subprocess stdout log");
            let stderr_file = File::create(&stderr_path).expect("create subprocess stderr log");

            let mut command = self.command(cwd);
            command
                .args(args)
                .stdout(Stdio::from(stdout_file))
                .stderr(Stdio::from(stderr_file));
            for (key, value) in extra_env {
                command.env(key, value);
            }

            eprintln!(
                "[default-build-e2e] stage={label} event=spawn binary={} model_root={} timeout_ms={}",
                verified_executable.display(),
                self.model_root.display(),
                timeout.as_millis()
            );
            let started = Instant::now();
            let mut child = command.spawn().expect("spawn production fsfs binary");
            let (status, timed_out) = loop {
                match child.try_wait().expect("poll production fsfs binary") {
                    Some(status) => break (status, false),
                    None if started.elapsed() >= timeout => {
                        child.kill().expect("terminate timed-out fsfs process");
                        let status = child.wait().expect("reap timed-out fsfs process");
                        break (status, true);
                    }
                    None => thread::sleep(Duration::from_millis(25)),
                }
            };
            let elapsed = started.elapsed();
            let stdout = fs::read_to_string(&stdout_path).expect("read subprocess stdout log");
            let stderr = fs::read_to_string(&stderr_path).expect("read subprocess stderr log");
            eprintln!(
                "[default-build-e2e] stage={label} event=exit status={:?} timed_out={} elapsed_ms={}\n[default-build-e2e] stdout:\n{}\n[default-build-e2e] stderr:\n{}",
                status.code(),
                timed_out,
                elapsed.as_millis(),
                stdout,
                stderr
            );

            CommandOutcome {
                status,
                elapsed,
                timed_out,
                stdout,
                stderr,
            }
        }
    }

    fn assert_finished_successfully(label: &str, outcome: &CommandOutcome) {
        assert!(
            !outcome.timed_out,
            "{label} exceeded its {:?} wall-clock budget; stdout:\n{}\nstderr:\n{}",
            outcome.elapsed, outcome.stdout, outcome.stderr
        );
        assert!(
            outcome.status.success(),
            "{label} failed with status {:?}; stdout:\n{}\nstderr:\n{}",
            outcome.status.code(),
            outcome.stdout,
            outcome.stderr
        );
    }

    fn parse_success_envelope(label: &str, outcome: &CommandOutcome) -> Value {
        assert_finished_successfully(label, outcome);
        let envelope: Value = serde_json::from_str(&outcome.stdout).unwrap_or_else(|error| {
            eprintln!(
                "[default-build-e2e] stage={label} event=envelope-parse-error error={error} stdout:\n{}\nstderr:\n{}",
                outcome.stdout, outcome.stderr
            );
            Value::Null
        });
        assert_eq!(
            envelope.get("ok").and_then(Value::as_bool),
            Some(true),
            "{label} envelope must report success: {envelope}"
        );
        envelope
    }

    fn model_status<'a>(envelope: &'a Value, tier: &str) -> &'a Value {
        envelope
            .pointer("/data/models")
            .and_then(Value::as_array)
            .and_then(|models| {
                models
                    .iter()
                    .find(|model| model.get("tier").and_then(Value::as_str) == Some(tier))
            })
            .unwrap_or_else(|| {
                eprintln!(
                    "[default-build-e2e] stage=status event=missing-model-tier tier={tier} envelope={envelope}"
                );
                envelope
            })
    }

    fn assert_index_completion(envelope: &Value, sentinel: &Value, count: usize, format: &str) {
        assert_eq!(envelope["ok"], true);
        assert_eq!(envelope["meta"]["command"], "index");
        assert_eq!(envelope["meta"]["format"], format);
        assert!(envelope["meta"]["duration_ms"].is_u64());
        assert_index_data(&envelope["data"], sentinel, count);
    }

    fn assert_index_data(data: &Value, sentinel: &Value, count: usize) {
        for (field, value) in sentinel.as_object().expect("sentinel object") {
            assert_eq!(&data[field], value, "published field {field}");
        }
        assert_eq!(data["indexed_files"], count);
        assert_eq!(data["semantic_indexed_files"], count);
        assert_eq!(data["generation_complete"], true);
        assert_eq!(data["semantic_deferred_files"], 0);
        assert_eq!(data["embedding_failures"], 0);
        assert_eq!(data["vector_generation"]["is_hash_control"], false);
        assert!(
            data["index_size_bytes"]
                .as_u64()
                .is_some_and(|size| size > 0)
        );
    }

    fn verify_index_output_formats(fsfs: &IsolatedFsfs, root: &Path, corpus: &Path, index: &Path) {
        for format in ["jsonl", "table", "toon", "csv"] {
            let outcome = fsfs.run(
                root,
                &format!("index-format-{format}"),
                [
                    "index",
                    corpus.to_str().unwrap(),
                    "--index-dir",
                    index.to_str().unwrap(),
                    "--format",
                    format,
                ],
                QUICKSTART_TIMEOUT,
            );
            assert_finished_successfully(format, &outcome);
            let sentinel: Value =
                serde_json::from_slice(&fs::read(index.join("index_sentinel.json")).unwrap())
                    .unwrap();
            if format == "table" {
                assert!(outcome.stdout.starts_with("Discovered 10 file(s)"));
                assert!(
                    outcome
                        .stdout
                        .contains("Indexed 10 file(s) (discovered 10, skipped 0)")
                );
                continue;
            }
            let envelope = match format {
                "jsonl" => {
                    assert_eq!(
                        outcome.stdout.lines().count(),
                        1,
                        "one completed generation"
                    );
                    assert!(outcome.stdout.ends_with('\n'));
                    parse_success_envelope(format, &outcome)
                }
                "toon" => serde_json::to_value(
                    frankensearch_fsfs::output_schema::decode_envelope_toon::<Value>(
                        &outcome.stdout,
                    )
                    .expect("all stdout must be valid TOON"),
                )
                .unwrap(),
                "csv" => {
                    let mut lines = outcome.stdout.lines();
                    assert_eq!(lines.next(), Some("data_json"));
                    let cell = lines.next().expect("single CSV data row");
                    assert_eq!(lines.next(), None, "no trailing empty records");
                    // A single quoted CSV field; JSON string newlines remain
                    // escaped inside that field. Check its actual data against
                    // disk, rather than accepting an arbitrary nonempty row.
                    let json = cell
                        .strip_prefix('"')
                        .unwrap()
                        .strip_suffix('"')
                        .unwrap()
                        .replace("\"\"", "\"");
                    let data: Value = serde_json::from_str(&json).unwrap();
                    assert_index_data(&data, &sentinel, QUICKSTART_DOCUMENT_COUNT);
                    continue;
                }
                _ => unreachable!(),
            };
            assert_index_completion(&envelope, &sentinel, QUICKSTART_DOCUMENT_COUNT, format);
            let quality = VectorIndex::open_read_only(&index.join("vector/quality.fsvi")).unwrap();
            assert_eq!(
                envelope["data"]["quality_generation"]["id"],
                quality.embedder_id()
            );
            assert_eq!(
                envelope["data"]["quality_generation"]["dimension"],
                quality.dimension()
            );
        }
        let empty = root.join("empty-corpus");
        let empty_index = root.join("empty-index");
        fs::create_dir_all(&empty).unwrap();
        for format in ["json", "jsonl"] {
            let outcome = fsfs.run(
                root,
                &format!("index-empty-{format}"),
                [
                    "index",
                    empty.to_str().unwrap(),
                    "--index-dir",
                    empty_index.to_str().unwrap(),
                    "--format",
                    format,
                    "--quiet",
                ],
                QUICKSTART_TIMEOUT,
            );
            let envelope = parse_success_envelope("empty index", &outcome);
            let sentinel: Value =
                serde_json::from_slice(&fs::read(empty_index.join("index_sentinel.json")).unwrap())
                    .unwrap();
            assert_index_completion(&envelope, &sentinel, 0, format);
        }
        eprintln!(
            "[default-build-e2e] stage=index-formats json=jsonl=toon=csv=table:verified nonempty=10 empty=0 counts=published-sentinel"
        );
    }

    fn verify_fast_only_policy(fsfs: &IsolatedFsfs, root: &Path, corpus: &Path, full_index: &Path) {
        let fast_index = root.join("fast-only-index");
        let config = root.join("quality-requested.toml");
        fs::write(&config, "[search]\nfast_only = false\n").unwrap();
        let outcome = fsfs.run_with_env(
            root,
            "index-fast-only-cli",
            [
                "index",
                corpus.to_str().unwrap(),
                "--config",
                config.to_str().unwrap(),
                "--index-dir",
                fast_index.to_str().unwrap(),
                "--fast-only",
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
            &[("FRANKENSEARCH_FAST_ONLY", "false")],
        );
        let envelope = parse_success_envelope("CLI fast-only indexing", &outcome);
        assert_eq!(envelope["data"]["quality_generation"], Value::Null);
        assert_eq!(
            envelope["data"]["semantic_indexed_files"],
            QUICKSTART_DOCUMENT_COUNT
        );
        assert!(
            !fast_index.join("vector/quality.fsvi").exists(),
            "CLI must prevent quality generation even with both models installed"
        );
        for (label, extra_args, env) in [
            (
                "cli",
                vec!["--fast-only"],
                vec![("FRANKENSEARCH_FAST_ONLY", "false")],
            ),
            ("env", vec![], vec![("FRANKENSEARCH_FAST_ONLY", "true")]),
        ] {
            let mut args = vec![
                "search",
                "network retry",
                "--index-dir",
                full_index.to_str().unwrap(),
                "--config",
                config.to_str().unwrap(),
                "--no-daemon",
                "--format",
                "jsonl",
            ];
            args.extend(extra_args);
            let outcome = fsfs.run_with_env(
                root,
                &format!("search-fast-only-{label}"),
                args,
                QUICKSTART_TIMEOUT,
                &env,
            );
            assert_finished_successfully(label, &outcome);
            let envelopes = outcome
                .stdout
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                envelopes.len(),
                1,
                "{label} must skip quality on an existing two-tier index"
            );
            assert_eq!(envelopes[0]["data"]["phase"], "initial");
            assert!(
                envelopes[0]["data"]["hits"]
                    .as_array()
                    .is_some_and(|hits| !hits.is_empty())
            );
        }
        eprintln!(
            "[default-build-e2e] stage=fast-only cli-over-env-over-file=true default-profile=performance actual-quality-file=absent search-cli=initial-only search-env=initial-only"
        );
    }

    fn write_quickstart_corpus(corpus: &Path) {
        fs::create_dir_all(corpus).expect("create quickstart corpus");
        fs::write(corpus.join("retry.md"), RETRY_DOCUMENT).expect("write retry document");
        fs::write(
            corpus.join("gardening.md"),
            "Tomato seedlings need warm soil, steady sunlight, compost, and careful watering.",
        )
        .expect("write gardening document");
        fs::write(
            corpus.join("accounting.md"),
            "Reconcile invoices against purchase orders before closing the monthly ledger.",
        )
        .expect("write accounting document");
        fs::write(
            corpus.join("astronomy.md"),
            "A telescope gathers faint starlight so astronomers can study distant galaxies.",
        )
        .expect("write astronomy document");
        fs::write(
            corpus.join("baking.md"),
            "Bread develops flavor through a slow fermentation before baking in a hot oven.",
        )
        .expect("write baking document");
        fs::write(
            corpus.join("music.md"),
            "A string quartet balances melody, harmony, rhythm, and dynamics across four players.",
        )
        .expect("write music document");
        fs::write(
            corpus.join("fitness.md"),
            "Progressive resistance training strengthens muscles when recovery and nutrition are adequate.",
        )
        .expect("write fitness document");
        fs::write(
            corpus.join("legal.md"),
            "A written contract records the parties, obligations, remedies, and governing law.",
        )
        .expect("write legal document");
        fs::write(
            corpus.join("database.md"),
            "A database transaction groups related updates into one atomic durable operation.",
        )
        .expect("write database document");
        fs::write(
            corpus.join("typography.md"),
            "Readable typography uses deliberate spacing, line length, contrast, and type hierarchy.",
        )
        .expect("write typography document");
    }

    fn write_legacy_same_id_receipt_over_corrupt_potion_cache(model_root: &Path) {
        let manifest = ModelManifest::potion_128m();
        let model_dir = model_root.join("potion-multilingual-128M");
        fs::create_dir_all(&model_dir).expect("create stale-receipt model cache");
        let mut file_states = serde_json::Map::new();
        for artifact in &manifest.files {
            let path = model_dir.join(&artifact.name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create stale-receipt artifact parent");
            }
            let file = File::create(&path).expect("create sparse corrupt model artifact");
            file.set_len(artifact.size)
                .expect("size sparse corrupt model artifact");
            let metadata = fs::metadata(&path).expect("stat sparse corrupt model artifact");
            let modified_unix_nanos = metadata
                .modified()
                .expect("read sparse artifact mtime")
                .duration_since(std::time::UNIX_EPOCH)
                .expect("artifact mtime after unix epoch")
                .as_nanos();
            file_states.insert(
                artifact.name.clone(),
                serde_json::json!({
                    "size_bytes": metadata.len(),
                    "modified_unix_nanos": u64::try_from(modified_unix_nanos)
                        .expect("artifact mtime fits u64 nanos")
                }),
            );
        }
        let legacy_marker = serde_json::json!({
            "manifest_id": manifest.id,
            "schema_version": frankensearch_embed::model_manifest::MANIFEST_SCHEMA_VERSION,
            "verified_at": 1,
            "file_states": file_states
        });
        fs::write(
            model_dir.join(".verified"),
            serde_json::to_vec_pretty(&legacy_marker).expect("serialize legacy marker"),
        )
        .expect("write legacy same-ID marker");
        eprintln!(
            "[default-build-e2e] stage=stale-receipt-fixture event=created manifest_id=potion-multilingual-128m receipt_schema=legacy-no-manifest-fingerprint corrupt_sparse_bytes=true"
        );
    }

    fn configured_model_root() -> PathBuf {
        for key in [
            "FSFS_DEFAULT_E2E_MODEL_DIR",
            "FRANKENSEARCH_BUNDLED_MODELS_SOURCE_DIR",
            "FRANKENSEARCH_MODEL_DIR",
        ] {
            if let Some(path) = std::env::var_os(key) {
                return PathBuf::from(path);
            }
        }

        std::env::var_os("HOME").map_or_else(
            || PathBuf::from(".local/share/frankensearch/models"),
            |home| {
                PathBuf::from(home)
                    .join(".local")
                    .join("share")
                    .join("frankensearch")
                    .join("models")
            },
        )
    }

    #[cfg(feature = "semantic-loaders")]
    fn verify_real_blend_and_deadline(fsfs: &IsolatedFsfs, root: &Path, index: &Path) {
        use sha2::Digest as _;
        use std::fmt::Write as _;
        let file_sha = |path: &Path| {
            let mut hex = String::with_capacity(64);
            for byte in sha2::Sha256::digest(fs::read(path).unwrap()) {
                write!(hex, "{byte:02x}").unwrap();
            }
            hex
        };
        let query = "How should a network client recover from transient failures using exponential backoff, bounded retries, and random jitter?";
        #[cfg(unix)]
        struct DaemonOwner(PathBuf);
        #[cfg(unix)]
        impl Drop for DaemonOwner {
            fn drop(&mut self) {
                use std::io::Write as _;
                if let Ok(mut socket) = std::os::unix::net::UnixStream::connect(&self.0) {
                    let _ = socket.write_all(b"quit\n");
                }
            }
        }
        #[cfg(unix)]
        let daemon = DaemonOwner(
            std::env::temp_dir().join(format!("fsfs-blend-{}.sock", std::process::id())),
        );
        let stack = frankensearch_embed::EmbedderStack::auto_detect_with_options(
            Some(&fsfs.model_root),
            &frankensearch_embed::DetectOptions {
                offline: Some(true),
            },
        )
        .expect("load actual comparison models");
        let fast = stack.fast_arc();
        let quality = stack
            .quality_arc()
            .expect("actual quality comparison model");
        let fast_index_path = index.join("vector/index.fsvi");
        let quality_index_path = index.join("vector/quality.fsvi");
        let fast_index =
            VectorIndex::open_read_only(&fast_index_path).expect("open actual fast generation");
        let quality_index = VectorIndex::open_read_only(&quality_index_path)
            .expect("open actual quality generation");
        let lexical_root = index.join("lexical");
        let lexical_pointer = frankensearch_quill::CurrentPointer::decode(
            &fs::read(lexical_root.join(frankensearch_quill::CURRENT_FILE_NAME)).unwrap(),
        )
        .expect("decode the actual lexical generation");
        assert_eq!(
            lexical_pointer.engine(),
            frankensearch_quill::BlueGreenEngine::Quill
        );
        let lexical_path = lexical_pointer.engine_dir(&lexical_root);
        let binary_sha = file_sha(&fsfs_binary());
        eprintln!(
            "[default-build-e2e] stage=blend-provenance binary_sha256={binary_sha} fast_id={} quality_id={} fast_identity={} quality_identity={} fast_index_sha256={} quality_index_sha256={}",
            fast.id(),
            quality.id(),
            fast.identity().unwrap().fingerprint(),
            quality.identity().unwrap().fingerprint(),
            file_sha(&fast_index_path),
            file_sha(&quality_index_path)
        );
        let scheduler = asupersync::runtime::RuntimeBuilder::current_thread()
            .blocking_threads(0, 2)
            .build()
            .unwrap();
        let task = scheduler.handle().spawn(async move {
            use frankensearch_core::LexicalRead as _;
            let cx = asupersync::Cx::current().expect("runtime-owned comparison context");
            let fast_vector = fast.embed(&cx, query).await.unwrap();
            let quality_vector = quality.embed(&cx, query).await.unwrap();
            let lexical = frankensearch_quill::QuillSearchIndex::open(
                &cx,
                lexical_path,
                frankensearch_quill::QuillConfig::default(),
            )
            .await
            .unwrap();
            let lexical_pool = lexical
                .search(&cx, query, QUICKSTART_DOCUMENT_COUNT)
                .await
                .unwrap();
            (
                fast_index
                    .search_top_k(&fast_vector, QUICKSTART_DOCUMENT_COUNT, None)
                    .unwrap(),
                quality_index
                    .search_top_k(&quality_vector, QUICKSTART_DOCUMENT_COUNT, None)
                    .unwrap(),
                lexical_pool,
            )
        });
        let (fast_pool, quality_pool, lexical_pool) = scheduler.block_on(task);
        for weight in [0.0_f32, 0.7, 1.0] {
            let config_path = root.join(format!("blend-{weight}.toml"));
            // This leg measures numerical policy parity under a comfortably
            // admitted budget. The stock-default lane above remains unchanged;
            // the separate 50ms leg below exercises actual deadline failure.
            fs::write(
                &config_path,
                format!("[search]\nquality_weight = {weight}\nquality_timeout_ms = 5000\n"),
            )
            .unwrap();
            let config_str = config_path.to_str().unwrap();
            let mut direct_blend = None;
            for route in ["direct", "daemon", "daemon-warm"] {
                let mut args = vec![
                    "search",
                    query,
                    "--config",
                    config_str,
                    "--index-dir",
                    index.to_str().unwrap(),
                    "--limit",
                    "all",
                    "--format",
                    "json",
                ];
                if route == "direct" {
                    args.push("--no-daemon");
                } else {
                    #[cfg(unix)]
                    args.extend(["--daemon-socket", daemon.0.to_str().unwrap()]);
                    #[cfg(not(unix))]
                    continue;
                }
                let outcome = fsfs.run_with_env(
                    root,
                    &format!("blend-{weight}-{route}"),
                    args,
                    QUICKSTART_TIMEOUT,
                    &[],
                );
                let envelope = parse_success_envelope("real policy search", &outcome);
                assert!(
                    !outcome
                        .stderr
                        .contains("falling back to in-process retrieval"),
                    "the selected serving route must execute: {}",
                    outcome.stderr
                );
                #[cfg(unix)]
                if route != "direct" {
                    use std::io::{Read as _, Write as _};
                    let mut socket = std::os::unix::net::UnixStream::connect(&daemon.0)
                        .expect("the same live daemon must serve successive policies");
                    socket.set_read_timeout(Some(QUICKSTART_TIMEOUT)).unwrap();
                    let request = serde_json::json!({
                        "query": query, "limit": usize::MAX, "quality_weight": weight,
                        "quality_timeout_ms": 5000, "rrf_k": 60.0, "fast_only": false
                    });
                    writeln!(socket, "{request}").unwrap();
                    socket.shutdown(std::net::Shutdown::Write).unwrap();
                    let mut raw = String::new();
                    socket.read_to_string(&mut raw).unwrap();
                    let response: Value = serde_json::from_str(&raw).unwrap();
                    assert_eq!(response["schema_version"], "fsfs.search.serve.v3");
                    use frankensearch_embed::model_manifest::ModelArtifactManifestV1;
                    let mut contracts = sha2::Sha256::new();
                    for manifest in [
                        ModelArtifactManifestV1::potion_128m_native().unwrap(),
                        ModelArtifactManifestV1::minilm_fastembed().unwrap(),
                        ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
                        ModelArtifactManifestV1::nomic_fastembed().unwrap(),
                    ] {
                        contracts.update(manifest.freeze().unwrap().fingerprint.as_bytes());
                    }
                    let mut expected_contracts = String::with_capacity(64);
                    for byte in contracts.finalize() {
                        write!(expected_contracts, "{byte:02x}").unwrap();
                    }
                    assert_eq!(
                        response["policy"]["embedding_contracts"], expected_contracts,
                        "the actual serving binary acknowledges its registered producers"
                    );
                    assert_eq!(response["policy"]["quality_weight_bits"], weight.to_bits());
                    assert_eq!(response["policy"]["quality_timeout_ms"], 5000);
                    assert_eq!(
                        response["cached"], true,
                        "matching policy uses the live daemon cache"
                    );
                }
                assert_eq!(
                    envelope.pointer("/data/phase").and_then(Value::as_str),
                    Some("refined")
                );
                let actual: frankensearch_fsfs::output_schema::SemanticBlendPayload =
                    serde_json::from_value(
                        envelope.pointer("/data/semantic_blend").unwrap().clone(),
                    )
                    .unwrap();
                let expected = frankensearch::blend_two_tier(&fast_pool, &quality_pool, weight);
                assert_eq!(actual.quality_weight, weight);
                assert_eq!(actual.hits.len(), QUICKSTART_DOCUMENT_COUNT);
                for expected_hit in &expected {
                    let observed = actual
                        .hits
                        .iter()
                        .find(|hit| hit.path == expected_hit.doc_id.as_str())
                        .unwrap();
                    assert!(
                        (observed.score - expected_hit.score).abs() < 1e-6,
                        "public facade/fsfs score disagreement: {weight} {route} {}",
                        observed.path
                    );
                }
                let blended_vectors = expected
                    .iter()
                    .map(|hit| frankensearch_core::VectorHit {
                        index: 0,
                        score: hit.score,
                        doc_id: hit.doc_id.clone(),
                    })
                    .collect::<Vec<_>>();
                let expected_fused = frankensearch::rrf_fuse(
                    &lexical_pool,
                    &blended_vectors,
                    QUICKSTART_DOCUMENT_COUNT,
                    0,
                    &frankensearch::RrfConfig::default(),
                );
                let output_hits = envelope.pointer("/data/hits").unwrap().as_array().unwrap();
                assert_eq!(output_hits.len(), expected_fused.len());
                for hit in &expected_fused {
                    let observed = output_hits
                        .iter()
                        .find(|row| row["path"] == hit.doc_id.as_str())
                        .unwrap();
                    assert!(
                        (observed["score"].as_f64().unwrap() - hit.rrf_score).abs() < 1e-12,
                        "independently retrieved Quill/public-facade RRF differs for {}",
                        hit.doc_id
                    );
                }
                // Existing planner tie rules are intentionally retained: fsfs
                // uses semantic score before doc ID after equal RRF/lexical
                // scores; the facade defaults to doc ID at that last boundary.
                // Compare every final ID and numeric score without pretending
                // those pre-existing equal-score tie policies are identical.
                if let Some(direct) = &direct_blend {
                    assert_eq!(&actual, direct, "daemon uses caller's policy");
                } else {
                    direct_blend = Some(actual.clone());
                }
                eprintln!(
                    "[default-build-e2e] stage=blend-parity weight={weight} route={route} independently_retrieved_fast={} independently_retrieved_quality={} independently_retrieved_quill={} returned={} blend_error_lt=0.000001 joint_rrf_error_lt=0.000000000001 no_quality_win_claim=true",
                    fast_pool.len(),
                    quality_pool.len(),
                    lexical_pool.len(),
                    actual.hits.len()
                );
            }
            let explain = fsfs.run_with_env(
                root,
                &format!("blend-{weight}-explain"),
                [
                    "explain",
                    "1",
                    "--config",
                    config_str,
                    "--index-dir",
                    index.to_str().unwrap(),
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
                &[],
            );
            let envelope = parse_success_envelope("cached policy explain", &explain);
            assert!(
                envelope
                    .pointer("/data/ranking/fusion/rrf/vector")
                    .and_then(Value::as_f64)
                    .is_some_and(|score| score > 0.0)
            );
            assert!(
                envelope
                    .pointer("/data/ranking/fusion/lexical_score")
                    .and_then(Value::as_f64)
                    .is_some()
            );
        }

        #[cfg(unix)]
        {
            let socket_path = daemon.0.clone();
            drop(daemon);
            let deadline = Instant::now() + QUICKSTART_TIMEOUT;
            while socket_path.exists() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(25));
            }
            assert!(
                !socket_path.exists(),
                "policy daemon must finish structured shutdown"
            );
        }

        let timeout_config = root.join("quality-deadline-50.toml");
        fs::write(&timeout_config, "[search]\nquality_timeout_ms = 50\n").unwrap();
        let outcome = fsfs.run_with_env(
            root,
            "actual-quality-deadline-50",
            [
                "search",
                query,
                "--config",
                timeout_config.to_str().unwrap(),
                "--index-dir",
                index.to_str().unwrap(),
                "--no-daemon",
                "--stream",
                "--format",
                "jsonl",
            ],
            QUICKSTART_TIMEOUT,
            &[
                ("FSFS_DISABLE_QUERY_CACHE", "1"),
                ("FRANKENSEARCH_LOG", "info"),
            ],
        );
        assert_finished_successfully("actual backend quality deadline", &outcome);
        let frames = outcome
            .stdout
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let initial = frames
            .iter()
            .position(|frame| {
                frame
                    .pointer("/payload/reason_code")
                    .and_then(Value::as_str)
                    == Some("query.stream.initial_ready")
            })
            .unwrap();
        let failures = frames
            .iter()
            .enumerate()
            .filter(|(_, frame)| {
                frame
                    .pointer("/payload/reason_code")
                    .and_then(Value::as_str)
                    == Some("query.stream.refinement_failed")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            failures.len(),
            1,
            "actual cold backend must respect 50ms boundary"
        );
        assert!(failures[0].0 > initial);
        let initial_hits = frames[initial + 1..failures[0].0]
            .iter()
            .filter(|frame| frame["event"] == "result")
            .map(|frame| frame["payload"].clone())
            .collect::<Vec<_>>();
        let failure_hits = frames[failures[0].0 + 1..]
            .iter()
            .filter(|frame| frame["event"] == "result")
            .map(|frame| frame["payload"].clone())
            .collect::<Vec<_>>();
        assert!(!initial_hits.is_empty(), "actual Initial returns real hits");
        assert_eq!(
            failure_hits, initial_hits,
            "timeout preserves every actual Initial item, score and rank"
        );
        assert!(
            failures[0]
                .1
                .pointer("/payload/message")
                .and_then(Value::as_str)
                .unwrap()
                .contains("50ms")
        );
        assert!(!frames.iter().any(|frame| {
            frame
                .pointer("/payload/reason_code")
                .and_then(Value::as_str)
                == Some("query.stream.refined_ready")
        }));
        eprintln!(
            "[default-build-e2e] stage=actual-quality-deadline budget_ms=50 initial_preserved=true failures=1 late_refined=0 process_join_ms={} backend_preemptible=false",
            outcome.elapsed.as_millis()
        );
    }

    fn verify_pinned_model_cache(model_root: &Path) -> Result<(), String> {
        let potion_dir = model_root.join("potion-multilingual-128M");
        let minilm_dir = model_root.join("all-MiniLM-L6-v2");
        eprintln!(
            "[default-build-e2e] stage=model-verification event=start model_root={}",
            model_root.display()
        );
        ModelManifest::potion_128m()
            .verify_dir(&potion_dir)
            .map_err(|error| {
                format!(
                    "pinned Potion cache verification failed at {}: {error}",
                    potion_dir.display()
                )
            })?;
        ModelManifest::minilm_v2()
            .verify_dir(&minilm_dir)
            .map_err(|error| {
                format!(
                    "pinned MiniLM cache verification failed at {}: {error}",
                    minilm_dir.display()
                )
            })?;
        eprintln!("[default-build-e2e] stage=model-verification event=verified");
        Ok(())
    }

    #[test]
    fn config_inspection_honors_valid_overrides_of_incomplete_source_layers() {
        let temp = tempfile::tempdir().unwrap();
        let fsfs = IsolatedFsfs::new(temp.path(), temp.path().join("models"));
        let config = temp.path().join("strict-quality.toml");
        fs::write(&config, "[pressure]\nprofile = \"strict\"\n[search]\nfast_only = false\nquality_timeout_ms = 250\n").unwrap();
        for (key, expected, source) in [
            ("search.fast_only", serde_json::json!(true), "cli"),
            (
                "search.quality_timeout_ms",
                serde_json::json!(250),
                "config",
            ),
        ] {
            let outcome = fsfs.run(
                temp.path(),
                key,
                [
                    "config",
                    "get",
                    key,
                    "--config",
                    config.to_str().unwrap(),
                    "--fast-only",
                    "--format",
                    "json",
                ],
                FAILURE_TIMEOUT,
            );
            let envelope = parse_success_envelope("inspect accepted CLI override", &outcome);
            assert_eq!(envelope["data"]["value"], expected);
            assert_eq!(envelope["data"]["source"], source);
        }
        for overrides in [vec!["--fast-only"], vec!["--profile", "performance"]] {
            let mut args = vec![
                "config",
                "--config",
                config.to_str().unwrap(),
                "--format",
                "table",
            ];
            args.extend(overrides);
            let outcome = fsfs.run(temp.path(), "config-table", args, FAILURE_TIMEOUT);
            assert_finished_successfully("table inspection of valid final configuration", &outcome);
            assert!(outcome.stdout.contains("search.fast_only"));
        }
        let rejected = fsfs.run(
            temp.path(),
            "winning-strict-false",
            [
                "config",
                "get",
                "search.fast_only",
                "--config",
                config.to_str().unwrap(),
                "--no-fast-only",
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );
        assert!(!rejected.timed_out);
        assert_eq!(rejected.status.code(), Some(2));
        let envelope: Value = serde_json::from_str(&rejected.stdout).unwrap();
        assert_eq!(envelope["ok"], false);
        assert_eq!(envelope["error"]["code"], "invalid_config");
        assert_eq!(envelope["error"]["field"], "search.fast_only");
        assert!(
            envelope["error"]["message"]
                .as_str()
                .unwrap()
                .contains("strict")
        );
    }

    #[test]
    fn index_rejected_target_emits_only_the_requested_error_envelope() {
        let temp = tempfile::tempdir().unwrap();
        let fsfs = IsolatedFsfs::new(temp.path(), temp.path().join("models"));
        let missing = temp.path().join("missing-source");
        for format in ["json", "jsonl"] {
            let outcome = fsfs.run(
                temp.path(),
                &format!("missing-target-{format}"),
                ["index", missing.to_str().unwrap(), "--format", format],
                FAILURE_TIMEOUT,
            );
            assert!(!outcome.timed_out);
            assert_eq!(outcome.status.code(), Some(2));
            let envelope: Value = serde_json::from_str(&outcome.stdout).unwrap();
            assert_eq!(envelope["ok"], false);
            assert_eq!(envelope["error"]["code"], "invalid_config");
            assert_eq!(envelope["error"]["exit_code"], 2);
            assert_eq!(envelope["meta"]["command"], "index");
            assert_eq!(envelope["meta"]["format"], format);
            assert!(envelope.get("data").is_none());
            assert!(
                envelope["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("target path does not exist")
            );
            if format == "jsonl" {
                assert_eq!(outcome.stdout.lines().count(), 1);
            }
        }
    }

    #[test]
    fn default_build_without_models_fails_closed_with_actionable_guidance() {
        log_binary_profile("missing-model");
        let temp = tempfile::tempdir().expect("create default-build failure fixture");
        let corpus = temp.path().join("corpus");
        let index = temp.path().join("index");
        write_quickstart_corpus(&corpus);
        let fsfs = IsolatedFsfs::new(temp.path(), temp.path().join("empty-model-cache"));

        let missing_status = fsfs.run(
            temp.path(),
            "status-missing-models",
            [
                "status",
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );
        let missing_status_envelope =
            parse_success_envelope("default-build missing-model status", &missing_status);
        for tier in ["fast", "quality"] {
            let status = model_status(&missing_status_envelope, tier);
            assert_eq!(
                status.get("verification_state").and_then(Value::as_str),
                Some("missing"),
                "status must classify an absent {tier} cache as missing: {status}"
            );
            assert_eq!(
                status.get("cached").and_then(Value::as_bool),
                Some(false),
                "status must not call an absent {tier} cache cached: {status}"
            );
        }
        eprintln!(
            "[default-build-e2e] stage=status-missing event=verified exit=0 ok=true fast=missing quality=missing"
        );

        let outcome = fsfs.run(
            temp.path(),
            "offline-missing-model",
            [
                "index",
                corpus.to_str().expect("UTF-8 corpus path"),
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );

        assert!(!outcome.timed_out, "missing-model failure must be prompt");
        assert_eq!(
            outcome.status.code(),
            Some(78),
            "missing semantic models are an EX_CONFIG failure; output:\n{}",
            outcome.combined_output()
        );
        let missing_model_error: Value = serde_json::from_str(&outcome.stdout)
            .expect("all missing-model stdout must be one error envelope, without discovery prose");
        assert_eq!(missing_model_error["ok"], false);
        assert_eq!(missing_model_error["error"]["code"], "embedder_unavailable");
        let rendered = outcome.combined_output().to_ascii_lowercase();
        assert!(
            rendered.contains("embedder_unavailable"),
            "failure must expose the typed error code: {rendered}"
        );
        assert!(
            rendered.contains("fsfs download-models"),
            "failure must include a directly runnable recovery command: {rendered}"
        );
        assert!(
            !rendered.contains("hash fallback") && !rendered.contains("fnv1a"),
            "production indexing must never advertise hash fallback: {rendered}"
        );
        assert!(
            !index.join("index_sentinel.json").exists(),
            "a rejected semantic generation must not publish a completion sentinel"
        );
        assert!(
            !index.join("vector/index.fsvi").exists(),
            "a rejected semantic generation must not publish an FSVI"
        );

        let missing_verify = fsfs.run(
            temp.path(),
            "verify-missing-model",
            [
                "download-models",
                "potion-multilingual-128m",
                "--verify",
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );
        assert!(
            !missing_verify.timed_out,
            "missing verification must be prompt"
        );
        assert_eq!(
            missing_verify.status.code(),
            Some(78),
            "missing verification must be a typed EX_CONFIG failure: {}",
            missing_verify.combined_output()
        );
        assert!(
            missing_verify
                .combined_output()
                .to_ascii_lowercase()
                .contains("model_not_found"),
            "missing verification must expose model_not_found: {}",
            missing_verify.combined_output()
        );

        write_legacy_same_id_receipt_over_corrupt_potion_cache(&fsfs.model_root);
        let corrupt_receipt_path = fsfs.model_root.join("potion-multilingual-128M/.verified");
        let legacy_receipt = fs::read(&corrupt_receipt_path)
            .expect("capture legacy receipt before observational commands");
        let corrupt_verify = fsfs.run(
            temp.path(),
            "verify-corrupt-model",
            [
                "download-models",
                "potion-multilingual-128m",
                "--verify",
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );
        assert!(
            !corrupt_verify.timed_out,
            "corrupt verification must be prompt"
        );
        assert!(
            !corrupt_verify.status.success(),
            "corrupt verification must return nonzero: {}",
            corrupt_verify.combined_output()
        );
        assert!(
            corrupt_verify
                .combined_output()
                .to_ascii_lowercase()
                .contains("hash_mismatch"),
            "corrupt verification must preserve its typed checksum error: {}",
            corrupt_verify.combined_output()
        );

        let corrupt_status = fsfs.run(
            temp.path(),
            "status-corrupt-model",
            [
                "status",
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );
        let corrupt_status_envelope =
            parse_success_envelope("default-build corrupt-model status", &corrupt_status);
        let corrupt_fast = model_status(&corrupt_status_envelope, "fast");
        assert_eq!(
            corrupt_fast
                .get("verification_state")
                .and_then(Value::as_str),
            Some("mismatch"),
            "status must reject corrupt bytes behind a legacy same-ID receipt: {corrupt_fast}"
        );
        assert_eq!(
            corrupt_fast.get("cached").and_then(Value::as_bool),
            Some(false),
            "a corrupt cache must never be reported as cached: {corrupt_fast}"
        );
        assert_eq!(
            fs::read(&corrupt_receipt_path).expect("read receipt after status"),
            legacy_receipt,
            "status must remain observational and must not rewrite a stale receipt"
        );

        let failed_doctor = fsfs.run(
            temp.path(),
            "doctor-corrupt-model",
            [
                "doctor",
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );
        assert!(!failed_doctor.timed_out, "failed doctor must be prompt");
        assert_eq!(
            failed_doctor.status.code(),
            Some(1),
            "a hard doctor verdict must return EX_RUNTIME: {}",
            failed_doctor.combined_output()
        );
        let failed_doctor_envelope: Value = serde_json::from_str(&failed_doctor.stdout)
            .expect("failed doctor must emit one machine-readable JSON envelope");
        assert_eq!(
            failed_doctor_envelope.get("ok").and_then(Value::as_bool),
            Some(false),
            "failed doctor must never emit a success envelope: {failed_doctor_envelope}"
        );
        assert_eq!(
            failed_doctor_envelope
                .pointer("/error/code")
                .and_then(Value::as_str),
            Some("subsystem_error"),
            "failed doctor needs a stable machine error code: {failed_doctor_envelope}"
        );
        assert!(
            failed_doctor_envelope
                .pointer("/error/context")
                .and_then(Value::as_str)
                .is_some_and(|context| context.contains("model.fast")),
            "failed doctor context must identify the failing check: {failed_doctor_envelope}"
        );
        assert!(
            failed_doctor_envelope.get("data").is_none()
                || failed_doctor_envelope.get("data") == Some(&Value::Null),
            "error envelopes must not smuggle success data: {failed_doctor_envelope}"
        );
        assert_eq!(
            fs::read(&corrupt_receipt_path).expect("read receipt after doctor"),
            legacy_receipt,
            "doctor must not mint or refresh a receipt"
        );
        eprintln!(
            "[default-build-e2e] stage=doctor-failure event=verified exit=1 ok=false code=subsystem_error check=model.fast stale_same_id_receipt_rejected=true"
        );
    }

    #[cfg(unix)]
    struct WatchChild(std::process::Child);

    #[cfg(unix)]
    impl Drop for WatchChild {
        fn drop(&mut self) {
            if self.0.try_wait().ok().flatten().is_none() {
                let _ = self.0.kill();
            }
            let _ = self.0.wait();
        }
    }

    #[cfg(unix)]
    fn wait_for_watch_output(
        child: &mut WatchChild,
        path: &Path,
        needle: &str,
        count: usize,
        timeout: Duration,
    ) -> String {
        let started = Instant::now();
        loop {
            let output = fs::read_to_string(path).expect("read watch output");
            if output.matches(needle).count() >= count {
                return output;
            }
            assert!(
                child.0.try_wait().expect("poll watcher").is_none(),
                "watcher exited before {needle}; output:\n{output}"
            );
            assert!(
                started.elapsed() < timeout,
                "watcher did not emit {needle} {count} times in {timeout:?}; output:\n{output}"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(unix)]
    #[test]
    #[ignore = "real-model watch handoff; requires the pinned model cache and semantic E2E opt-in"]
    fn default_build_watch_reconciles_handoff_and_persists_live_updates() -> Result<(), String> {
        log_binary_profile("real-model-watch");
        if std::env::var("FRANKENSEARCH_REQUIRE_SEMANTIC_E2E").as_deref() != Ok("1") {
            return Err(
                "set FRANKENSEARCH_REQUIRE_SEMANTIC_E2E=1 for real-model watch validation"
                    .to_owned(),
            );
        }
        let model_root = configured_model_root();
        verify_pinned_model_cache(&model_root)?;
        let temp = tempfile::tempdir().expect("watch fixture");
        let fsfs = IsolatedFsfs::new(temp.path(), model_root);
        let corpus = temp.path().join("corpus");
        let index = temp.path().join("index");
        fs::create_dir_all(&corpus).expect("watch corpus");
        fs::write(corpus.join("database.md"), "Oldquartz database transactions preserve atomicity using rollback and durable journals.").unwrap();
        fs::write(
            corpus.join("astronomy.md"),
            "Astronomers measure starlight to classify distant galaxies and planets.",
        )
        .unwrap();
        let corpus_arg = corpus.to_str().unwrap();
        let index_arg = index.to_str().unwrap();
        let initial = fsfs.run(
            temp.path(),
            "watch-initial-index",
            [
                "index",
                corpus_arg,
                "--index-dir",
                index_arg,
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        let envelope = parse_success_envelope("watch initial index", &initial);
        assert_eq!(envelope["data"]["indexed_files"], 2);
        assert_eq!(envelope["data"]["quality_generation"]["dimension"], 384);

        let stdout_path = fsfs.log_root.join("watch.stdout.log");
        let stderr_path = fsfs.log_root.join("watch.stderr.log");
        let mut watch = WatchChild(
            fsfs.command(temp.path())
                .args([
                    "index",
                    corpus_arg,
                    "--watch",
                    "--index-dir",
                    index_arg,
                    "--format",
                    "jsonl",
                ])
                .env("RUST_LOG", "info")
                .stdout(File::create(&stdout_path).unwrap())
                .stderr(File::create(&stderr_path).unwrap())
                .spawn()
                .expect("spawn real watcher"),
        );
        let publication = wait_for_watch_output(
            &mut watch,
            &stdout_path,
            "\"generation_complete\":true",
            1,
            QUICKSTART_TIMEOUT,
        );
        let publication: Value =
            serde_json::from_str(publication.trim()).expect("watch index completion envelope");
        assert_eq!(publication["data"]["indexed_files"], 2);
        // No watcher callback can observe these edits: model initialization
        // still separates the completed one-shot pass from notify registration.
        let before = fs::read_to_string(&stderr_path).unwrap();
        assert!(
            !before.contains("live ingest pipeline initialized for watch mode"),
            "the test missed the startup handoff window; no handoff claim is valid"
        );
        fs::write(
            corpus.join("handoff.md"),
            "Handoffquartz database recovery uses transaction rollback to preserve atomicity.",
        )
        .unwrap();
        fs::write(corpus.join("database.md"), "Newquartz restores database atomicity through transaction rollback and durable journals.").unwrap();
        let after_writes = fs::read_to_string(&stderr_path).unwrap();
        assert!(
            !after_writes.contains("live ingest pipeline initialized for watch mode"),
            "watcher initialization overlapped the fixture writes; no handoff claim is valid"
        );
        wait_for_watch_output(
            &mut watch,
            &stderr_path,
            "fsfs watch reconciliation completed",
            1,
            QUICKSTART_TIMEOUT,
        );

        // These are ordinary notify events, after the initial scan was applied.
        let before = fs::read_to_string(&stderr_path)
            .unwrap()
            .matches("fsfs watch batch applied")
            .count();
        fs::write(
            corpus.join("live.md"),
            "Liveamber handles transient failures with bounded retries and exponential backoff.",
        )
        .unwrap();
        wait_for_watch_output(
            &mut watch,
            &stderr_path,
            "fsfs watch batch applied",
            before + 1,
            FAILURE_TIMEOUT,
        );
        let before = fs::read_to_string(&stderr_path)
            .unwrap()
            .matches("fsfs watch batch applied")
            .count();
        fs::write(corpus.join("live.md"), "Livemalachite restores service after failures using bounded retries and exponential backoff.").unwrap();
        wait_for_watch_output(
            &mut watch,
            &stderr_path,
            "fsfs watch batch applied",
            before + 1,
            FAILURE_TIMEOUT,
        );

        let signal = Command::new("kill")
            .args(["-TERM", &watch.0.id().to_string()])
            .status()
            .expect("signal own watcher");
        assert!(signal.success());
        let stop_started = Instant::now();
        let status = loop {
            if let Some(status) = watch.0.try_wait().expect("poll graceful shutdown") {
                break status;
            }
            assert!(
                stop_started.elapsed() < FAILURE_TIMEOUT,
                "watch shutdown timed out"
            );
            thread::sleep(Duration::from_millis(10));
        };
        let stderr = fs::read_to_string(&stderr_path).unwrap();
        eprintln!("[default-build-e2e] stage=watch-shutdown status={status} stderr:\n{stderr}");
        assert!(status.success(), "graceful watcher shutdown failed");
        drop(watch);

        // Fresh handles must contain exactly one live row per file in BOTH
        // vector spaces. Query execution below verifies the lexical arm too.
        let stack = frankensearch_embed::EmbedderStack::auto_detect_with_options(
            Some(&fsfs.model_root),
            &frankensearch_embed::DetectOptions {
                offline: Some(true),
            },
        )
        .expect("load verified models for durable vector comparison");
        let scheduler = asupersync::runtime::RuntimeBuilder::current_thread()
            .blocking_threads(0, 2)
            .build()
            .unwrap();
        for (file, embedder) in [
            ("vector/index.fsvi", stack.fast_arc()),
            (
                "vector/quality.fsvi",
                stack.quality_arc().expect("quality comparison model"),
            ),
        ] {
            let vectors =
                VectorIndex::open_read_only(&index.join(file)).expect("reopen watched vector tier");
            let hits = vectors
                .search_top_k(&vec![1.0; vectors.dimension()], 10, None)
                .expect("enumerate live tier");
            let mut ids = hits
                .iter()
                .map(|hit| hit.doc_id.clone())
                .collect::<Vec<_>>();
            ids.sort();
            assert_eq!(
                ids,
                ["astronomy.md", "database.md", "handoff.md", "live.md"],
                "{file}"
            );
            assert_eq!(
                vectors.wal_record_count(),
                0,
                "{file}: graceful shutdown must compact"
            );
            let corpus = corpus.clone();
            let expected = scheduler.block_on(scheduler.handle().spawn(async move {
                use frankensearch_core::Canonicalizer as _;
                let cx = asupersync::Cx::current().expect("owned comparison context");
                let mut expected = Vec::new();
                for id in ids {
                    let text = frankensearch_core::DefaultCanonicalizer::default()
                        .canonicalize(&fs::read_to_string(corpus.join(&id)).unwrap());
                    expected.push((id, embedder.embed(&cx, &text).await.unwrap()));
                }
                expected
            }));
            for (id, expected) in expected {
                let hit = hits.iter().find(|hit| hit.doc_id == id).unwrap();
                let stored = vectors.vector_at_f32(hit.index as usize).unwrap();
                assert_eq!(stored.len(), expected.len());
                assert!(
                    stored
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| (a - b).abs() < 0.001),
                    "{file}: {id} does not encode the current file after f16 quantization"
                );
            }
        }
        let config = temp.path().join("watch-search.toml");
        // Functional coverage uses an explicit budget, not a latency claim.
        fs::write(&config, "[search]\nquality_timeout_ms = 5000\n").unwrap();
        for (label, query, expected) in [
            (
                "handoff",
                "how handoffquartz database recovery preserves atomicity",
                "handoff.md",
            ),
            (
                "handoff-modify",
                "how newquartz restores database atomicity",
                "database.md",
            ),
            (
                "live-modify",
                "how livemalachite restores service after failures",
                "live.md",
            ),
        ] {
            let outcome = fsfs.run(
                temp.path(),
                label,
                [
                    "search",
                    query,
                    "--no-daemon",
                    "--config",
                    config.to_str().unwrap(),
                    "--index-dir",
                    index_arg,
                    "--format",
                    "json",
                    "--limit",
                    "10",
                ],
                QUICKSTART_TIMEOUT,
            );
            let result = parse_success_envelope(label, &outcome);
            assert_eq!(result["data"]["phase"], "refined", "{label}: {result}");
            let hit = result["data"]["hits"]
                .as_array()
                .unwrap()
                .iter()
                .find(|hit| hit["path"] == expected)
                .expect("watched document in actual search results");
            assert!(hit["lexical_rank"].is_number(), "{label}: {hit}");
            assert!(hit["semantic_rank"].is_number(), "{label}: {hit}");
        }
        for old in ["Oldquartz", "Liveamber"] {
            let outcome = fsfs.run(
                temp.path(),
                old,
                [
                    "search",
                    old,
                    "--fast-only",
                    "--no-daemon",
                    "--index-dir",
                    index_arg,
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            let result = parse_success_envelope(old, &outcome);
            assert!(
                result["data"]["hits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|hit| hit["lexical_rank"].is_null()),
                "superseded lexical text remains: {result}"
            );
        }
        eprintln!(
            "[default-build-e2e] stage=watch-handoff event=verified startup_create=true startup_modify=true live_create=true live_modify=true lexical=true fast=true quality=true post_exit=true"
        );
        Ok(())
    }

    #[test]
    #[ignore = "mock-free model-backed quickstart; provision the pinned cache, then run with --ignored --nocapture"]
    fn default_build_indexes_and_returns_a_real_hybrid_result() -> Result<(), String> {
        log_binary_profile("real-model");
        if std::env::var("FRANKENSEARCH_REQUIRE_SEMANTIC_E2E")
            .ok()
            .as_deref()
            != Some("1")
        {
            return Err(
                "set FRANKENSEARCH_REQUIRE_SEMANTIC_E2E=1 to acknowledge the real-model CI lane"
                    .to_owned(),
            );
        }
        let model_root = configured_model_root();
        verify_pinned_model_cache(&model_root)?;

        let temp = tempfile::tempdir().expect("create default-build quickstart fixture");
        let corpus = temp.path().join("corpus");
        let index = temp.path().join("index");
        write_quickstart_corpus(&corpus);
        let fsfs = IsolatedFsfs::new(temp.path(), model_root.clone());

        for (label, model_id, manifest, install_dir) in [
            (
                "verify-potion-cli",
                "potion-multilingual-128m",
                ModelManifest::potion_128m(),
                "potion-multilingual-128M",
            ),
            (
                "verify-minilm-cli",
                "all-minilm-l6-v2",
                ModelManifest::minilm_v2(),
                "all-MiniLM-L6-v2",
            ),
        ] {
            let receipt_path = model_root.join(install_dir).join(".verified");
            fs::write(
                &receipt_path,
                br#"{"schema_version":0,"invalidated_by":"default-build-e2e"}"#,
            )
            .map_err(|error| {
                format!(
                    "invalidate prior receipt at {}: {error}",
                    receipt_path.display()
                )
            })?;
            let verify_outcome = fsfs.run(
                temp.path(),
                label,
                ["download-models", model_id, "--verify", "--format", "json"],
                QUICKSTART_TIMEOUT,
            );
            let verify_envelope =
                parse_success_envelope("default-build CLI model verification", &verify_outcome);
            assert_eq!(
                verify_envelope
                    .pointer("/data/operation")
                    .and_then(Value::as_str),
                Some("verify"),
                "CLI verify must report its operation: {verify_envelope}"
            );
            let verified_entry = verify_envelope
                .pointer("/data/models/0")
                .expect("CLI verify response must contain its selected model");
            assert_eq!(
                verified_entry.get("state").and_then(Value::as_str),
                Some("verified"),
                "CLI verify must report a verified state: {verified_entry}"
            );
            assert_eq!(
                verified_entry.get("verified").and_then(Value::as_bool),
                Some(true),
                "CLI verify must attest the verified result: {verified_entry}"
            );
            assert!(
                is_verification_cached(&manifest, &model_root.join(install_dir)),
                "explicit CLI verification must refresh a current full-SHA receipt for {model_id}"
            );
            eprintln!(
                "[default-build-e2e] stage=cli-model-verify event=verified model={model_id} exit=0 ok=true receipt_refreshed=true"
            );
        }

        let index_outcome = fsfs.run(
            temp.path(),
            "index",
            [
                "index",
                corpus.to_str().expect("UTF-8 corpus path"),
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        let index_envelope = parse_success_envelope("default-build index", &index_outcome);
        eprintln!(
            "[default-build-e2e] stage=index-one-shot event=verified timed_out=false exit=0 elapsed_ms={} watch_requested=false",
            index_outcome.elapsed.as_millis()
        );

        let sentinel_path = index.join("index_sentinel.json");
        let sentinel: Value = serde_json::from_slice(
            &fs::read(&sentinel_path).expect("read durable index completion sentinel"),
        )
        .expect("parse durable index completion sentinel");
        assert_index_completion(
            &index_envelope,
            &sentinel,
            QUICKSTART_DOCUMENT_COUNT,
            "json",
        );
        assert_eq!(
            sentinel.get("generation_complete").and_then(Value::as_bool),
            Some(true),
            "the completed generation must be explicitly sealed: {sentinel}"
        );
        assert_eq!(
            sentinel
                .get("indexed_files")
                .and_then(Value::as_u64)
                .and_then(|count| usize::try_from(count).ok()),
            Some(QUICKSTART_DOCUMENT_COUNT),
            "all quickstart files must be durably indexed: {sentinel}"
        );
        assert!(
            index.join("CURRENT").is_file() || index.join("lexical/CURRENT").is_file(),
            "the Quill lexical generation must publish a CURRENT pointer"
        );
        verify_index_output_formats(&fsfs, temp.path(), &corpus, &index);
        verify_fast_only_policy(&fsfs, temp.path(), &corpus, &index);

        let vector_path = index.join("vector/index.fsvi");
        let vector_index =
            VectorIndex::open_read_only(&vector_path).expect("inspect durable quickstart FSVI");
        assert_eq!(
            vector_index.record_count(),
            QUICKSTART_DOCUMENT_COUNT,
            "one vector per fixture file"
        );
        assert!(
            vector_index.dimension() > 0,
            "semantic vectors need dimensions"
        );
        let embedder_id = vector_index.embedder_id().to_ascii_lowercase();
        assert!(
            embedder_id.contains("potion") || embedder_id.contains("model2vec"),
            "the fast FSVI must name a real semantic producer, got {embedder_id}"
        );
        assert!(
            !embedder_id.contains("hash") && !embedder_id.contains("fnv1a"),
            "the durable FSVI must never carry a hash-control identity: {embedder_id}"
        );
        eprintln!(
            "[default-build-e2e] stage=durable-vector event=verified records={} dimension={} embedder_id={embedder_id}",
            vector_index.record_count(),
            vector_index.dimension()
        );

        // The two-tier promise: a standard `fsfs index` with both registered
        // models present publishes a quality-tier generation beside the fast
        // one, in its own (384-d MiniLM) space, covering the same documents.
        let quality_path = index.join("vector/quality.fsvi");
        assert!(
            quality_path.is_file(),
            "a standard index with the quality model present must publish {}",
            quality_path.display()
        );
        let quality_index =
            VectorIndex::open_read_only(&quality_path).expect("inspect durable quality FSVI");
        assert_eq!(
            quality_index.record_count(),
            QUICKSTART_DOCUMENT_COUNT,
            "one quality vector per fixture file"
        );
        assert_eq!(
            quality_index.dimension(),
            384,
            "the quality tier is the 384-d MiniLM space"
        );
        let quality_embedder_id = quality_index.embedder_id().to_ascii_lowercase();
        assert!(
            quality_embedder_id.contains("minilm"),
            "the quality FSVI must name the MiniLM producer, got {quality_embedder_id}"
        );
        assert_ne!(
            quality_embedder_id, embedder_id,
            "the quality tier must be a different embedding space from the fast tier"
        );
        eprintln!(
            "[default-build-e2e] stage=durable-quality-vector event=verified records={} dimension={} embedder_id={quality_embedder_id}",
            quality_index.record_count(),
            quality_index.dimension()
        );

        let status_outcome = fsfs.run(
            temp.path(),
            "status-verified-models",
            [
                "status",
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        let status_envelope =
            parse_success_envelope("default-build verified-model status", &status_outcome);
        for tier in ["fast", "quality"] {
            let status = model_status(&status_envelope, tier);
            assert_eq!(
                status.get("verification_state").and_then(Value::as_str),
                Some("verified"),
                "status must report manifest-verified {tier} bytes: {status}"
            );
            assert_eq!(
                status.get("cached").and_then(Value::as_bool),
                Some(true),
                "status cached=true is reserved for verified {tier} bytes: {status}"
            );
        }
        eprintln!(
            "[default-build-e2e] stage=status-verified event=verified exit=0 ok=true fast=verified quality=verified"
        );

        let doctor_outcome = fsfs.run(
            temp.path(),
            "doctor-verified-models",
            [
                "doctor",
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        let doctor_envelope =
            parse_success_envelope("default-build verified-model doctor", &doctor_outcome);
        assert_eq!(
            doctor_envelope
                .pointer("/data/fail_count")
                .and_then(Value::as_u64),
            Some(0),
            "loader-probed doctor must have no failures: {doctor_envelope}"
        );
        for check_name in ["model.fast", "model.quality"] {
            let check = doctor_envelope
                .pointer("/data/checks")
                .and_then(Value::as_array)
                .and_then(|checks| {
                    checks
                        .iter()
                        .find(|check| check.get("name").and_then(Value::as_str) == Some(check_name))
                })
                .unwrap_or_else(|| {
                    eprintln!(
                        "[default-build-e2e] stage=doctor-success event=missing-check check={check_name} envelope={doctor_envelope}"
                    );
                    &doctor_envelope
                });
            assert_eq!(
                check.get("verdict").and_then(Value::as_str),
                Some("pass"),
                "doctor must loader-probe {check_name}: {check}"
            );
            assert!(
                check
                    .get("detail")
                    .and_then(Value::as_str)
                    .is_some_and(|detail| detail.contains("manifest-verified and loadable")),
                "doctor must distinguish loader readiness from manifest status: {check}"
            );
        }
        eprintln!(
            "[default-build-e2e] stage=doctor-success event=verified exit=0 ok=true fail_count=0 loaders=fast,quality"
        );

        let semantic_only_index = temp.path().join("semantic-only-index");
        let semantic_only_vector = semantic_only_index.join("vector/index.fsvi");
        fs::create_dir_all(
            semantic_only_vector
                .parent()
                .expect("semantic-only vector path has a parent"),
        )
        .expect("create semantic-only vector directory");
        fs::copy(&vector_path, &semantic_only_vector)
            .expect("copy the sealed FSVI into a vector-only search root");
        assert!(
            !semantic_only_index.join("lexical").exists()
                && !semantic_only_index.join("CURRENT").exists(),
            "the semantic-only lane must have no lexical generation or CURRENT pointer"
        );
        eprintln!(
            "[default-build-e2e] stage=semantic-only-fixture event=verified lexical_root_present=false vector_present=true records={}",
            vector_index.record_count()
        );

        let document_terms = RETRY_DOCUMENT
            .split(|character: char| !character.is_alphanumeric())
            .filter(|term| !term.is_empty())
            .map(str::to_ascii_lowercase)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            QUICKSTART_DOCUMENT_COUNT > SEARCH_LIMIT.parse::<usize>().expect("numeric limit"),
            "the semantic lane needs more documents than its result limit"
        );
        for (ordinal, paraphrase) in SEMANTIC_PARAPHRASES.into_iter().enumerate() {
            let query_terms = paraphrase
                .split(|character: char| !character.is_alphanumeric())
                .filter(|term| !term.is_empty())
                .map(str::to_ascii_lowercase)
                .collect::<std::collections::BTreeSet<_>>();
            let lexical_overlap = document_terms
                .intersection(&query_terms)
                .cloned()
                .collect::<Vec<_>>();
            assert!(
                lexical_overlap.is_empty(),
                "semantic paraphrase {ordinal} must share zero exact terms with retry.md; overlap={lexical_overlap:?}"
            );

            let label = format!("semantic-only-search-{ordinal}");
            let semantic_search_outcome = fsfs.run(
                temp.path(),
                &label,
                [
                    "search",
                    paraphrase,
                    "--index-dir",
                    semantic_only_index
                        .to_str()
                        .expect("UTF-8 semantic-only index path"),
                    "--limit",
                    SEARCH_LIMIT,
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            assert_finished_successfully(
                "default-build semantic paraphrase search",
                &semantic_search_outcome,
            );
            let semantic_envelope: Value = serde_json::from_str(&semantic_search_outcome.stdout)
                .expect("parse semantic paraphrase search JSON envelope");
            assert_eq!(
                semantic_envelope.get("ok").and_then(Value::as_bool),
                Some(true),
                "semantic paraphrase search envelope must report success: {semantic_envelope}"
            );
            let semantic_hits = semantic_envelope
                .pointer("/data/hits")
                .and_then(Value::as_array)
                .expect("semantic paraphrase search data.hits array");
            let semantic_first = semantic_hits
                .first()
                .expect("semantic paraphrase search must return a hit");
            assert!(
                semantic_first
                    .get("path")
                    .and_then(Value::as_str)
                    .is_some_and(|path| path.ends_with("retry.md")),
                "semantic paraphrase {ordinal} must rank the known relevant document first: {semantic_first}"
            );
            assert_eq!(
                semantic_first.get("semantic_rank").and_then(Value::as_u64),
                Some(0),
                "retry.md must be semantic rank zero for paraphrase {ordinal}: {semantic_first}"
            );
            assert_eq!(
                semantic_first.get("lexical_rank").and_then(Value::as_u64),
                None,
                "a vector-only root must not manufacture a lexical rank: {semantic_first}"
            );
            assert_eq!(
                semantic_first
                    .get("in_both_sources")
                    .and_then(Value::as_bool),
                Some(false),
                "a vector-only root must report a semantic-only hit: {semantic_first}"
            );
            assert!(
                semantic_hits
                    .iter()
                    .all(|hit| hit.get("lexical_rank").is_none_or(Value::is_null)),
                "the semantic-only lane must have no lexical-ranked hits: {semantic_hits:?}"
            );
            eprintln!(
                "[default-build-e2e] stage=real-model-semantic-only event=verified query_ordinal={ordinal} query={paraphrase:?} path=retry.md semantic_rank=0 lexical_rank=absent in_both_sources=false exact_term_overlap=0"
            );
        }

        let hybrid_search_outcome = fsfs.run(
            temp.path(),
            "hybrid-search",
            [
                "search",
                "How should a network client recover from transient failures using exponential backoff, bounded retries, and random jitter?",
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--limit",
                SEARCH_LIMIT,
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        assert_finished_successfully("default-build hybrid search", &hybrid_search_outcome);
        let envelope: Value = serde_json::from_str(&hybrid_search_outcome.stdout)
            .expect("parse hybrid search JSON envelope");
        assert_eq!(
            envelope.get("ok").and_then(Value::as_bool),
            Some(true),
            "search envelope must report success: {envelope}"
        );
        assert_eq!(
            envelope.pointer("/data/phase").and_then(Value::as_str),
            Some("refined"),
            "a generation with a quality tier must finish in the REFINED phase: {envelope}"
        );
        let hits = envelope
            .pointer("/data/hits")
            .and_then(Value::as_array)
            .expect("search envelope data.hits array");
        let first = hits.first().expect("hybrid search must return a hit");
        assert!(
            first
                .get("path")
                .and_then(Value::as_str)
                .is_some_and(|path| path.ends_with("retry.md")),
            "the known relevant document must rank first: {first}"
        );
        assert_eq!(
            first.get("lexical_rank").and_then(Value::as_u64),
            Some(0),
            "the winning result must lead the lexical arm: {first}"
        );
        assert_eq!(
            first.get("semantic_rank").and_then(Value::as_u64),
            Some(0),
            "the winning result must lead the semantic arm: {first}"
        );
        assert_eq!(
            first.get("in_both_sources").and_then(Value::as_bool),
            Some(true),
            "the winning result must be a genuine hybrid hit: {first}"
        );
        eprintln!(
            "[default-build-e2e] stage=hybrid-control event=verified path=retry.md lexical_rank=0 semantic_rank=0 in_both_sources=true"
        );

        // Model loading follows the generation, not the process: a search over
        // the fast-only fixture (no quality tier) must never open the quality
        // model, while a search over the two-tier generation opens it exactly
        // once and finishes in REFINED. Both legs read the same log line, so
        // the negative is real rather than vacuous.
        let fast_only_outcome = fsfs.run_with_env(
            temp.path(),
            "search-loads-fast-tier-only",
            [
                "search",
                "bounded retries with exponential backoff",
                "--index-dir",
                semantic_only_index
                    .to_str()
                    .expect("UTF-8 semantic-only index path"),
                "--no-daemon",
                "--limit",
                SEARCH_LIMIT,
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
            &[("FRANKENSEARCH_LOG", "info")],
        );
        assert_finished_successfully("fast-tier-only search", &fast_only_outcome);
        assert!(
            fast_only_outcome.stderr.contains("Model2Vec model loaded"),
            "the fast tier must be loaded for a semantic search; stderr:\n{}",
            fast_only_outcome.stderr
        );
        assert!(
            !fast_only_outcome.stderr.contains("FastEmbed model loaded"),
            "a search over a fast-only generation must not load the quality model; stderr:\n{}",
            fast_only_outcome.stderr
        );
        let fast_only_envelope: Value = serde_json::from_str(&fast_only_outcome.stdout)
            .expect("parse fast-only search envelope");
        assert_eq!(
            fast_only_envelope
                .pointer("/data/phase")
                .and_then(Value::as_str),
            Some("initial"),
            "a fast-only generation serves INITIAL only: {fast_only_envelope}"
        );

        let two_tier_outcome = fsfs.run_with_env(
            temp.path(),
            "search-loads-quality-tier-once",
            [
                "search",
                "bounded retries with exponential backoff",
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--no-daemon",
                "--stream",
                "--limit",
                SEARCH_LIMIT,
                "--format",
                "jsonl",
            ],
            QUICKSTART_TIMEOUT,
            &[
                ("FRANKENSEARCH_LOG", "info"),
                ("FSFS_DISABLE_QUERY_CACHE", "1"),
            ],
        );
        assert_finished_successfully("two-tier streamed search", &two_tier_outcome);
        assert_eq!(
            two_tier_outcome
                .stderr
                .matches("FastEmbed model loaded")
                .count(),
            1,
            "a two-tier search must open the quality model exactly once; stderr:\n{}",
            two_tier_outcome.stderr
        );
        let stream_reason_codes = two_tier_outcome
            .stdout
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter_map(|frame| {
                frame
                    .pointer("/payload/reason_code")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect::<Vec<_>>();
        assert!(
            stream_reason_codes
                .iter()
                .any(|code| code == "query.stream.initial_ready"),
            "stream must announce INITIAL: {stream_reason_codes:?}"
        );
        assert!(
            stream_reason_codes
                .iter()
                .any(|code| code == "query.stream.refined_ready"),
            "stream must announce REFINED over a two-tier generation: {stream_reason_codes:?}"
        );
        eprintln!(
            "[default-build-e2e] stage=quality-tier-load event=verified fast_only_quality_loaded=false two_tier_quality_loads=1 refined_ready=true"
        );
        #[cfg(feature = "semantic-loaders")]
        verify_real_blend_and_deadline(&fsfs, temp.path(), &index);

        // Reality check 2026-09-01, G2: the auto-spawned query daemon must
        // outlive the search that spawned it (so the next search is warm), be
        // stoppable through its protocol, and reclaim itself after its idle
        // timeout so it can never linger as an orphan.
        #[cfg(unix)]
        {
            use std::io::Write as _;
            use std::os::unix::net::UnixStream;

            // AF_UNIX paths are limited to ~107 bytes, so the sockets live in
            // the runtime dir (or the system temp dir), never under the
            // fixture root, which can be arbitrarily deep.
            let socket_dir = std::env::var_os("XDG_RUNTIME_DIR")
                .map(PathBuf::from)
                .filter(|dir| dir.is_dir())
                .unwrap_or_else(std::env::temp_dir);
            let socket = socket_dir.join(format!("fsfs-e2e-query-{}.sock", std::process::id()));
            let idle_socket = socket_dir.join(format!("fsfs-e2e-idle-{}.sock", std::process::id()));
            assert!(
                socket.as_os_str().len() < 100 && idle_socket.as_os_str().len() < 100,
                "daemon socket path too long for AF_UNIX; set XDG_RUNTIME_DIR or TMPDIR to a short directory: {}",
                socket.display()
            );
            let socket_arg = socket.to_str().expect("UTF-8 socket path");
            let index_arg = index.to_str().expect("UTF-8 index path");
            let daemon_args = [
                "search",
                "bounded retries with exponential backoff",
                "--index-dir",
                index_arg,
                "--daemon-socket",
                socket_arg,
                "--limit",
                SEARCH_LIMIT,
                "--format",
                "json",
            ];

            let cold = fsfs.run(
                temp.path(),
                "daemon-search-cold",
                daemon_args,
                QUICKSTART_TIMEOUT,
            );
            assert_finished_successfully("daemon-backed cold search", &cold);

            // The daemon must still accept connections after its spawning
            // search has exited (the old PDEATHSIG hook killed it here).
            let mut survived = false;
            for _ in 0..40 {
                if UnixStream::connect(&socket).is_ok() {
                    survived = true;
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
            assert!(
                survived,
                "the query daemon must outlive the search that spawned it (socket {})",
                socket.display()
            );

            let warm = fsfs.run(
                temp.path(),
                "daemon-search-warm",
                daemon_args,
                QUICKSTART_TIMEOUT,
            );
            assert_finished_successfully("daemon-backed warm search", &warm);
            let warm_envelope: Value =
                serde_json::from_str(&warm.stdout).expect("parse warm daemon search envelope");
            assert_eq!(
                warm_envelope.get("ok").and_then(Value::as_bool),
                Some(true),
                "warm daemon search must succeed: {warm_envelope}"
            );
            assert!(
                warm.elapsed * 2 < cold.elapsed,
                "a warm daemon-backed search must not pay the model load again: cold={:?} warm={:?}",
                cold.elapsed,
                warm.elapsed
            );
            eprintln!(
                "[default-build-e2e] stage=daemon-survives-spawner event=verified cold_ms={} warm_ms={}",
                cold.elapsed.as_millis(),
                warm.elapsed.as_millis()
            );

            // Protocol stop reclaims the socket.
            let mut quit = UnixStream::connect(&socket).expect("connect to stop the daemon");
            quit.write_all(b"quit\n").expect("send quit to the daemon");
            drop(quit);
            let mut reclaimed = false;
            for _ in 0..200 {
                if !socket.exists() {
                    reclaimed = true;
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
            assert!(reclaimed, "the daemon must unlink its socket after quit");

            // Idle timeout: a daemon with a 1.5 s idle budget exits on its own
            // and leaves no socket behind. Planted negative: the run would
            // time out (and fail) if the daemon ignored its idle budget.
            let idle_outcome = fsfs.run(
                temp.path(),
                "daemon-idle-exit",
                [
                    "serve",
                    "--daemon-socket",
                    idle_socket.to_str().expect("UTF-8 idle socket path"),
                    "--idle-timeout-ms",
                    "1500",
                    "--index-dir",
                    index_arg,
                    "--format",
                    "jsonl",
                ],
                Duration::from_secs(90),
            );
            assert_finished_successfully("idle daemon self-exit", &idle_outcome);
            assert!(
                !idle_socket.exists(),
                "an idle-expired daemon must unlink its socket"
            );
            eprintln!(
                "[default-build-e2e] stage=daemon-idle-exit event=verified idle_timeout_ms=1500 wall_ms={}",
                idle_outcome.elapsed.as_millis()
            );
        }
        Ok(())
    }
}

#[cfg(feature = "embedded-models")]
#[test]
fn embedded_release_profile_retains_semantic_loaders() {
    eprintln!(
        "embedded-models is a supported release profile; the stock-default quickstart contract is exercised without this feature"
    );
    assert!(cfg!(feature = "semantic-loaders"));
}
