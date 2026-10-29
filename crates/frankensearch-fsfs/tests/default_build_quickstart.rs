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

    #[cfg(feature = "semantic-loaders")]
    use frankensearch_embed::model_manifest::is_verification_cached;
    use frankensearch_embed::model_manifest::{ModelManifest, ModelTier};
    #[cfg(feature = "semantic-loaders")]
    use frankensearch_index::VectorIndex;
    use serde_json::Value;

    const FAILURE_TIMEOUT: Duration = Duration::from_secs(30);
    #[cfg(feature = "semantic-loaders")]
    const QUICKSTART_TIMEOUT: Duration = Duration::from_secs(240);
    #[cfg(feature = "semantic-loaders")]
    const QUICKSTART_DOCUMENT_COUNT: usize = 10;
    const RETRY_DOCUMENT: &str = "Recover transient network failures with exponential backoff, bounded retries, and random jitter.";
    #[cfg(feature = "semantic-loaders")]
    const SEMANTIC_PARAPHRASES: [&str; 2] = [
        "staggered reconnects after brief outages",
        "delayed reattempts following temporary disruptions",
    ];
    #[cfg(feature = "semantic-loaders")]
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
            "[default-build-e2e] stage=stock-default-contract event=start lane={lane} binary={} binary_origin={binary_origin} harness_profile={harness_profile} semantic_loaders={} semantic_native={} embedded_models={}",
            verified_executable.display(),
            cfg!(feature = "semantic-loaders"),
            cfg!(feature = "semantic-native"),
            cfg!(feature = "embedded-models")
        );
    }

    #[derive(Debug)]
    struct IsolatedFsfs {
        binary: PathBuf,
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
                binary: fsfs_binary(),
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
            // ubs:ignore -- fsfs_binary validates the explicit test executable or uses Cargo's binary.
            let mut command = Command::new(&self.binary);
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
                .env_remove("FRANKENSEARCH_RANKING_PRIORS")
                .env_remove("FSFS_RANKING_PRIORS")
                .env_remove("FRANKENSEARCH_INDEXING_UPGRADE_REUSE")
                .env_remove("FSFS_INDEXING_UPGRADE_REUSE")
                .env_remove("FSFS_DISABLE_QUERY_CACHE")
                .env_remove("FRANKENSEARCH_API_PROVIDER")
                .env_remove("FRANKENSEARCH_API_MODEL")
                .env_remove("FRANKENSEARCH_API_DIMENSION")
                .env_remove("FRANKENSEARCH_API_IDENTITY_JSON")
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
            let verified_executable = &self.binary;
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

    /// The directory holding the active generation's files: `index` itself on
    /// the legacy layout, or the generation `FSFS-CURRENT` names in a store of
    /// complete generations (`FSFS_COMPLETE_GENERATIONS=1`), whose layout
    /// inside is the same.
    /// Whether `index` is a store of complete generations.
    #[cfg(all(unix, any(feature = "semantic-loaders", feature = "rerank")))]
    fn is_complete_store(index: &Path) -> bool {
        index.join("FSFS-CURRENT").exists()
    }

    /// Ask the query daemon on `socket` to stop. A client's `quit` stops the
    /// legacy daemon; a complete store's daemon takes only its versioned
    /// control message (what `fsfs daemon --stop` sends), so that no ordinary
    /// query can stop it.
    #[cfg(all(unix, any(feature = "semantic-loaders", feature = "rerank")))]
    fn request_daemon_stop(socket: &Path, complete: bool) {
        use std::io::{Read as _, Write as _};
        if let Ok(mut stream) = std::os::unix::net::UnixStream::connect(socket) {
            let _ = stream.set_read_timeout(Some(FAILURE_TIMEOUT));
            let _ = stream.set_write_timeout(Some(FAILURE_TIMEOUT));
            let request: &[u8] = if complete {
                b"{\"fsfs_complete_daemon\":\"shutdown\",\"version\":1}\n"
            } else {
                b"quit\n"
            };
            let _ = stream.write_all(request);
            let _ = stream.shutdown(std::net::Shutdown::Write);
            let _ = stream.read_to_end(&mut Vec::new());
        }
    }

    #[cfg(any(feature = "semantic-loaders", feature = "rerank"))]
    fn active_generation_root(index: &Path) -> PathBuf {
        let Ok(pointer) = fs::read_to_string(index.join("FSFS-CURRENT")) else {
            return index.to_path_buf();
        };
        let id = pointer
            .lines()
            .nth(1)
            .expect("FSFS-CURRENT names the active generation on its second line");
        index.join("generations").join(id)
    }

    #[cfg(feature = "semantic-loaders")]
    fn assert_index_completion(envelope: &Value, sentinel: &Value, count: usize, format: &str) {
        assert_eq!(envelope["ok"], true);
        assert_eq!(envelope["meta"]["command"], "index");
        assert_eq!(envelope["meta"]["format"], format);
        assert!(envelope["meta"]["duration_ms"].is_u64());
        assert_index_data(&envelope["data"], sentinel, count);
    }

    #[cfg(feature = "semantic-loaders")]
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

    #[cfg(feature = "semantic-loaders")]
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
            let sentinel: Value = serde_json::from_slice(
                &fs::read(active_generation_root(index).join("index_sentinel.json")).unwrap(),
            )
            .unwrap();
            if format == "table" {
                assert!(outcome.stdout.starts_with("Discovered 10 file(s)"));
                // JSONL already indexed the same corpus above. The next format
                // must report carrying all ten files, rather than rebuilding
                // their document embeddings just to change presentation.
                assert!(
                    outcome.stdout.contains(
                        "Indexed 10 file(s) (discovered 10, skipped 0; 10 unchanged, not reread)"
                    ),
                    "warm table output must retain the file counts and report reuse: {}",
                    outcome.stdout
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
            let quality = VectorIndex::open_read_only(
                &active_generation_root(index).join("vector/quality.fsvi"),
            )
            .unwrap();
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
            let sentinel: Value = serde_json::from_slice(
                &fs::read(active_generation_root(&empty_index).join("index_sentinel.json"))
                    .unwrap(),
            )
            .unwrap();
            assert_index_completion(&envelope, &sentinel, 0, format);
        }
        eprintln!(
            "[default-build-e2e] stage=index-formats json=jsonl=toon=csv=table:verified nonempty=10 empty=0 counts=published-sentinel"
        );
    }

    #[cfg(feature = "semantic-loaders")]
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
            !active_generation_root(&fast_index)
                .join("vector/quality.fsvi")
                .exists(),
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

    #[cfg(all(unix, feature = "semantic-loaders"))]
    fn verify_real_fast_model_cache_isolation(fsfs: &IsolatedFsfs, root: &Path, index: &Path) {
        let query = "producer policy controls retained network retries";
        let selected = root.join("cached-fast-model.toml");
        fs::write(
            &selected,
            "[indexing]\nfast_model = \"potion-multilingual-128M\"\n",
        )
        .unwrap();
        let search = |label: &str, config: &Path| {
            fsfs.run_with_env(
                root,
                label,
                [
                    "search",
                    query,
                    "--config",
                    config.to_str().unwrap(),
                    "--index-dir",
                    index.to_str().unwrap(),
                    "--no-daemon",
                    "--fast-only",
                    "--limit",
                    "3",
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
                &[("FRANKENSEARCH_LOG", "info")],
            )
        };
        let cold = search("fast-model-cache-cold", &selected);
        let original = parse_success_envelope("real fast-model cache seed", &cold);
        assert!(cold.stderr.contains("Model2Vec model loaded"), "{cold:?}");
        assert!(
            original["data"]["hits"]
                .as_array()
                .is_some_and(|hits| !hits.is_empty())
        );
        let matching_records = || {
            fs::read_dir(index.join("query_cache"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter_map(|path| {
                    let bytes = fs::read(&path).ok()?;
                    let record: Value = serde_json::from_slice(&bytes).ok()?;
                    (record["key"]["query"] == query).then_some((path, bytes, record))
                })
                .collect::<Vec<_>>()
        };
        let mut records = matching_records();
        assert_eq!(records.len(), 1, "the real query must persist one answer");
        let (cache_path, cache_bytes, record) = records.pop().unwrap();
        assert_eq!(record["schema_version"], "fsfs.search.cache.v21");
        assert_eq!(record["key"]["fast_model"], "potion-multilingual-128m");

        let warm = search("fast-model-cache-warm", &selected);
        let replayed = parse_success_envelope("real fast-model cache replay", &warm);
        assert_eq!(replayed["data"], original["data"]);
        assert!(
            !warm.stderr.contains("Model2Vec model loaded"),
            "the positive control must actually reuse the persisted answer: {warm:?}"
        );

        for (label, model) in [
            ("8m", "potion-base-8m"),
            ("32m", "potion-base-32m"),
            ("unknown", "missing-fast-model"),
            ("malformed", "potion/multilingual/128m"),
        ] {
            let config = root.join(format!("cached-fast-model-{label}.toml"));
            fs::write(&config, format!("[indexing]\nfast_model = \"{model}\"\n")).unwrap();
            let refused = search(&format!("fast-model-cache-refuse-{label}"), &config);
            assert!(!refused.timed_out, "{refused:?}");
            assert!(
                !refused.status.success(),
                "changing the producer must not replay the old answer: {refused:?}"
            );
            let error: Value = serde_json::from_str(&refused.stdout).unwrap();
            assert_eq!(error["ok"], false, "{error}");
            assert!(error["data"].is_null(), "{error}");
            assert!(
                error["error"]["code"]
                    .as_str()
                    .is_some_and(|code| !code.is_empty()),
                "producer mismatch must remain a typed failure: {error}"
            );
            if matches!(label, "unknown" | "malformed") {
                assert_eq!(error["error"]["code"], "invalid_config", "{error}");
                assert!(
                    refused.combined_output().contains("indexing.fast_model"),
                    "{refused:?}"
                );
            }
            assert_eq!(fs::read(&cache_path).unwrap(), cache_bytes);
            assert_eq!(matching_records().len(), 1);
        }

        // Case and surrounding whitespace identify the same registered producer.
        // Their canonical cache key should retain the valid existing answer.
        let alias = root.join("cached-fast-model-alias.toml");
        fs::write(
            &alias,
            "[indexing]\nfast_model = \"  POTION-MULTILINGUAL-128M  \"\n",
        )
        .unwrap();
        for (label, config) in [("alias", &alias), ("restored", &selected)] {
            let outcome = search(&format!("fast-model-cache-{label}"), config);
            let envelope = parse_success_envelope("retained fast-model answer", &outcome);
            assert_eq!(envelope["data"], original["data"]);
            assert!(
                !outcome.stderr.contains("Model2Vec model loaded"),
                "{outcome:?}"
            );
        }
        assert_eq!(fs::read(cache_path).unwrap(), cache_bytes);
        eprintln!(
            "[default-build-e2e] stage=fast-model-answer-cache real_potion=true warm_replay=true changed_producer_refused=true malformed_selection_refused=true canonical_alias_reused=true original_preserved=true"
        );
    }

    #[cfg(feature = "semantic-loaders")]
    fn verify_selected_quality_never_falls_back_to_cached_minilm(
        root: &Path,
        source_models: &Path,
    ) {
        let root = root.join("selected-quality");
        let models = root.join("models");
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        fs::write(corpus.join("retry.md"), RETRY_DOCUMENT).unwrap();
        // Keep the missing/corrupt selected-model cases independent of any
        // additional models the release qualification cache happens to hold.
        for (manifest, directory) in [
            (ModelManifest::potion_128m(), "potion-multilingual-128M"),
            (ModelManifest::minilm_v2(), "all-MiniLM-L6-v2"),
        ] {
            for artifact in &manifest.files {
                let source = source_models.join(directory).join(&artifact.name);
                let target = models.join(directory).join(&artifact.name);
                fs::create_dir_all(target.parent().unwrap()).unwrap();
                // Independent files keep this negative fixture from changing
                // the source inode's ctime and invalidating its live receipt.
                fs::copy(source, target).unwrap();
            }
            fs::copy(
                source_models.join(directory).join(".verified"),
                models.join(directory).join(".verified"),
            )
            .unwrap();
        }
        let fsfs = IsolatedFsfs::new(&root, models.clone());
        for (label, selected, reason) in [
            ("control", "all-MiniLM-L6-v2", "vector.quality_tier.built"),
            (
                "missing",
                "nomic-embed-text-v1.5",
                "vector.quality_tier.skipped.model_unavailable",
            ),
            (
                "corrupt",
                "nomic-embed-text-v1.5",
                "vector.quality_tier.skipped.model_unavailable",
            ),
            ("disabled", "", "vector.quality_tier.skipped.fast_only"),
        ] {
            if label == "corrupt" {
                for artifact in &ModelManifest::nomic_embed().files {
                    let path = models.join("nomic-embed-text-v1.5").join(&artifact.name);
                    fs::create_dir_all(path.parent().unwrap()).unwrap();
                    fs::write(path, b"unverified selected quality artifact").unwrap();
                }
            }
            let config = root.join(format!("{label}.toml"));
            fs::write(
                &config,
                format!(
                    "[indexing]\nquality_model = \"{selected}\"\n[search]\nfast_only = false\n"
                ),
            )
            .unwrap();
            let index = root.join(format!("index-{label}"));
            let outcome = fsfs.run(
                &root,
                &format!("quality-selection-{label}"),
                [
                    "index",
                    corpus.to_str().unwrap(),
                    "--config",
                    config.to_str().unwrap(),
                    "--index-dir",
                    index.to_str().unwrap(),
                    "--offline",
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            let envelope = parse_success_envelope("exact quality selection", &outcome);
            assert_eq!(envelope["data"]["generation_complete"], true);
            assert_eq!(envelope["data"]["indexed_files"], 1);
            assert!(
                envelope["data"]["reason_codes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|value| value == reason),
                "{envelope}"
            );
            let generation = active_generation_root(&index);
            assert!(generation.join("vector/index.fsvi").is_file());
            if label == "control" {
                let quality = VectorIndex::open_read_only(&generation.join("vector/quality.fsvi"))
                    .expect("the real MiniLM control must build a quality generation");
                assert_eq!(quality.record_count(), 1);
                assert!(
                    quality
                        .embedder_id()
                        .to_ascii_lowercase()
                        .contains("minilm")
                );
                continue;
            }
            assert!(envelope["data"]["quality_generation"].is_null());
            assert!(
                !generation.join("vector/quality.fsvi").exists(),
                "a cached MiniLM must never replace an unavailable or disabled selection"
            );
            if matches!(label, "missing" | "corrupt") {
                assert!(
                    outcome.stderr.contains("quality-tier model unavailable"),
                    "{outcome:?}"
                );
                let status = fsfs.run(
                    &root,
                    &format!("quality-selection-status-{label}"),
                    [
                        "status",
                        "--config",
                        config.to_str().unwrap(),
                        "--index-dir",
                        index.to_str().unwrap(),
                        "--format",
                        "json",
                    ],
                    QUICKSTART_TIMEOUT,
                );
                let inspected = parse_success_envelope("selected quality cache status", &status);
                let quality = model_status(&inspected, "quality");
                assert_eq!(quality["name"], "nomic-embed-text-v1.5");
                assert_eq!(quality["cached"], false);
                assert_eq!(
                    quality["verification_state"],
                    if label == "missing" {
                        "missing"
                    } else {
                        "mismatch"
                    }
                );
            }
            let searched = fsfs.run(
                &root,
                &format!("quality-selection-search-{label}"),
                [
                    "search",
                    SEMANTIC_PARAPHRASES[0],
                    "--config",
                    config.to_str().unwrap(),
                    "--index-dir",
                    index.to_str().unwrap(),
                    "--offline",
                    "--no-daemon",
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            let answer =
                parse_success_envelope("selected quality unavailable, real fast search", &searched);
            assert_eq!(answer["data"]["phase"], "initial", "{answer}");
            assert!(!answer["data"]["hits"].as_array().unwrap().is_empty());
        }
        eprintln!(
            "[default-build-e2e] stage=quality-selection real_minilm_control=true missing_selected_skips_quality=true corrupt_selected_skips_quality=true empty_selection_disables_quality=true real_fast_search=true"
        );
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

    #[cfg(feature = "semantic-loaders")]
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
    #[cfg(unix)]
    fn verify_real_progressive_daemon(fsfs: &IsolatedFsfs, root: &Path, index: &Path) {
        use std::io::{BufRead as _, Read as _, Write as _};
        use std::os::unix::net::UnixStream;
        struct Owner {
            child: std::process::Child,
            socket: PathBuf,
            complete: bool,
        }
        impl Drop for Owner {
            fn drop(&mut self) {
                request_daemon_stop(&self.socket, self.complete);
                let deadline = Instant::now() + FAILURE_TIMEOUT;
                while Instant::now() < deadline {
                    if let Ok(Some(status)) = self.child.try_wait() {
                        let _ = fs::remove_file(&self.socket);
                        let _ = fs::remove_file(self.socket.with_extension("lock"));
                        if !thread::panicking() {
                            assert!(status.success());
                        }
                        return;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                let _ = self.child.kill();
                let _ = self.child.wait();
                let _ = fs::remove_file(&self.socket);
                let _ = fs::remove_file(self.socket.with_extension("lock"));
                assert!(
                    thread::panicking(),
                    "progressive daemon failed graceful shutdown"
                );
            }
        }
        let socket =
            std::env::temp_dir().join(format!("fsfs-progressive-{}.sock", std::process::id()));
        let config = root.join("progressive.toml");
        fs::write(&config, "[search]\nquality_timeout_ms = 5000\n").unwrap();
        let log = fsfs.log_root.join("progressive-daemon.stderr.log");
        let start = Instant::now();
        let child = fsfs
            .command(root)
            .args([
                "serve",
                "--daemon",
                "--daemon-socket",
                socket.to_str().unwrap(),
                "--config",
                config.to_str().unwrap(),
                "--index-dir",
                index.to_str().unwrap(),
            ])
            .env("FRANKENSEARCH_LOG", "info")
            .stdout(File::create(fsfs.log_root.join("progressive-daemon.stdout.log")).unwrap())
            .stderr(File::create(&log).unwrap())
            .spawn()
            .unwrap();
        let mut owner = Owner {
            child,
            socket,
            complete: is_complete_store(index),
        };
        while !owner.socket.exists() {
            assert!(
                owner.child.try_wait().unwrap().is_none(),
                "progressive daemon exited"
            );
            assert!(
                start.elapsed() < QUICKSTART_TIMEOUT,
                "progressive daemon startup timed out"
            );
            thread::sleep(Duration::from_millis(10));
        }
        let mut ready = UnixStream::connect(&owner.socket).unwrap();
        ready.set_read_timeout(Some(QUICKSTART_TIMEOUT)).unwrap();
        ready.write_all(b"ready").unwrap();
        ready.shutdown(std::net::Shutdown::Write).unwrap();
        let mut raw = String::new();
        ready.read_to_string(&mut raw).unwrap();
        let ready: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(ready["pid"], owner.child.id());
        // Readiness precedes lazy model initialization; admission is checked
        // again after distinct real queries have loaded and used the model.
        assert_eq!(ready["semantic_admitted"], false);
        eprintln!(
            "[default-build-e2e] stage=progressive-daemon pid={} ready_ms={} fast_id={}",
            owner.child.id(),
            start.elapsed().as_millis(),
            ready["vector_generation_id"]
        );
        for (number, query) in [
            "How can reconnect attempts avoid repeating temporary network failures?",
            "What care helps young tomato plants establish healthy roots?",
        ]
        .into_iter()
        .enumerate()
        {
            let mut socket = UnixStream::connect(&owner.socket).unwrap();
            socket.set_read_timeout(Some(QUICKSTART_TIMEOUT)).unwrap();
            let request = serde_json::json!({"query":query,"limit":3,"quality_timeout_ms":5000,"stream":true});
            let started = Instant::now();
            writeln!(socket, "{request}").unwrap();
            socket.shutdown(std::net::Shutdown::Write).unwrap();
            let mut received =
                File::create(fsfs.log_root.join(format!("progressive-{number}.jsonl"))).unwrap();
            let mut frames = Vec::new();
            for line in std::io::BufReader::new(socket).lines() {
                let line = line.unwrap();
                let frame: Value = serde_json::from_str(&line).unwrap();
                let receipt =
                    serde_json::json!({"received_us":started.elapsed().as_micros(),"frame":frame});
                writeln!(received, "{receipt}").unwrap();
                // --nocapture retains actual arrival times and attestation
                // even after the temporary corpus is released by the test.
                eprintln!("[progressive-frame] query_index={number} {receipt}");
                frames.push(frame);
            }
            assert_eq!(frames.len(), 4, "{frames:?}");
            assert_eq!(frames[0]["event"], "attested");
            assert_eq!(
                frames[0]["policy"]["fast_model"],
                "potion-multilingual-128m"
            );
            assert_eq!(frames[0]["policy"]["quality_model"], "all-minilm-l6-v2");
            assert_eq!(
                frames[0]["cached"], false,
                "distinct queries must execute against the same model owner"
            );
            assert_eq!(
                frames[0]["vector_generation_id"],
                ready["vector_generation_id"]
            );
            assert_eq!(frames[1]["payload"]["phase"], "initial");
            assert_eq!(frames[2]["payload"]["phase"], "refined");
            assert_eq!(frames[3]["event"], "terminal");
            assert_eq!(frames[3]["ok"], true);
            let direct = fsfs.run(
                root,
                &format!("progressive-{number}-direct"),
                [
                    "search",
                    query,
                    "--no-daemon",
                    "--config",
                    config.to_str().unwrap(),
                    "--index-dir",
                    index.to_str().unwrap(),
                    "--limit",
                    "3",
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            let direct = parse_success_envelope("progressive direct reference", &direct);
            assert_eq!(frames[2]["payload"]["hits"], direct["data"]["hits"]);
            for format in ["jsonl", "toon"] {
                let stream = fsfs.run(
                    root,
                    &format!("progressive-{number}-{format}"),
                    [
                        "search",
                        query,
                        "--daemon",
                        "--daemon-socket",
                        owner.socket.to_str().unwrap(),
                        "--config",
                        config.to_str().unwrap(),
                        "--index-dir",
                        index.to_str().unwrap(),
                        "--limit",
                        "3",
                        "--stream",
                        "--format",
                        format,
                    ],
                    QUICKSTART_TIMEOUT,
                );
                assert_finished_successfully("progressive cached CLI replay", &stream);
                assert!(stream.stdout.contains("daemon_cache_hit"), "{stream:?}");
                assert!(
                    stream.stdout.contains("query.stream.initial_ready"),
                    "{stream:?}"
                );
                assert!(
                    stream.stdout.contains("query.stream.refined_ready"),
                    "{stream:?}"
                );
                assert!(!stream.stderr.contains("falling back"));
            }
            if number == 0 {
                for (label, field, model, invalid) in [
                    ("8m", "fast_model", "potion-base-8m", false),
                    ("32m", "fast_model", "potion-base-32m", false),
                    ("malformed", "fast_model", "potion/multilingual/128m", true),
                    (
                        "nomic-quality",
                        "quality_model",
                        "nomic-embed-text-v1.5",
                        false,
                    ),
                    (
                        "malformed-quality",
                        "quality_model",
                        "all/minilm/l6/v2",
                        true,
                    ),
                ] {
                    let changed_config = root.join(format!("daemon-fast-{label}.toml"));
                    fs::write(
                        &changed_config,
                        format!(
                            "[indexing]\n{field} = \"{model}\"\n[search]\nquality_timeout_ms = 5000\n"
                        ),
                    )
                    .unwrap();
                    let refused = fsfs.run(
                        root,
                        &format!("daemon-fast-producer-refuse-{label}"),
                        [
                            "search",
                            query,
                            "--daemon",
                            "--daemon-socket",
                            owner.socket.to_str().unwrap(),
                            "--config",
                            changed_config.to_str().unwrap(),
                            "--index-dir",
                            index.to_str().unwrap(),
                            "--limit",
                            "3",
                            "--stream",
                            "--format",
                            "jsonl",
                        ],
                        QUICKSTART_TIMEOUT,
                    );
                    assert!(!refused.timed_out, "{refused:?}");
                    assert!(
                        !refused.status.success(),
                        "a warmed daemon must not serve another selected producer: {refused:?}"
                    );
                    assert!(
                        !refused
                            .stderr
                            .contains("falling back to in-process retrieval"),
                        "{refused:?}"
                    );
                    let expected = if invalid {
                        format!("indexing.{field}")
                    } else {
                        "requested search policy".to_owned()
                    };
                    assert!(refused.combined_output().contains(&expected), "{refused:?}");
                    for line in refused.stdout.lines() {
                        let frame: Value = serde_json::from_str(line).unwrap();
                        assert_ne!(frame["event"], "result", "{frame}");
                        assert_ne!(
                            frame["payload"]["reason_code"], "query.stream.initial_ready",
                            "policy refusal must precede any answer: {frame}"
                        );
                    }
                }
            }
            assert!(owner.child.try_wait().unwrap().is_none());
        }
        // The same generation/query under two explicit ranking policies must
        // occupy separate cache entries. This prose query also checks the
        // existing neutral natural-language policy with actual producers.
        let mut neutral_hits = None;
        for (number, (priors, cached)) in
            [(false, false), (true, false), (false, true), (true, true)]
                .into_iter()
                .enumerate()
        {
            let mut socket = UnixStream::connect(&owner.socket).unwrap();
            socket.set_read_timeout(Some(QUICKSTART_TIMEOUT)).unwrap();
            writeln!(
                socket,
                "{}",
                serde_json::json!({
                    "query":"How should clients recover after temporary connection failures?",
                    "limit":3, "quality_timeout_ms":5000, "stream":true, "ranking_priors":priors,
                })
            )
            .unwrap();
            socket.shutdown(std::net::Shutdown::Write).unwrap();
            let frames: Vec<Value> = std::io::BufReader::new(socket)
                .lines()
                .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
                .collect();
            assert_eq!(frames.len(), 4, "{frames:?}");
            assert_eq!(frames[0]["policy"]["ranking_priors"], priors);
            assert_eq!(frames[0]["cached"], cached, "policy entry {number}");
            assert_eq!(frames[1]["payload"]["phase"], "initial");
            assert_eq!(frames[2]["payload"]["phase"], "refined");
            assert_eq!(frames[3]["ok"], true);
            let hits = (
                frames[1]["payload"]["hits"].clone(),
                frames[2]["payload"]["hits"].clone(),
            );
            if let Some(expected) = &neutral_hits {
                assert_eq!(&hits, expected);
            } else {
                neutral_hits = Some(hits);
            }
        }
        // Drop a real connection immediately after Initial, then prove the
        // same process remains usable before exercising signal shutdown.
        let mut disconnected = UnixStream::connect(&owner.socket).unwrap();
        disconnected
            .set_read_timeout(Some(QUICKSTART_TIMEOUT))
            .unwrap();
        writeln!(
            disconnected,
            "{}",
            serde_json::json!({
                "query":"How does structured concurrency join outstanding work?",
                "limit":3,"quality_timeout_ms":5000,"stream":true,
            })
        )
        .unwrap();
        disconnected.shutdown(std::net::Shutdown::Write).unwrap();
        let mut reader = std::io::BufReader::new(disconnected);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&line).unwrap()["payload"]["phase"],
            "initial"
        );
        reader.get_ref().shutdown(std::net::Shutdown::Both).unwrap();
        drop(reader);
        let mut ready = UnixStream::connect(&owner.socket).unwrap();
        ready.set_read_timeout(Some(QUICKSTART_TIMEOUT)).unwrap();
        ready.write_all(b"ready").unwrap();
        ready.shutdown(std::net::Shutdown::Write).unwrap();
        let mut raw = String::new();
        ready.read_to_string(&mut raw).unwrap();
        let pid = owner.child.id();
        let ready: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(ready["pid"], pid);
        assert_eq!(ready["semantic_admitted"], true);
        assert!(
            Command::new("kill")
                .args(["-INT", &pid.to_string()])
                .status()
                .unwrap()
                .success()
        );
        let deadline = Instant::now() + FAILURE_TIMEOUT;
        loop {
            if let Some(status) = owner.child.try_wait().unwrap() {
                assert!(status.success(), "signal shutdown: {status}");
                break;
            }
            assert!(Instant::now() < deadline, "signal shutdown timed out");
            thread::sleep(Duration::from_millis(10));
        }
        drop(owner);
        let diagnostics = fs::read_to_string(log).unwrap();
        eprintln!("[progressive-daemon-diagnostics]\n{diagnostics}");
        assert_eq!(
            diagnostics.matches("FastEmbed model loaded").count(),
            1,
            "{diagnostics}"
        );
        let initialization = diagnostics
            .lines()
            .filter(|line| line.contains("fsfs quality model initialization completed"))
            .collect::<Vec<_>>();
        assert_eq!(initialization.len(), 1, "{diagnostics}");
        eprintln!(
            "[default-build-e2e] stage=progressive-daemon cold_initialization={} distinct_completed_query_texts=3 uncached_policy_executions=4 load_amortization_denominator=4",
            initialization[0]
        );
        eprintln!(
            "[default-build-e2e] stage=progressive-daemon event=verified pid={pid} uncached_policy_executions=4 quality_loads=1 ordered_phases=true direct_parity=true ranking_policy_cache_separation=true prose_neutral=true jsonl=true toon=true disconnect=true signal_shutdown=true functional_only=true"
        );
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
        struct DaemonOwner(PathBuf, bool);
        #[cfg(unix)]
        impl Drop for DaemonOwner {
            fn drop(&mut self) {
                request_daemon_stop(&self.0, self.1);
                let deadline = Instant::now() + FAILURE_TIMEOUT;
                while self.0.exists() && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(10));
                }
                let _ = fs::remove_file(&self.0);
                let _ = fs::remove_file(self.0.with_extension("lock"));
            }
        }
        #[cfg(unix)]
        let daemon = DaemonOwner(
            std::env::temp_dir().join(format!("fsfs-blend-{}.sock", std::process::id())),
            is_complete_store(index),
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
        let generation = active_generation_root(index);
        let fast_index_path = generation.join("vector/index.fsvi");
        let quality_index_path = generation.join("vector/quality.fsvi");
        let fast_index =
            VectorIndex::open_read_only(&fast_index_path).expect("open actual fast generation");
        let quality_index = VectorIndex::open_read_only(&quality_index_path)
            .expect("open actual quality generation");
        let lexical_root = generation.join("lexical");
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
                    assert_eq!(response["schema_version"], "fsfs.search.serve.v18");
                    use frankensearch_embed::model_manifest::ModelArtifactManifestV1;
                    let mut contracts = sha2::Sha256::new();
                    for manifest in [
                        ModelArtifactManifestV1::potion_128m_native().unwrap(),
                        ModelArtifactManifestV1::potion_base_8m_native().unwrap(),
                        ModelArtifactManifestV1::potion_base_32m_native().unwrap(),
                        ModelArtifactManifestV1::minilm_fastembed().unwrap(),
                        ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
                        ModelArtifactManifestV1::nomic_fastembed().unwrap(),
                        ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap(),
                        ModelArtifactManifestV1::multilingual_minilm_native_frankentorch().unwrap(),
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
                    assert_eq!(response["policy"]["fast_model"], "potion-multilingual-128m");
                    assert_eq!(response["policy"]["quality_model"], "all-minilm-l6-v2");
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
        let refined = frames
            .iter()
            .enumerate()
            .filter(|(_, frame)| {
                frame
                    .pointer("/payload/reason_code")
                    .and_then(Value::as_str)
                    == Some("query.stream.refined_ready")
            })
            .collect::<Vec<_>>();
        assert_eq!(failures.len() + refined.len(), 1, "one quality outcome");
        if let Some((position, frame)) = refined.first() {
            assert!(*position > initial);
            assert_eq!(frame["payload"]["stage"], "retrieve.quality");
            assert!(
                frames[*position + 1..]
                    .iter()
                    .any(|frame| frame["event"] == "result")
            );
            eprintln!(
                "[default-build-e2e] stage=actual-quality-deadline budget_ms=50 refined=1 failures=0 process_join_ms={}",
                outcome.elapsed.as_millis()
            );
            return;
        }
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

    #[cfg(feature = "semantic-loaders")]
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

    #[cfg(all(feature = "semantic-native", not(feature = "semantic-loaders"), unix))]
    #[test]
    fn native_only_rollback_preserves_the_executable_and_unclassified_backup() {
        use sha2::{Digest as _, Sha256};
        use std::fmt::Write as _;

        let temp = tempfile::tempdir().unwrap();
        let mut fsfs = IsolatedFsfs::new(temp.path(), temp.path().join("models"));
        let executable = temp.path().join("fsfs-private-copy");
        fs::copy(&fsfs.binary, &executable).unwrap();
        fsfs.binary = executable.clone();
        let executable_before = fs::read(&executable).unwrap();
        let backup_dir = fsfs.xdg_data.join("frankensearch/backups");
        fs::create_dir_all(&backup_dir).unwrap();
        let backup = b"unclassified historical executable bytes";
        let backup_path = backup_dir.join("fsfs-1.0.0");
        fs::write(&backup_path, backup).unwrap();
        let mut backup_sha256 = String::new();
        for byte in Sha256::digest(backup) {
            write!(backup_sha256, "{byte:02x}").unwrap();
        }
        let manifest_path = backup_dir.join("rollback-manifest.json");
        let manifest = serde_json::to_vec(&serde_json::json!({"entries": [{
            "version": "1.0.0", "backed_up_at_epoch": 1,
            "original_path": executable, "binary_filename": "fsfs-1.0.0",
            "sha256": backup_sha256,
        }]}))
        .unwrap();
        fs::write(&manifest_path, &manifest).unwrap();
        for (label, args) in [
            (
                "rollback-latest",
                vec!["update", "--rollback", "--format", "json"],
            ),
            (
                "rollback-selected",
                vec!["update", "--rollback", "1.0.0", "--format", "json"],
            ),
            (
                "rollback-selected-check",
                vec![
                    "update",
                    "--rollback",
                    "1.0.0",
                    "--check",
                    "--format",
                    "json",
                ],
            ),
        ] {
            let outcome = fsfs.run(temp.path(), label, args, FAILURE_TIMEOUT);
            assert!(
                !outcome.timed_out && !outcome.status.success(),
                "{outcome:?}"
            );
            let error: Value = serde_json::from_str(&outcome.stdout).unwrap();
            assert_eq!(error["error"]["code"], "invalid_config", "{error}");
            assert!(
                outcome.combined_output().contains("update.profile"),
                "{outcome:?}"
            );
            assert_eq!(fs::read(&executable).unwrap(), executable_before);
            assert_eq!(fs::read(&backup_path).unwrap(), backup);
            assert_eq!(fs::read(&manifest_path).unwrap(), manifest);
        }
        let listing = fsfs.run(
            temp.path(),
            "rollback-list",
            ["update", "--rollback", "--check"],
            FAILURE_TIMEOUT,
        );
        assert_finished_successfully("rollback list", &listing);
        assert!(listing.stdout.contains("v1.0.0"), "{listing:?}");
        assert_eq!(fs::read(&executable).unwrap(), executable_before);
    }

    // Cold initialization is inside the public quality deadline. README's
    // native contract permits a typed timeout with retained Initial hits;
    // the real daemon test below separately requires uncached warm refinement.
    #[cfg(feature = "rerank")]
    fn validate_native_cold_result(payload: &Value) -> Result<(), &'static str> {
        let hits = payload["hits"].as_array().ok_or("missing hits")?;
        if hits.is_empty() || payload["returned_hits"].as_u64() != u64::try_from(hits.len()).ok() {
            return Err("cold search must retain populated hits");
        }
        match payload["phase"].as_str() {
            Some("refined")
                if payload["quality_timeout"].is_null() && payload["skip_reason"].is_null() =>
            {
                Ok(())
            }
            Some("refinement_failed")
                if payload["skip_reason"] == "quality_timeout"
                    && payload["quality_timeout"]["reason_code"] == "query.quality.timeout"
                    && payload["quality_timeout"]["budget_ms"] == 500
                    && payload["quality_timeout"]["elapsed_ms"]
                        .as_u64()
                        .is_some_and(|elapsed| elapsed >= 500) =>
            {
                Ok(())
            }
            _ => Err("cold search must refine or report its actual 500ms timeout"),
        }
    }

    #[cfg(feature = "rerank")]
    #[test]
    fn native_cold_contract_rejects_other_failures_and_empty_results() {
        let timeout = serde_json::json!({
            "phase": "refinement_failed", "hits": [{"path": "castaway.md"}],
            "returned_hits": 1, "skip_reason": "quality_timeout",
            "quality_timeout": {
                "reason_code": "query.quality.timeout", "budget_ms": 500, "elapsed_ms": 501
            }
        });
        assert_eq!(validate_native_cold_result(&timeout), Ok(()));
        for (pointer, replacement) in [
            ("/hits", serde_json::json!([])),
            ("/returned_hits", serde_json::json!(0)),
            ("/phase", serde_json::json!("initial")),
            (
                "/skip_reason",
                serde_json::json!("embedding_space_unverifiable"),
            ),
            (
                "/quality_timeout/reason_code",
                serde_json::json!("model_load_failed"),
            ),
            ("/quality_timeout/budget_ms", serde_json::json!(1000)),
            ("/quality_timeout/elapsed_ms", serde_json::json!(499)),
        ] {
            let mut invalid = timeout.clone();
            *invalid.pointer_mut(pointer).unwrap() = replacement;
            assert!(validate_native_cold_result(&invalid).is_err(), "{invalid}");
        }
        let mut refined = timeout;
        refined["phase"] = serde_json::json!("refined");
        assert!(validate_native_cold_result(&refined).is_err());
        refined["quality_timeout"] = Value::Null;
        assert!(validate_native_cold_result(&refined).is_err());
        refined["skip_reason"] = Value::Null;
        assert_eq!(validate_native_cold_result(&refined), Ok(()));
    }

    #[cfg(feature = "rerank")]
    #[test]
    #[ignore = "requires real native MiniLM, Potion and ONNX fixtures; explicit native CLI integration"]
    fn native_quality_model_runs_the_production_cli() {
        use frankensearch_core::Embedder;
        use frankensearch_rerank::{NativeEmbedder, NativeEmbeddingModel};

        log_binary_profile("native-quality");
        let temp = tempfile::tempdir().unwrap();
        let models = temp.path().join("models");
        let fsfs = IsolatedFsfs::new(temp.path(), models.clone());
        for (variable, directory, manifest) in [
            (
                "POTION_FIXTURE_DIR",
                "potion-multilingual-128M",
                ModelManifest::potion_128m(),
            ),
            (
                "MINILM_FIXTURE_DIR",
                "all-MiniLM-L6-v2-native",
                ModelManifest::minilm_v2_native(),
            ),
            (
                "FASTEMBED_MINILM_FIXTURE_DIR",
                "all-MiniLM-L6-v2",
                ModelManifest::minilm_v2(),
            ),
        ] {
            let source = std::env::var_os(variable)
                .map_or_else(|| configured_model_root().join(directory), PathBuf::from);
            manifest.verify_dir(&source).unwrap_or_else(|error| {
                // ubs:ignore -- Missing real model fixtures must fail this integration test.
                panic!("native CLI fixture {directory} at {} is unavailable: {error}; install it with fsfs download-models {} or set {variable}", source.display(), manifest.id)
            });
            let destination = models.join(directory);
            for file in &manifest.files {
                let target = destination.join(&file.name);
                fs::create_dir_all(target.parent().unwrap()).unwrap();
                // Linking/unlinking changes the source inode's ctime and races
                // other real-model verification tests. Use independent files.
                fs::copy(source.join(&file.name), &target).unwrap();
            }
            manifest.verify_dir(&destination).unwrap();
        }
        let native_directory = models.join("all-MiniLM-L6-v2-native");
        let native =
            NativeEmbedder::load_model(&native_directory, NativeEmbeddingModel::AllMiniLmL6V2F32)
                .unwrap();
        let fingerprint = native.identity().unwrap().fingerprint();
        // GOLDEN-CHANGE bd-7651e: the current 256-token F32 identity.
        assert_eq!(
            fingerprint, "8c37f4cf6f63cdbc937ff7b03369606555e0eb165779c0019b1591fcadaf0860",
            "native producer must name the current tokenizer protocol and sequence policy; historical identities are pinned separately"
        );
        let config = temp.path().join("native.toml");
        fs::write(
            &config,
            if cfg!(all(
                feature = "semantic-native",
                not(feature = "semantic-loaders")
            )) {
                // Exercise the profile's actual default without a quality override.
                ""
            } else {
                "[indexing]\nquality_model = \"all-MiniLM-L6-v2-native\"\n"
            },
        )
        .unwrap();
        let onnx_config = temp.path().join("onnx.toml");
        fs::write(
            &onnx_config,
            "[indexing]\nquality_model = \"all-MiniLM-L6-v2\"\n",
        )
        .unwrap();
        let corpus = temp.path().join("corpus");
        fs::create_dir(&corpus).unwrap();
        fs::write(
            corpus.join("castaway.md"),
            "Ben Gunn is a marooned sailor, a lone castaway stranded on Treasure Island.",
        )
        .unwrap();
        fs::write(
            corpus.join("cook.md"),
            "Long John Silver is the ship cook with a wooden leg and a talking parrot.",
        )
        .unwrap();
        let index = temp.path().join("index");
        let config_arg = config.to_str().unwrap();
        let index_arg = index.to_str().unwrap();
        #[cfg(all(feature = "semantic-native", not(feature = "semantic-loaders")))]
        {
            let verify = fsfs.run(
                temp.path(),
                "native-profile-default-verify",
                ["download-models", "--verify", "--format", "json"],
                QUICKSTART_TIMEOUT,
            );
            let verified = parse_success_envelope("native profile default verify", &verify);
            let entries = verified["data"]["models"].as_array().unwrap();
            assert_eq!(entries.len(), 2, "{verified}");
            assert!(
                entries
                    .iter()
                    .any(|entry| entry["id"] == "all-minilm-l6-v2-native"
                        && entry["verified"] == true),
                "{verified}"
            );
            assert!(
                entries
                    .iter()
                    .all(|entry| entry["id"] != "all-minilm-l6-v2"),
                "{verified}"
            );
            let unavailable = fsfs.run(
                temp.path(),
                "native-profile-onnx-doctor",
                [
                    "doctor",
                    "--config",
                    onnx_config.to_str().unwrap(),
                    "--index-dir",
                    index_arg,
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            assert!(!unavailable.timed_out, "{unavailable:?}");
            let doctor: Value = serde_json::from_str(&unavailable.stdout).unwrap();
            assert_eq!(unavailable.status.code(), Some(1), "{unavailable:?}");
            assert_eq!(doctor["ok"], false, "{doctor}");
            assert_eq!(doctor["error"]["code"], "subsystem_error", "{doctor}");
            let context = doctor["error"]["context"].as_str().unwrap();
            assert!(
                context.contains("model.quality") && context.contains("no ONNX quality loader"),
                "{doctor}"
            );
            let update = fsfs.run(
                temp.path(),
                "native-profile-update-refusal",
                ["update", "--format", "json"],
                FAILURE_TIMEOUT,
            );
            assert!(!update.timed_out && !update.status.success(), "{update:?}");
            assert!(
                update.combined_output().contains("update.profile"),
                "{update:?}"
            );
            assert!(
                update
                    .combined_output()
                    .contains("--features semantic-native"),
                "{update:?}"
            );
        }
        let verify = fsfs.run(
            temp.path(),
            "native-explicit-verify",
            [
                "download-models",
                "all-MiniLM-L6-v2-native",
                "--verify",
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );
        let verified = parse_success_envelope("native explicit verify", &verify);
        assert!(verified.to_string().contains("verified"));
        let build = fsfs.run(
            temp.path(),
            "native-index",
            [
                "index",
                corpus.to_str().unwrap(),
                "--config",
                config_arg,
                "--index-dir",
                index_arg,
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        parse_success_envelope("native index", &build);
        let quality_path = active_generation_root(&index).join("vector/quality.fsvi");
        {
            let quality = VectorIndex::open_read_only(&quality_path).unwrap();
            assert_eq!(quality.embedder_id(), "minilm-384-native-f32");
            assert_eq!(quality.embedder_revision(), fingerprint);
            assert_eq!(quality.record_count(), 2);
        }
        let status = fsfs.run(
            temp.path(),
            "native-status",
            [
                "status",
                "--config",
                config_arg,
                "--index-dir",
                index_arg,
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );
        let status = parse_success_envelope("native status", &status);
        assert_eq!(
            model_status(&status, "quality")["verification_state"],
            "verified"
        );
        let doctor = fsfs.run(
            temp.path(),
            "native-doctor",
            [
                "doctor",
                "--config",
                config_arg,
                "--index-dir",
                index_arg,
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        parse_success_envelope("native doctor", &doctor);

        let query = "a lone sailor stranded on an island";
        let search = |label: &str, native_config: bool, stream: bool, bypass_cache: bool| {
            let mut args = vec![
                "search",
                query,
                "--index-dir",
                index_arg,
                "--no-daemon",
                "--format",
                if stream { "jsonl" } else { "json" },
            ];
            if native_config {
                args.extend(["--config", config_arg]);
            } else {
                args.extend(["--config", onnx_config.to_str().unwrap()]);
            }
            if stream {
                args.push("--stream");
            }
            let cache_env = if bypass_cache {
                &[("FSFS_DISABLE_QUERY_CACHE", "1")][..]
            } else {
                &[]
            };
            fsfs.run_with_env(temp.path(), label, args, QUICKSTART_TIMEOUT, cache_env)
        };
        let initial_search = search("native-search", true, false, false);
        let result = parse_success_envelope("native search", &initial_search);
        assert_eq!(
            validate_native_cold_result(&result["data"]),
            Ok(()),
            "{result}"
        );
        let phases = |outcome: &CommandOutcome| {
            assert_finished_successfully("stream search", outcome);
            outcome
                .stdout
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .filter_map(|frame| {
                    frame
                        .pointer("/payload/reason_code")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>()
        };
        let native_stream = search("native-stream", true, true, false);
        let native_phases = phases(&native_stream);
        assert!(
            native_phases
                .iter()
                .any(|phase| phase == "query.stream.initial_ready")
        );
        if !native_phases
            .iter()
            .any(|phase| phase == "query.stream.refined_ready")
        {
            let frames: Vec<Value> = native_stream
                .stdout
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let failure = frames
                .iter()
                .find(|frame| frame["payload"]["reason_code"] == "query.stream.refinement_failed")
                .expect("cold stream must refine or report a quality timeout");
            assert!(
                failure["payload"]["message"]
                    .as_str()
                    .unwrap()
                    .starts_with("quality refinement exceeded 500ms (observed "),
                "{failure}"
            );
            assert!(
                frames.iter().any(|frame| frame["payload"]["item"]["path"]
                    .as_str()
                    .is_some_and(|path| path.ends_with("castaway.md"))),
                "{frames:?}"
            );
        }
        let wrong_backend = search("onnx-cannot-reuse-native-cache", false, true, false);
        let wrong_phases = phases(&wrong_backend);
        assert!(
            wrong_phases
                .iter()
                .any(|phase| phase == "query.stream.refinement_failed"),
            "{wrong_backend:?}"
        );
        assert!(
            !wrong_phases
                .iter()
                .any(|phase| phase == "query.stream.refined_ready")
        );

        #[cfg(unix)]
        {
            use std::io::{Read as _, Write as _};
            use std::os::unix::net::UnixStream;

            struct NativeDaemon(PathBuf, bool);
            impl Drop for NativeDaemon {
                fn drop(&mut self) {
                    request_daemon_stop(&self.0, self.1);
                }
            }
            let daemon_path = temp.path().join("native.sock");
            let complete = is_complete_store(&index);
            let daemon = NativeDaemon(daemon_path.clone(), complete);
            let args = [
                "search",
                query,
                "--config",
                config_arg,
                "--index-dir",
                index_arg,
                "--daemon",
                "--daemon-socket",
                daemon_path.to_str().unwrap(),
                "--format",
                "json",
            ];
            let served = fsfs.run(temp.path(), "native-daemon", args, QUICKSTART_TIMEOUT);
            let result = parse_success_envelope("native daemon", &served);
            assert_eq!(
                validate_native_cold_result(&result["data"]),
                Ok(()),
                "{result}"
            );
            assert!(
                !served
                    .stderr
                    .contains("falling back to in-process retrieval"),
                "{served:?}"
            );
            let mut socket = UnixStream::connect(&daemon_path).unwrap();
            socket.set_read_timeout(Some(QUICKSTART_TIMEOUT)).unwrap();
            writeln!(
                socket,
                "{}",
                serde_json::json!({"query": query, "limit": 10})
            )
            .unwrap();
            socket.shutdown(std::net::Shutdown::Write).unwrap();
            let mut raw = String::new();
            socket.read_to_string(&mut raw).unwrap();
            let response: Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(
                response["policy"]["quality_model"],
                "all-minilm-l6-v2-native"
            );
            assert_eq!(
                validate_native_cold_result(
                    response["payloads"].as_array().unwrap().last().unwrap()
                ),
                Ok(()),
                "{response}"
            );
            let wrong = fsfs.run(
                temp.path(),
                "onnx-cannot-reuse-native-daemon",
                [
                    "search",
                    query,
                    "--config",
                    onnx_config.to_str().unwrap(),
                    "--index-dir",
                    index_arg,
                    "--daemon",
                    "--daemon-socket",
                    daemon_path.to_str().unwrap(),
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            assert!(
                !wrong.status.success(),
                "policy refusal must reach the caller: {wrong:?}"
            );
            // A legacy daemon answers and the client refuses the unacknowledged
            // policy; a complete store's daemon refuses a client whose model
            // configuration differs before it searches.
            let refusal = if complete {
                "client and daemon configurations differ"
            } else {
                "did not acknowledge requested search policy"
            };
            assert!(wrong.combined_output().contains(refusal), "{wrong:?}");
            assert!(
                !wrong
                    .stderr
                    .contains("falling back to in-process retrieval")
            );
            let direct = fsfs.run(
                temp.path(),
                "onnx-cannot-read-native-generation-direct",
                [
                    "search",
                    query,
                    "--config",
                    onnx_config.to_str().unwrap(),
                    "--index-dir",
                    index_arg,
                    "--no-daemon",
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            let refused = parse_success_envelope("wrong native producer direct", &direct);
            assert_eq!(refused["data"]["phase"], "refinement_failed", "{refused}");
            assert_eq!(
                refused["data"]["degradation_advice"]["degrade.advice.embedding_space_unverifiable"]
                    ["failure"],
                "unverifiable_embedding_space",
                "{refused}"
            );
            drop(daemon);
            let deadline = Instant::now() + FAILURE_TIMEOUT;
            while daemon_path.exists() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(10));
            }
            assert!(
                !daemon_path.exists(),
                "owned daemon must release its index before append"
            );

            // A cold query may time out while its admitted constructor still
            // runs. Its successful model must remain usable by the next query.
            let cold_path = temp.path().join("native-cold.sock");
            let cold_log_path = fsfs.log_root.join("native-cold-daemon.stderr.log");
            let mut child = fsfs
                .command(temp.path())
                .args([
                    "serve",
                    "--daemon",
                    "--daemon-socket",
                    cold_path.to_str().unwrap(),
                    "--config",
                    config_arg,
                    "--index-dir",
                    index_arg,
                ])
                .env("RUST_LOG", "info")
                .stdout(File::create(fsfs.log_root.join("native-cold-daemon.stdout.log")).unwrap())
                .stderr(File::create(&cold_log_path).unwrap())
                .spawn()
                .unwrap();
            let cold_daemon = NativeDaemon(cold_path.clone(), complete);
            let deadline = Instant::now() + FAILURE_TIMEOUT;
            while !cold_path.exists() && Instant::now() < deadline {
                assert!(child.try_wait().unwrap().is_none(), "cold daemon exited");
                thread::sleep(Duration::from_millis(10));
            }
            let request = |budget_ms: u64| {
                let mut socket = UnixStream::connect(&cold_path).unwrap();
                socket.set_read_timeout(Some(QUICKSTART_TIMEOUT)).unwrap();
                writeln!(
                    socket,
                    "{}",
                    serde_json::json!({
                        "query": query, "limit": 10, "quality_timeout_ms": budget_ms
                    })
                )
                .unwrap();
                socket.shutdown(std::net::Shutdown::Write).unwrap();
                let mut raw = String::new();
                socket.read_to_string(&mut raw).unwrap();
                serde_json::from_str::<Value>(&raw).unwrap()
            };
            let cold = request(50);
            assert_eq!(cold["cached"], false, "{cold}");
            assert_eq!(cold["payloads"][0]["phase"], "initial", "{cold}");
            match cold["payloads"][1]["phase"].as_str() {
                Some("refinement_failed") => {
                    assert_eq!(cold["payloads"][1]["quality_timeout"]["budget_ms"], 50);
                    assert_eq!(cold["payloads"][1]["hits"], cold["payloads"][0]["hits"]);
                }
                Some("refined") => {
                    assert!(cold["payloads"][1]["quality_timeout"].is_null());
                    assert_eq!(
                        cold["payloads"][1]["semantic_blend"]["quality_embedder"],
                        "minilm-384-native-f32"
                    );
                    assert!(!cold["payloads"][1]["hits"].as_array().unwrap().is_empty());
                }
                _ => panic!("unexpected cold quality outcome: {cold}"),
            }

            // Observe the real loader finishing, not a guessed sleep interval.
            let loaded_message = "fsfs selected verified native F32 quality model";
            let deadline = Instant::now() + FAILURE_TIMEOUT;
            loop {
                let log = fs::read_to_string(&cold_log_path).unwrap();
                if log.contains(loaded_message) {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "native load did not finish: {log}"
                );
                assert!(child.try_wait().unwrap().is_none(), "cold daemon exited");
                thread::sleep(Duration::from_millis(10));
            }
            let warm = request(500);
            assert_eq!(
                warm["cached"], false,
                "the warm query must execute under its 500ms policy: {warm}"
            );
            assert_eq!(
                warm["payloads"].as_array().unwrap().last().unwrap()["phase"],
                "refined",
                "completed initialization must serve the next query: {warm}"
            );
            let log = fs::read_to_string(&cold_log_path).unwrap();
            assert_eq!(
                log.matches(loaded_message).count(),
                1,
                "a timeout must not force another model initialization: {log}"
            );
            drop(cold_daemon);
            assert!(child.wait().unwrap().success());
        }

        let append_path = temp.path().join("append.jsonl");
        let appended_id = corpus.join("second-castaway.md").display().to_string();
        fs::write(&append_path, format!("{}\n", serde_json::json!({"id": appended_id, "text": "A second marooned sailor survives alone on a remote island."}))).unwrap();
        let append_args = [
            "append-batch",
            "--file",
            append_path.to_str().unwrap(),
            "--config",
            config_arg,
            "--index-dir",
            index_arg,
            "--format",
            "json",
        ];
        let append = fsfs.run(
            temp.path(),
            "native-append",
            append_args,
            QUICKSTART_TIMEOUT,
        );
        assert_finished_successfully("native append", &append);
        let after_append = parse_success_envelope(
            "native after append",
            &search("native-search-appended", true, false, false),
        );
        assert_eq!(
            validate_native_cold_result(&after_append["data"]),
            Ok(()),
            "{after_append}"
        );
        assert!(after_append.to_string().contains("second-castaway.md"));

        let before = [
            fs::read(active_generation_root(&index).join("vector/index.fsvi")).unwrap(),
            fs::read(&quality_path).unwrap(),
        ];
        fs::rename(&native_directory, models.join("native-fixture-preserved")).unwrap();
        for corrupt in [false, true] {
            if corrupt {
                fs::create_dir(&native_directory).unwrap();
                for name in [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ] {
                    fs::copy(
                        models.join("native-fixture-preserved").join(name),
                        native_directory.join(name),
                    )
                    .unwrap();
                }
                fs::write(native_directory.join("model.safetensors"), b"corrupt").unwrap();
            }
            let label = if corrupt {
                "native-corrupt"
            } else {
                "native-missing"
            };
            // Force actual backend loading: a previously computed result from
            // this same producer can remain valid when model files disappear.
            let unavailable = search(label, true, true, true);
            let unavailable_phases = phases(&unavailable);
            assert!(
                unavailable_phases
                    .iter()
                    .any(|phase| phase == "query.stream.refinement_failed"),
                "{unavailable:?}"
            );
            assert!(
                !unavailable_phases
                    .iter()
                    .any(|phase| phase == "query.stream.refined_ready")
            );
            let refused = fsfs.run(
                temp.path(),
                &format!("{label}-append"),
                append_args,
                QUICKSTART_TIMEOUT,
            );
            assert!(
                !refused.timed_out && !refused.status.success(),
                "{refused:?}"
            );
            let error: Value = serde_json::from_str(&refused.stdout).unwrap();
            assert_eq!(
                error["error"]["code"],
                if corrupt {
                    "model_load_failed"
                } else {
                    "model_not_found"
                }
            );
            assert_eq!(
                [
                    fs::read(active_generation_root(&index).join("vector/index.fsvi")).unwrap(),
                    fs::read(&quality_path).unwrap()
                ],
                before
            );
        }
        eprintln!(
            "[native-cli] actual native index/search/append, producer refusal, missing/corrupt model and vector preservation verified"
        );
    }

    #[test]
    #[cfg(all(feature = "rerank", unix))]
    #[ignore = "requires real multilingual MiniLM, Potion and ONNX fixtures"]
    fn multilingual_quality_model_runs_cross_language_cli_search() {
        use std::io::{Read as _, Write as _};
        use std::os::unix::net::UnixStream;

        use frankensearch_core::Embedder;
        use frankensearch_rerank::NativeEmbedder;

        log_binary_profile("multilingual-quality");
        let temp = tempfile::tempdir().unwrap();
        let models = temp.path().join("models");
        let fsfs = IsolatedFsfs::new(temp.path(), models.clone());
        let model_id = "paraphrase-multilingual-minilm-l12-v2";
        let model_directory = "paraphrase-multilingual-MiniLM-L12-v2";
        for (variable, directory, manifest) in [
            (
                "POTION_FIXTURE_DIR",
                "potion-multilingual-128M",
                ModelManifest::potion_128m(),
            ),
            (
                "MULTILINGUAL_MINILM_FIXTURE_DIR",
                model_directory,
                ModelManifest::multilingual_minilm_l12_v2(),
            ),
            (
                "FASTEMBED_MINILM_FIXTURE_DIR",
                "all-MiniLM-L6-v2",
                ModelManifest::minilm_v2(),
            ),
        ] {
            let source = std::env::var_os(variable)
                .map_or_else(|| configured_model_root().join(directory), PathBuf::from);
            manifest.verify_dir(&source).unwrap_or_else(|error| {
                // ubs:ignore -- Missing real multilingual fixtures must fail this integration test.
                panic!(
                    "multilingual CLI fixture {} at {} is unavailable: {error}; set {variable}",
                    manifest.id,
                    source.display()
                )
            });
            let destination = models.join(directory);
            for file in &manifest.files {
                let target = destination.join(&file.name);
                fs::create_dir_all(target.parent().unwrap()).unwrap();
                fs::copy(source.join(&file.name), target).unwrap();
            }
            manifest.verify_dir(&destination).unwrap();
        }
        let native = NativeEmbedder::load_multilingual(models.join(model_directory)).unwrap();
        let fingerprint = native.identity().unwrap().fingerprint();
        let config = temp.path().join("multilingual.toml");
        fs::write(
            &config,
            format!("[indexing]\nquality_model = \"{model_id}\"\n"),
        )
        .unwrap();
        let corpus = temp.path().join("corpus");
        fs::create_dir(&corpus).unwrap();
        for (name, content) in [
            (
                "concurrency.md",
                "In Rust, structured concurrency keeps child tasks scoped and propagates cancellation safely.",
            ),
            (
                "bread.md",
                "A sourdough starter needs flour, water, and a warm kitchen.",
            ),
            (
                "deadlock.md",
                "数据库事务发生死锁时，应回滚其中一个事务，并按固定顺序重试锁操作。",
            ),
            ("pie.md", "这份食谱介绍如何烤制苹果派和准备奶油馅料。"),
            (
                "worker.md",
                "worker_queue.rs 必须在 async 任务取消时归还 reservation，避免消息丢失。",
            ),
            (
                "painting.md",
                "The watercolor landscape uses blue pigment and cold-press paper.",
            ),
        ] {
            fs::write(corpus.join(name), content).unwrap();
        }
        let index = temp.path().join("index");
        let config_arg = config.to_str().unwrap();
        let index_arg = index.to_str().unwrap();
        let build = fsfs.run(
            temp.path(),
            "multilingual-index",
            [
                "index",
                corpus.to_str().unwrap(),
                "--config",
                config_arg,
                "--index-dir",
                index_arg,
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        parse_success_envelope("multilingual index", &build);
        let quality_path = active_generation_root(&index).join("vector/quality.fsvi");
        {
            let quality = VectorIndex::open_read_only(&quality_path).unwrap();
            assert_eq!(
                quality.embedder_id(),
                native.id(),
                "explicit multilingual selection must not fall back to the installed ONNX model"
            );
            assert_eq!(quality.embedder_revision(), fingerprint);
            assert_eq!(quality.record_count(), 6);
        }
        for command in ["status", "doctor"] {
            let outcome = fsfs.run(
                temp.path(),
                &format!("multilingual-{command}"),
                [
                    command,
                    "--config",
                    config_arg,
                    "--index-dir",
                    index_arg,
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            let envelope = parse_success_envelope(command, &outcome);
            if command == "status" {
                assert_eq!(
                    model_status(&envelope, "quality")["verification_state"],
                    "verified"
                );
            }
        }

        struct Daemon {
            socket: PathBuf,
            complete: bool,
            child: std::process::Child,
        }
        impl Drop for Daemon {
            fn drop(&mut self) {
                request_daemon_stop(&self.socket, self.complete);
                let deadline = Instant::now() + FAILURE_TIMEOUT;
                while Instant::now() < deadline {
                    if let Ok(Some(status)) = self.child.try_wait() {
                        if !thread::panicking() {
                            assert!(status.success(), "multilingual daemon exited {status}");
                        }
                        return;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                eprintln!(
                    "multilingual test daemon failed to stop; terminating owned child {}",
                    self.child.id()
                );
                let _ = self.child.kill();
                let _ = self.child.wait();
                assert!(
                    thread::panicking(),
                    "multilingual daemon failed graceful shutdown"
                );
            }
        }
        let socket_path = temp.path().join("multilingual.sock");
        let log_path = fsfs.log_root.join("multilingual-daemon.stderr.log");
        let child = fsfs
            .command(temp.path())
            .args([
                "serve",
                "--daemon",
                "--daemon-socket",
                socket_path.to_str().unwrap(),
                "--config",
                config_arg,
                "--index-dir",
                index_arg,
            ])
            .env("RUST_LOG", "info")
            .stdout(File::create(fsfs.log_root.join("multilingual-daemon.stdout.log")).unwrap())
            .stderr(File::create(&log_path).unwrap())
            .spawn()
            .unwrap();
        let mut daemon = Daemon {
            socket: socket_path.clone(),
            complete: is_complete_store(&index),
            child,
        };
        let deadline = Instant::now() + FAILURE_TIMEOUT;
        while !socket_path.exists() && Instant::now() < deadline {
            assert!(daemon.child.try_wait().unwrap().is_none());
            thread::sleep(Duration::from_millis(10));
        }
        let request = |query: &str| {
            let mut socket = UnixStream::connect(&socket_path).unwrap();
            socket.set_read_timeout(Some(QUICKSTART_TIMEOUT)).unwrap();
            writeln!(socket, "{}", serde_json::json!({"query": query, "limit": 10, "mode": "full", "quality_timeout_ms": 500})).unwrap();
            socket.shutdown(std::net::Shutdown::Write).unwrap();
            let mut raw = String::new();
            socket.read_to_string(&mut raw).unwrap();
            serde_json::from_str::<Value>(&raw).unwrap()
        };
        let cold = request("How can structured concurrency handle cancellation safely?");
        assert_eq!(cold["ok"], true, "{cold}");
        assert_eq!(cold["payloads"][0]["phase"], "initial", "{cold}");
        let cold_last = cold["payloads"].as_array().unwrap().last().unwrap();
        if cold_last["phase"] != "refined" {
            assert_eq!(cold_last["phase"], "refinement_failed", "{cold}");
            assert_eq!(cold_last["skip_reason"], "quality_timeout", "{cold}");
            assert_eq!(cold_last["quality_timeout"]["budget_ms"], 500);
            assert_eq!(cold_last["hits"], cold["payloads"][0]["hits"]);
        }
        eprintln!(
            "[multilingual-cli] cold_default_budget_phase={}",
            cold_last["phase"]
        );
        let loaded = "fsfs selected verified multilingual native quality model";
        let deadline = Instant::now() + FAILURE_TIMEOUT;
        while !fs::read_to_string(&log_path).unwrap().contains(loaded) {
            assert!(
                Instant::now() < deadline,
                "multilingual model did not finish loading"
            );
            assert!(daemon.child.try_wait().unwrap().is_none());
            thread::sleep(Duration::from_millis(10));
        }
        let queries = [
            ("如何在 Rust 中处理任务取消和结构化并发？", "concurrency.md"),
            (
                "How should a database transaction deadlock be resolved?",
                "deadlock.md",
            ),
            (
                "修复 Rust async cancellation bug in worker_queue.rs",
                "worker.md",
            ),
        ];
        for (query, expected) in queries {
            let response = request(query);
            assert_eq!(response["ok"], true, "{response}");
            assert_eq!(
                response["cached"], false,
                "new multilingual query must execute"
            );
            assert_eq!(
                response["policy"]["quality_model"],
                "paraphrase-multilingual-minilm-l12-v2"
            );
            let refined = response["payloads"].as_array().unwrap().last().unwrap();
            assert_eq!(refined["phase"], "refined", "{response}");
            assert_eq!(refined["semantic_blend"]["quality_embedder"], native.id());
            assert!(
                refined["hits"][0]["path"]
                    .as_str()
                    .unwrap()
                    .ends_with(expected),
                "{response}"
            );
            let repeat = request(query);
            assert_eq!(repeat["cached"], true, "{repeat}");
            assert_eq!(repeat["payloads"], response["payloads"]);
        }
        assert_eq!(
            fs::read_to_string(&log_path)
                .unwrap()
                .matches(loaded)
                .count(),
            1
        );
        let onnx_config = temp.path().join("onnx.toml");
        fs::write(
            &onnx_config,
            "[indexing]\nquality_model = \"all-MiniLM-L6-v2\"\n",
        )
        .unwrap();
        let wrong = fsfs.run(
            temp.path(),
            "onnx-cannot-reuse-multilingual-daemon",
            [
                "search",
                queries[0].0,
                "--config",
                onnx_config.to_str().unwrap(),
                "--index-dir",
                index_arg,
                "--daemon",
                "--daemon-socket",
                socket_path.to_str().unwrap(),
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        assert!(
            !wrong.timed_out && !wrong.status.success(),
            "policy refusal must reach the caller: {wrong:?}"
        );
        // Complete-store clients bind the whole daemon configuration before
        // searching; legacy clients check the returned search policy instead.
        let refusal = if daemon.complete {
            let envelope: Value = serde_json::from_str(&wrong.stdout).unwrap();
            assert_eq!(envelope["ok"], false, "{envelope}");
            assert_eq!(envelope["error"]["code"], "invalid_config", "{envelope}");
            assert_eq!(
                envelope["error"]["field"], "complete_generation.daemon_config",
                "{envelope}"
            );
            "client and daemon configurations differ"
        } else {
            "did not acknowledge requested search policy"
        };
        assert!(wrong.combined_output().contains(refusal), "{wrong:?}");
        assert!(
            !wrong
                .stderr
                .contains("falling back to in-process retrieval")
        );
        let direct = fsfs.run(
            temp.path(),
            "onnx-cannot-read-multilingual-generation-direct",
            [
                "search",
                queries[0].0,
                "--config",
                onnx_config.to_str().unwrap(),
                "--index-dir",
                index_arg,
                "--no-daemon",
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        let refused = parse_success_envelope("wrong multilingual producer direct", &direct);
        assert_eq!(refused["data"]["phase"], "refinement_failed", "{refused}");
        assert_eq!(
            refused["data"]["degradation_advice"]["degrade.advice.embedding_space_unverifiable"]["failure"],
            "unverifiable_embedding_space",
            "{refused}"
        );
        drop(daemon);

        let appended_name = "retry-locks.md";
        let appended_id = corpus.join(appended_name).display().to_string();
        let append_path = temp.path().join("append.jsonl");
        let content = "数据库事务发生死锁后，回滚事务并按固定顺序重试锁操作。";
        fs::write(
            &append_path,
            format!(
                "{}\n",
                serde_json::json!({"id": appended_id, "text": content})
            ),
        )
        .unwrap();
        let predecessor_bytes = fs::read(&quality_path).unwrap();
        let append = fsfs.run(
            temp.path(),
            "multilingual-append",
            [
                "append-batch",
                "--file",
                append_path.to_str().unwrap(),
                "--config",
                config_arg,
                "--index-dir",
                index_arg,
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        let receipt = parse_success_envelope("multilingual append", &append);
        let successor_quality_path = active_generation_root(&index).join("vector/quality.fsvi");
        let complete = is_complete_store(&index);
        if complete {
            assert_eq!(receipt["data"]["appended"], 1);
            assert_eq!(receipt["data"]["publication"], "durable");
            assert_eq!(receipt["data"]["generation_changed"], true);
            assert_ne!(successor_quality_path, quality_path);
            assert_eq!(
                successor_quality_path,
                Path::new(receipt["data"]["generation_path"].as_str().unwrap())
                    .join("vector/quality.fsvi")
            );
            assert_eq!(fs::read(&quality_path).unwrap(), predecessor_bytes);
            let predecessor = VectorIndex::open_read_only(&quality_path).unwrap();
            assert_eq!(predecessor.record_count(), 6);
            assert!(!predecessor.live_doc_ids().unwrap().contains(&appended_id));
        }
        let quality = VectorIndex::open_read_only(&successor_quality_path).unwrap();
        assert_eq!(quality.embedder_revision(), fingerprint);
        assert_eq!(
            quality.quantization(),
            frankensearch_index::Quantization::F16
        );
        let expected_vector = frankensearch_core::SyncEmbed::embed_sync(&native, content).unwrap();
        let expected_path = temp.path().join("expected-native.fsvi");
        let mut expected_writer =
            VectorIndex::create(&expected_path, native.id(), native.dimension()).unwrap();
        expected_writer
            .write_record("expected", &expected_vector)
            .unwrap();
        expected_writer.finish().unwrap();
        let expected_index = VectorIndex::open_read_only(&expected_path).unwrap();
        let expected_persisted = expected_index.vector_at_f32(0).unwrap();
        let hits = quality.search_top_k(&expected_vector, 10, None).unwrap();
        let appended = hits
            .iter()
            .find(|hit| hit.doc_id == appended_id)
            .expect("the acknowledged append must be searchable in the selected image");
        let persisted = if complete {
            // Complete publication seals both tiers without pending WAL rows.
            assert_eq!(quality.record_count(), 7);
            assert_eq!(quality.wal_record_count(), 0);
            quality.vector_at_f32(appended.index as usize).unwrap()
        } else {
            quality
                .wal_records()
                .find(|(id, _)| *id == appended_id)
                .expect("the legacy append must persist a quality WAL row")
                .1
                .to_vec()
        };
        assert_eq!(
            persisted.as_slice(),
            expected_persisted.as_slice(),
            "append must embed content with the selected native producer, including F16 persistence"
        );
        eprintln!(
            "[multilingual-cli] real cross-language search/cache/daemon/append and incompatible producer refusal verified"
        );
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
    fn fresh_setup_defaults_select_only_production_ready_manifests() {
        let temp = tempfile::tempdir().expect("create fresh setup fixture");
        let fsfs = IsolatedFsfs::new(temp.path(), temp.path().join("models"));
        let defaults = ModelManifest::builtin_catalog();
        defaults.validate().expect("validate default model catalog");
        assert!(!defaults.models.is_empty());
        for manifest in &defaults.models {
            assert!(
                manifest.is_production_ready(),
                "fresh setup must not require an unqualified model: {}",
                manifest.id
            );
            assert!(manifest.files.iter().all(|file| file.size > 0));
            assert!(manifest.download_size_bytes > 0, "{}", manifest.id);
            assert_eq!(
                manifest.download_size_bytes,
                manifest.files.iter().map(|file| file.size).sum::<u64>(),
                "{} download budget must describe its verified artifacts",
                manifest.id
            );
        }
        assert!(
            defaults
                .models
                .iter()
                .all(|model| model.id != "flashrank-nano")
        );
        let reranker = defaults
            .models
            .iter()
            .find(|model| model.id == "ms-marco-minilm-l-6-v2")
            .expect("fresh setup retains the verified native reranker");
        assert_eq!(reranker.tier, Some(ModelTier::Reranker));
        assert!(
            reranker
                .files
                .iter()
                .any(|file| file.name.ends_with(".safetensors")),
            "the default reranker must provide weights for the native loader"
        );

        let mut registered = defaults.models;
        registered.extend(ModelManifest::opt_in_catalog().models);
        for key in ["indexing.fast_model", "indexing.quality_model"] {
            let outcome = fsfs.run(
                temp.path(),
                key,
                ["config", "get", key, "--format", "json"],
                FAILURE_TIMEOUT,
            );
            let envelope = parse_success_envelope("fresh setup default configuration", &outcome);
            assert_eq!(envelope["data"]["source"], "defaults", "{envelope}");
            let selected = envelope["data"]["value"]
                .as_str()
                .expect("configured model name");
            let manifest = registered
                .iter()
                .find(|model| model.id.eq_ignore_ascii_case(selected))
                .unwrap_or_else(|| panic!("{key}={selected} must have a registered manifest"));
            assert!(
                manifest.is_production_ready(),
                "{key}={selected} must be provisionable on a fresh machine"
            );
            assert!(manifest.total_size_bytes() > 0, "{key}={selected}");
        }
        assert_eq!(fs::read_dir(&fsfs.model_root).unwrap().count(), 0);
    }

    #[test]
    fn fresh_setup_lists_unqualified_flashrank_but_rejects_explicit_download() {
        let temp = tempfile::tempdir().expect("create unqualified model fixture");
        let fsfs = IsolatedFsfs::new(temp.path(), temp.path().join("models"));
        let manifest = ModelManifest::flashrank_nano();
        assert_eq!(manifest.repo, "prithivida/flashrank");
        assert_eq!(manifest.license, "CC-BY-SA-4.0");
        assert_eq!(manifest.total_size_bytes(), 3_421_983);
        assert_eq!(
            manifest.download_size_bytes,
            manifest.files.iter().map(|file| file.size).sum::<u64>()
        );
        assert!(!manifest.is_production_ready());
        assert!(
            ModelManifest::opt_in_catalog()
                .models
                .iter()
                .any(|model| model.id == manifest.id)
        );

        for (label, args) in [
            (
                "list-unqualified-models",
                vec!["download-models", "--list", "--format", "json"],
            ),
            (
                "verify-unqualified-model",
                vec![
                    "download-models",
                    "flashrank-nano",
                    "--verify",
                    "--format",
                    "json",
                ],
            ),
        ] {
            let outcome = fsfs.run(temp.path(), label, args, FAILURE_TIMEOUT);
            let envelope = parse_success_envelope(label, &outcome);
            let flashrank = envelope["data"]["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|model| model["id"] == manifest.id)
                .expect("FlashRank remains discoverable with an unavailable reason");
            assert_eq!(flashrank["state"], "unavailable", "{flashrank}");
            assert!(flashrank["verified"].is_null(), "{flashrank}");
            assert!(
                flashrank["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("production-ready")),
                "{flashrank}"
            );
        }

        for name in ["flashrank-nano", "flashrank"] {
            let outcome = fsfs.run(
                temp.path(),
                name,
                ["download-models", name, "--format", "json"],
                FAILURE_TIMEOUT,
            );
            assert!(!outcome.timed_out, "{outcome:?}");
            assert_eq!(outcome.status.code(), Some(2), "{outcome:?}");
            let envelope: Value = serde_json::from_str(&outcome.stdout).unwrap();
            assert_eq!(envelope["ok"], false, "{envelope}");
            assert_eq!(envelope["error"]["code"], "invalid_config");
            assert!(
                envelope["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("production-ready")),
                "{envelope}"
            );
        }
        assert_eq!(
            fs::read_dir(&fsfs.model_root).unwrap().count(),
            0,
            "rejecting an unqualified model must not leave staged artifacts or a verification receipt"
        );
    }

    #[test]
    fn offline_downloads_refuse_acquisition_without_mutating_missing_or_corrupt_caches() {
        let temp = tempfile::tempdir().expect("create offline acquisition fixture");
        let directory_entries = |path: &Path| {
            let mut entries = fs::read_dir(path)
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<Vec<_>>();
            entries.sort_unstable();
            entries
        };
        for corrupt in [false, true] {
            let case = if corrupt { "corrupt" } else { "missing" };
            let root = temp.path().join(case);
            let fsfs = IsolatedFsfs::new(&root, root.join("models"));
            let installed = fsfs.model_root.join("potion-base-8M");
            let corrupt_files: [(&str, &[u8]); 3] = [
                ("tokenizer.json", b"unverified tokenizer bytes"),
                ("model.safetensors", b"unverified model bytes"),
                (".verified", br#"{"schema_version":0,"verified":true}"#),
            ];
            if corrupt {
                fs::create_dir_all(&installed).unwrap();
                for (name, bytes) in corrupt_files {
                    fs::write(installed.join(name), bytes).unwrap();
                }
            }
            let root_before = directory_entries(&fsfs.model_root);
            let installed_before = corrupt.then(|| directory_entries(&installed));
            let assert_unchanged = || {
                assert_eq!(
                    directory_entries(&fsfs.model_root),
                    root_before,
                    "offline acquisition must not create a staging directory"
                );
                if let Some(before) = &installed_before {
                    assert_eq!(&directory_entries(&installed), before);
                    for (name, bytes) in corrupt_files {
                        assert_eq!(fs::read(installed.join(name)).unwrap(), bytes);
                    }
                }
            };

            for force in [false, true] {
                let mut args = vec![
                    "download-models",
                    "potion-base-8m",
                    "--offline",
                    "--format",
                    "json",
                ];
                if force {
                    args.push("--force");
                }
                let outcome = fsfs.run(
                    &root,
                    &format!("offline-{case}-force-{force}"),
                    args,
                    FAILURE_TIMEOUT,
                );
                assert!(
                    !outcome.timed_out,
                    "offline refusal must be prompt: {outcome:?}"
                );
                assert_eq!(outcome.status.code(), Some(2), "{outcome:?}");
                let envelope: Value = serde_json::from_str(&outcome.stdout).unwrap();
                assert_eq!(envelope["ok"], false, "{envelope}");
                assert_eq!(envelope["error"]["code"], "invalid_config");
                assert_eq!(envelope["error"]["field"], "indexing.offline");
                let message = envelope["error"]["message"].as_str().unwrap();
                assert!(message.contains("--offline"), "{message}");
                assert!(message.contains("potion-base-8m"), "{message}");
                assert!(envelope["data"].is_null());
                assert_unchanged();
            }

            let listed = fsfs.run(
                &root,
                &format!("offline-{case}-list"),
                [
                    "download-models",
                    "potion-base-8m",
                    "--offline",
                    "--list",
                    "--format",
                    "json",
                ],
                FAILURE_TIMEOUT,
            );
            let envelope = parse_success_envelope("offline model inspection", &listed);
            assert_eq!(envelope["data"]["operation"], "list");
            assert_eq!(envelope["data"]["models"].as_array().unwrap().len(), 1);
            assert_eq!(envelope["data"]["models"][0]["state"], case);
            assert_unchanged();

            let verified = fsfs.run(
                &root,
                &format!("offline-{case}-verify"),
                [
                    "download-models",
                    "potion-base-8m",
                    "--offline",
                    "--verify",
                    "--format",
                    "json",
                ],
                FAILURE_TIMEOUT,
            );
            assert!(!verified.timed_out, "{verified:?}");
            assert!(!verified.status.success(), "{verified:?}");
            let envelope: Value = serde_json::from_str(&verified.stdout).unwrap();
            assert_eq!(envelope["ok"], false);
            assert_eq!(
                envelope["error"]["code"],
                if corrupt {
                    "hash_mismatch"
                } else {
                    "model_not_found"
                },
                "offline verification must preserve the artifact error: {envelope}"
            );
            assert_unchanged();
        }
    }

    #[test]
    fn indexing_rejects_unknown_quality_selection_before_loading_or_publishing() {
        let temp = tempfile::tempdir().expect("create quality selection fixture");
        let corpus = temp.path().join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        fs::write(corpus.join("retry.md"), RETRY_DOCUMENT).unwrap();
        let fsfs = IsolatedFsfs::new(temp.path(), temp.path().join("models"));
        for (label, model) in [
            ("unknown", "missing-quality-model"),
            ("malformed", "all/minilm/l6/v2"),
        ] {
            let config = temp.path().join(format!("quality-{label}.toml"));
            fs::write(
                &config,
                format!(
                    "[indexing]\nfast_model = \"potion-base-8m\"\nquality_model = \"{model}\"\n"
                ),
            )
            .unwrap();
            let index = temp.path().join(format!("index-{label}"));
            let outcome = fsfs.run(
                temp.path(),
                &format!("index-quality-{label}"),
                [
                    "index",
                    corpus.to_str().unwrap(),
                    "--config",
                    config.to_str().unwrap(),
                    "--index-dir",
                    index.to_str().unwrap(),
                    "--offline",
                    "--format",
                    "json",
                ],
                FAILURE_TIMEOUT,
            );
            assert!(!outcome.timed_out, "{outcome:?}");
            assert_eq!(outcome.status.code(), Some(2), "{outcome:?}");
            let envelope: Value = serde_json::from_str(&outcome.stdout).unwrap();
            assert_eq!(envelope["ok"], false);
            assert_eq!(envelope["error"]["code"], "invalid_config");
            assert_eq!(envelope["error"]["field"], "indexing.quality_model");
            assert!(
                envelope["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(model)
            );
            assert!(!index.join("index_sentinel.json").exists());
            assert!(!index.join("vector/index.fsvi").exists());
            assert!(!index.join("CURRENT").exists());
            assert!(!index.join("FSFS-CURRENT").exists());
            assert_eq!(fs::read_dir(&fsfs.model_root).unwrap().count(), 0);
        }
    }

    #[test]
    fn upgrade_reuse_policy_is_discoverable_and_explicit_in_the_actual_cli() {
        let temp = tempfile::tempdir().unwrap();
        let fsfs = IsolatedFsfs::new(temp.path(), temp.path().join("empty-model-cache"));
        for (label, flags, expected) in [
            ("default", vec![], "same_executable"),
            (
                "enabled",
                vec!["--upgrade-reuse", "compatible-contract"],
                "compatible_contract",
            ),
            (
                "disabled",
                vec![
                    "--upgrade-reuse",
                    "compatible-contract",
                    "--upgrade-reuse",
                    "same-executable",
                ],
                "same_executable",
            ),
        ] {
            let mut args = vec![
                "config",
                "get",
                "indexing.upgrade_reuse",
                "--format",
                "json",
            ];
            args.extend(flags);
            let outcome = fsfs.run(temp.path(), label, args, FAILURE_TIMEOUT);
            assert!(!outcome.timed_out, "{}", outcome.combined_output());
            assert!(outcome.status.success(), "{}", outcome.combined_output());
            let envelope: Value = serde_json::from_str(&outcome.stdout).unwrap();
            assert_eq!(envelope["data"]["value"], expected, "{}", outcome.stdout);
        }
    }

    #[cfg(feature = "semantic-loaders")]
    #[test]
    #[ignore = "requires two source-bound executables and pinned real models; explicit upgrade lifecycle"]
    fn two_executables_reuse_only_an_unchanged_admitted_indexing_contract() {
        use sha2::Digest as _;
        let previous = PathBuf::from(
            std::env::var_os("FSFS_UPGRADE_REUSE_PREVIOUS_BINARY")
                .expect("supply the source-bound previous fsfs executable"),
        );
        assert!(previous.is_absolute() && previous.is_file());
        let current = fsfs_binary();
        let previous_sha = sha2::Sha256::digest(fs::read(&previous).unwrap());
        let current_sha = sha2::Sha256::digest(fs::read(&current).unwrap());
        assert_ne!(
            previous_sha, current_sha,
            "two genuinely different executables are required"
        );
        let models = configured_model_root();
        verify_pinned_model_cache(&models).expect("verified pinned real producers");
        let temp = tempfile::tempdir().unwrap();
        // Compare completed quality results on both executables. A cold
        // deadline fallback is a different phase, not evidence of index drift.
        // The dedicated cold-deadline cases retain their original 50ms budget.
        let ranking_config = temp.path().join("upgrade-ranking.toml");
        fs::write(&ranking_config, "[search]\nquality_timeout_ms = 5000\n").unwrap();
        let mut old = IsolatedFsfs::new(temp.path(), models.clone());
        old.binary = previous;
        let current = IsolatedFsfs::new(temp.path(), models);
        let observe = [
            ("RUST_LOG", "frankensearch_fsfs::indexing=debug"),
            ("FSFS_COMPLETE_GENERATIONS", "1"),
        ];
        // A separate incumbent per control prevents an earlier B run from
        // replacing the A receipt and accidentally turning upgrade into same-ELF reuse.
        for (control, flags, expected_carried) in [
            (
                "compatible",
                vec!["--upgrade-reuse", "compatible-contract"],
                QUICKSTART_DOCUMENT_COUNT,
            ),
            ("disabled", vec!["--upgrade-reuse", "same-executable"], 0),
            (
                "window-drift",
                vec!["--upgrade-reuse", "compatible-contract"],
                0,
            ),
            (
                "source-drift",
                vec!["--upgrade-reuse", "compatible-contract"],
                QUICKSTART_DOCUMENT_COUNT - 1,
            ),
            (
                "artifact-drift",
                vec!["--upgrade-reuse", "compatible-contract"],
                0,
            ),
            (
                "producer-drift",
                vec!["--upgrade-reuse", "compatible-contract"],
                0,
            ),
        ] {
            let corpus = temp.path().join(format!("{control}-corpus"));
            write_quickstart_corpus(&corpus);
            let index = temp.path().join(control);
            let base = [
                "index",
                corpus.to_str().unwrap(),
                "--index-dir",
                index.to_str().unwrap(),
                "--format",
                "json",
            ];
            let first = old.run_with_env(
                temp.path(),
                &format!("{control}-A"),
                base,
                QUICKSTART_TIMEOUT,
                &observe,
            );
            assert_finished_successfully("previous executable", &first);
            let prior_root = active_generation_root(&index);
            let prior_fast = fs::read(prior_root.join("vector/index.fsvi")).unwrap();
            let prior_quality = fs::read(prior_root.join("vector/quality.fsvi")).unwrap();
            let prior_receipt = fs::read(prior_root.join("FSFS-REUSE.json")).unwrap();
            let query = [
                "search",
                "network retries",
                "--no-daemon",
                "--config",
                ranking_config.to_str().unwrap(),
                "--index-dir",
                index.to_str().unwrap(),
                "--format",
                "json",
            ];
            let before = old.run(
                temp.path(),
                &format!("{control}-A-search"),
                query,
                QUICKSTART_TIMEOUT,
            );
            assert_finished_successfully("previous admitted search", &before);
            let before_page = parse_success_envelope(control, &before);
            assert_eq!(before_page["data"]["phase"], "refined", "{before_page}");
            let prior_selection = fs::read(index.join("FSFS-CURRENT")).unwrap();
            if control == "source-drift" {
                fs::write(
                    corpus.join("gardening.md"),
                    "Plant spring bulbs after preparing loose soil and adding compost.",
                )
                .unwrap();
            }
            if control == "artifact-drift" {
                let mut corrupted = prior_fast.clone();
                let last = corrupted.last_mut().expect("nonempty vector artifact");
                *last ^= 0xff;
                fs::write(prior_root.join("vector/index.fsvi"), corrupted).unwrap();
            }
            let mut args = base.to_vec();
            args.extend(flags);
            let producer_config = temp.path().join("unavailable-producer.toml");
            if control == "producer-drift" {
                fs::write(
                    &producer_config,
                    "[indexing]\nfast_model = 'potion-base-8m'\n",
                )
                .unwrap();
                args.extend(["--config", producer_config.to_str().unwrap()]);
            }
            let mut env = observe.to_vec();
            if control == "window-drift" {
                env.push(("FSFS_INDEXING_FAST_WINDOW_MAX_PER_FILE", "1"));
            }
            // A selected, unavailable producer must refuse before indexing,
            // rather than reuse rows from the old producer by matching dimension.
            let result = current.run_with_env(
                temp.path(),
                &format!("{control}-B"),
                args,
                QUICKSTART_TIMEOUT,
                &env,
            );
            if matches!(control, "artifact-drift" | "producer-drift") {
                assert!(!result.timed_out, "{}", result.combined_output());
                assert!(!result.status.success(), "{}", result.combined_output());
                assert_eq!(
                    fs::read(index.join("FSFS-CURRENT")).unwrap(),
                    prior_selection,
                    "failed admission must preserve the incumbent selection"
                );
                assert_eq!(
                    fs::read(prior_root.join("FSFS-REUSE.json")).unwrap(),
                    prior_receipt
                );
                assert!(
                    !result
                        .stderr
                        .contains("fsfs admitted indexing embedding batch"),
                    "refusal precedes embedding: {}",
                    result.stderr
                );
                eprintln!(
                    "[upgrade-reuse] control={control} source_documents={QUICKSTART_DOCUMENT_COUNT} refused=true incumbent_selection_preserved=true exit={:?}",
                    result.status.code()
                );
                continue;
            }
            assert_finished_successfully("current executable", &result);
            let envelope = parse_success_envelope(control, &result);
            assert_eq!(
                envelope["data"]["unchanged_files"],
                expected_carried,
                "{}",
                result.combined_output()
            );
            let calls = result
                .stderr
                .matches("fsfs admitted indexing embedding batch")
                .count();
            if control == "compatible" {
                assert_eq!(
                    calls, 0,
                    "no new indexing embedding admission: {}",
                    result.stderr
                );
                let next_root = active_generation_root(&index);
                assert_eq!(
                    fs::read(next_root.join("vector/index.fsvi")).unwrap(),
                    prior_fast
                );
                assert_eq!(
                    fs::read(next_root.join("vector/quality.fsvi")).unwrap(),
                    prior_quality
                );
            } else {
                assert!(
                    calls > 0,
                    "cold control must exercise the observation point: {}",
                    result.stderr
                );
            }
            assert_eq!(
                fs::read(prior_root.join("FSFS-REUSE.json")).unwrap(),
                prior_receipt,
                "a successor must preserve the incumbent receipt"
            );
            let query = [
                "search",
                "network retries",
                "--no-daemon",
                "--config",
                ranking_config.to_str().unwrap(),
                "--index-dir",
                index.to_str().unwrap(),
                "--format",
                "json",
            ];
            let searched = current.run(
                temp.path(),
                &format!("{control}-search"),
                query,
                QUICKSTART_TIMEOUT,
            );
            assert_finished_successfully("admitted successor search", &searched);
            let page = parse_success_envelope(control, &searched);
            assert_eq!(page["data"]["phase"], "refined", "{page}");
            assert!(!page["data"]["hits"].as_array().unwrap().is_empty());
            if control == "compatible" {
                assert_eq!(
                    page["data"]["hits"], before_page["data"]["hits"],
                    "unchanged rankings, scores and source snippets"
                );
            }
            eprintln!(
                "[upgrade-reuse] control={control} source_documents={QUICKSTART_DOCUMENT_COUNT} carried={expected_carried} indexing_embedding_admissions={calls} previous_sha={previous_sha:?} current_sha={current_sha:?} functional_only=true"
            );
        }
    }

    #[cfg(all(unix, feature = "semantic-loaders"))]
    #[test]
    #[ignore = "requires pinned real models; explicit ranking-prior product lifecycle"]
    fn ranking_prior_cli_opt_in_reorders_both_real_phases_and_explains_the_boost() {
        let models = configured_model_root();
        verify_pinned_model_cache(&models).expect("verified real models");
        let temp = tempfile::tempdir().unwrap();
        let corpus = temp.path().join("corpus");
        fs::create_dir_all(corpus.join("src")).unwrap();
        fs::create_dir_all(corpus.join("docs")).unwrap();
        // Equal relevant content holds semantic evidence constant. The lexical
        // tie favors docs; the exact underscore module spelling has the stronger
        // code-navigation prior. Both actual phases must change their winner.
        let relevant = "query execution engine scheduling and request processing";
        fs::write(corpus.join("docs/query-execution.md"), relevant).unwrap();
        fs::write(corpus.join("src/query_execution.rs"), relevant).unwrap();
        for number in 0..6 {
            fs::write(
                corpus.join(format!("notes-{number}.md")),
                "garden flowers soil watering and sunlight",
            )
            .unwrap();
        }
        let config = temp.path().join("fsfs.toml");
        fs::write(&config, "[search]\nquality_timeout_ms = 10000\n").unwrap();
        let index = temp.path().join("index");
        let fsfs = IsolatedFsfs::new(temp.path(), models);
        let indexed = fsfs.run_with_env(
            temp.path(),
            "prior-index",
            [
                "index",
                corpus.to_str().unwrap(),
                "--index-dir",
                index.to_str().unwrap(),
                "--config",
                config.to_str().unwrap(),
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
            &[("FSFS_COMPLETE_GENERATIONS", "1")],
        );
        assert_finished_successfully("prior fixture index", &indexed);
        struct StopDaemon(PathBuf);
        impl Drop for StopDaemon {
            fn drop(&mut self) {
                request_daemon_stop(&self.0, true);
            }
        }
        let _daemon_owner = StopDaemon(index.join("fsfs-query.sock"));
        let mut reference = [None, None];
        for transport in ["--no-daemon", "--daemon"] {
            for (label, flag, expected) in [
                ("off", "--no-ranking-priors", "docs/query-execution.md"),
                ("on", "--ranking-priors", "src/query_execution.rs"),
            ] {
                let outcome = fsfs.run(
                    temp.path(),
                    &format!("prior-{transport}-{label}"),
                    [
                        "search",
                        "query_execution",
                        transport,
                        flag,
                        "--index-dir",
                        index.to_str().unwrap(),
                        "--config",
                        config.to_str().unwrap(),
                        "--limit",
                        "5",
                        "--stream",
                        "--format",
                        "jsonl",
                    ],
                    QUICKSTART_TIMEOUT,
                );
                assert_finished_successfully("prior stream", &outcome);
                let frames: Vec<Value> = outcome
                    .stdout
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                assert_eq!(frames.last().unwrap()["payload"]["status"], "completed");
                let mut phases = [Vec::new(), Vec::new()];
                let mut selected = None;
                let mut seen = [false, false];
                for frame in &frames {
                    if frame["event"] == "progress" {
                        selected = match frame["payload"]["reason_code"].as_str() {
                            Some("query.stream.initial_ready") => Some(0),
                            Some("query.stream.refined_ready") => Some(1),
                            _ => selected,
                        };
                        if let Some(phase) = selected {
                            seen[phase] = true;
                        }
                    } else if frame["event"] == "result" {
                        phases[selected.expect("phase precedes results")]
                            .push(frame["payload"]["item"].clone());
                    }
                }
                assert_eq!(
                    seen,
                    [true, true],
                    "both actual semantic phases: {}",
                    outcome.stdout
                );
                for hits in &phases {
                    assert!(
                        hits[0]["path"].as_str().unwrap().ends_with(expected),
                        "discriminating {label} control: {}",
                        outcome.stdout
                    );
                }
                let policy_index = usize::from(label == "on");
                if transport == "--no-daemon" {
                    reference[policy_index] = Some(phases.clone());
                }
                if transport == "--daemon" {
                    assert_eq!(Some(&phases), reference[policy_index].as_ref());
                }
                let explained = fsfs.run(
                    temp.path(),
                    &format!("prior-explain-{transport}-{label}"),
                    [
                        "search",
                        "query_execution",
                        transport,
                        flag,
                        "--index-dir",
                        index.to_str().unwrap(),
                        "--config",
                        config.to_str().unwrap(),
                        "--limit",
                        "5",
                        "--explain",
                        "--format",
                        "json",
                    ],
                    QUICKSTART_TIMEOUT,
                );
                assert_finished_successfully("actual prior explanation", &explained);
                let page = parse_success_envelope(label, &explained);
                let explanations = page["data"]["explanations"].as_object().unwrap();
                let code = explanations
                    .iter()
                    .find(|(path, _)| path.ends_with("src/query_execution.rs"))
                    .unwrap()
                    .1;
                if label == "on" {
                    assert!(
                        code["fusion"]["prior_boost"]
                            .as_f64()
                            .is_some_and(|boost| boost > 0.0 && boost <= 0.01)
                    );
                } else {
                    assert!(code["fusion"].get("prior_boost").is_none());
                }
                eprintln!(
                    "[ranking-priors] transport={transport} policy={label} corpus_documents=8 phase_denominator=2 phases={phases:?} functional_only=true"
                );
            }
        }
        let stopped = fsfs.run(
            temp.path(),
            "prior-daemon-stop",
            [
                "daemon",
                "--stop",
                "--index-dir",
                index.to_str().unwrap(),
                "--config",
                config.to_str().unwrap(),
                "--format",
                "json",
            ],
            FAILURE_TIMEOUT,
        );
        assert_finished_successfully("stop prior test daemon", &stopped);
    }

    fn assert_toon_stream_frame_root_fields(raw: &Value) {
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
    }

    #[test]
    #[should_panic(expected = "TOON record contains non-frame fields")]
    fn toon_stream_root_guard_rejects_appended_ordinary_envelope() {
        // The old writer appended an ordinary envelope inside the terminal's
        // RS record. Typed decoding accepts the merged object and drops the
        // extra root fields, so inspect the raw object before that projection.
        let record = r#"v: 1
schema_version: fsfs.stream.query.v1
stream_id: stream-envelope-negative
seq: 1
ts: "2026-10-28T01:26:27Z"
command: search
event: terminal
payload:
  status: failed
  exit_code: 78
  failure_category: model
  error:
    code: embedder_unavailable
    message: No semantic models available
    exit_code: 78
  retry:
    policy: none
v: 1
ts: "2026-10-28T01:26:27Z"
ok: false
error:
  code: embedder_unavailable
  message: No semantic models available
  exit_code: 78
meta:
  command: search
  format: toon
"#;
        let frame: frankensearch_fsfs::stream_protocol::StreamFrame<Value> =
            frankensearch_fsfs::stream_protocol::decode_stream_frame_toon(record)
                .expect("typed decoder accepts the appended ordinary envelope");
        assert!(frankensearch_fsfs::stream_protocol::validate_stream_frame(&frame).valid);
        let raw: Value = toon_format::decode_default(record).expect("decode planted TOON record");
        assert_toon_stream_frame_root_fields(&raw);
    }

    #[cfg(feature = "semantic-loaders")]
    #[test]
    fn missing_models_stream_has_one_final_terminal_in_jsonl_and_toon() {
        let temp = tempfile::tempdir().unwrap();
        let index = temp.path().join("index");
        fs::create_dir_all(index.join("vector")).unwrap();
        // Storage/protocol fixture only: no semantic inference claim. The
        // empty model cache must refuse before admitting or scoring any row.
        let mut vectors = VectorIndex::create(
            &index.join("vector/index.fsvi"),
            "potion-multilingual-128M",
            256,
        )
        .unwrap();
        let mut row = vec![0.0; 256];
        row[0] = 1.0;
        vectors.write_record("one.md", &row).unwrap();
        vectors.finish().unwrap();
        let original = fs::read(index.join("vector/index.fsvi")).unwrap();
        let fsfs = IsolatedFsfs::new(temp.path(), temp.path().join("empty-model-cache"));
        for format in ["jsonl", "toon"] {
            let outcome = fsfs.run(
                temp.path(),
                &format!("missing-model-stream-{format}"),
                [
                    "search",
                    "network retries",
                    "--index-dir",
                    index.to_str().unwrap(),
                    "--no-daemon",
                    "--stream",
                    "--format",
                    format,
                ],
                FAILURE_TIMEOUT,
            );
            assert!(!outcome.timed_out, "{}", outcome.combined_output());
            assert_eq!(
                outcome.status.code(),
                Some(78),
                "{}",
                outcome.combined_output()
            );
            let frames: Vec<Value> = if format == "jsonl" {
                outcome
                    .stdout
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect()
            } else {
                assert!(outcome.stdout.starts_with('\u{1e}'));
                outcome
                    .stdout
                    .split('\u{1e}')
                    .skip(1)
                    .map(|record| {
                        let raw: Value = toon_format::decode_default(record.trim())
                            .expect("every TOON record is a complete object");
                        assert_toon_stream_frame_root_fields(&raw);
                        let frame: frankensearch_fsfs::stream_protocol::StreamFrame<Value> =
                            frankensearch_fsfs::stream_protocol::decode_stream_frame_toon(
                                record.trim(),
                            )
                            .unwrap();
                        serde_json::to_value(frame).unwrap()
                    })
                    .collect()
            };
            assert_eq!(
                frames.len(),
                2,
                "no envelope may follow terminal: {}",
                outcome.stdout
            );
            for (seq, frame) in frames.iter().enumerate() {
                assert_eq!(frame["schema_version"], "fsfs.stream.query.v1");
                assert_eq!(frame["seq"], seq);
                assert_eq!(frame["stream_id"], frames[0]["stream_id"]);
            }
            assert_eq!(frames[0]["event"], "started");
            assert_ne!(frames[0]["payload"]["semantic_admitted"], true);
            assert_eq!(frames[1]["event"], "terminal");
            assert_eq!(frames[1]["payload"]["status"], "failed");
            assert_eq!(frames[1]["payload"]["exit_code"], 78);
            assert_eq!(
                frames[1]["payload"]["error"]["code"],
                "embedder_unavailable"
            );
            assert_eq!(fs::read(index.join("vector/index.fsvi")).unwrap(), original);
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

    #[cfg(all(unix, feature = "semantic-loaders"))]
    struct WatchChild(std::process::Child);

    #[cfg(all(unix, feature = "semantic-loaders"))]
    impl Drop for WatchChild {
        fn drop(&mut self) {
            if self.0.try_wait().ok().flatten().is_none() {
                let _ = self.0.kill();
            }
            let _ = self.0.wait();
        }
    }

    #[cfg(all(unix, feature = "semantic-loaders"))]
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

    /// Each vector tier in `generation_root` holds exactly one live row per
    /// expected file, encoding that file's current text, and no WAL rows.
    #[cfg(all(unix, feature = "semantic-loaders"))]
    fn assert_vector_tiers_encode_current_files(
        model_root: &Path,
        generation_root: &Path,
        corpus: &Path,
        expected_ids: &[&str],
    ) {
        let stack = frankensearch_embed::EmbedderStack::auto_detect_with_options(
            Some(model_root),
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
            let vectors = VectorIndex::open_read_only(&generation_root.join(file))
                .expect("reopen watched vector tier");
            let hits = vectors
                .search_top_k(&vec![1.0; vectors.dimension()], 10, None)
                .expect("enumerate live tier");
            let mut ids = hits
                .iter()
                .map(|hit| hit.doc_id.clone())
                .collect::<Vec<_>>();
            ids.sort();
            assert_eq!(ids, expected_ids, "{file}");
            assert_eq!(vectors.wal_record_count(), 0, "{file}: no WAL rows remain");
            let corpus = corpus.to_path_buf();
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
    }

    #[cfg(all(unix, feature = "semantic-loaders"))]
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
        // The legacy watcher: a build pins the layout, and the watch below
        // follows the legacy index on disk.
        let initial = fsfs.run_with_env(
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
            &[("FSFS_COMPLETE_GENERATIONS", "0")],
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
                .env("FSFS_COMPLETE_GENERATIONS", "0")
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
        assert_vector_tiers_encode_current_files(
            &fsfs.model_root,
            &index,
            &corpus,
            &["astronomy.md", "database.md", "handoff.md", "live.md"],
        );
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

    /// The complete-generation layout's watch, judged by what a user sees:
    /// independent searches succeed while it runs (the legacy layout refuses
    /// them with its map lock), each edit becomes searchable through a
    /// published generation, SIGTERM stops it cleanly, and the selected
    /// generation's vectors encode the final files.
    #[cfg(all(unix, feature = "semantic-loaders"))]
    #[test]
    #[ignore = "real-model complete-generation watch; requires the pinned model cache and semantic E2E opt-in"]
    fn complete_generation_watch_serves_search_and_publishes_live_edits() -> Result<(), String> {
        log_binary_profile("real-model-complete-watch");
        if std::env::var("FRANKENSEARCH_REQUIRE_SEMANTIC_E2E").as_deref() != Ok("1") {
            return Err(
                "set FRANKENSEARCH_REQUIRE_SEMANTIC_E2E=1 for real-model watch validation"
                    .to_owned(),
            );
        }
        let model_root = configured_model_root();
        verify_pinned_model_cache(&model_root)?;
        let temp = tempfile::tempdir().expect("complete watch fixture");
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
        let initial = fsfs.run_with_env(
            temp.path(),
            "complete-watch-initial-index",
            [
                "index",
                corpus_arg,
                "--index-dir",
                index_arg,
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
            &[("FSFS_COMPLETE_GENERATIONS", "1")],
        );
        let envelope = parse_success_envelope("complete watch initial index", &initial);
        assert_eq!(envelope["data"]["indexed_files"], 2);
        assert!(
            index.join("FSFS-CURRENT").is_file(),
            "the complete-generation layout was selected"
        );

        let mut stdout_path = fsfs.log_root.join("complete-watch.stdout.log");
        let mut stderr_path = fsfs.log_root.join("complete-watch.stderr.log");
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
                .env("FSFS_COMPLETE_GENERATIONS", "1")
                .env("RUST_LOG", "info")
                .stdout(File::create(&stdout_path).unwrap())
                .stderr(File::create(&stderr_path).unwrap())
                .spawn()
                .expect("spawn real complete-generation watcher"),
        );
        let needle = "\"generation_complete\":true";
        let output = wait_for_watch_output(&mut watch, &stdout_path, needle, 1, QUICKSTART_TIMEOUT);
        let first: Value = serde_json::from_str(
            output
                .lines()
                .find(|line| line.contains(needle))
                .expect("a publication receipt line"),
        )
        .expect("watch publication receipt");
        assert_eq!(first["meta"]["command"], "watch");
        assert_eq!(first["data"]["indexed_files"], 2);
        assert_eq!(
            first["data"]["index_root"],
            first["data"]["generation_path"]
        );

        // Searches are separate processes on the store the watcher holds.
        // Polling them is the visibility criterion: one edit can publish more
        // than once, so receipt counts cannot say which edit a search sees.
        // A forwarded search goes through the store's daemon, which a plain
        // search uses whenever its socket exists, with no fallback.
        let lexical_paths = |label: &str, query: &str, forwarded: bool| -> Vec<String> {
            let mut args = vec!["search", query, "--fast-only"];
            if !forwarded {
                args.push("--no-daemon");
            }
            args.extend([
                "--index-dir",
                index_arg,
                "--format",
                "json",
                "--limit",
                "10",
            ]);
            let outcome = fsfs.run(temp.path(), label, args, QUICKSTART_TIMEOUT);
            parse_success_envelope(label, &outcome)["data"]["hits"]
                .as_array()
                .expect("search hits")
                .iter()
                .filter(|hit| hit["lexical_rank"].is_number())
                .map(|hit| hit["path"].as_str().expect("hit path").to_owned())
                .collect()
        };
        let await_search = |watch: &mut WatchChild,
                            label: &str,
                            query: &str,
                            path: &str,
                            present: bool,
                            forwarded| {
            let started = Instant::now();
            loop {
                let paths = lexical_paths(label, query, forwarded);
                if paths.iter().any(|hit| hit == path) == present {
                    eprintln!(
                        "[default-build-e2e] stage=complete-watch {label} visible_after_ms={}",
                        started.elapsed().as_millis()
                    );
                    return;
                }
                assert!(
                    watch.0.try_wait().expect("poll watcher").is_none(),
                    "watcher exited during {label}"
                );
                assert!(
                    started.elapsed() < QUICKSTART_TIMEOUT,
                    "{label}: {path} present={present} not reached for {query:?}; last hits {paths:?}"
                );
                thread::sleep(Duration::from_millis(200));
            }
        };
        // The store's daemon serves while the watcher publishes: every
        // forwarded search admits the generation selected at that moment.
        let socket = index.join("fsfs-query.sock");
        let mut live_daemon = WatchChild(
            fsfs.command(temp.path())
                .args(["daemon", "--index-dir", index_arg, "--idle-timeout-ms", "0"])
                .stdout(
                    File::create(fsfs.log_root.join("complete-live-daemon.stdout.log")).unwrap(),
                )
                .stderr(
                    File::create(fsfs.log_root.join("complete-live-daemon.stderr.log")).unwrap(),
                )
                .spawn()
                .expect("spawn complete-generation daemon during the watch"),
        );
        let listen_started = Instant::now();
        while !socket.exists() {
            assert!(
                live_daemon.0.try_wait().expect("poll daemon").is_none(),
                "daemon exited before listening"
            );
            assert!(
                listen_started.elapsed() < QUICKSTART_TIMEOUT,
                "daemon did not listen at {}",
                socket.display()
            );
            thread::sleep(Duration::from_millis(50));
        }
        await_search(
            &mut watch,
            "during-watch",
            "Oldquartz",
            "database.md",
            true,
            false,
        );
        await_search(
            &mut watch,
            "during-watch-daemon",
            "Oldquartz",
            "database.md",
            true,
            true,
        );
        fs::write(
            corpus.join("live.md"),
            "Liveamber handles transient failures with bounded retries and exponential backoff.",
        )
        .unwrap();
        await_search(
            &mut watch,
            "live-create",
            "Liveamber",
            "live.md",
            true,
            false,
        );
        fs::write(corpus.join("live.md"), "Livemalachite restores service after failures using bounded retries and exponential backoff.").unwrap();
        await_search(
            &mut watch,
            "live-modify",
            "Livemalachite",
            "live.md",
            true,
            false,
        );
        await_search(
            &mut watch,
            "live-modify-old-text",
            "Liveamber",
            "live.md",
            false,
            false,
        );
        // The deletion check below is only meaningful if this query finds
        // the file while it exists.
        await_search(
            &mut watch,
            "delete-control",
            "Astronomers",
            "astronomy.md",
            true,
            false,
        );
        fs::remove_file(corpus.join("astronomy.md")).unwrap();
        await_search(
            &mut watch,
            "live-delete",
            "Astronomers",
            "astronomy.md",
            false,
            false,
        );
        fs::write(
            corpus.join("moved.md"),
            "Livecobalt notes keep their rows when the watcher sees them renamed.",
        )
        .unwrap();
        await_search(
            &mut watch,
            "rename-control",
            "Livecobalt",
            "moved.md",
            true,
            true,
        );
        fs::rename(corpus.join("moved.md"), corpus.join("renamed.md")).unwrap();
        await_search(
            &mut watch,
            "rename-new-path",
            "Livecobalt",
            "renamed.md",
            true,
            true,
        );
        await_search(
            &mut watch,
            "rename-old-path",
            "Livecobalt",
            "moved.md",
            false,
            true,
        );
        await_search(
            &mut watch,
            "rename-direct",
            "Livecobalt",
            "renamed.md",
            true,
            false,
        );

        // A killed watcher must leave a complete readable generation. Keep an
        // actual old reader and the already-warm daemon alive while an edit
        // arrives with no watcher, then require restart to catch it up.
        await_search(
            &mut watch,
            "restart-cache-prime",
            "Livemalachite",
            "live.md",
            true,
            true,
        );
        let reader_runtime = asupersync::runtime::RuntimeBuilder::current_thread()
            .build()
            .unwrap();
        let reader_cx = reader_runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
        let reader_store =
            frankensearch_fsfs::generation_store::CompleteGenerationStore::open(&reader_cx, &index)
                .unwrap();
        let old_reader = reader_store.active(&reader_cx).unwrap().unwrap();
        let old_manifest = fs::read(old_reader.path().join("FSFS-BUNDLE.json")).unwrap();
        let original_daemon_pid = live_daemon.0.id();
        watch.0.kill().expect("kill only this owned watcher");
        let killed_started = Instant::now();
        let killed_status = loop {
            if let Some(status) = watch.0.try_wait().expect("poll killed watcher") {
                break status;
            }
            assert!(
                killed_started.elapsed() < QUICKSTART_TIMEOUT,
                "killed watcher did not exit"
            );
            thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(
            std::os::unix::process::ExitStatusExt::signal(&killed_status),
            Some(libc::SIGKILL),
            "watcher must actually stop without graceful teardown"
        );
        let old_selection = fs::read(index.join("FSFS-CURRENT")).unwrap();
        assert!(reader_store.active(&reader_cx).unwrap().is_some());
        fs::write(corpus.join("live.md"), "Restartsapphire restores service after failures using bounded retries and exponential backoff.").unwrap();
        assert_eq!(fs::read(index.join("FSFS-CURRENT")).unwrap(), old_selection);
        assert_eq!(
            fs::read(old_reader.path().join("FSFS-BUNDLE.json")).unwrap(),
            old_manifest
        );
        assert!(
            lexical_paths("stopped-watch-old", "Livemalachite", true)
                .iter()
                .any(|path| path == "live.md")
        );
        assert!(
            !lexical_paths("stopped-watch-new", "Restartsapphire", true)
                .iter()
                .any(|path| path == "live.md")
        );

        stdout_path = fsfs.log_root.join("complete-watch-restart.stdout.log");
        stderr_path = fsfs.log_root.join("complete-watch-restart.stderr.log");
        watch = WatchChild(
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
                .env("FSFS_COMPLETE_GENERATIONS", "1")
                .env("RUST_LOG", "info")
                .stdout(File::create(&stdout_path).unwrap())
                .stderr(File::create(&stderr_path).unwrap())
                .spawn()
                .expect("restart watcher over its existing complete store"),
        );
        await_search(
            &mut watch,
            "restart-daemon-new",
            "Restartsapphire",
            "live.md",
            true,
            true,
        );
        await_search(
            &mut watch,
            "restart-daemon-old",
            "Livemalachite",
            "live.md",
            false,
            true,
        );
        await_search(
            &mut watch,
            "restart-direct-new",
            "Restartsapphire",
            "live.md",
            true,
            false,
        );
        await_search(
            &mut watch,
            "restart-direct-old",
            "Livemalachite",
            "live.md",
            false,
            false,
        );
        assert_ne!(active_generation_root(&index).as_path(), old_reader.path());
        assert_eq!(
            fs::read(old_reader.path().join("FSFS-BUNDLE.json")).unwrap(),
            old_manifest,
            "the old reader must retain its immutable generation across restart"
        );
        assert_eq!(live_daemon.0.id(), original_daemon_pid);
        assert!(live_daemon.0.try_wait().unwrap().is_none());
        drop(old_reader);
        let status = terminate_and_wait(&mut live_daemon, "complete daemon during the watch");
        assert!(
            status.success(),
            "graceful complete daemon shutdown during the watch failed: {status}"
        );
        drop(live_daemon);

        let status = terminate_and_wait(&mut watch, "complete watch");
        let stdout = fs::read_to_string(&stdout_path).unwrap();
        let stderr = fs::read_to_string(&stderr_path).unwrap();
        eprintln!(
            "[default-build-e2e] stage=complete-watch-shutdown status={status} stdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(
            status.success(),
            "graceful complete watcher shutdown failed"
        );
        drop(watch);

        assert_vector_tiers_encode_current_files(
            &fsfs.model_root,
            &active_generation_root(&index),
            &corpus,
            &["database.md", "live.md", "renamed.md"],
        );
        let config = temp.path().join("complete-watch-search.toml");
        // Functional coverage uses an explicit budget, not a latency claim.
        fs::write(&config, "[search]\nquality_timeout_ms = 5000\n").unwrap();
        let outcome = fsfs.run(
            temp.path(),
            "complete-watch-refined",
            [
                "search",
                "how livemalachite restores service after failures",
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
        let result = parse_success_envelope("complete watch refined search", &outcome);
        assert_eq!(result["data"]["phase"], "refined", "{result}");
        let hit = result["data"]["hits"]
            .as_array()
            .unwrap()
            .iter()
            .find(|hit| hit["path"] == "live.md")
            .expect("watched document in actual search results");
        assert!(hit["lexical_rank"].is_number(), "{hit}");
        assert!(hit["semantic_rank"].is_number(), "{hit}");

        // The store's own daemon answers a plain search, which forwards
        // whenever the store socket exists and does not fall back once it
        // does, and it also stops cleanly on SIGTERM.
        let socket = index.join("fsfs-query.sock");
        let mut daemon = WatchChild(
            fsfs.command(temp.path())
                .args(["daemon", "--index-dir", index_arg, "--idle-timeout-ms", "0"])
                .stdout(File::create(fsfs.log_root.join("complete-daemon.stdout.log")).unwrap())
                .stderr(File::create(fsfs.log_root.join("complete-daemon.stderr.log")).unwrap())
                .spawn()
                .expect("spawn complete-generation daemon"),
        );
        let listen_started = Instant::now();
        while !socket.exists() {
            assert!(
                daemon.0.try_wait().expect("poll daemon").is_none(),
                "daemon exited before listening"
            );
            assert!(
                listen_started.elapsed() < QUICKSTART_TIMEOUT,
                "daemon did not listen at {}",
                socket.display()
            );
            thread::sleep(Duration::from_millis(50));
        }
        let forwarded = fsfs.run(
            temp.path(),
            "complete-daemon-search",
            [
                "search",
                "how livemalachite restores service after failures",
                "--index-dir",
                index_arg,
                "--format",
                "json",
                "--limit",
                "10",
            ],
            QUICKSTART_TIMEOUT,
        );
        let result = parse_success_envelope("complete daemon search", &forwarded);
        assert!(
            result["data"]["hits"]
                .as_array()
                .unwrap()
                .iter()
                .any(|hit| hit["path"] == "live.md"),
            "{result}"
        );
        let status = terminate_and_wait(&mut daemon, "complete daemon");
        assert!(
            status.success(),
            "graceful complete daemon shutdown failed: {status}"
        );
        assert!(!socket.exists(), "the daemon removes its own socket");
        drop(daemon);
        eprintln!(
            "[default-build-e2e] stage=complete-watch event=verified search_during_watch=true live_create=true live_modify=true live_delete=true live_rename=true daemon_during_watch=true hard_watch_restart=true restart_pending_edit=true retained_reader=true stable_daemon=true fast=true quality=true post_exit=true daemon=true"
        );
        Ok(())
    }

    /// Send SIGTERM to an owned child and wait for its exit status.
    #[cfg(all(unix, feature = "semantic-loaders"))]
    fn terminate_and_wait(child: &mut WatchChild, label: &str) -> std::process::ExitStatus {
        let signal = Command::new("kill")
            .args(["-TERM", &child.0.id().to_string()])
            .status()
            .expect("signal own child");
        assert!(signal.success());
        let started = Instant::now();
        loop {
            if let Some(status) = child.0.try_wait().expect("poll graceful shutdown") {
                return status;
            }
            assert!(
                started.elapsed() < FAILURE_TIMEOUT,
                "{label} shutdown timed out"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(feature = "semantic-loaders")]
    #[test]
    #[ignore = "requires pinned Potion and MiniLM caches; executes the production CLI"]
    fn doctor_rejects_stale_real_model_producer_revisions() -> Result<(), String> {
        log_binary_profile("doctor-producer-revision");
        let model_root = configured_model_root();
        verify_pinned_model_cache(&model_root)?;
        let temp = tempfile::tempdir().expect("doctor producer fixture");
        let corpus = temp.path().join("corpus");
        let index = temp.path().join("index");
        write_quickstart_corpus(&corpus);
        let fsfs = IsolatedFsfs::new(temp.path(), model_root);
        let indexed = fsfs.run(
            temp.path(),
            "doctor-producer-index",
            [
                "index",
                corpus.to_str().unwrap(),
                "--index-dir",
                index.to_str().unwrap(),
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
        );
        parse_success_envelope("doctor producer index", &indexed);

        for (tier, filename) in [("fast", "index.fsvi"), ("quality", "quality.fsvi")] {
            let source = active_generation_root(&index).join("vector").join(filename);
            let original = fs::read(&source).expect("real indexed vectors");
            let reader = VectorIndex::open_read_only(&source).expect("real generation reader");
            assert!(
                reader.record_count() > 0,
                "fixture must contain real model vectors"
            );
            let mut revision = reader.embedder_revision().as_bytes().to_vec();
            assert_eq!(revision.len(), 64, "complete producer fingerprint");
            revision[0] = if revision[0] == b'a' { b'b' } else { b'a' };
            let stale_revision = String::from_utf8(revision).expect("hex producer fingerprint");
            let isolated = temp.path().join(format!("doctor-{tier}"));
            fs::create_dir_all(isolated.join("vector")).expect("private vector directory");
            // Full-mode doctor also admits cross-tier document membership.
            // Keep the healthy counterpart so only the selected producer
            // revision differs in this otherwise valid two-tier fixture.
            let counterpart = if tier == "fast" {
                "quality.fsvi"
            } else {
                "index.fsvi"
            };
            fs::copy(
                active_generation_root(&index)
                    .join("vector")
                    .join(counterpart),
                isolated.join("vector").join(counterpart),
            )
            .expect("preserve healthy counterpart tier");
            let candidate = isolated.join("vector").join(filename);
            // Use the normal writer so the stale identity has a valid header
            // checksum. A raw byte flip would only exercise corruption refusal.
            let mut writer = VectorIndex::create_with_revision(
                &candidate,
                reader.embedder_id(),
                &stale_revision,
                reader.dimension(),
                reader.quantization(),
            )
            .expect("stale private generation writer")
            .with_publication_nonce(reader.publication_nonce());
            for row in 0..reader.record_count() {
                writer
                    .write_record(
                        reader.doc_id_at(row).unwrap(),
                        &reader.vector_at_f32(row).unwrap(),
                    )
                    .expect("retain genuine model vector");
            }
            writer.finish().expect("seal valid stale generation");
            {
                let stale_reader = VectorIndex::open_read_only(&candidate)
                    .expect("stale fixture must be structurally valid, including CRC");
                assert_eq!(stale_reader.embedder_revision(), stale_revision);
                assert_eq!(stale_reader.record_count(), reader.record_count());
                assert_eq!(stale_reader.quantization(), reader.quantization());
                for row in 0..reader.record_count() {
                    assert_eq!(
                        stale_reader.doc_id_at(row).unwrap(),
                        reader.doc_id_at(row).unwrap()
                    );
                    let actual = stale_reader.vector_at_f32(row).unwrap();
                    let expected = reader.vector_at_f32(row).unwrap();
                    assert!(
                        actual
                            .into_iter()
                            .map(f32::to_bits)
                            .eq(expected.into_iter().map(f32::to_bits)),
                        "fixture must preserve every genuine vector bit"
                    );
                }
            }
            drop(reader);
            let stale = fs::read(&candidate).expect("valid stale generation bytes");
            let refused = fsfs.run(
                temp.path(),
                &format!("doctor-{tier}-stale"),
                [
                    "doctor",
                    "--index-dir",
                    isolated.to_str().unwrap(),
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            assert!(!refused.timed_out, "doctor must terminate: {refused:?}");
            assert_eq!(refused.status.code(), Some(1), "{refused:?}");
            let envelope: Value =
                serde_json::from_str(&refused.stdout).expect("doctor refusal JSON");
            assert_eq!(envelope["ok"], false);
            assert_eq!(envelope["error"]["code"], "subsystem_error");
            let diagnostic = envelope["error"].to_string();
            assert!(
                diagnostic.contains(&format!("model.{tier}")),
                "{diagnostic}"
            );
            assert!(
                diagnostic.contains("Embedding space identity is unverifiable."),
                "{diagnostic}"
            );
            assert!(diagnostic.contains("fsfs index --full"), "{diagnostic}");
            assert_eq!(
                fs::read(&candidate).unwrap(),
                stale,
                "doctor must not repair in place"
            );
            assert_eq!(
                fs::read(&source).unwrap(),
                original,
                "source generation must stay intact"
            );

            fs::write(&candidate, &original).expect("restore exact producer in private fixture");
            let accepted = fsfs.run(
                temp.path(),
                &format!("doctor-{tier}-current"),
                [
                    "doctor",
                    "--index-dir",
                    isolated.to_str().unwrap(),
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            let envelope = parse_success_envelope("doctor exact producer control", &accepted);
            assert_eq!(envelope["data"]["fail_count"], 0);
            assert_eq!(fs::read(&candidate).unwrap(), original);
            eprintln!(
                "[default-build-e2e] stage=doctor-producer tier={tier} stale=refused current=accepted read_only=true"
            );
        }
        Ok(())
    }

    #[cfg(feature = "semantic-loaders")]
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
                [
                    "download-models",
                    model_id,
                    "--offline",
                    "--verify",
                    "--format",
                    "json",
                ],
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
            let receipt_before = fs::read(&receipt_path).unwrap();
            let model_directories = || {
                fs::read_dir(&model_root)
                    .unwrap()
                    .map(|entry| entry.unwrap().file_name())
                    .collect::<std::collections::BTreeSet<_>>()
            };
            let directories_before = model_directories();
            let cached_outcome = fsfs.run(
                temp.path(),
                &format!("offline-cached-{model_id}"),
                ["download-models", model_id, "--offline", "--format", "json"],
                QUICKSTART_TIMEOUT,
            );
            let cached = parse_success_envelope("offline verified cache reuse", &cached_outcome);
            assert_eq!(cached["data"]["operation"], "download");
            assert_eq!(cached["data"]["models"].as_array().unwrap().len(), 1);
            assert_eq!(cached["data"]["models"][0]["id"], model_id);
            assert_eq!(cached["data"]["models"][0]["state"], "cached");
            assert_eq!(cached["data"]["models"][0]["verified"], true);

            let forced_outcome = fsfs.run(
                temp.path(),
                &format!("offline-force-cached-{model_id}"),
                [
                    "download-models",
                    model_id,
                    "--offline",
                    "--force",
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
            );
            assert!(!forced_outcome.timed_out, "{forced_outcome:?}");
            assert_eq!(forced_outcome.status.code(), Some(2), "{forced_outcome:?}");
            let forced: Value = serde_json::from_str(&forced_outcome.stdout).unwrap();
            assert_eq!(forced["ok"], false);
            assert_eq!(forced["error"]["code"], "invalid_config");
            assert_eq!(forced["error"]["field"], "indexing.offline");
            assert!(
                forced["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(model_id)
            );
            assert_eq!(fs::read(&receipt_path).unwrap(), receipt_before);
            assert_eq!(model_directories(), directories_before);
            assert!(is_verification_cached(
                &manifest,
                &model_root.join(install_dir)
            ));
            eprintln!(
                "[default-build-e2e] stage=cli-model-verify event=verified model={model_id} exit=0 ok=true receipt_refreshed=true offline_cached_reuse=true offline_force_refused=true cache_preserved=true"
            );
        }

        verify_selected_quality_never_falls_back_to_cached_minilm(temp.path(), &model_root);

        // Exercise the FastEmbed caller itself with a real cached ONNX model,
        // then make that same receipt stale without changing size or mtime.
        // Auto-detection is deliberately bypassed: it can reject bad inputs
        // before the FastEmbed loader whose receipt wiring this proves.
        let cache_probe = temp.path().join("quality-receipt-control");
        let manifest = ModelManifest::minilm_v2();
        for entry in &manifest.files {
            let target = cache_probe.join(&entry.name);
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::copy(
                model_root.join("all-MiniLM-L6-v2").join(&entry.name),
                target,
            )
            .unwrap();
        }
        frankensearch_embed::model_manifest::verify_dir_and_record(&manifest, &cache_probe)
            .expect("mint real quality verification receipt");
        let cached = frankensearch_embed::FastEmbedEmbedder::load(&cache_probe)
            .expect("real FastEmbed load with a valid receipt");
        drop(cached);
        let tokenizer = cache_probe.join("tokenizer.json");
        let modified = fs::metadata(&tokenizer).unwrap().modified().unwrap();
        let original_marker = fs::read(cache_probe.join(".verified")).unwrap();
        let mut bytes = fs::read(&tokenizer).unwrap();
        bytes[0] ^= 1;
        fs::write(&tokenizer, &bytes).unwrap();
        File::options()
            .write(true)
            .open(&tokenizer)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert_eq!(
            fs::metadata(&tokenizer).unwrap().modified().unwrap(),
            modified
        );
        let rejected = frankensearch_embed::FastEmbedEmbedder::load(&cache_probe)
            .expect_err("a stale receipt must not admit changed tokenizer bytes");
        match rejected {
            frankensearch_core::SearchError::ModelLoadFailed { source, .. } => {
                assert!(
                    source
                        .to_string()
                        .contains("tokenizer.json:sha256-or-size-mismatch"),
                    "the frozen manifest must identify the corrupted tokenizer: {source}"
                );
            }
            // ubs:ignore -- An unexpected error must fail this checksum regression test.
            other => panic!("expected frozen artifact checksum rejection, got {other}"),
        }
        assert_eq!(
            fs::read(cache_probe.join(".verified")).unwrap(),
            original_marker
        );
        eprintln!(
            "[default-build-e2e] stage=fastembed-receipt event=verified real_onnx=true cached_load=true same_size_restored_mtime_corruption=frozen_artifact_checksum_rejection receipt_unchanged=true"
        );

        let index_outcome = fsfs.run_with_env(
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
            &[("FRANKENSEARCH_LOG", "info")],
        );
        let index_envelope = parse_success_envelope("default-build index", &index_outcome);
        for directory in ["potion-multilingual-128M", "all-MiniLM-L6-v2"] {
            let receipt: Value = serde_json::from_slice(
                &fs::read(model_root.join(directory).join(".verified"))
                    .expect("read authoritative loader verification receipt"),
            )
            .expect("parse authoritative loader verification receipt");
            let fingerprint = receipt["manifest_fingerprint"]
                .as_str()
                .expect("receipt fingerprint");
            assert!(
                index_outcome.stderr.lines().any(|line| {
                    line.contains("model verification receipt accepted")
                        && line.contains(&format!("receipt_manifest_fingerprint={fingerprint}"))
                }),
                "index must log the admitted {directory} receipt, not merely report model availability"
            );
        }
        eprintln!(
            "[default-build-e2e] stage=index-one-shot event=verified timed_out=false exit=0 elapsed_ms={} watch_requested=false",
            index_outcome.elapsed.as_millis()
        );

        let sentinel_path = active_generation_root(&index).join("index_sentinel.json");
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
            active_generation_root(&index)
                .join("lexical/CURRENT")
                .is_file()
                || index.join("CURRENT").is_file(),
            "the Quill lexical generation must publish a CURRENT pointer"
        );
        verify_index_output_formats(&fsfs, temp.path(), &corpus, &index);
        verify_fast_only_policy(&fsfs, temp.path(), &corpus, &index);

        let vector_path = active_generation_root(&index).join("vector/index.fsvi");
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
        let quality_path = active_generation_root(&index).join("vector/quality.fsvi");
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

        // The REFINED assertion below is about the quality tier, not a loaded
        // host's cold-start latency: like the file's other REFINED checks, the
        // stage gets 5 s (the 500 ms default, which counts cold model loading,
        // failed one gate run at load ~40 and passed its rerun).
        let refine_config = temp.path().join("hybrid-refine.toml");
        fs::write(&refine_config, "[search]\nquality_timeout_ms = 5000\n")
            .expect("write hybrid search config");
        let hybrid_search_outcome = fsfs.run(
            temp.path(),
            "hybrid-search",
            [
                "search",
                "How should a network client recover from transient failures using exponential backoff, bounded retries, and random jitter?",
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--config",
                refine_config.to_str().expect("UTF-8 config path"),
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
        let stream_frames = two_tier_outcome
            .stdout
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("every stream frame is JSON"))
            .collect::<Vec<_>>();
        let stream_reason_codes = stream_frames
            .iter()
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
        let phase_positions =
            ["query.stream.initial_ready", "query.stream.refined_ready"].map(|reason| {
                let positions = stream_frames
                    .iter()
                    .enumerate()
                    .filter_map(|(index, frame)| {
                        (frame["payload"]["reason_code"] == reason).then_some(index)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(positions.len(), 1, "one announcement per phase: {reason}");
                positions[0]
            });
        assert!(
            phase_positions[0] < phase_positions[1],
            "Initial precedes Refined"
        );
        for (begin, end) in [
            phase_positions.into(),
            (phase_positions[1], stream_frames.len() - 1),
        ] {
            let results = stream_frames[begin + 1..end]
                .iter()
                .filter(|frame| frame["event"] == "result")
                .collect::<Vec<_>>();
            assert!(
                !results.is_empty(),
                "phase announcement must carry actual results"
            );
            assert_eq!(results[0]["payload"]["item"]["path"], "retry.md");
            assert_eq!(results[0]["payload"]["item"]["in_both_sources"], true);
            assert_eq!(
                stream_frames[begin]["payload"]["completed_units"].as_u64(),
                Some(u64::try_from(results.len()).expect("result count fits u64"))
            );
        }
        let terminal = stream_frames.last().expect("stream terminal");
        assert_eq!(terminal["event"], "terminal");
        assert_eq!(terminal["payload"]["status"], "completed");
        assert_eq!(terminal["payload"]["exit_code"], 0);
        eprintln!(
            "[default-build-e2e] stage=quality-tier-load event=verified fast_only_quality_loaded=false two_tier_quality_loads=1 refined_ready=true"
        );
        #[cfg(feature = "semantic-loaders")]
        verify_real_blend_and_deadline(&fsfs, temp.path(), &index);
        // The raw progressive protocol (attested frames, per-request budgets)
        // is the legacy query daemon's; a complete store's daemon streams the
        // CLI frame protocol, which its own tests and the forwarded lane cover.
        #[cfg(all(unix, feature = "semantic-loaders"))]
        {
            let legacy_index = temp.path().join("legacy-index");
            let legacy = fsfs.run_with_env(
                temp.path(),
                "progressive-legacy-index",
                [
                    "index",
                    corpus.to_str().expect("UTF-8 corpus path"),
                    "--index-dir",
                    legacy_index.to_str().expect("UTF-8 index path"),
                    "--format",
                    "json",
                ],
                QUICKSTART_TIMEOUT,
                &[("FSFS_COMPLETE_GENERATIONS", "0")],
            );
            parse_success_envelope("legacy index for the progressive daemon", &legacy);
            verify_real_fast_model_cache_isolation(&fsfs, temp.path(), &legacy_index);
            verify_real_progressive_daemon(&fsfs, temp.path(), &legacy_index);
        }

        // Reality check 2026-09-01, G2: the auto-spawned query daemon must
        // outlive the search that spawned it (so the next search is warm), be
        // stoppable through its protocol, and reclaim itself after its idle
        // timeout so it can never linger as an orphan.
        #[cfg(unix)]
        {
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
            assert!(
                UnixStream::connect(&socket).is_ok(),
                "connect to stop the daemon"
            );
            request_daemon_stop(&socket, is_complete_store(&index));
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
    const { assert!(cfg!(feature = "semantic-loaders")) };
}

#[cfg(feature = "embedded-models")]
#[test]
fn embedded_materializer_child_verifies_models_and_observers_leave_cache_absent() {
    use frankensearch_embed::bundled_default_models::default_semantic_models_are_materialized;
    use frankensearch_fsfs::runtime::BUNDLED_MODEL_MATERIALIZER_FLAG;
    use std::process::{Command, Stdio};

    let root = tempfile::tempdir().expect("private materializer fixture");
    let models = root.path().join("models");
    let binary = env!("CARGO_BIN_EXE_fsfs");
    for command in ["status", "doctor"] {
        // ubs:ignore -- Cargo supplies this integration-test executable path, not user input.
        let output = Command::new(binary)
            .args([command, "--format", "json", "--index-dir"])
            .arg(root.path().join("index"))
            .env("FRANKENSEARCH_MODEL_DIR", &models)
            .env("FRANKENSEARCH_OFFLINE", "1")
            .env("FRANKENSEARCH_ALLOW_DOWNLOAD", "0")
            .env("FRANKENSEARCH_CHECK_UPDATES", "0")
            .stdin(Stdio::null())
            .output()
            .expect("run real observer CLI");
        assert!(
            output.status.success(),
            "{command} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!models.exists(), "{command} materialized bundled models");
    }
    // ubs:ignore -- Cargo supplies this integration-test executable path, not user input.
    let output = Command::new(binary)
        .arg(BUNDLED_MODEL_MATERIALIZER_FLAG)
        .arg(&models)
        .stdin(Stdio::null())
        .output()
        .expect("run the real materializer child");
    assert!(
        output.status.success(),
        "materializer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "child polluted the command output"
    );
    assert!(default_semantic_models_are_materialized(&models).unwrap());
    for (manifest, directory) in [
        (
            frankensearch_embed::ModelManifest::potion_128m(),
            "potion-multilingual-128M",
        ),
        (
            frankensearch_embed::ModelManifest::minilm_v2(),
            "all-MiniLM-L6-v2",
        ),
    ] {
        for file in manifest.files {
            frankensearch_embed::model_manifest::verify_file_sha256(
                &models.join(directory).join(file.name),
                &file.sha256,
                file.size,
            )
            .expect("materialized bytes match the published model manifest");
        }
    }
}

#[cfg(feature = "embedded-models")]
#[test]
fn embedded_materializer_child_propagates_unwritable_model_root() {
    let root = tempfile::tempdir().expect("private materializer failure fixture");
    let obstruction = root.path().join("not-a-directory");
    std::fs::write(&obstruction, b"preserve this file").unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fsfs"))
        .arg(frankensearch_fsfs::runtime::BUNDLED_MODEL_MATERIALIZER_FLAG)
        .arg(&obstruction)
        .output()
        .expect("run materializer against a real filesystem obstruction");
    assert!(!output.status.success());
    assert_eq!(std::fs::read(obstruction).unwrap(), b"preserve this file");
}
