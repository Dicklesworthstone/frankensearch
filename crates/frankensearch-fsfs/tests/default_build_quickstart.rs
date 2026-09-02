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

            let mut command = Command::new(&verified_executable);
            command
                .args(args)
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
                .env_remove("HUGGINGFACE_HUB_CACHE")
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
        assert_finished_successfully("default-build index", &index_outcome);
        eprintln!(
            "[default-build-e2e] stage=index-one-shot event=verified timed_out=false exit=0 elapsed_ms={} watch_requested=false",
            index_outcome.elapsed.as_millis()
        );

        let sentinel_path = index.join("index_sentinel.json");
        let sentinel: Value = serde_json::from_slice(
            &fs::read(&sentinel_path).expect("read durable index completion sentinel"),
        )
        .expect("parse durable index completion sentinel");
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

        // Reality check 2026-09-01, G1 cost half: the fsfs read path serves one
        // fast-tier generation, so a search must never open the quality model
        // (an ONNX session load on every process start). Positive observable:
        // the potion load is logged; planted negative: no FastEmbed load.
        let lazy_outcome = fsfs.run_with_env(
            temp.path(),
            "search-loads-fast-tier-only",
            [
                "search",
                "bounded retries with exponential backoff",
                "--index-dir",
                index.to_str().expect("UTF-8 index path"),
                "--no-daemon",
                "--limit",
                SEARCH_LIMIT,
                "--format",
                "json",
            ],
            QUICKSTART_TIMEOUT,
            &[("FRANKENSEARCH_LOG", "info")],
        );
        assert_finished_successfully("fast-tier-only search", &lazy_outcome);
        assert!(
            lazy_outcome.stderr.contains("Model2Vec model loaded"),
            "the fast tier must be loaded for a semantic search; stderr:\n{}",
            lazy_outcome.stderr
        );
        assert!(
            !lazy_outcome.stderr.contains("FastEmbed model loaded"),
            "a search over a fast-tier generation must not load the quality model; stderr:\n{}",
            lazy_outcome.stderr
        );
        eprintln!(
            "[default-build-e2e] stage=lazy-quality-load event=verified fast_loaded=true quality_loaded=false"
        );

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
