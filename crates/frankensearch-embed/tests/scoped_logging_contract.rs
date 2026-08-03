//! Fresh-process proof that `run_test_with_cx` does not implicitly bridge
//! `tokenizers`' `log` output into tracing.
//!
//! The `log` logger is process-global and cannot be uninstalled. The two
//! cases therefore run through this integration-test binary as child
//! processes, with bounded concurrent stdout/stderr drains.

use std::{
    ffi::OsString,
    io::{Read, Write},
    path::PathBuf,
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{self, RecvTimeoutError},
    thread,
    time::{Duration, Instant},
};

use asupersync::test_utils::{install_global_test_log_bridge, run_test_with_cx};
use sha2::{Digest, Sha256};
use tokenizers::{
    Tokenizer,
    models::wordlevel::WordLevel,
    normalizers::{NFD, Sequence, StripAccents},
    tokenizer::{NormalizedString, Normalizer},
};

const CHILD_CASE_ENV: &str = "FRANKENSEARCH_SCOPED_LOGGING_CHILD_CASE";
const CHILD_TEST: &str = "fresh_process_child";
const MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_LINES: usize = 10_000;
const CHILD_TIMEOUT: Duration = Duration::from_secs(20);
const TOKENIZERS_TRACE_MARKER: &str = "transform_range call";

#[derive(Debug)]
struct ChildOutput {
    status: ExitStatus,
    bytes: Vec<u8>,
    lines: usize,
    output_sha256: String,
}

#[derive(Debug)]
enum ChildFailure {
    OutputLimit {
        bytes: usize,
        lines: usize,
        status: ExitStatus,
        output_sha256: String,
    },
    Timeout {
        status: ExitStatus,
        output_sha256: String,
    },
}

fn output_sha256(bytes: &[u8], trailing: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.update(trailing);
    let digest = hasher.finalize();
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn assert_sha256_digest(digest: &str) {
    assert_eq!(digest.len(), 64, "output digest must be SHA-256");
    assert!(
        digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "output digest must be hexadecimal"
    );
}

fn normalizer() -> Sequence {
    Sequence::new(vec![NFD.into(), StripAccents.into()])
}

fn real_tokenizer() -> Tokenizer {
    let mut tokenizer = Tokenizer::new(WordLevel::default());
    tokenizer
        .with_normalizer(Some(normalizer()))
        .expect("real tokenizers normalizer configuration must be valid");
    tokenizer
}

fn exercise_real_tokenizer_body() {
    let mut normalized = NormalizedString::from("Crème brûlée");
    normalizer()
        .normalize(&mut normalized)
        .expect("NFD plus StripAccents must normalize a real NormalizedString");
    assert_eq!(normalized.get(), "Creme brulee");

    let tokenizer = real_tokenizer();
    let error = tokenizer
        .encode("Crème brûlée", false)
        .expect_err("the empty in-memory WordLevel must reject its missing UNK token");
    assert!(error.to_string().contains("Missing [UNK] token"));
}

fn exercise_real_tokenizer() {
    run_test_with_cx(|_cx| async move {
        // WordLevel has no vocabulary on purpose. `encode` reaches the configured
        // real normalizer before reporting the model's expected missing-UNK error.
        exercise_real_tokenizer_body();
    });
}

fn witness_worker_dispatch_contract() {
    run_test_with_cx(|_cx| async move {
        let explicit_dispatch = tracing::dispatcher::get_default(|dispatch| dispatch.clone());

        let unpropagated = thread::spawn(|| {
            tracing::dispatcher::get_default(|dispatch| {
                dispatch.is::<tracing::subscriber::NoSubscriber>()
            })
        });
        assert!(
            unpropagated
                .join()
                .expect("unpropagated worker must complete"),
            "worker unexpectedly inherited its caller dispatcher"
        );

        let propagated = thread::spawn(move || {
            tracing::dispatcher::with_default(&explicit_dispatch, || {
                tracing::dispatcher::get_default(|dispatch| {
                    !dispatch.is::<tracing::subscriber::NoSubscriber>()
                })
            })
        });
        assert!(
            propagated
                .join()
                .expect("explicitly propagated worker must complete"),
            "worker did not observe its explicit dispatcher"
        );
    });
}

fn witness_overlapping_runtimes() {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (go_a_tx, go_a_rx) = mpsc::channel();
    let (go_b_tx, go_b_rx) = mpsc::channel();

    let ready_a = ready_tx.clone();
    let worker_a = thread::spawn(move || {
        run_test_with_cx(|_cx| async move {
            ready_a.send(()).expect("runtime A must report readiness");
            go_a_rx
                .recv_timeout(CHILD_TIMEOUT)
                .expect("runtime A must be released");
            exercise_real_tokenizer_body();
        });
    });
    let worker_b = thread::spawn(move || {
        run_test_with_cx(|_cx| async move {
            ready_tx.send(()).expect("runtime B must report readiness");
            go_b_rx
                .recv_timeout(CHILD_TIMEOUT)
                .expect("runtime B must be released");
            exercise_real_tokenizer_body();
        });
    });

    ready_rx
        .recv_timeout(CHILD_TIMEOUT)
        .expect("runtime A must reach the overlap gate");
    ready_rx
        .recv_timeout(CHILD_TIMEOUT)
        .expect("runtime B must reach the overlap gate");
    go_a_tx
        .send(())
        .expect("runtime A release must be delivered");
    go_b_tx
        .send(())
        .expect("runtime B release must be delivered");
    worker_a.join().expect("runtime A must complete");
    worker_b.join().expect("runtime B must complete");
}

fn spawn_drain<R>(mut reader: R, stream_is_stderr: bool, tx: mpsc::Sender<(bool, Vec<u8>)>)
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut buffer = [0_u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) | Err(_) => {
                    let _ = tx.send((stream_is_stderr, Vec::new()));
                    return;
                }
                Ok(read) => {
                    if tx
                        .send((stream_is_stderr, buffer[..read].to_vec()))
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    });
}

fn terminate_and_reap(child: &mut Child) -> ExitStatus {
    if let Some(status) = child.try_wait().expect("child status must be observable") {
        return status;
    }
    let _ = child.kill();
    child.wait().expect("terminated child must be reaped")
}

fn read_bounded_child(mut child: Child, timeout: Duration) -> Result<ChildOutput, ChildFailure> {
    let stdout = child.stdout.take().expect("child stdout must be piped");
    let stderr = child.stderr.take().expect("child stderr must be piped");
    let (tx, rx) = mpsc::channel();
    spawn_drain(stdout, false, tx.clone());
    spawn_drain(stderr, true, tx);

    let started = Instant::now();
    let mut bytes = Vec::new();
    let mut lines = 0;
    let mut closed_streams = 0;
    let mut terminal_status = None;
    loop {
        if started.elapsed() > timeout {
            return Err(ChildFailure::Timeout {
                status: terminate_and_reap(&mut child),
                output_sha256: output_sha256(&bytes, &[]),
            });
        }

        match rx.recv_timeout(Duration::from_millis(10)) {
            Ok((_is_stderr, chunk)) if chunk.is_empty() => closed_streams += 1,
            Ok((_is_stderr, chunk)) => {
                for byte in &chunk {
                    if *byte == b'\n' {
                        lines += 1;
                    }
                }
                if bytes.len() + chunk.len() > MAX_OUTPUT_BYTES || lines > MAX_OUTPUT_LINES {
                    return Err(ChildFailure::OutputLimit {
                        bytes: bytes.len() + chunk.len(),
                        lines,
                        status: terminate_and_reap(&mut child),
                        output_sha256: output_sha256(&bytes, &chunk),
                    });
                }
                bytes.extend_from_slice(&chunk);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => closed_streams = 2,
        }

        if terminal_status.is_none() {
            terminal_status = child.try_wait().expect("child status must be observable");
        }
        if terminal_status.is_some() && closed_streams == 2 {
            break;
        }
    }

    let output_sha256 = output_sha256(&bytes, &[]);
    Ok(ChildOutput {
        status: terminal_status.expect("exited child must have a status"),
        bytes,
        lines,
        output_sha256,
    })
}

fn spawn_child(case: &str, rust_log: Option<OsString>) -> Child {
    let mut command = Command::new(std::env::current_exe().expect("test binary must exist"));
    command
        .arg(CHILD_TEST)
        .arg("--exact")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD_CASE_ENV, case)
        .env_remove("RUST_LOG")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(rust_log) = rust_log {
        command.env("RUST_LOG", rust_log);
    }
    command
        .spawn()
        .expect("fresh-process logging child must start")
}

fn run_child_with_rust_log(
    case: &str,
    rust_log: Option<OsString>,
) -> Result<ChildOutput, ChildFailure> {
    read_bounded_child(spawn_child(case, rust_log), CHILD_TIMEOUT)
}

fn run_child(case: &str, rust_log: Option<&str>) -> Result<ChildOutput, ChildFailure> {
    run_child_with_rust_log(case, rust_log.map(OsString::from))
}

fn run_child_with_timeout(
    case: &str,
    rust_log: Option<&str>,
    timeout: Duration,
) -> Result<ChildOutput, ChildFailure> {
    read_bounded_child(spawn_child(case, rust_log.map(OsString::from)), timeout)
}

fn run_passing_child(case: &str, rust_log: Option<&str>) -> ChildOutput {
    run_child(case, rust_log).unwrap_or_else(|failure| {
        panic!("fresh-process case {case:?} exceeded a bound: {failure:?}");
    })
}

#[cfg(unix)]
fn run_child_with_non_unicode_rust_log(case: &str) -> Result<ChildOutput, ChildFailure> {
    use std::os::unix::ffi::OsStringExt;

    run_child_with_rust_log(case, Some(OsString::from_vec(vec![b't', 0x80])))
}

#[cfg(unix)]
fn run_passing_child_with_non_unicode_rust_log(case: &str) -> ChildOutput {
    run_child_with_non_unicode_rust_log(case).unwrap_or_else(|failure| {
        panic!("fresh-process case {case:?} exceeded a bound: {failure:?}");
    })
}

fn assert_child_passed(case: &str, output: &ChildOutput) -> String {
    let text = String::from_utf8_lossy(&output.bytes).into_owned();
    assert_sha256_digest(&output.output_sha256);
    assert!(
        output.status.success(),
        "fresh-process case {case:?} failed after {} lines:\n{text}",
        output.lines,
    );
    text
}

fn assert_marker_is_suppressed(case: &str, output: &ChildOutput) {
    let output = assert_child_passed(case, output);
    assert!(
        !output.contains(TOKENIZERS_TRACE_MARKER),
        "{case} must not expose tokenizers TRACE records:\n{output}"
    );
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("embed manifest must be two directories below the workspace root")
        .to_path_buf()
}

fn lock_package_block<'a>(lock: &'a str, name: &str, version: &str) -> &'a str {
    lock.split("\n[[package]]")
        .find(|block| {
            block.contains(&format!("name = \"{name}\""))
                && block.contains(&format!("version = \"{version}\""))
        })
        .unwrap_or_else(|| panic!("Cargo.lock must contain {name} {version}"))
}

#[test]
fn fresh_process_contract_binds_pinned_dependency_and_source_identities() {
    let root = workspace_root();
    let lock = std::fs::read_to_string(root.join("Cargo.lock"))
        .expect("fresh-process contract must read the workspace Cargo.lock");
    for (name, version) in [("asupersync", "0.3.10"), ("tokenizers", "0.23.1")] {
        let block = lock_package_block(&lock, name, version);
        assert!(
            block.contains("source = \"registry+https://github.com/rust-lang/crates.io-index\""),
            "{name} {version} must be a registry-pinned dependency"
        );
        let checksum = block
            .lines()
            .find_map(|line| line.strip_prefix("checksum = \"")?.strip_suffix('"'))
            .expect("registry package must carry a Cargo.lock checksum");
        assert_eq!(checksum.len(), 64, "checksum must be a SHA-256 hex digest");
        assert!(
            checksum.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "checksum must be hexadecimal"
        );
    }

    assert!(
        std::env::current_exe()
            .expect("fresh-process test binary must exist")
            .is_file(),
        "fresh-process contract must execute a concrete test binary"
    );
    assert!(
        root.join("crates/frankensearch-embed/tests/scoped_logging_contract.rs")
            .is_file(),
        "fresh-process contract source must be present"
    );
}

#[test]
fn real_tokenizers_log_output_requires_explicit_bridge_and_trace() {
    let default_child = run_passing_child("default", None);
    assert_marker_is_suppressed("default", &default_child);

    let valid_without_bridge_child = run_passing_child("valid-trace-without-bridge", Some("trace"));
    assert_marker_is_suppressed("valid-trace-without-bridge", &valid_without_bridge_child);

    let empty_filter_child = run_passing_child("empty-rust-log", Some(""));
    assert_marker_is_suppressed("empty-rust-log", &empty_filter_child);

    let malformed_filter_child = run_passing_child("malformed-rust-log", Some("["));
    assert_marker_is_suppressed("malformed-rust-log", &malformed_filter_child);

    #[cfg(unix)]
    {
        let non_unicode_filter_child =
            run_passing_child_with_non_unicode_rust_log("nonunicode-rust-log");
        assert_marker_is_suppressed("nonunicode-rust-log", &non_unicode_filter_child);
    }

    let bridge_without_trace_child = run_passing_child("bridge-without-trace", Some("warn"));
    assert_marker_is_suppressed("bridge-without-trace", &bridge_without_trace_child);

    let bridge_off_child = run_passing_child("bridge-off", Some("off"));
    assert_marker_is_suppressed("bridge-off", &bridge_off_child);

    let panic_restoration_child = run_passing_child("panic-restoration", Some("trace"));
    assert_marker_is_suppressed("panic-restoration", &panic_restoration_child);

    let worker_dispatch_child = run_passing_child("worker-dispatch", Some("trace"));
    assert_marker_is_suppressed("worker-dispatch", &worker_dispatch_child);

    let overlapping_runtimes_child = run_passing_child("overlapping-runtimes", Some("trace"));
    assert_marker_is_suppressed("overlapping-runtimes", &overlapping_runtimes_child);

    let explicit_child = run_passing_child("explicit-bridge", Some("trace"));
    let explicit = assert_child_passed("explicit-bridge", &explicit_child);
    assert!(
        explicit.contains(TOKENIZERS_TRACE_MARKER),
        "explicit LogTracer plus TRACE must expose real tokenizers logs:\n{explicit}"
    );
}

#[test]
fn fresh_process_output_limit_reaps_noisy_child() {
    match run_child("line-overflow", None) {
        Err(ChildFailure::OutputLimit {
            bytes,
            lines,
            status,
            output_sha256,
        }) => {
            assert_sha256_digest(&output_sha256);
            assert!(bytes > 0, "bounded receipt must record observed output");
            assert!(lines > MAX_OUTPUT_LINES, "line limit did not trigger");
            assert!(
                !status.success(),
                "overflowing child must be terminated rather than succeed"
            );
        }
        Err(ChildFailure::Timeout {
            status,
            output_sha256,
        }) => {
            assert_sha256_digest(&output_sha256);
            panic!("noisy child hit wall-time bound instead of line cap: {status:?}");
        }
        Ok(output) => panic!(
            "noisy child unexpectedly passed after {} bytes and {} lines",
            output.bytes.len(),
            output.lines,
        ),
    }
}

#[test]
fn fresh_process_byte_limit_reaps_noisy_child() {
    match run_child("byte-overflow", None) {
        Err(ChildFailure::OutputLimit {
            bytes,
            lines,
            status,
            output_sha256,
        }) => {
            assert_sha256_digest(&output_sha256);
            assert!(bytes > MAX_OUTPUT_BYTES, "byte limit did not trigger");
            assert!(
                lines <= MAX_OUTPUT_LINES,
                "byte-only child reached the line cap before the byte cap"
            );
            assert!(
                !status.success(),
                "overflowing child must be terminated rather than succeed"
            );
        }
        Err(ChildFailure::Timeout {
            status,
            output_sha256,
        }) => {
            assert_sha256_digest(&output_sha256);
            panic!("byte-only child hit wall-time bound instead of byte cap: {status:?}");
        }
        Ok(output) => panic!(
            "byte-only child unexpectedly passed after {} bytes and {} lines",
            output.bytes.len(),
            output.lines,
        ),
    }
}

#[test]
fn fresh_process_timeout_reaps_stalled_child() {
    match run_child_with_timeout("timeout-stall", None, Duration::from_millis(50)) {
        Err(ChildFailure::Timeout {
            status,
            output_sha256,
        }) => {
            assert_sha256_digest(&output_sha256);
            assert!(
                !status.success(),
                "stalled child must be terminated rather than succeed"
            );
        }
        Err(ChildFailure::OutputLimit {
            bytes,
            lines,
            status,
            output_sha256,
        }) => panic!(
            "stalled child unexpectedly hit output cap: {bytes} bytes, {lines} lines, {status:?}, {output_sha256}"
        ),
        Ok(output) => panic!(
            "stalled child unexpectedly passed after {} bytes and {} lines",
            output.bytes.len(),
            output.lines,
        ),
    }
}

#[test]
fn fresh_process_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };

    match case.as_str() {
        "default"
        | "valid-trace-without-bridge"
        | "empty-rust-log"
        | "malformed-rust-log"
        | "nonunicode-rust-log" => exercise_real_tokenizer(),
        "bridge-without-trace" | "bridge-off" | "explicit-bridge" => {
            install_global_test_log_bridge()
                .expect("fresh child must permit an explicit global LogTracer");
            exercise_real_tokenizer();
        }
        "panic-restoration" => {
            let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_test_with_cx(|_cx| async { panic!("scoped dispatcher panic probe") });
            }));
            assert!(panic_result.is_err(), "panic probe unexpectedly returned");
            let restored = tracing::dispatcher::get_default(|dispatch| {
                dispatch.is::<tracing::subscriber::NoSubscriber>()
            });
            assert!(
                restored,
                "run_test_with_cx leaked its dispatcher after panic"
            );
            exercise_real_tokenizer();
        }
        "worker-dispatch" => witness_worker_dispatch_contract(),
        "overlapping-runtimes" => witness_overlapping_runtimes(),
        "line-overflow" => loop {
            println!("bounded-child-output-overflow");
        },
        "byte-overflow" => {
            let mut stdout = std::io::stdout().lock();
            let chunk = [b'x'; 8_192];
            loop {
                stdout
                    .write_all(&chunk)
                    .expect("byte-overflow child must write stdout");
                stdout
                    .flush()
                    .expect("byte-overflow child must flush stdout");
            }
        }
        "timeout-stall" => thread::sleep(Duration::from_secs(1)),
        other => panic!("unknown fresh-process logging child case {other:?}"),
    }
}
