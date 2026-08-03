//! Fresh-process proof that `run_test_with_cx` does not implicitly bridge
//! `tokenizers`' `log` output into tracing.
//!
//! The `log` logger is process-global and cannot be uninstalled. The two
//! cases therefore run through this integration-test binary as child
//! processes, with bounded concurrent stdout/stderr drains.

use std::{
    ffi::OsString,
    io::Read,
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{self, RecvTimeoutError},
    thread,
    time::{Duration, Instant},
};

use asupersync::test_utils::{install_global_test_log_bridge, run_test_with_cx};
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

fn exercise_real_tokenizer() {
    let mut normalized = NormalizedString::from("Crème brûlée");
    normalizer()
        .normalize(&mut normalized)
        .expect("NFD plus StripAccents must normalize a real NormalizedString");
    assert_eq!(normalized.get(), "Creme brulee");

    let tokenizer = real_tokenizer();
    run_test_with_cx(|_cx| async move {
        // WordLevel has no vocabulary on purpose. `encode` reaches the configured
        // real normalizer before reporting the model's expected missing-UNK error.
        let error = tokenizer
            .encode("Crème brûlée", false)
            .expect_err("the empty in-memory WordLevel must reject its missing UNK token");
        assert!(error.to_string().contains("Missing [UNK] token"));
    });
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

fn terminate_and_reap(child: &mut Child) {
    if child
        .try_wait()
        .expect("child status must be observable")
        .is_none()
    {
        let _ = child.kill();
    }
    let _ = child.wait();
}

fn read_bounded_child(mut child: Child) -> ChildOutput {
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
        if started.elapsed() > CHILD_TIMEOUT {
            terminate_and_reap(&mut child);
            panic!("fresh-process logging child exceeded {CHILD_TIMEOUT:?}");
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
                    terminate_and_reap(&mut child);
                    panic!(
                        "fresh-process logging child exceeded output bound: {} bytes, {lines} lines",
                        bytes.len() + chunk.len(),
                    );
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

    ChildOutput {
        status: terminal_status.expect("exited child must have a status"),
        bytes,
        lines,
    }
}

fn run_child_with_rust_log(case: &str, rust_log: Option<OsString>) -> ChildOutput {
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
    read_bounded_child(
        command
            .spawn()
            .expect("fresh-process logging child must start"),
    )
}

fn run_child(case: &str, rust_log: Option<&str>) -> ChildOutput {
    run_child_with_rust_log(case, rust_log.map(OsString::from))
}

#[cfg(unix)]
fn run_child_with_non_unicode_rust_log(case: &str) -> ChildOutput {
    use std::os::unix::ffi::OsStringExt;

    run_child_with_rust_log(case, Some(OsString::from_vec(vec![b't', 0x80])))
}

fn assert_child_passed(case: &str, output: &ChildOutput) -> String {
    let text = String::from_utf8_lossy(&output.bytes).into_owned();
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

#[test]
fn real_tokenizers_log_output_requires_explicit_bridge_and_trace() {
    let default_child = run_child("default", None);
    assert_marker_is_suppressed("default", &default_child);

    let valid_without_bridge_child = run_child("valid-trace-without-bridge", Some("trace"));
    assert_marker_is_suppressed("valid-trace-without-bridge", &valid_without_bridge_child);

    let empty_filter_child = run_child("empty-rust-log", Some(""));
    assert_marker_is_suppressed("empty-rust-log", &empty_filter_child);

    let malformed_filter_child = run_child("malformed-rust-log", Some("["));
    assert_marker_is_suppressed("malformed-rust-log", &malformed_filter_child);

    #[cfg(unix)]
    {
        let non_unicode_filter_child = run_child_with_non_unicode_rust_log("nonunicode-rust-log");
        assert_marker_is_suppressed("nonunicode-rust-log", &non_unicode_filter_child);
    }

    let bridge_without_trace_child = run_child("bridge-without-trace", Some("warn"));
    assert_marker_is_suppressed("bridge-without-trace", &bridge_without_trace_child);

    let bridge_off_child = run_child("bridge-off", Some("off"));
    assert_marker_is_suppressed("bridge-off", &bridge_off_child);

    let explicit_child = run_child("explicit-bridge", Some("trace"));
    let explicit = assert_child_passed("explicit-bridge", &explicit_child);
    assert!(
        explicit.contains(TOKENIZERS_TRACE_MARKER),
        "explicit LogTracer plus TRACE must expose real tokenizers logs:\n{explicit}"
    );
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
        other => panic!("unknown fresh-process logging child case {other:?}"),
    }
}
