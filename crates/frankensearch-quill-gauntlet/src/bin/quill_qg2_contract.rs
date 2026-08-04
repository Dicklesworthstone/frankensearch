#![forbid(unsafe_code)]

use std::ffi::OsStr;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::process::ExitCode;

use frankensearch_quill_gauntlet::{QG2_CONTRACT_REPORT_SCHEMA_VERSION, validate_qg2_contract};
use serde::Serialize;

const USAGE: &str = "usage: quill-qg2-contract --repo-root <path>";

#[derive(Debug, Serialize)]
struct InvocationFailure {
    schema_version: &'static str,
    status: &'static str,
    code: &'static str,
    expected: &'static str,
    observed: String,
    retry: &'static str,
}

enum Invocation {
    Help,
    Validate(PathBuf),
}

fn main() -> ExitCode {
    match parse_invocation(std::env::args_os().skip(1)) {
        Ok(Invocation::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Invocation::Validate(repo_root)) => {
            let report = validate_qg2_contract(&repo_root);
            let passed = report.is_pass();
            match write_json(&report) {
                Ok(()) if passed => ExitCode::SUCCESS,
                Ok(()) => ExitCode::from(1),
                Err(error) => {
                    eprintln!("failed to serialize QG-2 contract report: {error}");
                    ExitCode::from(70)
                }
            }
        }
        Err(observed) => {
            let failure = InvocationFailure {
                schema_version: QG2_CONTRACT_REPORT_SCHEMA_VERSION,
                status: "invocation_error",
                code: "qg2.cli.invalid_invocation",
                expected: USAGE,
                observed,
                retry: "Invoke the validator with exactly one --repo-root <path> pair.",
            };
            if let Err(error) = write_json(&failure) {
                eprintln!("failed to serialize QG-2 invocation error: {error}");
                return ExitCode::from(70);
            }
            ExitCode::from(64)
        }
    }
}

fn parse_invocation(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> Result<Invocation, String> {
    let Some(first) = arguments.next() else {
        return Err("missing --repo-root".to_owned());
    };
    if first == OsStr::new("--help") || first == OsStr::new("-h") {
        if let Some(extra) = arguments.next() {
            return Err(format!(
                "unexpected argument after --help: {}",
                extra.to_string_lossy()
            ));
        }
        return Ok(Invocation::Help);
    }
    if first != OsStr::new("--repo-root") {
        return Err(format!("unexpected argument: {}", first.to_string_lossy()));
    }
    let repo_root = arguments
        .next()
        .ok_or_else(|| "missing value for --repo-root".to_owned())?;
    if let Some(extra) = arguments.next() {
        return Err(format!(
            "unexpected trailing argument: {}",
            extra.to_string_lossy()
        ));
    }
    Ok(Invocation::Validate(PathBuf::from(repo_root)))
}

fn write_json(value: &impl Serialize) -> io::Result<()> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invocation_requires_exact_repo_root_pair() {
        assert!(matches!(
            parse_invocation(
                [
                    std::ffi::OsString::from("--repo-root"),
                    PathBuf::from(".").into_os_string()
                ]
                .into_iter()
            ),
            Ok(Invocation::Validate(_))
        ));
        assert!(parse_invocation(std::iter::empty()).is_err());
        assert!(
            parse_invocation(
                [
                    std::ffi::OsString::from("--repo-root"),
                    PathBuf::from(".").into_os_string(),
                    std::ffi::OsString::from("extra")
                ]
                .into_iter()
            )
            .is_err()
        );
    }
}
