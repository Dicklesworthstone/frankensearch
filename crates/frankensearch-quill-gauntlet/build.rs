#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const PRODUCER_CONTRACT_VERSION: &str = "frankensearch.quill-local-perf-producer.v4";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("Cargo always defines CARGO_MANIFEST_DIR"),
    );
    let repository = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("gauntlet crate must remain under <repository>/crates")
        .to_path_buf();
    let cargo_lock = repository.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", cargo_lock.display());

    let cargo_lock_sha256 = fs::read(&cargo_lock)
        .map(|bytes| sha256_hex(&bytes))
        .unwrap_or_else(|_| "unavailable".to_owned());
    let repository_identity_is_exact = git_output(&repository, &["rev-parse", "--show-toplevel"])
        .and_then(|path| fs::canonicalize(path).ok())
        .zip(fs::canonicalize(&repository).ok())
        .is_some_and(|(git_root, expected_root)| git_root == expected_root);
    let git_revision = repository_identity_is_exact
        .then(|| git_output(&repository, &["rev-parse", "HEAD"]))
        .flatten()
        .unwrap_or_else(|| "unavailable".to_owned());
    let git_dirty = !repository_identity_is_exact
        || git_output(
            &repository,
            &["status", "--porcelain=v1", "--untracked-files=all"],
        )
        .is_none_or(|status| !status.is_empty())
        || git_index_hides_worktree_changes(&repository);

    register_repository_inputs(&repository);
    register_git_identity_inputs(&repository);
    println!("cargo:rustc-env=QUILL_PERF_PRODUCER_CONTRACT_VERSION={PRODUCER_CONTRACT_VERSION}");
    println!("cargo:rustc-env=QUILL_PERF_PRODUCER_GIT_REVISION={git_revision}");
    println!("cargo:rustc-env=QUILL_PERF_PRODUCER_GIT_DIRTY={git_dirty}");
    println!("cargo:rustc-env=QUILL_PERF_PRODUCER_CARGO_LOCK_SHA256={cargo_lock_sha256}");
}

fn register_repository_inputs(repository: &Path) {
    let Ok(output) = git_command(repository, &["ls-files", "-z"]).output() else {
        return;
    };
    if !output.status.success() {
        return;
    }
    for path in output.stdout.split(|byte| *byte == 0) {
        if path.is_empty() {
            continue;
        }
        let path = std::str::from_utf8(path)
            .expect("tracked repository paths must be UTF-8 for exact Cargo invalidation");
        assert!(
            !path.bytes().any(|byte| byte.is_ascii_control()),
            "tracked repository paths must not inject Cargo line-protocol controls"
        );
        println!("cargo:rerun-if-changed={}", repository.join(path).display());
    }
}

fn register_git_identity_inputs(repository: &Path) {
    let Some(git_dir) = git_output(repository, &["rev-parse", "--absolute-git-dir"]) else {
        return;
    };
    let git_dir = PathBuf::from(git_dir);
    for path in [
        git_dir.join("HEAD"),
        git_dir.join("index"),
        git_dir.join("packed-refs"),
    ] {
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    let Some(symbolic_ref) = git_output(repository, &["symbolic-ref", "-q", "HEAD"]) else {
        return;
    };
    let Some(common_dir) = git_output(repository, &["rev-parse", "--git-common-dir"]) else {
        return;
    };
    let common_dir = absolute_git_path(repository, &common_dir);
    let packed_refs = common_dir.join("packed-refs");
    if packed_refs.exists() {
        println!("cargo:rerun-if-changed={}", packed_refs.display());
    }
    let ref_path = common_dir.join(symbolic_ref);
    if ref_path.exists() {
        println!("cargo:rerun-if-changed={}", ref_path.display());
    }
}

fn absolute_git_path(repository: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        repository.join(path)
    }
}

fn git_output(repository: &Path, args: &[&str]) -> Option<String> {
    let output = git_command(repository, args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_owned())
}

fn git_command(repository: &Path, args: &[&str]) -> Command {
    let mut command = Command::new("git");
    command.arg("-C").arg(repository).args(args);
    for (name, _) in env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"GIT_") {
            command.env_remove(name);
        }
    }
    command
}

fn git_index_hides_worktree_changes(repository: &Path) -> bool {
    let assume_unchanged = git_output(repository, &["ls-files", "-v"]).is_none_or(|output| {
        output
            .lines()
            .any(|line| line.as_bytes().first().is_some_and(u8::is_ascii_lowercase))
    });
    let skip_worktree = git_output(repository, &["ls-files", "-t"])
        .is_none_or(|output| output.lines().any(|line| line.starts_with("S ")));
    assume_unchanged || skip_worktree
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}
