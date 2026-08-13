use std::env;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const CANDIDATE_REV: &str = "18a5a1a9982138822c34d4c3fb29f4c883715069";
const BASELINE_VERSION: &str = "0.3.4";
const WORKSPACE_SOURCE_RECEIPT_SCHEMA: &str = "frankensearch.hnsw-workspace-source-receipt.v1";
const WORKSPACE_SOURCE_FIXED_INPUTS: &[&str] = &[
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    "crates/frankensearch-core/Cargo.toml",
    "crates/frankensearch-index/Cargo.toml",
    "crates/frankensearch-index/build.rs",
    "crates/frankensearch-index/src/bin/hnsw_patch_ab.admission.v1.json",
];

fn main() {
    for name in [
        "CARGO_HOME",
        "CARGO_PROFILE_RELEASE_PERF_LTO",
        "CARGO_PROFILE_RELEASE_PERF_CODEGEN_UNITS",
        "CARGO_PROFILE_RELEASE_PERF_OPT_LEVEL",
        "CARGO_ENCODED_RUSTFLAGS",
        "OPT_LEVEL",
        "PROFILE",
        "DEBUG",
        "RUSTC",
        "TARGET",
        "HOST",
        "CARGO_CFG_TARGET_FEATURE",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    emit("PROFILE_DIR", profile_directory());
    emit("PROFILE_FAMILY", required_env("PROFILE"));
    emit("OPT_LEVEL", required_env("OPT_LEVEL"));
    emit("DEBUG_INFO", required_env("DEBUG"));
    emit("HOST", required_env("HOST"));
    emit("TARGET", required_env("TARGET"));
    emit(
        "TARGET_FEATURES",
        env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default(),
    );
    emit(
        "LTO",
        env::var("CARGO_PROFILE_RELEASE_PERF_LTO").unwrap_or_else(|_| "<unset>".to_owned()),
    );
    emit(
        "CODEGEN_UNITS",
        env::var("CARGO_PROFILE_RELEASE_PERF_CODEGEN_UNITS")
            .unwrap_or_else(|_| "<unset>".to_owned()),
    );
    emit(
        "PROFILE_OPT_LEVEL",
        env::var("CARGO_PROFILE_RELEASE_PERF_OPT_LEVEL").unwrap_or_else(|_| "<unset>".to_owned()),
    );

    let rustc = required_env("RUSTC");
    let rustc_vv = Command::new(&rustc)
        .arg("-Vv")
        .output()
        .unwrap_or_else(|error| panic!("failed to execute {rustc:?} -Vv: {error}"));
    assert!(
        rustc_vv.status.success(),
        "{rustc:?} -Vv failed: {}",
        String::from_utf8_lossy(&rustc_vv.stderr)
    );
    emit("RUSTC_VV_SHA256", sha256_bytes(&rustc_vv.stdout));
    emit(
        "RUSTFLAGS_SHA256",
        sha256_bytes(
            env::var("CARGO_ENCODED_RUSTFLAGS")
                .unwrap_or_default()
                .as_bytes(),
        ),
    );

    if env::var_os("CARGO_FEATURE_HNSW_PATCH_AB").is_some() {
        let manifest_dir = PathBuf::from(required_env("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|error| {
                panic!(
                    "failed to resolve workspace root from {}: {error}",
                    manifest_dir.display()
                )
            });
        let workspace_receipt = workspace_source_receipt(&workspace);
        write_embedded_workspace_source_receipt(&workspace_receipt);
        let cargo_home = cargo_home();
        let candidate = find_candidate_checkout(&cargo_home);
        require_clean_tracked_checkout(&candidate);
        let baseline = find_registry_baseline(&cargo_home);
        emit("CANDIDATE_SOURCE_SHA256", hash_source_tree(&candidate));
        emit("BASELINE_SOURCE_SHA256", hash_source_tree(&baseline));
    } else {
        emit("CANDIDATE_SOURCE_SHA256", "<feature-disabled>");
        emit("BASELINE_SOURCE_SHA256", "<feature-disabled>");
    }
}

struct WorkspaceSourceInput {
    path: String,
    byte_len: u64,
    sha256: String,
}

struct WorkspaceSourceReceipt {
    aggregate_sha256: String,
    cargo_lock_sha256: String,
    inputs: Vec<WorkspaceSourceInput>,
}

fn workspace_source_receipt(workspace: &Path) -> WorkspaceSourceReceipt {
    let mut files: Vec<_> = WORKSPACE_SOURCE_FIXED_INPUTS
        .iter()
        .map(|relative| workspace.join(relative))
        .collect();
    collect_files(&workspace.join("crates/frankensearch-core/src"), &mut files);
    collect_files(
        &workspace.join("crates/frankensearch-index/src"),
        &mut files,
    );
    files.sort();
    files.dedup();

    let mut aggregate = Sha256::new();
    aggregate.update(WORKSPACE_SOURCE_RECEIPT_SCHEMA.as_bytes());
    let mut inputs = Vec::with_capacity(files.len());
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let metadata = fs::symlink_metadata(&path)
            .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", path.display()));
        assert!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "workspace build input is not a regular non-symlink file: {}",
            path.display()
        );
        let relative = path
            .strip_prefix(workspace)
            .unwrap_or_else(|_| panic!("workspace build input escaped {}", workspace.display()))
            .to_str()
            .unwrap_or_else(|| {
                panic!(
                    "workspace build input path is not valid UTF-8: {}",
                    path.display()
                )
            })
            .to_owned();
        assert!(
            relative
                .bytes()
                .all(|byte| !matches!(byte, b'\r' | b'\n' | b'\t')),
            "workspace build input path contains a control separator: {relative:?}"
        );
        let bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let byte_len =
            u64::try_from(bytes.len()).expect("workspace build input length must fit u64");
        let sha256 = sha256_bytes(&bytes);
        aggregate.update(
            u64::try_from(relative.len())
                .expect("workspace build input path length must fit u64")
                .to_le_bytes(),
        );
        aggregate.update(relative.as_bytes());
        aggregate.update(byte_len.to_le_bytes());
        aggregate.update(sha256.as_bytes());
        inputs.push(WorkspaceSourceInput {
            path: relative,
            byte_len,
            sha256,
        });
    }

    let cargo_lock_sha256 = inputs
        .iter()
        .find(|input| input.path == "Cargo.lock")
        .map(|input| input.sha256.clone())
        .expect("workspace source receipt must include Cargo.lock");
    WorkspaceSourceReceipt {
        aggregate_sha256: hex_bytes(&aggregate.finalize()),
        cargo_lock_sha256,
        inputs,
    }
}

fn write_embedded_workspace_source_receipt(receipt: &WorkspaceSourceReceipt) {
    let mut generated = String::new();
    writeln!(
        generated,
        "const EMBEDDED_WORKSPACE_SOURCE_RECEIPT_SCHEMA: &str = \
         {WORKSPACE_SOURCE_RECEIPT_SCHEMA:?};"
    )
    .expect("writing generated receipt cannot fail");
    writeln!(
        generated,
        "const EMBEDDED_WORKSPACE_SOURCE_AGGREGATE_SHA256: &str = {:?};",
        receipt.aggregate_sha256
    )
    .expect("writing generated receipt cannot fail");
    writeln!(
        generated,
        "const EMBEDDED_WORKSPACE_CARGO_LOCK_SHA256: &str = {:?};",
        receipt.cargo_lock_sha256
    )
    .expect("writing generated receipt cannot fail");
    writeln!(
        generated,
        "#[allow(clippy::unreadable_literal)]\n\
         const EMBEDDED_WORKSPACE_SOURCE_INPUTS: &[(&str, u64, &str)] = &["
    )
    .expect("writing generated receipt cannot fail");
    for input in &receipt.inputs {
        writeln!(
            generated,
            "    ({:?}, {}, {:?}),",
            input.path, input.byte_len, input.sha256
        )
        .expect("writing generated receipt cannot fail");
    }
    writeln!(generated, "];").expect("writing generated receipt cannot fail");

    let output = PathBuf::from(required_env("OUT_DIR")).join("hnsw_workspace_source_receipt.rs");
    fs::write(&output, generated).unwrap_or_else(|error| {
        panic!(
            "failed to write embedded workspace source receipt {}: {error}",
            output.display()
        )
    });
}

fn emit(name: &str, value: impl AsRef<str>) {
    let value = value.as_ref();
    assert!(
        !value.contains('\r') && !value.contains('\n'),
        "build attestation {name} contains a newline"
    );
    println!("cargo:rustc-env=FRANKENSEARCH_HNSW_{name}={value}");
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("Cargo did not provide required build input {name}"))
}

fn profile_directory() -> String {
    let out_dir = PathBuf::from(required_env("OUT_DIR"));
    let components: Vec<_> = out_dir.components().collect();
    let build = components
        .iter()
        .position(|component| component.as_os_str() == OsStr::new("build"))
        .unwrap_or_else(|| panic!("OUT_DIR has no build component: {}", out_dir.display()));
    assert!(build > 0, "OUT_DIR has no profile component");
    components[build - 1]
        .as_os_str()
        .to_string_lossy()
        .into_owned()
}

fn cargo_home() -> PathBuf {
    env::var_os("CARGO_HOME").map_or_else(
        || {
            PathBuf::from(
                env::var_os("HOME")
                    .unwrap_or_else(|| panic!("neither CARGO_HOME nor HOME is available")),
            )
            .join(".cargo")
        },
        PathBuf::from,
    )
}

fn find_candidate_checkout(cargo_home: &Path) -> PathBuf {
    let checkouts = cargo_home.join("git/checkouts");
    let mut matches = Vec::new();
    for repository in read_dirs(&checkouts) {
        for checkout in read_dirs(&repository) {
            if checkout.join("Cargo.toml").is_file()
                && git_text(&checkout, &["rev-parse", "HEAD"]).as_deref() == Some(CANDIDATE_REV)
            {
                matches.push(checkout);
            }
        }
    }
    unique_match(matches, "candidate hnsw_rs git checkout")
}

fn find_registry_baseline(cargo_home: &Path) -> PathBuf {
    let registry_src = cargo_home.join("registry/src");
    let mut matches = Vec::new();
    for registry in read_dirs(&registry_src) {
        let candidate = registry.join(format!("hnsw_rs-{BASELINE_VERSION}"));
        if candidate.join("Cargo.toml").is_file() {
            matches.push(candidate);
        }
    }
    unique_match(matches, "published hnsw_rs 0.3.4 source")
}

fn unique_match(mut matches: Vec<PathBuf>, description: &str) -> PathBuf {
    matches.sort();
    matches.dedup();
    match matches.as_slice() {
        [path] => path.clone(),
        [] => panic!("could not locate {description} in Cargo's source cache"),
        _ => panic!("found multiple {description} candidates: {matches:?}"),
    }
}

fn read_dirs(path: &Path) -> Vec<PathBuf> {
    fs::read_dir(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to inspect an entry under {}: {error}",
                        path.display()
                    )
                })
                .path()
        })
        .filter(|entry| entry.is_dir())
        .collect()
}

fn require_clean_tracked_checkout(checkout: &Path) {
    let head = git_text(checkout, &["rev-parse", "HEAD"])
        .unwrap_or_else(|| panic!("candidate checkout has no readable HEAD"));
    assert_eq!(head, CANDIDATE_REV, "candidate checkout revision changed");
    for args in [
        ["diff", "--quiet", "HEAD", "--"].as_slice(),
        ["diff", "--cached", "--quiet", "HEAD", "--"].as_slice(),
    ] {
        let status = git_command(checkout)
            .args(args)
            .status()
            .unwrap_or_else(|error| panic!("failed to inspect candidate tracked dirt: {error}"));
        assert!(
            status.success(),
            "candidate checkout has tracked modifications: {}",
            checkout.display()
        );
    }
}

fn git_text(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = git_command(cwd).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn git_command(cwd: &Path) -> Command {
    let mut command = Command::new("git");
    command
        .arg("-c")
        .arg(format!("safe.directory={}", cwd.display()))
        .current_dir(cwd);
    command
}

fn hash_source_tree(root: &Path) -> String {
    let mut files = vec![root.join("Cargo.toml")];
    let build_rs = root.join("build.rs");
    if build_rs.is_file() {
        files.push(build_rs);
    }
    collect_files(&root.join("src"), &mut files);
    files.sort();
    files.dedup();
    assert!(
        files.iter().all(|path| path.is_file()),
        "source attestation includes a missing file"
    );

    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.hnsw-source-tree.v1");
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path
            .strip_prefix(root)
            .unwrap_or_else(|_| panic!("source file escaped {}", root.display()));
        let relative = relative.as_os_str().as_encoded_bytes();
        let mut file = fs::File::open(&path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", path.display()));
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        hasher.update(
            u64::try_from(relative.len())
                .expect("relative source path length must fit u64")
                .to_le_bytes(),
        );
        hasher.update(relative);
        hasher.update(
            u64::try_from(bytes.len())
                .expect("source file length must fit u64")
                .to_le_bytes(),
        );
        hasher.update(&bytes);
    }
    hex_bytes(&hasher.finalize())
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) {
    for entry in read_entries(path) {
        let metadata = fs::symlink_metadata(&entry)
            .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", entry.display()));
        assert!(
            !metadata.file_type().is_symlink(),
            "dependency source contains a symlink: {}",
            entry.display()
        );
        if metadata.is_dir() {
            collect_files(&entry, files);
        } else if metadata.is_file() {
            files.push(entry);
        }
    }
}

fn read_entries(path: &Path) -> Vec<PathBuf> {
    let mut entries: Vec<_> = fs::read_dir(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to inspect an entry under {}: {error}",
                        path.display()
                    )
                })
                .path()
        })
        .collect();
    entries.sort();
    entries
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}
