//! Persistent preparation handoff for the recovery executable.
//!
//! Files contain only a target, its original selection guard and optional repair
//! origin, never query text, previews, model credentials or a final outcome.
//! They are trusted operator records, not signatures or persistent reader pins.
//! A later process must re-admit the target and producers through the runtime.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

use frankensearch_storage::ContentHasher;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{Cx, RecoveryState, RepairSource, Report, SearchError, SearchResult, invalid};

const FORMAT: &str = "fsfs.recovery.preparation";
const MAX_BYTES: u64 = 8192;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SavedPreparation {
    format: String,
    schema_version: u16,
    // The native canonical path spelling is deliberately platform-bound, like
    // the existing restore guard. Relocation requires a new explicit decision.
    store_path_sha256: String,
    pub generation_id: String,
    pub manifest_sha256: String,
    pub expected_selection_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repaired_from: Option<RepairSource>,
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_generation(value: &str) -> bool {
    let mut parts = value.split('-');
    parts.next() == Some("g")
        && parts.next().is_some_and(|part| is_hex(part, 32))
        && parts.next().is_some_and(|part| is_hex(part, 8))
        && parts.next().is_some_and(|part| is_hex(part, 16))
        && parts.next().is_none()
}

fn store_path_sha256(canonical: &Path) -> String {
    let mut hash = Sha256::new();
    hash.update(b"fsfs.recovery.preparation.store-path.v1\0");
    hash.update(canonical.as_os_str().as_encoded_bytes());
    let sha256: [u8; 32] = hash.finalize().into();
    ContentHasher::to_hex(&sha256)
}

impl SavedPreparation {
    fn from_report(root: &Path, report: &Report) -> SearchResult<Self> {
        if report.state != RecoveryState::Prepared
            || report.selection_write_performed
            || report.durability_confirmed
            || !report.producer_admission_checked
            || report.sync_error_kind.is_some()
        {
            return Err(invalid(
                "only an admitted preparation can be saved; a publication outcome is not a resume receipt",
            ));
        }
        let receipt = Self {
            format: FORMAT.to_owned(),
            schema_version: 1,
            store_path_sha256: store_path_sha256(root),
            generation_id: report.generation_id.clone(),
            manifest_sha256: report.manifest_sha256.clone(),
            expected_selection_sha256: report.expected_selection_sha256.clone().ok_or_else(
                || invalid("a saved preparation requires its original selection guard"),
            )?,
            repaired_from: report.repaired_from.clone(),
        };
        receipt.validate()?;
        Ok(receipt)
    }

    fn validate(&self) -> SearchResult<()> {
        if self.format != FORMAT || self.schema_version != 1 {
            return Err(invalid("unsupported preparation receipt format or version"));
        }
        if !is_generation(&self.generation_id)
            || !is_hex(&self.manifest_sha256, 64)
            || !is_hex(&self.expected_selection_sha256, 64)
            || !is_hex(&self.store_path_sha256, 64)
        {
            return Err(invalid(
                "invalid preparation target, manifest, store binding or selection guard",
            ));
        }
        if let Some(origin) = &self.repaired_from
            && (!is_generation(&origin.generation_id) || !is_hex(&origin.manifest_sha256, 64))
        {
            return Err(invalid("invalid preparation repair origin"));
        }
        Ok(())
    }
}

fn decode(reader: impl Read) -> SearchResult<SavedPreparation> {
    let mut bytes = Vec::new();
    reader.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(invalid("preparation receipt exceeds the 8192-byte limit"));
    }
    // Deserialize the original bytes, not a Value: duplicate fields and
    // trailing documents must not turn an ambiguous receipt into authority.
    let receipt: SavedPreparation = serde_json::from_slice(&bytes).map_err(|error| {
        invalid(format!(
            "invalid preparation receipt ({:?} at line {}, column {})",
            error.classify(),
            error.line(),
            error.column(),
        ))
    })?;
    receipt.validate()?;
    Ok(receipt)
}

pub fn read(path: &Path, root: &Path) -> SearchResult<SavedPreparation> {
    #[cfg(unix)]
    let file = {
        use rustix::fs::{Mode, OFlags};
        // A named pipe must not hang argument admission waiting for a writer.
        // NOFOLLOW is set on the actual open, not only a prior pathname check.
        let descriptor = rustix::fs::open(
            path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(io::Error::from)?;
        File::from(descriptor)
    };
    #[cfg(not(unix))]
    let file = {
        if !fs::symlink_metadata(path)?.is_file() {
            return Err(invalid(
                "preparation receipt must be a regular non-symlink file",
            ));
        }
        File::open(path)?
    };
    if !file.metadata()?.is_file() {
        return Err(invalid("preparation receipt must be a regular file"));
    }
    let receipt = decode(file)?;
    if receipt.store_path_sha256 != store_path_sha256(&fs::canonicalize(root)?) {
        return Err(invalid(
            "preparation receipt belongs to a different store; it cannot be relocated or rebased",
        ));
    }
    Ok(receipt)
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|_| SearchError::Cancelled {
        phase: "recovery.preparation_file".to_owned(),
        reason: "preparation receipt I/O cancelled before selection publication".to_owned(),
    })
}

/// Save before restoration, never after a visible publication. A failure here
/// prevents selection, but may leave a partial file which decode will reject.
/// Do not delete, truncate, reuse or overwrite that path on a retry.
/// Parent directories must already exist and remain trusted during the call.
pub fn write(cx: &Cx, path: &Path, root: &Path, report: &Report) -> SearchResult<()> {
    checkpoint(cx)?;
    let root = fs::canonicalize(root)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent)?;
    if parent.starts_with(&root) {
        return Err(invalid(
            "preparation receipts must be saved outside the index store; sealed bundles and store control files cannot be modified",
        ));
    }
    let name = path
        .file_name()
        .ok_or_else(|| invalid("preparation receipt requires a file name"))?;
    let destination = parent.join(name);
    let receipt = SavedPreparation::from_report(&root, report)?;
    let mut bytes =
        serde_json::to_vec(&receipt).map_err(|_| invalid("cannot encode preparation receipt"))?;
    bytes.push(b'\n');
    if bytes.len() as u64 > MAX_BYTES {
        return Err(invalid("preparation receipt exceeds the 8192-byte limit"));
    }
    checkpoint(cx)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(destination)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    // Non-Unix std has no portable directory synchronization primitive. There
    // the file is flushed, without a directory-durability claim.
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    checkpoint(cx)
}

#[cfg(test)]
mod tests {
    use super::super::{Options, RecoveryAction};
    use super::*;
    use std::ffi::OsString;

    fn report() -> Report {
        Report {
            schema_version: 1,
            state: RecoveryState::Prepared,
            generation_id: format!("g-{}-{}-{}", "a".repeat(32), "b".repeat(8), "c".repeat(16)),
            manifest_sha256: "d".repeat(64),
            selection_write_performed: false,
            durability_confirmed: false,
            producer_admission_checked: true,
            repaired_from: Some(RepairSource {
                generation_id: format!("g-{}-{}-{}", "1".repeat(32), "2".repeat(8), "3".repeat(16)),
                manifest_sha256: "e".repeat(64),
            }),
            expected_selection_sha256: Some("f".repeat(64)),
            sync_error_kind: None,
            preview: Vec::new(),
        }
    }

    fn arguments(root: &Path, receipt: &Path, action: &str) -> Vec<OsString> {
        vec![
            "--index-dir".into(),
            root.as_os_str().to_owned(),
            "--resume-file".into(),
            receipt.as_os_str().to_owned(),
            action.into(),
        ]
    }

    #[test]
    fn file_roundtrip_preserves_target_guard_and_origin_without_overwriting() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().join("store");
            fs::create_dir(&root).unwrap();
            let path = directory.path().join("preparation.json");
            let expected = report();
            write(&cx, &path, &root, &expected).unwrap();
            let bytes = fs::read(&path).unwrap();
            let loaded = read(&path, &root).unwrap();
            assert_eq!(loaded.generation_id, expected.generation_id);
            assert_eq!(loaded.manifest_sha256, expected.manifest_sha256);
            assert_eq!(
                Some(loaded.expected_selection_sha256),
                expected.expected_selection_sha256
            );
            assert_eq!(
                loaded.repaired_from.unwrap().generation_id,
                expected.repaired_from.as_ref().unwrap().generation_id
            );
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            for key in [
                "query",
                "preview",
                "config",
                "state",
                "selection_write_performed",
                "durability_confirmed",
            ] {
                assert!(
                    value.get(key).is_none(),
                    "{key} must not be persisted as a preparation fact"
                );
            }
            assert!(
                matches!(write(&cx, &path, &root, &expected), Err(SearchError::Io(error))
                if error.kind() == io::ErrorKind::AlreadyExists)
            );
            assert_eq!(fs::read(&path).unwrap(), bytes);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(
                    fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
        });
    }

    #[test]
    fn receipt_rejects_schema_identity_duplicates_trailing_data_and_private_errors() {
        let root = Path::new("/fixture");
        let receipt = SavedPreparation::from_report(root, &report()).unwrap();
        let valid = serde_json::to_vec(&receipt).unwrap();
        assert!(decode(valid.as_slice()).is_ok());
        for (field, value) in [
            ("format", serde_json::json!("other")),
            ("schema_version", serde_json::json!(2)),
            ("generation_id", serde_json::json!("../outside")),
            ("manifest_sha256", serde_json::json!("A".repeat(64))),
            (
                "expected_selection_sha256",
                serde_json::json!("f".repeat(63)),
            ),
            ("store_path_sha256", serde_json::json!("g".repeat(64))),
            (
                "schema_version",
                serde_json::json!("private-receipt-canary"),
            ),
            ("unexpected", serde_json::json!(true)),
        ] {
            let mut value_json = serde_json::to_value(&receipt).unwrap();
            value_json[field] = value;
            let encoded = serde_json::to_vec(&value_json).unwrap();
            let error = decode(encoded.as_slice()).unwrap_err();
            assert!(!error.to_string().contains("private-receipt-canary"));
        }
        let mut duplicate = valid.clone();
        duplicate.pop();
        duplicate.extend_from_slice(b",\"schema_version\":1}");
        assert!(decode(duplicate.as_slice()).is_err());
        let mut trailing = valid.clone();
        trailing.extend_from_slice(b" {}");
        assert!(decode(trailing.as_slice()).is_err());
        for end in [0, 1, valid.len() / 2, valid.len() - 1] {
            assert!(decode(&valid[..end]).is_err());
        }
        let mut exact = valid;
        exact.resize(usize::try_from(MAX_BYTES).unwrap(), b' ');
        assert!(decode(exact.as_slice()).is_ok());
        exact.push(b' ');
        assert!(decode(exact.as_slice()).is_err());
        assert!(
            decode(io::repeat(b' ')).is_err(),
            "an unbounded reader must stop at the byte ceiling"
        );
    }

    #[test]
    fn saving_rejects_nonprepared_outcomes_and_cancellation_without_creating_a_file() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().join("store");
            fs::create_dir(&root).unwrap();
            let path = directory.path().join("preparation.json");
            for state in [
                RecoveryState::Inspected,
                RecoveryState::RestoredDurable,
                RecoveryState::RestoredDurabilityUncertain,
            ] {
                let mut facts = report();
                facts.state = state;
                assert!(write(&cx, &path, &root, &facts).is_err());
                assert!(!path.exists());
            }
            let mut facts = report();
            facts.expected_selection_sha256 = None;
            assert!(write(&cx, &path, &root, &facts).is_err());
            assert!(!path.exists());
            cx.set_cancel_requested(true);
            assert!(matches!(
                write(&cx, &path, &root, &report()),
                Err(SearchError::Cancelled { .. })
            ));
            assert!(!path.exists());
            cx.set_cancel_requested(false);
        });
    }

    #[test]
    fn saved_files_cannot_mutate_the_store_or_move_between_store_roots() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().join("store");
            let other = directory.path().join("other-store");
            fs::create_dir(&root).unwrap();
            fs::create_dir(&other).unwrap();
            let path = root.join("preparation.json");
            assert!(write(&cx, &path, &root, &report()).is_err());
            assert!(!path.exists());
            let path = directory.path().join("preparation.json");
            write(&cx, &path, &root, &report()).unwrap();
            assert!(read(&path, &other).is_err());
            assert!(read(&path, &root.join(".")).is_ok());
            assert!(read(&root, &root).is_err());
        });
    }

    #[cfg(unix)]
    #[test]
    fn receipt_io_rejects_symlinks_and_pipes_without_following_or_waiting() {
        use std::os::unix::fs::symlink;
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().join("store");
            fs::create_dir(&root).unwrap();
            let protected = directory.path().join("protected");
            fs::write(&protected, b"private original").unwrap();
            let link = directory.path().join("link");
            symlink(&protected, &link).unwrap();
            assert!(write(&cx, &link, &root, &report()).is_err());
            assert!(read(&link, &root).is_err());
            assert_eq!(fs::read(&protected).unwrap(), b"private original");
            let alias = directory.path().join("store-alias");
            symlink(&root, &alias).unwrap();
            assert!(write(&cx, &alias.join("receipt"), &root, &report()).is_err());
            assert!(!root.join("receipt").exists());
            // rustix has no mkfifoat on Apple targets; the other FIFO tests are
            // Linux-only too.
            #[cfg(target_os = "linux")]
            {
                let fifo = directory.path().join("pipe");
                rustix::fs::mkfifoat(
                    rustix::fs::CWD,
                    &fifo,
                    rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
                )
                .unwrap();
                assert!(read(&fifo, &root).is_err());
            }
        });
    }

    #[test]
    fn cli_file_resume_requires_explicit_action_and_preserves_the_original_receipt() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().join("store");
            fs::create_dir(&root).unwrap();
            let path = directory.path().join("preparation.json");
            write(&cx, &path, &root, &report()).unwrap();
            let resumed = Options::parse(arguments(&root, &path, "--apply")).unwrap();
            assert_eq!(resumed.action, RecoveryAction::Apply);
            assert_eq!(resumed.generation, report().generation_id);
            assert_eq!(resumed.digest, report().manifest_sha256);
            assert_eq!(
                resumed.expected_selection_sha256,
                report().expected_selection_sha256
            );
            assert!(!resumed.repair_vectors);
            assert_eq!(
                resumed.resume_origin.unwrap().generation_id,
                report().repaired_from.unwrap().generation_id
            );
            let confirmation =
                Options::parse(arguments(&root, &path, "--confirm-durability")).unwrap();
            assert_eq!(confirmation.action, RecoveryAction::ConfirmDurability);
            assert!(confirmation.expected_selection_sha256.is_none());
            assert_eq!(confirmation.generation, report().generation_id);
            for flags in [
                vec![],
                vec!["--prepare"],
                vec!["--apply", "--repair-vectors"],
                vec!["--apply", "--max-copy-bytes", "1"],
                vec!["--apply", "--generation", "manual"],
                vec!["--apply", "--manifest-sha256", "manual"],
                vec!["--apply", "--expected-selection-sha256", "manual"],
                vec!["--confirm-durability", "--query", "query"],
            ] {
                let mut input = arguments(&root, &path, "--apply");
                input.pop();
                input.extend(flags.into_iter().map(OsString::from));
                assert!(Options::parse(input).is_err());
            }
        });
    }

    #[cfg(unix)]
    #[test]
    fn saved_resume_rechecks_the_store_guard_and_confirmation_never_republishes() {
        use super::super::{CompleteGenerationStore, FsfsRuntime, GenerationPublication, execute};
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().join("store");
            let store = CompleteGenerationStore::create(&cx, &root).unwrap();
            let publish = || {
                let build = store.begin(&cx).unwrap();
                fs::write(
                    build.path().join("payload"),
                    b"store fixture, not an engine",
                )
                .unwrap();
                let GenerationPublication::Durable(generation) =
                    build.publish(&cx, |_, _| Ok(())).unwrap()
                else {
                    panic!("fixture must publish durably");
                };
                generation
            };
            let target = publish();
            let plan = store.prepare_restore(&cx, &target).unwrap();
            let facts = Report::inspected(&target).prepared(plan.expected_selection_sha256());
            let path = directory.path().join("preparation.json");
            write(&cx, &path, &root, &facts).unwrap();
            drop(plan);
            let options = Options::parse(arguments(&root, &path, "--confirm-durability")).unwrap();
            let confirmed = execute(&cx, None, &options).await.unwrap();
            assert_eq!(confirmed.state, RecoveryState::DurabilityConfirmed);
            assert!(!confirmed.selection_write_performed && !confirmed.producer_admission_checked);
            let winner = publish();
            assert!(execute(&cx, None, &options).await.is_err());
            let mut config = frankensearch_fsfs::FsfsConfig::default();
            "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
            config.indexing.offline = true;
            let runtime = FsfsRuntime::new(config);
            let options = Options::parse(arguments(&root, &path, "--apply")).unwrap();
            let error = execute(&cx, Some(&runtime), &options).await.unwrap_err();
            assert!(
                matches!(error, SearchError::InvalidConfig { ref reason, .. } if reason.contains("cannot rebase"))
            );
            assert_eq!(store.active(&cx).unwrap(), Some(winner));
        });
    }
}
