//! Create-only handoff for an interrupted inactive source rebuild.
//!
//! Unlike a preparation receipt, this contains no sealed manifest or successful
//! preview. The runtime must re-admit the scope, normal writer ownership,
//! durable checkpoint rows, source hashes and actual producers on every resume.

use std::io::Read;
use std::path::Path;

use frankensearch_fsfs::runtime::RetainedRebuildCheckpoint;
use serde::{Deserialize, Serialize};

use super::{Cx, PreviewRequirement, SearchResult, invalid, preparation_file};

const FORMAT: &str = "fsfs.recovery.inactive-build";
const MAX_BYTES: u64 = 8192;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SavedBuild {
    format: String,
    schema_version: u16,
    pub checkpoint: RetainedRebuildCheckpoint,
    pub preview_policy_sha256: String,
}

impl SavedBuild {
    fn validate(&self) -> SearchResult<()> {
        if self.format != FORMAT
            || self.schema_version != 1
            || self.preview_policy_sha256.len() != 64
            || !self
                .preview_policy_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid(
                "invalid inactive-build receipt format or preview binding",
            ));
        }
        Ok(())
    }
}

pub fn read(path: &Path) -> SearchResult<SavedBuild> {
    let mut bytes = Vec::new();
    preparation_file::open_receipt(path)?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(invalid(
            "inactive-build receipt exceeds the 8192-byte limit",
        ));
    }
    let saved: SavedBuild = serde_json::from_slice(&bytes).map_err(|error| {
        invalid(format!(
            "invalid inactive-build receipt ({:?} at line {}, column {})",
            error.classify(),
            error.line(),
            error.column(),
        ))
    })?;
    saved.validate()?;
    Ok(saved)
}

pub fn write(
    cx: &Cx,
    path: &Path,
    root: &Path,
    checkpoint: &RetainedRebuildCheckpoint,
    preview: &PreviewRequirement,
) -> SearchResult<()> {
    let PreviewRequirement::Query {
        policy_sha256,
        searchable: true,
    } = preview
    else {
        return Err(invalid(
            "a saved source build requires an exact useful-preview policy",
        ));
    };
    let saved = SavedBuild {
        format: FORMAT.to_owned(),
        schema_version: 1,
        checkpoint: checkpoint.clone(),
        preview_policy_sha256: policy_sha256.clone(),
    };
    saved.validate()?;
    let mut bytes =
        serde_json::to_vec(&saved).map_err(|_| invalid("cannot encode inactive-build receipt"))?;
    bytes.push(b'\n');
    // Same bounded create-new, no-follow/regular-file and private Unix mode
    // rules as prepared receipts. File and parent are synced before indexing.
    preparation_file::write_receipt(cx, path, root, &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn checkpoint() -> RetainedRebuildCheckpoint {
        serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "generation_id": format!("g-{}-{}-{}", "a".repeat(32), "b".repeat(8), "c".repeat(16)),
            "expected_selection_sha256": "d".repeat(64),
            "session": "fixture-session",
            "executable_sha256": "e".repeat(64),
            "indexing_contract_sha256": "f".repeat(64),
            "source_path_sha256": "0".repeat(64),
        }))
        .unwrap()
    }

    #[test]
    fn saved_build_is_create_only_bounded_and_separate_from_sealed_preparation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().join("store");
            fs::create_dir(&root).unwrap();
            let path = directory.path().join("build.json");
            let expected = checkpoint();
            let preview = PreviewRequirement::Query {
                policy_sha256: "1".repeat(64),
                searchable: true,
            };
            write(&cx, &path, &root, &expected, &preview).unwrap();
            let before = fs::read(&path).unwrap();
            let reopened = read(&path).unwrap();
            assert_eq!(
                serde_json::to_value(&reopened.checkpoint).unwrap(),
                serde_json::to_value(&expected).unwrap()
            );
            assert_eq!(reopened.preview_policy_sha256, "1".repeat(64));
            assert!(
                !String::from_utf8(before.clone())
                    .unwrap()
                    .contains("manifest_sha256")
            );
            assert!(preparation_file::read(&path, &root).is_err());
            assert!(write(&cx, &path, &root, &expected, &preview).is_err());
            assert_eq!(fs::read(&path).unwrap(), before);
            assert!(write(&cx, &root.join("receipt.json"), &root, &expected, &preview).is_err());
            assert!(!root.join("receipt.json").exists());
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(
                    fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
            cx.set_cancel_requested(true);
            let cancelled = directory.path().join("cancelled.json");
            assert!(write(&cx, &cancelled, &root, &expected, &preview).is_err());
            assert!(!cancelled.exists());
            cx.set_cancel_requested(false);
        });
    }

    #[test]
    fn saved_build_rejects_ambiguous_oversized_and_nonregular_input() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("build.json");
        let saved = SavedBuild {
            format: FORMAT.to_owned(),
            schema_version: 1,
            checkpoint: checkpoint(),
            preview_policy_sha256: "1".repeat(64),
        };
        let valid = serde_json::to_vec(&saved).unwrap();
        let mut unknown = serde_json::to_value(&saved).unwrap();
        unknown["manifest_sha256"] = serde_json::json!("2".repeat(64));
        let mut duplicate = valid.clone();
        duplicate.pop();
        duplicate.extend_from_slice(b",\"schema_version\":1}");
        let mut trailing = valid.clone();
        trailing.extend_from_slice(b"{}");
        for bytes in [
            serde_json::to_vec(&unknown).unwrap(),
            duplicate,
            trailing,
            vec![b' '; usize::try_from(MAX_BYTES + 1).unwrap()],
        ] {
            fs::write(&path, bytes).unwrap();
            assert!(read(&path).is_err());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            fs::write(&path, valid).unwrap();
            let link = directory.path().join("linked.json");
            symlink(&path, &link).unwrap();
            assert!(read(&link).is_err());
            #[cfg(target_os = "linux")]
            {
                let fifo = directory.path().join("fifo");
                rustix::fs::mkfifoat(
                    rustix::fs::CWD,
                    &fifo,
                    rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
                )
                .unwrap();
                assert!(read(&fifo).is_err());
            }
        }
        assert!(read(directory.path()).is_err());
    }
}
