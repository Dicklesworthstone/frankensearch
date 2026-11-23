//! Bare relative paths must support the same durable lifecycle as absolute paths.

#![cfg(unix)]

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use frankensearch_index::{Quantization, VectorIndex, wal_path_for};

const CHILD_STAGE: &str = "FRANKENSEARCH_RELATIVE_PATH_DURABILITY_STAGE";

#[test]
fn bare_relative_paths_survive_wal_rewrite_compaction_and_restart() {
    if let Ok(stage) = std::env::var(CHILD_STAGE) {
        run_stage(&stage);
        return;
    }

    let directory = tempfile::tempdir().expect("private lifecycle directory");
    let executable = std::env::current_exe().expect("integration test executable");
    // Each stage starts in a new process, so it can only observe the previous
    // stage's durable bytes. No test changes the shared process's working dir.
    for stage in [
        "create",
        "append",
        "rewrite",
        "compact",
        "remove_wal",
        "verify",
    ] {
        let output = Command::new(&executable)
            .args([
                "--exact",
                "bare_relative_paths_survive_wal_rewrite_compaction_and_restart",
                "--nocapture",
            ])
            .env(CHILD_STAGE, stage)
            .current_dir(directory.path())
            .output()
            .expect("run fresh lifecycle process");
        assert!(
            output.status.success(),
            "relative-path stage {stage} failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

fn run_stage(stage: &str) {
    for (name, quantization) in [
        ("f16.fsvi", Quantization::F16),
        ("f32.fsvi", Quantization::F32),
    ] {
        let path = Path::new(name);
        if stage == "create" {
            let mut writer =
                VectorIndex::create_with_revision(path, "relative-path", "v1", 4, quantization)
                    .expect("create relative writer");
            writer
                .write_record("main", &[1.0, 0.0, 0.0, 0.0])
                .expect("write main vector");
            writer.finish().expect("durably publish relative index");
            continue;
        }

        let expected: &[&str] = match stage {
            "append" => &["main"],
            "rewrite" => &["main", "wal-first", "wal-second"],
            "compact" | "verify" => &["main", "wal-second"],
            "remove_wal" => &["main", "wal-second", "after-compact"],
            other => panic!("unknown lifecycle stage {other}"),
        };
        let reader = VectorIndex::open_read_only(path).expect("reopen previous stage");
        assert_eq!(
            reader.live_doc_ids().expect("read durable document ids"),
            expected
                .iter()
                .map(|id| (*id).to_owned())
                .collect::<HashSet<_>>(),
            "{name}, before {stage}",
        );
        if matches!(stage, "compact" | "remove_wal" | "verify") {
            let hits = reader
                .search_top_k(&[0.0, 0.0, 0.0, 1.0], 1, None)
                .expect("search replacement vector after restart");
            assert_eq!(hits[0].doc_id, "wal-second");
            assert_eq!(hits[0].score.to_bits(), 1.0_f32.to_bits());
        }
        drop(reader);

        if stage == "verify" {
            assert!(!wal_path_for(path).exists(), "removed WAL stays absent");
            continue;
        }
        let mut index = VectorIndex::open_writer(path).expect("open relative writer");
        match stage {
            "append" => index
                .append_batch(&[
                    ("wal-first".to_owned(), vec![0.0, 1.0, 0.0, 0.0]),
                    ("wal-second".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
                ])
                .expect("durably create relative WAL"),
            "rewrite" => {
                assert!(
                    index
                        .soft_delete("wal-first")
                        .expect("rewrite relative WAL")
                );
                index
                    .append("wal-second", &[0.0, 0.0, 0.0, 1.0])
                    .expect("durably replace existing WAL vector");
            }
            "compact" => {
                let stats = index.compact().expect("durably compact relative index");
                assert_eq!(stats.wal_records, 1);
                assert!(!wal_path_for(path).exists(), "compaction retires WAL");
                index
                    .append("after-compact", &[0.0, 1.0, 0.0, 0.0])
                    .expect("durably recreate relative WAL");
            }
            "remove_wal" => {
                assert!(
                    index
                        .soft_delete("after-compact")
                        .expect("durably remove relative WAL")
                );
            }
            other => panic!("unexpected mutation stage {other}"),
        }
    }
}
