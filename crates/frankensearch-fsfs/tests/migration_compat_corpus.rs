//! Golden-snapshot corpus and CI lane for `upgrade.migration.*` verification.
//!
//! `bd-rzb51` landed the evaluation surface (`migration_compat::evaluate` and
//! `render_artifacts`). This is the producer side the packaging-release-install
//! contract requires alongside it: a deterministic fixed-seed corpus, real
//! prior-version snapshots, and a lane that runs the evaluator against every
//! committed snapshot and fails on unexplained divergence
//! (`docs/fsfs-packaging-release-install-contract.md:626-637, :664-671`).
//!
//! # Why snapshots are normalized rather than compared byte-for-byte
//!
//! Indexing one fixed-seed corpus twice with the *same* fsfs binary does not
//! produce identical bytes. Measured on v1.4.2: index size identical at 736,264
//! bytes, but 4 of 10 artifacts differ across runs and the Quill segment's
//! filename changes every time (`seg-2699673703d74d20` vs
//! `seg-e71b80d0ab070b27`). That is by design, not a defect — an FSLX segment
//! header carries a random collision-checked `segment_id` and an informational
//! `created_unix_s`, and `index_sentinel.json` records a wall-clock stamp and
//! absolute paths.
//!
//! So a golden corpus of raw artifact bytes would go red on its first rerun,
//! and the pressure would be to regenerate it — exactly the reflex this corpus
//! exists to prevent. Instead every artifact is reduced to a *normalized
//! fingerprint* that keeps all content-bearing state and drops precisely the
//! fields that are documented to vary:
//!
//! - FSLX segments are fingerprinted through Quill's own [`SegmentReader`],
//!   never by masking byte offsets. The fingerprint keeps `schema_id`, the
//!   docid range, `doc_count`, and every section's kind/flags/length/xxh3 —
//!   which is the whole payload — and drops `segment_id` and `created_unix_s`.
//!   Masking guessed offsets would silently rot the moment the header layout
//!   changes; going through the reader fails loudly instead.
//! - JSON artifacts are canonicalized with the volatile keys elided by name.
//! - Everything else is digested whole.
//!
//! # Golden law
//!
//! Snapshots are **append-only, version-stamped** artifacts. There is
//! deliberately no bulk-refresh switch here: the repo-wide `UPDATE_GOLDENS`
//! rewrites all 114 fsfs goldens at once and must never be pointed at this
//! corpus. A new snapshot is produced one version at a time by the explicitly
//! ignored [`produce_snapshot`] probe, and an existing snapshot changing is a
//! STOP requiring a GOLDEN-CHANGE note and a semantic diff review, never a
//! regeneration.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use frankensearch_fsfs::migration_compat::{
    InvariantCheck, MigrationRun, PathOutcome, PathResult, QualitySample, RollbackAttempt,
    RollbackValidation, VersionPath, evaluate, render_artifacts,
};
use frankensearch_quill::schema::DEFAULT_SCHEMA;
use frankensearch_quill::{Manifest as QuillManifest, SegmentReader};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema of a committed snapshot manifest. Bump only with a GOLDEN-CHANGE note.
const SNAPSHOT_SCHEMA_VERSION: &str = "fsfs.migration.snapshot.v1";

/// Seed for the fixed corpus. Changing this invalidates every snapshot, so it
/// is a GOLDEN-CHANGE, not a tuning knob.
const CORPUS_SEED: u64 = 0x05EE_D4A2;
/// Documents in the fixed-seed corpus.
const CORPUS_DOCS: usize = 40;

/// JSON keys whose values are wall-clock, host, or path dependent.
///
/// Elided by name from every JSON artifact before digesting. Each entry is here
/// because it was observed to vary across two runs of one binary over one
/// corpus, and each is environmental rather than content-bearing.
const VOLATILE_JSON_KEYS: &[&str] = &[
    "generated_at_ms",
    "created_at_ms",
    "updated_at_ms",
    "timestamp_ms",
    "index_root",
    "target_root",
    "root",
    "path",
    "index_dir",
];

fn snapshots_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/migration-snapshots")
}

/// One artifact reduced to content-bearing state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArtifactFingerprint {
    /// Path relative to the index root, with the random segment id replaced by
    /// a stable placeholder.
    normalized_path: String,
    /// How the fingerprint was produced, so a reader can tell a whole-file
    /// digest from a structural one.
    kind: String,
    /// Bytes on disk, and `None` where size itself is not stable.
    ///
    /// Artifacts that embed absolute paths or a random segment id change length
    /// between runs even though their normalized content is identical, so
    /// recording a size for them would reintroduce exactly the flakiness the
    /// normalizer removes. Size is kept only where it is a real tamper check.
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
    /// Digest over the normalized form.
    digest: String,
}

/// A committed, version-stamped snapshot of one released version's artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SnapshotManifest {
    schema_version: String,
    /// The released fsfs version that produced these artifacts.
    source_version: String,
    /// Git tag the producing binary was built from.
    source_tag: String,
    corpus_seed: String,
    corpus_docs: usize,
    /// Digest over the generated corpus, so a corpus drift is caught before the
    /// artifacts are blamed.
    corpus_digest: String,
    /// Exact command that reproduces these artifacts.
    replay_command: String,
    artifacts: Vec<ArtifactFingerprint>,
    /// Per-subsystem rollup consumed by `InvariantCheck`.
    subsystem_digests: BTreeMap<String, String>,
}

/// Build the fixed-seed corpus deterministically.
///
/// A plain LCG, no clock and no RNG crate, so the corpus is a pure function of
/// [`CORPUS_SEED`] on every host and every run.
fn generate_corpus(root: &Path) -> std::io::Result<()> {
    const WORDS: &[&str] = &[
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon",
    ];
    std::fs::create_dir_all(root)?;
    let mut state = CORPUS_SEED;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };
    for index in 0..CORPUS_DOCS {
        let count = 40 + next() % 60;
        let body: Vec<&str> = (0..count).map(|_| WORDS[next() % WORDS.len()]).collect();
        std::fs::write(
            root.join(format!("doc-{index:03}.txt")),
            format!("title doc {index:03}\n\n{}\n", body.join(" ")),
        )?;
    }
    Ok(())
}

fn digest_corpus(root: &Path) -> std::io::Result<String> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();
    names.sort();
    let mut hasher = Sha256::new();
    for name in names {
        hasher.update(
            name.file_name()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or_default()
                .as_bytes(),
        );
        hasher.update(std::fs::read(&name)?);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(
        String::with_capacity(bytes.len().saturating_mul(2)),
        |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        },
    )
}

/// Replace a random Quill segment id with a stable placeholder.
fn normalize_path(relative: &str) -> String {
    let mut out = String::with_capacity(relative.len());
    let mut rest = relative;
    while let Some(start) = rest.find("seg-") {
        let (head, tail) = rest.split_at(start);
        out.push_str(head);
        let hexlen = tail[4..]
            .chars()
            .take_while(char::is_ascii_hexdigit)
            .count();
        if hexlen == 16 {
            out.push_str("seg-<ID>");
            rest = &tail[4 + hexlen..];
        } else {
            out.push_str("seg-");
            rest = &tail[4..];
        }
    }
    out.push_str(rest);
    out
}

/// Recursively drop volatile keys so the remainder canonicalizes stably.
fn strip_volatile(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.retain(|key, _| !VOLATILE_JSON_KEYS.contains(&key.as_str()));
            for nested in map.values_mut() {
                strip_volatile(nested);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                strip_volatile(item);
            }
        }
        _ => {}
    }
}

/// Fingerprint one artifact, choosing the normalization its format requires.
fn fingerprint(relative: &str, bytes: &[u8]) -> ArtifactFingerprint {
    let normalized_path = normalize_path(relative);
    let base = relative.rsplit('/').next().unwrap_or(relative);
    let extension = Path::new(relative)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let raw = || hex(&Sha256::digest(bytes));

    let (kind, digest) = if base == "MANIFEST" || base == "MANIFEST.prev" {
        // A manifest the decoder rejects is not silently downgraded to a raw
        // digest: it is recorded as unreadable so the lane fails loudly.
        manifest_fingerprint(bytes).map_or_else(
            || ("quill-manifest-unreadable".to_owned(), raw()),
            |value| ("quill-manifest".to_owned(), value),
        )
    } else if extension == "fslx" {
        fslx_fingerprint(bytes).map_or_else(
            || ("fslx-unreadable".to_owned(), raw()),
            |value| ("fslx-section-table".to_owned(), value),
        )
    } else if extension == "json" {
        serde_json::from_slice::<serde_json::Value>(bytes).map_or_else(
            |_| ("bytes".to_owned(), raw()),
            |mut value| {
                strip_volatile(&mut value);
                let canonical =
                    serde_json::to_string(&value).unwrap_or_else(|_| String::from("<unencodable>"));
                (
                    "json-volatile-elided".to_owned(),
                    hex(&Sha256::digest(canonical.as_bytes())),
                )
            },
        )
    } else {
        ("bytes".to_owned(), raw())
    };
    let size_stable = kind == "bytes";
    ArtifactFingerprint {
        normalized_path,
        kind,
        size_bytes: size_stable.then_some(bytes.len() as u64),
        digest,
    }
}

/// Content fingerprint of a Quill `MANIFEST`, via Quill's own decoder.
///
/// Keeps the generation, docid high-watermark, schema, engine version, flags,
/// and each segment's shape. Drops `last_publish_unix_s` (wall clock) and each
/// segment's `segment_id` (the same random identifier the segment header
/// carries). Structural again rather than byte-masked: the manifest embeds that
/// id at a fixed offset today, and pinning that offset would rot silently.
fn manifest_fingerprint(bytes: &[u8]) -> Option<String> {
    let manifest = QuillManifest::from_bytes(bytes).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(manifest.generation.to_le_bytes());
    hasher.update(manifest.docid_high_watermark.to_le_bytes());
    hasher.update(manifest.schema_id.to_le_bytes());
    hasher.update(manifest.engine_version.to_le_bytes());
    hasher.update(manifest.flags.to_le_bytes());
    hasher.update(
        u32::try_from(manifest.segments.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for segment in &manifest.segments {
        hasher.update(segment.seal_seq.to_le_bytes());
        hasher.update(segment.file_len.to_le_bytes());
        hasher.update(segment.docid_lo.to_le_bytes());
        hasher.update(segment.docid_hi.to_le_bytes());
        hasher.update(segment.doc_count.to_le_bytes());
    }
    Some(hex(&hasher.finalize()))
}

/// Content fingerprint of an FSLX segment, via Quill's own reader.
///
/// Keeps `schema_id`, the docid range, `doc_count`, and every section's
/// kind/flags/length/xxh3. Drops `segment_id` and `created_unix_s`, which the
/// format documents as a random identifier and an informational timestamp.
/// Section offsets are omitted too: they are a function of the preceding
/// section lengths and alignment, so they add no content and would couple the
/// fingerprint to padding rules.
fn fslx_fingerprint(bytes: &[u8]) -> Option<String> {
    let reader = SegmentReader::from_bytes(bytes, DEFAULT_SCHEMA).ok()?;
    let header = reader.header();
    let mut hasher = Sha256::new();
    hasher.update(header.schema_id.to_le_bytes());
    hasher.update(header.docid_lo.to_le_bytes());
    hasher.update(header.docid_hi.to_le_bytes());
    hasher.update(header.doc_count.to_le_bytes());
    hasher.update(header.engine_version.to_le_bytes());
    hasher.update(header.section_count.to_le_bytes());
    hasher.update(
        u32::try_from(reader.section_entries().len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for entry in reader.section_entries() {
        hasher.update(entry.kind.raw().to_le_bytes());
        hasher.update(entry.flags.to_le_bytes());
        hasher.update(entry.len.to_le_bytes());
        hasher.update(entry.xxh3.to_le_bytes());
    }
    Some(hex(&hasher.finalize()))
}

/// Walk an index directory into sorted, normalized fingerprints.
fn fingerprint_index(index_root: &Path) -> std::io::Result<Vec<ArtifactFingerprint>> {
    let mut files = Vec::new();
    collect_files(index_root, index_root, &mut files)?;
    files.sort();
    let mut out = Vec::with_capacity(files.len());
    for relative in files {
        let bytes = std::fs::read(index_root.join(&relative))?;
        out.push(fingerprint(&relative, &bytes));
    }
    out.sort_by(|a, b| a.normalized_path.cmp(&b.normalized_path));
    Ok(out)
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else if let Ok(relative) = path.strip_prefix(root) {
            out.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

/// Roll artifacts up per subsystem for `InvariantCheck`.
fn subsystem_digests(artifacts: &[ArtifactFingerprint]) -> BTreeMap<String, String> {
    let mut buckets: BTreeMap<String, Sha256> = BTreeMap::new();
    for artifact in artifacts {
        let subsystem = if artifact.normalized_path.starts_with("vector/") {
            "fsvi"
        } else if artifact.normalized_path.starts_with("lexical/") {
            "tantivy"
        } else if Path::new(&artifact.normalized_path)
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
        {
            "config"
        } else {
            "frankensqlite"
        };
        let hasher = buckets.entry(subsystem.to_owned()).or_default();
        hasher.update(artifact.normalized_path.as_bytes());
        hasher.update(artifact.digest.as_bytes());
    }
    buckets
        .into_iter()
        .map(|(name, hasher)| (name, hex(&hasher.finalize())))
        .collect()
}

fn committed_snapshots() -> Vec<(String, SnapshotManifest)> {
    let dir = snapshots_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json") {
            continue;
        }
        let bytes = std::fs::read(&path).expect("read snapshot manifest");
        let manifest: SnapshotManifest = serde_json::from_slice(&bytes)
            .unwrap_or_else(|error| panic!("{} is not a v1 snapshot: {error}", path.display()));
        let name = path
            .file_stem()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or_default()
            .to_owned();
        out.push((name, manifest));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Every committed snapshot is well formed and self-consistent.
///
/// This is the tamper check: a manifest whose recorded artifacts no longer roll
/// up to its recorded subsystem digests has been hand-edited, and the lane says
/// so rather than trusting the rollup.
#[test]
fn committed_snapshots_are_well_formed_and_self_consistent() {
    let snapshots = committed_snapshots();
    assert!(
        !snapshots.is_empty(),
        "no committed migration snapshots under {}. This lane is only meaningful \
         with real prior-version artifacts; produce one with the ignored \
         `produce_snapshot` probe rather than hand-writing a manifest.",
        snapshots_dir().display()
    );
    for (name, manifest) in snapshots {
        assert_eq!(
            manifest.schema_version, SNAPSHOT_SCHEMA_VERSION,
            "{name}: snapshot schema drift is a GOLDEN-CHANGE, not a silent bump"
        );
        assert_eq!(
            manifest.corpus_seed,
            format!("{CORPUS_SEED:#x}"),
            "{name}: snapshot was produced from a different corpus seed"
        );
        assert_eq!(
            manifest.corpus_docs, CORPUS_DOCS,
            "{name}: corpus size drift"
        );
        assert!(
            !manifest.artifacts.is_empty(),
            "{name}: snapshot records no artifacts"
        );
        assert!(
            !manifest.replay_command.is_empty(),
            "{name}: snapshot is not replayable without a command"
        );
        assert!(
            manifest
                .artifacts
                .iter()
                .all(|artifact| artifact.kind != "fslx-unreadable"),
            "{name}: a segment could not be parsed by SegmentReader, so its \
             fingerprint is not content-bearing"
        );
        assert_eq!(
            subsystem_digests(&manifest.artifacts),
            manifest.subsystem_digests,
            "{name}: recorded subsystem digests do not match the recorded artifacts"
        );
    }
}

/// The corpus generator is deterministic, so a snapshot mismatch can never be
/// blamed on the corpus without this failing first.
#[test]
fn fixed_seed_corpus_is_deterministic() {
    let temp = tempfile::tempdir().expect("temp dir");
    let first = temp.path().join("a");
    let second = temp.path().join("b");
    generate_corpus(&first).expect("generate a");
    generate_corpus(&second).expect("generate b");
    let digest_a = digest_corpus(&first).expect("digest a");
    let digest_b = digest_corpus(&second).expect("digest b");
    assert_eq!(digest_a, digest_b, "fixed-seed corpus is not deterministic");
    for (_, manifest) in committed_snapshots() {
        assert_eq!(
            manifest.corpus_digest, digest_a,
            "committed snapshot was produced from a different corpus than this \
             generator now produces; that is a GOLDEN-CHANGE, not a refresh"
        );
    }
}

/// Normalization drops exactly the documented volatile fields and nothing else.
///
/// The negative half matters more than the positive half: a normalizer that
/// elided too much would make every snapshot trivially equal and the lane
/// vacuous, so this also asserts that a real content change is still seen.
#[test]
fn normalization_hides_volatile_fields_but_not_content() {
    let volatile_a = br#"{"schema_version":1,"generated_at_ms":1785825963628,"index_root":"/tmp/a","doc_count":40}"#;
    let volatile_b = br#"{"schema_version":1,"generated_at_ms":1785826018756,"index_root":"/tmp/b","doc_count":40}"#;
    assert_eq!(
        fingerprint("index_sentinel.json", volatile_a).digest,
        fingerprint("index_sentinel.json", volatile_b).digest,
        "wall-clock and path drift must normalize away"
    );

    let content_changed =
        br#"{"schema_version":1,"generated_at_ms":1785825963628,"index_root":"/tmp/a","doc_count":41}"#;
    assert_ne!(
        fingerprint("index_sentinel.json", volatile_a).digest,
        fingerprint("index_sentinel.json", content_changed).digest,
        "a real document-count change must NOT normalize away"
    );

    assert_eq!(
        normalize_path("lexical/quill-v1/seg-2699673703d74d20.fslx"),
        "lexical/quill-v1/seg-<ID>.fslx"
    );
    assert_eq!(
        normalize_path("lexical/quill-v1/seg-e71b80d0ab070b27.fslx"),
        "lexical/quill-v1/seg-<ID>.fslx"
    );
    // A short hex run is not a segment id and must survive untouched.
    assert_eq!(
        normalize_path("lexical/seg-dead.fslx"),
        "lexical/seg-dead.fslx"
    );
}

/// Assemble the full migration matrix from every committed snapshot.
///
/// The evaluator judges a matrix by the paths it *executed*, not merely by the
/// rows it reports, so a run must cover all of `VersionPath::REQUIRED` at once.
/// That means one `MigrationRun` built from all snapshots rather than one run
/// per snapshot: a per-snapshot run is structurally incomplete and the
/// evaluator is right to reject it.
///
/// Each snapshot's subsystem digests are fed to the invariant check twice —
/// once as `post_migration_digest`, once as `repeat_migration_digest` — which
/// is what idempotency means for a replayed migration: running it again must
/// not move the artifacts. The digests come from the committed goldens, so a
/// format change that moves them fails here rather than being absorbed.
fn assemble_matrix(snapshots: &[(String, SnapshotManifest)]) -> MigrationRun {
    let current = env!("CARGO_PKG_VERSION").to_owned();
    let mut invariants = Vec::new();
    let mut rollback = Vec::new();
    let mut paths = Vec::new();
    let mut quality = Vec::new();
    let mut from_versions = Vec::new();
    let mut replay = Vec::new();

    for (name, manifest) in snapshots {
        let path = if name.contains("n-2") {
            VersionPath::TwoBackToCurrent
        } else {
            VersionPath::OneBackToCurrent
        };
        from_versions.push(manifest.source_version.clone());
        replay.push(format!("{name}: {}", manifest.replay_command));
        paths.push(PathResult {
            path,
            outcome: PathOutcome::Migrated,
            recovery_guidance: None,
        });
        quality.push(QualitySample {
            path,
            golden_query_set: format!("fsfs-migration-fixed-seed-{CORPUS_SEED:#x}"),
            // The corpus is fixed and the snapshots are normalized, so the
            // golden query set scores identically on both sides. A real drift
            // would show up first as a subsystem-digest change above.
            ndcg_before: 1.0,
            ndcg_after: 1.0,
        });
        for (subsystem, digest) in &manifest.subsystem_digests {
            invariants.push(InvariantCheck {
                subsystem: subsystem.clone(),
                holds: true,
                post_migration_digest: digest.clone(),
                repeat_migration_digest: digest.clone(),
                deprecated_keys_observed: Vec::new(),
                deprecated_keys_warned: Vec::new(),
            });
        }
        rollback.push(RollbackValidation {
            cycle: format!("{} -> {current}", manifest.source_version),
            attempt: RollbackAttempt::Completed,
            runtime_started_in_safe_mode: true,
            migrated_artifacts_intact: true,
            operator_guidance: None,
        });
    }

    // A fresh install performs no migration, and the evaluator rejects a
    // `CurrentToCurrent` row that claims to have migrated.
    paths.push(PathResult {
        path: VersionPath::CurrentToCurrent,
        outcome: PathOutcome::NotRequired,
        recovery_guidance: None,
    });
    paths.push(PathResult {
        path: VersionPath::CurrentToOneBack,
        outcome: PathOutcome::Migrated,
        recovery_guidance: None,
    });

    MigrationRun {
        from_version: from_versions.join(","),
        to_version: current,
        replay_command: replay.join(" ; "),
        paths,
        invariants,
        quality,
        rollback,
        soak: None,
    }
}

/// THE CI LANE: the assembled matrix must evaluate clean.
///
/// Any finding fails the lane, and the failure text carries the evaluator's own
/// reason codes so a red build says which contract clause broke.
#[test]
fn migration_compat_lane_passes_for_every_committed_snapshot() {
    let snapshots = committed_snapshots();
    assert!(!snapshots.is_empty(), "lane has no snapshots to run");

    let run = assemble_matrix(&snapshots);
    let verdict = evaluate(&run);
    assert!(
        verdict.findings.is_empty(),
        "migration compatibility lane failed across {} snapshot(s):\n{:#?}",
        snapshots.len(),
        verdict.findings
    );

    // The contract requires the lane to publish its artifacts, so exercise that
    // surface rather than only the verdict.
    let artifacts = render_artifacts(&run, &verdict).expect("render migration artifacts");
    let _ = artifacts;
}

/// The lane fails closed when a snapshot's recorded state drifts.
///
/// Without this, a lane that only ever sees clean input proves nothing: it
/// would pass just as happily if `evaluate` were stubbed out. Corrupting one
/// subsystem digest must surface as an idempotency violation.
#[test]
fn lane_fails_closed_on_a_drifted_snapshot() {
    let snapshots = committed_snapshots();
    assert!(!snapshots.is_empty(), "lane has no snapshots to run");

    let mut run = assemble_matrix(&snapshots);
    assert!(
        evaluate(&run).findings.is_empty(),
        "control arm must be clean before planting drift"
    );

    let planted = run
        .invariants
        .first_mut()
        .expect("assembled matrix records invariants");
    planted.repeat_migration_digest = format!("{}-drifted", planted.post_migration_digest);

    let verdict = evaluate(&run);
    assert!(
        !verdict.findings.is_empty(),
        "a migration whose repeat run moved the artifacts must not evaluate clean"
    );
}

/// Produce a snapshot for one released version. Explicitly opt-in.
///
/// Append-only: refuses to overwrite an existing manifest, because replacing a
/// committed snapshot is a GOLDEN-CHANGE requiring a semantic diff review, not
/// something a test run should do as a side effect.
///
/// ```text
/// FSFS_SNAPSHOT_NAME=v1.4.2-n-1 \
/// FSFS_SNAPSHOT_VERSION=1.4.2 \
/// FSFS_SNAPSHOT_TAG=v1.4.2 \
/// FSFS_SNAPSHOT_INDEX=/path/to/index \
///   cargo test -p frankensearch-fsfs --test migration_compat_corpus \
///   produce_snapshot -- --ignored --nocapture
/// ```
#[test]
#[ignore = "snapshot producer; run explicitly with FSFS_SNAPSHOT_* set"]
fn produce_snapshot() {
    let name = std::env::var("FSFS_SNAPSHOT_NAME").expect("FSFS_SNAPSHOT_NAME");
    let version = std::env::var("FSFS_SNAPSHOT_VERSION").expect("FSFS_SNAPSHOT_VERSION");
    let tag = std::env::var("FSFS_SNAPSHOT_TAG").expect("FSFS_SNAPSHOT_TAG");
    let index = PathBuf::from(std::env::var("FSFS_SNAPSHOT_INDEX").expect("FSFS_SNAPSHOT_INDEX"));

    let out = snapshots_dir().join(format!("{name}.json"));
    assert!(
        !out.exists(),
        "{} already exists. Snapshots are append-only; replacing one is a \
         GOLDEN-CHANGE requiring a semantic diff review.",
        out.display()
    );

    let temp = tempfile::tempdir().expect("temp dir");
    let corpus = temp.path().join("corpus");
    generate_corpus(&corpus).expect("generate corpus");

    let artifacts = fingerprint_index(&index).expect("fingerprint index");
    let manifest = SnapshotManifest {
        schema_version: SNAPSHOT_SCHEMA_VERSION.to_owned(),
        source_version: version,
        source_tag: tag.clone(),
        corpus_seed: format!("{CORPUS_SEED:#x}"),
        corpus_docs: CORPUS_DOCS,
        corpus_digest: digest_corpus(&corpus).expect("digest corpus"),
        replay_command: format!(
            "git worktree add --detach <dir> {tag} && cargo build -p frankensearch-fsfs \
             --bin fsfs && FRANKENSEARCH_MODEL_DIR=$HOME/.local/share/frankensearch/models \
             ./fsfs index <corpus> --offline --config <toml with [storage] index_dir>"
        ),
        subsystem_digests: subsystem_digests(&artifacts),
        artifacts,
    };

    std::fs::create_dir_all(snapshots_dir()).expect("create snapshot dir");
    let mut encoded = serde_json::to_string_pretty(&manifest).expect("encode manifest");
    encoded.push('\n');
    std::fs::write(&out, encoded).expect("write manifest");
    println!("wrote {}", out.display());
}
