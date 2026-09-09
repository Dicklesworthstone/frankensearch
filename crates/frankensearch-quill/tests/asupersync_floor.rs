//! Declared-floor guard for the `asupersync` dependency (gh#44).
//!
//! `frankensearch-quill` polls cancellation on its per-posting hot path with
//! [`Cx::is_cancelled`], which asupersync added in 0.4.10. The 0.2.3 and 0.2.4
//! crates were published declaring `asupersync >= 0.4.4`, so a consumer whose
//! lockfile held 0.4.9 or below resolved the requirement fine and then failed
//! with `E0599` from inside this crate. A fresh resolve always picked 0.4.10,
//! which is why the workspace build never noticed.
//!
//! The lockfile cannot catch that class of regression: it records the version
//! that was resolved, not the floor that was declared. This test reads the
//! declared requirement itself, so lowering the floor below the API this crate
//! actually calls fails the suite instead of shipping.

use std::fs;
use std::path::{Path, PathBuf};

use asupersync::Cx;

/// First asupersync release that has `Cx::is_cancelled`.
const REQUIRED_FLOOR: (u64, u64, u64) = (0, 4, 10);

/// Compile-time witness for WHY the floor is what it is: this is the method
/// whose absence below 0.4.10 broke consumers. If a future refactor stops
/// calling it, this reference still documents the floor's origin; if
/// asupersync ever removes it, this file stops compiling.
#[allow(dead_code)]
fn floor_witness(cx: &Cx) -> bool {
    cx.is_cancelled()
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/frankensearch-quill sits two levels below the workspace root")
        .to_path_buf()
}

fn read_toml(path: &Path) -> toml::Table {
    let text =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    text.parse::<toml::Table>()
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Parse a `major.minor.patch` triple, tolerating a trailing pre-release tag.
fn parse_version(text: &str) -> Option<(u64, u64, u64)> {
    let core = text.split(['-', '+']).next()?;
    let mut parts = core.split('.').map(|part| part.trim().parse::<u64>());
    let major = parts.next()?.ok()?;
    let minor = parts.next()?.ok()?;
    let patch = parts.next()?.ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// The lowest version a Cargo requirement string admits.
///
/// Handles the comparator forms Cargo accepts for a floor: `>=X.Y.Z`,
/// `^X.Y.Z`, bare `X.Y.Z`, `=X.Y.Z`, and `~X.Y.Z`, any of them comma-joined
/// with a ceiling such as `<0.5`. A `>` comparator is rejected as ambiguous
/// rather than approximated.
fn requirement_floor(requirement: &str) -> (u64, u64, u64) {
    let mut floor = None;
    for comparator in requirement.split(',') {
        let comparator = comparator.trim();
        let candidate = if let Some(rest) = comparator.strip_prefix(">=") {
            rest
        } else if comparator.starts_with('<') {
            continue;
        } else if comparator.starts_with('>') {
            panic!(
                "asupersync requirement {requirement:?} uses an exclusive lower bound; declare the floor with >="
            );
        } else {
            comparator.trim_start_matches(['^', '=', '~'])
        };
        let parsed = parse_version(candidate)
            .unwrap_or_else(|| panic!("unparseable asupersync comparator {comparator:?}"));
        assert!(
            floor.is_none(),
            "asupersync requirement {requirement:?} declares more than one lower bound"
        );
        floor = Some(parsed);
    }
    floor
        .unwrap_or_else(|| panic!("asupersync requirement {requirement:?} declares no lower bound"))
}

fn dependency_table<'a>(manifest: &'a toml::Table, section: &str) -> &'a toml::Table {
    manifest
        .get(section)
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| panic!("manifest has no [{section}] table"))
}

#[test]
fn workspace_asupersync_floor_covers_cx_is_cancelled() {
    let root = workspace_root();
    let workspace = read_toml(&root.join("Cargo.toml"));
    let workspace_deps =
        dependency_table(dependency_table(&workspace, "workspace"), "dependencies");
    let asupersync = workspace_deps
        .get("asupersync")
        .expect("[workspace.dependencies] declares asupersync");
    let requirement = match asupersync {
        toml::Value::String(requirement) => requirement.as_str(),
        toml::Value::Table(table) => table
            .get("version")
            .and_then(toml::Value::as_str)
            .expect("[workspace.dependencies].asupersync has a version requirement"),
        other => panic!("unexpected asupersync dependency shape: {other:?}"),
    };
    let floor = requirement_floor(requirement);
    assert!(
        floor >= REQUIRED_FLOOR,
        "workspace declares asupersync {requirement:?} (floor {floor:?}) but \
         frankensearch-quill calls Cx::is_cancelled, which needs >= {REQUIRED_FLOOR:?} (gh#44)"
    );
}

#[test]
fn quill_takes_asupersync_from_the_workspace_in_every_section() {
    let manifest = read_toml(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"));
    for section in ["dependencies", "dev-dependencies"] {
        let entry = dependency_table(&manifest, section)
            .get("asupersync")
            .unwrap_or_else(|| panic!("[{section}] declares asupersync"));
        let inherits = entry
            .as_table()
            .and_then(|table| table.get("workspace"))
            .and_then(toml::Value::as_bool)
            .unwrap_or(false);
        assert!(
            inherits,
            "[{section}].asupersync must inherit the workspace floor (found {entry:?}); \
             a crate-level requirement would bypass the gh#44 floor guard"
        );
        assert!(
            entry
                .as_table()
                .is_some_and(|table| !table.contains_key("version")),
            "[{section}].asupersync must not carry its own version next to `workspace = true`"
        );
    }
}

#[test]
fn requirement_floor_parses_every_cargo_comparator_shape() {
    assert_eq!(requirement_floor(">=0.4.10, <0.5"), (0, 4, 10));
    assert_eq!(requirement_floor("^0.4.10"), (0, 4, 10));
    assert_eq!(requirement_floor("0.4.10"), (0, 4, 10));
    assert_eq!(requirement_floor("=0.4.10"), (0, 4, 10));
    assert_eq!(requirement_floor("~0.4.10"), (0, 4, 10));
    assert_eq!(requirement_floor("<0.5, >=0.4.4"), (0, 4, 4));
    assert!((0, 4, 9) < REQUIRED_FLOOR);
    assert!((0, 4, 10) >= REQUIRED_FLOOR);
    assert!((0, 5, 0) > REQUIRED_FLOOR);
}
