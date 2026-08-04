//! bd-ctzo C5: validate the machine-readable activation call-site census.
//!
//! `docs/CTZO_ACTIVATION_CENSUS.json` names every PRODUCTION entry point that
//! was migrated onto typed owner-backed activation. A checked-in list proves
//! nothing on its own — it rots the moment someone adds a call site — so this
//! suite is the proof half:
//!
//! - every named symbol still exists, in the file the census names, in that
//!   file's PRODUCTION half (above `#[cfg(test)] mod tests`);
//! - the number of production call sites of the activation constructor and of
//!   the coverage receipt matches the census exactly, so adding or deleting
//!   one without updating the census fails here;
//! - the new types are consumed OUTSIDE `frankensearch-core`, which is the
//!   half of C5 that distinguishes "defined" from "used".
//!
//! The census is read from the workspace source tree, not from a build
//! artifact, because the claim is about what the shipped source does.

use std::path::{Path, PathBuf};

/// Workspace root, derived from this crate's manifest directory.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<crate> sits two levels below the workspace root")
        .to_path_buf()
}

fn read(relative: &str) -> String {
    let path = workspace_root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("census names {relative}, which could not be read: {error}");
    })
}

/// The production half of a source file: everything above the inline test
/// module. Consuming a symbol only from a test would satisfy "the symbol
/// exists" while leaving production unmigrated, which is exactly the
/// confusion C5 exists to prevent.
fn production_half(source: &str) -> &str {
    source
        .split_once("\n#[cfg(test)]")
        .map_or(source, |(production, _)| production)
}

fn census() -> serde_json::Value {
    serde_json::from_str(&read("docs/CTZO_ACTIVATION_CENSUS.json"))
        .expect("the census must be valid JSON")
}

#[test]
fn every_named_entry_point_exists_in_production_source() {
    let census = census();
    let entries = census["entry_points"]
        .as_array()
        .expect("entry_points is an array");
    assert!(
        entries.len() >= 8,
        "the census lost entries; it named 8 migrated entry points"
    );

    for entry in entries {
        let file = entry["file"].as_str().expect("file");
        let symbol = entry["symbol"].as_str().expect("symbol");
        let source = read(file);
        let production = production_half(&source);
        assert!(
            production.contains(symbol),
            "census names `{symbol}` in {file}, but it is absent from that file's \
             production half — either the symbol was renamed or it now lives only in tests"
        );
        assert!(
            file.contains("/src/"),
            "a census entry must name a production source file, got {file}"
        );
    }
}

#[test]
fn production_call_site_counts_match_the_census_exactly() {
    let census = census();
    for site in census["production_call_sites"]
        .as_array()
        .expect("production_call_sites is an array")
    {
        let file = site["file"].as_str().expect("file");
        let callee = site["callee"].as_str().expect("callee");
        let expected = usize::try_from(site["count"].as_u64().expect("count")).expect("count fits");

        let source = read(file);
        let production = production_half(&source);
        // The definition line is not a call site; count only invocations.
        let actual = production
            .lines()
            .filter(|line| {
                let trimmed = line.trim_start();
                !trimmed.starts_with("///")
                    && !trimmed.starts_with("//")
                    && !trimmed.starts_with("pub fn ")
                    && !trimmed.starts_with("fn ")
                    && line.contains(callee)
            })
            .count();
        assert_eq!(
            actual, expected,
            "{file} has {actual} production call sites of `{callee}` but the census says \
             {expected}; update docs/CTZO_ACTIVATION_CENSUS.json in the same change"
        );
    }
}

#[test]
fn the_activation_types_are_consumed_outside_their_defining_crate() {
    let census = census();
    let definition_crate = census["definition_crate"]
        .as_str()
        .expect("definition_crate");
    assert_eq!(definition_crate, "frankensearch-core");

    let consumers: Vec<&str> = census["entry_points"]
        .as_array()
        .expect("entry_points")
        .iter()
        .filter_map(|entry| entry["crate"].as_str())
        .filter(|krate| *krate != definition_crate)
        .collect();
    assert!(
        consumers.contains(&"frankensearch-index"),
        "the index crate must consume the activation types"
    );
    assert!(
        consumers.contains(&"frankensearch-fusion"),
        "the fusion crate must consume the activation types"
    );

    // And the strong form: a type defined in core is named in a production
    // file of another crate. `docs` alone would not prove consumption.
    let searcher = read("crates/frankensearch-fusion/src/searcher.rs");
    assert!(
        production_half(&searcher).contains("TieredQueryEmbeddings"),
        "the async searcher must consume TieredQueryEmbeddings in production code"
    );
    let sync_searcher = read("crates/frankensearch-fusion/src/sync_searcher.rs");
    assert!(
        production_half(&sync_searcher).contains("TieredQueryEmbeddings"),
        "the sync searcher must consume TieredQueryEmbeddings in production code"
    );
}

/// The census must not claim the downstream work bd-ctzo explicitly excludes.
#[test]
fn the_census_claims_no_downstream_work() {
    let raw = read("docs/CTZO_ACTIVATION_CENSUS.json").to_lowercase();
    for forbidden in [
        "fsfs persistence",
        "raw api retirement",
        "daemon enforcement",
        "recovery",
        "cass",
    ] {
        assert!(
            !raw.contains(forbidden),
            "the census must not claim `{forbidden}`; bd-ctzo scopes that downstream"
        );
    }
}
