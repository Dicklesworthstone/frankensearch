# Crates.io publication contract

Frankensearch publishes a heterogeneous bundle of crates from one immutable
Git commit. A successful GitHub release, a successful workspace build, or an
`already uploaded` response from crates.io is not publication evidence.

The authoritative preflight is:

```bash
scripts/check_crates_publish_contract.sh \
  --mode gate \
  --scope workspace \
  --registry-census live \
  --release-tag "crates-v$(cargo metadata --no-deps --format-version 1 |
    jq -r '.packages[] | select(.name == "frankensearch") | .version')"
```

The command never publishes. It emits a JSON receipt and exits non-zero unless
the exact candidate can be published without overwriting or silently reusing
different source bytes.

## Tag and source identity

Binary `fsfs` releases and crate-bundle releases use separate namespaces:

- `v<fsfs-version>` identifies a binary release.
- `crates-v<facade-version>` identifies a crates.io bundle.

The crate-bundle tag version names the facade, not every crate. The receipt
records the actual version of every package, the complete publication order,
the candidate Git SHA, the `Cargo.lock` SHA-256, the Rust/Cargo versions, and
the registry identity of every occupied `(crate, version)` pair.

A dirty tracked worktree is never a release candidate. Non-ignored untracked
files are also forbidden: Cargo package selection can include them even though
they have no committed provenance. The planner records their repository-relative
paths and emits `UNTRACKED_PACKAGE_FILES`. Ignored untracked files remain
allowed because Cargo excludes Git-ignored files by default.

`--allow-dirty` exists only for the planner's synthetic self-tests. Release
gates and `cargo package` / `cargo publish` invocations MUST NOT use it.

## Publication scope

`--scope workspace` is the release policy. It includes every workspace member
whose Cargo metadata permits publication. This prevents the release job from
silently omitting independent products such as `frankensearch-fsfs`,
`frankensearch-tui`, or `frankensearch-ops`.

A workspace member that should never be published must say so explicitly:

```toml
[package]
publish = false
```

`--scope facade` exists for focused development checks. It recursively follows
all normal and build dependencies of the `frankensearch` facade, including
optional dependencies, because optional dependencies are still present in the
packaged manifest.

The planner derives a deterministic topological order from `cargo metadata`.
No crate list is duplicated in the workflow. Internal dependencies must have a
registry version requirement and must appear before their dependents.

## Registry census

The live census performs credential-free reads only:

1. Query the exact crate/version from the crates.io API.
2. Download the immutable `.crate` archive when the version exists.
3. verify the archive SHA-256 against crates.io metadata.
4. Read `.cargo_vcs_info.json` from the archive.
5. Compare the published source SHA and dirty bit to the candidate.
6. If the source matches, build the candidate `.crate` archive and require its
   SHA-256 to match the registry archive exactly.

An occupied version is skippable only when its archive names the clean exact
candidate source commit and the candidate archive is byte-identical. A generic
`already uploaded` error is never accepted. If source or content differs, or
provenance is absent, the version is occupied and must be bumped together with
every affected internal dependency requirement.

The census can be captured independently and injected with
`--registry-census <path>`. Its schema is printed by `--help`. Gate mode
requires either a live or injected census.

The download and dependency rules follow Cargo's registry contracts:

- [Cargo registry index and download endpoint](https://doc.rust-lang.org/cargo/reference/registry-index.html)
- [Cargo registry web API](https://doc.rust-lang.org/cargo/reference/registry-web-api.html)
- [Cargo dependency version requirements](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html)

## Typed blockers

The receipt uses stable blocker codes so CI, Beads, and release tooling can
route work without parsing prose.

| Code | Meaning |
|---|---|
| `DIRTY_TRACKED_WORKTREE` | The candidate bytes are not the named Git commit. |
| `UNTRACKED_PACKAGE_FILES` | Non-ignored untracked files may enter Cargo package selection without committed provenance. |
| `INTERNAL_DEPENDENCY_CYCLE` | No valid crates.io publication order exists. |
| `INTERNAL_DEPENDENCY_VERSION_REQUIRED` | A path dependency would lose its usable registry requirement when packaged. |
| `INTERNAL_DEPENDENCY_VERSION_MISMATCH` | A dependent does not require the exact candidate internal version. |
| `INTERNAL_DEPENDENCY_NOT_PUBLISHABLE` | A registry package depends on a workspace package marked non-publishable. |
| `DEPENDENCY_GIT_VERSION_REQUIRED` | Cargo cannot rewrite a git-only dependency to an existing registry version. |
| `PACKAGE_*_MISSING` | Required crates.io package metadata is absent. |
| `REGISTRY_CENSUS_*` | Registry identity is missing, malformed, or unverifiable. |
| `PUBLISHED_VERSION_SOURCE_MISMATCH` | The version is occupied by different source bytes. |
| `PUBLISHED_VERSION_CONTENT_MISMATCH` | The source SHA matches but the candidate archive does not. |
| `RELEASE_TAG_*` | The crate-bundle tag is absent or maps to the wrong facade version. |

Known campaign prerequisites are attached directly to blocker rows. The
external HNSW release/removal route is `bd-mczj`; the Frankentorch registry-name
or package-graph resolution is `bd-8nqz.6-ft-registry`.

## Evidence ladder

The metadata receipt is necessary but not sufficient. Once it is green, release
evidence proceeds in this order for every planned package:

```bash
cargo package --list -p <crate>
cargo package --locked -p <crate>
cargo publish --locked --dry-run -p <crate>
```

Then install the packaged artifacts into a temporary local registry and build a
clean consumer with no path, Git, or `[patch]` overrides. The consumer matrix
must cover:

- no default features;
- `lexical`;
- `hybrid`;
- `persistent`;
- `durable`;
- `full`;
- `lexical-tantivy`;
- `cass-compat`;
- all features.

Backend-sensitive lanes must execute at least one lexical or hybrid query; a
successful dependency resolution alone does not prove that the selected engine
is wired correctly.

Actual publication remains a separately authorized, credentialed action. The
preflight receipt contains no token and is safe to upload as a CI artifact.

## Local verification

```bash
bash -n scripts/check_crates_publish_contract.sh
scripts/check_crates_publish_contract.sh --mode self-test

scripts/check_crates_publish_contract.sh \
  --mode audit \
  --scope workspace \
  --registry-census live \
  --output /tmp/frankensearch-crates-publish-contract/audit.json

jq '{status, blocker_codes, packages}' \
  /tmp/frankensearch-crates-publish-contract/audit.json
```

The self-test uses a committed temporary Git repository and proves that a clean
tree passes, ignored untracked files remain allowed, and non-ignored untracked
files fail with `UNTRACKED_PACKAGE_FILES`.

`audit` always preserves its receipt when readiness blockers exist. `gate`
uses the same analysis but exits non-zero unless the receipt status is
`ready`.
