#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA="$ROOT_DIR/schemas/fsfs-packaging-release-install-v1.schema.json"
FAILURES=0

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_packaging_release_install_contract.sh [--mode unit|integration|e2e|installer|model-features|all]

Validates fsfs packaging/release/install fixtures and model-feature boundaries.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  unit|integration|e2e|installer|model-features|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|integration|e2e|installer|model-features|all)" >&2
    exit 2
    ;;
esac

if [[ "$MODE" != "model-features" ]]; then
  if [[ ! -f "$SCHEMA" ]]; then
    echo "ERROR: schema not found: $SCHEMA" >&2
    exit 2
  fi
  if ! command -v jsonschema >/dev/null 2>&1; then
    echo "ERROR: jsonschema CLI not found in PATH" >&2
    exit 2
  fi
fi

check_valid() {
  local scope="$1"
  local file="$2"
  if jsonschema -i "$file" "$SCHEMA" >/dev/null 2>&1; then
    echo "[$scope][OK]   valid fixture accepted: $file"
  else
    echo "[$scope][FAIL] valid fixture rejected: $file"
    FAILURES=$((FAILURES + 1))
  fi
}

check_invalid() {
  local scope="$1"
  local file="$2"
  if jsonschema -i "$file" "$SCHEMA" >/dev/null 2>&1; then
    echo "[$scope][FAIL] invalid fixture unexpectedly accepted: $file"
    FAILURES=$((FAILURES + 1))
  else
    echo "[$scope][OK]   invalid fixture rejected: $file"
  fi
}

check_unit() {
  echo "[unit] validating release matrix, artifact naming, and policy coverage"
  check_valid "unit" "$ROOT_DIR/schemas/fixtures/fsfs-packaging-release-install-contract-v1.json"
  check_invalid "unit" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-packaging-release-install-invalid-missing-target-v1.json"
}

check_integration() {
  echo "[integration] validating release manifest artifact + checksum envelope"
  check_valid "integration" "$ROOT_DIR/schemas/fixtures/fsfs-packaging-release-install-release-manifest-v1.json"
  check_invalid "integration" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-packaging-release-install-invalid-missing-integrity-v1.json"
}

check_e2e() {
  echo "[e2e] validating upgrade path and rollback telemetry expectations"
  check_valid "e2e" "$ROOT_DIR/schemas/fixtures/fsfs-packaging-release-install-upgrade-plan-v1.json"
}

check_installer_behavior() {
  local installer="$ROOT_DIR/install.sh"
  local archive="$ROOT_DIR/README.md"
  local expected=""
  local installer_shell="${FSFS_INSTALL_TEST_BASH:-bash}"

  echo "[installer] exercising fail-closed checksum and profile routing behavior"
  if command -v sha256sum >/dev/null 2>&1; then
    expected=$(sha256sum -- "$archive" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    expected=$(shasum -a 256 -- "$archive" | awk '{print $1}')
  else
    echo "[installer][FAIL] checker requires sha256sum or shasum for the positive checksum case"
    FAILURES=$((FAILURES + 1))
    return
  fi

  if FSFS_INSTALL_CONTRACT_TEST=1 "$installer_shell" "$installer" checksum "$archive" "$expected" >/dev/null; then
    echo "[installer][OK]   matching release checksum admitted"
  else
    echo "[installer][FAIL] matching release checksum rejected"
    FAILURES=$((FAILURES + 1))
  fi

  if FSFS_INSTALL_CONTRACT_TEST=1 "$installer_shell" "$installer" checksum "$archive" "$(printf '0%.0s' {1..64})" >/dev/null 2>&1; then
    echo "[installer][FAIL] checksum mismatch unexpectedly admitted"
    FAILURES=$((FAILURES + 1))
  else
    echo "[installer][OK]   checksum mismatch rejected"
  fi

  if FSFS_INSTALL_CONTRACT_TEST=1 "$installer_shell" "$installer" checksum "$archive" "$expected" none >/dev/null 2>&1; then
    echo "[installer][FAIL] missing SHA-256 tool unexpectedly admitted"
    FAILURES=$((FAILURES + 1))
  else
    echo "[installer][OK]   missing SHA-256 tool fails closed"
  fi

  if FSFS_INSTALL_CONTRACT_TEST=1 "$installer_shell" "$installer" manifest "$ROOT_DIR/README.md" "missing-release-artifact.tar.xz" >/dev/null 2>&1; then
    echo "[installer][FAIL] missing checksum manifest entry unexpectedly admitted"
    FAILURES=$((FAILURES + 1))
  else
    echo "[installer][OK]   missing checksum manifest entry fails closed"
  fi

  if FSFS_INSTALL_CONTRACT_TEST=1 "$installer_shell" "$installer" sidecar "$installer/missing.sha256" >/dev/null 2>&1; then
    echo "[installer][FAIL] absent checksum sidecar unexpectedly admitted"
    FAILURES=$((FAILURES + 1))
  else
    echo "[installer][OK]   absent checksum sidecar fails closed"
  fi

  local actual_route expected_route route_case explicit_lite full_artifact target
  for route_case in \
    "0 1 x86_64-unknown-linux-musl artifact-full" \
    "1 0 x86_64-apple-darwin source-lite" \
    "0 0 x86_64-unknown-linux-musl source-default" \
    "0 0 aarch64-apple-darwin source-default" \
    "0 1 x86_64-apple-darwin unsupported-semantic" \
    "0 0 x86_64-apple-darwin unsupported-semantic"; do
    read -r explicit_lite full_artifact target expected_route <<<"$route_case"
    if actual_route=$(FSFS_INSTALL_CONTRACT_TEST=1 "$installer_shell" "$installer" route "$explicit_lite" "$full_artifact" "$target") \
      && [[ "$actual_route" == "$expected_route" ]]; then
      echo "[installer][OK]   route target=$target lite=$explicit_lite artifact=$full_artifact -> $actual_route"
    else
      echo "[installer][FAIL] route target=$target lite=$explicit_lite artifact=$full_artifact expected=$expected_route actual=${actual_route:-<error>}"
      FAILURES=$((FAILURES + 1))
    fi
  done

  local unsupported_output unsupported_status
  unsupported_status=0
  unsupported_output=$(FSFS_INSTALL_CONTRACT_TEST=1 "$installer_shell" "$installer" unsupported x86_64-apple-darwin 2>&1) \
    || unsupported_status=$?
  if [[ "$unsupported_status" -eq 78 \
    && "$unsupported_output" == *"unsupported_platform"* \
    && "$unsupported_output" == *"--lite"* ]]; then
    echo "[installer][OK]   Intel macOS semantic install fails with typed actionable EX_CONFIG"
  else
    echo "[installer][FAIL] Intel macOS unsupported outcome status=$unsupported_status output=$unsupported_output"
    FAILURES=$((FAILURES + 1))
  fi
}

check_model_features() {
  echo "[model-features] validating loader-capable defaults, explicit lite, and embedded release lanes"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[model-features][FAIL] python3 is required for TOML and workflow checks"
    FAILURES=$((FAILURES + 1))
    return
  fi

  if python3 - "$ROOT_DIR" <<'PY'
import pathlib
import re
import sys
import tomllib

root = pathlib.Path(sys.argv[1])
manifest_path = root / "crates/frankensearch-fsfs/Cargo.toml"
workflow_path = root / ".github/workflows/ci.yml"
installer_path = root / "install.sh"
crate_readme_path = root / "crates/frankensearch-fsfs/README.md"
root_readme_path = root / "README.md"
contract_path = root / "docs/fsfs-packaging-release-install-contract.md"
quickstart_path = root / "crates/frankensearch-fsfs/tests/default_build_quickstart.rs"
embed_manifest_path = root / "crates/frankensearch-embed/src/model_manifest.rs"
embed_lib_path = root / "crates/frankensearch-embed/src/lib.rs"
bundled_models_path = root / "crates/frankensearch-embed/src/bundled_default_models.rs"
runtime_path = root / "crates/frankensearch-fsfs/src/runtime.rs"
auto_detect_path = root / "crates/frankensearch-embed/src/auto_detect.rs"

with manifest_path.open("rb") as handle:
    manifest = tomllib.load(handle)

failures: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


features = manifest.get("features", {})
require(
    features.get("default") == ["semantic-loaders"],
    "frankensearch-fsfs default must select semantic-loaders",
)
require(
    features.get("semantic-loaders")
    == [
        "frankensearch-embed/model2vec",
        "frankensearch-embed/fastembed",
    ],
    "semantic-loaders must compile both registered model implementations",
)
require(
    features.get("embedded-models")
    == [
        "semantic-loaders",
        "frankensearch-embed/bundled-default-models",
    ],
    "embedded-models must add pinned bundled bytes to the semantic loaders",
)
embed_dependency = manifest.get("dependencies", {}).get("frankensearch-embed", {})
require(
    embed_dependency.get("default-features") is False
    and embed_dependency.get("features") == ["hash", "download"],
    "fsfs must keep hash-control and verified acquisition explicit at the dependency boundary",
)
quickstart_target = next(
    (
        target
        for target in manifest.get("test", [])
        if target.get("name") == "default_build_quickstart"
    ),
    None,
)
require(
    quickstart_target is not None
    and quickstart_target.get("required-features") == ["semantic-loaders"],
    "default quickstart must compile under stock loader-capable defaults",
)
quickstart = quickstart_path.read_text(encoding="utf-8")
require(
    '#[cfg(not(feature = "embedded-models"))]\nmod loader_only {' in quickstart,
    "stock-default quickstart tests must run only when embedded bytes are absent",
)
require(
    'compile_error!(' not in quickstart
    and "fn embedded_release_profile_retains_semantic_loaders()" in quickstart,
    "the supported embedded profile must compile an explicit semantic-loader sentinel",
)
for evidence_marker in (
    "stage=stock-default-contract event=start",
    "stage=status-missing event=verified",
    "stage=doctor-failure event=verified",
    "stage=model-verification event=verified",
    "stage=cli-model-verify event=verified",
    "stage=durable-vector event=verified",
    "stage=index-one-shot event=verified",
    "stage=status-verified event=verified",
    "stage=doctor-success event=verified",
    "stage=real-model-semantic-only event=verified",
    "stage=hybrid-control event=verified",
):
    require(
        evidence_marker in quickstart,
        f"default quickstart missing stable success evidence marker: {evidence_marker}",
    )

workflow = workflow_path.read_text(encoding="utf-8")


def job(name: str) -> str:
    match = re.search(
        rf"^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        failures.append(f"workflow job missing: {name}")
        return ""
    return match.group("body")


quality = job("quality")
installer_platform = job("installer-platform")
release = job("release-build")
lite = job("release-build-lite")
release_publish = job("release-publish")
crates_publish = job("publish-crates")
quality_lines = {line.strip() for line in quality.splitlines()}
release_lines = {line.strip() for line in release.splitlines()}

release_targets = re.findall(r"^\s+target:\s+([^\s#]+)", release, flags=re.MULTILINE)
lite_targets = re.findall(r"^\s+target:\s+([^\s#]+)", lite, flags=re.MULTILINE)
require(
    release_targets == ["aarch64-apple-darwin", "x86_64-pc-windows-msvc"],
    f"full embedded release target matrix drifted: {release_targets}",
)
require(
    lite_targets
    == [
        "x86_64-unknown-linux-musl",
        "aarch64-unknown-linux-musl",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
    ],
    f"explicit lite release target matrix drifted: {lite_targets}",
)


def has_command(lines: set[str], command: str) -> bool:
    return command in lines or f"run: {command}" in lines

provision = "scripts/rch-ensure-deps.sh --models-only"
verify = "scripts/rch-ensure-deps.sh --models-only --check"
default_check = "cargo check --workspace --all-targets"
model_free_check = "cargo check -p frankensearch-fsfs --no-default-features"
embedded_check = (
    "cargo check -p frankensearch-fsfs --no-default-features "
    "--features embedded-models"
)
embedded_build = (
    'cargo build -p frankensearch-fsfs --release '
    '--target "${{ matrix.target }}" --no-default-features '
    "--features embedded-models"
)

for command in (default_check, model_free_check, provision, verify, embedded_check):
    require(has_command(quality_lines, command), f"quality job missing command: {command}")
if all(command in quality for command in (default_check, provision, embedded_check)):
    require(
        quality.index(default_check) < quality.index(provision) < quality.index(embedded_check),
        "quality must compile artifact-independent defaults before provisioning embedded inputs",
    )

missing_e2e = "loader_only::default_build_without_models_fails_closed_with_actionable_guidance"
real_e2e = "loader_only::default_build_indexes_and_returns_a_real_hybrid_result"
require(
    missing_e2e in quality
    and real_e2e in quality
    and "FRANKENSEARCH_REQUIRE_SEMANTIC_E2E: \"1\"" in quality
    and "-- --ignored --exact --nocapture" in quality,
    "quality must execute both fully-qualified stock-default quickstart lanes after model provisioning",
)
require(
    "fsfs-default-semantic-e2e-${{ github.run_id }}-${{ github.run_attempt }}" in quality
    and "if-no-files-found: error" in quality
    and "receipt.json" in quality,
    "stock-default semantic E2E must publish a durable fail-closed evidence bundle",
)
for lane in ("missing-corrupt", "real-model-semantic-only", "real-model-hybrid"):
    require(lane in quality, f"stock-default semantic E2E receipt missing lane: {lane}")

for platform in (
    "ubuntu-latest",
    "macos-15",
    "macos-15-intel",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
):
    require(platform in installer_platform, f"installer platform job missing lane: {platform}")
require(
    "/bin/bash scripts/check_fsfs_packaging_release_install_contract.sh --mode installer"
    in installer_platform
    and "install.sh install-built target/release/fsfs" in installer_platform
    and "--no-default-features" in installer_platform,
    "installer platform job must use the platform system shell and install a built semantic/lite profile",
)

for command in (provision, verify, embedded_build):
    require(has_command(release_lines, command), f"full release job missing command: {command}")
if all(command in release for command in (provision, verify, embedded_build)):
    require(
        release.index(provision) < release.index(verify) < release.index(embedded_build),
        "full release must provision, verify, then build embedded models",
    )
require(
    'archive_base="fsfs-${version_bare}-${TARGET_TRIPLE}"' in release
    and '"profile": "embedded"' in release
    and '"semantic_loaders": true' in release
    and '"embedded_models": true' in release,
    "full release artifacts must carry embedded-profile capability metadata",
)

lite_builds = [
    line.strip()
    for line in lite.splitlines()
    if line.strip().startswith(("cargo build ", "cargo zigbuild "))
]
require(len(lite_builds) == 2, "lite release job must retain both Cargo and zigbuild lanes")
require(
    bool(lite_builds) and all("--no-default-features" in line for line in lite_builds),
    "every lite release build must use --no-default-features",
)
require(
    "embedded-models" not in lite,
    "lite release job must not select or provision embedded models",
)
require(
    'archive_base="fsfs-lite-${version_bare}-${TARGET_TRIPLE}"' in lite
    and '"profile": "lite"' in lite
    and '"semantic_loaders": false' in lite
    and '"embedded_models": false' in lite,
    "lite release artifacts must remain distinctly named and carry model-free capability metadata",
)

require(
    "needs.release-build.result == 'success'" in release_publish
    and "needs.release-build-lite.result == 'success'" in release_publish
    and "needs.release-build.result == 'success' ||" not in release_publish
    and "Validate exact six-profile release inventory" in release_publish
    and "release.inventory.exact_six: PASS" in release_publish
    and "release.inventory.checksum_mismatch" in release_publish,
    "release-publish must require both build jobs and verify the exact six artifact/checksum pairs",
)

for forbidden in ("--models-only", "embedded-models", "FRANKENSEARCH_BUNDLED_MODELS_SOURCE_DIR"):
    require(
        forbidden not in crates_publish,
        f"publish-crates must not require model input: {forbidden}",
    )

installer = installer_path.read_text(encoding="utf-8")
crate_readme = crate_readme_path.read_text(encoding="utf-8")
root_readme = root_readme_path.read_text(encoding="utf-8")
contract = contract_path.read_text(encoding="utf-8")
embed_manifest = embed_manifest_path.read_text(encoding="utf-8")
embed_lib = embed_lib_path.read_text(encoding="utf-8")
bundled_models = bundled_models_path.read_text(encoding="utf-8")
runtime = runtime_path.read_text(encoding="utf-8")
auto_detect = auto_detect_path.read_text(encoding="utf-8")
require(
    "cargo build --release -p frankensearch-fsfs --no-default-features" in installer,
    "installer --lite source lane must remain explicitly model-free",
)
require(
    "cargo build --release -p frankensearch-fsfs)" in installer,
    "ordinary source installation must exercise the loader-capable Cargo default",
)
require(
    "plain build" in crate_readme
    and "fsfs download-models potion-multilingual-128m" in crate_readme,
    "crate README must document the no-feature default semantic provisioning path",
)
require(
    "Building default semantic-loader variant" in installer
    and "download-models potion-multilingual-128m" in installer,
    "installer source default must compile loaders and prescribe verified model acquisition",
)


def require_quickstart_order(block: str, renderer: str) -> None:
    commands = [
        "download-models potion-multilingual-128m",
        "download-models all-minilm-l6-v2",
        "download-models potion-multilingual-128m --verify",
        "download-models all-minilm-l6-v2 --verify",
        "index /path/to/files",
        'search "your query"' if renderer == "gum" else 'search \\"your query\\"',
    ]
    missing = [command for command in commands if command not in block]
    require(not missing, f"installer {renderer} quickstart missing commands: {missing}")
    if not missing:
        positions = [block.index(command) for command in commands]
        require(
            positions == sorted(positions),
            f"installer {renderer} quickstart must download, verify, index, then search",
        )


source_start = installer.index('info "Semantic loaders installed.')
source_end = installer.index('ok "Done. Binary at:', source_start)
source_quickstart = installer[source_start:source_end]
gum_start = installer.index('if [ "$HAS_GUM" -eq 1 ]', source_end)
plain_start = installer.index("\nelse\n", gum_start)
require_quickstart_order(source_quickstart, "source")
require_quickstart_order(installer[gum_start:plain_start], "gum")
require_quickstart_order(installer[plain_start:], "plain")
require(
    "Full artifact download failed; building the loader-capable default from source" in installer
    and "fsfs-lite-${VERSION_BARE}" not in installer,
    "ordinary installation must never silently substitute the explicit loader-free lite profile",
)
require(
    "unsupported-semantic" in installer
    and "unsupported_platform" in installer
    and "x86_64-apple-darwin" in installer,
    "Intel macOS must fail with a typed semantic-install outcome instead of an impossible source fallback",
)
unsupported_gate = (
    'if [ "$LITE" -eq 0 ] && [ "$TARGET" = "x86_64-apple-darwin" ]; then\n'
    '  fail_unsupported_semantic_platform "$TARGET"'
)
require(
    unsupported_gate in installer
    and installer.index(unsupported_gate) < installer.index("\nresolve_version\n")
    and "versionless" not in installer
    and "FALLBACK_TAR" not in installer
    and "FALLBACK_URL" not in installer,
    "Intel macOS must reject before version/network resolution and no generic artifact fallback may remain",
)
require(
    'CHECKSUM="SKIP"' not in installer
    and "skipping checksum verification" not in installer.lower()
    and "refusing to install an unverified artifact" in installer,
    "release artifact installation must fail closed when integrity cannot be verified",
)
require(
    "plain default" in contract
    and "no Cargo feature flag or rebuild is permitted" in contract,
    "release contract must require a no-hidden-feature default semantic path",
)
require(
    "The full embedded profile MUST be produced for exactly these targets" in contract
    and "The explicit model-free lite profile MUST be produced for exactly these targets"
    in contract
    and "`profile` (`embedded` or `lite`)" in contract,
    "release contract must publish truthful full/lite target and capability matrices",
)
require(
    "typed nonzero error" in contract
    and "Status MUST NOT claim that a verified cache is loadable" in contract
    and "`fsfs doctor` MUST" in contract,
    "release contract must distinguish manifest status from loader-probed readiness",
)
require(
    re.search(
        r"Downloading model files alone MUST NOT be\s+presented as enabling semantic execution in this deliberately stripped binary",
        contract,
    ) is not None,
    "release contract must preserve the explicit lite capability boundary",
)
require(
    "## Quick Start (60 Seconds)" not in root_readme
    and "fsfs download-models potion-multilingual-128m" in root_readme
    and "fsfs download-models all-minilm-l6-v2 --verify" in root_readme
    and root_readme.index("fsfs download-models potion-multilingual-128m")
    < root_readme.index("fsfs index ./my-project"),
    "root Quick Start must provision and verify both semantic tiers before indexing",
)
require(
    "pub fn write_verification_marker" not in embed_manifest
    and "write_verification_marker" not in embed_lib
    and "pub fn verify_dir_and_record" in embed_manifest
    and embed_manifest.count("write_verification_marker_atomic(") == 2,
    "only the full-SHA verification boundary may call the private atomic receipt writer",
)
require(
    "verify_dir_and_record(self, staged_dir)?" in embed_manifest
    and "verify_dir_and_record(manifest, model_dir)" in bundled_models
    and "verify_dir_and_record(&manifest, &destination)?" in runtime,
    "explicit verification, staged download promotion, and bundled promotion must mint through one boundary",
)
collect_model_statuses = runtime[
    runtime.index("fn collect_model_statuses") : runtime.index("fn collect_model_status(")
]
require(
    "ensure_default_semantic_models" not in collect_model_statuses
    and "materialize_bundled_default_models" not in auto_detect
    and ".fsfs_doctor_probe" not in runtime
    and "effective writability is unknown/not tested" in runtime,
    "auto-detect, status, and doctor must remain observational",
)
require(
    "fn prepare_bundled_semantic_models_for_execution" in runtime
    and "self.prepare_bundled_semantic_models_for_execution()?;" in runtime
    and "bundled_semantic_models_status_and_doctor_are_observational" in runtime
    and "bundled_semantic_models_materialize_once_at_execution" in runtime,
    "semantic execution must own bundled preparation with observer and idempotence tests",
)

if failures:
    for failure in failures:
        print(f"[model-features][FAIL] {failure}")
    raise SystemExit(1)

print("[model-features][OK] source defaults compile loaders without embedding model artifacts")
PY
  then
    :
  else
    FAILURES=$((FAILURES + 1))
  fi

  check_installer_behavior

  local default_tree lite_tree all_features_tree
  if ! default_tree="$(
    cargo tree --locked -p frankensearch-fsfs --edges normal --prefix none
  )"; then
    echo "[model-features][FAIL] cargo tree failed for the default profile"
    FAILURES=$((FAILURES + 1))
    return
  fi
  if ! lite_tree="$(
    cargo tree --locked -p frankensearch-fsfs --no-default-features --edges normal --prefix none
  )"; then
    echo "[model-features][FAIL] cargo tree failed for the explicit lite profile"
    FAILURES=$((FAILURES + 1))
    return
  fi
  if ! all_features_tree="$(
    cargo tree --locked -p frankensearch-fsfs --all-features --edges normal --prefix none
  )"; then
    echo "[model-features][FAIL] cargo tree failed for the all-features profile"
    FAILURES=$((FAILURES + 1))
    return
  fi

  local package
  for package in fastembed safetensors tokenizers; do
    if ! grep -Eq "^${package} v" <<<"$default_tree"; then
      echo "[model-features][FAIL] default profile lacks required loader package: $package"
      FAILURES=$((FAILURES + 1))
    fi
    if grep -Eq "^${package} v" <<<"$lite_tree"; then
      echo "[model-features][FAIL] explicit lite profile contains loader package: $package"
      FAILURES=$((FAILURES + 1))
    fi
  done

  local lane tree forbidden
  for lane in default lite all-features; do
    case "$lane" in
      default) tree="$default_tree" ;;
      lite) tree="$lite_tree" ;;
      all-features) tree="$all_features_tree" ;;
    esac
    for forbidden in tokio hyper reqwest axum tower async-std smol; do
      if grep -Eq "^${forbidden}([ -][^ ]*)? v" <<<"$tree"; then
        echo "[model-features][FAIL] $lane profile contains forbidden async package family: $forbidden"
        FAILURES=$((FAILURES + 1))
      fi
    done
  done

  if ((FAILURES == 0)); then
    echo "[model-features][OK] default has both real loaders; lite omits them; all profiles exclude forbidden async stacks"
  fi
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  check_unit
fi
if [[ "$MODE" == "integration" || "$MODE" == "all" ]]; then
  check_integration
fi
if [[ "$MODE" == "e2e" || "$MODE" == "all" ]]; then
  check_e2e
fi
if [[ "$MODE" == "installer" || "$MODE" == "all" || "$MODE" == "model-features" ]]; then
  # The --locked callers downstream (this script's cargo-tree profile audit,
  # and the CI installer-platform build step that runs after `--mode installer`)
  # need a current lockfile, but Cargo.lock is deliberately untracked
  # (.gitignore): fresh checkouts and CI runners have none, and a dev tree's
  # local copy can be stale after manifest changes — either way --locked fails
  # with "cannot create/update the lock file". Refresh it here (the
  # consumer-smoke idiom in ci.yml) so every --locked caller in the job
  # resolves against this one snapshot.
  (cd "$ROOT_DIR" && cargo generate-lockfile)
fi
if [[ "$MODE" == "installer" || "$MODE" == "all" ]]; then
  check_installer_behavior
fi
if [[ "$MODE" == "model-features" || "$MODE" == "all" ]]; then
  check_model_features
fi

if ((FAILURES > 0)); then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
