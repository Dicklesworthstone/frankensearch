#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA="$ROOT_DIR/schemas/fsfs-packaging-release-install-v1.schema.json"
FAILURES=0

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_packaging_release_install_contract.sh [--mode unit|integration|e2e|model-features|all]

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
  unit|integration|e2e|model-features|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|integration|e2e|model-features|all)" >&2
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

check_model_features() {
  echo "[model-features] validating model-free defaults and explicit embedded release lane"
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
contract_path = root / "docs/fsfs-packaging-release-install-contract.md"

with manifest_path.open("rb") as handle:
    manifest = tomllib.load(handle)

failures: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


features = manifest.get("features", {})
require(features.get("default") == [], "frankensearch-fsfs default features must be empty")
require(
    features.get("embedded-models")
    == [
        "frankensearch-embed/bundled-default-models",
        "frankensearch-embed/model2vec",
        "frankensearch-embed/fastembed",
    ],
    "embedded-models must select the pinned bundled model implementation features",
)
embed_dependency = manifest.get("dependencies", {}).get("frankensearch-embed", {})
require(
    embed_dependency.get("default-features") is False
    and embed_dependency.get("features") == ["hash", "download"],
    "model-free fsfs dependency must expose only hash-control and model acquisition",
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
release = job("release-build")
lite = job("release-build-lite")
publish = job("publish-crates")
quality_lines = {line.strip() for line in quality.splitlines()}
release_lines = {line.strip() for line in release.splitlines()}


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
        "quality must compile model-free defaults before provisioning embedded inputs",
    )

for command in (provision, verify, embedded_build):
    require(has_command(release_lines, command), f"full release job missing command: {command}")
if all(command in release for command in (provision, verify, embedded_build)):
    require(
        release.index(provision) < release.index(verify) < release.index(embedded_build),
        "full release must provision, verify, then build embedded models",
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

for forbidden in ("--models-only", "embedded-models", "FRANKENSEARCH_BUNDLED_MODELS_SOURCE_DIR"):
    require(forbidden not in publish, f"publish-crates must not require model input: {forbidden}")

installer = installer_path.read_text(encoding="utf-8")
crate_readme = crate_readme_path.read_text(encoding="utf-8")
contract = contract_path.read_text(encoding="utf-8")
require(
    "cargo build --release -p frankensearch-fsfs --no-default-features" in installer,
    "installer --lite source lane must remain explicitly model-free",
)
require(
    "cargo build --release -p frankensearch-fsfs)" in installer,
    "ordinary source installation must exercise the model-free Cargo default",
)
require(
    re.search(r"cannot\s+execute Model2Vec or\s+FastEmbed", crate_readme) is not None,
    "crate README must state that downloaded files do not add uncompiled model backends",
)
require(
    "Downloaded files alone do not add those compiled capabilities." in installer,
    "installer must not claim that model acquisition activates a model-free binary",
)
require(
    re.search(
        r"Downloading model files\s+alone MUST NOT be presented as enabling semantic execution",
        contract,
    )
    is not None,
    "release contract must forbid download-only semantic activation claims",
)

if failures:
    for failure in failures:
        print(f"[model-features][FAIL] {failure}")
    raise SystemExit(1)

print("[model-features][OK] model artifact admission is explicit and release-scoped")
PY
  then
    :
  else
    FAILURES=$((FAILURES + 1))
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
if [[ "$MODE" == "model-features" || "$MODE" == "all" ]]; then
  check_model_features
fi

if ((FAILURES > 0)); then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
