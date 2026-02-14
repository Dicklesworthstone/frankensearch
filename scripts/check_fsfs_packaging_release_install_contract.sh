#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA="$ROOT_DIR/schemas/fsfs-packaging-release-install-v1.schema.json"
FAILURES=0

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_packaging_release_install_contract.sh [--mode unit|integration|e2e|all]

Validates fsfs packaging/release/install fixtures for bd-2hz.11.1.
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
  unit|integration|e2e|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|integration|e2e|all)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$SCHEMA" ]]; then
  echo "ERROR: schema not found: $SCHEMA" >&2
  exit 2
fi

if ! command -v jsonschema >/dev/null 2>&1; then
  echo "ERROR: jsonschema CLI not found in PATH" >&2
  exit 2
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

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  check_unit
fi
if [[ "$MODE" == "integration" || "$MODE" == "all" ]]; then
  check_integration
fi
if [[ "$MODE" == "e2e" || "$MODE" == "all" ]]; then
  check_e2e
fi

if ((FAILURES > 0)); then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
