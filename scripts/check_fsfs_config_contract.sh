#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA="$ROOT_DIR/schemas/fsfs-config-v1.schema.json"
FAILURES=0

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_config_contract.sh [--mode unit|integration|e2e|all]

Validates fsfs config contract fixtures for bd-2hz.13.
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

# jsonschema tool resolution: the standalone `jsonschema` CLI was removed in
# python-jsonschema 4.x, so modern runners ship only the module. Accept either.
if command -v jsonschema >/dev/null 2>&1; then
  JSONSCHEMA_MODE="cli"
elif python3 -c 'import jsonschema' >/dev/null 2>&1; then
  JSONSCHEMA_MODE="module"
else
  echo "ERROR: no jsonschema validator found: install the python3 jsonschema module (python3 -m pip install jsonschema)" >&2
  exit 2
fi

jsonschema_validate() {
  local file="$1"
  if [[ "${JSONSCHEMA_MODE}" == "cli" ]]; then
    jsonschema -i "$file" "$SCHEMA" >/dev/null 2>&1
  else
    python3 - "$file" "$SCHEMA" >/dev/null 2>&1 <<'PY'
import json
import sys

import jsonschema

with open(sys.argv[1], encoding="utf-8") as handle:
    instance = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    schema = json.load(handle)
jsonschema.validate(instance, schema)
PY
  fi
}

check_valid() {
  local scope="$1"
  local file="$2"
  if jsonschema_validate "$file"; then
    echo "[$scope][OK]   valid fixture accepted: $file"
  else
    echo "[$scope][FAIL] valid fixture rejected: $file"
    FAILURES=$((FAILURES + 1))
  fi
}

check_invalid() {
  local scope="$1"
  local file="$2"
  if jsonschema_validate "$file"; then
    echo "[$scope][FAIL] invalid fixture unexpectedly accepted: $file"
    FAILURES=$((FAILURES + 1))
  else
    echo "[$scope][OK]   invalid fixture rejected: $file"
  fi
}

check_unit() {
  echo "[unit] validating config contract precedence, default safety, and unknown-key warning policy"
  check_valid "unit" "$ROOT_DIR/schemas/fixtures/fsfs-config-contract-v1.json"
  check_invalid "unit" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-config-invalid-precedence-v1.json"
  check_invalid "unit" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-config-invalid-unknown-policy-v1.json"
  check_invalid "unit" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-config-invalid-default-redaction-v1.json"
}

check_integration() {
  echo "[integration] validating effective config resolution, path expansion, and conflict warning semantics"
  check_valid "integration" "$ROOT_DIR/schemas/fixtures/fsfs-config-effective-v1.json"
  check_invalid "integration" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-config-invalid-fast-only-missing-warning-v1.json"
}

check_e2e() {
  echo "[e2e] validating config_loaded telemetry payload and reason-code diagnostics"
  check_valid "e2e" "$ROOT_DIR/schemas/fixtures/fsfs-config-load-event-v1.json"
  check_invalid "e2e" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-config-invalid-load-event-missing-reason-code-v1.json"
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
