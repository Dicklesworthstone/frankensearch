#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA="$ROOT_DIR/schemas/fsfs-model-cache-diagnostics-v1.schema.json"
FAILURES=0

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_model_cache_diagnostics.sh [--mode unit|warm|cold|offline|all]

Validates deterministic fsfs model-cache diagnostics fixtures for bd-pkl0.11.
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
  unit|warm|cold|offline|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|warm|cold|offline|all)" >&2
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
  echo "[unit] validating model-cache diagnostics contract"
  check_valid "unit" "$ROOT_DIR/schemas/fixtures/fsfs-model-cache-diagnostics-contract-v1.json"
}

check_warm() {
  echo "[warm] validating ready quality-model cache diagnostics"
  check_valid "warm" "$ROOT_DIR/schemas/fixtures/fsfs-model-cache-diagnostics-warm-v1.json"
}

check_cold() {
  echo "[cold] validating cold-start warmup diagnostics"
  check_valid "cold" "$ROOT_DIR/schemas/fixtures/fsfs-model-cache-diagnostics-cold-v1.json"
}

check_offline() {
  echo "[offline] validating offline fallback and path-redaction failures"
  check_valid "offline" "$ROOT_DIR/schemas/fixtures/fsfs-model-cache-diagnostics-missing-offline-v1.json"
  check_invalid "offline" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-model-cache-diagnostics-invalid-raw-path-v1.json"
  check_invalid "offline" "$ROOT_DIR/schemas/fixtures-invalid/fsfs-model-cache-diagnostics-invalid-missing-advice-v1.json"
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  check_unit
fi
if [[ "$MODE" == "warm" || "$MODE" == "all" ]]; then
  check_warm
fi
if [[ "$MODE" == "cold" || "$MODE" == "all" ]]; then
  check_cold
fi
if [[ "$MODE" == "offline" || "$MODE" == "all" ]]; then
  check_offline
fi

if ((FAILURES > 0)); then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
