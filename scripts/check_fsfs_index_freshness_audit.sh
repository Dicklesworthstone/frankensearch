#!/usr/bin/env bash
set -euo pipefail

MODE="all"
RUN_ID="bd-pkl0.2-index-freshness"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA_VERSION="fsfs-index-freshness-audit-v1"

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_index_freshness_audit.sh [--mode unit|integration|e2e|all] [--run-id <id>]

Validates the fsfs index freshness audit contract for bd-pkl0.2.
Emits structured JSONL progress events to stdout. Set FSFS_AUDIT_USE_RCH=1
to run cargo checks through rch.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
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

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|"")
    echo "ERROR: --run-id must contain only letters, digits, '.', '_', or '-'" >&2
    exit 2
    ;;
esac

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/frankensearch-fsfs-index-freshness-audit-target}"
REPLAY_COMMAND="scripts/check_fsfs_index_freshness_audit.sh --mode e2e --run-id ${RUN_ID}"
AUDIT_JSONL="runs/${RUN_ID}/index_freshness/audit-events.jsonl"
SUMMARY_JSON="runs/${RUN_ID}/index_freshness/summary.json"

emit_event() {
  local phase="$1"
  local status="$2"
  printf '{"schema_version":"%s","run_id":"%s","phase":"%s","status":"%s","audit_jsonl":"%s","summary_json":"%s","replay_command":"%s"}\n' \
    "$SCHEMA_VERSION" "$RUN_ID" "$phase" "$status" "$AUDIT_JSONL" "$SUMMARY_JSON" "$REPLAY_COMMAND"
}

run_cargo() {
  if [[ "${FSFS_AUDIT_USE_RCH:-0}" == "1" ]]; then
    RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- cargo "$@"
  else
    cargo "$@"
  fi
}

check_unit() {
  emit_event "unit" "start"
  (
    cd "$ROOT_DIR"
    run_cargo test -p frankensearch-fsfs --lib index_freshness_audit -- --nocapture
  )
  emit_event "unit" "pass"
}

check_integration() {
  emit_event "integration" "start"
  (
    cd "$ROOT_DIR"
    run_cargo test -p frankensearch-fsfs --test schema_conformance test_index_freshness_audit_report_conformance -- --exact --nocapture
  )
  emit_event "integration" "pass"
}

check_e2e() {
  emit_event "e2e" "start"
  (
    cd "$ROOT_DIR"
    run_cargo test -p frankensearch-fsfs --test schema_conformance test_index_freshness_audit_report_conformance -- --exact --nocapture
  )
  emit_event "e2e" "pass"
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

emit_event "summary" "pass"
