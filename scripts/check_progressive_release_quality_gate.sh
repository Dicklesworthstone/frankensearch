#!/usr/bin/env bash
set -euo pipefail

MODE="all"
RUN_ID="bd-pkl0.3-progressive-release-gate"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEMA_VERSION="fsfs-progressive-release-quality-gate-v1"
ARTIFACT_ROOT="${FSFS_RELEASE_GATE_ARTIFACT_ROOT:-/tmp/frankensearch-progressive-release-gate/${RUN_ID}}"
EVENTS_JSONL="${ARTIFACT_ROOT}/gate-events.jsonl"
SUMMARY_JSON="${ARTIFACT_ROOT}/summary.json"
SUMMARY_MD="${ARTIFACT_ROOT}/summary.md"
REPLAY_COMMAND="scripts/check_progressive_release_quality_gate.sh --mode all --run-id ${RUN_ID}"

usage() {
  cat <<USAGE
Usage: scripts/check_progressive_release_quality_gate.sh [--mode unit|integration|features|quality|e2e|all] [--run-id <id>]

Validates the progressive release quality gate pack for bd-pkl0.3.
Cargo work uses rch by default. Set FSFS_RELEASE_GATE_USE_RCH=0 only for
explicit local debugging.
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
      ARTIFACT_ROOT="${FSFS_RELEASE_GATE_ARTIFACT_ROOT:-/tmp/frankensearch-progressive-release-gate/${RUN_ID}}"
      EVENTS_JSONL="${ARTIFACT_ROOT}/gate-events.jsonl"
      SUMMARY_JSON="${ARTIFACT_ROOT}/summary.json"
      SUMMARY_MD="${ARTIFACT_ROOT}/summary.md"
      REPLAY_COMMAND="scripts/check_progressive_release_quality_gate.sh --mode all --run-id ${RUN_ID}"
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
  unit|integration|features|quality|e2e|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|integration|features|quality|e2e|all)" >&2
    exit 2
    ;;
esac

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|"")
    echo "ERROR: --run-id must contain only letters, digits, '.', '_', or '-'" >&2
    exit 2
    ;;
esac

mkdir -p "$ARTIFACT_ROOT"
: >"$EVENTS_JSONL"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/rch_target_frankensearch_${AGENT_NAME:-agent}_bd_pkl0_3_release_gate}"

emit_event() {
  local phase="$1"
  local status="$2"
  local reason_code="$3"
  local line
  line="$(printf '{"schema_version":"%s","run_id":"%s","phase":"%s","status":"%s","reason_code":"%s","events_jsonl":"%s","summary_json":"%s","summary_md":"%s","replay_command":"%s"}' \
    "$SCHEMA_VERSION" "$RUN_ID" "$phase" "$status" "$reason_code" "$EVENTS_JSONL" "$SUMMARY_JSON" "$SUMMARY_MD" "$REPLAY_COMMAND")"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >>"$EVENTS_JSONL"
}

run_cargo() {
  if [[ "${FSFS_RELEASE_GATE_USE_RCH:-1}" == "1" ]]; then
    RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- cargo "$@"
  else
    cargo "$@"
  fi
}

write_summary() {
  local status="$1"
  cat >"$SUMMARY_JSON" <<JSON
{"schema_version":"${SCHEMA_VERSION}","run_id":"${RUN_ID}","status":"${status}","events_jsonl":"${EVENTS_JSONL}","summary_md":"${SUMMARY_MD}","replay_command":"${REPLAY_COMMAND}"}
JSON
  cat >"$SUMMARY_MD" <<MD
# Progressive Release Quality Gate

- run_id: ${RUN_ID}
- status: ${status}
- events_jsonl: ${EVENTS_JSONL}
- summary_json: ${SUMMARY_JSON}
- replay_command: ${REPLAY_COMMAND}

## Feature Lanes

- hash-only: frankensearch --features hash
- hybrid: frankensearch --features hybrid
- durable: frankensearch --features durable
- full: frankensearch --features full
MD
}

check_unit() {
  emit_event "unit" "start" "RELEASE_GATE_UNIT_START"
  (
    cd "$ROOT_DIR"
    run_cargo test -p frankensearch-fsfs --lib progressive_release_quality_gate -- --nocapture
  )
  emit_event "unit" "pass" "RELEASE_GATE_UNIT_PASS"
}

check_integration() {
  emit_event "integration" "start" "RELEASE_GATE_INTEGRATION_START"
  (
    cd "$ROOT_DIR"
    run_cargo test -p frankensearch-fsfs --test schema_conformance test_progressive_release_quality_gate_pack_conformance -- --exact --nocapture
  )
  emit_event "integration" "pass" "RELEASE_GATE_INTEGRATION_PASS"
}

check_feature_lane() {
  local lane="$1"
  local features="$2"
  emit_event "features:${lane}" "start" "RELEASE_GATE_FEATURE_LANE_START"
  (
    cd "$ROOT_DIR"
    run_cargo check -p frankensearch --lib --no-default-features --features "$features"
  )
  emit_event "features:${lane}" "pass" "RELEASE_GATE_FEATURE_LANE_PASS"
}

check_features() {
  check_feature_lane "hash-only" "hash"
  check_feature_lane "hybrid" "hybrid"
  check_feature_lane "durable" "durable"
  check_feature_lane "full" "full"
}

check_quality() {
  emit_event "quality" "start" "RELEASE_GATE_QUALITY_START"
  (
    cd "$ROOT_DIR"
    run_cargo test -p frankensearch-fsfs --test search_quality_harness quality_harness_reports_metrics_by_query_slice -- --exact --nocapture
  )
  emit_event "quality" "pass" "RELEASE_GATE_QUALITY_PASS"
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  check_unit
fi
if [[ "$MODE" == "integration" || "$MODE" == "all" ]]; then
  check_integration
fi
if [[ "$MODE" == "features" || "$MODE" == "all" ]]; then
  check_features
fi
if [[ "$MODE" == "quality" || "$MODE" == "e2e" || "$MODE" == "all" ]]; then
  check_quality
fi

write_summary "pass"
emit_event "summary" "pass" "RELEASE_GATE_SUMMARY_PASS"
