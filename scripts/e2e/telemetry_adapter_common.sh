#!/usr/bin/env bash
set -euo pipefail

readonly TELEMETRY_ADAPTER_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly TELEMETRY_ADAPTER_LOG_ROOT="${TELEMETRY_ADAPTER_REPO_ROOT}/test_logs/telemetry_adapters"

TELEMETRY_ADAPTER_FINALIZED=0
TELEMETRY_ADAPTER_RUN_STATUS="ok"
TELEMETRY_ADAPTER_RUN_MESSAGE=""
TELEMETRY_ADAPTER_RUN_REASON_CODE="telemetry_adapter.session.ok"
TELEMETRY_ADAPTER_LAST_FAILURE_STAGE=""
TELEMETRY_ADAPTER_LAST_FAILURE_EXIT_CODE=""
TELEMETRY_ADAPTER_EXECUTION_MODE="live"

telemetry_adapter_escape_json() {
  local value="${1:-}"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  printf "%s" "$value"
}

telemetry_adapter_now_iso() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

telemetry_adapter_emit_event() {
  local stage="$1"
  local status="$2"
  local reason_code="$3"
  local detail="$4"
  printf '{"v":1,"schema":"telemetry-adapter-e2e-event-v1","run_id":"%s","ts":"%s","body":{"host":"%s","mode":"%s","stage":"%s","status":"%s","reason_code":"%s","detail":"%s"}}\n' \
    "$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_RUN_ID")" \
    "$(telemetry_adapter_now_iso)" \
    "$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_HOST")" \
    "$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_MODE")" \
    "$(telemetry_adapter_escape_json "$stage")" \
    "$(telemetry_adapter_escape_json "$status")" \
    "$(telemetry_adapter_escape_json "$reason_code")" \
    "$(telemetry_adapter_escape_json "$detail")" \
    >>"$TELEMETRY_ADAPTER_EVENTS_JSONL"
}

telemetry_adapter_require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found in PATH: $cmd" >&2
    exit 2
  fi
}

telemetry_adapter_set_execution_mode() {
  local mode="${1:-live}"
  case "$mode" in
    live|dry) TELEMETRY_ADAPTER_EXECUTION_MODE="$mode" ;;
    *)
      echo "ERROR: invalid execution mode '$mode' (expected live|dry)" >&2
      exit 2
      ;;
  esac
}

telemetry_adapter_is_dry() {
  [[ "$TELEMETRY_ADAPTER_EXECUTION_MODE" == "dry" ]]
}

telemetry_adapter_init() {
  local host="$1"
  local mode="$2"
  local replay_command="$3"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"

  TELEMETRY_ADAPTER_HOST="$host"
  TELEMETRY_ADAPTER_MODE="$mode"
  TELEMETRY_ADAPTER_RUN_ID="${host}-${mode}-${stamp}"
  TELEMETRY_ADAPTER_RUN_DIR="${TELEMETRY_ADAPTER_LOG_ROOT}/${TELEMETRY_ADAPTER_RUN_ID}"
  TELEMETRY_ADAPTER_EVENTS_JSONL="${TELEMETRY_ADAPTER_RUN_DIR}/structured_events.jsonl"
  TELEMETRY_ADAPTER_TRANSCRIPT_TXT="${TELEMETRY_ADAPTER_RUN_DIR}/terminal_transcript.txt"
  TELEMETRY_ADAPTER_REPLAY_TXT="${TELEMETRY_ADAPTER_RUN_DIR}/replay_command.txt"
  TELEMETRY_ADAPTER_SUMMARY_JSON="${TELEMETRY_ADAPTER_RUN_DIR}/summary.json"
  TELEMETRY_ADAPTER_SUMMARY_MD="${TELEMETRY_ADAPTER_RUN_DIR}/summary.md"
  TELEMETRY_ADAPTER_MANIFEST_JSON="${TELEMETRY_ADAPTER_RUN_DIR}/manifest.json"
  TELEMETRY_ADAPTER_CARGO_TARGET_DIR="${TELEMETRY_ADAPTER_RUN_DIR}/cargo-target"

  mkdir -p "$TELEMETRY_ADAPTER_RUN_DIR"
  mkdir -p "$TELEMETRY_ADAPTER_CARGO_TARGET_DIR"

  : >"$TELEMETRY_ADAPTER_EVENTS_JSONL"
  : >"$TELEMETRY_ADAPTER_TRANSCRIPT_TXT"
  printf "%s\n" "$replay_command" >"$TELEMETRY_ADAPTER_REPLAY_TXT"

  telemetry_adapter_emit_event \
    "session.init" \
    "ok" \
    "telemetry_adapter.session.init" \
    "initialized telemetry adapter run directory (execution_mode=${TELEMETRY_ADAPTER_EXECUTION_MODE})"
}

telemetry_adapter_record_command_header() {
  local repo="$1"
  shift
  local -a cmd=("$@")
  {
    printf '$ (cd %q && CARGO_TARGET_DIR=%q rch exec --' "$repo" "$TELEMETRY_ADAPTER_CARGO_TARGET_DIR"
    printf ' %q' "${cmd[@]}"
    printf ')\n'
  } >>"$TELEMETRY_ADAPTER_TRANSCRIPT_TXT"
}

telemetry_adapter_run_rch_cargo() {
  local stage="$1"
  local repo="$2"
  shift 2
  local -a cargo_cmd=("$@")
  local start_ts
  local end_ts
  local duration_s
  local exit_code

  telemetry_adapter_emit_event \
    "$stage" \
    "started" \
    "telemetry_adapter.stage.started" \
    "running cargo command through rch (execution_mode=${TELEMETRY_ADAPTER_EXECUTION_MODE})"
  telemetry_adapter_record_command_header "$repo" "${cargo_cmd[@]}"

  if telemetry_adapter_is_dry; then
    telemetry_adapter_emit_event \
      "$stage" \
      "ok" \
      "telemetry_adapter.stage.skipped_dry_run" \
      "dry-run: skipped command execution"
    printf '[%s] dry-run: skipped command execution\n' "$stage" >>"$TELEMETRY_ADAPTER_TRANSCRIPT_TXT"
    return 0
  fi

  start_ts="$(date +%s)"
  if (
    cd "$repo" &&
    CARGO_TARGET_DIR="$TELEMETRY_ADAPTER_CARGO_TARGET_DIR" rch exec -- "${cargo_cmd[@]}"
  ) >>"$TELEMETRY_ADAPTER_TRANSCRIPT_TXT" 2>&1; then
    end_ts="$(date +%s)"
    duration_s=$((end_ts - start_ts))
    telemetry_adapter_emit_event \
      "$stage" \
      "ok" \
      "telemetry_adapter.stage.ok" \
      "completed in ${duration_s}s"
    return 0
  fi

  exit_code=$?
  end_ts="$(date +%s)"
  duration_s=$((end_ts - start_ts))
  TELEMETRY_ADAPTER_LAST_FAILURE_STAGE="$stage"
  TELEMETRY_ADAPTER_LAST_FAILURE_EXIT_CODE="$exit_code"
  TELEMETRY_ADAPTER_RUN_STATUS="fail"
  TELEMETRY_ADAPTER_RUN_REASON_CODE="telemetry_adapter.stage.failed"
  TELEMETRY_ADAPTER_RUN_MESSAGE="stage ${stage} failed (exit_code=${exit_code}, duration_s=${duration_s})"
  telemetry_adapter_emit_event \
    "$stage" \
    "fail" \
    "telemetry_adapter.stage.failed" \
    "exit_code=${exit_code} duration_s=${duration_s}"
  return "$exit_code"
}

telemetry_adapter_set_status() {
  local status="$1"
  local message="$2"
  local reason_code="$3"
  TELEMETRY_ADAPTER_RUN_STATUS="$status"
  TELEMETRY_ADAPTER_RUN_MESSAGE="$message"
  TELEMETRY_ADAPTER_RUN_REASON_CODE="$reason_code"
}

telemetry_adapter_write_manifest() {
  local status="$1"
  local reason_code="$2"
  cat >"$TELEMETRY_ADAPTER_MANIFEST_JSON" <<EOF_MANIFEST
{"schema":"telemetry-adapter-e2e-manifest-v1","v":1,"host":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_HOST")","mode":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_MODE")","execution_mode":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_EXECUTION_MODE")","run_id":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_RUN_ID")","status":"$(telemetry_adapter_escape_json "$status")","reason_code":"$(telemetry_adapter_escape_json "$reason_code")","ts":"$(telemetry_adapter_now_iso)","artifacts":{"events":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_EVENTS_JSONL")","transcript":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_TRANSCRIPT_TXT")","replay":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_REPLAY_TXT")","summary_json":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_SUMMARY_JSON")","summary_md":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_SUMMARY_MD")"}}
EOF_MANIFEST
}

telemetry_adapter_finalize() {
  local status="$1"
  local message="$2"
  local reason_code="$3"

  TELEMETRY_ADAPTER_FINALIZED=1

  cat >"$TELEMETRY_ADAPTER_SUMMARY_JSON" <<EOF_SUMMARY_JSON
{"host":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_HOST")","mode":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_MODE")","execution_mode":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_EXECUTION_MODE")","run_id":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_RUN_ID")","status":"$(telemetry_adapter_escape_json "$status")","reason_code":"$(telemetry_adapter_escape_json "$reason_code")","message":"$(telemetry_adapter_escape_json "$message")","events":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_EVENTS_JSONL")","transcript":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_TRANSCRIPT_TXT")","replay":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_REPLAY_TXT")","summary_md":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_SUMMARY_MD")","manifest":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_MANIFEST_JSON")","ts":"$(telemetry_adapter_now_iso)"}
EOF_SUMMARY_JSON

  cat >"$TELEMETRY_ADAPTER_SUMMARY_MD" <<EOF_SUMMARY_MD
# Telemetry Adapter Lane Summary

- host: ${TELEMETRY_ADAPTER_HOST}
- mode: ${TELEMETRY_ADAPTER_MODE}
- execution_mode: ${TELEMETRY_ADAPTER_EXECUTION_MODE}
- run_id: ${TELEMETRY_ADAPTER_RUN_ID}
- status: ${status}
- reason_code: ${reason_code}
- message: ${message}
- events: ${TELEMETRY_ADAPTER_EVENTS_JSONL}
- transcript: ${TELEMETRY_ADAPTER_TRANSCRIPT_TXT}
- replay: ${TELEMETRY_ADAPTER_REPLAY_TXT}
- ts: $(telemetry_adapter_now_iso)
EOF_SUMMARY_MD

  telemetry_adapter_write_manifest "$status" "$reason_code"
  telemetry_adapter_emit_event "session.finalize" "$status" "$reason_code" "$message"

  echo "Artifacts:"
  echo "  execution:  $TELEMETRY_ADAPTER_EXECUTION_MODE"
  echo "  run_dir:    $TELEMETRY_ADAPTER_RUN_DIR"
  echo "  events:     $TELEMETRY_ADAPTER_EVENTS_JSONL"
  echo "  transcript: $TELEMETRY_ADAPTER_TRANSCRIPT_TXT"
  echo "  replay:     $TELEMETRY_ADAPTER_REPLAY_TXT"
  echo "  summary_md: $TELEMETRY_ADAPTER_SUMMARY_MD"
  echo "  manifest:   $TELEMETRY_ADAPTER_MANIFEST_JSON"
  echo "  summary:    $TELEMETRY_ADAPTER_SUMMARY_JSON"
}

telemetry_adapter_on_exit() {
  local exit_code="$1"

  if [[ "$TELEMETRY_ADAPTER_FINALIZED" -eq 1 ]]; then
    return 0
  fi

  if [[ "$exit_code" -eq 0 && "$TELEMETRY_ADAPTER_RUN_STATUS" == "ok" ]]; then
    local ok_message="${TELEMETRY_ADAPTER_RUN_MESSAGE:-telemetry adapter lane passed}"
    telemetry_adapter_finalize "ok" "$ok_message" "$TELEMETRY_ADAPTER_RUN_REASON_CODE"
    return 0
  fi

  local failed_stage="${TELEMETRY_ADAPTER_LAST_FAILURE_STAGE:-unknown-stage}"
  local failed_exit="${TELEMETRY_ADAPTER_LAST_FAILURE_EXIT_CODE:-$exit_code}"
  local fail_message="${TELEMETRY_ADAPTER_RUN_MESSAGE:-stage ${failed_stage} failed (exit_code=${failed_exit})}"
  telemetry_adapter_finalize "fail" "$fail_message" "$TELEMETRY_ADAPTER_RUN_REASON_CODE"
}

telemetry_adapter_install_exit_trap() {
  trap 'telemetry_adapter_on_exit "$?"' EXIT
}
