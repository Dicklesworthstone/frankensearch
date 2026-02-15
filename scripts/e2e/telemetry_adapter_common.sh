#!/usr/bin/env bash
set -euo pipefail

TELEMETRY_ADAPTER_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly TELEMETRY_ADAPTER_REPO_ROOT
readonly TELEMETRY_ADAPTER_LOG_ROOT="${TELEMETRY_ADAPTER_REPO_ROOT}/test_logs/telemetry_adapters"

TELEMETRY_ADAPTER_FINALIZED=0
TELEMETRY_ADAPTER_RUN_STATUS="ok"
TELEMETRY_ADAPTER_RUN_MESSAGE=""
TELEMETRY_ADAPTER_RUN_REASON_CODE="telemetry_adapter.session.ok"
TELEMETRY_ADAPTER_LAST_FAILURE_STAGE=""
TELEMETRY_ADAPTER_LAST_FAILURE_EXIT_CODE=""
TELEMETRY_ADAPTER_ACTIVE_STAGE=""
TELEMETRY_ADAPTER_EXECUTION_MODE="live"
TELEMETRY_ADAPTER_STAGE_STARTED_COUNT=0
TELEMETRY_ADAPTER_STAGE_COMPLETED_COUNT=0
TELEMETRY_ADAPTER_RCH_TIMEOUT_SECS="${TELEMETRY_ADAPTER_RCH_TIMEOUT_SECS:-420}"

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

telemetry_adapter_mark_stage_started() {
  TELEMETRY_ADAPTER_STAGE_STARTED_COUNT=$((TELEMETRY_ADAPTER_STAGE_STARTED_COUNT + 1))
}

telemetry_adapter_mark_stage_completed() {
  TELEMETRY_ADAPTER_STAGE_COMPLETED_COUNT=$((TELEMETRY_ADAPTER_STAGE_COMPLETED_COUNT + 1))
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
  local epoch_ns
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  epoch_ns="$(date -u +%s%N 2>/dev/null || printf "%s000000000" "$(date -u +%s)")"

  TELEMETRY_ADAPTER_HOST="$host"
  TELEMETRY_ADAPTER_MODE="$mode"
  TELEMETRY_ADAPTER_RUN_ID="${host}-${mode}-${stamp}-${epoch_ns}-$$"
  TELEMETRY_ADAPTER_RUN_DIR="${TELEMETRY_ADAPTER_LOG_ROOT}/${TELEMETRY_ADAPTER_RUN_ID}"
  TELEMETRY_ADAPTER_EVENTS_JSONL="${TELEMETRY_ADAPTER_RUN_DIR}/structured_events.jsonl"
  TELEMETRY_ADAPTER_TRANSCRIPT_TXT="${TELEMETRY_ADAPTER_RUN_DIR}/terminal_transcript.txt"
  TELEMETRY_ADAPTER_REPLAY_TXT="${TELEMETRY_ADAPTER_RUN_DIR}/replay_command.txt"
  TELEMETRY_ADAPTER_SUMMARY_JSON="${TELEMETRY_ADAPTER_RUN_DIR}/summary.json"
  TELEMETRY_ADAPTER_SUMMARY_MD="${TELEMETRY_ADAPTER_RUN_DIR}/summary.md"
  TELEMETRY_ADAPTER_MANIFEST_JSON="${TELEMETRY_ADAPTER_RUN_DIR}/manifest.json"
  TELEMETRY_ADAPTER_CARGO_TARGET_DIR="target/telemetry-adapter/${TELEMETRY_ADAPTER_RUN_ID}"

  mkdir -p "$TELEMETRY_ADAPTER_RUN_DIR"

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
    printf '$ (cd %q && CARGO_TARGET_DIR=%q' "$repo" "$TELEMETRY_ADAPTER_CARGO_TARGET_DIR"
    printf ' %q' "${cmd[@]}"
    printf ')\n'
  } >>"$TELEMETRY_ADAPTER_TRANSCRIPT_TXT"
}

telemetry_adapter_classify_stage_failure_reason() {
  local exit_code="$1"
  local transcript_tail
  transcript_tail="$(tail -n 200 "$TELEMETRY_ADAPTER_TRANSCRIPT_TXT" 2>/dev/null || true)"

  if [[ "$exit_code" -eq 124 || "$exit_code" -eq 137 ]]; then
    printf "%s" "telemetry_adapter.stage.timeout"
    return 0
  fi

  if grep -Fq "failed to load manifest for dependency" <<<"$transcript_tail"; then
    printf "%s" "telemetry_adapter.stage.remote_path_dependency_missing"
    return 0
  fi

  if grep -Fq "failed to select a version for the requirement" <<<"$transcript_tail"; then
    printf "%s" "telemetry_adapter.stage.remote_dependency_resolution_failed"
    return 0
  fi

  if grep -Fq "does not contain this feature" <<<"$transcript_tail"; then
    printf "%s" "telemetry_adapter.stage.invalid_feature_flag"
    return 0
  fi

  if grep -Fq "no such command: \`+nightly\`" <<<"$transcript_tail"; then
    printf "%s" "telemetry_adapter.stage.invalid_toolchain_invocation"
    return 0
  fi

  printf "%s" "telemetry_adapter.stage.failed"
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
  local timeout_seconds="${TELEMETRY_ADAPTER_RCH_TIMEOUT_SECS}"
  local timeout_reason=""
  local failure_reason=""
  local -a stage_cmd

  TELEMETRY_ADAPTER_ACTIVE_STAGE="$stage"
  telemetry_adapter_mark_stage_started

  stage_cmd=(rch exec -- "${cargo_cmd[@]}")
  if command -v timeout >/dev/null 2>&1; then
    if [[ "$timeout_seconds" =~ ^[0-9]+$ ]] && (( timeout_seconds > 0 )); then
      stage_cmd=(timeout -s TERM "${timeout_seconds}" "${stage_cmd[@]}")
    fi
  fi

  telemetry_adapter_emit_event \
    "$stage" \
    "started" \
    "telemetry_adapter.stage.started" \
    "running cargo command through rch (execution_mode=${TELEMETRY_ADAPTER_EXECUTION_MODE}, timeout_s=${timeout_seconds})"
  telemetry_adapter_record_command_header "$repo" "${stage_cmd[@]}"

  if telemetry_adapter_is_dry; then
    telemetry_adapter_emit_event \
      "$stage" \
      "ok" \
      "telemetry_adapter.stage.skipped_dry_run" \
      "dry-run: skipped command execution"
    printf '[%s] dry-run: skipped command execution\n' "$stage" >>"$TELEMETRY_ADAPTER_TRANSCRIPT_TXT"
    telemetry_adapter_mark_stage_completed
    TELEMETRY_ADAPTER_ACTIVE_STAGE=""
    return 0
  fi

  start_ts="$(date +%s)"
  if (
    cd "$repo" &&
    CARGO_TARGET_DIR="$TELEMETRY_ADAPTER_CARGO_TARGET_DIR" "${stage_cmd[@]}"
  ) >>"$TELEMETRY_ADAPTER_TRANSCRIPT_TXT" 2>&1; then
    end_ts="$(date +%s)"
    duration_s=$((end_ts - start_ts))
    telemetry_adapter_emit_event \
      "$stage" \
      "ok" \
      "telemetry_adapter.stage.ok" \
      "completed in ${duration_s}s"
    telemetry_adapter_mark_stage_completed
    TELEMETRY_ADAPTER_ACTIVE_STAGE=""
    return 0
  else
    exit_code=$?
  fi

  end_ts="$(date +%s)"
  duration_s=$((end_ts - start_ts))

  if [[ "$exit_code" -eq 124 || "$exit_code" -eq 137 ]]; then
    timeout_reason="telemetry_adapter.stage.timeout"
    TELEMETRY_ADAPTER_LAST_FAILURE_STAGE="$stage"
    TELEMETRY_ADAPTER_LAST_FAILURE_EXIT_CODE="$exit_code"
    TELEMETRY_ADAPTER_RUN_STATUS="fail"
    TELEMETRY_ADAPTER_RUN_REASON_CODE="$timeout_reason"
    TELEMETRY_ADAPTER_RUN_MESSAGE="stage ${stage} timed out (exit_code=${exit_code}, timeout_s=${timeout_seconds}, duration_s=${duration_s})"
    telemetry_adapter_emit_event \
      "$stage" \
      "fail" \
      "$timeout_reason" \
      "exit_code=${exit_code} timeout_s=${timeout_seconds} duration_s=${duration_s}"
    telemetry_adapter_mark_stage_completed
    TELEMETRY_ADAPTER_ACTIVE_STAGE=""
    return "$exit_code"
  fi

  TELEMETRY_ADAPTER_LAST_FAILURE_STAGE="$stage"
  TELEMETRY_ADAPTER_LAST_FAILURE_EXIT_CODE="$exit_code"
  failure_reason="$(telemetry_adapter_classify_stage_failure_reason "$exit_code")"
  TELEMETRY_ADAPTER_RUN_STATUS="fail"
  TELEMETRY_ADAPTER_RUN_REASON_CODE="$failure_reason"
  TELEMETRY_ADAPTER_RUN_MESSAGE="stage ${stage} failed (reason_code=${failure_reason}, exit_code=${exit_code}, duration_s=${duration_s})"
  telemetry_adapter_emit_event \
    "$stage" \
    "fail" \
    "$failure_reason" \
    "reason_code=${failure_reason} exit_code=${exit_code} duration_s=${duration_s}"
  telemetry_adapter_mark_stage_completed
  TELEMETRY_ADAPTER_ACTIVE_STAGE=""
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
{"schema":"telemetry-adapter-e2e-manifest-v1","v":1,"host":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_HOST")","mode":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_MODE")","execution_mode":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_EXECUTION_MODE")","run_id":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_RUN_ID")","status":"$(telemetry_adapter_escape_json "$status")","reason_code":"$(telemetry_adapter_escape_json "$reason_code")","stage_started_count":${TELEMETRY_ADAPTER_STAGE_STARTED_COUNT},"stage_completed_count":${TELEMETRY_ADAPTER_STAGE_COMPLETED_COUNT},"active_stage":"$(telemetry_adapter_escape_json "${TELEMETRY_ADAPTER_ACTIVE_STAGE}")","ts":"$(telemetry_adapter_now_iso)","artifacts":{"events":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_EVENTS_JSONL")","transcript":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_TRANSCRIPT_TXT")","replay":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_REPLAY_TXT")","summary_json":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_SUMMARY_JSON")","summary_md":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_SUMMARY_MD")"}}
EOF_MANIFEST
}

telemetry_adapter_finalize() {
  local status="$1"
  local message="$2"
  local reason_code="$3"

  TELEMETRY_ADAPTER_FINALIZED=1

  cat >"$TELEMETRY_ADAPTER_SUMMARY_JSON" <<EOF_SUMMARY_JSON
{"host":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_HOST")","mode":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_MODE")","execution_mode":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_EXECUTION_MODE")","run_id":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_RUN_ID")","status":"$(telemetry_adapter_escape_json "$status")","reason_code":"$(telemetry_adapter_escape_json "$reason_code")","message":"$(telemetry_adapter_escape_json "$message")","stage_started_count":${TELEMETRY_ADAPTER_STAGE_STARTED_COUNT},"stage_completed_count":${TELEMETRY_ADAPTER_STAGE_COMPLETED_COUNT},"active_stage":"$(telemetry_adapter_escape_json "${TELEMETRY_ADAPTER_ACTIVE_STAGE}")","events":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_EVENTS_JSONL")","transcript":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_TRANSCRIPT_TXT")","replay":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_REPLAY_TXT")","summary_md":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_SUMMARY_MD")","manifest":"$(telemetry_adapter_escape_json "$TELEMETRY_ADAPTER_MANIFEST_JSON")","ts":"$(telemetry_adapter_now_iso)"}
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
- stage_started_count: ${TELEMETRY_ADAPTER_STAGE_STARTED_COUNT}
- stage_completed_count: ${TELEMETRY_ADAPTER_STAGE_COMPLETED_COUNT}
- active_stage: ${TELEMETRY_ADAPTER_ACTIVE_STAGE:-none}
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
    if [[ -n "${TELEMETRY_ADAPTER_ACTIVE_STAGE}" || "$TELEMETRY_ADAPTER_STAGE_STARTED_COUNT" -ne "$TELEMETRY_ADAPTER_STAGE_COMPLETED_COUNT" ]]; then
      local incomplete_stage="${TELEMETRY_ADAPTER_ACTIVE_STAGE:-unknown-stage}"
      local incomplete_message="incomplete stage accounting detected (active_stage=${incomplete_stage}, started=${TELEMETRY_ADAPTER_STAGE_STARTED_COUNT}, completed=${TELEMETRY_ADAPTER_STAGE_COMPLETED_COUNT})"
      telemetry_adapter_finalize "fail" "$incomplete_message" "telemetry_adapter.session.incomplete"
      return 0
    fi
    local ok_message="${TELEMETRY_ADAPTER_RUN_MESSAGE:-telemetry adapter lane passed}"
    telemetry_adapter_finalize "ok" "$ok_message" "$TELEMETRY_ADAPTER_RUN_REASON_CODE"
    return 0
  fi

  local failed_stage="${TELEMETRY_ADAPTER_LAST_FAILURE_STAGE:-${TELEMETRY_ADAPTER_ACTIVE_STAGE:-unknown-stage}}"
  local failed_exit="${TELEMETRY_ADAPTER_LAST_FAILURE_EXIT_CODE:-$exit_code}"
  local fail_reason="${TELEMETRY_ADAPTER_RUN_REASON_CODE}"
  if [[ "$fail_reason" == "telemetry_adapter.session.ok" ]]; then
    fail_reason="telemetry_adapter.session.failed"
  fi
  local fail_message="${TELEMETRY_ADAPTER_RUN_MESSAGE:-stage ${failed_stage} failed (exit_code=${failed_exit})}"
  telemetry_adapter_finalize "fail" "$fail_message" "$fail_reason"
}

telemetry_adapter_on_signal() {
  local signal_name="$1"
  local signal_exit_code="$2"
  local interrupted_stage="${TELEMETRY_ADAPTER_ACTIVE_STAGE:-unknown-stage}"

  TELEMETRY_ADAPTER_LAST_FAILURE_STAGE="$interrupted_stage"
  TELEMETRY_ADAPTER_LAST_FAILURE_EXIT_CODE="$signal_exit_code"
  TELEMETRY_ADAPTER_RUN_STATUS="fail"
  TELEMETRY_ADAPTER_RUN_REASON_CODE="telemetry_adapter.session.interrupted"
  TELEMETRY_ADAPTER_RUN_MESSAGE="received ${signal_name} during stage ${interrupted_stage} (exit_code=${signal_exit_code})"

  telemetry_adapter_emit_event \
    "session.interrupted" \
    "fail" \
    "telemetry_adapter.session.interrupted" \
    "signal=${signal_name} stage=${interrupted_stage} exit_code=${signal_exit_code}"

  exit "$signal_exit_code"
}

telemetry_adapter_install_exit_trap() {
  trap 'telemetry_adapter_on_signal INT 130' INT
  trap 'telemetry_adapter_on_signal TERM 143' TERM
  trap 'telemetry_adapter_on_signal HUP 129' HUP
  trap 'telemetry_adapter_on_exit "$?"' EXIT
}
