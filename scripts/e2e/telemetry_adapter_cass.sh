#!/usr/bin/env bash
set -euo pipefail

MODE="all"
EXECUTION_MODE="live"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/scripts/e2e/telemetry_adapter_common.sh"

usage() {
  cat <<USAGE
Usage: scripts/e2e/telemetry_adapter_cass.sh [--mode unit|integration|e2e|all] [--execution live|dry] [--dry-run]

Runs cass host-adapter telemetry validation lanes for bd-2yu.5.9.
All cargo commands are offloaded through rch.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --execution)
      EXECUTION_MODE="${2:-}"
      shift 2
      ;;
    --dry-run)
      EXECUTION_MODE="dry"
      shift
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

telemetry_adapter_set_execution_mode "$EXECUTION_MODE"
if ! telemetry_adapter_is_dry; then
  telemetry_adapter_require_command rch
fi
telemetry_adapter_init "cass" "$MODE" "${ROOT_DIR}/scripts/e2e/telemetry_adapter_cass.sh --mode ${MODE} --execution ${EXECUTION_MODE}"
telemetry_adapter_install_exit_trap

run_unit() {
  telemetry_adapter_run_rch_cargo \
    "unit.core.cass_hint_aliases" \
    "$ROOT_DIR" \
    cargo test -p frankensearch-core host_adapter::tests::hint_resolves_cass_aliases -- --nocapture
  telemetry_adapter_run_rch_cargo \
    "unit.core.cass_hint_adapter_style" \
    "$ROOT_DIR" \
    cargo test -p frankensearch-core host_adapter::tests::hint_resolves_cass_adapter_style_names -- --nocapture
}

run_integration() {
  telemetry_adapter_run_rch_cargo \
    "integration.ops.discovery_storage_attribution" \
    "$ROOT_DIR" \
    cargo test -p frankensearch-ops --test data_pipeline_integration pipeline_discovery_attribution_aligns_with_storage_rollups_and_anomalies -- --nocapture
}

run_e2e() {
  telemetry_adapter_run_rch_cargo \
    "e2e.cass_host_repo_migration_check" \
    "/data/projects/coding_agent_session_search" \
    cargo +nightly check --all-targets --features frankensearch-migration
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  run_unit
fi
if [[ "$MODE" == "integration" || "$MODE" == "all" ]]; then
  run_integration
fi
if [[ "$MODE" == "e2e" || "$MODE" == "all" ]]; then
  run_e2e
fi

telemetry_adapter_set_status \
  "ok" \
  "cass telemetry adapter lane passed (execution_mode=${EXECUTION_MODE})" \
  "telemetry_adapter.lane.passed"
echo "Result: PASS"
