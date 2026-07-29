#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_cli_e2e_contract.sh [--mode unit|integration|e2e|all]

Runs fsfs CLI e2e contract checks for bd-2hz.10.4.
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

cd "$ROOT_DIR"

run_unit() {
  echo "[unit] fsfs cli_e2e module tests"
  cargo test -p frankensearch-fsfs cli_e2e::tests:: -- --nocapture
}

run_integration() {
  echo "[integration] fsfs cli_e2e contract suite"
  cargo test -p frankensearch-fsfs --features embedded-models --test cli_e2e_contract -- --nocapture
}

run_e2e() {
  echo "[e2e] degraded CLI flow replay contract"
  cargo test -p frankensearch-fsfs --features embedded-models --test cli_e2e_contract -- --nocapture --exact scenario_cli_degrade_path
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

echo "Result: PASS"
