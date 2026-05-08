#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="all"
LANE="all"
RUN_ID="${FRANKENSEARCH_FEATURE_MATRIX_RUN_ID:-bd-pkl0.13-feature-matrix}"
ARTIFACT_DIR="${FRANKENSEARCH_FEATURE_MATRIX_ARTIFACT_DIR:-/tmp/frankensearch-feature-matrix/${RUN_ID}}"
SCHEMA_VERSION="feature-smoke-lanes-v1"
REQUIRED_LANES=(default semantic hybrid persistent durable full full-fts5)

usage() {
  cat <<USAGE
Usage: scripts/check_feature_matrix.sh [OPTIONS]

Validates the per-feature minimal smoke lanes for bd-pkl0.13.

Options:
  --mode <validate|compile|behavior|all>   Which checks to run (default: all)
  --lane <lane|all>                        Lane to run (default: all)
  --artifact-dir <path>                    Artifact output directory
  --run-id <id>                            Stable run identifier for artifact payloads
  -h, --help                               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --lane)
      LANE="${2:-}"
      shift 2
      ;;
    --artifact-dir)
      ARTIFACT_DIR="${2:-}"
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
      echo "ERROR: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  validate|compile|behavior|all) ;;
  *)
    echo "ERROR: invalid --mode '$MODE' (expected validate|compile|behavior|all)" >&2
    exit 2
    ;;
esac

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|"")
    echo "ERROR: --run-id must contain only letters, digits, '.', '_', or '-'" >&2
    exit 2
    ;;
esac

if [[ -z "$ARTIFACT_DIR" ]]; then
  echo "ERROR: --artifact-dir cannot be empty" >&2
  exit 2
fi

lane_exists() {
  local lane="$1"
  local required
  for required in "${REQUIRED_LANES[@]}"; do
    if [[ "$lane" == "$required" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "$LANE" != "all" ]] && ! lane_exists "$LANE"; then
  echo "ERROR: invalid --lane '$LANE' (expected one of: ${REQUIRED_LANES[*]}, all)" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_DIR"

if [[ "${FRANKENSEARCH_FEATURE_MATRIX_USE_RCH:-0}" == "1" ]]; then
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/rch_target_frankensearch_${AGENT_NAME:-agent}_feature_matrix}"
fi

run_cargo() {
  if [[ "${FRANKENSEARCH_FEATURE_MATRIX_USE_RCH:-0}" == "1" ]]; then
    RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- cargo "$@"
  else
    cargo "$@"
  fi
}

lane_features() {
  case "$1" in
    default) echo "default" ;;
    semantic) echo "semantic" ;;
    hybrid) echo "hybrid" ;;
    persistent) echo "persistent" ;;
    durable) echo "durable" ;;
    full) echo "full" ;;
    full-fts5) echo "full-fts5" ;;
    *)
      echo "ERROR: unknown lane '$1'" >&2
      return 2
      ;;
  esac
}

lane_compile_command() {
  case "$1" in
    default) echo "cargo check -p frankensearch --all-targets" ;;
    semantic) echo "cargo check -p frankensearch --lib --no-default-features --features semantic" ;;
    hybrid) echo "cargo check -p frankensearch --lib --no-default-features --features hybrid" ;;
    persistent) echo "cargo check -p frankensearch --lib --no-default-features --features persistent" ;;
    durable) echo "cargo check -p frankensearch --lib --no-default-features --features durable" ;;
    full) echo "cargo check -p frankensearch --lib --no-default-features --features full" ;;
    full-fts5) echo "cargo check -p frankensearch --lib --no-default-features --features full-fts5" ;;
    *)
      echo "ERROR: unknown lane '$1'" >&2
      return 2
      ;;
  esac
}

lane_behavior_command() {
  case "$1" in
    default) echo "cargo test -p frankensearch --lib feature_matrix_smoke::default_lane_behavior -- --exact" ;;
    semantic) echo "cargo test -p frankensearch --lib --no-default-features --features semantic feature_matrix_smoke::semantic_lane_behavior -- --exact" ;;
    hybrid) echo "cargo test -p frankensearch --lib --no-default-features --features hybrid feature_matrix_smoke::hybrid_lane_behavior -- --exact" ;;
    persistent) echo "cargo test -p frankensearch --lib --no-default-features --features persistent feature_matrix_smoke::persistent_lane_behavior -- --exact" ;;
    durable) echo "cargo test -p frankensearch --lib --no-default-features --features durable feature_matrix_smoke::durable_lane_behavior -- --exact" ;;
    full) echo "cargo test -p frankensearch --lib --no-default-features --features full feature_matrix_smoke::full_lane_behavior -- --exact" ;;
    full-fts5) echo "cargo test -p frankensearch --lib --no-default-features --features full-fts5 feature_matrix_smoke::full_fts5_lane_behavior -- --exact" ;;
    *)
      echo "ERROR: unknown lane '$1'" >&2
      return 2
      ;;
  esac
}

lane_artifact_name() {
  echo "feature-smoke-$1.json"
}

run_compile_lane() {
  local lane="$1"
  echo "[feature-matrix][$lane] $(lane_compile_command "$lane")"
  case "$lane" in
    default) run_cargo check -p frankensearch --all-targets ;;
    semantic) run_cargo check -p frankensearch --lib --no-default-features --features semantic ;;
    hybrid) run_cargo check -p frankensearch --lib --no-default-features --features hybrid ;;
    persistent) run_cargo check -p frankensearch --lib --no-default-features --features persistent ;;
    durable) run_cargo check -p frankensearch --lib --no-default-features --features durable ;;
    full) run_cargo check -p frankensearch --lib --no-default-features --features full ;;
    full-fts5) run_cargo check -p frankensearch --lib --no-default-features --features full-fts5 ;;
  esac
}

run_behavior_lane() {
  local lane="$1"
  echo "[feature-matrix][$lane] $(lane_behavior_command "$lane")"
  case "$lane" in
    default) run_cargo test -p frankensearch --lib feature_matrix_smoke::default_lane_behavior -- --exact ;;
    semantic) run_cargo test -p frankensearch --lib --no-default-features --features semantic feature_matrix_smoke::semantic_lane_behavior -- --exact ;;
    hybrid) run_cargo test -p frankensearch --lib --no-default-features --features hybrid feature_matrix_smoke::hybrid_lane_behavior -- --exact ;;
    persistent) run_cargo test -p frankensearch --lib --no-default-features --features persistent feature_matrix_smoke::persistent_lane_behavior -- --exact ;;
    durable) run_cargo test -p frankensearch --lib --no-default-features --features durable feature_matrix_smoke::durable_lane_behavior -- --exact ;;
    full) run_cargo test -p frankensearch --lib --no-default-features --features full feature_matrix_smoke::full_lane_behavior -- --exact ;;
    full-fts5) run_cargo test -p frankensearch --lib --no-default-features --features full-fts5 feature_matrix_smoke::full_fts5_lane_behavior -- --exact ;;
  esac
}

write_lane_artifact() {
  local lane="$1"
  local status="$2"
  local artifact_name artifact_path
  artifact_name="$(lane_artifact_name "$lane")"
  artifact_path="${ARTIFACT_DIR}/${artifact_name}"

  jq -n \
    --arg schema "$SCHEMA_VERSION" \
    --arg run_id "$RUN_ID" \
    --arg lane "$lane" \
    --arg features "$(lane_features "$lane")" \
    --arg compile_command "$(lane_compile_command "$lane")" \
    --arg behavior_test_command "$(lane_behavior_command "$lane")" \
    --arg artifact_name "$artifact_name" \
    --arg status "$status" \
    '{
      schema: $schema,
      run_id: $run_id,
      lane: $lane,
      features: $features,
      compile_command: $compile_command,
      behavior_test_command: $behavior_test_command,
      artifact_name: $artifact_name,
      status: $status
    }' >"$artifact_path"
}

write_matrix_artifact() {
  local matrix_path="${ARTIFACT_DIR}/feature-smoke-matrix.json"
  local lane
  {
    printf '{"schema":"%s","run_id":"%s","required_lanes":[' "$SCHEMA_VERSION" "$RUN_ID"
    local first=1
    for lane in "${REQUIRED_LANES[@]}"; do
      if [[ "$first" -eq 0 ]]; then
        printf ','
      fi
      first=0
      jq -cn \
        --arg lane "$lane" \
        --arg features "$(lane_features "$lane")" \
        --arg compile_command "$(lane_compile_command "$lane")" \
        --arg behavior_test_command "$(lane_behavior_command "$lane")" \
        --arg artifact_name "$(lane_artifact_name "$lane")" \
        '{
          lane: $lane,
          features: $features,
          compile_command: $compile_command,
          behavior_test_command: $behavior_test_command,
          artifact_name: $artifact_name
        }'
    done
    printf ']}\n'
  } >"$matrix_path"
}

validate_lane_coverage() {
  local lane
  for lane in "${REQUIRED_LANES[@]}"; do
    [[ -n "$(lane_features "$lane")" ]]
    [[ -n "$(lane_compile_command "$lane")" ]]
    [[ -n "$(lane_behavior_command "$lane")" ]]
    [[ "$(lane_artifact_name "$lane")" == "feature-smoke-${lane}.json" ]]
  done
  write_matrix_artifact
}

selected_lanes() {
  if [[ "$LANE" == "all" ]]; then
    printf '%s\n' "${REQUIRED_LANES[@]}"
  else
    printf '%s\n' "$LANE"
  fi
}

run_lane() {
  local lane="$1"
  local status="pass"
  if [[ "$MODE" == "compile" || "$MODE" == "all" ]]; then
    run_compile_lane "$lane"
  fi
  if [[ "$MODE" == "behavior" || "$MODE" == "all" ]]; then
    run_behavior_lane "$lane"
  fi
  write_lane_artifact "$lane" "$status"
}

(
  cd "$ROOT_DIR"
  validate_lane_coverage
  if [[ "$MODE" != "validate" ]]; then
    while IFS= read -r lane; do
      [[ -z "$lane" ]] && continue
      run_lane "$lane"
    done < <(selected_lanes)
  fi
)

echo "[feature-matrix] PASS"
