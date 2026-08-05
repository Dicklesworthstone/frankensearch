#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="all"
LANE="all"
RUN_ID="${FRANKENSEARCH_FEATURE_MATRIX_RUN_ID:-bd-pkl0.13-feature-matrix}"
ARTIFACT_DIR="${FRANKENSEARCH_FEATURE_MATRIX_ARTIFACT_DIR:-/tmp/frankensearch-feature-matrix/${RUN_ID}}"
SCHEMA_VERSION="feature-smoke-lanes-v2"
# ONE RECORD PER LANE — the single source of truth (bd-yv2nt).
#
# A lane used to be DECLARED in three functions (features, compile command,
# behavior command) and EXECUTED in two more, five arms with nothing tying them
# together. Registering only the declared three produced a lane that REPORTED
# SUCCESS WHILE RUNNING NOTHING, and it shipped that way twice within an hour:
# `both` under bd-8nqz.4 slice 4, then `all-features` under slice 5. Two agents
# making the same mistake is a shape problem, not carelessness.
#
# Everything below is now derived from these records: the lane list, the three
# declaration accessors, and — the part that closes the hole — the argv the
# executors actually run. There is no second place to forget.
#
# Fields are `|`-separated: name | cargo features | compile command | behavior
# command. Commands are split on whitespace into argv, so no field may contain
# `|` and no argument may contain spaces; `validate_lane_coverage` enforces both
# shapes before anything runs.
LANE_SPECS=(
  "default|default|cargo check -p frankensearch --all-targets|cargo test -p frankensearch --lib feature_matrix_smoke::default_lane_behavior -- --exact --nocapture"
  "both|lexical,lexical-tantivy|cargo check -p frankensearch --all-targets --no-default-features --features hash,lexical,lexical-tantivy|cargo test -p frankensearch --lib --no-default-features --features hash,lexical,lexical-tantivy feature_matrix_smoke::both_backends_select_deterministically -- --exact --nocapture"
  "all-features|all-features|cargo check -p frankensearch --all-targets --all-features|cargo test -p frankensearch --lib --all-features feature_matrix_smoke::both_backends_select_deterministically -- --exact --nocapture"
  "quill|quill|cargo check -p frankensearch --lib --no-default-features --features quill|cargo test -p frankensearch --lib --no-default-features --features quill feature_matrix_smoke::quill_lane_behavior -- --exact --nocapture"
  "lexical-tantivy|lexical-tantivy|cargo check -p frankensearch --lib --no-default-features --features lexical-tantivy|cargo test -p frankensearch --lib --no-default-features --features lexical-tantivy feature_matrix_smoke::lexical_tantivy_lane_behavior -- --exact --nocapture"
  "cass-compat|cass-compat|cargo check -p frankensearch --lib --no-default-features --features cass-compat|cargo test -p frankensearch --lib --no-default-features --features cass-compat feature_matrix_smoke::cass_compat_lane_behavior -- --exact --nocapture"
  "semantic|semantic|cargo check -p frankensearch --lib --no-default-features --features semantic|cargo test -p frankensearch --lib --no-default-features --features semantic feature_matrix_smoke::semantic_lane_behavior -- --exact --nocapture"
  "hybrid|hybrid|cargo check -p frankensearch --lib --no-default-features --features hybrid|cargo test -p frankensearch --lib --no-default-features --features hybrid feature_matrix_smoke::hybrid_lane_behavior -- --exact --nocapture"
  "persistent|persistent|cargo check -p frankensearch --lib --no-default-features --features persistent|cargo test -p frankensearch --lib --no-default-features --features persistent feature_matrix_smoke::persistent_lane_behavior -- --exact --nocapture"
  "durable|durable|cargo check -p frankensearch --lib --no-default-features --features durable|cargo test -p frankensearch --lib --no-default-features --features durable feature_matrix_smoke::durable_lane_behavior -- --exact --nocapture"
  "full|full|cargo check -p frankensearch --lib --no-default-features --features full|cargo test -p frankensearch --lib --no-default-features --features full feature_matrix_smoke::full_lane_behavior -- --exact --nocapture"
  "full-fts5|full-fts5|cargo check -p frankensearch --lib --no-default-features --features full-fts5|cargo test -p frankensearch --lib --no-default-features --features full-fts5 feature_matrix_smoke::full_fts5_lane_behavior -- --exact --nocapture"
)

# Derived, never hand-maintained: a lane exists exactly when it has a record.
REQUIRED_LANES=()
for _lane_spec in "${LANE_SPECS[@]}"; do
  REQUIRED_LANES+=("${_lane_spec%%|*}")
done
unset _lane_spec

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
    RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- cargo "$@"
  else
    cargo "$@"
  fi
}

# Return the whole record for a lane. The fail-closed default is retained from
# the previous shape: an unknown lane is an error, never an empty string that a
# caller could mistake for "nothing to do".
lane_spec_record() {
  local lane="$1"
  local spec
  for spec in "${LANE_SPECS[@]}"; do
    if [[ "${spec%%|*}" == "$lane" ]]; then
      printf '%s\n' "$spec"
      return 0
    fi
  done
  echo "ERROR: unknown lane '$lane'" >&2
  return 2
}

# Field 1 is the name; 2 features; 3 the compile command; 4 the behavior command.
lane_spec_field() {
  local lane="$1"
  local index="$2"
  local record
  record="$(lane_spec_record "$lane")" || return 2
  local -a fields
  IFS='|' read -ra fields <<<"$record"
  if [[ "${#fields[@]}" -ne 4 ]]; then
    echo "ERROR: lane '$lane' has a malformed spec: expected 4 '|'-separated fields, got ${#fields[@]}" >&2
    return 2
  fi
  printf '%s\n' "${fields[$((index - 1))]}"
}

# A declared command must be a runnable cargo invocation. This is what makes the
# declaration and the execution the same object: if a lane cannot be turned into
# argv here, it cannot be declared either, so "declared but never executed" has
# no representation.
assert_lane_command() {
  local lane="$1"
  local kind="$2"
  local command="$3"
  local -a argv
  read -ra argv <<<"$command"
  if [[ "${#argv[@]}" -lt 2 || "${argv[0]}" != "cargo" ]]; then
    echo "ERROR: lane '$lane' has no runnable $kind command (expected a cargo invocation, got '$command')" >&2
    return 1
  fi
}

lane_features() {
  lane_spec_field "$1" 2
}

lane_compile_command() {
  lane_spec_field "$1" 3
}

lane_behavior_command() {
  lane_spec_field "$1" 4
}

lane_artifact_name() {
  echo "feature-smoke-$1.json"
}

run_compile_lane() {
  local lane="$1"
  local command
  command="$(lane_compile_command "$lane")"
  echo "[feature-matrix][$lane] $command"
  local -a argv
  read -ra argv <<<"$command"
  assert_lane_command "$lane" compile "$command"
  # argv[0] is the literal `cargo`; run_cargo supplies it (and the rch wrapper).
  run_cargo "${argv[@]:1}"
}

run_behavior_lane() {
  local lane="$1"
  local command output
  command="$(lane_behavior_command "$lane")"
  echo "[feature-matrix][$lane] $command"
  local -a argv
  read -ra argv <<<"$command"
  assert_lane_command "$lane" behavior "$command"
  # The exit code is deliberately swallowed so the cargo output still prints:
  # the vacuity guard below turns any outcome that is not exactly one passing
  # test into a named error, so nothing is lost, and a failing lane now shows
  # WHY instead of dying silently under `set -e` before the capture is echoed.
  output="$(run_cargo "${argv[@]:1}" 2>&1)" || true
  printf '%s\n' "$output"
  if [[ "$output" != *"test result: ok. 1 passed; 0 failed;"* ]]; then
    echo "ERROR: feature lane '$lane' did not execute exactly one behavior test" >&2
    return 1
  fi
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

# The validator now checks the TERRITORY, not the map (bd-yv2nt).
#
# Its predecessor asserted that each lane had a features / compile-command /
# behavior-command / artifact-name entry — all four of them DECLARED values. A
# lane could satisfy every one of those and still have no executing arm, which
# is exactly what shipped twice. Because the executors now derive their argv
# from the declaration, proving the declaration is RUNNABLE proves the lane
# runs: there is no second definition left to disagree with it.
validate_lane_coverage() {
  local lane
  local seen=""
  for lane in "${REQUIRED_LANES[@]}"; do
    if [[ -z "$lane" ]]; then
      echo "ERROR: LANE_SPECS contains a record with an empty lane name" >&2
      return 1
    fi
    case "$lane" in
      *[!A-Za-z0-9._-]*)
        echo "ERROR: lane name '$lane' must contain only letters, digits, '.', '_', or '-'" >&2
        return 1
        ;;
    esac
    case " $seen " in
      *" $lane "*)
        echo "ERROR: duplicate lane '$lane' in LANE_SPECS" >&2
        return 1
        ;;
    esac
    seen="$seen $lane"

    # Shape first, so a partial record reports THAT and not a misleading
    # downstream symptom: a spec missing its commands would otherwise surface
    # as "declares no cargo features", which is not what is wrong with it.
    local record
    local -a fields
    record="$(lane_spec_record "$lane")"
    IFS='|' read -ra fields <<<"$record"
    if [[ "${#fields[@]}" -ne 4 ]]; then
      echo "ERROR: lane '$lane' has a malformed spec: expected 4 '|'-separated fields (name|features|compile|behavior), got ${#fields[@]}" >&2
      return 1
    fi

    if [[ -z "$(lane_features "$lane")" ]]; then
      echo "ERROR: lane '$lane' declares no cargo features" >&2
      return 1
    fi
    assert_lane_command "$lane" compile "$(lane_compile_command "$lane")"
    assert_lane_command "$lane" behavior "$(lane_behavior_command "$lane")"
    [[ "$(lane_artifact_name "$lane")" == "feature-smoke-${lane}.json" ]]
  done
  write_matrix_artifact
}

# bd-8nqz.5: the CASS lexical compatibility lane must resolve Tantivy IN and
# Quill OUT.
#
# This lives here, at the RESOLVED DEPENDENCY GRAPH, and not as a cfg! assertion
# inside a #[cfg(all(cass-compat, not(quill)))] test: inside that gate the
# assertion is a compile-time constant and the gate presupposes exactly what it
# claims to prove. `cargo tree` asks the resolver instead, so a feature edit that
# started pulling Quill into cass-compat fails here even though every lane would
# still compile.
validate_cass_compat_backend_graph() {
  local tree
  if ! tree="$(cargo tree -p frankensearch --no-default-features --features cass-compat --prefix none 2>/dev/null)"; then
    echo "ERROR: cass-compat dependency graph could not be resolved" >&2
    return 1
  fi
  if ! grep -q '^frankensearch-lexical ' <<<"${tree}"; then
    echo "ERROR: cass-compat must resolve frankensearch-lexical (Tantivy)" >&2
    return 1
  fi
  if grep -q '^frankensearch-quill ' <<<"${tree}"; then
    echo "ERROR: cass-compat resolved frankensearch-quill; the CASS compatibility lane must exclude Quill unless a consumer requests it independently" >&2
    return 1
  fi
  echo "[feature-matrix][OK] cass-compat resolves Tantivy and excludes Quill"
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
  validate_cass_compat_backend_graph
  if [[ "$MODE" != "validate" ]]; then
    while IFS= read -r lane; do
      [[ -z "$lane" ]] && continue
      run_lane "$lane"
    done < <(selected_lanes)
  fi
)

echo "[feature-matrix] PASS"
