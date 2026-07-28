#!/usr/bin/env bash
set -euo pipefail

TRUSTED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="gate"
BASE_GIT_SHA=""
PATCH_PATH=""
EXPECTED_PATCH_SHA256=""
ARTIFACT_DIR=""
RUN_ID="${FRANKENSEARCH_QG10_RUN_ID:-bd-8nqz.4.2}"
SELECTED_LANE="all"
CONTRACT_RELATIVE_PATH="docs/contracts/quill-facade-source-contract-v1.json"
MAX_LOG_BYTES="${FRANKENSEARCH_QG10_MAX_LOG_BYTES:-5242880}"
SCHEMA_VERSION="frankensearch-qg10-overlay-receipt-v1"
LANE_SCHEMA_VERSION="frankensearch-qg10-lane-receipt-v1"
BEHAVIOR_SCHEMA_VERSION="frankensearch-feature-behavior-v2"
CONTRACT_SCHEMA_VERSION="frankensearch-qg10-source-contract-v1"
STATIC_FEATURE_CONTRACT_SHA256="c73adaac35ac307f129d6aedc69f2f676658db518452b41d12dd2e7e84d32ea7"
STATIC_TARGET_CONTRACT_SHA256="b76b28bd0cb1d87be7d2dd87dfffa1c21d8cdb916c24df34d515d8c38ee46810"
STATIC_NAMESPACE_CONTRACT_SHA256="7900fff62767788737a6d3807018ca3c1b0541aaaa6929c3aca75ea12bb1254c"
STATIC_LANE_CONTRACT_SHA256="6bdbacc3c57d009988c718e2a1ebc0ab1c879e59231dd69f5ad7d5ff15122d04"
REQUIRED_LANES=(
  no-default
  default
  hash
  quill
  lexical
  lexical-bench
  lexical-tantivy
  cass-compat
  both-backends
  semantic
  hybrid
  persistent
  durable
  full
  full-fts5
  all-features
)

usage() {
  cat <<'USAGE'
Usage: scripts/check_feature_matrix_overlay.sh [OPTIONS]

Build and verify the prospective QG-10 facade receipt from the exact tuple
{base_git_sha, canonical_flip_patch_sha256}. This script does not change the
checked-out source tree and does not flip facade defaults.

Modes:
  gate       Run every selected source-workspace proof and require the reviewed
             contract hashes. A full all-lane pass is the only admissible receipt.
  audit      Run the same proof but emit observed hashes for contract review;
             the receipt is always non-admissible.
  verify     Re-hash an existing artifact directory and reject any drift.
  self-test  Exercise fail-closed identity, tamper, log-bound, lane, dependency,
             and source-order controls without compiling the workspace.

Required for gate/audit:
  --base-git-sha <40-hex>                  Frozen base commit; must equal HEAD.
  --canonical-flip-patch <path>            Full canonical prospective flip patch.
  --canonical-flip-patch-sha256 <64-hex>   Expected byte hash of that patch.
  --artifact-dir <path>                    New, empty directory outside the repo.

Other options:
  --lane <lane|all>                        Run one lane or the full matrix.
  --run-id <id>                            Stable receipt run identifier.
  --contract-relative-path <path>          Candidate-tree contract path.
  -h, --help                               Show this help.

Environment:
  FRANKENSEARCH_QG10_USE_RCH=1             Execute Cargo/Rustdoc through strict RCH.
  FRANKENSEARCH_QG10_RCH_WORKER=<worker>   Optional pinned remote worker.
  FRANKENSEARCH_QG10_MAX_LOG_BYTES=<n>     Capture-time per-log byte ceiling.

The overlay worktree is intentionally retained for audit. This script never
removes files or worktrees.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --base-git-sha)
      BASE_GIT_SHA="${2:-}"
      shift 2
      ;;
    --canonical-flip-patch)
      PATCH_PATH="${2:-}"
      shift 2
      ;;
    --canonical-flip-patch-sha256)
      EXPECTED_PATCH_SHA256="${2:-}"
      shift 2
      ;;
    --artifact-dir)
      ARTIFACT_DIR="${2:-}"
      shift 2
      ;;
    --lane)
      SELECTED_LANE="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --contract-relative-path)
      CONTRACT_RELATIVE_PATH="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR [ARG_UNKNOWN]: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  gate|audit|verify|self-test) ;;
  *)
    echo "ERROR [MODE_INVALID]: expected gate, audit, verify, or self-test" >&2
    exit 2
    ;;
esac

case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|"")
    echo "ERROR [RUN_ID_INVALID]: use only letters, digits, '.', '_', or '-'" >&2
    exit 2
    ;;
esac

case "$CONTRACT_RELATIVE_PATH" in
  /*|../*|*/../*|*/..|"")
    echo "ERROR [CONTRACT_PATH_INVALID]: contract path must stay inside the candidate tree" >&2
    exit 2
    ;;
esac

case "$MAX_LOG_BYTES" in
  *[!0-9]*|"")
    echo "ERROR [LOG_LIMIT_INVALID]: FRANKENSEARCH_QG10_MAX_LOG_BYTES must be an integer" >&2
    exit 2
    ;;
esac
if (( MAX_LOG_BYTES == 0 )); then
  echo "ERROR [LOG_LIMIT_INVALID]: log limit must be greater than zero" >&2
  exit 2
fi

lane_exists() {
  local candidate_lane="$1"
  local required_lane
  for required_lane in "${REQUIRED_LANES[@]}"; do
    if [[ "$candidate_lane" == "$required_lane" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "$SELECTED_LANE" != "all" ]] && ! lane_exists "$SELECTED_LANE"; then
  echo "ERROR [LANE_INVALID]: unknown lane '$SELECTED_LANE'" >&2
  exit 2
fi

for required_tool in git cargo rustc jq awk sed sort sha256sum cmp realpath rg; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    echo "ERROR [TOOL_MISSING]: required command not found: $required_tool" >&2
    exit 2
  fi
done

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

canonical_json_hash() {
  local json_path="$1"
  jq -S -c . "$json_path" | sha256sum | awk '{print $1}'
}

canonical_json_expression_hash() {
  local json_path="$1"
  local expression="$2"
  jq -S -c "$expression" "$json_path" | sha256sum | awk '{print $1}'
}

json_array_from_lines() {
  jq -Rsc 'split("\n") | map(select(length > 0))'
}

selected_lanes() {
  if [[ "$SELECTED_LANE" == "all" ]]; then
    printf '%s\n' "${REQUIRED_LANES[@]}"
  else
    printf '%s\n' "$SELECTED_LANE"
  fi
}

validate_exact_sha() {
  local label="$1"
  local value="$2"
  local width="$3"
  if [[ ${#value} -ne "$width" || ! "$value" =~ ^[0-9a-f]+$ ]]; then
    echo "ERROR [${label}_INVALID]: expected exactly ${width} lowercase hex characters" >&2
    return 2
  fi
}

artifact_path_is_safe() {
  local candidate_path="$1"
  case "$candidate_path" in
    ""|"/"|"/tmp"|"/data"|"/data/projects")
      echo "ERROR [ARTIFACT_PATH_UNSAFE]: refusing broad artifact directory '$candidate_path'" >&2
      return 2
      ;;
  esac
  local resolved_candidate
  resolved_candidate="$(realpath -m "$candidate_path")"
  case "$resolved_candidate/" in
    "$TRUSTED_ROOT/"*)
      echo "ERROR [ARTIFACT_PATH_IN_REPO]: artifacts must live outside the source tree" >&2
      return 2
      ;;
  esac
}

prepare_empty_artifact_dir() {
  artifact_path_is_safe "$ARTIFACT_DIR"
  mkdir -p "$ARTIFACT_DIR"
  if [[ -n "$(find "$ARTIFACT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR [ARTIFACT_DIR_NOT_EMPTY]: stale artifacts are never reused" >&2
    return 2
  fi
}

validate_static_contract() {
  local contract_path="$1"
  if ! jq -e \
    --arg schema "$CONTRACT_SCHEMA_VERSION" \
    --arg generator "scripts/check_feature_matrix_overlay.sh" \
    '.schema == $schema
      and .generator == $generator
      and .claim_scope.kind == "source-workspace-only"
      and .claim_scope.performance_claim == false
      and .claim_scope.registry_or_package_claim == false
      and .claim_scope.registry_or_package_owner == "bd-8nqz.6"' \
    "$contract_path" >/dev/null
  then
    echo "ERROR [CONTRACT_HEADER_INVALID]: source contract header/scope is not exact" >&2
    return 1
  fi

  local feature_hash target_hash namespace_hash lane_hash
  feature_hash="$(canonical_json_expression_hash "$contract_path" '.required_facade_features')"
  target_hash="$(canonical_json_expression_hash "$contract_path" '.required_facade_targets')"
  namespace_hash="$(canonical_json_expression_hash "$contract_path" '.namespace_contract')"
  lane_hash="$(canonical_json_expression_hash "$contract_path" '.required_lanes')"

  if [[ "$feature_hash" != "$STATIC_FEATURE_CONTRACT_SHA256" ]]; then
    echo "ERROR [FEATURE_CONTRACT_DRIFT]: facade feature census changed without validator review" >&2
    return 1
  fi
  if [[ "$target_hash" != "$STATIC_TARGET_CONTRACT_SHA256" ]]; then
    echo "ERROR [TARGET_CONTRACT_DRIFT]: target census changed without validator review" >&2
    return 1
  fi
  if [[ "$namespace_hash" != "$STATIC_NAMESPACE_CONTRACT_SHA256" ]]; then
    echo "ERROR [NAMESPACE_CONTRACT_DRIFT]: backend naming/selection contract changed" >&2
    return 1
  fi
  if [[ "$lane_hash" != "$STATIC_LANE_CONTRACT_SHA256" ]]; then
    echo "ERROR [LANE_CONTRACT_DRIFT]: required lane contract changed without validator review" >&2
    return 1
  fi

  local actual_lane_names expected_lane_names
  actual_lane_names="$(jq -c '[.required_lanes[].name]' "$contract_path")"
  expected_lane_names="$(printf '%s\n' "${REQUIRED_LANES[@]}" | json_array_from_lines)"
  if [[ "$actual_lane_names" != "$expected_lane_names" ]]; then
    echo "ERROR [LANE_CENSUS_INCOMPLETE]: required lane names/order are not exhaustive" >&2
    return 1
  fi

  if [[ "$MODE" == "gate" ]]; then
    if ! jq -e '
      .review_status == "reviewed"
      and (
        .reviewed_inventory
        | [
            .cargo_lock_sha256,
            .workspace_manifest_sha256,
            .facade_manifest_sha256,
            .toolchain_sha256,
            .facade_features_sha256,
            .target_inventory_sha256,
            .schema_inventory_sha256,
            .source_publish_order_sha256
          ]
        | all(type == "string" and test("^[0-9a-f]{64}$"))
      )
      and (
        [
          .reviewed_inventory.resolved_features_by_lane[],
          .reviewed_inventory.dependency_packages_by_lane[],
          .reviewed_inventory.public_api_by_lane[]
        ]
        | all(type == "string" and test("^[0-9a-f]{64}$"))
      )' "$contract_path" >/dev/null
    then
      echo "ERROR [CONTRACT_UNREVIEWED]: gate requires every generated inventory hash to be reviewed" >&2
      return 1
    fi
  fi
}

lane_contract_json() {
  local contract_path="$1"
  local lane_name="$2"
  jq -c --arg lane "$lane_name" '.required_lanes[] | select(.name == $lane)' "$contract_path"
}

set_lane_cargo_args() {
  local contract_path="$1"
  local lane_name="$2"
  local feature_mode
  feature_mode="$(jq -r --arg lane "$lane_name" '.required_lanes[] | select(.name == $lane) | .feature_mode' "$contract_path")"
  mapfile -t LANE_REQUESTED_FEATURES < <(
    jq -r --arg lane "$lane_name" \
      '.required_lanes[] | select(.name == $lane) | .requested_features[]' \
      "$contract_path"
  )
  LANE_CARGO_ARGS=()
  case "$feature_mode" in
    default) ;;
    no-default)
      LANE_CARGO_ARGS+=(--no-default-features)
      if (( ${#LANE_REQUESTED_FEATURES[@]} > 0 )); then
        local feature_csv
        feature_csv="$(IFS=,; echo "${LANE_REQUESTED_FEATURES[*]}")"
        LANE_CARGO_ARGS+=(--features "$feature_csv")
      fi
      ;;
    all-features)
      LANE_CARGO_ARGS+=(--all-features)
      ;;
    *)
      echo "ERROR [FEATURE_MODE_INVALID]: lane '$lane_name' has '$feature_mode'" >&2
      return 2
      ;;
  esac
}

dependency_contract_passes() {
  local contract_path="$1"
  local lane_name="$2"
  local package_json_path="$3"
  local required_package forbidden_prefix
  while IFS= read -r required_package; do
    [[ -z "$required_package" ]] && continue
    if ! jq -e --arg package "$required_package" 'index($package) != null' "$package_json_path" >/dev/null; then
      echo "ERROR [DEPENDENCY_REQUIRED_MISSING]: lane '$lane_name' lacks '$required_package'" >&2
      return 1
    fi
  done < <(
    jq -r --arg lane "$lane_name" \
      '.required_lanes[] | select(.name == $lane) | .required_packages[]' \
      "$contract_path"
  )
  while IFS= read -r forbidden_prefix; do
    [[ -z "$forbidden_prefix" ]] && continue
    if jq -e --arg prefix "$forbidden_prefix" \
      'any(.[]; . == $prefix or startswith($prefix + "-"))' \
      "$package_json_path" >/dev/null
    then
      echo "ERROR [DEPENDENCY_FORBIDDEN_PRESENT]: lane '$lane_name' resolved '$forbidden_prefix*'" >&2
      return 1
    fi
  done < <(
    jq -r --arg lane "$lane_name" \
      '.required_lanes[] | select(.name == $lane) | .forbidden_package_prefixes[]' \
      "$contract_path"
  )
}

source_publish_order_passes() {
  local order_json_path="$1"
  local quill_position fusion_position facade_position
  quill_position="$(jq -r '.[] | select(.name == "frankensearch-quill") | .position' "$order_json_path")"
  fusion_position="$(jq -r '.[] | select(.name == "frankensearch-fusion") | .position' "$order_json_path")"
  facade_position="$(jq -r '.[] | select(.name == "frankensearch") | .position' "$order_json_path")"
  if [[ ! "$quill_position" =~ ^[0-9]+$ || ! "$fusion_position" =~ ^[0-9]+$ || ! "$facade_position" =~ ^[0-9]+$ ]]; then
    echo "ERROR [SOURCE_ORDER_PACKAGE_MISSING]: Quill/fusion/facade must all be publishable workspace nodes" >&2
    return 1
  fi
  if (( quill_position >= fusion_position || quill_position >= facade_position )); then
    echo "ERROR [SOURCE_ORDER_QUILL_LATE]: Quill must precede fusion and facade" >&2
    return 1
  fi

  if ! jq -e '
    . as $packages
    | all(
        .[];
        . as $package
        | all(
            .internal_dependencies[];
            . as $dependency
            | (($packages[] | select(.name == $dependency) | .position) < $package.position)
          )
      )' "$order_json_path" >/dev/null
  then
    echo "ERROR [SOURCE_ORDER_NOT_TOPOLOGICAL]: a workspace dependency follows its consumer" >&2
    return 1
  fi
}

BASE_TREE_SHA=""
PATCH_SHA256=""
CANDIDATE_TREE_SHA=""
SYNTHETIC_COMMIT_SHA=""
OVERLAY_ROOT=""
CONTRACT_FILE=""
BASE_CLEAN=false
OVERLAY_CLEAN=false

validate_overlay_inputs() {
  local trusted_contract_file="${TRUSTED_ROOT}/${CONTRACT_RELATIVE_PATH}"
  validate_exact_sha "BASE_GIT_SHA" "$BASE_GIT_SHA" 40
  validate_exact_sha "PATCH_SHA256" "$EXPECTED_PATCH_SHA256" 64
  if [[ ! -f "$trusted_contract_file" ]]; then
    echo "ERROR [TRUSTED_CONTRACT_MISSING]: base lacks '$CONTRACT_RELATIVE_PATH'" >&2
    return 2
  fi
  validate_static_contract "$trusted_contract_file"
  if [[ -z "$PATCH_PATH" || ! -f "$PATCH_PATH" ]]; then
    echo "ERROR [PATCH_MISSING]: canonical flip patch is not a regular file" >&2
    return 2
  fi
  PATCH_PATH="$(realpath "$PATCH_PATH")"
  PATCH_SHA256="$(sha256_file "$PATCH_PATH")"
  if [[ "$PATCH_SHA256" != "$EXPECTED_PATCH_SHA256" ]]; then
    echo "ERROR [PATCH_HASH_MISMATCH]: canonical patch bytes changed" >&2
    return 1
  fi

  local resolved_base current_head current_state
  resolved_base="$(git -C "$TRUSTED_ROOT" rev-parse "${BASE_GIT_SHA}^{commit}")"
  if [[ "$resolved_base" != "$BASE_GIT_SHA" ]]; then
    echo "ERROR [BASE_SHA_MISMATCH]: base must be a full exact commit identity" >&2
    return 1
  fi
  current_head="$(git -C "$TRUSTED_ROOT" rev-parse HEAD)"
  if [[ "$current_head" != "$BASE_GIT_SHA" ]]; then
    echo "ERROR [BASE_HEAD_DRIFT]: trusted checkout HEAD is not the frozen base" >&2
    return 1
  fi
  current_state="$(git -C "$TRUSTED_ROOT" status --porcelain=v1 --untracked-files=all)"
  if [[ -n "$current_state" ]]; then
    echo "ERROR [BASE_DIRTY]: trusted base checkout contains tracked or untracked drift" >&2
    return 1
  fi
  BASE_CLEAN=true
  BASE_TREE_SHA="$(git -C "$TRUSTED_ROOT" rev-parse "${BASE_GIT_SHA}^{tree}")"
}

materialize_overlay() {
  local index_path base_epoch commit_message
  index_path="${ARTIFACT_DIR}/prospective-overlay.index"
  if [[ -e "$index_path" ]]; then
    echo "ERROR [OVERLAY_INDEX_EXISTS]: refusing stale temporary index" >&2
    return 2
  fi

  GIT_INDEX_FILE="$index_path" git -C "$TRUSTED_ROOT" read-tree "$BASE_GIT_SHA"
  if ! GIT_INDEX_FILE="$index_path" git -C "$TRUSTED_ROOT" apply \
    --cached \
    --whitespace=error-all \
    "$PATCH_PATH"
  then
    echo "ERROR [PATCH_APPLY_FAILED]: patch does not apply exactly to the frozen base index" >&2
    return 1
  fi
  CANDIDATE_TREE_SHA="$(
    GIT_INDEX_FILE="$index_path" git -C "$TRUSTED_ROOT" write-tree
  )"
  if [[ "$CANDIDATE_TREE_SHA" == "$BASE_TREE_SHA" ]]; then
    echo "ERROR [PATCH_EMPTY]: canonical flip patch produces the base tree unchanged" >&2
    return 1
  fi

  base_epoch="$(git -C "$TRUSTED_ROOT" show -s --format=%ct "$BASE_GIT_SHA")"
  commit_message="$(
    printf 'QG-10 prospective overlay\n\nbase=%s\npatch_sha256=%s\n' \
      "$BASE_GIT_SHA" \
      "$PATCH_SHA256"
  )"
  SYNTHETIC_COMMIT_SHA="$(
    printf '%s\n' "$commit_message" \
      | GIT_AUTHOR_NAME="QG10 Overlay" \
        GIT_AUTHOR_EMAIL="qg10-overlay@invalid.example" \
        GIT_COMMITTER_NAME="QG10 Overlay" \
        GIT_COMMITTER_EMAIL="qg10-overlay@invalid.example" \
        GIT_AUTHOR_DATE="@${base_epoch} +0000" \
        GIT_COMMITTER_DATE="@${base_epoch} +0000" \
        git -C "$TRUSTED_ROOT" commit-tree "$CANDIDATE_TREE_SHA" -p "$BASE_GIT_SHA"
  )"

  OVERLAY_ROOT="${ARTIFACT_DIR}.overlay-${BASE_GIT_SHA:0:12}-${PATCH_SHA256:0:12}"
  if [[ -e "$OVERLAY_ROOT" ]]; then
    echo "ERROR [OVERLAY_PATH_EXISTS]: refusing to replace '$OVERLAY_ROOT'" >&2
    return 2
  fi
  git -C "$TRUSTED_ROOT" worktree add --detach "$OVERLAY_ROOT" "$SYNTHETIC_COMMIT_SHA"

  local checked_tree overlay_state
  checked_tree="$(git -C "$OVERLAY_ROOT" rev-parse 'HEAD^{tree}')"
  if [[ "$checked_tree" != "$CANDIDATE_TREE_SHA" ]]; then
    echo "ERROR [OVERLAY_TREE_MISMATCH]: materialized tree does not match temporary-index tree" >&2
    return 1
  fi
  overlay_state="$(git -C "$OVERLAY_ROOT" status --porcelain=v1 --untracked-files=all)"
  if [[ -n "$overlay_state" ]]; then
    echo "ERROR [OVERLAY_DIRTY]: freshly materialized candidate is not clean" >&2
    return 1
  fi
  OVERLAY_CLEAN=true
  CONTRACT_FILE="${OVERLAY_ROOT}/${CONTRACT_RELATIVE_PATH}"
  if [[ ! -f "$CONTRACT_FILE" ]]; then
    echo "ERROR [CONTRACT_MISSING]: candidate lacks '$CONTRACT_RELATIVE_PATH'" >&2
    return 1
  fi
  if ! cmp -s "${TRUSTED_ROOT}/${CONTRACT_RELATIVE_PATH}" "$CONTRACT_FILE"; then
    echo "ERROR [CONTRACT_PATCHED]: canonical flip patch may not rewrite its trusted contract" >&2
    return 1
  fi
}

CURRENT_LOG_OVERLAY_ROOT=""
CURRENT_LOG_REPOSITORY_ROOT=""

run_logged() {
  local log_path="$1"
  shift
  local command_exit filter_exit
  local -a pipeline_outcomes
  set +e
  "$@" 2>&1 \
    | LC_ALL=C awk \
      -v limit="$MAX_LOG_BYTES" \
      -v overlay="$CURRENT_LOG_OVERLAY_ROOT" \
      -v repository="$CURRENT_LOG_REPOSITORY_ROOT" '
        function replace_literal(value, needle, replacement, offset, result) {
          if (needle == "") {
            return value
          }
          result = ""
          while ((offset = index(value, needle)) != 0) {
            result = result substr(value, 1, offset - 1) replacement
            value = substr(value, offset + length(needle))
          }
          return result value
        }
        BEGIN {
          bytes = 0
          overflow = 0
        }
        {
          line = $0 ORS
          line = replace_literal(line, overlay, "<overlay>")
          line = replace_literal(line, repository, "<repository>")
          gsub(/\/tmp\/rch\/[A-Za-z0-9._\/-]+/, "<remote-workspace>", line)
          gsub(/\/data\/projects\/[A-Za-z0-9._\/-]+/, "<host-workspace>", line)
          line_bytes = length(line)
          if (bytes + line_bytes <= limit) {
            printf "%s", line
          } else {
            remaining = limit - bytes
            if (remaining > 0) {
              printf "%s", substr(line, 1, remaining)
            }
            overflow = 1
          }
          bytes += line_bytes
        }
        END {
          if (overflow) {
            exit 86
          }
        }
      ' >"$log_path"
  pipeline_outcomes=("${PIPESTATUS[@]}")
  set -e
  command_exit="${pipeline_outcomes[0]}"
  filter_exit="${pipeline_outcomes[1]}"
  if (( filter_exit == 86 )); then
    echo "ERROR [LOG_LIMIT_EXCEEDED]: '$log_path' exceeded ${MAX_LOG_BYTES} bytes" >&2
    return 86
  fi
  if (( filter_exit != 0 )); then
    echo "ERROR [LOG_FILTER_FAILED]: '$log_path' filter exit ${filter_exit}" >&2
    return "$filter_exit"
  fi
  if (( command_exit != 0 )); then
    return "$command_exit"
  fi
  return 0
}

CURRENT_TARGET_DIR=""

run_overlay_tool() {
  if [[ "${FRANKENSEARCH_QG10_USE_RCH:-0}" == "1" ]]; then
    local allowlist="CARGO_TARGET_DIR"
    local rustdoc_flags="${CURRENT_RUSTDOCFLAGS:-}"
    local -a rch_environment=(
      "CARGO_TARGET_DIR=${CURRENT_TARGET_DIR}"
      "RUSTDOCFLAGS=${rustdoc_flags}"
      "RCH_REQUIRE_REMOTE=1"
      "RCH_NO_SELF_HEALING=1"
    )
    if [[ -n "$rustdoc_flags" ]]; then
      allowlist="${allowlist},RUSTDOCFLAGS"
    fi
    rch_environment+=("RCH_ENV_ALLOWLIST=${allowlist}")
    if [[ -n "${FRANKENSEARCH_QG10_RCH_WORKER:-}" ]]; then
      rch_environment+=("RCH_WORKER=${FRANKENSEARCH_QG10_RCH_WORKER}")
    fi
    (
      cd "$OVERLAY_ROOT"
      env "${rch_environment[@]}" \
        rch --no-self-healing exec -- "$@"
    )
  else
    (
      cd "$OVERLAY_ROOT"
      CARGO_TARGET_DIR="$CURRENT_TARGET_DIR" \
      RUSTDOCFLAGS="${CURRENT_RUSTDOCFLAGS:-}" \
        "$@"
    )
  fi
}

run_overlay_read_tool() {
  (
    cd "$OVERLAY_ROOT"
    "$@"
  )
}

run_trusted_publish_planner() {
  (
    cd "$TRUSTED_ROOT"
    scripts/check_crates_publish_contract.sh "$@"
  )
}

extract_metadata_json() {
  local log_path="$1"
  local output_path="$2"
  local json_lines
  json_lines="$(rg -c '^\{' "$log_path" || true)"
  if [[ "$json_lines" != "1" ]]; then
    echo "ERROR [METADATA_OUTPUT_INVALID]: expected one JSON object in '$log_path'" >&2
    return 1
  fi
  rg '^\{' "$log_path" >"$output_path"
  jq -e '.packages and .resolve' "$output_path" >/dev/null
}

CARGO_LOCK_SHA256=""
WORKSPACE_MANIFEST_SHA256=""
FACADE_MANIFEST_SHA256=""
TOOLCHAIN_SHA256=""
FACADE_FEATURES_SHA256=""
TARGET_INVENTORY_SHA256=""
SCHEMA_INVENTORY_SHA256=""
SOURCE_PUBLISH_ORDER_SHA256=""

capture_global_inventory() {
  if [[ ! -f "${OVERLAY_ROOT}/Cargo.lock" ]]; then
    echo "ERROR [CARGO_LOCK_MISSING]: canonical flip patch must carry the reviewed lockfile" >&2
    return 1
  fi
  local metadata_log="${ARTIFACT_DIR}/facade-static-metadata.log"
  local metadata_json="${ARTIFACT_DIR}/facade-static-metadata.json"
  CURRENT_TARGET_DIR="${OVERLAY_ROOT}/target/qg10-static"
  if ! run_logged "$metadata_log" \
    run_overlay_read_tool cargo metadata \
      --locked \
      --format-version 1 \
      --manifest-path frankensearch/Cargo.toml \
      --all-features
  then
    echo "ERROR [STATIC_METADATA_FAILED]: candidate metadata could not be resolved with Cargo.lock" >&2
    return 1
  fi
  extract_metadata_json "$metadata_log" "$metadata_json"

  jq -S '
    .packages[]
    | select(.name == "frankensearch")
    | .features
    | keys
    | sort
  ' "$metadata_json" >"${ARTIFACT_DIR}/facade-features.json"

  jq -S '
    .packages[]
    | select(.name == "frankensearch")
    | .targets
    | map({
        name,
        kind: (.kind | sort),
        crate_types: (.crate_types | sort),
        required_features: ((."required-features" // []) | sort)
      })
    | sort_by(.name, .kind)
  ' "$metadata_json" >"${ARTIFACT_DIR}/facade-targets.json"

  if ! cmp -s \
    <(jq -S '.required_facade_features' "$CONTRACT_FILE") \
    "${ARTIFACT_DIR}/facade-features.json"
  then
    echo "ERROR [FACADE_FEATURE_CENSUS_DRIFT]: declared facade feature names changed" >&2
    return 1
  fi
  if ! cmp -s \
    <(jq -S '.required_facade_targets' "$CONTRACT_FILE") \
    "${ARTIFACT_DIR}/facade-targets.json"
  then
    echo "ERROR [FACADE_TARGET_CENSUS_DRIFT]: example/bench/test/bin/lib target inventory changed" >&2
    return 1
  fi

  local schema_entries="${ARTIFACT_DIR}/schema-inventory.jsonl"
  : >"$schema_entries"
  while IFS= read -r schema_file; do
    [[ -z "$schema_file" ]] && continue
    jq -cn \
      --arg path "$schema_file" \
      --arg sha256 "$(sha256_file "${OVERLAY_ROOT}/${schema_file}")" \
      '{kind: "json-schema", path: $path, sha256: $sha256}' \
      >>"$schema_entries"
  done < <(
    cd "$OVERLAY_ROOT"
    rg --files schemas \
      | LC_ALL=C sort \
      | awk '/\.schema\.json$/'
  )

  while IFS= read -r constant_line; do
    [[ -z "$constant_line" ]] && continue
    jq -cn \
      --arg declaration "$constant_line" \
      '{kind: "public-schema-constant", declaration: $declaration}' \
      >>"$schema_entries"
  done < <(
    cd "$OVERLAY_ROOT"
    rg -n --no-heading \
      'pub (const|static) [A-Z0-9_]*(SCHEMA|FORMAT|PROTOCOL)[A-Z0-9_]*(VERSION|HASH|ID)[A-Z0-9_]*' \
      frankensearch/src crates \
      | LC_ALL=C sort
  )
  jq -S -s 'sort_by(.kind, .path // "", .declaration // "")' \
    "$schema_entries" >"${ARTIFACT_DIR}/schema-inventory.json"

  CARGO_LOCK_SHA256="$(sha256_file "${OVERLAY_ROOT}/Cargo.lock")"
  WORKSPACE_MANIFEST_SHA256="$(sha256_file "${OVERLAY_ROOT}/Cargo.toml")"
  FACADE_MANIFEST_SHA256="$(sha256_file "${OVERLAY_ROOT}/frankensearch/Cargo.toml")"
  TOOLCHAIN_SHA256="$(sha256_file "${OVERLAY_ROOT}/rust-toolchain.toml")"
  FACADE_FEATURES_SHA256="$(canonical_json_hash "${ARTIFACT_DIR}/facade-features.json")"
  TARGET_INVENTORY_SHA256="$(canonical_json_hash "${ARTIFACT_DIR}/facade-targets.json")"
  SCHEMA_INVENTORY_SHA256="$(canonical_json_hash "${ARTIFACT_DIR}/schema-inventory.json")"
}

extract_dependency_packages() {
  local tree_log="$1"
  local package_json="$2"
  awk '
    /^[A-Za-z0-9_][A-Za-z0-9_.-]* v[0-9]/ {
      print $1
    }
  ' "$tree_log" \
    | LC_ALL=C sort -u \
    | jq -Rsc 'split("\n") | map(select(length > 0))' \
    >"$package_json"
}

extract_public_api() {
  local rustdoc_json="$1"
  local public_api_json="$2"
  jq -S '
    {
      reachable_paths: (
        [
          .paths
          | to_entries[]
          | select(.value.path[0] == "frankensearch")
          | {
              path: (.value.path | join("::")),
              kind: .value.kind
            }
        ]
        | sort_by(.path, .kind)
      ),
      public_items: (
        [
          .index
          | to_entries[]
          | select(.value.visibility == "public")
          | {
              id: .key,
              crate_id: .value.crate_id,
              name: .value.name,
              attrs: .value.attrs,
              deprecation: .value.deprecation,
              inner: .value.inner
            }
        ]
        | sort_by(.id)
      )
    }
  ' "$rustdoc_json" >"$public_api_json"
  if [[ "$(jq '.reachable_paths | length' "$public_api_json")" -eq 0 ]] \
    || [[ "$(jq '.public_items | length' "$public_api_json")" -eq 0 ]]
  then
    echo "ERROR [PUBLIC_API_EMPTY]: rustdoc JSON yielded no facade paths" >&2
    return 1
  fi
}

content_address_file() {
  local stable_path="$1"
  local prefix="$2"
  local digest addressed_path
  digest="$(sha256_file "$stable_path")"
  addressed_path="$(dirname "$stable_path")/${prefix}.${digest}.json"
  if [[ -e "$addressed_path" ]]; then
    if ! cmp -s "$stable_path" "$addressed_path"; then
      echo "ERROR [CONTENT_ADDRESS_COLLISION]: '$addressed_path'" >&2
      return 1
    fi
  else
    cp "$stable_path" "$addressed_path"
  fi
  printf '%s\n' "$digest"
}

write_lane_receipt() {
  local lane_name="$1"
  local lane_outcome="$2"
  local failure_phase="$3"
  local resolved_features_path="$4"
  local dependency_packages_path="$5"
  local public_api_path="$6"
  local runtime_observation_path="$7"
  shift 7
  local log_paths=("$@")
  local logs_json="[]"
  local log_path
  for log_path in "${log_paths[@]}"; do
    if [[ -f "$log_path" ]]; then
      local log_relative_path="${log_path#"${ARTIFACT_DIR}/"}"
      if [[ "$log_relative_path" == "$log_path" ]]; then
        echo "ERROR [LANE_LOG_OUTSIDE_ARTIFACT_DIR]: '$log_path'" >&2
        return 1
      fi
      logs_json="$(
        jq -cn \
          --argjson prior "$logs_json" \
          --arg file "$log_relative_path" \
          --arg sha256 "$(sha256_file "$log_path")" \
          '$prior + [{file: $file, sha256: $sha256}]'
      )"
    fi
  done

  local runtime_observation="null"
  if [[ -f "$runtime_observation_path" ]]; then
    runtime_observation="$(jq -c . "$runtime_observation_path")"
  fi
  local resolved_features="[]"
  local dependency_packages="[]"
  local public_api_sha256=""
  if [[ -f "$resolved_features_path" ]]; then
    resolved_features="$(jq -c . "$resolved_features_path")"
  fi
  if [[ -f "$dependency_packages_path" ]]; then
    dependency_packages="$(jq -c . "$dependency_packages_path")"
  fi
  if [[ -f "$public_api_path" ]]; then
    public_api_sha256="$(canonical_json_hash "$public_api_path")"
  fi

  local receipt_path="${ARTIFACT_DIR}/qg10-lane-${lane_name}.json"
  jq -n \
    --arg schema "$LANE_SCHEMA_VERSION" \
    --arg lane "$lane_name" \
    --arg outcome "$lane_outcome" \
    --arg failure_phase "$failure_phase" \
    --arg base_git_sha "$BASE_GIT_SHA" \
    --arg patch_sha256 "$PATCH_SHA256" \
    --arg candidate_tree_sha "$CANDIDATE_TREE_SHA" \
    --argjson lane_contract "$(lane_contract_json "$CONTRACT_FILE" "$lane_name")" \
    --argjson resolved_features "$resolved_features" \
    --argjson dependency_packages "$dependency_packages" \
    --arg public_api_sha256 "$public_api_sha256" \
    --argjson runtime_observation "$runtime_observation" \
    --argjson logs "$logs_json" \
    '{
      schema: $schema,
      lane: $lane,
      outcome: $outcome,
      failure_phase: (if $failure_phase == "" then null else $failure_phase end),
      candidate: {
        base_git_sha: $base_git_sha,
        canonical_flip_patch_sha256: $patch_sha256,
        candidate_tree_sha: $candidate_tree_sha
      },
      contract: $lane_contract,
      observed: {
        resolved_features: $resolved_features,
        dependency_packages: $dependency_packages,
        public_api_sha256: (if $public_api_sha256 == "" then null else $public_api_sha256 end),
        runtime: $runtime_observation
      },
      logs: $logs
    }' >"$receipt_path"
  content_address_file "$receipt_path" "qg10-lane-${lane_name}" >/dev/null
}

run_lane() {
  local lane_name="$1"
  local lane_dir="${ARTIFACT_DIR}/lane-${lane_name}"
  mkdir -p "$lane_dir"
  CURRENT_TARGET_DIR="${OVERLAY_ROOT}/target/qg10-${lane_name}"
  set_lane_cargo_args "$CONTRACT_FILE" "$lane_name"

  local metadata_log="${lane_dir}/metadata.log"
  local metadata_json="${lane_dir}/metadata.json"
  local resolved_features_json="${lane_dir}/resolved-features.json"
  local tree_log="${lane_dir}/dependencies.log"
  local dependency_packages_json="${lane_dir}/dependency-packages.json"
  local compile_log="${lane_dir}/check-all-targets.log"
  local doctest_log="${lane_dir}/doctest.log"
  local behavior_log="${lane_dir}/runtime-behavior.log"
  local rustdoc_log="${lane_dir}/rustdoc-json.log"
  local public_api_json="${lane_dir}/public-api.json"
  local runtime_observation_json="${lane_dir}/runtime-observation.json"
  local lane_outcome="pass"
  local failure_phase=""

  echo "[qg10][$lane_name] cargo metadata"
  if ! run_logged "$metadata_log" \
    run_overlay_read_tool cargo metadata \
      --locked \
      --format-version 1 \
      --manifest-path frankensearch/Cargo.toml \
      "${LANE_CARGO_ARGS[@]}"
  then
    lane_outcome="fail"
    failure_phase="metadata"
  elif ! extract_metadata_json "$metadata_log" "$metadata_json"; then
    lane_outcome="fail"
    failure_phase="metadata"
  else
    jq -S '
      . as $metadata
      | ($metadata.packages[] | select(.name == "frankensearch") | .id) as $facade_id
      | $metadata.resolve.nodes[]
      | select(.id == $facade_id)
      | .features
      | sort
    ' "$metadata_json" >"$resolved_features_json"
  fi

  if [[ "$lane_outcome" == "pass" ]]; then
    echo "[qg10][$lane_name] cargo tree"
    if ! run_logged "$tree_log" \
      run_overlay_read_tool cargo tree \
        --locked \
        -p frankensearch \
        -e normal \
        --prefix none \
        "${LANE_CARGO_ARGS[@]}"
    then
      lane_outcome="fail"
      failure_phase="dependencies"
    else
      extract_dependency_packages "$tree_log" "$dependency_packages_json"
      if ! dependency_contract_passes \
        "$CONTRACT_FILE" \
        "$lane_name" \
        "$dependency_packages_json"
      then
        lane_outcome="fail"
        failure_phase="dependencies"
      fi
    fi
  fi

  if [[ "$lane_outcome" == "pass" ]]; then
    echo "[qg10][$lane_name] cargo check --all-targets"
    if ! run_logged "$compile_log" \
      run_overlay_tool cargo check \
        --locked \
        -p frankensearch \
        --all-targets \
        "${LANE_CARGO_ARGS[@]}"
    then
      lane_outcome="fail"
      failure_phase="check-all-targets"
    fi
  fi

  if [[ "$lane_outcome" == "pass" ]]; then
    echo "[qg10][$lane_name] cargo test --doc"
    if ! run_logged "$doctest_log" \
      run_overlay_tool cargo test \
        --locked \
        -p frankensearch \
        --doc \
        "${LANE_CARGO_ARGS[@]}"
    then
      lane_outcome="fail"
      failure_phase="doctest"
    fi
  fi

  if [[ "$lane_outcome" == "pass" ]]; then
    local behavior_test expected_backend expected_selection
    behavior_test="$(
      jq -r --arg lane "$lane_name" \
        '.required_lanes[] | select(.name == $lane) | .behavior_test' \
        "$CONTRACT_FILE"
    )"
    expected_backend="$(
      jq -r --arg lane "$lane_name" \
        '.required_lanes[] | select(.name == $lane) | .expected_lexical_backend' \
        "$CONTRACT_FILE"
    )"
    expected_selection="$(
      jq -r --arg lane "$lane_name" \
        '.required_lanes[] | select(.name == $lane) | .expected_selected_backend' \
        "$CONTRACT_FILE"
    )"
    echo "[qg10][$lane_name] runtime backend probe"
    if ! run_logged "$behavior_log" \
      run_overlay_tool cargo test \
        --locked \
        -p frankensearch \
        --lib \
        "${LANE_CARGO_ARGS[@]}" \
        "$behavior_test" \
        -- \
        --exact \
        --nocapture
    then
      lane_outcome="fail"
      failure_phase="runtime"
    else
      local test_summary_count observation_count
      test_summary_count="$(
        rg -F -c 'test result: ok. 1 passed; 0 failed;' "$behavior_log" \
          || printf '0\n'
      )"
      observation_count="$(
        { rg -F "\"schema\":\"${BEHAVIOR_SCHEMA_VERSION}\"" "$behavior_log" || true; } \
          | jq -Rsc 'split("\n") | map(select(length > 0)) | length'
      )"
      if [[ "$test_summary_count" != "1" || "$observation_count" != "1" ]]; then
        echo "ERROR [RUNTIME_RECEIPT_COUNT]: lane '$lane_name' must execute one test and emit one receipt" >&2
        lane_outcome="fail"
        failure_phase="runtime"
      else
        rg -F "\"schema\":\"${BEHAVIOR_SCHEMA_VERSION}\"" "$behavior_log" \
          >"$runtime_observation_json"
        if ! jq -e \
          --arg schema "$BEHAVIOR_SCHEMA_VERSION" \
          --arg lane "$lane_name" \
          --arg backend "$expected_backend" \
          --arg selected "$expected_selection" \
          '.schema == $schema
            and .lane == $lane
            and .status == "pass"
            and .observations.lexical_backend == $backend
            and .observations.selected_backend == $selected' \
          "$runtime_observation_json" >/dev/null
        then
          echo "ERROR [RUNTIME_BACKEND_MISMATCH]: lane '$lane_name' did not prove backend and selection" >&2
          lane_outcome="fail"
          failure_phase="runtime"
        fi
      fi
    fi
  fi

  if [[ "$lane_outcome" == "pass" ]]; then
    echo "[qg10][$lane_name] rustdoc public API JSON"
    if ! run_logged "$rustdoc_log" \
      CURRENT_RUSTDOCFLAGS="-Z unstable-options --output-format json" \
      run_overlay_tool cargo doc \
        --locked \
        -p frankensearch \
        --lib \
        --no-deps \
        "${LANE_CARGO_ARGS[@]}" \
        --quiet
    then
      lane_outcome="fail"
      failure_phase="rustdoc"
    else
      local rustdoc_json="${CURRENT_TARGET_DIR}/doc/frankensearch.json"
      if [[ ! -f "$rustdoc_json" ]]; then
        echo "ERROR [RUSTDOC_JSON_MISSING]: lane '$lane_name' produced no facade JSON" >&2
        lane_outcome="fail"
        failure_phase="rustdoc"
      elif ! extract_public_api "$rustdoc_json" "$public_api_json"; then
        lane_outcome="fail"
        failure_phase="public-api"
      fi
    fi
  fi

  write_lane_receipt \
    "$lane_name" \
    "$lane_outcome" \
    "$failure_phase" \
    "$resolved_features_json" \
    "$dependency_packages_json" \
    "$public_api_json" \
    "$runtime_observation_json" \
    "$metadata_log" \
    "$tree_log" \
    "$compile_log" \
    "$doctest_log" \
    "$behavior_log" \
    "$rustdoc_log"

  [[ "$lane_outcome" == "pass" ]]
}

derive_source_publish_order() {
  local all_features_metadata="${ARTIFACT_DIR}/lane-all-features/metadata.json"
  if [[ ! -f "$all_features_metadata" ]]; then
    echo "ERROR [SOURCE_ORDER_METADATA_MISSING]: all-features metadata is required" >&2
    return 1
  fi
  local planner_receipt="${ARTIFACT_DIR}/source-workspace-publish-plan.raw.json"
  local planner_log="${ARTIFACT_DIR}/source-workspace-publish-plan.log"
  CURRENT_LOG_OVERLAY_ROOT="$OVERLAY_ROOT"
  CURRENT_LOG_REPOSITORY_ROOT="$TRUSTED_ROOT"
  if ! run_logged "$planner_log" \
    run_trusted_publish_planner \
      --mode audit \
      --scope workspace \
      --metadata "$all_features_metadata" \
      --source-sha "$CANDIDATE_TREE_SHA" \
      --allow-dirty \
      --output "$planner_receipt"
  then
    echo "ERROR [SOURCE_ORDER_PLANNER_FAILED]: bd-8nqz.6 planner could not derive workspace order" >&2
    return 1
  fi
  jq -S '
    [
      .packages[]
      | {
          position,
          name,
          internal_dependencies
        }
    ]
    | sort_by(.position)
  ' "$planner_receipt" >"${ARTIFACT_DIR}/source-workspace-publish-order.json"
  source_publish_order_passes "${ARTIFACT_DIR}/source-workspace-publish-order.json"
  SOURCE_PUBLISH_ORDER_SHA256="$(
    canonical_json_hash "${ARTIFACT_DIR}/source-workspace-publish-order.json"
  )"
}

write_observed_contract_values() {
  local lane_name receipt_path
  local resolved_map="{}"
  local dependency_map="{}"
  local public_api_map="{}"
  while IFS= read -r lane_name; do
    [[ -z "$lane_name" ]] && continue
    receipt_path="${ARTIFACT_DIR}/qg10-lane-${lane_name}.json"
    if [[ ! -f "$receipt_path" ]]; then
      continue
    fi
    resolved_map="$(
      jq -cn \
        --argjson prior "$resolved_map" \
        --arg lane "$lane_name" \
        --arg value "$(
          jq -c '.observed.resolved_features' "$receipt_path" \
            | sha256sum \
            | awk '{print $1}'
        )" \
        '$prior + {($lane): $value}'
    )"
    dependency_map="$(
      jq -cn \
        --argjson prior "$dependency_map" \
        --arg lane "$lane_name" \
        --arg value "$(
          jq -c '.observed.dependency_packages' "$receipt_path" \
            | sha256sum \
            | awk '{print $1}'
        )" \
        '$prior + {($lane): $value}'
    )"
    public_api_map="$(
      jq -cn \
        --argjson prior "$public_api_map" \
        --arg lane "$lane_name" \
        --arg value "$(jq -r '.observed.public_api_sha256 // ""' "$receipt_path")" \
        '$prior + {($lane): (if $value == "" then null else $value end)}'
    )"
  done < <(selected_lanes)

  jq -n \
    --arg cargo_lock_sha256 "$CARGO_LOCK_SHA256" \
    --arg workspace_manifest_sha256 "$WORKSPACE_MANIFEST_SHA256" \
    --arg facade_manifest_sha256 "$FACADE_MANIFEST_SHA256" \
    --arg toolchain_sha256 "$TOOLCHAIN_SHA256" \
    --arg facade_features_sha256 "$FACADE_FEATURES_SHA256" \
    --arg target_inventory_sha256 "$TARGET_INVENTORY_SHA256" \
    --arg schema_inventory_sha256 "$SCHEMA_INVENTORY_SHA256" \
    --arg source_publish_order_sha256 "${SOURCE_PUBLISH_ORDER_SHA256:-}" \
    --argjson resolved_features_by_lane "$resolved_map" \
    --argjson dependency_packages_by_lane "$dependency_map" \
    --argjson public_api_by_lane "$public_api_map" \
    '{
      cargo_lock_sha256: $cargo_lock_sha256,
      workspace_manifest_sha256: $workspace_manifest_sha256,
      facade_manifest_sha256: $facade_manifest_sha256,
      toolchain_sha256: $toolchain_sha256,
      facade_features_sha256: $facade_features_sha256,
      target_inventory_sha256: $target_inventory_sha256,
      schema_inventory_sha256: $schema_inventory_sha256,
      source_publish_order_sha256: (
        if $source_publish_order_sha256 == ""
        then null
        else $source_publish_order_sha256
        end
      ),
      resolved_features_by_lane: $resolved_features_by_lane,
      dependency_packages_by_lane: $dependency_packages_by_lane,
      public_api_by_lane: $public_api_by_lane
    }' >"${ARTIFACT_DIR}/observed-contract-values.json"
}

reviewed_contract_matches_observed() {
  local observed_path="${ARTIFACT_DIR}/observed-contract-values.json"
  if ! cmp -s \
    <(jq -S '.reviewed_inventory' "$CONTRACT_FILE") \
    <(jq -S . "$observed_path")
  then
    echo "ERROR [REVIEWED_INVENTORY_DRIFT]: generated facade/dependency/API/schema/order hashes differ" >&2
    return 1
  fi
}

write_final_receipt() {
  local overall_outcome="$1"
  local failure_phase="$2"
  local contract_sha256 lane_receipts_json artifact_bindings_json
  contract_sha256="$(sha256_file "$CONTRACT_FILE")"
  lane_receipts_json="$(
    local lane_name receipt_path
    local receipt_paths=()
    while IFS= read -r lane_name; do
      [[ -z "$lane_name" ]] && continue
      receipt_path="${ARTIFACT_DIR}/qg10-lane-${lane_name}.json"
      [[ -f "$receipt_path" ]] && receipt_paths+=("$receipt_path")
    done < <(selected_lanes)
    if (( ${#receipt_paths[@]} == 0 )); then
      printf '[]\n'
    else
      jq -s '.' "${receipt_paths[@]}"
    fi
  )"

  artifact_bindings_json="$(
    local bound_file
    local bindings="[]"
    while IFS= read -r bound_file; do
      [[ -z "$bound_file" ]] && continue
      bindings="$(
        jq -cn \
          --argjson prior "$bindings" \
          --arg file "$(basename "$bound_file")" \
          --arg sha256 "$(sha256_file "$bound_file")" \
          '$prior + [{file: $file, sha256: $sha256}]'
      )"
    done < <({
      printf '%s\n' \
        "${ARTIFACT_DIR}/facade-features.json" \
        "${ARTIFACT_DIR}/facade-targets.json" \
        "${ARTIFACT_DIR}/schema-inventory.json" \
        "${ARTIFACT_DIR}/source-workspace-publish-order.json" \
        "${ARTIFACT_DIR}/observed-contract-values.json"
      local lane_name
      while IFS= read -r lane_name; do
        [[ -n "$lane_name" ]] \
          && printf '%s\n' "${ARTIFACT_DIR}/qg10-lane-${lane_name}.json"
      done < <(selected_lanes)
    } | while IFS= read -r candidate_file; do
      [[ -f "$candidate_file" ]] && printf '%s\n' "$candidate_file"
    done)
    printf '%s\n' "$bindings"
  )"

  local complete_matrix=false
  local release_admissible=false
  if [[ "$SELECTED_LANE" == "all" ]]; then
    complete_matrix=true
  fi
  if [[ \
    "$MODE" == "gate" \
    && "$complete_matrix" == true \
    && "$overall_outcome" == "pass" \
    && "$BASE_CLEAN" == true \
    && "$OVERLAY_CLEAN" == true \
  ]]; then
    release_admissible=true
  fi

  local receipt_path="${ARTIFACT_DIR}/qg10-overlay-receipt.json"
  jq -n \
    --arg schema "$SCHEMA_VERSION" \
    --arg run_id "$RUN_ID" \
    --arg mode "$MODE" \
    --arg selected_lane "$SELECTED_LANE" \
    --arg outcome "$overall_outcome" \
    --arg failure_phase "$failure_phase" \
    --arg base_git_sha "$BASE_GIT_SHA" \
    --arg base_tree_sha "$BASE_TREE_SHA" \
    --arg patch_sha256 "$PATCH_SHA256" \
    --arg candidate_tree_sha "$CANDIDATE_TREE_SHA" \
    --arg synthetic_commit_sha "$SYNTHETIC_COMMIT_SHA" \
    --arg trusted_runner_sha256 "$(sha256_file "${TRUSTED_ROOT}/scripts/check_feature_matrix_overlay.sh")" \
    --arg contract_relative_path "$CONTRACT_RELATIVE_PATH" \
    --arg contract_sha256 "$contract_sha256" \
    --argjson base_clean "$BASE_CLEAN" \
    --argjson overlay_clean "$OVERLAY_CLEAN" \
    --argjson complete_matrix "$complete_matrix" \
    --argjson release_admissible "$release_admissible" \
    --argjson lane_receipts "$lane_receipts_json" \
    --argjson artifact_bindings "$artifact_bindings_json" \
    '{
      schema: $schema,
      run_id: $run_id,
      mode: $mode,
      selected_lane: $selected_lane,
      outcome: $outcome,
      failure_phase: (if $failure_phase == "" then null else $failure_phase end),
      candidate: {
        base_git_sha: $base_git_sha,
        base_tree_sha: $base_tree_sha,
        canonical_flip_patch_sha256: $patch_sha256,
        candidate_tree_sha: $candidate_tree_sha,
        synthetic_commit_sha: $synthetic_commit_sha,
        base_clean: $base_clean,
        overlay_clean: $overlay_clean
      },
      validator: {
        trusted_runner_sha256: $trusted_runner_sha256,
        contract_relative_path: $contract_relative_path,
        contract_sha256: $contract_sha256
      },
      claim: {
        complete_matrix: $complete_matrix,
        release_admissible: $release_admissible,
        source_workspace_only: true,
        performance_claim: false,
        registry_or_package_claim: false,
        registry_or_package_owner: "bd-8nqz.6"
      },
      lane_receipts: $lane_receipts,
      artifact_bindings: $artifact_bindings
    }' >"$receipt_path"
  content_address_file "$receipt_path" "qg10-overlay-receipt" >/dev/null
}

safe_receipt_relative_path() {
  case "$1" in
    ""|/*|../*|*/../*|*/..|./*|*/./*|*/.)
      return 1
      ;;
  esac
}

verify_receipt_directory() {
  local receipt_dir="$1"
  local stable_receipt="${receipt_dir}/qg10-overlay-receipt.json"
  if [[ ! -f "$stable_receipt" ]]; then
    echo "ERROR [RECEIPT_MISSING]: no stable overlay receipt" >&2
    return 1
  fi
  if ! jq -e --arg schema "$SCHEMA_VERSION" '.schema == $schema' "$stable_receipt" >/dev/null; then
    echo "ERROR [RECEIPT_SCHEMA_INVALID]: stable receipt schema mismatch" >&2
    return 1
  fi

  local stable_sha addressed_count addressed_path
  stable_sha="$(sha256_file "$stable_receipt")"
  addressed_count="$(
    find "$receipt_dir" \
      -maxdepth 1 \
      -type f \
      -name "qg10-overlay-receipt.${stable_sha}.json" \
      | wc -l \
      | awk '{print $1}'
  )"
  if [[ "$addressed_count" != "1" ]]; then
    echo "ERROR [RECEIPT_ADDRESS_MISSING]: expected exactly one content-addressed receipt" >&2
    return 1
  fi
  addressed_path="${receipt_dir}/qg10-overlay-receipt.${stable_sha}.json"
  if ! cmp -s "$stable_receipt" "$addressed_path"; then
    echo "ERROR [RECEIPT_ADDRESS_DRIFT]: stable and content-addressed bytes differ" >&2
    return 1
  fi

  local bound_file expected_sha actual_sha
  while IFS=$'\t' read -r bound_file expected_sha; do
    if ! safe_receipt_relative_path "$bound_file"; then
      echo "ERROR [RECEIPT_PATH_UNSAFE]: '$bound_file'" >&2
      return 1
    fi
    if [[ ! -f "${receipt_dir}/${bound_file}" ]]; then
      echo "ERROR [RECEIPT_ARTIFACT_MISSING]: '$bound_file'" >&2
      return 1
    fi
    actual_sha="$(sha256_file "${receipt_dir}/${bound_file}")"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      echo "ERROR [RECEIPT_ARTIFACT_TAMPER]: '$bound_file'" >&2
      return 1
    fi
  done < <(
    jq -r '.artifact_bindings[] | [.file, .sha256] | @tsv' "$stable_receipt"
  )

  local lane_receipt lane_name lane_path lane_sha lane_addressed_path lane_log
  while IFS= read -r lane_receipt; do
    [[ -z "$lane_receipt" ]] && continue
    lane_name="$(jq -r '.lane' <<<"$lane_receipt")"
    case "$lane_name" in
      *[!A-Za-z0-9._-]*|"")
        echo "ERROR [LANE_RECEIPT_NAME_UNSAFE]: '$lane_name'" >&2
        return 1
        ;;
    esac
    lane_path="${receipt_dir}/qg10-lane-${lane_name}.json"
    if [[ ! -f "$lane_path" ]]; then
      echo "ERROR [LANE_RECEIPT_MISSING]: '$lane_name'" >&2
      return 1
    fi
    lane_sha="$(sha256_file "$lane_path")"
    lane_addressed_path="${receipt_dir}/qg10-lane-${lane_name}.${lane_sha}.json"
    if [[ ! -f "$lane_addressed_path" ]] || ! cmp -s "$lane_path" "$lane_addressed_path"; then
      echo "ERROR [LANE_RECEIPT_ADDRESS_DRIFT]: '$lane_name'" >&2
      return 1
    fi
    while IFS=$'\t' read -r lane_log expected_sha; do
      if ! safe_receipt_relative_path "$lane_log"; then
        echo "ERROR [LANE_LOG_PATH_UNSAFE]: '$lane_log'" >&2
        return 1
      fi
      local located_log="${receipt_dir}/${lane_log}"
      if [[ ! -f "$located_log" || "$(sha256_file "$located_log")" != "$expected_sha" ]]; then
        echo "ERROR [LANE_LOG_TAMPER]: '$lane_log'" >&2
        return 1
      fi
    done < <(jq -r '.logs[] | [.file, .sha256] | @tsv' <<<"$lane_receipt")
  done < <(jq -c '.lane_receipts[]' "$stable_receipt")

  echo "[qg10] receipt verification PASS"
}

run_self_test() {
  local self_test_root
  self_test_root="$(mktemp -d /tmp/frankensearch-qg10-overlay-self-test.XXXXXX)"
  local contract_path="${TRUSTED_ROOT}/docs/contracts/quill-facade-source-contract-v1.json"
  validate_static_contract "$contract_path"

  local missing_lane_contract="${self_test_root}/missing-lane.json"
  jq 'del(.required_lanes[0])' "$contract_path" >"$missing_lane_contract"
  if validate_static_contract "$missing_lane_contract" >/dev/null 2>&1; then
    echo "ERROR [SELF_TEST_FALSE_PASS]: missing lane was accepted" >&2
    return 1
  fi

  local lexical_packages="${self_test_root}/lexical-packages.json"
  jq -n '["frankensearch", "frankensearch-quill", "tantivy"]' >"$lexical_packages"
  if dependency_contract_passes "$contract_path" lexical "$lexical_packages" >/dev/null 2>&1; then
    echo "ERROR [SELF_TEST_FALSE_PASS]: unified-feature Tantivy masking was accepted" >&2
    return 1
  fi

  local valid_order="${self_test_root}/valid-order.json"
  local invalid_order="${self_test_root}/invalid-order.json"
  jq -n '[
    {position: 0, name: "frankensearch-core", internal_dependencies: []},
    {position: 1, name: "frankensearch-quill", internal_dependencies: ["frankensearch-core"]},
    {position: 2, name: "frankensearch-fusion", internal_dependencies: ["frankensearch-quill"]},
    {position: 3, name: "frankensearch", internal_dependencies: ["frankensearch-fusion", "frankensearch-quill"]}
  ]' >"$valid_order"
  jq '.[1].position = 3 | .[3].position = 1 | sort_by(.position)' \
    "$valid_order" >"$invalid_order"
  source_publish_order_passes "$valid_order"
  if source_publish_order_passes "$invalid_order" >/dev/null 2>&1; then
    echo "ERROR [SELF_TEST_FALSE_PASS]: late Quill order was accepted" >&2
    return 1
  fi

  local old_limit="$MAX_LOG_BYTES"
  local bounded_log="${self_test_root}/bounded.log"
  MAX_LOG_BYTES=64
  CURRENT_LOG_OVERLAY_ROOT="/data/projects/private-overlay"
  CURRENT_LOG_REPOSITORY_ROOT="/data/projects/private-repository"
  if run_logged "$bounded_log" bash -c \
    'printf "/data/projects/private-overlay/%0128d\\n" 0' >/dev/null 2>&1
  then
    echo "ERROR [SELF_TEST_FALSE_PASS]: over-limit log was accepted" >&2
    return 1
  fi
  if [[ "$(wc -c <"$bounded_log" | awk '{print $1}')" != "64" ]]; then
    echo "ERROR [SELF_TEST_LOG_BOUND]: bounded log did not stop at exactly 64 bytes" >&2
    return 1
  fi
  if rg -F -q "/data/projects/private-overlay" "$bounded_log"; then
    echo "ERROR [SELF_TEST_REDACTION]: host overlay path leaked" >&2
    return 1
  fi
  MAX_LOG_BYTES="$old_limit"

  local patch_a="${self_test_root}/patch-a.diff"
  local patch_b="${self_test_root}/patch-b.diff"
  printf 'canonical patch bytes\n' >"$patch_a"
  printf 'tampered patch bytes\n' >"$patch_b"
  if [[ "$(sha256_file "$patch_a")" == "$(sha256_file "$patch_b")" ]]; then
    echo "ERROR [SELF_TEST_PATCH_HASH]: distinct patches hashed equally" >&2
    return 1
  fi

  local saved_trusted_root="$TRUSTED_ROOT"
  local saved_artifact_dir="$ARTIFACT_DIR"
  local saved_base_git_sha="$BASE_GIT_SHA"
  local saved_patch_path="$PATCH_PATH"
  local saved_expected_patch_sha256="$EXPECTED_PATCH_SHA256"
  local fixture_repo="${self_test_root}/git-overlay-repository"
  local fixture_artifacts="${self_test_root}/git-overlay-artifacts"
  mkdir -p \
    "$fixture_repo/docs/contracts" \
    "$fixture_artifacts"
  git -C "$fixture_repo" init -q
  git -C "$fixture_repo" config user.name "QG10 Self Test"
  git -C "$fixture_repo" config user.email "qg10-self-test@invalid.example"
  printf 'base\n' >"${fixture_repo}/tracked.txt"
  cp "$contract_path" \
    "${fixture_repo}/${CONTRACT_RELATIVE_PATH}"
  git -C "$fixture_repo" add tracked.txt "$CONTRACT_RELATIVE_PATH"
  git -C "$fixture_repo" commit -q -m "self-test base"
  local fixture_patch="${self_test_root}/git-overlay.diff"
  printf '%s\n' \
    'diff --git a/tracked.txt b/tracked.txt' \
    '--- a/tracked.txt' \
    '+++ b/tracked.txt' \
    '@@ -1 +1,2 @@' \
    ' base' \
    '+candidate' \
    >"$fixture_patch"
  TRUSTED_ROOT="$fixture_repo"
  ARTIFACT_DIR="$fixture_artifacts"
  BASE_GIT_SHA="$(git -C "$fixture_repo" rev-parse HEAD)"
  PATCH_PATH="$fixture_patch"
  EXPECTED_PATCH_SHA256="$(sha256_file "$fixture_patch")"
  BASE_CLEAN=false
  OVERLAY_CLEAN=false
  validate_overlay_inputs
  materialize_overlay
  if [[ "$(sed -n '2p' "${OVERLAY_ROOT}/tracked.txt")" != "candidate" ]]; then
    echo "ERROR [SELF_TEST_OVERLAY_CONTENT]: synthetic worktree lacks patch bytes" >&2
    return 1
  fi
  printf '\n' >>"$fixture_patch"
  if validate_overlay_inputs >/dev/null 2>&1; then
    echo "ERROR [SELF_TEST_FALSE_PASS]: canonical patch drift was accepted" >&2
    return 1
  fi
  EXPECTED_PATCH_SHA256="$(sha256_file "$fixture_patch")"
  git -C "$fixture_repo" commit -q --allow-empty -m "self-test base drift"
  if validate_overlay_inputs >/dev/null 2>&1; then
    echo "ERROR [SELF_TEST_FALSE_PASS]: base HEAD drift was accepted" >&2
    return 1
  fi
  BASE_GIT_SHA="$(git -C "$fixture_repo" rev-parse HEAD)"
  validate_overlay_inputs
  printf 'untracked drift\n' >"${fixture_repo}/dirty-marker"
  if validate_overlay_inputs >/dev/null 2>&1; then
    echo "ERROR [SELF_TEST_FALSE_PASS]: dirty base checkout was accepted" >&2
    return 1
  fi
  TRUSTED_ROOT="$saved_trusted_root"
  ARTIFACT_DIR="$saved_artifact_dir"
  BASE_GIT_SHA="$saved_base_git_sha"
  PATCH_PATH="$saved_patch_path"
  EXPECTED_PATCH_SHA256="$saved_expected_patch_sha256"
  BASE_CLEAN=false
  OVERLAY_CLEAN=false

  local receipt_dir="${self_test_root}/receipt-positive"
  mkdir -p "${receipt_dir}/lane-self-test"
  printf 'bound artifact\n' >"${receipt_dir}/bound.json"
  printf 'bounded lane log\n' >"${receipt_dir}/lane-self-test/probe.log"
  jq -n \
    --arg schema "$LANE_SCHEMA_VERSION" \
    --arg log_sha256 "$(sha256_file "${receipt_dir}/lane-self-test/probe.log")" \
    '{
      schema: $schema,
      lane: "self-test",
      logs: [{file: "lane-self-test/probe.log", sha256: $log_sha256}]
    }' >"${receipt_dir}/qg10-lane-self-test.json"
  content_address_file \
    "${receipt_dir}/qg10-lane-self-test.json" \
    "qg10-lane-self-test" >/dev/null
  jq -n \
    --arg schema "$SCHEMA_VERSION" \
    --arg bound_sha256 "$(sha256_file "${receipt_dir}/bound.json")" \
    --arg lane_sha256 "$(sha256_file "${receipt_dir}/qg10-lane-self-test.json")" \
    --slurpfile lane "${receipt_dir}/qg10-lane-self-test.json" \
    '{
      schema: $schema,
      lane_receipts: $lane,
      artifact_bindings: [
        {file: "bound.json", sha256: $bound_sha256},
        {file: "qg10-lane-self-test.json", sha256: $lane_sha256}
      ]
    }' >"${receipt_dir}/qg10-overlay-receipt.json"
  content_address_file \
    "${receipt_dir}/qg10-overlay-receipt.json" \
    "qg10-overlay-receipt" >/dev/null
  verify_receipt_directory "$receipt_dir" >/dev/null
  printf 'tamper\n' >>"${receipt_dir}/lane-self-test/probe.log"
  if verify_receipt_directory "$receipt_dir" >/dev/null 2>&1; then
    echo "ERROR [SELF_TEST_FALSE_PASS]: lane log tamper was accepted" >&2
    return 1
  fi

  local artifact_tamper_dir="${self_test_root}/receipt-artifact-tamper"
  mkdir -p "$artifact_tamper_dir"
  printf 'bound artifact\n' >"${artifact_tamper_dir}/bound.json"
  jq -n \
    --arg schema "$SCHEMA_VERSION" \
    --arg sha256 "$(sha256_file "${artifact_tamper_dir}/bound.json")" \
    '{
      schema: $schema,
      lane_receipts: [],
      artifact_bindings: [{file: "bound.json", sha256: $sha256}]
    }' >"${artifact_tamper_dir}/qg10-overlay-receipt.json"
  content_address_file \
    "${artifact_tamper_dir}/qg10-overlay-receipt.json" \
    "qg10-overlay-receipt" >/dev/null
  verify_receipt_directory "$artifact_tamper_dir" >/dev/null
  printf 'tamper\n' >>"${artifact_tamper_dir}/bound.json"
  if verify_receipt_directory "$artifact_tamper_dir" >/dev/null 2>&1; then
    echo "ERROR [SELF_TEST_FALSE_PASS]: artifact tamper was accepted" >&2
    return 1
  fi

  local manifest_tamper_dir="${self_test_root}/receipt-manifest-tamper"
  mkdir -p "$manifest_tamper_dir"
  printf 'bound artifact\n' >"${manifest_tamper_dir}/bound.json"
  jq -n \
    --arg schema "$SCHEMA_VERSION" \
    --arg sha256 "$(sha256_file "${manifest_tamper_dir}/bound.json")" \
    '{
      schema: $schema,
      lane_receipts: [],
      artifact_bindings: [{file: "bound.json", sha256: $sha256}]
    }' >"${manifest_tamper_dir}/qg10-overlay-receipt.json"
  content_address_file \
    "${manifest_tamper_dir}/qg10-overlay-receipt.json" \
    "qg10-overlay-receipt" >/dev/null
  printf '\n' >>"${manifest_tamper_dir}/qg10-overlay-receipt.json"
  if verify_receipt_directory "$manifest_tamper_dir" >/dev/null 2>&1; then
    echo "ERROR [SELF_TEST_FALSE_PASS]: generated-manifest tamper was accepted" >&2
    return 1
  fi

  echo "[qg10] self-test PASS"
}

if [[ "$MODE" == "self-test" ]]; then
  run_self_test
  exit 0
fi

if [[ -z "$ARTIFACT_DIR" ]]; then
  echo "ERROR [ARTIFACT_DIR_REQUIRED]: pass --artifact-dir" >&2
  exit 2
fi
ARTIFACT_DIR="$(realpath -m "$ARTIFACT_DIR")"

if [[ "$MODE" == "verify" ]]; then
  if [[ ! -d "$ARTIFACT_DIR" ]]; then
    echo "ERROR [ARTIFACT_DIR_MISSING]: cannot verify '$ARTIFACT_DIR'" >&2
    exit 2
  fi
  verify_receipt_directory "$ARTIFACT_DIR"
  exit 0
fi

prepare_empty_artifact_dir
validate_overlay_inputs
materialize_overlay
CURRENT_LOG_OVERLAY_ROOT="$OVERLAY_ROOT"
CURRENT_LOG_REPOSITORY_ROOT="$TRUSTED_ROOT"
validate_static_contract "$CONTRACT_FILE"

overall_outcome="pass"
overall_failure_phase=""
if ! capture_global_inventory; then
  overall_outcome="fail"
  overall_failure_phase="global-inventory"
fi

if [[ "$overall_outcome" == "pass" ]]; then
  while IFS= read -r lane_name; do
    [[ -z "$lane_name" ]] && continue
    if ! run_lane "$lane_name"; then
      overall_outcome="fail"
      [[ -z "$overall_failure_phase" ]] && overall_failure_phase="lane-${lane_name}"
    fi
  done < <(selected_lanes)
fi

if [[ "$SELECTED_LANE" == "all" ]]; then
  if ! derive_source_publish_order; then
    overall_outcome="fail"
    [[ -z "$overall_failure_phase" ]] && overall_failure_phase="source-publish-order"
  fi
else
  echo "[qg10] partial lane: source-workspace order is deferred to the all-lane receipt"
fi

write_observed_contract_values
if [[ "$MODE" == "gate" ]] && ! reviewed_contract_matches_observed; then
  overall_outcome="fail"
  [[ -z "$overall_failure_phase" ]] && overall_failure_phase="reviewed-contract"
fi

write_final_receipt "$overall_outcome" "$overall_failure_phase"
verify_receipt_directory "$ARTIFACT_DIR"

if [[ "$overall_outcome" != "pass" ]]; then
  echo "[qg10] FAIL (${overall_failure_phase})" >&2
  exit 1
fi

echo "[qg10] PASS"
