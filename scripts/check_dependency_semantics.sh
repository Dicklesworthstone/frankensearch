#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES_FILE="$ROOT_DIR/.beads/issues.jsonl"

usage() {
  cat <<USAGE
Usage: scripts/check_dependency_semantics.sh [--mode unit|integration|all] [--issues <path>]

Checks dependency-semantics policy invariants for bd-17dv retrofit scope and
bd-1pkl composition-matrix linkage policy.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --issues)
      ISSUES_FILE="${2:-}"
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

if [[ ! -f "$ISSUES_FILE" ]]; then
  echo "ERROR: issues file not found: $ISSUES_FILE" >&2
  exit 2
fi

case "$MODE" in
  unit|integration|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|integration|all)" >&2
    exit 2
    ;;
esac

RETROFIT_TARGETS=(bd-z3j bd-i37 bd-2rq bd-2u4 bd-2tv bd-6sj bd-1co bd-sot)
COMPOSITION_TARGETS=(bd-1pkl)
DATA="$(jq -cs '.' "$ISSUES_FILE")"
FAILURES=0

check_unit() {
  echo "[unit] checking target-level dependency semantics"

  for issue_id in "${RETROFIT_TARGETS[@]}"; do
    local dep_type
    dep_type="$(jq -r --arg id "$issue_id" '
      ([.[] | select(.id == $id)] | .[0]) as $issue
      | (($issue.dependencies // []) | map(select(.depends_on_id == "bd-3un")) | .[0].type) // ""
    ' <<<"$DATA")"

    if [[ "$dep_type" != "parent-child" ]]; then
      echo "[unit][FAIL] $issue_id has bd-3un dependency type '$dep_type' (expected 'parent-child')"
      FAILURES=$((FAILURES + 1))
    else
      echo "[unit][OK]   $issue_id uses parent-child for bd-3un"
    fi

    local comments
    comments="$(jq -r --arg id "$issue_id" '
      ([.[] | select(.id == $id)] | .[0]) as $issue
      | ($issue.comments // []) | map(.text) | join("\n")
    ' <<<"$DATA")"

    if ! grep -Fq "[bd-17dv retrofit] DEP_SEMANTICS" <<<"$comments"; then
      echo "[unit][FAIL] $issue_id missing retrofit DEP_SEMANTICS annotation comment"
      FAILURES=$((FAILURES + 1))
    else
      echo "[unit][OK]   $issue_id has retrofit DEP_SEMANTICS annotation"
    fi

    if ! grep -Eq 'HARD_DEP[[:space:]]+bd-' <<<"$comments"; then
      echo "[unit][FAIL] $issue_id missing at least one HARD_DEP example in annotation"
      FAILURES=$((FAILURES + 1))
    fi
  done

  echo "[unit] checking composition-matrix dependency semantics"

  for issue_id in "${COMPOSITION_TARGETS[@]}"; do
    local dep_type
    dep_type="$(jq -r --arg id "$issue_id" '
      ([.[] | select(.id == $id)] | .[0]) as $issue
      | (($issue.dependencies // []) | map(select(.depends_on_id == "bd-3un.52")) | .[0].type) // ""
    ' <<<"$DATA")"

    if [[ -z "$dep_type" ]]; then
      echo "[unit][FAIL] $issue_id missing required dependency edge to bd-3un.52"
      FAILURES=$((FAILURES + 1))
    else
      echo "[unit][OK]   $issue_id links to bd-3un.52 using dependency type '$dep_type'"
    fi

    local comments
    comments="$(jq -r --arg id "$issue_id" '
      ([.[] | select(.id == $id)] | .[0]) as $issue
      | ($issue.comments // []) | map(.text) | join("\n")
    ' <<<"$DATA")"

    if ! grep -Fq "[bd-1pkl composition-matrix] DEP_SEMANTICS" <<<"$comments"; then
      echo "[unit][FAIL] $issue_id missing composition DEP_SEMANTICS annotation comment"
      FAILURES=$((FAILURES + 1))
    else
      echo "[unit][OK]   $issue_id has composition DEP_SEMANTICS annotation"
    fi

    if ! grep -Eq 'MATRIX_LINK[[:space:]:]+bd-3un\.52' <<<"$comments"; then
      echo "[unit][FAIL] $issue_id missing MATRIX_LINK to bd-3un.52 in annotation"
      FAILURES=$((FAILURES + 1))
    fi

    if ! grep -Eq 'FALLBACK_SEMANTICS' <<<"$comments"; then
      echo "[unit][FAIL] $issue_id missing FALLBACK_SEMANTICS in annotation"
      FAILURES=$((FAILURES + 1))
    fi

    if ! grep -Eq 'INTERACTION_TEST_PLAN' <<<"$comments"; then
      echo "[unit][FAIL] $issue_id missing INTERACTION_TEST_PLAN in annotation"
      FAILURES=$((FAILURES + 1))
    fi
  done
}

check_integration() {
  echo "[integration] checking ambiguous blocker patterns"

  local scoped_ambiguous
  scoped_ambiguous="$(jq -r '
    .[] as $issue
    | (($issue.dependencies // [])[]? | select(.depends_on_id == "bd-3un" and .type == "blocks") | $issue.id)
  ' <<<"$DATA" | grep -E '^(bd-z3j|bd-i37|bd-2rq|bd-2u4|bd-2tv|bd-6sj|bd-1co|bd-sot)$' || true)"

  if [[ -n "$scoped_ambiguous" ]]; then
    echo "[integration][FAIL] retrofit-scope issues still using ambiguous bd-3un blocker edges:"
    echo "$scoped_ambiguous" | sed 's/^/  - /'
    FAILURES=$((FAILURES + 1))
  else
    echo "[integration][OK]   retrofit-scope issues have no bd-3un blocker edges"
  fi

  local global_candidates
  global_candidates="$(jq -r '
    .[] as $issue
    | (($issue.dependencies // [])[]? | select(.depends_on_id == "bd-3un" and .type == "blocks") | $issue.id)
  ' <<<"$DATA" | sort -u || true)"

  if [[ -n "$global_candidates" ]]; then
    echo "[integration][INFO] candidate ambiguous epic blockers outside retrofit scope (review queue):"
    echo "$global_candidates" | sed 's/^/  - /'
  else
    echo "[integration][OK]   no global bd-3un blocker edges found"
  fi

  local missing_composition_link
  missing_composition_link="$(jq -r '
    .[] as $issue
    | select($issue.id == "bd-1pkl")
    | ((($issue.dependencies // []) | map(select(.depends_on_id == "bd-3un.52")) | length) == 0)
  ' <<<"$DATA")"

  if [[ "$missing_composition_link" == "true" ]]; then
    echo "[integration][FAIL] bd-1pkl is missing dependency linkage to bd-3un.52"
    FAILURES=$((FAILURES + 1))
  else
    echo "[integration][OK]   bd-1pkl composition dependency linkage is present"
  fi
}

if [[ "$MODE" == "unit" || "$MODE" == "all" ]]; then
  check_unit
fi
if [[ "$MODE" == "integration" || "$MODE" == "all" ]]; then
  check_integration
fi

if [[ "$FAILURES" -gt 0 ]]; then
  echo "Result: FAIL ($FAILURES violation(s))"
  exit 1
fi

echo "Result: PASS"
