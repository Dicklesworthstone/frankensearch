#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES_FILE="$ROOT_DIR/.beads/issues.jsonl"
FAILURES=0

RULE_POLICY_UNIT="SDOC-POLICY-000"
RULE_MISSING_MATRIX_MARKER="SDOC-MATRIX-001"
RULE_MISSING_MATRIX_SECTION="SDOC-MATRIX-002"
RULE_MISSING_EXCEPTION_MARKER="SDOC-MATRIX-003"
RULE_MISSING_COMPOSITION_MARKER="SDOC-MATRIX-004"
RULE_MISSING_COMPOSITION_SECTION="SDOC-MATRIX-005"
RULE_INVALID_COMPOSITION_EXCEPTION="SDOC-MATRIX-006"
RULE_SCANNER_PARSE_ERROR="SDOC-SCAN-001"
RULE_SCANNER_NORMALIZATION_ERROR="SDOC-SCAN-002"

DATA="[]"
NORMALIZED="[]"

report_finding() {
  local rule_id="$1"
  local severity="$2"
  local bead_id="$3"
  local message="$4"
  local fix_hint="$5"

  jq -cn \
    --arg rule_id "$rule_id" \
    --arg severity "$severity" \
    --arg bead_id "$bead_id" \
    --arg message "$message" \
    --arg fix_hint "$fix_hint" \
    '{
      rule_id: $rule_id,
      severity: $severity,
      bead_id: $bead_id,
      message: $message,
      fix_hint: $fix_hint
    }'
}

classify_bead_class() {
  local issue_type="$1"
  local labels_csv="${2:-}"

  case ",$labels_csv," in
    *,release-gate,*|*,gate,*)
      echo "gate"
      return
      ;;
  esac

  case "$issue_type" in
    task|feature|bug) echo "implementation" ;;
    epic) echo "program" ;;
    question|docs) echo "exploratory" ;;
    *) echo "implementation" ;;
  esac
}

severity_for_missing_matrix() {
  local bead_class="$1"

  case "$bead_class" in
    implementation|gate) echo "error" ;;
    program) echo "warning" ;;
    exploratory) echo "info" ;;
    *) echo "error" ;;
  esac
}

yes_no_from_grep() {
  local pattern="$1"
  local haystack="$2"
  if grep -Eq "$pattern" <<<"$haystack"; then
    echo "yes"
  else
    echo "no"
  fi
}

contains_literal() {
  local needle="$1"
  local haystack="$2"
  if grep -Fq "$needle" <<<"$haystack"; then
    echo "yes"
  else
    echo "no"
  fi
}

is_advanced_ranking_control_issue() {
  local issue_type="$1"
  local labels_csv="$2"
  local title="$3"
  local description="$4"
  local haystack

  case "$issue_type" in
    task|feature|bug) ;;
    *) echo "no"; return ;;
  esac

  haystack="$(printf '%s\n%s\n%s' "$labels_csv" "$title" "$description" | tr '[:upper:]' '[:lower:]')"

  if grep -Eq '(^|,)(program|planning)(,|$)' <<<"$(tr '[:upper:]' '[:lower:]' <<<"$labels_csv")" \
    && ! grep -Eq '(^|,)(ranking|control|adaptive)(,|$)' <<<"$(tr '[:upper:]' '[:lower:]' <<<"$labels_csv")"; then
    echo "no"
    return
  fi

  if grep -Eq 'advanced ranking/control|ranking/control' <<<"$haystack"; then
    echo "yes"
    return
  fi

  if grep -Eq '(^|,)(ranking|control|adaptive)(,|$)' <<<"$(tr '[:upper:]' '[:lower:]' <<<"$labels_csv")"; then
    echo "yes"
    return
  fi

  echo "no"
}

composition_marker_present() {
  local comments="$1"
  contains_literal "[bd-1pkl composition-matrix] COMPOSITION_MATRIX" "$comments"
}

composition_exception_present() {
  local comments="$1"
  if [[ "$(contains_literal "[bd-264r test-matrix] EXCEPTION" "$comments")" == "yes" ]] \
    && grep -Eq 'rule_id[[:space:]:=]+SDOC-MATRIX-004' <<<"$comments"; then
    echo "yes"
  else
    echo "no"
  fi
}

composition_required_sections_present() {
  local comments="$1"
  local required=("MATRIX_LINK:" "FALLBACK_SEMANTICS:" "INTERACTION_TEST_PLAN:")

  for section in "${required[@]}"; do
    if ! grep -Fqi "$section" <<<"$comments"; then
      echo "no"
      return
    fi
  done
  echo "yes"
}

composition_semantics_present() {
  local comments="$1"

  if ! grep -Eq 'MATRIX_LINK:[[:space:]]*(bd-3un\.52|.*successor.*bd-3un\.52)' <<<"$comments"; then
    echo "no"
    return
  fi
  if ! grep -Eq 'FALLBACK_SEMANTICS:.*ON_EXHAUSTION|ON_EXHAUSTION' <<<"$comments"; then
    echo "no"
    return
  fi
  if ! grep -Eq 'INTERACTION_TEST_PLAN:.*interaction_(unit|integration)|interaction_(unit|integration)' <<<"$comments"; then
    echo "no"
    return
  fi
  echo "yes"
}

composition_exception_metadata_complete() {
  local comments="$1"
  local required=("rule_id" "owner" "justification" "expires_on" "follow_up_bead")

  for field in "${required[@]}"; do
    if ! grep -Eq "${field}[[:space:]:=]+" <<<"$comments"; then
      echo "no"
      return
    fi
  done
  echo "yes"
}

check_composition_requirement() {
  local issue_id="$1"
  local comments="$2"

  if [[ "$(composition_marker_present "$comments")" != "yes" ]] \
    && [[ "$(composition_exception_present "$comments")" != "yes" ]]; then
    report_finding \
      "$RULE_MISSING_COMPOSITION_MARKER" \
      "error" \
      "$issue_id" \
      "missing composition-matrix marker for advanced ranking/control bead" \
      "add '[bd-1pkl composition-matrix] COMPOSITION_MATRIX' with MATRIX_LINK/FALLBACK_SEMANTICS/INTERACTION_TEST_PLAN or add EXCEPTION for SDOC-MATRIX-004"
    FAILURES=$((FAILURES + 1))
    return
  fi

  if [[ "$(composition_exception_present "$comments")" == "yes" ]]; then
    if [[ "$(composition_exception_metadata_complete "$comments")" != "yes" ]]; then
      report_finding \
        "$RULE_INVALID_COMPOSITION_EXCEPTION" \
        "error" \
        "$issue_id" \
        "composition-matrix exception is missing required metadata fields" \
        "include rule_id, owner, justification, expires_on, and follow_up_bead in the EXCEPTION block"
      FAILURES=$((FAILURES + 1))
    fi
    return
  fi

  if [[ "$(composition_required_sections_present "$comments")" != "yes" ]]; then
    report_finding \
      "$RULE_MISSING_COMPOSITION_SECTION" \
      "error" \
      "$issue_id" \
      "composition-matrix annotation missing required sections" \
      "include MATRIX_LINK, FALLBACK_SEMANTICS, and INTERACTION_TEST_PLAN fields"
    FAILURES=$((FAILURES + 1))
    return
  fi

  if [[ "$(composition_semantics_present "$comments")" != "yes" ]]; then
    report_finding \
      "$RULE_MISSING_COMPOSITION_SECTION" \
      "error" \
      "$issue_id" \
      "composition-matrix annotation is missing concrete linkage/fallback/test details" \
      "ensure MATRIX_LINK points to bd-3un.52 (or successor), FALLBACK_SEMANTICS includes ON_EXHAUSTION, and INTERACTION_TEST_PLAN names interaction_unit/integration coverage"
    FAILURES=$((FAILURES + 1))
    return
  fi

  echo "[policy][OK] $issue_id satisfies composition-matrix gate contract"
}

assert_eq() {
  local actual="$1"
  local expected="$2"
  local context="$3"

  if [[ "$actual" != "$expected" ]]; then
    report_finding \
      "$RULE_POLICY_UNIT" \
      "error" \
      "policy-selftest" \
      "$context expected '$expected' but found '$actual'" \
      "update classify_bead_class()/severity_for_missing_matrix() to match policy contract"
    FAILURES=$((FAILURES + 1))
  else
    echo "[unit][OK]   $context"
  fi
}

run_policy_unit_tests() {
  echo "[unit] validating lint policy classification and severity mappings"

  assert_eq "$(classify_bead_class "task" "")" "implementation" "task bead class"
  assert_eq "$(classify_bead_class "epic" "")" "program" "epic bead class"
  assert_eq "$(classify_bead_class "question" "")" "exploratory" "question bead class"
  assert_eq \
    "$(classify_bead_class "task" "ci,release-gate,lint")" \
    "gate" \
    "release-gate label bead class"

  assert_eq \
    "$(severity_for_missing_matrix "implementation")" \
    "error" \
    "implementation missing-matrix severity"
  assert_eq \
    "$(severity_for_missing_matrix "program")" \
    "warning" \
    "program missing-matrix severity"
  assert_eq \
    "$(severity_for_missing_matrix "exploratory")" \
    "info" \
    "exploratory missing-matrix severity"
}

load_issue_data_from_file() {
  local issues_file="$1"
  local line_no=0
  local line parsed
  local -a parsed_rows=()

  while IFS= read -r line || [[ -n "$line" ]]; do
    line_no=$((line_no + 1))

    if [[ -z "$line" ]]; then
      report_finding \
        "$RULE_SCANNER_PARSE_ERROR" \
        "error" \
        "line:$line_no" \
        "empty JSONL record" \
        "remove blank lines from issues.jsonl"
      FAILURES=$((FAILURES + 1))
      continue
    fi

    if ! parsed="$(jq -c '.' <<<"$line" 2>/dev/null)"; then
      report_finding \
        "$RULE_SCANNER_PARSE_ERROR" \
        "error" \
        "line:$line_no" \
        "malformed JSON record" \
        "ensure each line is a valid JSON object"
      FAILURES=$((FAILURES + 1))
      continue
    fi

    parsed="$(jq -c --argjson source_line "$line_no" '. + {__source_line: $source_line}' <<<"$parsed")"
    parsed_rows+=("$parsed")
  done < "$issues_file"

  if [[ "${#parsed_rows[@]}" -eq 0 ]]; then
    DATA="[]"
    return
  fi

  DATA="$(printf '%s\n' "${parsed_rows[@]}" | jq -cs '.')"
}

normalize_issue_model() {
  NORMALIZED="$(jq -c '
    def scalar_text:
      if . == null then ""
      elif type == "string" then .
      elif type == "number" or type == "boolean" then tostring
      elif type == "array" then map(
        if type == "string" then . else tostring end
      ) | join("\n")
      elif type == "object" then
        if has("text") then (.text | scalar_text)
        elif has("message") then (.message | scalar_text)
        else tojson
        end
      else tostring
      end;

    def normalize_labels:
      (.labels // []) as $labels
      | if ($labels | type) == "array" then $labels else [$labels] end
      | map(tostring)
      | sort
      | unique;

    def normalize_comments:
      (.comments // []) as $comments
      | if ($comments | type) == "array" then $comments else [$comments] end
      | map(
          if type == "object" then {
            author: ((.author // .created_by // "") | tostring),
            text: ((.text // .message // .body // "") | scalar_text),
            created_at: ((.created_at // "") | tostring)
          }
          elif type == "string" then {
            author: "",
            text: .,
            created_at: ""
          }
          else {
            author: "",
            text: tostring,
            created_at: ""
          }
          end
        );

    def normalize_dependencies:
      (.dependencies // []) as $deps
      | if ($deps | type) == "array" then $deps else [$deps] end
      | map(
          if type == "object" then {
            depends_on_id: ((.depends_on_id // .id // .dependency_id // "") | tostring),
            dep_type: ((.type // .dependency_type // "") | tostring)
          }
          elif type == "string" then {
            depends_on_id: .,
            dep_type: ""
          }
          else {
            depends_on_id: "",
            dep_type: ""
          }
          end
        );

    map(
      . as $issue
      | (normalize_comments) as $comments
      | (normalize_dependencies) as $deps
      | {
          id: ((.id // "") | tostring),
          issue_type: ((.issue_type // "task") | tostring),
          status: ((.status // "open") | tostring),
          priority: ((.priority // 4) | tonumber? // 4),
          title: ((.title // "") | scalar_text),
          description: ((.description // "") | scalar_text),
          acceptance_criteria: ((.acceptance_criteria // "") | scalar_text),
          notes: ((.notes // "") | scalar_text),
          labels: normalize_labels,
          comments: $comments,
          comment_text: ($comments | map(.text) | join("\n")),
          dependencies: $deps,
          source_line: (.__source_line // -1)
        }
    )
    | sort_by(.id, .source_line)
  ' <<<"$DATA")"
}

validate_normalized_model() {
  local errors
  errors="$(jq -r '
    .[] as $issue
    | (
        if ($issue.id | length) == 0 then
          "missing_id\tline:\($issue.source_line)\tid"
        else
          empty
        end
      ),
      (
        $issue.dependencies[]
        | select((.dep_type | length) > 0 and (.depends_on_id | length) == 0)
        | "missing_dep_target\t\($issue.id)\tdependencies.depends_on_id"
      )
  ' <<<"$NORMALIZED")"

  if [[ -z "$errors" ]]; then
    return
  fi

  while IFS=$'\t' read -r err_code bead_id field_path; do
    [[ -z "$err_code" ]] && continue
    report_finding \
      "$RULE_SCANNER_NORMALIZATION_ERROR" \
      "error" \
      "$bead_id" \
      "normalization error ($err_code) at $field_path" \
      "fix malformed record fields in .beads/issues.jsonl"
    FAILURES=$((FAILURES + 1))
  done <<<"$errors"
}

run_scanner_unit_tests() {
  echo "[unit] validating deterministic scanner normalization"

  local test_data test_normalized
  test_data="$(jq -cn '
    [
      {
        id: "bd-z",
        issue_type: "task",
        status: "open",
        labels: "lint",
        comments: "single comment string",
        dependencies: {depends_on_id: "bd-a", type: "blocks"},
        acceptance_criteria: ["a", "b"]
      },
      {
        issue_type: "feature",
        status: "open",
        comments: [{text: "missing id comment"}],
        dependencies: [{type: "blocks"}]
      },
      {
        id: "bd-a",
        issue_type: "epic",
        status: "in_progress",
        labels: ["release-gate", "ci"],
        comments: [{author: "ops", text: "obj comment"}],
        dependencies: ["bd-root"]
      }
    ]
  ')"

  test_normalized="$(jq -c '
    def scalar_text:
      if . == null then ""
      elif type == "string" then .
      elif type == "number" or type == "boolean" then tostring
      elif type == "array" then map(
        if type == "string" then . else tostring end
      ) | join("\n")
      elif type == "object" then
        if has("text") then (.text | scalar_text)
        elif has("message") then (.message | scalar_text)
        else tojson
        end
      else tostring
      end;
    def normalize_labels:
      (.labels // []) as $labels
      | if ($labels | type) == "array" then $labels else [$labels] end
      | map(tostring)
      | sort
      | unique;
    def normalize_comments:
      (.comments // []) as $comments
      | if ($comments | type) == "array" then $comments else [$comments] end
      | map(
          if type == "object" then {
            author: ((.author // .created_by // "") | tostring),
            text: ((.text // .message // .body // "") | scalar_text),
            created_at: ((.created_at // "") | tostring)
          }
          elif type == "string" then {author: "", text: ., created_at: ""}
          else {author: "", text: tostring, created_at: ""}
          end
        );
    def normalize_dependencies:
      (.dependencies // []) as $deps
      | if ($deps | type) == "array" then $deps else [$deps] end
      | map(
          if type == "object" then {
            depends_on_id: ((.depends_on_id // .id // .dependency_id // "") | tostring),
            dep_type: ((.type // .dependency_type // "") | tostring)
          }
          elif type == "string" then {depends_on_id: ., dep_type: ""}
          else {depends_on_id: "", dep_type: ""}
          end
        );
    map(
      . as $issue
      | (normalize_comments) as $comments
      | (normalize_dependencies) as $deps
      | {
          id: ((.id // "") | tostring),
          issue_type: ((.issue_type // "task") | tostring),
          status: ((.status // "open") | tostring),
          priority: ((.priority // 4) | tonumber? // 4),
          title: ((.title // "") | scalar_text),
          description: ((.description // "") | scalar_text),
          acceptance_criteria: ((.acceptance_criteria // "") | scalar_text),
          notes: ((.notes // "") | scalar_text),
          labels: normalize_labels,
          comments: $comments,
          comment_text: ($comments | map(.text) | join("\n")),
          dependencies: $deps,
          source_line: (.__source_line // -1)
        }
    )
    | sort_by(.id, .source_line)
  ' <<<"$test_data")"

  assert_eq "$(jq -r 'length' <<<"$test_normalized")" "3" "normalized fixture count"
  assert_eq "$(jq -r '.[0].id' <<<"$test_normalized")" "" "deterministic sort with malformed id first"
  assert_eq \
    "$(jq -r '.[] | select(.id == "bd-z") | .acceptance_criteria' <<<"$test_normalized")" \
    "a"$'\n'"b" \
    "array acceptance criteria stringification"
  assert_eq \
    "$(jq -r '.[] | select(.id == "bd-z") | .comments[0].text' <<<"$test_normalized")" \
    "single comment string" \
    "string comment normalization"
  assert_eq \
    "$(jq -r '.[] | select(.id == "bd-a") | .dependencies[0].depends_on_id' <<<"$test_normalized")" \
    "bd-root" \
    "string dependency normalization"
  assert_eq \
    "$(jq -r '.[] | select(.id == "") | .dependencies[0].depends_on_id' <<<"$test_normalized")" \
    "" \
    "malformed dependency target normalization"
}

run_composition_policy_unit_tests() {
  echo "[unit] validating composition-matrix gate helpers"

  assert_eq \
    "$(is_advanced_ranking_control_issue "task" "ranking,policy" "Any title" "normal text")" \
    "yes" \
    "advanced ranking/control detection from labels"
  assert_eq \
    "$(is_advanced_ranking_control_issue "task" "ci,gate" "advanced ranking/control gate" "normal text")" \
    "yes" \
    "advanced ranking/control detection from title phrase"
  assert_eq \
    "$(is_advanced_ranking_control_issue "epic" "control-plane" "Epic title" "normal text")" \
    "no" \
    "non-implementation bead excluded from advanced gate"

  local compliant_comment
  compliant_comment=$'[bd-1pkl composition-matrix] COMPOSITION_MATRIX\nMATRIX_LINK: bd-3un.52\nFALLBACK_SEMANTICS: ON_EXHAUSTION -> lexical_only (reason_code=composition.matrix.exhausted)\nINTERACTION_TEST_PLAN: interaction_unit + interaction_integration lanes'
  assert_eq \
    "$(composition_marker_present "$compliant_comment")" \
    "yes" \
    "composition marker recognition"
  assert_eq \
    "$(composition_required_sections_present "$compliant_comment")" \
    "yes" \
    "composition required sections recognition"
  assert_eq \
    "$(composition_semantics_present "$compliant_comment")" \
    "yes" \
    "composition semantics completeness"

  local exception_comment
  exception_comment=$'[bd-264r test-matrix] EXCEPTION\nrule_id: SDOC-MATRIX-004\nowner: infra\njustification: temporary waiver\nexpires_on: 2026-03-31\nfollow_up_bead: bd-9999'
  assert_eq \
    "$(composition_exception_present "$exception_comment")" \
    "yes" \
    "composition exception marker recognition"
  assert_eq \
    "$(composition_exception_metadata_complete "$exception_comment")" \
    "yes" \
    "composition exception metadata recognition"
}

usage() {
  cat <<USAGE
Usage: scripts/check_bead_test_matrix.sh [--mode unit|integration|all] [--issues <path>]

Validates per-bead test matrix policy anchors for bd-264r.
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

case "$MODE" in
  unit|integration|all) ;;
  *)
    echo "ERROR: invalid mode '$MODE' (expected unit|integration|all)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$ISSUES_FILE" ]]; then
  echo "ERROR: issues file not found: $ISSUES_FILE" >&2
  exit 2
fi

load_issue_data_from_file "$ISSUES_FILE"
normalize_issue_model
validate_normalized_model

WAVE1=(bd-3un.31 bd-3un.32 bd-3un.40 bd-3un.52)
WAVE2_EXCEPTIONS=(bd-2hz.10 bd-2yu.8)
COMPOSITION_REQUIRED_TARGETS=(bd-1pkl)

check_wave1_matrix() {
  local issue_id="$1"
  local comments
  comments="$(jq -r --arg id "$issue_id" '
    ([.[] | select(.id == $id)] | .[0].comment_text) // ""
  ' <<<"$NORMALIZED")"

  if ! grep -Fq "[bd-264r test-matrix] TEST_MATRIX" <<<"$comments"; then
    report_finding \
      "$RULE_MISSING_MATRIX_MARKER" \
      "error" \
      "$issue_id" \
      "missing TEST_MATRIX annotation marker" \
      "add '[bd-264r test-matrix] TEST_MATRIX' comment with full matrix template"
    FAILURES=$((FAILURES + 1))
    return
  fi

  local required_sections=("Unit tests:" "Integration tests:" "E2E tests:" "Performance" "Logs/artifacts")
  for section in "${required_sections[@]}"; do
    if ! grep -Fqi "$section" <<<"$comments"; then
      report_finding \
        "$RULE_MISSING_MATRIX_SECTION" \
        "error" \
        "$issue_id" \
        "missing required section '$section'" \
        "add '$section' entry under the TEST_MATRIX annotation"
      FAILURES=$((FAILURES + 1))
    fi
  done

  echo "[unit][OK]   $issue_id has explicit TEST_MATRIX annotation"
}

check_wave2_exception() {
  local issue_id="$1"
  local comments
  comments="$(jq -r --arg id "$issue_id" '
    ([.[] | select(.id == $id)] | .[0].comment_text) // ""
  ' <<<"$NORMALIZED")"

  if ! grep -Fq "[bd-264r test-matrix] EXCEPTION" <<<"$comments"; then
    report_finding \
      "$RULE_MISSING_EXCEPTION_MARKER" \
      "error" \
      "$issue_id" \
      "missing EXCEPTION annotation marker" \
      "add '[bd-264r test-matrix] EXCEPTION' with explicit rationale"
    FAILURES=$((FAILURES + 1))
  else
    echo "[unit][OK]   $issue_id has explicit EXCEPTION rationale"
  fi
}

check_unit() {
  run_policy_unit_tests
  run_scanner_unit_tests
  run_composition_policy_unit_tests
  echo "[unit] validating wave-1 and wave-2 policy anchors"

  for issue_id in "${WAVE1[@]}"; do
    check_wave1_matrix "$issue_id"
  done

  for issue_id in "${WAVE2_EXCEPTIONS[@]}"; do
    check_wave2_exception "$issue_id"
  done

  echo "[unit] validating composition-matrix policy anchors"
  for issue_id in "${COMPOSITION_REQUIRED_TARGETS[@]}"; do
    local comments
    comments="$(jq -r --arg id "$issue_id" '
      ([.[] | select(.id == $id)] | .[0].comment_text) // ""
    ' <<<"$NORMALIZED")"
    check_composition_requirement "$issue_id" "$comments"
  done
}

check_integration() {
  echo "[integration] enforcing composition-matrix contract for advanced ranking/control beads"

  local advanced_count=0
  local row issue_id issue_type title description labels_csv comments

  while IFS= read -r row; do
    issue_id="$(jq -r '.id' <<<"$row")"
    issue_type="$(jq -r '.issue_type' <<<"$row")"
    title="$(jq -r '.title' <<<"$row")"
    description="$(jq -r '.description' <<<"$row")"
    labels_csv="$(jq -r '(.labels // []) | join(",")' <<<"$row")"
    comments="$(jq -r '.comment_text // ""' <<<"$row")"

    if [[ "$(is_advanced_ranking_control_issue "$issue_type" "$labels_csv" "$title" "$description")" != "yes" ]]; then
      continue
    fi

    advanced_count=$((advanced_count + 1))
    check_composition_requirement "$issue_id" "$comments"
  done < <(jq -c '
    .[]
    | select(.status == "open" or .status == "in_progress")
  ' <<<"$NORMALIZED")

  if [[ "$advanced_count" -eq 0 ]]; then
    echo "[integration][OK]   no advanced ranking/control candidates detected"
  else
    echo "[integration][OK]   evaluated $advanced_count advanced ranking/control candidate(s)"
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
