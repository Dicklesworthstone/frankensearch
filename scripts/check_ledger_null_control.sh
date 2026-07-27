#!/usr/bin/env bash
# Candidate and write-side preflight for the performance ledgers.
#
# Exit 0 = clear, exit 2 = BLOCKED, exit 64 = usage or repository error.
#
# The six resurrection classes are:
#   VALID-PROFILE, VALID-MECHANISM, VALID-AB,
#   VOID-CV, VOID-ZEROSELF, VOID-NONULL.
#
# This guard deliberately does not infer a verdict from a loose keyword hit.
# It gates the complete newly added section from the staged/committed blob.
# A new REJECT needs a positive, same-invocation A/A record or a counted
# no-change mechanism. A new KEEP needs the executing ELF/binary SHA-256 and
# an explicit comparison class. Only a side-by-side, same-invocation run
# against the named actual legacy incumbent is campaign/competitive evidence;
# an in-repo before/after comparison is SELF-SPEEDUP maintenance.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="staged"
SINCE_REF=""
CANDIDATE=""
SURFACE=""
LEDGER_OVERRIDE=""
SELF_CHECK_FIXTURE=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/check_ledger_null_control.sh --candidate <lever> --surface <target>
  scripts/check_ledger_null_control.sh [--staged | --since <ref> | --all]
  scripts/check_ledger_null_control.sh --selfcheck
  scripts/check_ledger_null_control.sh --install-hook

Options:
  --candidate <text>  Search NEGATIVE_EVIDENCE before proposing a lever.
  --surface <text>    Function/path/target surface paired with --candidate.
  --staged            Gate newly added staged rows (default; pre-commit).
  --since <ref>       Gate rows added between merge-base(ref, HEAD) and HEAD.
  --all               Mechanical whole-ledger report; never blocks.
  --selfcheck         Exercise fail-closed synthetic contract cases.
  --ledger <path>     Override the default ledger set (test/diagnostic use).
  --install-hook      Point this checkout at the tracked .githooks directory.

Row contract:
  REJECT: record either
    * "A/A null: <numeric evidence> ... same invocation", or
    * a counted instructions/cycles/syscalls/allocations/faults line explicitly
      saying the count is unchanged, identical, flat, or the same.
  KEEP: record "ELF sha256: <64 hex>" or "binary sha256: <64 hex>", plus:
    * "Comparison class: SELF-SPEEDUP" for in-repo before/after maintenance; or
    * "Comparison class: INCUMBENT" with "Actual legacy incumbent: <identity>",
      a numeric incumbent ratio, a same-invocation numeric A/A null, and an
      explicit statement that candidate and incumbent ran side-by-side in that
      same invocation.
  SELF-SPEEDUP rows must not claim a campaign or competitive win.

Exit 0 means clear. Exit 2 means BLOCKED. Exit 64 means usage/IO failure.
USAGE
}

need_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "ERROR: $1 requires a value" >&2
    exit 64
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate)
      need_value "$@"
      MODE="candidate"
      CANDIDATE="$2"
      shift 2
      ;;
    --surface)
      need_value "$@"
      SURFACE="$2"
      shift 2
      ;;
    --staged)
      MODE="staged"
      shift
      ;;
    --since)
      need_value "$@"
      MODE="since"
      SINCE_REF="$2"
      shift 2
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --selfcheck)
      MODE="selfcheck"
      shift
      ;;
    --ledger)
      need_value "$@"
      LEDGER_OVERRIDE="$2"
      shift 2
      ;;
    --install-hook)
      MODE="install-hook"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

resolve_ledger() {
  local requested="$1"
  if [[ "${requested}" == /* ]]; then
    LEDGER_ABS="${requested}"
  else
    LEDGER_ABS="${ROOT_DIR}/${requested}"
  fi
  if [[ ! -f "${LEDGER_ABS}" ]]; then
    echo "ERROR: ledger not found: ${LEDGER_ABS}" >&2
    exit 64
  fi
  LEDGER_REL="${LEDGER_ABS#"${ROOT_DIR}/"}"
  if [[ "${LEDGER_REL}" == "${LEDGER_ABS}" ]]; then
    echo "ERROR: ledger must be inside ${ROOT_DIR}: ${LEDGER_ABS}" >&2
    exit 64
  fi
}

install_hook() {
  local tracked_hook="${ROOT_DIR}/.githooks/pre-commit"
  local configured
  if [[ ! -x "${tracked_hook}" ]]; then
    echo "ERROR: tracked pre-commit hook is missing or not executable: ${tracked_hook}" >&2
    return 64
  fi
  configured="$(git -C "${ROOT_DIR}" config --get core.hooksPath || true)"
  if [[ -n "${configured}" && "${configured}" != ".githooks" ]]; then
    echo "BLOCKED: refusing to replace existing core.hooksPath=${configured}" >&2
    return 2
  fi
  git -C "${ROOT_DIR}" config core.hooksPath .githooks
  echo "[ledger-preflight] installed: core.hooksPath=.githooks"
}

candidate_preflight() {
  resolve_ledger "${LEDGER_OVERRIDE:-docs/NEGATIVE_EVIDENCE.md}"
  if [[ -z "${CANDIDATE}" || -z "${SURFACE}" ]]; then
    echo "ERROR: --candidate and --surface are both required" >&2
    return 64
  fi

  awk -v candidate="${CANDIDATE}" -v surface="${SURFACE}" \
      -v ledger="${LEDGER_REL}" '
    function is_entry_heading(line) {
      return line ~ /^##+ 20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/
    }
    function flush(    low, retry_text) {
      if (header == "") return
      low = tolower(text)
      if (index(low, tolower(candidate)) == 0 &&
          index(low, tolower(surface)) == 0) {
        header = ""; text = ""; retry = ""; return
      }
      hits++
      print "  " ledger ":" start_line
      print "    " header
      retry_text = (retry == "" ? "(none recorded)" : retry)
      print "    retry: " retry_text
      print ""
      header = ""; text = ""; retry = ""
    }
    {
      if (is_entry_heading($0)) {
        flush()
        header = $0
        sub(/^##+ /, "", header)
        start_line = FNR
        text = $0 "\n"
        next
      }
      if (header != "") {
        text = text $0 "\n"
        if (retry == "" && tolower($0) ~ /retry (predicate|condition|only|if|on|when)/) {
          retry = $0
          sub(/^[[:space:]>*-]+/, "", retry)
        }
      }
    }
    END {
      flush()
      if (hits == 0) {
        print "[ledger-preflight] CLEAR — no prior row matched candidate=\"" \
              candidate "\" surface=\"" surface "\""
        exit 0
      }
      print "[ledger-preflight] BLOCKED — prior negative-evidence row(s) cover candidate=\"" \
            candidate "\" surface=\"" surface "\""
      print "Satisfy and cite the retry predicate, or switch veins."
      exit 2
    }
  ' "${LEDGER_ABS}"
}

diff_stream() {
  local rel="$1"
  # A sentinel avoids the empty-first-file FNR==NR trap in awk.
  printf '%s\n' "__FRANKENSEARCH_DIFF_SENTINEL__"
  case "${MODE}" in
    staged)
      git -C "${ROOT_DIR}" diff --cached -U0 -- "${rel}"
      ;;
    since)
      git -C "${ROOT_DIR}" diff -U0 "${SINCE_REF}...HEAD" -- "${rel}"
      ;;
    all)
      # No hunks are needed in report mode; awk checks every entry.
      ;;
    selfcheck)
      # Synthetic self-check entries are all considered newly added.
      ;;
  esac
}

blob_stream() {
  local rel="$1"
  local abs="$2"
  case "${MODE}" in
    staged)
      if ! git -C "${ROOT_DIR}" show ":${rel}" 2>/dev/null; then
        # An untracked override is useful for diagnostics; default ledgers are
        # tracked, so reaching this branch during normal pre-commit is an error.
        if [[ -n "${LEDGER_OVERRIDE}" ]]; then
          command cat "${abs}"
        else
          return 64
        fi
      fi
      ;;
    since|all)
      command cat "${abs}"
      ;;
    selfcheck)
      printf '%s\n' "${SELF_CHECK_FIXTURE}"
      ;;
  esac
}

lint_one() {
  local requested="$1"
  resolve_ledger "${requested}"

  awk -v mode="${MODE}" -v ledger="${LEDGER_REL}" '
    function has_hex64(line,    fields, n, i) {
      n = split(line, fields, /[^[:xdigit:]]+/)
      for (i = 1; i <= n; i++) {
        if (length(fields[i]) == 64) return 1
      }
      return 0
    }
    function is_entry_heading(line) {
      # This ledger has used both ## and ### for entries. Date-bearing headings
      # are entries; undated headings are grouping labels.
      return line ~ /^##+ 20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/
    }
    function note_line(line,    low, absent, metric, decisive, value) {
      low = tolower(line)

      if (low ~ /same[- ]invocation|same binary invocation/) {
        same_invocation = 1
      }
      if (low ~ /side[- ]by[- ]side/) {
        side_by_side = 1
      }

      absent = (low ~ /no (a\/a|null[- ]control)|without (an )?(a\/a|null[- ]control)|\
(a\/a|null[- ]control).*(missing|absent|not recorded|none)/)
      if (!absent &&
          low ~ /(a\/a|null[- ]control|null floor|null_median_ratio)/ &&
          line ~ /[0-9][.][0-9]/) {
        numeric_null = 1
      }

      metric = (low ~ /(instructions?|cycles?|syscalls?|allocations?|page faults?|faults?)/)
      decisive = (low ~ /(unchanged|identical|flat|same count|counts? (were |are )?the same|\
no (counted )?change|0([.]0+)?% change)/)
      absent = (low ~ /(not measured|not recorded|missing|unavailable|no counted mechanism)/)
      if (metric && decisive && !absent) {
        counted_mechanism = 1
      }

      if (low ~ /(elf|binary|executable)/ &&
          low ~ /sha-?256|sha256/ &&
          has_hex64(line)) {
        binary_sha = 1
      }

      if (low ~ /(comparison|win)[-_ ]*class[[:space:]]*:[[:space:]]*self[-_ ]*speedup/) {
        self_speedup_class = 1
      }
      if (low ~ /(comparison|win)[-_ ]*class[[:space:]]*:[[:space:]]*incumbent/) {
        incumbent_class = 1
      }
      if (low ~ /actual legacy incumbent[[:space:]]*:/) {
        value = low
        sub(/^.*actual legacy incumbent[[:space:]]*:[[:space:]]*/, "", value)
        sub(/[[:space:]]*$/, "", value)
        if (value != "" &&
            value !~ /^(n\/a|none|unknown|not recorded|no counterpart)([[:space:]]|$)/) {
          actual_incumbent = 1
        }
      }
      if ((low ~ /incumbent.*ratio|ratio.*incumbent/) &&
          line ~ /[0-9][.][0-9]/) {
        incumbent_ratio = 1
      }
      if (low ~ /(campaign|competitive)[-_ ]+(win|speedup|claim|advantage)/ ||
          low ~ /(beats?|faster than|dominates).*actual legacy incumbent/) {
        competitive_claim = 1
      }

      if (low ~ /invalid[-_ ]?cv|cv[_ -]*only|cv[_ -]*gate|\
coefficient of variation[_ -]*gate/ ||
          low ~ /(decision|verdict|outcome|status).*(cv|coefficient of variation)/ ||
          low ~ /(reject|no[- ]?ship|no[- ]?land).*(cv|coefficient of variation)/ ||
          low ~ /(cv|coefficient of variation).*(reject|invalid|inadmissible)/) {
        cv_verdict = 1
      }
    }
    function flush(    upper, explicit_reject, explicit_keep, exempt,
                       is_reject, is_keep, new_entry, has_null,
                       incumbent_complete) {
      if (header == "") return
      new_entry = (mode == "all" || mode == "selfcheck" || added[start_line])
      if (!new_entry) {
        header = ""; evidence = ""; return
      }

      checked++
      # Verdict-bearing body lines are authoritative too. A neutral heading
      # with "Decision: KEEP" must not bypass KEEP admission.
      upper = toupper(header "\n" decision_lines)
      explicit_reject = (upper ~ /REJECT|REFUT|NO[- ]?SHIP|NO[- ]?LAND|\
NOT A (WIN|LEVER)|REGRESS|WASH/)
      explicit_keep = (upper ~ /KEEP|LANDED|SHIPPED|MEASURED WIN/)
      exempt = (upper ~ /SURVEY|ROUTE[- ]NEXT|CORRECTION|RETRACT|RESOLVED|\
AUDIT|INVENTORY|METHODOLOGY|BLOCKED|UNTIMED|INVALID|HOLD/)

      is_keep = explicit_keep
      is_reject = explicit_reject
      if (ledger ~ /NEGATIVE_EVIDENCE[.]md$/ &&
          !explicit_keep && !exempt) {
        is_reject = 1
      }
      has_null = numeric_null && same_invocation
      incumbent_complete = actual_incumbent && incumbent_ratio &&
                            has_null && side_by_side

      if (is_reject && !has_null && !counted_mechanism) {
        violations++
        if (mode != "all" || violations <= 20) {
          print "BLOCKED REJECT " ledger ":" start_line
          print "  " header
          print "  missing: same-invocation numeric A/A null OR counted no-change mechanism"
        }
      }
      if (is_reject && cv_verdict) {
        violations++
        if (mode != "all" || violations <= 20) {
          print "BLOCKED CV-VERDICT " ledger ":" start_line
          print "  " header
          print "  forbidden: CV-based decision; gate on median CI versus the A/A null floor"
        }
      }
      if (is_keep && !binary_sha) {
        violations++
        if (mode != "all" || violations <= 20) {
          print "BLOCKED KEEP " ledger ":" start_line
          print "  " header
          print "  missing: executing ELF/binary SHA-256"
        }
      }
      if (is_keep && !self_speedup_class && !incumbent_class) {
        violations++
        if (mode != "all" || violations <= 20) {
          print "BLOCKED KEEP-CLASS " ledger ":" start_line
          print "  " header
          print "  missing: Comparison class: SELF-SPEEDUP or INCUMBENT"
        }
      }
      if (is_keep && self_speedup_class && incumbent_class) {
        violations++
        if (mode != "all" || violations <= 20) {
          print "BLOCKED KEEP-CLASS " ledger ":" start_line
          print "  " header
          print "  conflict: a KEEP cannot be both SELF-SPEEDUP and INCUMBENT"
        }
      }
      if (is_keep && competitive_claim && !incumbent_class) {
        violations++
        if (mode != "all" || violations <= 20) {
          print "BLOCKED SELF-SPEEDUP-CLAIM " ledger ":" start_line
          print "  " header
          print "  forbidden: self-speedup maintenance presented as campaign/competitive output"
        }
      }
      if (is_keep && incumbent_class && !incumbent_complete) {
        violations++
        if (mode != "all" || violations <= 20) {
          print "BLOCKED INCUMBENT-WIN " ledger ":" start_line
          print "  " header
          print "  missing: named actual legacy incumbent, numeric incumbent ratio,"
          print "           same-invocation numeric A/A null, or side-by-side execution"
        }
      }

      header = ""; evidence = ""; decision_lines = ""
      numeric_null = 0; same_invocation = 0
      counted_mechanism = 0; binary_sha = 0; cv_verdict = 0
      side_by_side = 0; self_speedup_class = 0; incumbent_class = 0
      actual_incumbent = 0; incumbent_ratio = 0; competitive_claim = 0
    }

    FNR == NR {
      if ($0 ~ /^@@ /) {
        hunk = $0
        sub(/^@@ -[^ ]+ [+]/, "", hunk)
        sub(/ .*/, "", hunk)
        split(hunk, range, ",")
        first = range[1] + 0
        count = (length(range[2]) ? range[2] + 0 : 1)
        for (i = 0; i < count; i++) added[first + i] = 1
      }
      next
    }

    {
      if (is_entry_heading($0)) {
        flush()
        header = $0
        sub(/^##+ /, "", header)
        start_line = FNR
        note_line($0)
        next
      }
      if (header != "") {
        note_line($0)
        low = tolower($0)
        if (low ~ /(verdict|decision|outcome|status)[[:space:]:-]/) {
          decision_lines = decision_lines "\n" $0
        }
      }
    }
    END {
      flush()
      if (mode == "all") {
        print "[ledger-gate] mechanical report " ledger \
              ": checked=" (checked + 0) " violations=" (violations + 0)
        exit 0
      }
      if (violations > 0) exit 2
      print "[ledger-gate] OK " ledger ": checked_new_rows=" (checked + 0)
      exit 0
    }
  ' <(diff_stream "${LEDGER_REL}") <(blob_stream "${LEDGER_REL}" "${LEDGER_ABS}")
}

lint_ledgers() {
  local ledgers
  local status=0

  if [[ "${MODE}" == "since" && -z "${SINCE_REF}" ]]; then
    echo "ERROR: --since requires a git ref" >&2
    return 64
  fi
  if [[ "${MODE}" == "since" ]] &&
     ! git -C "${ROOT_DIR}" rev-parse --verify --quiet "${SINCE_REF}^{commit}" >/dev/null; then
    echo "ERROR: --since ref is not a commit: ${SINCE_REF}" >&2
    return 64
  fi
  if [[ -n "${LEDGER_OVERRIDE}" ]]; then
    ledgers=("${LEDGER_OVERRIDE}")
  else
    ledgers=("docs/NEGATIVE_EVIDENCE.md" "docs/PERF_LEDGER.md")
  fi

  for ledger in "${ledgers[@]}"; do
    resolve_ledger "${ledger}"
    if [[ "${MODE}" == "staged" && -z "${LEDGER_OVERRIDE}" ]] &&
       ! git -C "${ROOT_DIR}" cat-file -e ":${LEDGER_REL}" 2>/dev/null; then
      echo "ERROR: staged ledger blob is unavailable: ${LEDGER_REL}" >&2
      return 64
    fi
    if lint_one "${ledger}"; then
      local rc=0
    else
      local rc=$?
      if [[ ${rc} -eq 64 ]]; then
        return 64
      fi
      status=2
    fi
  done

  if [[ ${status} -eq 2 ]]; then
    cat >&2 <<'EOF'

[ledger-gate] BLOCKED. A new REJECT must distinguish the lever from the
harness with a numeric same-invocation A/A null, or refute it with a counted
unchanged mechanism. A new KEEP must identify the executing ELF/binary SHA-256
and classify the comparison. SELF-SPEEDUP is maintenance, never campaign or
competitive output. INCUMBENT requires the named actual legacy incumbent,
numeric ratio, same-invocation A/A null, and side-by-side execution in that
invocation. CV may be diagnostic only; the decision gate is median CI versus
the A/A null floor.
EOF
  fi
  return "${status}"
}

run_selfcheck_case() {
  local label="$1"
  local expected="$2"
  local ledger="$3"
  local fixture="$4"
  local output rc

  SELF_CHECK_TOTAL=$((SELF_CHECK_TOTAL + 1))
  SELF_CHECK_FIXTURE="${fixture}"
  if output="$(lint_one "${ledger}" 2>&1)"; then
    rc=0
  else
    rc=$?
  fi

  if [[ ${rc} -ne ${expected} ]]; then
    echo "[ledger-selfcheck] FAIL ${ledger} ${label}: expected exit ${expected}, got ${rc}" >&2
    printf '%s\n' "${output}" >&2
    return 1
  fi
  SELF_CHECK_PASSED=$((SELF_CHECK_PASSED + 1))
  echo "[ledger-selfcheck] PASS ${ledger} ${label}: exit ${rc}"
}

run_selfcheck_suite() {
  local ledger="$1"
  local sha
  sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

  run_selfcheck_case "VOID-NONULL reject is blocked" 2 "${ledger}" $'### 2099-01-01 — REJECT: synthetic no-null row\nA/B median ratio: 1.001\nDecision: reject as no improvement.' || SELF_CHECK_FAILED=1
  run_selfcheck_case "CV-only reject is blocked" 2 "${ledger}" $'### 2099-01-02 — REJECT / INVALID-CV: synthetic CV row\nA/A null: 1.000 [0.900, 1.100], same invocation\nA/B median CI: 0.980 [0.950, 1.020]\nDecision: reject because arm CV exceeded 5%.' || SELF_CHECK_FAILED=1
  run_selfcheck_case "null-floor reject is admitted" 0 "${ledger}" $'### 2099-01-03 — REJECT: synthetic null-contained row\nA/A null: 1.000 [0.980, 1.020], same invocation\nA/B median CI: 1.001 [0.990, 1.010]\nDecision: no-ship because the effect remains inside the A/A null floor.' || SELF_CHECK_FAILED=1
  run_selfcheck_case "counted-mechanism reject is admitted" 0 "${ledger}" $'### 2099-01-04 — REJECT: synthetic counted-mechanism row\nInstructions count unchanged: baseline 100, candidate 100.\nDecision: reject because the counted mechanism removed no work.' || SELF_CHECK_FAILED=1
  run_selfcheck_case "KEEP without ELF SHA is blocked" 2 "${ledger}" $'### 2099-01-05 — KEEP: synthetic unbound binary\nComparison class: SELF-SPEEDUP\nA/B median CI: 0.900 [0.880, 0.920].' || SELF_CHECK_FAILED=1
  run_selfcheck_case "KEEP without comparison class is blocked" 2 "${ledger}" $"### 2099-01-06 — KEEP: synthetic unclassified binary
ELF sha256: ${sha}
A/B median CI: 0.900 [0.880, 0.920]." || SELF_CHECK_FAILED=1
  run_selfcheck_case "SELF-SPEEDUP maintenance is admitted" 0 "${ledger}" $"### 2099-01-07 — KEEP: synthetic maintenance speedup
Comparison class: SELF-SPEEDUP
ELF sha256: ${sha}
A/B median CI: 0.900 [0.880, 0.920]." || SELF_CHECK_FAILED=1
  run_selfcheck_case "SELF-SPEEDUP competitive claim is blocked" 2 "${ledger}" $"### 2099-01-08 — KEEP: synthetic mislabeled result
Comparison class: SELF-SPEEDUP
ELF sha256: ${sha}
Competitive win: 1.11x.
A/B median CI: 0.900 [0.880, 0.920]." || SELF_CHECK_FAILED=1
  run_selfcheck_case "INCUMBENT without actual identity is blocked" 2 "${ledger}" $"### 2099-01-09 — KEEP: synthetic unnamed incumbent
Comparison class: INCUMBENT
ELF sha256: ${sha}
A/A null: 1.000 [0.990, 1.010], same invocation
Incumbent ratio: 0.900.
Invocation: candidate and incumbent ran side-by-side in the same invocation." || SELF_CHECK_FAILED=1
  run_selfcheck_case "INCUMBENT split invocation is blocked" 2 "${ledger}" $"### 2099-01-10 — KEEP: synthetic split incumbent
Comparison class: INCUMBENT
Actual legacy incumbent: Tantivy 0.26.1 shipping backend
ELF sha256: ${sha}
A/A null: 1.000 [0.990, 1.010].
Incumbent ratio: 0.900.
Invocation: candidate and incumbent ran side-by-side in separate invocations." || SELF_CHECK_FAILED=1
  run_selfcheck_case "same-invocation INCUMBENT win is admitted" 0 "${ledger}" $"### 2099-01-11 — KEEP: synthetic incumbent result
Comparison class: INCUMBENT
Actual legacy incumbent: Tantivy 0.26.1 shipping backend
ELF sha256: ${sha}
A/A null: 1.000 [0.990, 1.010], same invocation
Incumbent ratio: 0.900.
Invocation: candidate and incumbent ran side-by-side in the same invocation." || SELF_CHECK_FAILED=1
  run_selfcheck_case "body-only KEEP without class is blocked" 2 "${ledger}" $"### 2099-01-12 — synthetic body-level verdict
ELF sha256: ${sha}
Decision: KEEP." || SELF_CHECK_FAILED=1
  run_selfcheck_case "body-only SELF-SPEEDUP is admitted" 0 "${ledger}" $"### 2099-01-13 — synthetic body-level maintenance
Comparison class: SELF-SPEEDUP
ELF sha256: ${sha}
Decision: KEEP." || SELF_CHECK_FAILED=1
}

run_selfcheck() {
  SELF_CHECK_FAILED=0
  SELF_CHECK_TOTAL=0
  SELF_CHECK_PASSED=0

  run_selfcheck_suite "docs/NEGATIVE_EVIDENCE.md"
  run_selfcheck_suite "docs/PERF_LEDGER.md"

  if [[ ${SELF_CHECK_FAILED} -ne 0 ]]; then
    echo "[ledger-selfcheck] BLOCKED: one or more contract cases failed" >&2
    return 2
  fi
  echo "[ledger-selfcheck] OK: ${SELF_CHECK_PASSED}/${SELF_CHECK_TOTAL} contract cases across both ledgers"
}

case "${MODE}" in
  candidate)
    candidate_preflight
    ;;
  install-hook)
    install_hook
    ;;
  selfcheck)
    run_selfcheck
    ;;
  staged|since|all)
    lint_ledgers
    ;;
esac
