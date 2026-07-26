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
# no-change mechanism. A new KEEP needs the executing ELF/binary SHA-256.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="staged"
SINCE_REF=""
CANDIDATE=""
SURFACE=""
LEDGER_OVERRIDE=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/check_ledger_null_control.sh --candidate <lever> --surface <target>
  scripts/check_ledger_null_control.sh [--staged | --since <ref> | --all]
  scripts/check_ledger_null_control.sh --install-hook

Options:
  --candidate <text>  Search NEGATIVE_EVIDENCE before proposing a lever.
  --surface <text>    Function/path/target surface paired with --candidate.
  --staged            Gate newly added staged rows (default; pre-commit).
  --since <ref>       Gate rows added between merge-base(ref, HEAD) and HEAD.
  --all               Mechanical whole-ledger report; never blocks.
  --ledger <path>     Override the default ledger set (test/diagnostic use).
  --install-hook      Point this checkout at the tracked .githooks directory.

Row contract:
  REJECT: record either
    * "A/A null: <numeric evidence> ... same invocation", or
    * a counted instructions/cycles/syscalls/allocations/faults line explicitly
      saying the count is unchanged, identical, flat, or the same.
  KEEP: record "ELF sha256: <64 hex>" or "binary sha256: <64 hex>".

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
    function note_line(line,    low, absent, metric, decisive) {
      low = tolower(line)

      if (low ~ /same[- ]invocation|same binary invocation/) {
        same_invocation = 1
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
    }
    function flush(    upper, explicit_reject, explicit_keep, exempt,
                       is_reject, is_keep, new_entry, has_null) {
      if (header == "") return
      new_entry = (mode == "all" || added[start_line])
      if (!new_entry) {
        header = ""; evidence = ""; return
      }

      checked++
      upper = toupper(header)
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

      if (is_reject && !has_null && !counted_mechanism) {
        violations++
        if (mode != "all" || violations <= 20) {
          print "BLOCKED REJECT " ledger ":" start_line
          print "  " header
          print "  missing: same-invocation numeric A/A null OR counted no-change mechanism"
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

      header = ""; evidence = ""; decision_lines = ""
      numeric_null = 0; same_invocation = 0
      counted_mechanism = 0; binary_sha = 0
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
unchanged mechanism. A new KEEP must identify the executing ELF/binary SHA-256.
Never use cv_pct as the decision gate.
EOF
  fi
  return "${status}"
}

case "${MODE}" in
  candidate)
    candidate_preflight
    ;;
  install-hook)
    install_hook
    ;;
  staged|since|all)
    lint_ledgers
    ;;
esac
