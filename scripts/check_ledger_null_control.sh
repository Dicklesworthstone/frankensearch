#!/usr/bin/env bash
# check_ledger_null_control.sh — refuse a REJECT row that cannot distinguish
# the lever from the harness.
#
# WHY THIS EXISTS
# ---------------
# The 2026-07-25 Ledger Resurrection audit classified 345 REJECT-verdict rows in
# docs/NEGATIVE_EVIDENCE.md under the frankenfs six-class taxonomy:
#
#     VOID-NONULL     205    <- 93% of all voids
#     VOID-ZEROSELF    11
#     VOID-UNMEASURED   4
#     VOID-CV           0
#     VOID total      220 / 345 = 63.8%
#
# VOID-NONULL means: an A/B ran, the row was rejected on a near-1.0 wall ratio,
# and NO A/A null control and NO counted mechanism were recorded — so the row
# cannot distinguish "the lever does nothing" from "the harness cannot see it".
# Those rejections are unfalsifiable, and because the house rule is to grep the
# ledger before proposing a lever, each one permanently suppresses a candidate
# that may never have been measured at all.
#
# The fleet's own evidence is that this DECAYS and that auditing once is not
# enough: repos that audited once and institutionalized the check sit at ~1.7%
# void; repos that never did sit at 25-91%. So this is the write-side gate.
#
# CONTRACT
# --------
# A newly added REJECT row must record at least one of:
#   (a) an A/A null control  — the effect can be compared against the bench's
#       own noise floor; or
#   (b) a counted mechanism  — instructions / cycles / syscalls / allocations /
#       page faults / "already auto-vectorizes" / "no work removed". A null
#       control cannot change the fact that no work was removed, so a mechanism
#       refutation is sound WITHOUT one (frankenfs VALID-MECHANISM).
#
# Rows that never obtained a measurement must say so (BLOCKED / UNTIMED /
# INVALID) rather than claiming REJECT — an unmeasured lever is not a rejected
# lever, and conflating them is what produced the VOID-UNMEASURED class.
#
# Exit codes:  0 = OK   2 = BLOCKED (contract violation)   1 = usage/IO error
#
# Only NEWLY ADDED rows are gated. The 220 historical void rows are left alone
# deliberately: repository Rule 1 forbids deleting them, and rewriting history
# is not the goal — stopping the bleeding is.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER="${ROOT_DIR}/docs/NEGATIVE_EVIDENCE.md"
MODE="staged"
SINCE_REF=""

usage() {
  cat <<'USAGE'
Usage: scripts/check_ledger_null_control.sh [--staged | --since <ref> | --all]
                                            [--ledger <path>]

  --staged        Gate rows added in the staged diff (default; pre-commit use).
  --since <ref>   Gate rows added since a git ref (CI use, e.g. origin/main).
  --all           Audit the whole ledger and REPORT ONLY (never blocks). Use to
                  measure the historical backlog, not as a gate.

Exit: 0 ok · 2 BLOCKED (a new REJECT row lacks a null control AND a mechanism)
      1 usage/IO error
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --staged) MODE="staged"; shift ;;
    --all)    MODE="all"; shift ;;
    --since)  MODE="since"; SINCE_REF="${2:-}"; shift 2 ;;
    --ledger) LEDGER="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -f "${LEDGER}" ]]; then
  echo "ERROR: ledger not found: ${LEDGER}" >&2
  exit 1
fi
if [[ "${MODE}" == "since" && -z "${SINCE_REF}" ]]; then
  echo "ERROR: --since requires a git ref" >&2
  exit 1
fi

REL_LEDGER="${LEDGER#"${ROOT_DIR}/"}"

# ─── Collect the text to inspect ────────────────────────────────────────────
# The ledger is append-only, so added lines from a diff reconstruct whole
# sections. Strip the leading '+' and drop diff headers.
collect() {
  case "${MODE}" in
    staged)
      git -C "${ROOT_DIR}" diff --cached -U0 -- "${REL_LEDGER}" 2>/dev/null \
        | sed -n 's/^+\([^+].*\)$/\1/p; s/^+$//p' || true
      ;;
    since)
      git -C "${ROOT_DIR}" diff -U0 "${SINCE_REF}" -- "${REL_LEDGER}" 2>/dev/null \
        | sed -n 's/^+\([^+].*\)$/\1/p; s/^+$//p' || true
      ;;
    all)
      cat "${LEDGER}"
      ;;
  esac
}

# Stream to a temp file rather than a shell variable: --all inspects ~16k lines,
# and round-tripping that through "$(...)" plus printf is pathologically slow.
WORK="$(mktemp)"
VIOL="$(mktemp)"
cleanup() { rm -f "${WORK}" "${VIOL}"; }
trap cleanup EXIT

collect >"${WORK}"
if [[ ! -s "${WORK}" ]]; then
  echo "[ledger-gate] no new ledger rows to check."
  exit 0
fi

# ─── Adjudicate each '### ' section ─────────────────────────────────────────
# A section is GATED when its header announces a rejected candidate. Rows that
# are surveys, corrections, keeps, or explicit blockers are not rejections.
awk '
  function flush_section() {
    if (header == "") return
    is_reject = (header ~ /REJECT|REJECTED|WASH|regress|REGRESS|not a win|NO-LAND|below.bar|BELOW-FLOOR/)
    is_exempt = (header ~ /SURVEY|ROUTE-NEXT|CORRECTION|RETRACT|RESOLVED|LANDED|KEEP|AUDIT|inventory|METHODOLOGY|Methodology|BLOCKED|UNTIMED|INVALID|HOLD|NULL \(profiling/)
    if (is_reject && !is_exempt) {
      has_null = (body ~ /A\/A/)
      has_mech = (body ~ /instruction|cycles|syscall|allocation|page fault|perf stat|auto-?vector|no work (is |was )?removed|same per-byte work|identical (instruction|work)|zero-gain/)
      if (!has_null && !has_mech) print header
    }
    header = ""; body = ""
  }
  /^### / { flush_section(); header = $0; body = ""; next }
  { body = body "\n" $0 }
  END { flush_section() }
' "${WORK}" >"${VIOL}"

if [[ ! -s "${VIOL}" ]]; then
  echo "[ledger-gate] OK — every new REJECT row records a null control or a counted mechanism."
  exit 0
fi

COUNT="$(grep -c '^###' "${VIOL}" || true)"

if [[ "${MODE}" == "all" ]]; then
  echo "[ledger-gate] REPORT (--all never blocks): ${COUNT} historical rows lack both."
  head -20 "${VIOL}"
  exit 0
fi

cat >&2 <<EOF
[ledger-gate] BLOCKED — ${COUNT} new REJECT row(s) record neither an A/A null
control nor a counted mechanism, so they cannot distinguish the lever from the
harness. This is the VOID-NONULL class: 205 of 220 voids in this ledger.

EOF
cat "${VIOL}" >&2
cat >&2 <<'EOF'

To clear this, add ONE of:
  * an A/A null control from the SAME invocation (frankensearch_core::bench_support
    paired_median_ratio + decidable_against), so the effect is compared against
    the bench's own floor; or
  * a counted mechanism showing no work was removed — instructions, cycles,
    syscalls, allocations, or page faults unchanged. That is sound WITHOUT a
    null control, because a null cannot change "no work was removed".

If the candidate never actually got timed, say so: title the row BLOCKED or
UNTIMED rather than REJECT. An unmeasured lever is not a rejected lever, and
recording it as one suppresses a candidate that was never measured.

Never gate on cv_pct — it is unreachable below ~12% on this hardware and does
not track decidability (campaign 2.3).
EOF
exit 2
