#!/usr/bin/env bash
# Executable quick-start gate (bd-fsfs-executable-quickstart-ci-ve3ul).
#
# Runs the public source-install quick start against the REAL binary and a tiny
# deterministic corpus, exactly as the README documents it — no hidden feature
# flags, no mocks. Fails closed on every regression class the 2026-08-01
# reality check found the hard way (docs/evidence/reality-check-20260801.md):
#
#   NONTERMINATION       `fsfs index` completes but never exits (exit 124 under cap)
#   MODEL-UNAVAILABLE    default build cannot index at all (exit 78)
#   MISSING-ARTIFACT     no sentinel/CURRENT/FSVI/Quill durability artifacts
#   INCOMPLETE-SENTINEL  sentinel exists but generation_complete != true
#   EMPTY-RESULTS        search returns no hits for a query with a known answer
#   WRONG-RANKING        the known-correct document is not rank 1
#   HASH-DEGRADATION     binary silently fell back to hash embedders
#   PROCESS-LEAK         child fsfs processes survive the run
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX_DEADLINE_SECS=180
SEARCH_DEADLINE_SECS=90
SKIP_BUILD=0
KEEP_ARTIFACTS=0
BINARY_OVERRIDE=""

usage() {
  cat <<USAGE
Usage: scripts/check_fsfs_executable_quickstart.sh [options]

Options:
  --binary PATH       Use an existing fsfs binary instead of building one.
  --skip-build        Alias for requiring --binary (fail if binary missing).
  --keep-artifacts    Do not delete the work directory on success.
  --index-deadline N  Seconds allowed for one bounded index run (default ${INDEX_DEADLINE_SECS}).
  -h|--help           Show this help.

The gate builds the binary with the README's documented source-install
feature set (cargo build -p frankensearch-fsfs --features embedded-models),
so it exercises exactly what the README tells a user to run — the feature is
explicit in the documentation, never hidden. Model inputs must already be
provisioned and SHA-verified via: scripts/rch-ensure-deps.sh --models-only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY_OVERRIDE="${2:?}"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --keep-artifacts) KEEP_ARTIFACTS=1; shift ;;
    --index-deadline) INDEX_DEADLINE_SECS="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

fail() {
  local class="$1"; shift
  echo "" >&2
  echo "QUICKSTART GATE FAIL [${class}]: $*" >&2
  for log in "${WORK_DIR}"/index-run1.stderr "${WORK_DIR}"/index-run1.stdout \
             "${WORK_DIR}"/search.stderr "${WORK_DIR}"/search.json; do
    if [[ -s "${log}" ]]; then
      echo "---- tail ${log} ----" >&2
      tail -n 25 "${log}" >&2 || true
    fi
  done
  echo "Artifacts retained at: ${WORK_DIR}" >&2
  exit 1
}

cd "${ROOT_DIR}"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fsfs-quickstart.XXXXXXXX")"
cleanup() {
  if [[ "${KEEP_ARTIFACTS}" -eq 0 && -n "${GATE_PASSED:-}" ]]; then
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT

echo "[gate] work dir: ${WORK_DIR}"

# ── 0. Model provisioning preflight (explicit, SHA-verified, fail closed) ──
if ! scripts/rch-ensure-deps.sh --models-only --check >"${WORK_DIR}/models-check.log" 2>&1; then
  fail "MODEL-PROVISIONING" \
    "bundled model inputs absent or hash-mismatched; run scripts/rch-ensure-deps.sh --models-only first"
fi

# ── 1. Build the binary exactly as the README documents (default features) ──
if [[ -n "${BINARY_OVERRIDE}" ]]; then
  FSFS_BIN="${BINARY_OVERRIDE}"
  [[ -x "${FSFS_BIN}" ]] || fail "SETUP" "--binary ${FSFS_BIN} is not executable"
else
  [[ "${SKIP_BUILD}" -eq 1 ]] && fail "SETUP" "--skip-build requires --binary"
  echo "[gate] building: cargo build -p frankensearch-fsfs --bin fsfs --features embedded-models (documented source install)"
  cargo build -p frankensearch-fsfs --bin fsfs --features embedded-models
  FSFS_BIN="${CARGO_TARGET_DIR:-target}/debug/fsfs"
  [[ -x "${FSFS_BIN}" ]] || fail "SETUP" "build produced no binary at ${FSFS_BIN}"
fi

BIN_SHA256="$(sha256sum "${FSFS_BIN}" | cut -d' ' -f1)"
GIT_REV="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"

# ── 2. Deterministic fixture corpus with one unambiguous best answer ──
CORPUS_DIR="${WORK_DIR}/corpus"
INDEX_DIR="${WORK_DIR}/index"
mkdir -p "${CORPUS_DIR}" "${INDEX_DIR}"
printf 'Retry backoff uses exponential delay with jitter to avoid thundering herds.\n' > "${CORPUS_DIR}/retry.md"
printf 'The parser tokenizes UTF-8 input into normalized terms.\n' > "${CORPUS_DIR}/parser.md"
printf 'Structured concurrency ensures no orphan tasks survive scope exit.\n' > "${CORPUS_DIR}/concurrency.md"

run_env=(env FRANKENSEARCH_INDEX_DIR="${INDEX_DIR}" FRANKENSEARCH_CHECK_UPDATES=0)
if [[ -n "${FRANKENSEARCH_MODEL_DIR:-}" ]]; then
  run_env+=(FRANKENSEARCH_MODEL_DIR="${FRANKENSEARCH_MODEL_DIR}")
fi

# ── 3. Bounded index run: must exit 0 on its own ──
echo "[gate] index run 1 (deadline ${INDEX_DEADLINE_SECS}s)"
set +e
"${run_env[@]}" timeout "${INDEX_DEADLINE_SECS}" "${FSFS_BIN}" index "${CORPUS_DIR}" \
  >"${WORK_DIR}/index-run1.stdout" 2>"${WORK_DIR}/index-run1.stderr"
INDEX_EXIT=$?
set -e
case "${INDEX_EXIT}" in
  0) ;;
  124) fail "NONTERMINATION" \
        "fsfs index did not exit within ${INDEX_DEADLINE_SECS}s (the bd-fsfs-index-command-quiescence-v53qo regression)" ;;
  78) fail "MODEL-UNAVAILABLE" \
        "fsfs index exited 78 embedder_unavailable — documented default build cannot index (the bd-fsfs-default-build-usable-6mtid regression)" ;;
  *) fail "INDEX-ERROR" "fsfs index exited ${INDEX_EXIT}" ;;
esac

# ── 4. Process-leak check ──
if pgrep -f "fsfs index ${CORPUS_DIR}" >/dev/null 2>&1; then
  fail "PROCESS-LEAK" "an fsfs index process for this corpus is still alive after exit"
fi

# ── 5. Durable artifact assertions ──
SENTINEL="${INDEX_DIR}/index_sentinel.json"
[[ -s "${SENTINEL}" ]] || fail "MISSING-ARTIFACT" "no index_sentinel.json"
[[ -s "${INDEX_DIR}/vector/index.fsvi" ]] || fail "MISSING-ARTIFACT" "no vector/index.fsvi"
[[ -s "${INDEX_DIR}/lexical/CURRENT" ]] || fail "MISSING-ARTIFACT" "no lexical/CURRENT pointer"
find "${INDEX_DIR}/lexical" -name MANIFEST -size +0c | grep -q . \
  || fail "MISSING-ARTIFACT" "no lexical engine MANIFEST"
python3 - "$SENTINEL" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("generation_complete") is not True:
    print(f"QUICKSTART GATE FAIL [INCOMPLETE-SENTINEL]: generation_complete={d.get('generation_complete')!r}", file=sys.stderr)
    sys.exit(1)
if d.get("indexed_files") != 3:
    print(f"QUICKSTART GATE FAIL [INCOMPLETE-SENTINEL]: indexed_files={d.get('indexed_files')!r}, expected 3", file=sys.stderr)
    sys.exit(1)
PY

# ── 6. Repeatability: a second bounded run must also exit 0 ──
echo "[gate] index run 2 (repeat, deadline ${INDEX_DEADLINE_SECS}s)"
"${run_env[@]}" timeout "${INDEX_DEADLINE_SECS}" "${FSFS_BIN}" index "${CORPUS_DIR}" \
  >"${WORK_DIR}/index-run2.stdout" 2>"${WORK_DIR}/index-run2.stderr" \
  || fail "NONTERMINATION" "repeat fsfs index run failed or timed out (exit $?)"

# ── 7. Bounded search: nonempty, correctly ranked, hybrid ──
echo "[gate] search (deadline ${SEARCH_DEADLINE_SECS}s)"
"${run_env[@]}" timeout "${SEARCH_DEADLINE_SECS}" "${FSFS_BIN}" search \
  "how does retry backoff work" --limit 3 --format json \
  >"${WORK_DIR}/search.json" 2>"${WORK_DIR}/search.stderr" \
  || fail "SEARCH-ERROR" "fsfs search failed or timed out (exit $?)"
python3 - "${WORK_DIR}/search.json" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
def die(cls, msg):
    print(f"QUICKSTART GATE FAIL [{cls}]: {msg}", file=sys.stderr)
    sys.exit(1)
if not d.get("ok"):
    die("SEARCH-ERROR", f"payload ok={d.get('ok')!r}")
hits = (d.get("data") or {}).get("hits") or []
if not hits:
    die("EMPTY-RESULTS", "search returned no hits for a query with a known answer")
top = hits[0]
if top.get("path") != "retry.md":
    die("WRONG-RANKING", f"expected retry.md at rank 1, got {top.get('path')!r}")
if top.get("in_both_sources") is not True:
    die("WRONG-RANKING", "top hit is not corroborated by both lexical and semantic sources")
fresh = (d.get("data") or {}).get("index_freshness") or {}
if fresh.get("degraded"):
    die("HASH-DEGRADATION", "index_freshness reports degraded=true")
PY

# ── 8. Real embedder identity attestation (no silent hash fallback) ──
echo "[gate] embedder identity attestation"
"${run_env[@]}" timeout "${SEARCH_DEADLINE_SECS}" "${FSFS_BIN}" status --no-watch-mode --format json \
  >"${WORK_DIR}/status.json" 2>"${WORK_DIR}/status.stderr" \
  || fail "SEARCH-ERROR" "fsfs status failed (exit $?)"
python3 - "${WORK_DIR}/status.json" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
def die(cls, msg):
    print(f"QUICKSTART GATE FAIL [{cls}]: {msg}", file=sys.stderr)
    sys.exit(1)
models = (d.get("data") or {}).get("models") or []
tiers = {m.get("tier"): (m.get("name") or "") for m in models}
fast = tiers.get("fast", "")
quality = tiers.get("quality", "")
if not fast or "hash" in fast.lower():
    die("HASH-DEGRADATION", f"fast tier is {fast!r}; expected a real semantic model")
if not quality or "hash" in quality.lower():
    die("HASH-DEGRADATION", f"quality tier is {quality!r}; expected a real semantic model")
PY

GATE_PASSED=1
echo ""
echo "[gate] PASS"
echo "  binary       : ${FSFS_BIN}"
echo "  binary sha256: ${BIN_SHA256}"
echo "  source rev   : ${GIT_REV}"
echo "  replay       : scripts/check_fsfs_executable_quickstart.sh"
echo "Result: PASS"
