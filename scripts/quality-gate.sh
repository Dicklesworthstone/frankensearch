#!/usr/bin/env bash
# frankensearch quality gate — the CI replacement, run through `dsr quality --tool frankensearch`
# (configured in ~/.config/dsr/repos.d/frankensearch.yaml) or directly by hand.
#
# GitHub Actions is not used for this repository (owner decision 2026-09-01); every gate that
# used to live in .github/workflows/ci.yml that still matters runs here, on a real host, in
# one pass, and fails closed. Stages:
#
#   fmt        cargo fmt --check
#   check      cargo check --workspace --all-targets
#   clippy     cargo clippy --workspace --all-targets -- -D warnings (pedantic+nursery lints)
#   tests      library unit tests for every crate except the gauntlet harness (its 894-test
#              unit binary alone takes >50 min; it has its own lane in the perf ratchet)
#   fsfs       every fsfs test binary buildable on the default feature set
#   facade     the library crate's integration tests on the product feature set (`hybrid`:
#              potion + MiniLM loaders + Quill), including the real-model two-tier lane
#              (IndexBuilder + TwoTierSearcher yield INITIAL then REFINED through the public
#              API) which is hard-required when the registered models are present
#   examples   frankensearch/examples/run_all.sh (the validate_* e2e scripts + bench_quick);
#              opt-in via QUALITY_GATE_STAGES because it compiles and runs four examples
#              (~2.5 min debug)
#   e2e        the real-model quickstart lane (index + INITIAL/REFINED search on the real
#              binary) when the registered models are present; otherwise a typed SKIP that
#              still fails the gate unless QUALITY_GATE_ALLOW_MODEL_SKIP=1
#   quickstart scripts/check_fsfs_executable_quickstart.sh against the freshly built binary
#
# Environment:
#   QUALITY_GATE_STAGES        comma list to run (default: all of the above)
#   QUALITY_GATE_MODEL_DIR     registered model cache (default: ~/.local/share/frankensearch/models)
#   QUALITY_GATE_ALLOW_MODEL_SKIP=1  let the e2e stage SKIP instead of FAIL when models are absent
#   CARGO_BUILD_JOBS           forwarded to cargo (default 16)
#
# The rch compile-offload hook on the dev fleet is bypassed here on purpose: a gate must run
# on the host it reports for.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export RCH_DISABLE=1
export RCH_CARGO_WRAPPER_BYPASS=1
unset RCH_REQUIRE_REMOTE || true
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}"
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-never}"
MODEL_DIR="${QUALITY_GATE_MODEL_DIR:-$HOME/.local/share/frankensearch/models}"
STAGES="${QUALITY_GATE_STAGES:-fmt,check,clippy,tests,fsfs,facade,e2e,quickstart}"

# AF_UNIX socket paths are capped at ~107 bytes; the daemon tests need a short runtime dir.
if [ -z "${XDG_RUNTIME_DIR:-}" ] || [ ! -d "${XDG_RUNTIME_DIR:-/nonexistent}" ]; then
  export XDG_RUNTIME_DIR="/tmp/fsfs-gate-$$"
  mkdir -p "$XDG_RUNTIME_DIR"
fi
export TMPDIR="${TMPDIR:-/tmp}"

started=$(date -u +%FT%TZ)
failures=()
pass() { printf '[quality-gate] PASS %-10s %s\n' "$1" "$2"; }
fail() { printf '[quality-gate] FAIL %-10s %s\n' "$1" "$2"; failures+=("$1"); }
skip() { printf '[quality-gate] SKIP %-10s %s\n' "$1" "$2"; }
want() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

run_stage() {
  local name="$1"; shift
  local t0=$SECONDS
  if "$@"; then pass "$name" "($((SECONDS - t0))s)"; else fail "$name" "($((SECONDS - t0))s): $*"; fi
}

echo "[quality-gate] repo=$REPO_ROOT rev=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ') started=$started host=$(hostname)"

want fmt       && run_stage fmt    cargo fmt --check
want check     && run_stage check  cargo check --locked --workspace --all-targets
want clippy    && run_stage clippy cargo clippy --locked --workspace --all-targets -- -D warnings
want tests     && run_stage tests  cargo test --locked --workspace --lib --exclude frankensearch-quill-gauntlet
want fsfs      && run_stage fsfs   cargo test --locked -p frankensearch-fsfs --tests

# The library crate's own integration tests on the product feature set. With the registered
# models present the real-model two-tier lane is REQUIRED (it panics instead of skipping), so a
# green stage means the public API actually served INITIAL then REFINED with potion + MiniLM.
if want facade; then
  if [ -d "$MODEL_DIR/potion-multilingual-128M" ] && [ -d "$MODEL_DIR/all-MiniLM-L6-v2" ]; then
    run_stage facade env FRANKENSEARCH_MODEL_DIR="$MODEL_DIR" FRANKENSEARCH_REQUIRE_SEMANTIC_E2E=1 \
      cargo test --locked -p frankensearch --tests --features hybrid
  elif [ "${QUALITY_GATE_ALLOW_MODEL_SKIP:-0}" = "1" ]; then
    skip facade "real-model lane skipped: registered models absent under $MODEL_DIR (allowed by QUALITY_GATE_ALLOW_MODEL_SKIP=1)"
    run_stage facade env FRANKENSEARCH_MODEL_DIR="$MODEL_DIR" \
      cargo test --locked -p frankensearch --tests --features hybrid
  else
    fail facade "registered models absent under $MODEL_DIR; run scripts/rch-ensure-deps.sh --models-only or set QUALITY_GATE_ALLOW_MODEL_SKIP=1"
  fi
fi

if want examples; then
  run_stage examples env FRANKENSEARCH_MODEL_DIR="$MODEL_DIR" frankensearch/examples/run_all.sh
fi

if want e2e; then
  if [ -d "$MODEL_DIR/potion-multilingual-128M" ] && [ -d "$MODEL_DIR/all-MiniLM-L6-v2" ]; then
    run_stage e2e env FSFS_DEFAULT_E2E_MODEL_DIR="$MODEL_DIR" FRANKENSEARCH_REQUIRE_SEMANTIC_E2E=1 \
      cargo test --locked -p frankensearch-fsfs --test default_build_quickstart -- --include-ignored
  elif [ "${QUALITY_GATE_ALLOW_MODEL_SKIP:-0}" = "1" ]; then
    skip e2e "registered models absent under $MODEL_DIR (allowed by QUALITY_GATE_ALLOW_MODEL_SKIP=1)"
  else
    fail e2e "registered models absent under $MODEL_DIR; run scripts/rch-ensure-deps.sh --models-only or set QUALITY_GATE_ALLOW_MODEL_SKIP=1"
  fi
fi

if want quickstart; then
  run_stage quickstart env FRANKENSEARCH_MODEL_DIR="$MODEL_DIR" scripts/check_fsfs_executable_quickstart.sh
fi

echo "[quality-gate] finished=$(date -u +%FT%TZ) failures=${#failures[@]}"
if [ "${#failures[@]}" -ne 0 ]; then
  printf '[quality-gate] FAILED stages: %s\n' "${failures[*]}"
  exit 1
fi
echo "[quality-gate] OK"
