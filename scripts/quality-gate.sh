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
#   clippy     cargo clippy --workspace --all-targets -- -D warnings (pedantic+nursery lints),
#              then the facade crate again on --features hybrid (the product feature set)
#   cross      cargo check -p frankensearch-index --target x86_64-pc-windows-msvc — the
#              non-Unix compile guard for #42 (a unix-only import outside any cfg, and a
#              fallback platform module missing entry points the portable code calls). Needs
#              only the target's std, never a Windows host or a linker. Override the triple
#              with QUALITY_GATE_CROSS_TARGET.
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
#   perf       the library two-tier latency/index-cost receipt (frankensearch/tests/
#              latency_receipt.rs) on --release with the registered models: 1,000-document
#              corpus, 50 timed queries, p50/p95/p99 per phase, written to
#              docs/evidence/perf/library-two-tier-latency-<date>-<host>.json (override with
#              QUALITY_GATE_PERF_RECEIPT_OUT). Opt-in: it builds the facade in release
#              and takes minutes; commit the receipt it writes.
#   e2e        the real-model quickstart lane (index + INITIAL/REFINED search on the real
#              binary) when the registered models are present; otherwise a typed SKIP that
#              still fails the gate unless QUALITY_GATE_ALLOW_MODEL_SKIP=1
#   quickstart scripts/check_fsfs_executable_quickstart.sh against the freshly built binary
#
# Environment:
#   QUALITY_GATE_STAGES        comma list to run (default: all of the above)
#   QUALITY_GATE_MODEL_DIR     registered model cache (default: ~/.local/share/frankensearch/models)
#   QUALITY_GATE_ALLOW_MODEL_SKIP=1  let the e2e stage SKIP instead of FAIL when models are absent
#   QUALITY_GATE_CROSS_TARGET  triple for the cross stage (default x86_64-pc-windows-msvc)
#   QUALITY_GATE_ALLOW_CROSS_SKIP=1  let the cross stage SKIP when that target's std is absent
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
STAGES="${QUALITY_GATE_STAGES:-fmt,check,clippy,cross,tests,fsfs,facade,e2e,quickstart}"
# Non-Unix compile guard target (#42). Overridable so the same stage can prove
# any other host triple the index crate claims to build for.
CROSS_TARGET="${QUALITY_GATE_CROSS_TARGET:-x86_64-pc-windows-msvc}"

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
# The facade's product feature set (loaders + Quill) is what users build; it was never linted
# until bd-fhy2j, and its future-Send analysis needs the crate's raised recursion limit.
want clippy    && run_stage clippy-hybrid cargo clippy --locked -p frankensearch --features hybrid --all-targets -- -D warnings

# Cross-target compile guard (#42). The generation-root work landed a
# `use std::os::unix::…` outside any cfg and a fallback platform module missing
# three entry points the platform-independent publisher calls, so v1.8.0 failed
# to compile for x86_64-pc-windows-msvc with ten name/type-resolution errors.
# Every one of them is caught by a plain `cargo check` against the target from
# any host — no Windows machine, and no linker, required.
if want cross; then
  if ! rustup target list --installed 2>/dev/null | grep -qx "$CROSS_TARGET"; then
    rustup target add "$CROSS_TARGET" >/dev/null 2>&1 || true
  fi
  if rustup target list --installed 2>/dev/null | grep -qx "$CROSS_TARGET"; then
    run_stage cross cargo check --locked -p frankensearch-index --target "$CROSS_TARGET"
  elif [ "${QUALITY_GATE_ALLOW_CROSS_SKIP:-0}" = "1" ]; then
    skip cross "std for $CROSS_TARGET is not installed (allowed by QUALITY_GATE_ALLOW_CROSS_SKIP=1)"
  else
    fail cross "std for $CROSS_TARGET is not installed; run 'rustup target add $CROSS_TARGET' or set QUALITY_GATE_ALLOW_CROSS_SKIP=1"
  fi
fi

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

# Release-profile latency/index-cost receipt for the library two-tier path. The lane refuses a
# hash-only stack, so it needs the registered models; the JSON it writes is the evidence the
# README's envelope rows cite, and is meant to be committed.
if want perf; then
  if [ -d "$MODEL_DIR/all-MiniLM-L6-v2" ]; then
    perf_out="${QUALITY_GATE_PERF_RECEIPT_OUT:-docs/evidence/perf/library-two-tier-latency-$(date -u +%Y%m%d)-$(hostname).json}"
    # cargo runs the test with the PACKAGE directory as cwd, so a relative
    # receipt path would land under frankensearch/ instead of the repo root.
    case "$perf_out" in /*) ;; *) perf_out="$REPO_ROOT/$perf_out" ;; esac
    run_stage perf env FRANKENSEARCH_MODEL_DIR="$MODEL_DIR" FRANKENSEARCH_PERF_RECEIPT=1 \
      FRANKENSEARCH_PERF_RECEIPT_OUT="$perf_out" \
      cargo test --locked --release -p frankensearch --features hybrid --test latency_receipt -- --nocapture
    echo "[quality-gate] perf receipt: $perf_out"
    # The product half: the fsfs binary's own index cost, cold start, and daemon-served
    # query latency (plus the rerank series), written beside the library receipt.
    perf_fsfs_out="${QUALITY_GATE_PERF_FSFS_RECEIPT_OUT:-docs/evidence/perf/fsfs-latency-$(date -u +%Y%m%d)-$(hostname).json}"
    case "$perf_fsfs_out" in /*) ;; *) perf_fsfs_out="$REPO_ROOT/$perf_fsfs_out" ;; esac
    run_stage perf-fsfs env FRANKENSEARCH_MODEL_DIR="$MODEL_DIR" FRANKENSEARCH_PERF_RECEIPT=1 \
      FRANKENSEARCH_PERF_RECEIPT_OUT="$perf_fsfs_out" \
      cargo test --locked --release -p frankensearch-fsfs --test fsfs_latency_receipt -- --nocapture
    echo "[quality-gate] perf-fsfs receipt: $perf_fsfs_out"
  else
    fail perf "registered models absent under $MODEL_DIR; the latency receipt needs potion + MiniLM"
  fi
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
