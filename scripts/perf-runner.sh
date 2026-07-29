#!/usr/bin/env bash
# perf-runner.sh — registered-host launcher for the E8-H performance campaign.
#
# The Rust producer owns the exclusive lease, start/end host probes, benchmark
# child, exact run log, artifact-manifest binding, receipt sealing, and
# self-admission. This shell surface only builds the typed local producer,
# establishes the requested Linux affinity/NUMA envelope, and launches that
# producer. The producer itself builds and resolves the benchmark executable
# from the clean source snapshot. This script never manufactures JSON or writes
# promotion history.
#
# Usage:
#   scripts/perf-runner.sh \
#     --gate <QG-1..QG-10> \
#     --class <trj-zen3-Nc[-smt2]|m4-macos> \
#     --run-id <unique-pass-id> \
#     --run-window <shared-candidate-rerun-window> \
#     [--thread-budget <N>] \
#     [--cpu-list <taskset-list>] \
#     [--apple-mode <p-plus-e>] \
#     [--runs <N>] \
#     [--foreground] \
#     [--out <directory>]
#
# Linux runs require --cpu-list. The producer proves that the selected CPUs
# match the physical width and SMT suffix encoded by the class and that every
# selected CPU uses the performance governor under NUMA-node-0 binding.
# M4 runs currently admit only explicit P+E execution. P-only is
# non-admissible; any ad-hoc P-only measurement is diagnostic-only until the
# producer can prove scheduler assignment. QG-1/QG-8 remain blocked until their
# normative matrices have M4-specific endpoints.
# Timed runs are always local; there is no RCH override.

set -euo pipefail

GATE=""
CLASS=""
RUN_ID=""
RUN_WINDOW=""
THREAD_BUDGET=""
CPU_LIST=""
APPLE_MODE=""
RUNS="10"
FOREGROUND=0
OUT_ROOT="${PERF_RUNNER_OUT:-$HOME/.frankensearch-perf-runs}"

usage() {
    printf '%s\n' \
        "Usage:" \
        "  scripts/perf-runner.sh --gate <QG-1..QG-10> \\" \
        "    --class <trj-zen3-Nc[-smt2]|m4-macos> \\" \
        "    --run-id <unique-pass-id> --run-window <shared-window> \\" \
        "    [--thread-budget <N>] [--cpu-list <taskset-list>] \\" \
        "    [--apple-mode <p-plus-e>] [--runs <N>] [--foreground] \\" \
        "    [--out <absolute-directory>]"
}
die() { echo "perf-runner: $*" >&2; exit 64; }

while [ $# -gt 0 ]; do
    case "$1" in
        --gate) GATE="${2:?--gate needs a value}"; shift 2 ;;
        --class) CLASS="${2:?--class needs a value}"; shift 2 ;;
        --run-id) RUN_ID="${2:?--run-id needs a value}"; shift 2 ;;
        --run-window) RUN_WINDOW="${2:?--run-window needs a value}"; shift 2 ;;
        --thread-budget) THREAD_BUDGET="${2:?--thread-budget needs a value}"; shift 2 ;;
        --cpu-list) CPU_LIST="${2:?--cpu-list needs a value}"; shift 2 ;;
        --apple-mode) APPLE_MODE="${2:?--apple-mode needs a value}"; shift 2 ;;
        --runs) RUNS="${2:?--runs needs a value}"; shift 2 ;;
        --foreground) FOREGROUND=1; shift ;;
        --out) OUT_ROOT="${2:?--out needs a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done

[[ "$GATE" =~ ^QG-([1-9]|10)$ ]] || die "--gate must be QG-1 through QG-10"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-id must use [A-Za-z0-9._-]"
[[ "$RUN_WINDOW" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-window must use [A-Za-z0-9._-]"
[[ "$RUNS" =~ ^[0-9]+$ ]] && [ "$RUNS" -ge 10 ] ||
    die "--runs must preserve the >=10-run evidence law"

OS="$(uname -s)"
if [[ "$CLASS" =~ ^trj-zen3-([1-9]|[1-5][0-9]|6[0-4])c(-smt2)?$ ]]; then
    [ "$OS" = "Linux" ] || die "class $CLASS requires Linux"
    [ -n "$CPU_LIST" ] || die "registered TRJ runs require --cpu-list"
    [ -z "$APPLE_MODE" ] || die "TRJ runs do not accept --apple-mode"
    command -v taskset >/dev/null 2>&1 || die "taskset is required for TRJ runs"
    command -v numactl >/dev/null 2>&1 || die "numactl is required for TRJ runs"
    PHYSICAL_WIDTH="${BASH_REMATCH[1]}"
    THREADS_PER_CORE=1
    [[ "${BASH_REMATCH[2]}" == "-smt2" ]] && THREADS_PER_CORE=2
    CLASS_CAPACITY=$((PHYSICAL_WIDTH * THREADS_PER_CORE))
    THREAD_BUDGET="${THREAD_BUDGET:-$CLASS_CAPACITY}"
    APPLE_MODE="not-applicable"
    LEASE_FAMILY="trj-zen3"
elif [ "$CLASS" = "m4-macos" ]; then
    [ "$OS" = "Darwin" ] || die "class m4-macos requires macOS"
    [ -z "$CPU_LIST" ] || die "M4 scheduler-pool runs do not accept --cpu-list"
    [ "$APPLE_MODE" = "p-plus-e" ] ||
        die "M4 promotion runs currently require --apple-mode p-plus-e"
    case "$GATE" in
        QG-1|QG-8)
            die "$GATE on M4 is blocked until the normative matrix has class-specific 10P/14P+E endpoints"
            ;;
        QG-3|QG-4|QG-5)
            die "$GATE on macOS is blocked until both arms attest symmetric F_FULLFSYNC treatment"
            ;;
    esac
    CLASS_CAPACITY=14
    THREAD_BUDGET="${THREAD_BUDGET:-$CLASS_CAPACITY}"
    LEASE_FAMILY="m4-macos"
else
    die "--class must name a registered trj-zen3-Nc[-smt2] or m4-macos class"
fi

[[ "$THREAD_BUDGET" =~ ^[0-9]+$ ]] &&
    [ "$THREAD_BUDGET" -ge 1 ] &&
    [ "$THREAD_BUDGET" -le "$CLASS_CAPACITY" ] ||
    die "--thread-budget must be within 1..$CLASS_CAPACITY for $CLASS"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
    die "registered performance production requires a clean source tree"
fi
case "$OUT_ROOT" in
    /*) ;;
    *) die "--out must be an absolute directory outside the source repository" ;;
esac
OUT_PARENT="$(cd "$(dirname "$OUT_ROOT")" && pwd -P)" ||
    die "--out parent must already exist"
OUT_ROOT="$OUT_PARENT/$(basename "$OUT_ROOT")"
case "$OUT_ROOT/" in
    "$REPO_ROOT/"*) die "--out must remain outside the source repository" ;;
esac
PERF_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.frankensearch-perf-target-$CLASS}"
case "$PERF_TARGET_DIR" in
    /*) ;;
    *) die "CARGO_TARGET_DIR must be absolute and outside the source repository" ;;
esac
TARGET_PARENT="$(cd "$(dirname "$PERF_TARGET_DIR")" && pwd -P)" ||
    die "CARGO_TARGET_DIR parent must already exist"
PERF_TARGET_DIR="$TARGET_PARENT/$(basename "$PERF_TARGET_DIR")"
case "$PERF_TARGET_DIR/" in
    "$REPO_ROOT/"*) die "CARGO_TARGET_DIR must remain outside the source repository" ;;
esac

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CLASS_ROOT="$OUT_ROOT/$CLASS"
RUN_DIR="$CLASS_ROOT/$STAMP-$RUN_ID"
mkdir -p "$CLASS_ROOT"
mkdir "$RUN_DIR" || die "run directory already exists: $RUN_DIR"

# Keep compilation outside the repository and force it to remain on this host.
export CARGO_TARGET_DIR="$PERF_TARGET_DIR"
export RCH_DISABLE=1
export RCH_CARGO_WRAPPER_BYPASS=1
export RCH_MIN_LOCAL_TIME_MS=999999999
export QUILL_PERF_RUNS="$RUNS"
export RAYON_NUM_THREADS="$THREAD_BUDGET"

BUILD_MESSAGES="$RUN_DIR/build-messages.jsonl"
cd "$REPO_ROOT"
command -v jq >/dev/null 2>&1 || die "jq is required to resolve exact Cargo executables"
cargo build \
    --locked \
    --profile release-perf \
    -p frankensearch-quill-gauntlet \
    --bin quill-perf-finalize \
    --message-format=json-render-diagnostics \
    > "$BUILD_MESSAGES"
FINALIZER_ELF="$(
    jq -r '
        select(.reason == "compiler-artifact")
        | select(.target.name == "quill-perf-finalize")
        | .executable // empty
    ' "$BUILD_MESSAGES" | tail -n 1
)"
[ -x "$FINALIZER_ELF" ] || die "Cargo did not report an executable typed finalizer"

LEASE_ROOT="$OUT_ROOT/.leases"
mkdir -p "$LEASE_ROOT"
LEASE_PATH="$LEASE_ROOT/$LEASE_FAMILY.lock"
PRODUCER=(
    "$FINALIZER_ELF"
    --gate "$GATE"
    --class "$CLASS"
    --run-id "$RUN_ID"
    --run-window "$RUN_WINDOW"
    --thread-budget "$THREAD_BUDGET"
    --apple-mode "$APPLE_MODE"
    --lease-path "$LEASE_PATH"
    --output-dir "$RUN_DIR"
)
if [ "$OS" = "Linux" ]; then
    LAUNCH=(
        taskset -c "$CPU_LIST"
        numactl --physcpubind="$CPU_LIST" --membind=0
        "${PRODUCER[@]}"
    )
else
    LAUNCH=("${PRODUCER[@]}")
fi

echo "run dir:       $RUN_DIR"
echo "gate/class:    $GATE / $CLASS"
echo "run identity:  $RUN_ID (window $RUN_WINDOW)"
echo "thread budget: $THREAD_BUDGET"
echo "benchmark ELF: built and resolved by the typed producer"

if [ "$FOREGROUND" -eq 1 ]; then
    set +e
    "${LAUNCH[@]}" 2>&1 | tee "$RUN_DIR/launcher.log"
    STATUS=${PIPESTATUS[0]}
    set -e
    echo "producer exit: $STATUS"
    exit "$STATUS"
fi

if command -v setsid >/dev/null 2>&1; then
    nohup setsid "${LAUNCH[@]}" > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
else
    nohup "${LAUNCH[@]}" > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
fi
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/producer.pid"
echo "detached:      pid $PID"
echo "follow:        tail -f $RUN_DIR/launcher.log"
echo "committed:     test -f $RUN_DIR/$GATE.runner.json"
