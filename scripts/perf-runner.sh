#!/usr/bin/env bash
# perf-runner.sh — registered-host launcher for the E8-H performance campaign.
#
# The Rust producer owns the exclusive lease, start/end host probes, benchmark
# child, exact run log, artifact-manifest binding, receipt sealing, and
# self-admission. This shell surface requires an out-of-band prebuilt typed
# producer, establishes the requested Linux affinity/NUMA envelope, and
# launches it. The producer acquires the canonical host-global lease before it
# builds and resolves the benchmark executable from the clean source snapshot.
# Registered hosts provide one shared mount namespace and do not unlink or
# rename the lease inode while a campaign is active.
# This script never compiles during a measurement invocation, manufactures
# JSON, or writes promotion history.
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
# The thread budget defaults to, and if supplied must equal, the maximum width
# in the selected gate's complete frozen matrix; it never defaults to host
# capacity.
# M4 is a recognized, fingerprinted optimization target, but every current M4
# promotion invocation fails closed until the producer can attest the actual
# executing image through a supported O_EXEC or loaded-image mechanism.
# Diagnostic Apple profiling happens outside this promotion producer.
# Timed runs are always local; there is no RCH override.

set -euo pipefail
export LC_ALL=C

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
[ "${#RUN_ID}" -le 96 ] || die "--run-id must be at most 96 bytes"
[ "${#RUN_WINDOW}" -le 96 ] || die "--run-window must be at most 96 bytes"
[[ "$RUNS" =~ ^[0-9]+$ ]] && [ "$RUNS" -ge 10 ] && [ "$RUNS" -le 100 ] ||
    die "--runs must remain within 10..100"

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
    APPLE_MODE="not-applicable"
elif [ "$CLASS" = "m4-macos" ]; then
    [ "$OS" = "Darwin" ] || die "class m4-macos requires macOS"
    [ -z "$CPU_LIST" ] || die "M4 scheduler-pool runs do not accept --cpu-list"
    die "M4 promotion is unavailable until the producer can attest the actual executing image through a supported O_EXEC or loaded-image mechanism; current M4 work is diagnostic-only"
else
    die "--class must name a registered trj-zen3-Nc[-smt2] or m4-macos class"
fi

case "$GATE" in
    QG-1) NORMATIVE_THREAD_BUDGET=128 ;;
    QG-7) NORMATIVE_THREAD_BUDGET=8 ;;
    QG-8) NORMATIVE_THREAD_BUDGET=32 ;;
    *) NORMATIVE_THREAD_BUDGET=1 ;;
esac
THREAD_BUDGET="${THREAD_BUDGET:-$NORMATIVE_THREAD_BUDGET}"
[[ "$THREAD_BUDGET" =~ ^[0-9]+$ ]] &&
    [ "$THREAD_BUDGET" -eq "$NORMATIVE_THREAD_BUDGET" ] ||
    die "--thread-budget must equal the frozen $GATE matrix maximum $NORMATIVE_THREAD_BUDGET"
[ "$THREAD_BUDGET" -le "$CLASS_CAPACITY" ] ||
    die "$CLASS cannot execute the full $GATE matrix width $THREAD_BUDGET"
case "$GATE" in
    QG-3|QG-4|QG-5)
        die "$GATE is promotion-unavailable on every host until both arms emit a non-declarative symmetric durability-treatment witness"
        ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
case "$OUT_ROOT" in
    /*) ;;
    *) die "--out must be an absolute directory outside the source repository" ;;
esac
[ ! -L "$OUT_ROOT" ] && [ -d "$OUT_ROOT" ] ||
    die "--out must be an existing non-symlink directory"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd -P)" ||
    die "--out must resolve cleanly"
case "$OUT_ROOT/" in
    "$REPO_ROOT/"*) die "--out must remain outside the source repository" ;;
esac
PERF_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.frankensearch-perf-target-$CLASS}"
case "$PERF_TARGET_DIR" in
    /*) ;;
    *) die "CARGO_TARGET_DIR must be absolute and outside the source repository" ;;
esac
[ ! -L "$PERF_TARGET_DIR" ] && [ -d "$PERF_TARGET_DIR" ] ||
    die "CARGO_TARGET_DIR must be an existing non-symlink directory"
PERF_TARGET_DIR="$(cd "$PERF_TARGET_DIR" && pwd -P)" ||
    die "CARGO_TARGET_DIR must resolve cleanly"
case "$PERF_TARGET_DIR/" in
    "$REPO_ROOT/"*) die "CARGO_TARGET_DIR must remain outside the source repository" ;;
esac
FINALIZER_ELF="$PERF_TARGET_DIR/release-perf/quill-perf-finalize"
[ -x "$FINALIZER_ELF" ] ||
    die "typed finalizer is not prebuilt at $FINALIZER_ELF; build it locally outside measurement windows with: RCH_DISABLE=1 RCH_CARGO_WRAPPER_BYPASS=1 CARGO_TARGET_DIR=$PERF_TARGET_DIR cargo build --locked --profile release-perf -p frankensearch-quill-gauntlet --bin quill-perf-finalize"
exec 9<"$FINALIZER_ELF" ||
    die "cannot hold the verified typed-finalizer executable open"
HELD_FINALIZER_ELF="/proc/self/fd/9"
[ -r "$HELD_FINALIZER_ELF" ] ||
    die "cannot address the held typed-finalizer executable"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CLASS_ROOT="$OUT_ROOT/$CLASS"
RUN_DIR="$CLASS_ROOT/$STAMP-$RUN_ID"
[ ! -L "$CLASS_ROOT" ] ||
    die "machine-class output root must not be a symbolic link"
[ -d "$CLASS_ROOT" ] ||
    die "machine-class output root must already exist before the measurement window"
CLASS_ROOT="$(cd "$CLASS_ROOT" && pwd -P)" ||
    die "machine-class output root must resolve cleanly"
case "$CLASS_ROOT/" in
    "$OUT_ROOT/"*) ;;
    *) die "machine-class output root escaped --out" ;;
esac
RUN_DIR="$CLASS_ROOT/$STAMP-$RUN_ID"
[ ! -e "$RUN_DIR" ] && [ ! -L "$RUN_DIR" ] ||
    die "run directory already exists: $RUN_DIR"

# Force the typed producer and benchmark child to remain on this host.
export CARGO_TARGET_DIR="$PERF_TARGET_DIR"
export RCH_DISABLE=1
export RCH_CARGO_WRAPPER_BYPASS=1
export RCH_MIN_LOCAL_TIME_MS=999999999
export QUILL_PERF_HELD_PRODUCER_FD=9

cd "$REPO_ROOT"
PRODUCER=(
    "$HELD_FINALIZER_ELF"
    --gate "$GATE"
    --class "$CLASS"
    --run-id "$RUN_ID"
    --run-window "$RUN_WINDOW"
    --thread-budget "$THREAD_BUDGET"
    --runs "$RUNS"
    --apple-mode "$APPLE_MODE"
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
    "${LAUNCH[@]}"
    exit $?
fi

if command -v setsid >/dev/null 2>&1; then
    nohup setsid "${LAUNCH[@]}" > /dev/null 2>&1 < /dev/null &
else
    nohup "${LAUNCH[@]}" > /dev/null 2>&1 < /dev/null &
fi
PID=$!
echo "detached:      pid $PID"
echo "artifacts:     $RUN_DIR (created only after locked validation)"
echo "committed:     test -f $RUN_DIR/$GATE.runner.json"
