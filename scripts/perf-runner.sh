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
#     --hardware-class <registered-hardware-class> \
#     --execution-profile <registered-execution-profile> \
#     --run-id <unique-pass-id> \
#     --run-window <shared-candidate-rerun-window> \
#     [--cpu-list <taskset-list>] \
#     [--runs <N>] \
#     [--foreground] \
#     [--out <directory>]
#
# Linux runs require --cpu-list. The producer proves that the selected CPUs
# match the frozen execution profile and that every selected CPU uses the
# performance governor under NUMA-node-0 binding. Execution capacity and the
# gate's maximum exercised width come only from the frozen registry.
# M4 is a recognized, fingerprinted optimization target, but every current M4
# promotion invocation fails closed until the producer can attest the actual
# executing image through a supported O_EXEC or loaded-image mechanism.
# Diagnostic Apple profiling happens outside this promotion producer.
# Timed runs are always local; there is no RCH override.

set -euo pipefail
export LC_ALL=C

GATE=""
HARDWARE_CLASS=""
EXECUTION_PROFILE=""
RUN_ID=""
RUN_WINDOW=""
CPU_LIST=""
RUNS="10"
FOREGROUND=0
OUT_ROOT="${PERF_RUNNER_OUT:-$HOME/.frankensearch-perf-runs}"

usage() {
    printf '%s\n' \
        "Usage:" \
        "  scripts/perf-runner.sh --gate <QG-1..QG-10> \\" \
        "    --hardware-class <registered-hardware-class> \\" \
        "    --execution-profile <registered-execution-profile> \\" \
        "    --run-id <unique-pass-id> --run-window <shared-window> \\" \
        "    [--cpu-list <taskset-list>] [--runs <N>] [--foreground] \\" \
        "    [--out <absolute-directory>]"
}
die() { echo "perf-runner: $*" >&2; exit 64; }

while [ $# -gt 0 ]; do
    case "$1" in
        --gate) GATE="${2:?--gate needs a value}"; shift 2 ;;
        --hardware-class) HARDWARE_CLASS="${2:?--hardware-class needs a value}"; shift 2 ;;
        --execution-profile) EXECUTION_PROFILE="${2:?--execution-profile needs a value}"; shift 2 ;;
        --run-id) RUN_ID="${2:?--run-id needs a value}"; shift 2 ;;
        --run-window) RUN_WINDOW="${2:?--run-window needs a value}"; shift 2 ;;
        --cpu-list) CPU_LIST="${2:?--cpu-list needs a value}"; shift 2 ;;
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
PROFILE_KEY="$HARDWARE_CLASS:$EXECUTION_PROFILE"
case "$PROFILE_KEY" in
    trj-zen3-5995wx:physical-64|trj-zen3-5995wx:smt2-128)
        [ "$OS" = "Linux" ] || die "profile $PROFILE_KEY requires Linux"
        [ -n "$CPU_LIST" ] || die "registered Threadripper profiles require --cpu-list"
        command -v taskset >/dev/null 2>&1 ||
            die "taskset is required for Threadripper profiles"
        command -v numactl >/dev/null 2>&1 ||
            die "numactl is required for Threadripper profiles"
        ;;
    m4-macos:scheduler-10)
        [ "$OS" = "Darwin" ] || die "profile $PROFILE_KEY requires macOS"
        [ -z "$CPU_LIST" ] || die "scheduler-managed Apple profiles do not accept --cpu-list"
        die "M4 promotion is unavailable until the producer can attest the actual executing image through a supported O_EXEC or loaded-image mechanism; current M4 work is diagnostic-only"
        ;;
    m5-macos:scheduler-14)
        die "profile $PROFILE_KEY is registered but unavailable until a real M5 host fingerprint lands"
        ;;
    x86-vps-ovh:x86-diagnostic)
        die "profile $PROFILE_KEY is diagnostic-only and cannot run through the promotion producer"
        ;;
    *)
        die "unregistered hardware/execution profile pair: $PROFILE_KEY"
        ;;
esac

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
PERF_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.frankensearch-perf-target-$HARDWARE_CLASS-$EXECUTION_PROFILE}"
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
PROFILE_ROOT="$OUT_ROOT/$HARDWARE_CLASS.$EXECUTION_PROFILE"
[ ! -L "$PROFILE_ROOT" ] ||
    die "machine-profile output root must not be a symbolic link"
[ -d "$PROFILE_ROOT" ] ||
    die "machine-profile output root must already exist before the measurement window"
PROFILE_ROOT="$(cd "$PROFILE_ROOT" && pwd -P)" ||
    die "machine-profile output root must resolve cleanly"
case "$PROFILE_ROOT/" in
    "$OUT_ROOT/"*) ;;
    *) die "machine-profile output root escaped --out" ;;
esac
RUN_DIR="$PROFILE_ROOT/$STAMP-$RUN_ID"
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
    --hardware-class "$HARDWARE_CLASS"
    --execution-profile "$EXECUTION_PROFILE"
    --run-id "$RUN_ID"
    --run-window "$RUN_WINDOW"
    --runs "$RUNS"
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
echo "gate/profile:  $GATE / $HARDWARE_CLASS.$EXECUTION_PROFILE"
echo "run identity:  $RUN_ID (window $RUN_WINDOW)"
echo "run capacity:  derived from the frozen execution profile"
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
