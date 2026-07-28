#!/usr/bin/env bash
# perf-runner.sh — machine-class provenance runner for the E8-H hyperopt campaign.
#
# Runs a benchmark command (default: the QG perf_matrix bench) on a non-rch
# machine class (trj-zen-128c, m4-macos, m5-macos) or an rch-class host
# (x86-vps-ovh), capturing the per-class provenance that
# docs/contracts/quill-hyperopt-campaign.md § 2 makes mandatory:
# CPU/NUMA topology, governor + SMT state (Linux), thermal pressure + page
# size (macOS), git revision + dirty-state hash, toolchain versions.
#
# The benchmark process itself remains responsible for printing its own ELF
# SHA-256 before Criterion output (quill-perf-gates.md § .bench-history) —
# this runner records everything the process cannot know about its host.
#
# Usage:
#   scripts/perf-runner.sh --class <id> [options] [-- <command...>]
#
#   --class <id>       one of: x86-vps-ovh | trj-zen-128c | m4-macos | m5-macos
#   --label <name>     short label folded into the output directory name
#   --calibrate-aa     record this run as an A/A-only calibration run
#                      (exports QUILL_PERF_CALIBRATE_AA=1; harness-side
#                      consumption is bd-quill-e8-hyperopt P0 scope)
#   --foreground       run attached instead of detached (default: detached)
#   --allow-rch        permit rch offload for this run (compile-only labels).
#                      By DEFAULT the runner exports RCH_DISABLE=1 so the
#                      command executes on THIS machine — a timed window that
#                      silently offloads to an rch worker measures the wrong
#                      machine (Law 6). Never pass this for timed runs.
#   --out <dir>        output root (default: ~/.frankensearch-perf-runs)
#   -- <command...>    command to run; default:
#                      cargo bench -p frankensearch-quill-gauntlet
#                        --bench perf_matrix --features perf-harness
#                        --profile release-perf
#
# Environment passed through and recorded: QUILL_PERF_SCALE,
# QUILL_PERF_FIXTURE, RAYON_NUM_THREADS.
#
# Law 6 (machine-class scoping): artifacts from this runner are only
# comparable within one class. Law 7 (macOS): durability-adjacent numbers
# need F_FULLFSYNC symmetry attested — this runner records the OS but the
# harness owns the attestation.

set -euo pipefail

CLASS=""
LABEL="run"
CALIBRATE_AA=0
FOREGROUND=0
ALLOW_RCH=0
OUT_ROOT="${PERF_RUNNER_OUT:-$HOME/.frankensearch-perf-runs}"

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --class) CLASS="${2:?--class needs a value}"; shift 2 ;;
        --label) LABEL="${2:?--label needs a value}"; shift 2 ;;
        --calibrate-aa) CALIBRATE_AA=1; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        --allow-rch) ALLOW_RCH=1; shift ;;
        --out) OUT_ROOT="${2:?--out needs a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) echo "unknown argument: $1 (use --help)" >&2; exit 64 ;;
    esac
done

case "$CLASS" in
    # trj-zen3-<width>c per the committed-baseline convention
    # (QG-2.trj-zen3-16c.latest.json); trj-zen-128c is the superseded label.
    x86-vps-ovh|trj-zen-128c|trj-zen3-*) EXPECT_OS="Linux" ;;
    m4-macos|m5-macos) EXPECT_OS="Darwin" ;;
    "") echo "error: --class is required" >&2; exit 64 ;;
    *) echo "error: unknown machine class '$CLASS'" >&2; exit 64 ;;
esac

OS="$(uname -s)"
if [ "$OS" != "$EXPECT_OS" ]; then
    echo "error: class $CLASS expects $EXPECT_OS but this host is $OS" >&2
    exit 65
fi

if [ $# -gt 0 ]; then
    CMD=("$@")
else
    CMD=(cargo bench -p frankensearch-quill-gauntlet --bench perf_matrix
        --features perf-harness --profile release-perf)
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUT_ROOT/$CLASS/$STAMP-$LABEL"
mkdir -p "$RUN_DIR"

# Isolated, per-class persistent target dir: avoids lock contention with the
# agent swarm's builds AND avoids a cold rebuild on every run.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.frankensearch-perf-target-$CLASS}"

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1; else shasum -a 256 | cut -d' ' -f1; fi; }

GIT_REV="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
DIRTY_HASH="$( (git -C "$REPO_ROOT" status --porcelain; git -C "$REPO_ROOT" diff) 2>/dev/null | sha )"

# --- per-OS fingerprint ------------------------------------------------------
FP_DIR="$RUN_DIR/fingerprint"
mkdir -p "$FP_DIR"
uname -a > "$FP_DIR/uname.txt"
GOVERNOR="n/a"; SMT="n/a"; THERMAL="n/a"; PAGE_SIZE="n/a"
if [ "$OS" = "Linux" ]; then
    lscpu > "$FP_DIR/lscpu.txt" 2>/dev/null || true
    lscpu -e > "$FP_DIR/lscpu-e.txt" 2>/dev/null || true
    command -v numactl >/dev/null 2>&1 && numactl -H > "$FP_DIR/numactl-H.txt" 2>/dev/null || true
    [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ] && \
        GOVERNOR="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    [ -r /sys/devices/system/cpu/smt/control ] && \
        SMT="$(cat /sys/devices/system/cpu/smt/control)"
    PAGE_SIZE="$(getconf PAGE_SIZE 2>/dev/null || echo n/a)"
    if [ "$CLASS" = "trj-zen-128c" ] && [ "$GOVERNOR" != "performance" ]; then
        echo "WARNING: governor is '$GOVERNOR', not 'performance' — trj timed" >&2
        echo "         windows are inadmissible per campaign contract § 2." >&2
    fi
else
    sysctl machdep.cpu hw.ncpu hw.memsize hw.pagesize hw.perflevel0 hw.perflevel1 \
        > "$FP_DIR/sysctl.txt" 2>/dev/null || true
    THERMAL="$(pmset -g therm 2>/dev/null | tr '\n' ' ' | tr -s ' ' || echo n/a)"
    PAGE_SIZE="$(sysctl -n hw.pagesize 2>/dev/null || echo n/a)"
fi

# --- local-execution guarantee ------------------------------------------------
# Some hosts (trj) shim cargo through rch; an offloaded "timed" run would
# execute on a different machine than the one fingerprinted above. Default to
# forcing local execution; --allow-rch opts out for compile-only labels.
if [ "$ALLOW_RCH" -eq 0 ]; then
    export RCH_DISABLE=1
    export RCH_MIN_LOCAL_TIME_MS=999999999
    # trj's cargo is a fail-closed rch wrapper; this is its documented bypass
    # (execs the real ~/.cargo/bin/cargo). Harmless on hosts without the shim.
    export RCH_CARGO_WRAPPER_BYPASS=1
fi

# --- provenance --------------------------------------------------------------
[ "$CALIBRATE_AA" -eq 1 ] && export QUILL_PERF_CALIBRATE_AA=1
CMD_STR="$(printf '%q ' "${CMD[@]}")"
# Escape for embedding in JSON string values (backslashes first, then quotes).
json_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
CMD_JSON="$(json_escape "$CMD_STR")"
THERMAL_JSON="$(json_escape "$THERMAL")"
cat > "$RUN_DIR/provenance.json" <<EOF
{
  "schema": "frankensearch.perf-runner.v1",
  "machine_class": "$CLASS",
  "label": "$LABEL",
  "calibrate_aa": $CALIBRATE_AA,
  "rch_disabled": $([ "$ALLOW_RCH" -eq 0 ] && echo true || echo false),
  "timestamp_utc": "$STAMP",
  "hostname": "$(hostname)",
  "os": "$OS",
  "git_rev": "$GIT_REV",
  "dirty_state_sha256": "$DIRTY_HASH",
  "rustc": "$(rustc -V 2>/dev/null || echo unknown)",
  "cargo": "$(cargo -V 2>/dev/null || echo unknown)",
  "governor": "$GOVERNOR",
  "smt": "$SMT",
  "thermal_pressure": "$THERMAL_JSON",
  "page_size": "$PAGE_SIZE",
  "cargo_target_dir": "$CARGO_TARGET_DIR",
  "quill_perf_scale": "${QUILL_PERF_SCALE:-unset}",
  "quill_perf_fixture": "${QUILL_PERF_FIXTURE:-unset}",
  "rayon_num_threads": "${RAYON_NUM_THREADS:-unset}",
  "command": "$CMD_JSON"
}
EOF

echo "run dir:    $RUN_DIR"
echo "class:      $CLASS  (git $GIT_REV, dirty $DIRTY_HASH)"
echo "command:    $CMD_STR"

# --- execute -----------------------------------------------------------------
cd "$REPO_ROOT"
if [ "$FOREGROUND" -eq 1 ]; then
    "${CMD[@]}" 2>&1 | tee "$RUN_DIR/run.log"
    STATUS=${PIPESTATUS[0]}
    echo "$STATUS" > "$RUN_DIR/exit-status"
    echo "exit:       $STATUS"
    exit "$STATUS"
else
    # Detached: survives SSH disconnect. setsid where available (Linux);
    # macOS ships no setsid binary, nohup alone suffices there.
    RUNNER="nohup"
    command -v setsid >/dev/null 2>&1 && RUNNER="setsid nohup"
    # shellcheck disable=SC2086
    # </dev/null is load-bearing: without it the child inherits sshd's stdin
    # on macOS (no setsid there) and a remote kickoff hangs until build end.
    $RUNNER bash -c '"$@" > "$0/run.log" 2>&1; echo $? > "$0/exit-status"' \
        "$RUN_DIR" "${CMD[@]}" > /dev/null 2>&1 < /dev/null &
    PID=$!
    echo "$PID" > "$RUN_DIR/runner.pid"
    echo "detached:   pid $PID"
    echo "follow:     tail -f $RUN_DIR/run.log"
    echo "exit code:  cat $RUN_DIR/exit-status   (written on completion)"
fi
