#!/usr/bin/env bash
set -u
set -o pipefail

readonly ROOT=/tmp/frankensearch-qg1-ccc37c8e
readonly ELF="$ROOT/bin/perf_matrix-clean-53ab4c0975f0ad21"
readonly SOURCE_REV=ccc37c8e611cd313201108ffe9260376a977b4bd
readonly ELF_SHA256=53ab4c0975f0ad2148e37f35641dfd56e78acd8048d01cdb8b1194aa8ab9b637
readonly SWEEP_ID=qg1-trj-h1h2-ccc37c8e-clean-r10-20260731T0349Z
readonly RUN_WINDOW=qg1-trj-h1h2-ccc37c8e-clean-r10-20260731
readonly BASE="$ROOT/rows/$SWEEP_ID"
readonly SCRATCH="$ROOT/shared-scratch/$SWEEP_ID"
readonly HASH_DOMAIN=sorted-newline-environment-v1
readonly -a WIDTHS=(1 2 4 8 16 32 64 96 128)

capture_host() {
    date -u +%FT%TZ
    hostname
    uname -a
    printf 'boot_id='
    cat /proc/sys/kernel/random/boot_id
    uptime
    printf 'remote_shell_pid=%s remote_shell_ppid=%s\n' "$$" "$PPID"
    grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status
    lscpu
    free -b
    df -B1 / /tmp
    for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -r "$governor" ]]; then
            cat "$governor"
        fi
    done | sort | uniq -c
    ps -eo pid,ppid,stat,comm,psr,%cpu,%mem,etimes,args --sort=-%cpu | head -40
    sha256sum "$ELF"
    stat -c '%s %A %n' "$ELF"
}

actual_elf_sha256=$(sha256sum "$ELF" | cut -d' ' -f1)
if [[ "$actual_elf_sha256" != "$ELF_SHA256" ]]; then
    printf '[trj-sweep-abort] ELF SHA-256 mismatch expected=%s actual=%s path=%s\n' \
        "$ELF_SHA256" "$actual_elf_sha256" "$ELF" >&2
    exit 65
fi

mkdir -p "$BASE" "$SCRATCH"

printf '[trj-sweep-ready] sweep_id=%s remote_shell_pid=%s remote_shell_ppid=%s host=%s boot_id=%s widths=%s\n' \
    "$SWEEP_ID" "$$" "$PPID" "$(hostname)" \
    "$(cat /proc/sys/kernel/random/boot_id)" "${WIDTHS[*]}"
printf '[trj-sweep-ready] elf_sha256=%s source_rev=%s base=%s\n' \
    "$ELF_SHA256" "$SOURCE_REV" "$BASE"
printf '[trj-sweep-ready] build_binding=rch-base-clean-no-overlay rch_base=%s overlay_files=0\n' \
    "$SOURCE_REV"
printf '[trj-sweep-handshake] type START after Agent Mail records the live PID tree\n'

IFS= read -r start_token
if [[ "$start_token" != START ]]; then
    printf '[trj-sweep-abort] expected START, received %q\n' "$start_token" >&2
    exit 64
fi

sweep_status=0
sweep_started_ns=$(date +%s%N)
printf '[trj-sweep-start] timestamp=%s shell_pid=%s\n' "$(date -u +%FT%TZ)" "$$"

for threads in "${WIDTHS[@]}"; do
    row="$BASE/t$threads"
    tmp="$row/tmp"
    run_id="$SWEEP_ID-t$threads"
    mkdir -p "$row" "$tmp"

    capture_host > "$row/host-before.txt"

    env_lines=(
        "HOME=/home/ubuntu"
        "LANG=C"
        "LC_ALL=C"
        "PATH=/usr/local/bin:/usr/bin:/bin"
        "QUILL_PERF_BOOTSTRAP_SEED=5860671082138523204"
        "QUILL_PERF_BUILD_PROFILE=release-perf"
        "QUILL_PERF_FIXTURE=bulk/xlarge/$threads/positions_on"
        "QUILL_PERF_GATE=QG-1"
        "QUILL_PERF_GIT_REV=$SOURCE_REV"
        "QUILL_PERF_OUTPUT_DIR=$row"
        "QUILL_PERF_RUNS=10"
        "QUILL_PERF_RUN_ID=$run_id"
        "QUILL_PERF_RUN_WINDOW=$RUN_WINDOW"
        "QUILL_PERF_RUSTC=/nonexistent"
        "QUILL_PERF_SCALE=full"
        "QUILL_PERF_SCRATCH_DIR=$SCRATCH"
        "QUILL_PERF_TIMING_MODE=continuous"
        "QUILL_PERF_TYPED_PRODUCER=1"
        "QUILL_PERF_WARMUP_ROUNDS=1"
        "QUILL_PERF_WORK_RECEIPTS=on"
        "RAYON_NUM_THREADS=$threads"
        "RCH_CARGO_WRAPPER_BYPASS=1"
        "RCH_DISABLE=1"
        "TMPDIR=$tmp"
    )
    environment_sha256=$(
        printf '%s\n' "${env_lines[@]}" |
            LC_ALL=C sort |
            sha256sum |
            cut -d' ' -f1
    )
    {
        printf 'hash_domain=%s\n' "$HASH_DOMAIN"
        printf '%s\n' "${env_lines[@]}" | LC_ALL=C sort
        printf 'QUILL_PERF_ENVIRONMENT_SHA256=%s\n' "$environment_sha256"
    } > "$row/environment.txt"
    printf '%q ' env -i "${env_lines[@]}" \
        "QUILL_PERF_ENVIRONMENT_SHA256=$environment_sha256" \
        "$ELF" --noplot --bench > "$row/command.txt"
    printf '\n' >> "$row/command.txt"

    row_started_ns=$(date +%s%N)
    printf '[trj-row-start] timestamp=%s threads=%s run_id=%s environment_sha256=%s\n' \
        "$(date -u +%FT%TZ)" "$threads" "$run_id" "$environment_sha256"

    env -i "${env_lines[@]}" \
        "QUILL_PERF_ENVIRONMENT_SHA256=$environment_sha256" \
        "$ELF" --noplot --bench 2>&1 |
        tee "$row/run.log" |
        stdbuf -oL grep -E \
            '^(bench_elf_sha256=|\[quill-perf-oracle\]|\[qg1-work-receipt\]|\[qg1-continuous-summary\]|\[qg1-work-receipt-summary\]|\[quill-evidence\]|\[quill-perf\]|thread .+ panicked|error:)'
    benchmark_status=${PIPESTATUS[0]}
    row_status=$benchmark_status

    self_reported_sha256=$(
        sed -nE 's/^bench_elf_sha256=([0-9a-f]{64}).*/\1/p' "$row/run.log" |
            head -1
    )
    receipt_count=0
    receipt_sha256=missing
    wall_mismatches=missing
    terminal_failures=missing
    if [[ -f "$row/work-receipts.jsonl" ]]; then
        receipt_count=$(wc -l < "$row/work-receipts.jsonl")
        receipt_sha256=$(sha256sum "$row/work-receipts.jsonl" | cut -d' ' -f1)
        wall_mismatches=$(
            jq -s '
                map(select(
                    .concurrency.wall_ns
                    != ([.phases[] | select(.phase == "quiescence_joined") | .window_elapsed_ns][0])
                ))
                | length
            ' "$row/work-receipts.jsonl"
        )
        terminal_failures=$(
            jq -s '
                map(select(
                    (.terminal.drained != true)
                    or (.terminal.pending_docs_zero != true)
                    or (.terminal.retryable != false)
                ))
                | length
            ' "$row/work-receipts.jsonl"
        )
    fi

    evidence_identity_ok=false
    last_h1_h2_wall_equal=false
    laws_attested=missing
    if [[ -f "$row/QG-1.json" ]]; then
        evidence_identity_ok=$(
            jq -r --arg sha "$ELF_SHA256" --arg rev "$SOURCE_REV" \
                '(.bench_elf_sha256 == $sha) and (.git_rev == $rev)' \
                "$row/QG-1.json"
        )
        last_h1_h2_wall_equal=$(
            jq -r '
                .work_receipts.cells as $work
                | .continuous_timing.cells as $continuous
                | [
                    range(0; $work | length)
                    | (
                        ($work[.].last_quill_receipt.concurrency.wall_ns
                            == $continuous[.].last_quill_receipt.timeline.window_total_ns)
                        and
                        ($work[.].last_tantivy_receipt.concurrency.wall_ns
                            == $continuous[.].last_tantivy_receipt.timeline.window_total_ns)
                    )
                ]
                | all
            ' "$row/QG-1.json"
        )
        laws_attested=$(jq -r '.laws_attested' "$row/QG-1.json")
    fi

    if [[ "$self_reported_sha256" != "$ELF_SHA256" ||
          "$receipt_count" -ne 66 ||
          "$wall_mismatches" != 0 ||
          "$terminal_failures" != 0 ||
          "$evidence_identity_ok" != true ||
          "$last_h1_h2_wall_equal" != true ||
          "$laws_attested" != false ]]; then
        row_status=66
    fi

    row_ended_ns=$(date +%s%N)
    capture_host > "$row/host-after.txt"
    {
        printf 'status=%s\n' "$row_status"
        printf 'benchmark_status=%s\n' "$benchmark_status"
        printf 'started_ns=%s\n' "$row_started_ns"
        printf 'ended_ns=%s\n' "$row_ended_ns"
        printf 'elapsed_ns=%s\n' "$((row_ended_ns - row_started_ns))"
        printf 'remote_shell_pid=%s\n' "$$"
        printf 'source_rev=%s\n' "$SOURCE_REV"
        printf 'elf_sha256=%s\n' "$ELF_SHA256"
        printf 'self_reported_sha256=%s\n' "$self_reported_sha256"
        printf 'environment_sha256=%s\n' "$environment_sha256"
        printf 'receipt_count=%s\n' "$receipt_count"
        printf 'receipt_sha256=%s\n' "$receipt_sha256"
        printf 'wall_mismatches=%s\n' "$wall_mismatches"
        printf 'terminal_failures=%s\n' "$terminal_failures"
        printf 'evidence_identity_ok=%s\n' "$evidence_identity_ok"
        printf 'last_h1_h2_wall_equal=%s\n' "$last_h1_h2_wall_equal"
        printf 'laws_attested=%s\n' "$laws_attested"
    } > "$row/status.txt"

    if [[ -f "$row/QG-1.json" ]]; then
        jq -r --arg threads "$threads" '
            "[trj-row-result] threads=\($threads) host=\(.machine_fingerprint) laws_attested=\(.laws_attested) quill=\([.cells[] | select(.engine == "quill") | .value][0]) tantivy=\([.cells[] | select(.engine == "tantivy") | .value][0])"
        ' "$row/QG-1.json" || true
    fi
    printf '[trj-row-validation] threads=%s self_sha=%s receipts=%s wall_mismatches=%s terminal_failures=%s identity_ok=%s h1_h2_equal=%s laws_attested=%s\n' \
        "$threads" "$self_reported_sha256" "$receipt_count" "$wall_mismatches" \
        "$terminal_failures" "$evidence_identity_ok" "$last_h1_h2_wall_equal" \
        "$laws_attested"
    printf '[trj-row-terminal] timestamp=%s threads=%s status=%s elapsed_ns=%s\n' \
        "$(date -u +%FT%TZ)" "$threads" "$row_status" "$((row_ended_ns - row_started_ns))"

    if ((row_status != 0)); then
        sweep_status=$row_status
        break
    fi
done

sweep_ended_ns=$(date +%s%N)
{
    printf 'status=%s\n' "$sweep_status"
    printf 'started_ns=%s\n' "$sweep_started_ns"
    printf 'ended_ns=%s\n' "$sweep_ended_ns"
    printf 'elapsed_ns=%s\n' "$((sweep_ended_ns - sweep_started_ns))"
    printf 'remote_shell_pid=%s\n' "$$"
    printf 'source_rev=%s\n' "$SOURCE_REV"
    printf 'elf_sha256=%s\n' "$ELF_SHA256"
} > "$BASE/sweep-status.txt"
printf '[trj-sweep-terminal] timestamp=%s status=%s elapsed_ns=%s\n' \
    "$(date -u +%FT%TZ)" "$sweep_status" "$((sweep_ended_ns - sweep_started_ns))"
exit "$sweep_status"
