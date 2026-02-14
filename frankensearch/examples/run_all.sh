#!/usr/bin/env bash
# Run all frankensearch e2e validation scripts (bd-3un.40).
#
# Usage:
#   ./frankensearch/examples/run_all.sh          # debug build
#   ./frankensearch/examples/run_all.sh --release # release build (for bench)
#
# Exit code: 0 if all pass, 1 if any fail.

set -euo pipefail

PROFILE="${1:-}"
CARGO_FLAGS=()
if [[ "$PROFILE" == "--release" ]]; then
    CARGO_FLAGS+=(--release)
    echo -e "\n\033[1;36m=== Running in RELEASE mode ===\033[0m"
else
    echo -e "\n\033[1;36m=== Running in DEBUG mode ===\033[0m"
fi

FAILED=0

run_example() {
    local name="$1"
    echo -e "\n\033[1;33m──── $name ────\033[0m"
    if RUST_LOG=off cargo run --example "$name" "${CARGO_FLAGS[@]}" 2>&1; then
        echo -e "\033[32m✓ $name passed\033[0m"
    else
        echo -e "\033[31m✗ $name FAILED\033[0m"
        FAILED=1
    fi
}

run_example validate_full_pipeline
run_example validate_index_io
run_example validate_fusion
run_example bench_quick

echo ""
if [[ "$FAILED" -eq 0 ]]; then
    echo -e "\033[1;32m=== All validations passed ===\033[0m"
else
    echo -e "\033[1;31m=== Some validations FAILED ===\033[0m"
    exit 1
fi
