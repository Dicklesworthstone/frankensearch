#!/usr/bin/env bats

load "helpers/common.bash"

setup() {
  REPO_ROOT="$(repo_root)"
  setup_installer_test_env
}

@test "e2e_smoke.sh completes local install/index/search/uninstall lifecycle" {
  run env \
    FSFS_E2E_TMPDIR="$TEST_ROOT/e2e-smoke-pass" \
    FSFS_E2E_INSTALL_MODE=local \
    bash "$REPO_ROOT/tests/installer/e2e_smoke.sh"
  [ "$status" -eq 0 ]
  [[ "$output" == *"E2E SMOKE TEST PASSED"* ]]
}

@test "e2e_smoke.sh reports install-phase failure for invalid install mode" {
  run env \
    FSFS_E2E_TMPDIR="$TEST_ROOT/e2e-smoke-invalid-mode" \
    FSFS_E2E_INSTALL_MODE=invalid \
    bash "$REPO_ROOT/tests/installer/e2e_smoke.sh"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Unknown FSFS_E2E_INSTALL_MODE: invalid"* ]]
  [[ "$output" == *"E2E installer smoke failed during phase: install"* ]]
}
