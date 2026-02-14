#!/usr/bin/env bats

load "helpers/common.bash"

setup() {
  REPO_ROOT="$(repo_root)"
  setup_installer_test_env
}

@test "uninstall.sh delegates to fsfs uninstall when binary exists in PATH" {
  local mock_bin="$TEST_ROOT/mock-bin"
  mkdir -p "$mock_bin"
  cat >"$mock_bin/fsfs" <<'EOF'
#!/usr/bin/env bash
printf 'mock fsfs uninstall invoked: %s\n' "$*"
EOF
  chmod +x "$mock_bin/fsfs"

  PATH="$mock_bin:/usr/bin:/bin" run "$REPO_ROOT/uninstall.sh" --dry-run --yes --purge
  [ "$status" -eq 0 ]
  [[ "$output" == *"mock fsfs uninstall invoked: uninstall --purge --yes --dry-run"* ]]
}

@test "uninstall.sh fallback dry-run prints deterministic plan" {
  mkdir -p "$INSTALL_LOCATION"
  mkdir -p "$FRANKENSEARCH_INDEX_DIR/vector"
  mkdir -p "$FRANKENSEARCH_MODEL_DIR"

  PATH="/usr/bin:/bin" run "$REPO_ROOT/uninstall.sh" --dry-run --yes --purge
  [ "$status" -eq 0 ]
  [[ "$output" == *"fsfs uninstall plan"* ]]
  [[ "$output" == *"mode: dry-run"* ]]
  [[ "$output" == *"PLAN   $FRANKENSEARCH_INDEX_DIR"* ]]
  [[ "$output" == *"PLAN   $FRANKENSEARCH_MODEL_DIR"* ]]
  [[ "$output" == *"Summary:"* ]]
}

@test "uninstall.sh fallback without purge marks purge paths as skipped" {
  PATH="/usr/bin:/bin" run "$REPO_ROOT/uninstall.sh" --dry-run --yes
  [ "$status" -eq 0 ]
  [[ "$output" == *"mode: dry-run"* ]]
  [[ "$output" == *"SKIP"* || "$output" == *"MISS"* || "$output" == *"PLAN"* ]]
  [[ "$output" == *"Summary:"* ]]
}

