#!/usr/bin/env bats

load "helpers/common.bash"

setup() {
  REPO_ROOT="$(repo_root)"
  setup_installer_test_env
}

@test "install.sh --help exposes scaffold options" {
  run bash "$REPO_ROOT/install.sh" --help
  [ "$status" -eq 0 ]
  [[ "$output" == *"Usage: install.sh"* ]]
  [[ "$output" == *"--demo"* ]]
  [[ "$output" == *"--model-cache PATH"* ]]
}

@test "install.sh --demo renders summary rows with overrides" {
  run bash "$REPO_ROOT/install.sh" \
    --demo \
    --dest "$HOME/bin/fsfs-custom" \
    --model-cache "$HOME/models" \
    --agents "claude,cursor" \
    --path-status "configured"
  [ "$status" -eq 0 ]
  [[ "$output" == *"Installation Summary"* ]]
  [[ "$output" == *"Installation location:"* ]]
  [[ "$output" == *"$HOME/bin/fsfs-custom"* ]]
  [[ "$output" == *"Configured agents:"* ]]
  [[ "$output" == *"claude,cursor"* ]]
}

@test "install.sh rejects unknown flags with usage error" {
  run bash "$REPO_ROOT/install.sh" --definitely-not-valid
  [ "$status" -eq 2 ]
  [[ "$output" == *"Unknown argument: --definitely-not-valid"* ]]
}

@test "install.sh --demo respects NO_COLOR without ANSI escapes" {
  run env NO_COLOR=1 bash "$REPO_ROOT/install.sh" --demo --no-gum
  [ "$status" -eq 0 ]
  [[ "$output" == *"Installation Summary"* ]]
  [[ "$output" != *$'\033'* ]]
}

@test "install.sh --demo degrades cleanly when piped (no ANSI escapes)" {
  run bash -c "bash \"$REPO_ROOT/install.sh\" --demo --no-gum | cat"
  [ "$status" -eq 0 ]
  [[ "$output" == *"Installation Summary"* ]]
  [[ "$output" != *$'\033'* ]]
}
