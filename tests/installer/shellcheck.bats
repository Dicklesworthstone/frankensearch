#!/usr/bin/env bats

load "helpers/common.bash"

setup() {
  REPO_ROOT="$(repo_root)"
}

@test "shellcheck passes for installer scripts" {
  run shellcheck -x "$REPO_ROOT/install.sh" "$REPO_ROOT/uninstall.sh"
  [ "$status" -eq 0 ]
}

