#!/usr/bin/env bash

repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  printf '%s' "$here"
}

setup_installer_test_env() {
  export TEST_ROOT
  TEST_ROOT="$(mktemp -d)"
  export HOME="$TEST_ROOT/home"
  export XDG_CONFIG_HOME="$HOME/.config"
  export XDG_CACHE_HOME="$HOME/.cache"
  export XDG_DATA_HOME="$HOME/.local/share"
  export INSTALL_LOCATION="$HOME/.local/bin/fsfs"
  export FRANKENSEARCH_INDEX_DIR="$TEST_ROOT/project/.frankensearch"
  export FRANKENSEARCH_MODEL_DIR="$XDG_DATA_HOME/frankensearch/models"

  mkdir -p \
    "$HOME/.local/bin" \
    "$HOME/.config" \
    "$HOME/.cache" \
    "$HOME/.local/share" \
    "$TEST_ROOT/project"
}
