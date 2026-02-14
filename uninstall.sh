#!/usr/bin/env bash

set -euo pipefail

PURGE=false
YES=false
DRY_RUN=false

print_usage() {
  cat <<'USAGE'
Usage: uninstall.sh [options]

Options:
  --purge     Also remove model/cache/config artifacts.
  --yes       Skip interactive confirmation.
  --dry-run   Show planned removals without deleting files.
  --help      Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge)
      PURGE=true
      shift
      ;;
    --yes)
      YES=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      print_usage >&2
      exit 2
      ;;
  esac
done

FORWARD_FLAGS=()
if [[ "$PURGE" == true ]]; then
  FORWARD_FLAGS+=("--purge")
fi
if [[ "$YES" == true ]]; then
  FORWARD_FLAGS+=("--yes")
fi
if [[ "$DRY_RUN" == true ]]; then
  FORWARD_FLAGS+=("--dry-run")
fi

if command -v fsfs >/dev/null 2>&1; then
  exec fsfs uninstall "${FORWARD_FLAGS[@]}"
fi

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"

BINARY_PATH="${INSTALL_LOCATION:-$HOME/.local/bin/fsfs}"
INDEX_DIR="${FRANKENSEARCH_INDEX_DIR:-$PWD/.frankensearch}"
MODEL_DIR="${FRANKENSEARCH_MODEL_DIR:-$DATA_HOME/frankensearch/models}"
CONFIG_DIR="$CONFIG_HOME/frankensearch"
CACHE_DIR="$CACHE_HOME/frankensearch"

BASH_COMPLETION="$DATA_HOME/bash-completion/completions/fsfs"
ZSH_COMPLETION_SITE="$DATA_HOME/zsh/site-functions/_fsfs"
ZSH_COMPLETION_HOME="$HOME/.zfunc/_fsfs"
FISH_COMPLETION="$CONFIG_HOME/fish/completions/fsfs.fish"
CLAUDE_HOOK_FSFS="$HOME/.claude/hooks/fsfs.sh"
CLAUDE_HOOK_FRANKENSEARCH="$HOME/.claude/hooks/frankensearch.sh"
CLAUDE_CODE_HOOK_FSFS="$HOME/.config/claude-code/hooks/fsfs.sh"
CLAUDE_CODE_HOOK_FRANKENSEARCH="$HOME/.config/claude-code/hooks/frankensearch.sh"
CURSOR_HOOK_FSFS="$HOME/.config/cursor/hooks/fsfs.sh"
CURSOR_HOOK_FRANKENSEARCH="$HOME/.config/cursor/hooks/frankensearch.sh"

ALWAYS_TARGETS=(
  "$BINARY_PATH"
  "$INDEX_DIR"
  "$BASH_COMPLETION"
  "$ZSH_COMPLETION_SITE"
  "$ZSH_COMPLETION_HOME"
  "$FISH_COMPLETION"
  "$CLAUDE_HOOK_FSFS"
  "$CLAUDE_HOOK_FRANKENSEARCH"
  "$CLAUDE_CODE_HOOK_FSFS"
  "$CLAUDE_CODE_HOOK_FRANKENSEARCH"
  "$CURSOR_HOOK_FSFS"
  "$CURSOR_HOOK_FRANKENSEARCH"
)
PURGE_TARGETS=(
  "$MODEL_DIR"
  "$CONFIG_DIR"
  "$CACHE_DIR"
)

safe_guard_path() {
  local path="$1"
  [[ -n "$path" ]] || return 1
  [[ "$path" != "/" ]] || return 1
  [[ "$path" != "$HOME" ]] || return 1
  return 0
}

show_plan() {
  printf 'fsfs uninstall plan\n'
  printf '  mode: %s\n' "$([[ "$DRY_RUN" == true ]] && printf 'dry-run' || printf 'execute')"
  printf '  purge: %s\n' "$PURGE"
  printf '\n'
  printf 'Always removed:\n'
  local target
  for target in "${ALWAYS_TARGETS[@]}"; do
    printf '  - %s\n' "$target"
  done
  printf '\n'
  printf 'Purge-only targets:\n'
  for target in "${PURGE_TARGETS[@]}"; do
    printf '  - %s\n' "$target"
  done
}

if [[ "$YES" == false && "$DRY_RUN" == false ]]; then
  show_plan
  printf '\nProceed with uninstall? [y/N]: '
  read -r answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *)
      printf 'Cancelled.\n'
      exit 0
      ;;
  esac
fi

removed=0
skipped=0
failed=0

remove_path() {
  local path="$1"
  local purge_only="$2"

  if [[ "$purge_only" == "true" && "$PURGE" == false ]]; then
    printf 'SKIP   %s (requires --purge)\n' "$path"
    skipped=$((skipped + 1))
    return
  fi
  if [[ ! -e "$path" && ! -L "$path" ]]; then
    printf 'MISS   %s\n' "$path"
    skipped=$((skipped + 1))
    return
  fi
  if ! safe_guard_path "$path"; then
    printf 'ERROR  %s (unsafe path)\n' "$path"
    failed=$((failed + 1))
    return
  fi
  if [[ "$DRY_RUN" == true ]]; then
    printf 'PLAN   %s\n' "$path"
    skipped=$((skipped + 1))
    return
  fi

  if [[ -d "$path" && ! -L "$path" ]]; then
    if rm -rf "$path"; then
      printf 'OK     %s\n' "$path"
      removed=$((removed + 1))
    else
      printf 'ERROR  %s\n' "$path"
      failed=$((failed + 1))
    fi
  else
    if rm -f "$path"; then
      printf 'OK     %s\n' "$path"
      removed=$((removed + 1))
    else
      printf 'ERROR  %s\n' "$path"
      failed=$((failed + 1))
    fi
  fi
}

show_plan
printf '\n'

for path in "${ALWAYS_TARGETS[@]}"; do
  remove_path "$path" false
done
for path in "${PURGE_TARGETS[@]}"; do
  remove_path "$path" true
done

printf '\nSummary: removed=%d skipped=%d failed=%d\n' "$removed" "$skipped" "$failed"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
