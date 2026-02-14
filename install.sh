#!/usr/bin/env bash

set -euo pipefail

# Installer output primitives for fsfs rollout tasks.
# This script currently focuses on presentation helpers and summary rendering.

QUIET=false
NO_GUM=false
DEMO=false

INSTALL_LOCATION="${INSTALL_LOCATION:-$HOME/.local/bin/fsfs}"
MODEL_CACHE_LOCATION="${MODEL_CACHE_LOCATION:-$HOME/.cache/frankensearch/models}"
CONFIGURED_AGENTS="${CONFIGURED_AGENTS:-none}"
PATH_STATUS="${PATH_STATUS:-not configured}"

USE_COLOR=false
USE_UNICODE=false
HAS_GUM=false

ANSI_RESET=""
ANSI_BLUE=""
ANSI_GREEN=""
ANSI_YELLOW=""
ANSI_RED=""
ANSI_BOLD=""

SYMBOL_INFO="->"
SYMBOL_OK="[ok]"
SYMBOL_WARN="!"
SYMBOL_ERR="x"

print_usage() {
  cat <<'USAGE'
Usage: install.sh [options]

Options:
  --demo                    Run output helper demo mode.
  --dest PATH               Installation location shown in summary output.
  --model-cache PATH        Model cache location shown in summary output.
  --agents LIST             Configured agents shown in summary output.
  --path-status TEXT        PATH status shown in summary output.
  --quiet                   Suppress non-essential info lines.
  --no-gum                  Force ANSI/plain fallback even when gum exists.
  --help                    Show this help.
USAGE
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --demo)
        DEMO=true
        shift
        ;;
      --dest)
        INSTALL_LOCATION="${2:-}"
        shift 2
        ;;
      --model-cache)
        MODEL_CACHE_LOCATION="${2:-}"
        shift 2
        ;;
      --agents)
        CONFIGURED_AGENTS="${2:-}"
        shift 2
        ;;
      --path-status)
        PATH_STATUS="${2:-}"
        shift 2
        ;;
      --quiet)
        QUIET=true
        shift
        ;;
      --no-gum)
        NO_GUM=true
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
}

configure_output_mode() {
  local is_tty=false
  local no_color_requested=false
  if [[ -t 1 ]]; then
    is_tty=true
  fi

  local utf8_locale=false
  case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
    *UTF-8*|*utf8*|*UTF8*)
      utf8_locale=true
      ;;
  esac

  if [[ -n "${NO_COLOR:-}" ]]; then
    no_color_requested=true
    USE_COLOR=false
  elif [[ "$is_tty" == true ]]; then
    USE_COLOR=true
  else
    USE_COLOR=false
  fi

  if [[ "$utf8_locale" == true && "$is_tty" == true ]]; then
    USE_UNICODE=true
  else
    USE_UNICODE=false
  fi

  if [[ "$USE_COLOR" == true ]]; then
    ANSI_RESET=$'\033[0m'
    ANSI_BLUE=$'\033[34m'
    ANSI_GREEN=$'\033[32m'
    ANSI_YELLOW=$'\033[33m'
    ANSI_RED=$'\033[31m'
    ANSI_BOLD=$'\033[1m'
  fi

  if [[ "$USE_UNICODE" == true ]]; then
    SYMBOL_INFO="→"
    SYMBOL_OK="✓"
    SYMBOL_WARN="▲"
    SYMBOL_ERR="✗"
  fi

  if [[ "$NO_GUM" == false && "$is_tty" == true && "$no_color_requested" == false ]] && command -v gum >/dev/null 2>&1; then
    HAS_GUM=true
  else
    HAS_GUM=false
  fi
}

emit_line() {
  local color="$1"
  local symbol="$2"
  local message="$3"

  if [[ "$HAS_GUM" == true ]]; then
    gum style --foreground "$color" "${symbol} ${message}"
    return
  fi

  if [[ "$USE_COLOR" == true ]]; then
    local ansi_color="$ANSI_BLUE"
    case "$color" in
      green) ansi_color="$ANSI_GREEN" ;;
      yellow) ansi_color="$ANSI_YELLOW" ;;
      red) ansi_color="$ANSI_RED" ;;
      blue) ansi_color="$ANSI_BLUE" ;;
    esac
    printf '%b%s%b %s\n' "$ansi_color" "$symbol" "$ANSI_RESET" "$message"
  else
    printf '%s %s\n' "$symbol" "$message"
  fi
}

info() {
  if [[ "$QUIET" == true ]]; then
    return
  fi
  emit_line "blue" "$SYMBOL_INFO" "$*"
}

ok() {
  emit_line "green" "$SYMBOL_OK" "$*"
}

warn() {
  emit_line "yellow" "$SYMBOL_WARN" "$*"
}

err() {
  emit_line "red" "$SYMBOL_ERR" "$*"
}

banner() {
  local title="$1"

  if [[ "$HAS_GUM" == true ]]; then
    gum style --border normal --border-foreground 63 --padding "0 1" "$title"
    return
  fi

  local border_width
  border_width=$(( ${#title} + 2 ))
  local line
  line="$(printf '%*s' "$border_width" '' | tr ' ' '─')"

  if [[ "$USE_UNICODE" == true ]]; then
    if [[ "$USE_COLOR" == true ]]; then
      printf '%b┌%s┐%b\n' "$ANSI_BLUE" "$line" "$ANSI_RESET"
      printf '%b│ %s │%b\n' "$ANSI_BLUE" "$title" "$ANSI_RESET"
      printf '%b└%s┘%b\n' "$ANSI_BLUE" "$line" "$ANSI_RESET"
    else
      printf '┌%s┐\n' "$line"
      printf '│ %s │\n' "$title"
      printf '└%s┘\n' "$line"
    fi
  else
    line="$(printf '%*s' "$border_width" '' | tr ' ' '-')"
    printf '+%s+\n' "$line"
    printf '| %s |\n' "$title"
    printf '+%s+\n' "$line"
  fi
}

run_with_spinner() {
  local title="$1"
  shift

  if [[ $# -eq 0 ]]; then
    err "run_with_spinner requires a command"
    return 2
  fi

  if [[ "$HAS_GUM" == true ]]; then
    gum spin --spinner dot --title "$title" -- "$@"
    return $?
  fi

  if [[ ! -t 1 || "$QUIET" == true ]]; then
    "$@"
    return $?
  fi

  local frames="|/-\\"
  local i=0
  "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    printf '\r%c %s' "${frames:i++%${#frames}:1}" "$title"
    sleep 0.1
  done

  wait "$pid"
  local status=$?
  printf '\r'
  if [[ $status -eq 0 ]]; then
    ok "$title"
  else
    err "$title failed"
  fi
  return $status
}

summary_row() {
  local label="$1"
  local value="$2"

  if [[ "$HAS_GUM" == true ]]; then
    gum style --bold "$label:" "$(gum style --foreground 246 "$value")"
    return
  fi

  if [[ "$USE_COLOR" == true ]]; then
    printf '%b%-24s%b %s\n' "$ANSI_BOLD" "${label}:" "$ANSI_RESET" "$value"
  else
    printf '%-24s %s\n' "${label}:" "$value"
  fi
}

final_summary() {
  local install_location="$1"
  local cache_location="$2"
  local configured_agents="$3"
  local path_status="$4"
  shift 4
  local next_steps=("$@")

  banner "Installation Summary"
  summary_row "Installation location" "$install_location"
  summary_row "Model cache location" "$cache_location"
  summary_row "Configured agents" "$configured_agents"
  summary_row "PATH status" "$path_status"

  if [[ ${#next_steps[@]} -gt 0 ]]; then
    if [[ "$HAS_GUM" == true ]]; then
      gum style --bold "Next steps:"
      local step
      for step in "${next_steps[@]}"; do
        gum style "  - $step"
      done
    else
      if [[ "$USE_COLOR" == true ]]; then
        printf '%bNext steps:%b\n' "$ANSI_BOLD" "$ANSI_RESET"
      else
        printf 'Next steps:\n'
      fi
      local step
      for step in "${next_steps[@]}"; do
        printf '  - %s\n' "$step"
      done
    fi
  fi
}

demo_mode() {
  banner "frankensearch installer output demo"
  info "preflight checks started"
  run_with_spinner "simulating model download" sleep 1
  ok "binary install complete"
  if [[ "$HAS_GUM" == true ]]; then
    info "gum renderer active"
  else
    warn "gum not found; using fallback renderer"
  fi
  err "sample failure output"
  final_summary \
    "$INSTALL_LOCATION" \
    "$MODEL_CACHE_LOCATION" \
    "$CONFIGURED_AGENTS" \
    "$PATH_STATUS" \
    "Run 'fsfs --help'" \
    "Run 'fsfs index <path>'" \
    "Run 'fsfs search \"query\"'"
}

main() {
  parse_args "$@"
  configure_output_mode

  if [[ "$DEMO" == true ]]; then
    demo_mode
    return 0
  fi

  banner "frankensearch installer scaffold"
  info "output helper subsystem initialized"
  warn "installer execution stages are not wired in this bead"
  final_summary \
    "$INSTALL_LOCATION" \
    "$MODEL_CACHE_LOCATION" \
    "$CONFIGURED_AGENTS" \
    "$PATH_STATUS" \
    "Re-run with --demo to preview output primitives"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
