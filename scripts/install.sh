#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
INSTALL_LOCK_DIR="${TMPDIR:-/tmp}/frankensearch-fsfs-install.lock"
INSTALL_LOCK_PID_FILE="${INSTALL_LOCK_DIR}/pid"
INSTALL_LOCK_TS_FILE="${INSTALL_LOCK_DIR}/started_at"
MIN_DISK_MB=200

DEFAULT_DEST_DIR="${HOME}/.local/bin"
SYSTEM_DEST_DIR="/usr/local/bin"
DEFAULT_REPO_SLUG="Dicklesworthstone/frankensearch"
DEFAULT_BINARY_NAME="fsfs"

VERSION="latest"
DEST_DIR="${DEFAULT_DEST_DIR}"
DEST_EXPLICIT=false
SYSTEM_INSTALL=false
FORCE=false
VERIFY=false
FROM_SOURCE=false
OFFLINE=false
QUIET=false
NO_GUM=false
NO_CONFIGURE=false
EASY_MODE=false
CHECKSUM=""
ALLOW_ROOT=false

TARGET_OS=""
TARGET_ARCH=""
TARGET_TRIPLE=""
RESOLVED_VERSION=""
TEMP_DIR=""
HAVE_GUM=false
USE_COLOR=false

AGENT_NAMES=()
AGENT_DETECTED=()
AGENT_VERSIONS=()
AGENT_TARGETS=()
AGENT_RESULTS=()

COLOR_RESET=""
COLOR_BOLD=""
COLOR_INFO=""
COLOR_OK=""
COLOR_WARN=""
COLOR_ERR=""

print_usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [options]

Installer scaffold for frankensearch fsfs.

Options:
  --version <tag>        Install a specific release tag (default: latest)
  --dest <dir>           Installation directory (default: ${DEFAULT_DEST_DIR})
  --system               Install to ${SYSTEM_DEST_DIR}
  --force                Overwrite existing installation
  --verify               Enable checksum verification
  --from-source          Build/install from source instead of a release binary
  --offline              Disable network checks and remote version lookup
  --quiet                Reduce log output
  --no-gum               Disable gum formatting even when gum exists
  --no-configure         Skip shell configuration stage
  --easy-mode            Auto-configure detected agent integrations without prompts
  --checksum <sha256>    Expected SHA-256 for release artifact
  --yes-i-want-to-run-as-root
                         Allow execution as root
  --help                 Show this help text
EOF
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

log_gum() {
  local style="$1"
  local text="$2"

  if [[ "${HAVE_GUM}" == true ]]; then
    case "${style}" in
      info)
        gum style --foreground 33 "[INFO] ${text}"
        ;;
      ok)
        gum style --foreground 42 "[OK] ${text}"
        ;;
      warn)
        gum style --foreground 214 "[WARN] ${text}"
        ;;
      err)
        gum style --foreground 196 "[ERROR] ${text}" >&2
        ;;
      plain)
        gum style "${text}"
        ;;
    esac
    return
  fi
}

configure_output() {
  if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    USE_COLOR=true
    COLOR_RESET=$'\033[0m'
    COLOR_BOLD=$'\033[1m'
    COLOR_INFO=$'\033[34m'
    COLOR_OK=$'\033[32m'
    COLOR_WARN=$'\033[33m'
    COLOR_ERR=$'\033[31m'
  fi

  if [[ "${NO_GUM}" == false && -t 1 ]] && has_cmd gum; then
    HAVE_GUM=true
  fi
}

log_plain() {
  local prefix="$1"
  local color="$2"
  local text="$3"

  if [[ "${USE_COLOR}" == true ]]; then
    printf '%b%s%b %s\n' "${color}" "${prefix}" "${COLOR_RESET}" "${text}"
  else
    printf '%s %s\n' "${prefix}" "${text}"
  fi
}

info() {
  local text="$*"
  if [[ "${QUIET}" == true ]]; then
    return
  fi

  if [[ "${HAVE_GUM}" == true ]]; then
    log_gum info "${text}"
  else
    log_plain "[INFO]" "${COLOR_INFO}" "${text}"
  fi
}

ok() {
  local text="$*"
  if [[ "${HAVE_GUM}" == true ]]; then
    log_gum ok "${text}"
  else
    log_plain "[OK]" "${COLOR_OK}" "${text}"
  fi
}

warn() {
  local text="$*"
  if [[ "${HAVE_GUM}" == true ]]; then
    log_gum warn "${text}"
  else
    log_plain "[WARN]" "${COLOR_WARN}" "${text}"
  fi
}

err() {
  local text="$*"
  if [[ "${HAVE_GUM}" == true ]]; then
    log_gum err "${text}"
  else
    log_plain "[ERROR]" "${COLOR_ERR}" "${text}" >&2
  fi
}

die() {
  err "$*"
  exit 1
}

need_arg() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    die "Flag ${flag} requires a value"
  fi
}

validate_checksum() {
  if [[ -z "${CHECKSUM}" ]]; then
    return
  fi

  if [[ ! "${CHECKSUM}" =~ ^[A-Fa-f0-9]{64}$ ]]; then
    die "--checksum must be a 64-character hexadecimal SHA-256 digest"
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --version)
        need_arg "$1" "${2:-}"
        VERSION="$2"
        shift 2
        ;;
      --dest)
        need_arg "$1" "${2:-}"
        DEST_DIR="$2"
        DEST_EXPLICIT=true
        shift 2
        ;;
      --system)
        SYSTEM_INSTALL=true
        shift
        ;;
      --force)
        FORCE=true
        shift
        ;;
      --verify)
        VERIFY=true
        shift
        ;;
      --from-source)
        FROM_SOURCE=true
        shift
        ;;
      --offline)
        OFFLINE=true
        shift
        ;;
      --quiet)
        QUIET=true
        shift
        ;;
      --no-gum)
        NO_GUM=true
        shift
        ;;
      --no-configure)
        NO_CONFIGURE=true
        shift
        ;;
      --easy-mode)
        EASY_MODE=true
        shift
        ;;
      --checksum)
        need_arg "$1" "${2:-}"
        CHECKSUM="$2"
        shift 2
        ;;
      --yes-i-want-to-run-as-root)
        ALLOW_ROOT=true
        shift
        ;;
      --help|-h)
        print_usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done

  if [[ "${SYSTEM_INSTALL}" == true && "${DEST_EXPLICIT}" == false ]]; then
    DEST_DIR="${SYSTEM_DEST_DIR}"
  fi
}

release_lock() {
  if [[ -d "${INSTALL_LOCK_DIR}" ]]; then
    rm -f "${INSTALL_LOCK_PID_FILE}" "${INSTALL_LOCK_TS_FILE}" || true
    rmdir "${INSTALL_LOCK_DIR}" 2>/dev/null || true
  fi
}

cleanup_temp_dir() {
  if [[ -n "${TEMP_DIR}" && -d "${TEMP_DIR}" ]]; then
    rm -f "${TEMP_DIR}"/* 2>/dev/null || true
    rmdir "${TEMP_DIR}" 2>/dev/null || true
  fi
}

on_exit() {
  local code=$?
  cleanup_temp_dir
  release_lock

  if [[ ${code} -ne 0 ]]; then
    err "Installer failed with exit code ${code}"
  fi
}

acquire_lock() {
  if mkdir "${INSTALL_LOCK_DIR}" 2>/dev/null; then
    printf '%s\n' "$$" > "${INSTALL_LOCK_PID_FILE}"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "${INSTALL_LOCK_TS_FILE}"
    return
  fi

  local existing_pid=""
  if [[ -f "${INSTALL_LOCK_PID_FILE}" ]]; then
    existing_pid="$(tr -d '[:space:]' < "${INSTALL_LOCK_PID_FILE}" || true)"
  fi

  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    die "Another installer process is running (pid ${existing_pid}). Use --force only after that process exits."
  fi

  warn "Found stale installer lock; attempting recovery."
  release_lock

  if mkdir "${INSTALL_LOCK_DIR}" 2>/dev/null; then
    printf '%s\n' "$$" > "${INSTALL_LOCK_PID_FILE}"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "${INSTALL_LOCK_TS_FILE}"
  else
    die "Failed to acquire installer lock at ${INSTALL_LOCK_DIR}"
  fi
}

detect_platform() {
  local uname_s
  local uname_m
  uname_s="$(uname -s)"
  uname_m="$(uname -m)"

  case "${uname_s}" in
    Linux) TARGET_OS="unknown-linux-musl" ;;
    Darwin) TARGET_OS="apple-darwin" ;;
    *)
      die "Unsupported operating system: ${uname_s}"
      ;;
  esac

  case "${uname_m}" in
    x86_64|amd64) TARGET_ARCH="x86_64" ;;
    aarch64|arm64) TARGET_ARCH="aarch64" ;;
    *)
      die "Unsupported architecture: ${uname_m}"
      ;;
  esac

  TARGET_TRIPLE="${TARGET_ARCH}-${TARGET_OS}"
  ok "Detected platform ${TARGET_TRIPLE}"
}

check_not_root() {
  if [[ "${EUID}" -eq 0 && "${ALLOW_ROOT}" == false ]]; then
    die "Refusing to run as root. Re-run with --yes-i-want-to-run-as-root if this is intentional."
  fi
}

check_disk_space() {
  local probe_path="${DEST_DIR}"
  if [[ ! -d "${probe_path}" ]]; then
    probe_path="$(dirname "${probe_path}")"
  fi

  if [[ ! -d "${probe_path}" ]]; then
    probe_path="."
  fi

  local free_kb
  free_kb="$(df -Pk "${probe_path}" | awk 'NR==2 {print $4}')"
  if [[ -z "${free_kb}" ]]; then
    die "Unable to determine free disk space for ${probe_path}"
  fi

  local required_kb=$((MIN_DISK_MB * 1024))
  if (( free_kb < required_kb )); then
    die "At least ${MIN_DISK_MB}MB free disk space is required. Available: $((free_kb / 1024))MB."
  fi

  info "Disk space check passed (${free_kb}KB available)"
}

check_write_permissions() {
  local target_parent
  if [[ -d "${DEST_DIR}" ]]; then
    target_parent="${DEST_DIR}"
  else
    target_parent="$(dirname "${DEST_DIR}")"
  fi

  if [[ ! -d "${target_parent}" ]]; then
    die "Destination parent directory does not exist: ${target_parent}"
  fi

  if [[ ! -w "${target_parent}" ]]; then
    die "No write permission for destination parent: ${target_parent}"
  fi

  mkdir -p "${DEST_DIR}" || die "Failed to create destination directory ${DEST_DIR}"
  local probe_file="${DEST_DIR}/.fsfs-install-write-probe-$$"
  : > "${probe_file}" || die "Write test failed for ${DEST_DIR}"
  rm -f "${probe_file}" || die "Failed to clean write probe in ${DEST_DIR}"
  info "Destination write check passed (${DEST_DIR})"
}

check_network_connectivity() {
  if [[ "${OFFLINE}" == true ]]; then
    info "Offline mode enabled; skipping network checks"
    return
  fi

  local checks=(
    "https://api.github.com"
    "https://huggingface.co"
  )

  local endpoint
  for endpoint in "${checks[@]}"; do
    if has_cmd curl; then
      curl --silent --show-error --location --head --max-time 8 "${endpoint}" >/dev/null \
        || die "Network connectivity check failed for ${endpoint}"
    elif has_cmd wget; then
      wget --spider --timeout=8 "${endpoint}" >/dev/null 2>&1 \
        || die "Network connectivity check failed for ${endpoint}"
    else
      die "Need curl or wget for connectivity checks"
    fi
  done

  info "Network preflight checks passed"
}

check_existing_installation() {
  local target_bin="${DEST_DIR}/${DEFAULT_BINARY_NAME}"
  if [[ -f "${target_bin}" && "${FORCE}" == false ]]; then
    die "Existing installation found at ${target_bin}. Re-run with --force to overwrite."
  fi

  if command -v "${DEFAULT_BINARY_NAME}" >/dev/null 2>&1; then
    local current_bin
    current_bin="$(command -v "${DEFAULT_BINARY_NAME}")"
    if [[ "${current_bin}" != "${target_bin}" && "${FORCE}" == false ]]; then
      warn "Existing ${DEFAULT_BINARY_NAME} found at ${current_bin}. Use --force to overwrite ${target_bin}."
    fi
  fi
}

detect_command_version() {
  local cmd="$1"
  if ! has_cmd "${cmd}"; then
    printf 'not-installed'
    return
  fi

  local version_line=""
  version_line="$("${cmd}" --version 2>/dev/null | head -n 1 || true)"
  if [[ -n "${version_line}" ]]; then
    printf '%s' "${version_line}"
    return
  fi

  printf 'unknown'
}

register_agent_detection() {
  local name="$1"
  local detected="$2"
  local version="$3"
  local target="$4"

  AGENT_NAMES+=("${name}")
  AGENT_DETECTED+=("${detected}")
  AGENT_VERSIONS+=("${version}")
  AGENT_TARGETS+=("${target}")
  AGENT_RESULTS+=("pending")
}

detect_agent_integrations() {
  AGENT_NAMES=()
  AGENT_DETECTED=()
  AGENT_VERSIONS=()
  AGENT_TARGETS=()
  AGENT_RESULTS=()

  local claude_detected="no"
  if [[ -d "${HOME}/.claude" ]] || has_cmd claude; then
    claude_detected="yes"
  fi
  register_agent_detection \
    "claude-code" \
    "${claude_detected}" \
    "$(detect_command_version claude)" \
    "${HOME}/.claude/settings.json + ${HOME}/.claude/skills/frankensearch-fsfs/SKILL.md"

  local cursor_detected="no"
  if [[ -d "${HOME}/.cursor" ]] || has_cmd cursor; then
    cursor_detected="yes"
  fi
  register_agent_detection \
    "cursor" \
    "${cursor_detected}" \
    "$(detect_command_version cursor)" \
    "${HOME}/.cursor/settings.json"

  local aider_detected="no"
  if [[ -d "${HOME}/.aider" ]] || has_cmd aider || [[ -f "${HOME}/.aider.conf.yml" ]]; then
    aider_detected="yes"
  fi
  register_agent_detection \
    "aider" \
    "${aider_detected}" \
    "$(detect_command_version aider)" \
    "${HOME}/.aider.conf.yml"

  local continue_detected="no"
  if [[ -d "${HOME}/.continue" ]]; then
    continue_detected="yes"
  fi
  register_agent_detection \
    "continue-dev" \
    "${continue_detected}" \
    "unknown" \
    "${HOME}/.continue/config.json"

  local codeium_detected="no"
  if [[ -d "${HOME}/.codeium" ]] || has_cmd codeium; then
    codeium_detected="yes"
  fi
  register_agent_detection \
    "codeium" \
    "${codeium_detected}" \
    "$(detect_command_version codeium)" \
    "${HOME}/.codeium/config.json"

  local copilot_detected="no"
  if [[ -d "${HOME}/.config/github-copilot" ]]; then
    copilot_detected="yes"
  fi
  register_agent_detection \
    "github-copilot" \
    "${copilot_detected}" \
    "unknown" \
    "${HOME}/.config/github-copilot/hosts.json"

  local amazon_q_detected="no"
  if [[ -d "${HOME}/.amazon-q" ]] || [[ -d "${HOME}/.aws/amazonq" ]]; then
    amazon_q_detected="yes"
  fi
  register_agent_detection \
    "amazon-q" \
    "${amazon_q_detected}" \
    "unknown" \
    "${HOME}/.aws/amazonq/config.toml"
}

backup_file_if_present() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    return 0
  fi

  local stamp
  stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  local backup_path="${path}.bak.${stamp}"
  cp "${path}" "${backup_path}" || return 1
  info "Backed up ${path} -> ${backup_path}"
}

prompt_yes_no() {
  local prompt="$1"
  local default_choice="${2:-yes}"
  local default_hint="[Y/n]"
  local answer=""

  if [[ "${default_choice}" == "no" ]]; then
    default_hint="[y/N]"
  fi

  read -r -p "${prompt} ${default_hint} " answer || return 1

  if [[ -z "${answer}" ]]; then
    [[ "${default_choice}" == "yes" ]]
    return
  fi

  [[ "${answer}" =~ ^[Yy]$ ]]
}

should_run_optional_step() {
  local step_name="$1"
  local prompt="$2"
  local default_choice="${3:-yes}"

  if [[ "${EASY_MODE}" == true ]]; then
    return 0
  fi

  if [[ ! -t 0 ]]; then
    info "Skipping ${step_name} in non-interactive mode. Use --easy-mode to run it automatically."
    return 1
  fi

  prompt_yes_no "${prompt}" "${default_choice}"
}

should_run_config_step() {
  local step_name="$1"
  local prompt="$2"
  local default_choice="${3:-yes}"

  if [[ "${NO_CONFIGURE}" == true ]]; then
    info "Skipping ${step_name} because --no-configure was provided."
    return 1
  fi

  should_run_optional_step "${step_name}" "${prompt}" "${default_choice}"
}

should_configure_agent() {
  local agent="$1"
  if [[ "${NO_CONFIGURE}" == true ]]; then
    return 1
  fi

  if [[ "${EASY_MODE}" == true ]]; then
    return 0
  fi

  if [[ ! -t 0 ]]; then
    return 1
  fi

  prompt_yes_no "Configure ${agent} integration?" "no"
}

LAST_CONFIG_RESULT=""

resolve_fsfs_binary_for_completion() {
  local dest_candidate="${DEST_DIR}/${DEFAULT_BINARY_NAME}"
  if [[ -x "${dest_candidate}" ]]; then
    printf '%s' "${dest_candidate}"
    return 0
  fi

  if has_cmd "${DEFAULT_BINARY_NAME}"; then
    command -v "${DEFAULT_BINARY_NAME}"
    return 0
  fi

  return 1
}

detect_completion_shell() {
  local raw_shell="${SHELL:-}"
  local shell_name
  shell_name="$(basename "${raw_shell}")"

  case "${shell_name}" in
    bash|zsh|fish)
      printf '%s' "${shell_name}"
      ;;
    *)
      return 1
      ;;
  esac
}

completion_install_path_for_shell() {
  local shell_name="$1"
  local data_home="${XDG_DATA_HOME:-${HOME}/.local/share}"

  case "${shell_name}" in
    bash)
      printf '%s/bash-completion/completions/%s' "${data_home}" "${DEFAULT_BINARY_NAME}"
      ;;
    zsh)
      printf '%s/zsh/site-functions/_%s' "${data_home}" "${DEFAULT_BINARY_NAME}"
      ;;
    fish)
      printf '%s/fish/completions/%s.fish' "${data_home}" "${DEFAULT_BINARY_NAME}"
      ;;
    *)
      return 1
      ;;
  esac
}

shell_rc_path_for_shell() {
  local shell_name="$1"

  case "${shell_name}" in
    bash)
      printf '%s/.bashrc' "${HOME}"
      ;;
    zsh)
      printf '%s/.zshrc' "${HOME}"
      ;;
    fish)
      printf '%s/.config/fish/config.fish' "${HOME}"
      ;;
    *)
      printf '%s/.profile' "${HOME}"
      ;;
  esac
}

path_export_line_for_shell() {
  local shell_name="$1"

  case "${shell_name}" in
    fish)
      printf 'fish_add_path -g "%s"' "${DEST_DIR}"
      ;;
    *)
      printf "export PATH=\"%s:\$PATH\"" "${DEST_DIR}"
      ;;
  esac
}

configure_shell_path() {
  local shell_name
  shell_name="$(basename "${SHELL:-}")"
  if [[ -z "${shell_name}" ]]; then
    shell_name="sh"
  fi

  local rc_path
  rc_path="$(shell_rc_path_for_shell "${shell_name}")"
  local path_line
  path_line="$(path_export_line_for_shell "${shell_name}")"

  mkdir -p "$(dirname "${rc_path}")" || {
    warn "Failed to create parent directory for ${rc_path}; skipping PATH update."
    return 0
  }

  if [[ -f "${rc_path}" ]] && grep -Fq "${path_line}" "${rc_path}"; then
    info "PATH update already present in ${rc_path}"
    return 0
  fi

  backup_file_if_present "${rc_path}" || {
    warn "Failed to back up ${rc_path}; skipping PATH update."
    return 0
  }

  if [[ ! -f "${rc_path}" ]]; then
    touch "${rc_path}" || {
      warn "Could not create ${rc_path}; skipping PATH update."
      return 0
    }
  fi

  {
    printf '\n# Added by frankensearch fsfs installer for PATH setup\n'
    printf '%s\n' "${path_line}"
  } >> "${rc_path}" || {
    warn "Failed to append PATH update to ${rc_path}; skipping."
    return 0
  }

  ok "Updated ${rc_path} with fsfs PATH entry (${DEST_DIR})"

  if [[ ":${PATH}:" != *":${DEST_DIR}:"* ]]; then
    warn "Current shell PATH not updated yet. Restart shell or run: export PATH=\"${DEST_DIR}:\$PATH\""
  fi
}

maybe_configure_shell_path() {
  if ! should_run_config_step \
    "PATH setup" \
    "Add ${DEST_DIR} to your shell startup PATH?" \
    "yes"; then
    return 0
  fi

  configure_shell_path
}

ensure_zsh_fpath_for_easy_mode() {
  local completion_dir="$1"
  local zshrc_path="${HOME}/.zshrc"
  local fpath_line="fpath=(\"${completion_dir}\" \$fpath)"

  if [[ "${EASY_MODE}" != true ]]; then
    return 0
  fi

  if [[ -f "${zshrc_path}" ]] && grep -Fq "${fpath_line}" "${zshrc_path}"; then
    return 0
  fi

  if [[ ! -f "${zshrc_path}" ]]; then
    touch "${zshrc_path}" || return 1
  fi

  {
    printf '\n# Added by frankensearch fsfs installer for zsh completions\n'
    printf '%s\n' "${fpath_line}"
  } >> "${zshrc_path}" || return 1

  info "Updated ${zshrc_path} to include ${completion_dir} in fpath."
}

install_shell_completion() {
  local shell_name=""
  if ! shell_name="$(detect_completion_shell)"; then
    warn "Could not detect a supported shell from SHELL='${SHELL:-unset}'; skipping completion install."
    return 0
  fi

  local fsfs_bin=""
  if ! fsfs_bin="$(resolve_fsfs_binary_for_completion)"; then
    warn "Could not locate an executable fsfs binary for completion generation; skipping completion install."
    return 0
  fi

  local completion_target=""
  completion_target="$(completion_install_path_for_shell "${shell_name}")" || {
    warn "No completion install path mapping for shell '${shell_name}'; skipping."
    return 0
  }

  local completion_dir
  completion_dir="$(dirname "${completion_target}")"
  mkdir -p "${completion_dir}" || {
    warn "Failed to create completion directory ${completion_dir}; skipping completion install."
    return 0
  }

  local completion_script=""
  completion_script="$("${fsfs_bin}" completions "${shell_name}" 2>/dev/null || true)"
  if [[ -z "${completion_script}" ]]; then
    warn "Completion generation failed via '${fsfs_bin} completions ${shell_name}'; skipping completion install."
    return 0
  fi

  backup_file_if_present "${completion_target}" || {
    warn "Failed to backup existing completion file at ${completion_target}; skipping completion install."
    return 0
  }

  printf '%s\n' "${completion_script}" > "${completion_target}" || {
    warn "Failed to write completion file ${completion_target}; skipping."
    return 0
  }

  if [[ ! -s "${completion_target}" ]]; then
    warn "Completion file ${completion_target} is empty after write; skipping."
    return 0
  fi

  ok "Installed ${shell_name} completions to ${completion_target}"

  if [[ "${shell_name}" == "zsh" ]]; then
    ensure_zsh_fpath_for_easy_mode "${completion_dir}" || {
      warn "Could not ensure zsh fpath contains ${completion_dir}"
      return 0
    }
  fi
}

maybe_install_shell_completion() {
  if ! should_run_config_step \
    "shell completion install" \
    "Install ${DEFAULT_BINARY_NAME} shell completions now?" \
    "yes"; then
    return 0
  fi

  install_shell_completion
}

run_initial_model_download() {
  local fsfs_bin=""
  if ! fsfs_bin="$(resolve_fsfs_binary_for_completion)"; then
    warn "Could not locate an executable fsfs binary; skipping model pre-download."
    return 0
  fi

  if "${fsfs_bin}" download >/dev/null 2>&1; then
    ok "Initial model download completed."
  else
    warn "Model pre-download command failed (${fsfs_bin} download)."
  fi
}

maybe_run_initial_model_download() {
  if ! should_run_optional_step \
    "model pre-download" \
    "Download initial embedding models now for faster first search?" \
    "yes"; then
    return 0
  fi

  run_initial_model_download
}

run_post_install_doctor() {
  local fsfs_bin=""
  if ! fsfs_bin="$(resolve_fsfs_binary_for_completion)"; then
    warn "Could not locate an executable fsfs binary; skipping doctor verification."
    return 0
  fi

  if "${fsfs_bin}" doctor >/dev/null 2>&1; then
    ok "Post-install doctor check passed."
  else
    warn "Doctor check failed (${fsfs_bin} doctor)."
  fi
}

maybe_run_post_install_doctor() {
  if ! should_run_optional_step \
    "doctor verification" \
    "Run fsfs doctor now to verify the installation?" \
    "yes"; then
    return 0
  fi

  run_post_install_doctor
}

configure_claude_code() {
  local fsfs_bin="${DEST_DIR}/${DEFAULT_BINARY_NAME}"
  local claude_root="${HOME}/.claude"
  local settings_path="${claude_root}/settings.json"
  local skills_dir="${claude_root}/skills/frankensearch-fsfs"
  local skill_path="${skills_dir}/SKILL.md"
  local had_existing=false

  mkdir -p "${claude_root}" "${skills_dir}" || {
    LAST_CONFIG_RESULT="failed (mkdir)"
    return 1
  }

  if [[ -f "${settings_path}" ]] || [[ -f "${skill_path}" ]]; then
    had_existing=true
  fi

  backup_file_if_present "${settings_path}" || {
    LAST_CONFIG_RESULT="failed (backup settings)"
    return 1
  }
  backup_file_if_present "${skill_path}" || {
    LAST_CONFIG_RESULT="failed (backup skill)"
    return 1
  }

  if has_cmd jq; then
    if [[ -f "${settings_path}" ]]; then
      jq \
        --arg cmd "${fsfs_bin}" \
        '.mcpServers = (.mcpServers // {}) | .mcpServers["frankensearch-fsfs"] = {"command": $cmd, "args": ["search", "--format", "json", "--limit", "20"]}' \
        "${settings_path}" > "${TEMP_DIR}/claude-settings.json" || {
        LAST_CONFIG_RESULT="failed (invalid claude settings.json)"
        return 1
      }
    else
      jq -n \
        --arg cmd "${fsfs_bin}" \
        '{mcpServers: {"frankensearch-fsfs": {command: $cmd, args: ["search", "--format", "json", "--limit", "20"]}}}' \
        > "${TEMP_DIR}/claude-settings.json" || {
        LAST_CONFIG_RESULT="failed (render claude settings)"
        return 1
      }
    fi
    mv "${TEMP_DIR}/claude-settings.json" "${settings_path}" || {
      LAST_CONFIG_RESULT="failed (write claude settings)"
      return 1
    }
  else
    warn "jq not found; writing minimal Claude settings without merge support."
    cat > "${settings_path}" <<EOF
{
  "mcpServers": {
    "frankensearch-fsfs": {
      "command": "${fsfs_bin}",
      "args": ["search", "--format", "json", "--limit", "20"]
    }
  }
}
EOF
  fi

  cat > "${skill_path}" <<EOF
# frankensearch-fsfs installer managed skill

Use fsfs for semantic codebase search.

Examples:
- \`${fsfs_bin} search "error handling"\`
- \`${fsfs_bin} search --format json --limit 20 "query"\`

When searching large repos, prefer fsfs over naive grep for semantic recall.
EOF

  if [[ ! -f "${skill_path}" ]]; then
    LAST_CONFIG_RESULT="failed (skill verify)"
    return 1
  fi

  if has_cmd jq; then
    jq -e '.mcpServers["frankensearch-fsfs"].command' "${settings_path}" >/dev/null 2>&1 || {
      LAST_CONFIG_RESULT="failed (settings verify)"
      return 1
    }
  fi

  if [[ "${had_existing}" == true ]]; then
    LAST_CONFIG_RESULT="merged"
  else
    LAST_CONFIG_RESULT="created"
  fi
}

configure_cursor() {
  local fsfs_bin="${DEST_DIR}/${DEFAULT_BINARY_NAME}"
  local cursor_dir="${HOME}/.cursor"
  local settings_path="${cursor_dir}/settings.json"
  local had_existing=false

  mkdir -p "${cursor_dir}" || {
    LAST_CONFIG_RESULT="failed (mkdir)"
    return 1
  }

  if [[ -f "${settings_path}" ]]; then
    had_existing=true
  fi
  backup_file_if_present "${settings_path}" || {
    LAST_CONFIG_RESULT="failed (backup)"
    return 1
  }

  if has_cmd jq; then
    if [[ -f "${settings_path}" ]]; then
      jq \
        --arg bin "${fsfs_bin}" \
        '. + {"frankensearch.enabled": true, "frankensearch.fsfsPath": $bin, "frankensearch.searchCommand": ($bin + " search --format json --limit 20")}' \
        "${settings_path}" > "${TEMP_DIR}/cursor-settings.json" || {
        LAST_CONFIG_RESULT="failed (invalid cursor settings.json)"
        return 1
      }
    else
      jq -n \
        --arg bin "${fsfs_bin}" \
        '{"frankensearch.enabled": true, "frankensearch.fsfsPath": $bin, "frankensearch.searchCommand": ($bin + " search --format json --limit 20")}' \
        > "${TEMP_DIR}/cursor-settings.json" || {
        LAST_CONFIG_RESULT="failed (render cursor settings)"
        return 1
      }
    fi
  else
    LAST_CONFIG_RESULT="failed (jq required)"
    return 1
  fi

  mv "${TEMP_DIR}/cursor-settings.json" "${settings_path}" || {
    LAST_CONFIG_RESULT="failed (write settings)"
    return 1
  }

  jq -e '.["frankensearch.enabled"] == true and .["frankensearch.fsfsPath"] != null' "${settings_path}" >/dev/null 2>&1 || {
    LAST_CONFIG_RESULT="failed (verify)"
    return 1
  }

  if [[ "${had_existing}" == true ]]; then
    LAST_CONFIG_RESULT="merged"
  else
    LAST_CONFIG_RESULT="created"
  fi
}

configure_detected_agents() {
  local i=0
  for i in "${!AGENT_NAMES[@]}"; do
    local name="${AGENT_NAMES[$i]}"
    local detected="${AGENT_DETECTED[$i]}"

    if [[ "${detected}" != "yes" ]]; then
      AGENT_RESULTS[i]="skipped (not detected)"
      continue
    fi

    if ! should_configure_agent "${name}"; then
      if [[ "${NO_CONFIGURE}" == true ]]; then
        AGENT_RESULTS[i]="skipped (--no-configure)"
      elif [[ "${EASY_MODE}" == false && ! -t 0 ]]; then
        AGENT_RESULTS[i]="skipped (non-interactive)"
      else
        AGENT_RESULTS[i]="skipped (user)"
      fi
      continue
    fi

    case "${name}" in
      claude-code)
        if configure_claude_code; then
          AGENT_RESULTS[i]="${LAST_CONFIG_RESULT}"
        else
          AGENT_RESULTS[i]="${LAST_CONFIG_RESULT}"
        fi
        ;;
      cursor)
        if configure_cursor; then
          AGENT_RESULTS[i]="${LAST_CONFIG_RESULT}"
        else
          AGENT_RESULTS[i]="${LAST_CONFIG_RESULT}"
        fi
        ;;
      *)
        AGENT_RESULTS[i]="skipped (detection-only)"
        ;;
    esac
  done
}

print_agent_report_table() {
  if [[ "${USE_COLOR}" == true ]]; then
    printf '\n%bAI Agent Integration Report%b\n' "${COLOR_BOLD}" "${COLOR_RESET}"
  else
    printf '\nAI Agent Integration Report\n'
  fi
  printf '%-16s %-8s %-24s %-48s %-24s\n' "Agent" "Detected" "Version" "Target" "Result"
  printf '%-16s %-8s %-24s %-48s %-24s\n' "-----" "--------" "-------" "------" "------"

  local i=0
  for i in "${!AGENT_NAMES[@]}"; do
    printf '%-16s %-8s %-24s %-48s %-24s\n' \
      "${AGENT_NAMES[$i]}" \
      "${AGENT_DETECTED[$i]}" \
      "${AGENT_VERSIONS[$i]}" \
      "${AGENT_TARGETS[$i]}" \
      "${AGENT_RESULTS[$i]}"
  done
}

resolve_version() {
  if [[ "${VERSION}" != "latest" ]]; then
    RESOLVED_VERSION="${VERSION}"
    return
  fi

  if [[ "${OFFLINE}" == true ]]; then
    RESOLVED_VERSION="latest"
    warn "Offline mode active; cannot resolve latest version tag from network."
    return
  fi

  local repo_slug="${FRANKENSEARCH_REPO:-${DEFAULT_REPO_SLUG}}"
  local api_url="https://api.github.com/repos/${repo_slug}/releases/latest"
  local tag_name=""

  if has_cmd curl; then
    tag_name="$(curl --silent --show-error --location --max-time 10 "${api_url}" \
      | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
  elif has_cmd wget; then
    tag_name="$(wget -qO- "${api_url}" \
      | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
  fi

  if [[ -z "${tag_name}" ]]; then
    RESOLVED_VERSION="latest"
    warn "Could not resolve latest release tag; keeping 'latest' selector."
  else
    RESOLVED_VERSION="${tag_name}"
    info "Resolved latest version to ${RESOLVED_VERSION}"
  fi
}

artifact_name() {
  printf '%s-%s.tar.xz' "${DEFAULT_BINARY_NAME}" "${TARGET_TRIPLE}"
}

checksum_file_name() {
  printf '%s.sha256' "$(artifact_name)"
}

checksum_url() {
  local repo_slug="${FRANKENSEARCH_REPO:-${DEFAULT_REPO_SLUG}}"
  local checksum_file
  checksum_file="$(checksum_file_name)"

  if [[ "${RESOLVED_VERSION}" == "latest" ]]; then
    printf 'https://github.com/%s/releases/latest/download/%s\n' "${repo_slug}" "${checksum_file}"
  else
    printf 'https://github.com/%s/releases/download/%s/%s\n' "${repo_slug}" "${RESOLVED_VERSION}" "${checksum_file}"
  fi
}

checksums_txt_url() {
  local repo_slug="${FRANKENSEARCH_REPO:-${DEFAULT_REPO_SLUG}}"

  if [[ "${RESOLVED_VERSION}" == "latest" ]]; then
    printf 'https://github.com/%s/releases/latest/download/checksums.txt\n' "${repo_slug}"
  else
    printf 'https://github.com/%s/releases/download/%s/checksums.txt\n' "${repo_slug}" "${RESOLVED_VERSION}"
  fi
}

artifact_url() {
  local repo_slug="${FRANKENSEARCH_REPO:-${DEFAULT_REPO_SLUG}}"
  local artifact
  artifact="$(artifact_name)"

  if [[ "${RESOLVED_VERSION}" == "latest" ]]; then
    printf 'https://github.com/%s/releases/latest/download/%s\n' "${repo_slug}" "${artifact}"
  else
    printf 'https://github.com/%s/releases/download/%s/%s\n' "${repo_slug}" "${RESOLVED_VERSION}" "${artifact}"
  fi
}

run_preflight_checks() {
  detect_platform
  check_not_root
  check_disk_space
  check_write_permissions
  check_network_connectivity
  check_existing_installation
}

# ---------------------------------------------------------------------------
# Binary download, verification, extraction, and installation
# ---------------------------------------------------------------------------

http_download() {
  local url="$1"
  local output="$2"

  if has_cmd curl; then
    curl --fail --silent --show-error --location --max-time 120 \
      --retry 3 --retry-delay 2 \
      -o "${output}" "${url}"
  elif has_cmd wget; then
    wget --quiet --tries=3 --timeout=120 \
      -O "${output}" "${url}"
  else
    die "Need curl or wget for downloads"
  fi
}

download_artifact() {
  local url
  url="$(artifact_url)"
  local artifact_file
  artifact_file="${TEMP_DIR}/$(artifact_name)"

  info "Downloading binary archive from ${url}"
  if ! http_download "${url}" "${artifact_file}"; then
    return 1
  fi

  if [[ ! -f "${artifact_file}" || ! -s "${artifact_file}" ]]; then
    err "Downloaded artifact is missing or empty"
    return 1
  fi

  local size_bytes
  size_bytes="$(wc -c < "${artifact_file}" | tr -d ' ')"
  ok "Downloaded archive (${size_bytes} bytes)"
}

resolve_expected_checksum() {
  # If caller provided --checksum, use that directly.
  if [[ -n "${CHECKSUM}" ]]; then
    printf '%s' "${CHECKSUM}"
    return 0
  fi

  local artifact
  artifact="$(artifact_name)"
  local checksum_file
  checksum_file="${TEMP_DIR}/$(checksum_file_name)"

  # Try per-artifact .sha256 file first.
  local per_artifact_url
  per_artifact_url="$(checksum_url)"
  info "Downloading checksum from ${per_artifact_url}"
  if http_download "${per_artifact_url}" "${checksum_file}" 2>/dev/null; then
    if [[ -f "${checksum_file}" && -s "${checksum_file}" ]]; then
      # Format: "<hex>  <filename>" or just "<hex>"
      local hash
      hash="$(awk '{print $1}' "${checksum_file}" | head -n 1 | tr -d '[:space:]')"
      if [[ "${hash}" =~ ^[A-Fa-f0-9]{64}$ ]]; then
        printf '%s' "${hash}"
        return 0
      fi
    fi
  fi

  # Fallback: try checksums.txt (all artifacts in one file).
  local checksums_txt="${TEMP_DIR}/checksums.txt"
  local checksums_url
  checksums_url="$(checksums_txt_url)"
  info "Per-artifact checksum not found; trying checksums.txt from ${checksums_url}"
  if http_download "${checksums_url}" "${checksums_txt}" 2>/dev/null; then
    if [[ -f "${checksums_txt}" && -s "${checksums_txt}" ]]; then
      # Format: "<hex>  <filename>" per line
      local hash
      hash="$(grep -F "${artifact}" "${checksums_txt}" | awk '{print $1}' | head -n 1 | tr -d '[:space:]')"
      if [[ "${hash}" =~ ^[A-Fa-f0-9]{64}$ ]]; then
        printf '%s' "${hash}"
        return 0
      fi
    fi
  fi

  # No checksum source available.
  return 1
}

compute_sha256() {
  local file="$1"

  if has_cmd sha256sum; then
    sha256sum "${file}" | awk '{print $1}'
  elif has_cmd shasum; then
    shasum -a 256 "${file}" | awk '{print $1}'
  elif has_cmd openssl; then
    openssl dgst -sha256 -hex "${file}" | awk '{print $NF}'
  else
    die "No SHA-256 tool found (need sha256sum, shasum, or openssl)"
  fi
}

verify_artifact_checksum() {
  local artifact_file
  artifact_file="${TEMP_DIR}/$(artifact_name)"

  local expected_hash=""
  if expected_hash="$(resolve_expected_checksum)"; then
    info "Verifying SHA-256 checksum..."
    local actual_hash
    actual_hash="$(compute_sha256 "${artifact_file}")"

    # Normalize to lowercase for comparison.
    expected_hash="$(printf '%s' "${expected_hash}" | tr '[:upper:]' '[:lower:]')"
    actual_hash="$(printf '%s' "${actual_hash}" | tr '[:upper:]' '[:lower:]')"

    if [[ "${actual_hash}" != "${expected_hash}" ]]; then
      err "SHA-256 checksum mismatch!"
      err "  Expected: ${expected_hash}"
      err "  Actual:   ${actual_hash}"
      die "Artifact verification failed. The download may be corrupted or tampered with."
    fi

    ok "SHA-256 checksum verified: ${actual_hash}"
  else
    if [[ "${VERIFY}" == true ]]; then
      die "Checksum verification was requested (--verify) but no checksum is available. Provide one with --checksum or ensure the release includes .sha256 files."
    fi
    warn "No checksum available for verification. Skipping integrity check."
    warn "Use --checksum <sha256> to provide one, or --verify to require it."
  fi
}

extract_archive() {
  local artifact_file
  artifact_file="${TEMP_DIR}/$(artifact_name)"
  local extract_dir="${TEMP_DIR}/extract"
  mkdir -p "${extract_dir}"

  info "Extracting archive..."

  # Detect format by extension and available tools.
  local artifact_basename
  artifact_basename="$(basename "${artifact_file}")"

  case "${artifact_basename}" in
    *.tar.xz)
      if has_cmd xz; then
        tar -xJf "${artifact_file}" -C "${extract_dir}" || die "Failed to extract .tar.xz archive (tar + xz)"
      elif has_cmd unxz; then
        unxz --keep --stdout "${artifact_file}" | tar -xf - -C "${extract_dir}" || die "Failed to extract .tar.xz archive (unxz + tar)"
      else
        die "Cannot extract .tar.xz archive: need xz or unxz. Install xz-utils (apt) or xz (brew)."
      fi
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "${artifact_file}" -C "${extract_dir}" || die "Failed to extract .tar.gz archive"
      ;;
    *.zip)
      if has_cmd unzip; then
        unzip -q "${artifact_file}" -d "${extract_dir}" || die "Failed to extract .zip archive"
      else
        die "Cannot extract .zip archive: need unzip"
      fi
      ;;
    *)
      die "Unknown archive format: ${artifact_basename}"
      ;;
  esac

  # Locate the binary inside the extracted tree.
  local binary_path=""

  # Direct in extract dir.
  if [[ -f "${extract_dir}/${DEFAULT_BINARY_NAME}" ]]; then
    binary_path="${extract_dir}/${DEFAULT_BINARY_NAME}"
  else
    # Search one level deep (archives often have a top-level directory).
    binary_path="$(find "${extract_dir}" -maxdepth 2 -name "${DEFAULT_BINARY_NAME}" -type f | head -n 1)"
  fi

  if [[ -z "${binary_path}" || ! -f "${binary_path}" ]]; then
    die "Could not find '${DEFAULT_BINARY_NAME}' binary inside the extracted archive"
  fi

  # Stage the binary for installation.
  cp "${binary_path}" "${TEMP_DIR}/${DEFAULT_BINARY_NAME}" || die "Failed to stage binary"
  ok "Extracted binary: ${DEFAULT_BINARY_NAME}"
}

install_binary() {
  local staged_binary="${TEMP_DIR}/${DEFAULT_BINARY_NAME}"
  local target_binary="${DEST_DIR}/${DEFAULT_BINARY_NAME}"

  if [[ ! -f "${staged_binary}" ]]; then
    die "Staged binary not found at ${staged_binary}"
  fi

  # Ensure destination directory exists.
  mkdir -p "${DEST_DIR}" || die "Failed to create destination directory ${DEST_DIR}"

  # Install with executable permissions.
  # Use full path to avoid shell aliases (e.g., install -> apt install).
  if [[ -x /usr/bin/install ]]; then
    /usr/bin/install -m 0755 "${staged_binary}" "${target_binary}" || die "Failed to install binary to ${target_binary}"
  else
    cp "${staged_binary}" "${target_binary}" || die "Failed to copy binary to ${target_binary}"
    chmod 0755 "${target_binary}" || die "Failed to set permissions on ${target_binary}"
  fi

  # Verify the installed binary is executable.
  if [[ ! -x "${target_binary}" ]]; then
    die "Installed binary is not executable: ${target_binary}"
  fi

  ok "Installed ${DEFAULT_BINARY_NAME} to ${target_binary}"
}

verify_installation() {
  local target_binary="${DEST_DIR}/${DEFAULT_BINARY_NAME}"

  info "Verifying installation..."

  local version_output=""
  version_output="$("${target_binary}" --version 2>&1 || true)"

  if [[ -z "${version_output}" ]]; then
    warn "Binary at ${target_binary} did not produce --version output."
    warn "The binary may require shared libraries not present on this system."
    return 1
  fi

  ok "Binary verification passed: ${version_output}"
}

print_plan() {
  local url
  url="$(artifact_url)"

  if [[ "${USE_COLOR}" == true ]]; then
    printf '%bInstallation Plan%b\n' "${COLOR_BOLD}" "${COLOR_RESET}"
  else
    printf 'Installation Plan\n'
  fi
  printf '  Version selector : %s\n' "${VERSION}"
  printf '  Resolved version : %s\n' "${RESOLVED_VERSION}"
  printf '  Target triple    : %s\n' "${TARGET_TRIPLE}"
  printf '  Destination      : %s\n' "${DEST_DIR}"
  printf '  Artifact URL     : %s\n' "${url}"
  printf '  Verify checksum  : %s\n' "${VERIFY}"
  printf '  From source      : %s\n' "${FROM_SOURCE}"
  printf '  Configure shell  : %s\n' "$([[ "${NO_CONFIGURE}" == true ]] && echo "false" || echo "true")"
}

run_install() {
  if [[ "${FROM_SOURCE}" == true ]]; then
    info "FROM_SOURCE=true: building from source (see bd-2w7x.19)."
    die "Build-from-source is not yet implemented. Remove --from-source to download a prebuilt binary."
  fi

  # Stage 1: Download the binary archive.
  if ! download_artifact; then
    err "Binary download failed."
    info "You can try building from source with: ${SCRIPT_NAME} --from-source"
    die "Download failed for $(artifact_url)"
  fi

  # Stage 2: Verify the checksum.
  verify_artifact_checksum

  # Stage 3: Extract the archive.
  extract_archive

  # Stage 4: Install the binary.
  install_binary

  # Stage 5: Verify the binary runs.
  if ! verify_installation; then
    warn "Binary verification failed but installation completed."
    warn "You may need to install runtime dependencies (e.g., ONNX Runtime)."
  fi

  # Post-install configuration stages.
  maybe_configure_shell_path
  maybe_install_shell_completion

  detect_agent_integrations
  configure_detected_agents
  print_agent_report_table
  maybe_run_initial_model_download
  maybe_run_post_install_doctor

  ok "Installation completed successfully."
  info "Run '${DEFAULT_BINARY_NAME} --help' to get started."
}

main() {
  parse_args "$@"
  configure_output
  validate_checksum

  if [[ "${OFFLINE}" == false ]] && ! has_cmd curl && ! has_cmd wget; then
    die "Need curl or wget for online installation mode"
  fi

  TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/frankensearch-install.XXXXXX")"

  trap on_exit EXIT INT TERM
  acquire_lock
  run_preflight_checks
  resolve_version
  print_plan
  run_install
}

main "$@"
